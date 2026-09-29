# Op 01 — GEMM（矩阵乘法）

**重要性：★★★★★**。GEMM 是一切深度学习计算的地基——LLM 里每个 linear
层、attention 的 QK^T 和 AV、卷积的 im2col 形态，全是矩阵乘。它也是 K3 上
编译链路覆盖最全的算子：矩阵引擎、TCM、pack/unpack、K 分块循环全都用到。
学会这一章，等于学会这本书的一半。

## 1.1 这个算子在算什么

数学定义：`C[M,N] = A[M,K] @ B[K,N]`，即

```
C[i,j] = Σ_{k=0..K-1} A[i,k] · B[k,j]
```

一个 4×4 的具体例子（心算可验证）：

```
A = [[1,0,0,0],        B = [[1,2,3,4],        C = A@B
     [0,1,0,0],             [5,6,7,8],         = B（单位阵乘任何矩阵）
     [0,0,1,0],             [9,10,11,12],
     [0,0,0,1]]             [13,14,15,16]]
```

三个尺寸的记忆法：**M 是输出的行数、N 是输出的列数、K 是“消掉”的维度**
（A 的列数必须等于 B 的行数）。每个输出元素需要 K 次乘加，总计算量
2·M·N·K 次浮点运算（一次乘 + 一次加算 2 次）。

## 1.2 为什么 LLM 里它无处不在

以 Qwen3-0.6B 为例，一次 forward 里的 GEMM：

| 位置 | 形状 | 说明 |
|---|---|---|
| q/k/v_proj | [tokens, 1024] @ [1024, ·] | 每层 3 个 |
| o_proj | [tokens, 1024] @ [1024, 1024] | 每层 1 个 |
| mlp gate/up/down | [tokens, 1024] @ [1024, 3072] 等 | 每层 3 个，最大的计算量 |
| lm_head | [tokens, 1024] @ [1024, 151936] | N 巨大（词表），单列讨论 |

28 层 × 每层 7 个大矩阵乘——**prefill 的墙钟时间大头就是这些 GEMM**。
所以 K3 上 GEMM 走不走矩阵引擎，直接决定端到端速度（本章 1.8 有实测数字）。

## 1.3 PyTorch golden：先有参照物

写 kernel 前永远先把 golden 跑出来：

```python
import torch
torch.manual_seed(0)
M, N, K = 256, 256, 256
a = torch.randn((M, K), dtype=torch.float16)
b = torch.randn((K, N), dtype=torch.float16)
c_torch = torch.matmul(a, b)          # 这就是我们要复现的结果
```

注意 dtype 用 f16——这是 LLM 推理的常态，也是 K3 矩阵引擎的原生 dtype。
容差预期：f16 GEMM 对 f16 golden 用 `atol=1e-2, rtol=1e-2`（K=256 的累加
长度下这是安全值）。

## 1.4 从暴力三重循环到分块并行

最朴素的 GEMM 是三重循环 `for i: for j: for k: C[i,j] += A[i,k]*B[k,j]`。
并行化的思路（回忆 basics/03 的 SPMD 模型）：

1. **把 C 切成 tile**：每个 program 负责一个 `BM×BN` 的输出块；
2. **grid 是二维的**：`grid = (M/BM, N/BN)`，`pid_m/pid_n` 定位自己负责的
   输出块；
3. **一个 program 内部**：要算出自己的 `BM×BN` 块，需要 A 的 `BM×K` 行条
   和 B 的 `K×BN` 列条——沿 K 维分块循环，每次取 `BM×BK` 和 `BK×BN` 两小
   块做乘加，累加到 f32 累加器：

```
        B: K×BN 列条
        ┌──────┐
        │▓▓▓▓▓▓│ ← 第 k 块 (BK×BN)
        │      │
A: M×K  └──────┘
┌─┬──────┐
│ │▓▓▓▓▓▓│ ← A 的第 k 块 (BM×BK)      C 的一个 tile (BM×BN)
├─┼──────┤        acc = Σ_k  A块 @ B块
│▓│      │  ← pid_m 这一行
└─┴──────┘
```

这就是标准的 **GEMM K-loop** 模式，`tl.dot(a_block, b_block)` 一条语句完成
一个小块矩阵乘，编译器自动把它映射到 K3 矩阵引擎（basics/06 §6.2 的
mmt4d/vfwmadot 通路）。

## 1.5 完整可运行程序（block_ptr 版）

来源：`python/examples/mm_block_ptr.py`（仓库 CI 的标准 case，256³ f16）。
block_ptr 是“带形状和边界的指针”，越界处理交给编译器，代码更干净：

```python
# gemm.py — C = A @ B（block_ptr 版，单 K 块）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,      # A 的行/列步长（元素个数）
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # 我负责 C 的第几行块
    pid_n = tl.program_id(1)   # 第几列块

    # A 的行条视图：从 (pid_m*BLOCK_M, 0) 开始取 BLOCK_M × BLOCK_K
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),          # 列优先遍历序（内存连续性提示）
    )
    # B 的列条视图：从 (0, pid_n*BLOCK_N) 开始取 BLOCK_K × BLOCK_N
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    # boundary_check：越界部分自动按 padding 处理（不用手写 mask！）
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    b = tl.load(b_block_ptr, boundary_check=(0, 1))

    acc = tl.dot(a, b)         # f16 @ f16 → f32（自动提升，军规内建）

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    # 注意 .to(...)：block-ptr store 是严格类型检查，无隐式 cast
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"incompatible dimensions: {K} vs {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N = 32, 32
    BLOCK_K = triton.next_power_of_2(K)    # 本例 K 一次装下，不用循环

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)

    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)
    torch.testing.assert_close(c_triton, c_torch, atol=1e-2, rtol=1e-2)
    print(f"PASS: ({M}x{K}) @ ({K}x{N})")
```

```bash
python gemm.py
# PASS: (256x256) @ (256x256)      ← 第一次跑要编译，稍等
```

### 逐段精讲

- **为什么传 6 个 stride**：kernel 不应该假设 tensor 是连续的。
  `stride_am` = A 相邻两行的元素间距（连续 tensor 就是 K）。显式传 stride
  后，非连续视图（切片、转置）也能直接算——生产算子（FlagGems mm）都这么做。
- **`make_block_ptr` 五要素**：base（原始指针）、shape（全矩阵尺寸）、
  strides、offsets（本块的起点）、block_shape（本块尺寸）。它像一个
  “带 GPS 的指针”：知道自己在多大的矩阵里、走一步多远，因此
  `boundary_check=(0,1)` 能自动处理“最后一块不满”的越界（第 0 维和第 1 维
  都检查）——等价于手写 mask，但不会写错。
- **`tl.dot(a, b)`**：块级矩阵乘，`(BM,BK) @ (BK,BN) → (BM,BN)`。f16 输入
  自动得到 f32 输出——**K3 的矩阵引擎硬件本身就是 f16 乘 f32 累加**
  （vfwmadot），dtype 军规在 GEMM 这里是硬件白送的。
- **`acc.to(c_ptr.dtype.element_ty)` 不能省**：普通指针的 `tl.store` 会隐式
  cast，但 **block-ptr store 是严格类型检查**——f32 值存进 f16 block 直接报
  “Block element type(fp16) and value element type(fp32) mismatch”。这是
  新手在 block_ptr 上的第一大坑。
- **grid 大小**：256/32 × 256/32 = 8×8 = 64 个 program。回忆 basics/04：
  libspert <0.6.3 时 grid>512 会被静默丢弃，64 很安全；生产 kernel 的
  BLOCK 调小让 grid 超 512 时就要小心（0.6.3+ 无此问题）。

### K 大时：K-loop 版

上面 `BLOCK_K = next_power_of_2(K)` 一次装下整个 K，只适合 K 小的场景。
K 大时（TCM 装不下，见 basics/06 §6.3）必须沿 K 分块循环，手写 mask 版本
（不用 block_ptr，看另一种边界处理写法）：

```python
@triton.jit
def mm_kloop_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                    sam, sak, sbk, sbn, scm, scn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)     # 本块的行下标向量
    offs_n = pid_n * BN + tl.arange(0, BN)     # 列下标向量
    offs_k = tl.arange(0, BK)
    # 2D 下标 = 行向量[:, None] + 列向量[None, :]（广播成矩阵）
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)      # f32 累加器——军规
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)                          # 部分和留在 f32
        a_ptrs += BK * sak                           # 指针步进到下一个 K 块
        b_ptrs += BK * sbk

    c = acc.to(c_ptr.dtype.element_ty)               # 只在最终输出 cast 回 f16
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, c,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

三个要点：

1. `offs_m[:, None] * sam + offs_k[None, :] * sak` 是**二维下标广播**的
   标准写法：`(BM,1) + (1,BK)` 广播成 `(BM,BK)` 指针矩阵；
2. 循环里 mask 同时防 M/K（或 K/N）两个方向的越界，`other=0.0` 保证越界
   lane 对点积贡献为 0；
3. **部分和永远留在 f32**（`acc += tl.dot(...)`），只在最后 store 前降回
   f16。“每 K 块先截断成 f16 再累加”是错误写法——见 1.7 的契约说明。

## 1.6 验证清单

```bash
python gemm.py                                    # 本章程序
python3 python/examples/mm_block_ptr.py           # 仓库原版 CI case
python3 python/examples/test_smt_mm.py            # smt 矩阵引擎三分支（1.8）
python3 -m pytest python/tests/test_blas_ops.py -q   # 仓库 blas 测试套
```

自查（比“跑过了”更硬的证据）：

```bash
# 确认 kernel 真的用了矩阵引擎：dump IR 看 mmt4d / vfwmadot
SPINE_TRITON_DUMP_PATH=./dumps python gemm.py
grep -c mmt4d ./dumps/mm_kernel/*.linalgdir      # >0 = 走了矩阵引擎通路
# 最终指令层：
grep -c vfwmadot ./dumps/mm_kernel/*.ll
```

## 1.7 K3 上的关键契约与约束

**f16 dot 的 f32 累加契约**：`tl.dot(f16, f16) → f32` 的 K 归约**必须**
f32 累加。这不只是数值建议，而是曾经真实违反过的契约：编译器历史 bug——
循环内 dot 生成 `linalg.matmul outs(f16-fill)`，每 K 块部分和先截断 f16 再
累加，12.7% 元素超差（修复后与“f16 精确乘积 + f32 累加”的 CPU 仿真
99.9% bit 一致）。它波及所有 K-blocking 的 f16 mm/bmm/conv。给你的启示：
**f16 GEMM 数值不对时，先怀疑“哪里把中间和降回了 f16”**，验证方法是三
模型对拍（emuA 精确乘积+f32 累加 / emuB f16 舍入乘积 / 实测）。

| 约束 | 数值/规则 | 违反后果 |
|---|---|---|
| TCM scratch | 256KB/worker | 大 tile 中间量 spill 超量 → NULL → 段错误 |
| BN 上限 | output tile ≤ 32×256（8192 元素）安全；16384 元素崩 | segfault（tl.dot 与 smt.dot 同限，共享 mmt4d lowering） |
| BK | K 大必须分块 | BK=K 一次展开需 608KB → 溢出 hang |
| grid | ≤512（libspert <0.6.3）；0.6.3+ 引擎自动分块 | 旧 runtime 静默丢弃 → 输出=未初始化内存；更坑：丢弃 config 计时~0 反而**赢得 autotune** |
| N 特别大 | N > 131072（lm_head）不走 smt.dot | 单 launch grid 爆；chunked 2 launches 的 dispatch 开销反超（237ms vs 180ms 实测）→ 用 vfwmacc.vv 路径 |
| dtype | f16/f32 可用；**bf16 撞 MatmulConfigAnalysis UNREACHABLE** | rc=134（平台豁免项，非你的错） |

## 1.8 层次 3：smt 矩阵引擎显式编排（性能形态）

标准 `tl.dot` 已被自动映射到矩阵引擎，但生产级 GEMM（FlagGems
`_spacemit/ops/mm.py`）在 K3 上用 **smt.dot** 显式编排：MICRO tile 参数化 +
TCM 中转 + pack 视图。骨架（完整可跑版本 = `python/examples/test_smt_mm.py`）：

```python
if SPLIT_M:
    b_desc = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(b_desc, (0, 0), (BK, BN), (MICRO_K, MICRO_N))   # pack 视图
    for i in smt.parallel(0, sub_num):        # bind_sub_block 默认 False！
        a_sub = ...                            # A 的 sub-tile
        # smt.dot 返回 packed 格式，必须 unpack 再累加（micro=(1,1)）：
        partial = smt.view(smt.dot(a_sub, b), (0, 0), (SUB_M, BN), (1, 1))
        acc += partial
```

K3 f16 MICRO tile = M/K/N **16/8/32**（`tle.mma_cube("f16")` 可查）。
细节全在 [basics/06](../basics/06-smt-matrix-engine.md)。

**实测收益**（Qwen3-0.6B，K3）：

| 形态 | prefill | 说明 |
|---|---|---|
| 纯 vfwmacc.vv（无 smt.dot） | ~21 tok/s | 基线 |
| + smt.dot dispatch 链 | ~40 tok/s | proj mm 五 shape：3.51ms vs 21.49ms（6x） |
| + `logits_to_keep=1` | ~52 tok/s | lm_head 只算最后一个 token（transformers 原生 API） |

生产 dispatch 链的形状感知逻辑（读 FlagGems mm.py 时的地图）：

```
M == 1 ?  ──是──► mm_m1_raw（vfwmacc.vf，B 寄存器广播，decode 场景）
   │否
   ▼
_mm_smt_path（smt.dot，N ≤ 131072）        # 各 proj mm
   ▼ 不满足
_mm_tldot_path（tl.dot，N ≤ 4096）         # smt 失败时的备用
   ▼ 不满足
_mm_transposed_tail（vfwmacc.vv，任意 N）  # lm_head
```

N 不对齐时的 tail 处理：`N % 8 == 0` 即可命中 fast path——main tile
`(N//NB)*NB` 走矩阵引擎 kernel（kernel 带 `CS`=原始行 stride 参数，直接写
full-N 输出的前 n_main 列），tail 走 native `torch.addmm` 再 copy。

## 1.9 常见错误与判读

| 症状 | 根因 | 动作 |
|---|---|---|
| `Block element type ... mismatch` | block-ptr store 没显式 `.to()` | 1.5 精讲第 4 条 |
| 输出全零 / 99.9% 垃圾值 | grid>512 被旧 libspert 丢弃；或旧 spine-mlir 的 f32 unpack 写丢弃 bug（b86dd23 已修） | 查 libspert 版本；换 cache 重跑；再查二进制代际 |
| K∈(64,96) 区间数据依赖错值 | 负 vl 无符号钳位 → 全宽 OOB 读（b86dd23 已修） | 同上，先确认二进制代际 |
| f16 结果 diff ~175（大得离谱） | smt.dot packed 结果没 unpack 就累加 | `smt.view(..., (1,1))` |
| f16 结果 diff 小但 ~12% 元素超差 | f16 截断累加（1.7 契约） | 查哪里把部分和降回 f16 |
| `KeyError: MICRO_M` | config pre_hook 给无 MICRO_* 参数的 kernel 注入（调用方绕过 port 直接 import 了 general kernel） | 查调用方 import 路径 |
| 换机器/换二进制后“修复失效” | TRITON_CACHE_DIR 没换，旧 .so 复用 | basics/04 §4.2 |
| rc=134 无 Python 栈 | spine-opt 层崩溃（bf16 matmul、TCM、pack 断言…） | basics/04 §4.4 离线复现 |

## 1.10 练习

1. 跑通 1.5，然后把 M/N/K 改成 `(128, 192, 96)`（全部不对齐 32）——
   block_ptr 的 boundary_check 应该继续 PASS。再试 K-loop 版，确认手写
   mask 也 PASS。
2. `BLOCK_M=BLOCK_N=32` 改成 64：grid 从 64 变 16，结果应不变。再改成
   `BLOCK_N=512`——预测会发生什么（1.7 的表），跑一次验证你的预测。
3. 用 `SPINE_TRITON_DUMP_PATH` 对比 BLOCK_K=64 与 BLOCK_K=256 的
   `.linalgdir`：数一数 `linalg.matmul`（或 mmt4d）出现的次数——体会
   “K 分块数 = 循环体复制/迭代次数”。
4. 思考题：为什么 lm_head（N=151936）不走 smt.dot？用 1.7 表里的 grid
   约束算一算 BN=256 时 grid_n 是多少。

下一篇：[02-layernorm.md](02-layernorm.md) —— 归约+逐元素复合算子。
