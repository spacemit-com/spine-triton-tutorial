# Op 05 — GEMV / mv（矩阵 × 向量）

**重要性：★★★★☆**。`y[N] = W[N,K] @ x[K]`——GEMM 的 M=1 特例，却是
**LLM decode 阶段的命脉**：自回归生成每步只有 1 个新 token，q/k/v/o_proj
和 MLP 的全部矩阵乘都退化成矩阵×向量。decode 快不快，第一决定因素就是
GEMV。它也是理解 K3“访存受限算子怎么压榨硬件”的最佳案例：从标准 tl 写法
→ raw 向量写法 → 矩阵引擎写法，三层形态在本章全部登场。

## 5.1 这个算子在算什么，为什么它和 GEMM 性格不同

定义：

```
y[i] = Σ_{k=0..K-1} W[i,k] · x[k]        # 每行一个点积
```

手算例子——`W = [[1,2,3],[4,5,6]]`（N=2, K=3），`x = [1,0,-1]`：

```
y[0] = 1·1 + 2·0 + 3·(-1) = -2
y[1] = 4·1 + 5·0 + 6·(-1) = -2
```

**与 GEMM 的本质区别是算术强度**。GEMM（M=N=K=256）算 2·256³ 次运算、
搬 3·256² 个元素——每个数据被复用几百次，是 compute-bound。GEMV 算
2·N·K 次运算、搬 N·K（权重）+ K（向量）个元素——**每个权重只用一次**，
约 1 FLOP/byte，是彻底的 **memory-bound**：性能上限 = 权重读取带宽，
优化目标是“搬得少、搬得顺”，不是“算得多”。

这决定了后面所有设计：x 向量小到可以常驻寄存器反复用；W 每行只碰一次；
输出只有 N 个标量。

**decode 现场**：Qwen3-0.6B 每 token 每层要过 7 个 proj GEMV（K 或 N 达
1024~3072），28 层——每生成一个 token 要把模型几乎全部权重从内存读一遍。
这就是“decode 速度 ≈ 内存带宽 / 模型大小”这条经验公式的来源。

## 5.2 标准 tl 写法：每 program 若干行

并行方案：**行间独立 → 按 N 切块**，`grid = (cdiv(N, BLOCK_N),)`，每个
program 算 BLOCK_N 行的点积。行内沿 K 分块循环。

（kernel 结构来源：FlagGems 主线 `ops/mv.py` 的 `mv_kernel`——生产级写法，
变量名按本章约定改成 W/x/y。）

```python
# gemv.py — y = W @ x（每 program BLOCK_N 行）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["N", "K"])
def mv_kernel(W, x, y, N, K,
              stride_wn, stride_wk, stride_xk, stride_yn,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    # 2D 下标广播：(BLOCK_N,1) 行 × (1,BLOCK_K) 列
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_k = tl.arange(0, BLOCK_K)[None, :]
    n_mask = offset_n < N

    W_ptrs = W + offset_n * stride_wn + offset_k * stride_wk
    x_ptrs = x + offset_k * stride_xk

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)   # ← 军规：f32 累加
    for k in range(0, K, BLOCK_K):
        k_mask = k + offset_k < K
        w = tl.load(W_ptrs, mask=n_mask & k_mask, other=0.0).to(tl.float32)
        v = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        acc += w * v                    # 逐元素乘，按“列 lane”分别累加
        W_ptrs += BLOCK_K * stride_wk
        x_ptrs += BLOCK_K * stride_xk

    row_sum = tl.sum(acc, axis=1)       # 最后一次性水平归约 → (BLOCK_N,)
    tl.store(y + offset_n * stride_yn, row_sum, mask=n_mask)


def mv(W, x):
    N, K = W.shape
    y = torch.empty((N,), device=W.device, dtype=W.dtype)
    BLOCK_N, BLOCK_K = 32, min(128, triton.next_power_of_2(K))
    grid = (triton.cdiv(N, BLOCK_N),)
    mv_kernel[grid](W, x, y, N, K,
                    W.stride(0), W.stride(1), x.stride(0), y.stride(0),
                    BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    N, K = 1024, 512
    W = torch.randn(N, K, dtype=torch.float16)
    x = torch.randn(K, dtype=torch.float16)

    y = mv(W, x)
    y_ref = torch.mv(W.float(), x.float()).half()
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("PASS  max diff =", (y - y_ref).abs().max().item())
```

```bash
python gemv.py
# PASS  max diff = 0.00xx
```

### 逐段精讲

- **`acc` 是 (BLOCK_N, BLOCK_K) 二维的，不是每行一个标量**。K 循环里每个
  `acc[n, k_lane]` 各自累加自己那一列的部分积，**循环结束后才
  `tl.sum(acc, axis=1)` 做一次水平归约**。这与 norm 家族的向量累加器同款
  （ops/02 §2.4）：把归约推迟到最后，循环体内全是纯向量乘加，没有归约
  开销。
- **`.to(tl.float32)` 出现在乘之前**，且 acc 是 f32——这不是风格问题，是
  踩过血的坑：spacemit port 版 mv 曾用 `A.dtype.element_ty`（f16）做 acc，
  K=160 分 5 块时 f16 部分和累积误差超容差，测试 1F。**mainline 用
  `tl.float32` 是对的**。排查这类 port 算子问题的标准手段就是对照 mainline
  逐行 diff（本例正是这么定位的）。
- **`other=0.0` 天然安全**：越界 lane 读成 0，乘加贡献为 0，对求和归约
  是吸收元（对照 softmax 必须 -inf——归约算子决定填充值）。
- **`do_not_specialize=["N", "K"]`**：Triton 会对 int 参数做值特化（=1 的
  参数直接变 constexpr）和 16 整除性特化。特化后不同 shape 各编译一份还
  算好，真正的坑是 K=BLOCK_K 恰好整块时特化路径曾把参数折成 constexpr
  导致下游拿不到 `.handle` 报错（raw 版 host 的同款问题）。显式关掉特化，
  一个二进制通吃所有 shape。
- **grid = cdiv(1024, 32) = 32 个 program**，安全范围（旧版 wheel 对超大
  grid 有上限，basics/04 §4.5；新版运行时自动分块。grid 大小与性能的
  关系见 5.4——反直觉）。
- **store 隐式 cast**：`row_sum` 是 f32，普通指针 store 自动降到 y 的
  f16（对照 block-ptr 的严格检查，ops/02 §2.6 坑 3）。

## 5.3 K3 硬件视角：M=1 为什么值得特殊照顾

标准 `tl.dot` 走矩阵引擎（mmt4d + pack/unpack，ops/01 §1.7），但 M=1 时：

- 矩阵引擎的原子块是 MICRO 16×8×32——M=1 意味着 16 行里 15 行是空的，
  **利用率 1/16**；
- pack/unpack 的搬运开销相对 1 行的计算量变得巨大。

K3 上 M×1 场景的正确通路是 **vfwmacc**（`vector_ext.batch_macc`，见
basics/06 §6.2）：`for k: acc += lhs[:,k]标量 × rhs[k,:]向量`——x 向量
广播成标量乘，W 的行留在向量寄存器，**零浪费**。但 batch_macc 有约束
“输出列数 n ≥64 且 %64==0”，而 mv 直接套是 n=1。解法是**转置映射**：

```
y[N] = W[N,K] @ x[K]          （n=1，不满足）
⇔ y[1,N] = x[1,K] @ W^T[K,N]  （n=N，满足）
   lhs = x  memref<1×K>       （标量广播）
   rhs = W^T vector<K×N>      （转置读）
```

`W^T` 不需要 host 侧真转置——raw 层的 `load_2d_t` 原语用 strided view
（`strides=[1,N]`，硬件 vlse strided 读）直接从行主序 W 里按转置布局读。
这就是 FlagGems 生产 dispatch 里 `M==1 → mm_m1_raw（vfwmacc.vf）/
mm_m1_t_raw（转置形态 vfwmacc.vv）` 的由来（ops/01 §1.8 的 dispatch 图）。

## 5.4 spine_raw 向量版：4 行展开 + strip-mine

（来源：`python/tests/raw/test_raw_mv_svector.py`，纯向量单元形态。）
raw 版把“每 program 算 BLOCK 行”的组织搬进 eDSL，核心结构：

```python
@tle.raw_kernel
def mv_block_style2(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True),
                    K: tle.index, row_base: tle.index, row_end: tle.index):
    nvl = tle.vconfig(-1, 1)             # f16 VLMAX = 64（K3 vlen=1024）
    Kfloor = (K // nvl) * nvl            # 满 tile 覆盖的 K 区间
    for ni in tle.range(row_base, row_end, 4):        # 每次 4 行
        acc0 = tle.vzero(f32); acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32); acc3 = tle.vzero(f32)
        for ki in tle.range(0, Kfloor, nvl):          # 主循环：满 tile 快路
            va = tle.vload(A, ki)                     # x 段，4 行共用！
            vb0 = tle.vload(B, ni * K + ki)           # B 第 ni 行段
            vb1 = tle.vload(B, (ni + 1) * K + ki)
            vb2 = tle.vload(B, (ni + 2) * K + ki)
            vb3 = tle.vload(B, (ni + 3) * K + ki)
            acc0 = tle.vmacc(acc0, vb0, va)           # acc += v * 标量广播
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        for ki in tle.range(Kfloor, K, nvl):          # 尾循环：0 或 1 次
            nvl = tle.vconfig(K - ki, 1)              # 收窄 → fill-0 pad
            ta = tle.vload(A, ki)                     # 独立临时名！
            tb0 = tle.vload(B, ni * K + ki)
            # ... tb1/tb2/tb3 同构，acc0..3 继续 vmacc
        tle.sstore(C, ni,     tle.vreduce_sum(acc0))  # 水平归约 → 标量 store
        tle.sstore(C, ni + 1, tle.vreduce_sum(acc1))
        tle.sstore(C, ni + 2, tle.vreduce_sum(acc2))
        tle.sstore(C, ni + 3, tle.vreduce_sum(acc3))
```

host 侧按 `BLOCK` 行切 grid，`tle.call()` 注入（basics/05 §5.6）：

```python
@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)   # 末 program 不越 N
    _sr_call(mv_block_style2, outputs=[], inputs=[B, A, C, K, row_base, row_end])
```

四个设计点：

1. **4 行展开**：`va`（x 的 K 段）load 一次喂 4 个 vmacc——x 的访存被
   摊薄 4 倍。这是 memory-bound 算子的经典手法（提高每字节搬运的计算
   复用）。
2. **主/尾双循环 strip-mine**：主循环不收窄 vconfig，走满 tile 快路（零
   pad 零分支）；尾循环 `vconfig(K-ki)` 收窄后 vload 自动 fill-0。整除时
   尾循环天然跑 0 次——用迭代次数代替分支（default path 无 if，basics/05
   §5.4）。
3. **尾循环用独立临时名 ta/tb\***：与主循环 va/vb\* 同名会触发 loop-env
   泄露类的 iter_arg 误判（basics/05 坑 4 的表亲）。
4. **phantom 行吸收**：`N % BLOCK != 0` 时末 program 的 4 行组可能超出
   N，`sstore` 是无条件的——host 把 C 分配到 `Np = ceil(N/BLOCK)*BLOCK`
   吸收 phantom 写，取结果切 `C[:N]`。**多分配、不越界、零 copy**。

任意 shape（K 非 64 倍数、N 非 BLOCK 倍数）测试矩阵全过——见该文件
`_SHAPES_ARB` 的 15 组。

## 5.5 矩阵引擎版（vfwmacc）：K3 实测最优形态

（来源：`test_raw_mv_cbm.py` + FlagGems `_spacemit/ops/mv.py` 生产版；
这一层读个轮廓即可，动手价值在前两层。）

按 5.3 的转置映射用 `batch_macc` 实现后，K3 实测出一串**反直觉**结论：

- **大 NB（少 program）反而更快**：NB=64（grid 16）0.165ms、NB=256
  （grid 4）0.118ms、NB=512/1024 持平——K3 上 **dispatch 开销主导**，
  “更多 program 更并行”的 GPU 直觉失效。生产选 NB=256。
- **kernel 内再切 4×64 sub-tile**：NB=256 的块内用 4 个独立
  `vector<1×64>` acc，每 K-block 做 4 次 pack + batch_macc。64 = f16 一个
  scalable 寄存器宽（macc 零吞吐损失）；32 不足一个寄存器宽、会触发
  编译器崩溃；每块 buf 4KB << L1 32KB。
- **A^T pack 用硬件 `spestruct.pack`**（对标 mm 的 pack 例程），取代软件
  linalg.generic 循环。旧软件 pack 在大 M 下是灾难（M=512 只有 mainline
  的 0.35x），硬件 pack 救回 1.17x（~3.3x 加速）。它的两个布局坑（都
  真实炸过）：`inner_tiles=[8, NB]` 不能写反（写反会越过 K 边界读出
  NaN）；单 tile ≤512 元素（更大的整块会触发编译器断言崩溃，必须切成
  K/8 个小 tile）。
- **动态 M**：kernel 内 `for kb in tle.range(nk)`（nk=M/32）K 块循环，
  列 stride 支持动态 SSA。约束：`M%32==0 && N%256==0 && f16 &&
  contiguous`。

**性能总账**（N=4096，vs FlagGems 通用路径）：M=32 时 2.10x，M 增大
优势递减（M=2048 1.15x），M=4096 略输 3%（0.97x）——M 大了以后它本质上
在跟 GEMM 抢通路，交给 tl.dot/smt.dot 更合适。**GEMV 专用 kernel 的
价值区间就是 M 小（decode）的场景**。

## 5.6 常见错误与判读

| 症状 | 根因 | 动作 |
|---|---|---|
| f16 结果随 K 增大误差滚大、个别 shape 1F | acc 用了 f16（port 版历史 bug） | 5.2 精讲第 2 条；对照 mainline diff |
| 不同 shape 各编译一份 / K 整除 shape 报 `.handle` | int 参数被值特化 | `do_not_specialize` |
| 尾部行结果对但堆损坏 / 后续 kernel 崩 | phantom 行越界写（N%BLOCK≠0） | C 分配到 Np 吸收（5.4 第 4 条） |
| raw kernel 报 unregistered dialect `vector_ext` | 用了 assembly 语法（中层转换工具不认识该 dialect，parse 失败） | generic form 字符串（basics/05 坑 5） |
| 矩阵引擎版 NaN | pack 的 `inner_tiles` 写反 | 5.5 的两个布局坑；tile ≤512 元素 |
| M=160 类 shape 失败但 M=128 过 | 超出验证范围 + port acc dtype bug 叠加 | 先修 acc dtype，再扩验证矩阵 |
| 输出全零 / 垃圾 | 旧版 wheel 丢弃超大 grid，或旧版编译器越界读（新版均已修） | 升级 wheel + 新 cache；basics/04 §4.5 分层排查 |

## 5.7 验证

```bash
python gemv.py                                            # 本章 tl 版
python3 -m pytest python/tests/raw/test_raw_mv_svector.py -q   # raw 向量版（含任意 shape）
python3 -m pytest python/tests/raw/test_raw_mv_cbm.py -q       # raw 矩阵引擎版
python3 python/tests/raw/perf_mv.py                            # 性能对照
```

golden：`torch.mv(W.float(), x.float())`；f16 rtol/atol 1e-2。IR 层自查
（dump 后）：后端输出 IR 无 `from/to_scalable` 残留、指令层产物（.ll）含
`riscv.vle/vse`（basics/04 §4.4）。

> 跑 raw 测试套件的整体注意事项（bindings PYTHONPATH、4 个诊断脚本要
> --ignore、独立 cache）见 basics/05 §5.11。

## 5.8 练习

1. 跑通 5.2。把 `BLOCK_N` 从 32 改到 4 和 128，N=4096 下观察 grid 变化与
   结果正确性；想想 5.5 的“dispatch 开销主导”预言哪种更快（在 K3 上
   实测）。
2. 把 acc 的 dtype 改成 `W.dtype.element_ty`（f16），K 取 1024 重跑——
   复现 port 历史 bug 的误差签名（对照 5.2 精讲第 2 条）。改回来。
3. 跑 `test_raw_mv_svector.py`，然后数一数 style2 kernel 里 `tle.vload(A,...)`
   出现几次、`tle.vload(B,...)` 几次——解释为什么 4 行展开能把 x 的访存
   摊薄（5.4 第 1 条），并算 K=512 时 A 和 B 各被读多少字节。
4. 思考题：为什么 GEMV 的 `other=0.0` 安全而 softmax 必须 `-inf`？用
   “归约算子的单位元”框架（basics/03 §3.3）分别写出 sum/max/exp-sum 三个
   归约的单位元。
5. 进阶：把 5.2 改成 M>1 的 mini-GEMM（`x` 变 (M,K) 矩阵、acc 变三维或
   循环 M），对照 ops/01 的 tl.dot 版，讨论 M 多大时应该切换到 tl.dot
   通路（提示：5.5 的性能总账）。

下一篇：[06-reduction.md](06-reduction.md) —— sum/max/argmax 的归约全景。
