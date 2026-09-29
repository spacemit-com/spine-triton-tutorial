# Op 01 — GEMM (mm)

一切 LLM 推理算子的地基：`C[M,N] = A[M,K] @ B[K,N]`（及 `addmm`：
`C = beta·C + alpha·A@B`）。K3 上 GEMM 走矩阵引擎（vfwmadot/vfwmacc），
是性能敏感度最高、也是编译链路覆盖最全的算子。

## 三个层次的写法

### 层次 1：标准 tl.dot K-loop（先保证正确）

```python
@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
              sam, sak, sbk, sbn, scm, scn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)      # f32 累加器——军规
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)                          # f16 输入 → f32 输出
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    c = acc.to(c_ptr.dtype.element_ty)               # 只在最终输出 cast 回 f16
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, c,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

**f16 累加契约**：`tl.dot(f16, f16) → f32` 的 K 归约必须 f32 累加。历史 bug：
循环内 dot 曾生成 `linalg.matmul outs(f16-fill)`——每 K 块部分和先截断 f16 再
累加，fail_frac=0.127。已修（mixed named matmul：f16 ins / f32 outs），波及
所有 K-blocking 的 f16 mm/bmm/conv。**写 kernel 时任何“把中间和存回 f16 再累加”
的变体都违反契约**。数值验证方法：CPU 仿真三模型对比（emuA=f16 精确乘积+f32
累加=正确语义；emuB=f16 舍入乘积；gems 实测），看 bit 一致率。

### 层次 2：block_ptr（边界处理交给编译器）

（来源：`python/examples/mm_block_ptr.py`，CI 标准 case 256³ f16）

```python
a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=[M, K], strides=[sam, sak],
                                offsets=[pid_m * BM, 0], block_shape=[BM, BK],
                                order=[1, 0])
# 循环内: a = tl.load(a_block_ptr); a_block_ptr = tl.advance(a_block_ptr, [0, BK])
```

block-ptr 的 load/store 自动处理越界（border padding），但注意 **store 是严格
类型检查**：`tl.store(block_ptr, f32_val)` 对 f16 block 直接报 mismatch，必须
显式 `.to()`。

### 层次 3：smt 矩阵引擎（性能形态）

（来源：`python/examples/test_smt_mm.py`）SPLIT_M/SPLIT_MN/SPLIT_K 三分支 +
MICRO tile 参数化，见 [basics/06](../basics/06-smt-matrix-engine.md)。骨架：

```python
if SPLIT_M:
    b_desc = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(b_desc, (0, 0), (BK, BN), (MICRO_K, MICRO_N))   # pack 视图
    for i in smt.parallel(0, sub_num, bind_sub_block=False):      # 默认 False！
        a_sub = ...                                                # A 的 sub-tile
        partial = smt.view(smt.dot(a_sub, b), (0, 0), (SUB_M, BN), (1, 1))  # unpack 再累加
        acc += partial
```

K3 f16 MICRO = M/K/N **16/8/32**（`tle.mma_cube("f16")` 可查）。smt.dot 路径
相对 vfwmacc.vv 可达 **6x**（proj mm 5 shape 合计 3.51ms vs 21.49ms）。

## K3 关键约束

| 约束 | 数值/规则 | 违反后果 |
|---|---|---|
| TCM scratch | 256KB / worker | 大 tile 中间量单次 spill 超量 → NULL → 段错误（BN=512 crash、BK=K 608KB hang 均此族） |
| BN 上限 | output tile ≤ 32×256（8192 elem）安全；16384 elem 崩 | segfault（tl.dot 与 smt.dot 同限，共享 mmt4d lowering） |
| grid | ≤512（libspert <0.6.3）；≥0.6.3 自动分块无限制 | 0.6.0/0.6.2 静默丢弃 → 输出未初始化内存；更坑的是丢弃 config 计时~0 反而**赢得 autotune** |
| N 特别大 | N > 131072（lm_head）不走 smt.dot | 单 launch grid 爆 / chunked 2 launches dispatch 开销反超；用 vfwmacc.vv 路径 |
| dtype | f16/f32；**bf16 linalg.matmul 撞 MatmulConfigAnalysis UNREACHABLE** | rc=134（平台豁免项） |

## 生产 dispatch 链（FlagGems `_spacemit/ops/mm.py` 参考架构)

```
M == 1 ?  ──是──► mm_m1_raw (vfwmacc.vf, B 寄存器广播) / mm_m1_t_raw
   │否
   ▼
_mm_smt_path (smt.dot, N ≤ 131072)          # proj mms
   ▼ 不满足
_mm_tldot_path (tl.dot, N ≤ 4096)           # smt 失败备用
   ▼ 不满足
_mm_transposed_tail (vfwmacc.vv, any N)     # lm_head
```

**tail 处理**（N 不对齐 NB 时）：`N % 8 == 0` 即可命中 fast path——main tile
`(N//NB)*NB` 走矩阵引擎 kernel，tail 走 native `torch.addmm`（beta=0 写 tmp 再
copy）。kernel 带 `CS`（c_row_stride=原始 N）参数让 main tile 直接写入 full-N
输出的前 n_main 列，无需 contiguous temp。

## 验证与性能基准

```bash
python3 python/examples/mm_block_ptr.py          # 256³ f16 CI 标准 case
python3 python/examples/test_smt_mm.py           # SPLIT_M/MN/K 三分支 512³
python3 -m pytest python/tests/test_blas_ops.py -q
```

性能参考（Qwen3-0.6B，K3）：vfwmacc-only ~21 tok/s → smt.dot dispatch 链
~40 tok/s（prefill 1.91x）；再叠加 `logits_to_keep=1`（transformers 原生 API，
lm_head 只算最后一个 token）→ ~52 tok/s。

profiling 用 CPU proton（`PROTON_KERNEL_CAPTURE=1`，host region 计时经
`proton_enter_kernel/exit_kernel`），拆到 lm_head/attn/mlp 粒度。

## 常见失败速查

| 症状 | 根因 |
|---|---|
| 输出全零 / 99.9% 垃圾 | grid>512 被 0.6.0/0.6.2 丢弃；或 f32 unpack 写外部 dest 被丢（旧 spine-mlir bug，b86dd23 已修——先确认二进制代际） |
| K∈(64,96) 区间数据依赖错值 | 负 vl 无符号钳位到 VLMAX 全宽 OOB（b86dd23 已修，同上） |
| f16 结果 diff ~175 | smt.dot packed 结果直接与普通 acc 相加，忘了 unpack |
| `KeyError: MICRO_M` | config pre_hook 对无 MICRO_* 参数的 kernel 注入（调用方绕过 port 直接 import general kernel） |
| 换机器/换二进制后“修复失效” | TRITON_CACHE_DIR 没换，旧 .so 复用 |

下一篇：[02-layernorm.md](02-layernorm.md)
