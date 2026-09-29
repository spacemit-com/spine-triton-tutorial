# Op 05 — GEMV / mv（M=1 矩阵向量乘）

`C[N] = A[N,K] @ B[K]`。LLM decode 阶段每 token 的全部 proj 都是 GEMV，
是 decode 性能的决定算子。它也是 K3 矩阵引擎 **vfwmacc** 通路的最佳教学案例：
标准 tl.dot 在 M=1 时利用率极低，正确形态需要显式映射。

## 为什么 tl.dot 不行

矩阵引擎的 `vector_ext.matmul`（vfwmadot）单条 m1 形态只算 `acc[0,0]`；
linalg.mmt4d 通路要 pack 物化 A/B。M×1 场景唯一能做到 **B 零搬运**（寄存器
广播）的是 `vector_ext.batch_macc` → **vfwmacc**：

```
batch_macc(lhs: memref<m×k>, rhs: vector<k×n>, acc: vector<m×n>)
  → acc[m,n] += Σ_k lhs[m,k] · rhs[k,n]
展开语义：for k: acc += lhs[:,k](标量) * rhs[k,:](向量)
约束：n % 64 == 0 且 n ≥ 64（f16 一个 scalable 寄存器 = 64 元素）
```

## 转置映射（关键一步）

`C[N] = A[N,32] @ B[32]` 直接套 batch_macc 是 n=1，不满足约束。转成：

```
C[1,N] = B[1,32] @ A^T[32,N]
lhs = B   memref<1×32>        （标量广播）
rhs = A^T vector<32×N>        （转置读）
acc = C   vector<1×N>
```

`A^T` 用 `load_2d_t` 原语从原始行主序 A 直接 strided view
（`strides=[1,M]`，vlse strided 读），**不需要 host pack**：

```python
# spine_raw 2D 原语（codegen.py）
batch_macc / view_2d / load_2d[_at] / load_2d_t / pack_2d_t
splat_2d / store_2d[_at] / matmul / extract_elem
```

## 最终架构（K3 实测最优）

（参考：`python/tests/raw/test_raw_mv_svector.py`、`test_raw_mv_cbm.py`、
`bench_mv_fused_vs_svector.py`）

- **host 侧 NB=256**：grid=(N/256,)。反直觉但实测：**大 NB（少 program）反而
  快**——NB=64(grid16) 0.165ms、NB=256(grid4) 0.118ms、NB=512/1024 持平。
  K3 上 dispatch 开销主导，"更多 program 更并行" 的 GPU 直觉不成立。
- **kernel 内 4×NB_sub=64 sub-tile**：4 个独立 `vector<1×64>` acc，每 K-block
  做 4 次 pack + batch_macc。NB_sub=64 的 buf=4KB << L1(32KB)；
  NB_sub=32 是 sub-vscale 会 crash；macc(NB=64) 即 1 条 vfwmacc 零吞吐损失。
- **A^T pack 用硬件 `spestruct.pack`**（对标 mm 的 `spe_pack_f16_m{NB}_n8`），
  取代软件 linalg.generic 循环。布局关键坑：dest 连续 K×NB 须
  `inner_dims_pos=[1,0]` + `inner_tiles=[8, NB]`（**不是 [NB, 8]**——写反会越过
  K 边界读出 NaN）；单 tile 元素数 ≤512（64×8），整块 [32×64]=2048 会触发
  SplitLargeShapeScalable assert crash，必须切 K/8 个 tile。
- **动态 M**（K 块循环）：`for kb in tle.range(nk)`（nk=M/32）累加到 acc
  iter_arg；B 取 `koff:koff+32`，A^T offset=`col*M+koff`；列 stride 支持动态
  SSA（`strided<[1,?]>` 传 %M）。host kernel 传 M、NK 必须
  `@triton.jit(do_not_specialize=["M","NK"])`——否则 NK=1 被值特化成 constexpr。
  约束：`M%32==0 && N%256==0 && f16 && contiguous`。

## 性能参考（N=4096，K3）

| M | vs FlagGems 通用路径 |
|---|---|
| 32 | 2.10x~2.94x |
| 32~2048 | 全胜，随 M 增大递减（1.15x） |
| 4096 | 0.97x（略输 3%） |

kernel 0.118ms（对照 reduce_add 0.3ms、普通实现 0.4ms）。软件 pack 旧版在
M=512 只有 0.35x——spestruct.pack 把大 M 救回 1.17x（~3.3x 加速）。

## raw kernel 里的写法要点

```python
# vector_ext op 必须 generic form（spine-triton-opt 未注册该 dialect，
# assembly 语法 parse 失败 → dsl_region 不展开 → 漏到 linalg 报 unregistered）
"vector_ext.batch_macc"(%lhs, %rhs, %acc) : (...) -> ...
"spestruct.pack"(...)   # 硬件 pack 例程，同理 generic form
```

f16 输入 → f32 acc（vfwmacc widening）→ fptrunc → f16 输出。

## 与 mm 的关系

FlagGems 生产 dispatch：`M==1` 走 `mm_m1_raw`（vfwmacc.vf，B 广播）/
`mm_m1_t_raw`（B 转置形态 vfwmacc.vv）；`M>1` 走 smt.dot / tl.dot /
vfwmacc.vv 链（见 ops/01）。lm_head（N=151936）即使 M>1 也保留 vfwmacc.vv。

## 验证

```bash
python3 -m pytest python/tests/raw/test_raw_mv_svector.py -q
python3 -m pytest python/tests/raw/test_raw_mv_cbm.py -q     # 矩阵引擎版
python3 python/tests/raw/perf_mv.py                           # 性能
```

golden `torch.mv` / `torch.matmul`；f16 rtol/atol 1e-2。
调试时留意 e2e 后 IR 无 `from/to_scalable` 残留、llc 输出含 `riscv.vle/vse`。

> 注意 M=160 一类非 32 倍数/超出验证范围的 shape：spacemit port 的 mv 曾有
> acc 用 f16 累加的 port bug（mainline 是 tl.float32）——port 算子对照 mainline
> 找偏差是标准 triage 手段。

下一篇：[06-reduction.md](06-reduction.md)
