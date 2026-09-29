# 06 — smt 扩展与 K3 矩阵引擎

`triton.language.extra.smt` 是对 K3 矩阵引擎（vfwmadot / vfwmacc 指令族）和
TCM 的显式控制接口。mm/bmm/attention 这类矩阵乘算子要跑出矩阵引擎性能，
最终都会落到这一层（由编译器自动或你手动）。

## 硬件背景

- **vfwmacc**（`vector_ext.batch_macc` lower 而来）：`acc[m,n] += Σ_k lhs[m,k]·rhs[k,n]`，
  f16 乘、f32 累加。M×1（GEMV）和 per-row 场景的主力，rhs 可寄存器广播零复制。
- **vfwmadot**（`vector_ext.matmul` / mmt4d lower 而来）：矩阵引擎点积通路。
  注意单条 m1 形态只算 `acc[0,0]`——直接拿它做完整 matmul 是错的，完整 GEMM
  要走 mmt4d/linalg.matmul 的 pack+unpack 编排（编译器负责）。
- **MICRO tile**：矩阵引擎的原子计算块。K3（arch `0xA064`）f16 合法配置为
  **M/K/N = 16/8/32**。运行时可查：`tle.mma_cube("f16")`。MICRO 值必须来自
  当前 arch 的合法表（config_pre_hook 里有按 arch 的表），传错会静默算错或崩。
- **TCM**：每 worker 256KB。大 tile 中间量 spill 单次分配超过它就 **NULL →
  段错误**（生成的 kernel 不检查分配结果）。经验边界：8192 元素 tile OK、
  16384 元素崩。BN=512 的 tl.dot/smt.dot crash、BK=K 一次性展开 hang 都是这族。

## smt API

```python
import triton.language.extra.smt as smt

smt.alloc(shape, type=...)        # TCM 分配（默认 dtype=f32！f16 tile 必须显式 type=）
smt.view(desc, offsets, shape, micro)   # subview / pack 视图
smt.descriptor_load(block_ptr, offsets) # 描述符块加载
smt.dot(a, b)                     # 矩阵引擎点积（结果是 packed MICRO-tile 格式！）
smt.parallel(...)                 # 并行循环（range 子类，带 bind_sub_block 语义）
smt.compile_hint(ptr, name, val)  # 编译提示
```

两个关键语义：

1. **`smt.dot` 返回 packed 格式**，不能直接与普通 `tl.zeros` 累加器相加
   （相加结果 diff 巨大）。必须先 unpack 再累加：

   ```python
   partial = smt.view(smt.dot(a, b), (0, 0), (BM, BN), (1, 1))
   acc += partial
   ```

2. **`smt.parallel(bind_sub_block=...)`**：bind_sub_block=True 时编译器把循环
   降级成 `scf.forall` → `spert.parallel`，这条链在当前 spine-opt 上有
   `UseDefLists.h:198 use_empty()` 崩溃；**默认值已改为 False**（循环保持普通
   `scf.for`，端到端可用）。写新 kernel 不要显式传 True。

## SPLIT_M / SPLIT_MN / SPLIT_K 三种编排

（来源：`python/examples/test_smt_mm.py`，三分支 512³ f16 均已端到端验证）

GEMM tile 内再切 MICRO 块的三种模式：

- **SPLIT_M**：M 维切 sub-block，b tile 经 `smt.descriptor_load` +
  `smt.view(..., (MICRO_K, MICRO_N))` 做 pack 视图复用。
- **SPLIT_MN**：M、N 双维切；注意 `smt.alloc` 的 dtype 必须显式
  `type=b_ptr.dtype.element_ty`（默认 f32，f16 store 会撞
  “Block element type(fp32) and value element type(fp16) mismatch”——
  block-ptr store 是严格类型检查，无隐式 cast）。
- **SPLIT_K**：K 维切，走 `tl.range` 普通循环累加。

BLOCK_SIZE_K 受 TCM 约束：BK=256 时“noloop BK=K”需要 608KB 会溢出 hang，
K 大时必须分块。

## mbarrier（异步同步）

`smt.mbarrier/alloc/barrier_arrive/barrier_wait` 编译出对
`spine_mbarrier_*` 四符号的引用，需要 runtime 侧实现：

- QEMU RPC：由 rpc-runtime 的 **Event 适配层**提供（基于 libspert 0.6.3 公开
  Event 接口，协程感知阻塞；事件池 64 槽，耗尽降级 spin）。
- 官方 libspert **没有** mbarrier 协议对象（barrier 类 API 只有融合
  arrive+wait 的不透明句柄）。
- 实践建议：**新 kernel 尽量不依赖 mbarrier**——`bind_sub_block=False` 下
  `smt.parallel` 是顺序 `scf.for`，同 program 内 TCM 的 store→load 按程序序
  执行，天然无需同步（test_smt_mm 的去 mbarrier 版本数值仍正确）。

## 什么时候需要手写 smt？

大多数情况不需要——`tl.dot` + block_ptr 的标准 GEMM 写法会被编译器自动
lower 到矩阵引擎（mmt4d 通路）。手写 smt 的动机只有：

1. 显式控制 TCM 分配与 pack 布局（如 SPLIT_N 的 b tile 中转）；
2. 编译器自动通路失败/次优时的降级或提优手段；
3. spine_raw 层的矩阵引擎 intrinsic（`batch_macc`/`vmadot`，见 ops/05）。

无论哪层，验收标准一致：`nm -D --undefined-only kernel.so` 无意外符号、
数值 vs torch golden（f16 rtol/atol 1e-2）、性能对照 `perf.py`/proton。

下一篇：进入算子实战 [../ops/01-gemm.md](../ops/01-gemm.md)
