# spine-triton Tutorial

面向 SpacemiT K3（RISC-V + RVV + 矩阵引擎）的 Triton 编译器 **spine-triton** 入门教程。

spine-triton 让你用熟悉的 `@triton.jit` Python DSL 写 kernel，编译到 RISC-V
可执行代码（RVV scalable vector / K3 vfwmacc 矩阵单元），在 K3 板卡上原生运行，
或在 x86 主机上通过 QEMU RPC 远程执行。

## 目录

### Part 1 — 基础（Skills）

| 文件 | 内容 |
|---|---|
| [basics/01-introduction.md](basics/01-introduction.md) | spine-triton 是什么：架构、编译流水线、三层语言接口、相关仓库 |
| [basics/02-installation.md](basics/02-installation.md) | 安装：预编译 wheel、x86/RPC 源码构建、riscv64 交叉编译、FlagTree wheel |
| [basics/03-quickstart.md](basics/03-quickstart.md) | 第一个 kernel：vec add、CPUDriver、grid 语义、dtype 约束 |
| [basics/04-running-and-debugging.md](basics/04-running-and-debugging.md) | 运行形态（native / QEMU RPC）、环境变量、IR dump、缓存管理、常见坑 |
| [basics/05-spine-raw-edsl.md](basics/05-spine-raw-edsl.md) | spine_raw eDSL：向量级原语直接产 linalg/vector MLIR，LLVM-direct 模式 |
| [basics/06-smt-matrix-engine.md](basics/06-smt-matrix-engine.md) | smt 扩展：K3 矩阵引擎（vfwmadot/vfwmacc）、TCM、MICRO tile、alloc/view |

### Part 2 — 算子实战（按重要性排序）

| # | 算子 | 文件 | 核心内容 |
|---|---|---|---|
| 1 | GEMM (mm) | [ops/01-gemm.md](ops/01-gemm.md) | tl.dot → block_ptr → smt 矩阵引擎三层写法、MICRO tile、tail 处理 |
| 2 | LayerNorm | [ops/02-layernorm.md](ops/02-layernorm.md) | fused fwd kernel、f32 累加约定、spine_raw 三趟版 |
| 3 | RMSNorm | [ops/03-rmsnorm.md](ops/03-rmsnorm.md) | reduce → rsqrt → 广播模式、fused add+rmsnorm |
| 4 | Softmax | [ops/04-softmax.md](ops/04-softmax.md) | row-parallel、数值稳定三趟、fp16 的 tl.exp dtype 坑 |
| 5 | GEMV (mv) | [ops/05-gemv.md](ops/05-gemv.md) | M=1 的 vfwmacc.vf 转置映射、B 零搬运、动态 M K-block 循环 |
| 6 | Reduction | [ops/06-reduction.md](ops/06-reduction.md) | sum/mean/max/argmax 通用模式、行长与 vscale、精度容差 |
| 7 | Elementwise & Fusion | [ops/07-elementwise-fused.md](ops/07-elementwise-fused.md) | pointwise 家族、silu/gelu、libdevice shim、constexpr 解包坑 |
| 8 | Attention | [ops/08-attention.md](ops/08-attention.md) | GQA QKT+AV kernel、per-head grid、D=128 分 chunk、SDPA 注册路径 |
| 9 | Cumsum / Scan | [ops/09-cumsum-scan.md](ops/09-cumsum-scan.md) | Hillis-Steele scan、协程栈需求、整型 dtype 已知限制 |
| 10 | Quantization | [ops/10-quantization.md](ops/10-quantization.md) | per-token-group-quant-fp8、bf16 向量限制、dtype 判读方法 |

## 学习路线建议

1. **只跑标准 Triton kernel**：读 01–04，然后从 `ops/01-gemm.md` 开始。
   标准 `tl.*` 写法在 spine-triton 上与 GPU Triton 基本一致，重点看每章的
   “K3 关键点 / 已知坑”一节。
2. **要压榨矩阵引擎性能**（mm / mv / attention）：加读 basics/06，然后
   ops/01、05、08。
3. **要写向量级定制 kernel**（fuse、超越函数、特殊布局）：加读 basics/05，
   然后 ops/02、04、06 里的 spine_raw 版本。

## 代码参照

教程里的代码全部取材自真实仓库，正文会标注来源路径：

- `spine-triton/python/examples/` — 可运行示例（mm_block_ptr.py、test_smt_mm.py、test_layernorm.py、test_softmax.py…）
- `spine-triton/python/tests/` — 单测（test_blas_ops.py、test_norm_ops.py、test_reduction_ops.py…）
- `spine-triton/python/tests/raw/` — spine_raw eDSL 测试（test_raw_layernorm.py、test_raw_mv_svector.py…）
- `FlagGems`（spacemit 后端 `_spacemit/ops/`）— 生产级算子注册实现（mm.py、softmax.py、layernorm.py、flash_attention.py…）

## 约定

- “K3” 指 SpacemiT K3（arch id `0xA064`/A100，vlen=1024，f16 MICRO tile M/K/N = 16/8/32）。
- 所有 kernel 默认 f16 输入 / f32 累加，这是数值正确性的第一约定。
- 文中 `spine-opt`、`llc` 等工具来自 spine-mlir 发行包，随 spine-triton 安装部署。

## License

MIT（与 spine-triton 一致）。
