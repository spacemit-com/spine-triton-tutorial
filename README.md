# spine-triton 教程（零基础版）

这是一套**面向完全新手**的 spine-triton 教程。

spine-triton 是 SpacemiT（进迭时空）的 Triton 编译器后端：它让你用 Python 写
高性能算子（kernel），编译成 RISC-V 机器码，跑在 K3 芯片上。默认玩法很直接：
在 K3 板卡上 `pip install` 官方 wheel，然后像写 PyTorch 一样写 kernel。
（进阶篇会讲怎么在 x86 开发机上交叉编译出板上可用的 wheel。）

## 你需要的基础

- 会 Python，用过 PyTorch（知道 `torch.Tensor`、`torch.randn`、shape/dtype 是什么）。
- **不需要**会 Triton、CUDA、MLIR、RISC-V、汇编——这些都会在教程里从零讲。
- 一块 SpacemiT K3 板卡（riscv64 Linux，Python 3.12）。进阶的交叉编译流程另需一台 x86 Linux 开发机。

## 怎么使用这套教程

1. **按顺序读 Part 1**（basics/，共 6 章）。每章末尾的“动手”环节一定要真的跑一遍，
   跑通了再往下走。
2. Part 2 的算子章节（ops/）可以按需挑读，但建议至少读完 01（GEMM）、
   04（Softmax）、06（Reduction）——它们是理解其余算子的基础。
3. 所有代码块都是**完整可运行**的（除非明确标注为“片段”）。遇到报错，
   先查当章的“常见错误”一节，再查 [basics/04](basics/04-running-and-debugging.md)
   的排错手册。
4. 进阶章节（basics/05、basics/06）第一遍读不懂是正常的，标记一下跳过，
   做完 ops 前几章再回来。

## 目录

### Part 1 — 基础（Skills）

| 章 | 文件 | 你将学会 |
|---|---|---|
| 1 | [basics/01-introduction.md](basics/01-introduction.md) | kernel 是什么、为什么需要 Triton、spine-triton 把 Python 变成机器码的完整路径、K3 芯片 60 秒速览 |
| 2 | [basics/02-installation.md](basics/02-installation.md) | 在 K3 上 pip 安装、验证与排错；（进阶）在 x86 上交叉编译 wheel 再装到板上 |
| 3 | [basics/03-quickstart.md](basics/03-quickstart.md) | 写出并跑通第一个 kernel（向量加法），逐行理解 grid/program/block/mask，完成 4 个练习 |
| 4 | [basics/04-running-and-debugging.md](basics/04-running-and-debugging.md) | kernel 运行时到底发生了什么：编译缓存、IR dump、环境变量、报错定位三板斧 |
| 5 | [basics/05-spine-raw-edsl.md](basics/05-spine-raw-edsl.md) | （进阶）spine_raw eDSL：用向量级原语直接控制 RVV，绕过标准路径的限制 |
| 6 | [basics/06-smt-matrix-engine.md](basics/06-smt-matrix-engine.md) | （进阶）smt 扩展：K3 矩阵引擎、MICRO tile、TCM，GEMM 性能形态的底层机制 |

### Part 2 — 算子实战（按重要性排序）

| # | 算子 | 文件 | 一句话简介 |
|---|---|---|---|
| 1 | GEMM（矩阵乘） | [ops/01-gemm.md](ops/01-gemm.md) | 一切深度学习计算的核心；三个层次的写法：tl.dot → block_ptr → 矩阵引擎 |
| 2 | LayerNorm | [ops/02-layernorm.md](ops/02-layernorm.md) | “归约 + 逐元素”复合算子的代表；f32 累加军规的主战场 |
| 3 | RMSNorm | [ops/03-rmsnorm.md](ops/03-rmsnorm.md) | LayerNorm 的简化版，LLM 主流；reduce → rsqrt → 广播模式 |
| 4 | Softmax | [ops/04-softmax.md](ops/04-softmax.md) | 数值稳定性入门必修；K3 上 fp16 + tl.exp 的第一大坑 |
| 5 | GEMV（矩阵×向量） | [ops/05-gemv.md](ops/05-gemv.md) | LLM decode 的命脉；vfwmacc 转置映射的完整案例研究 |
| 6 | Reduction（归约） | [ops/06-reduction.md](ops/06-reduction.md) | sum/mean/max/argmax 通用模式；行长与 vscale 的边界行为 |
| 7 | Elementwise 与融合 | [ops/07-elementwise-fused.md](ops/07-elementwise-fused.md) | add/silu/gelu；math 函数链路、libdevice、epilogue 融合 |
| 8 | Attention | [ops/08-attention.md](ops/08-attention.md) | QK^T + softmax + AV；GQA per-head kernel、性能收益的边界在哪 |
| 9 | Cumsum / Scan | [ops/09-cumsum-scan.md](ops/09-cumsum-scan.md) | 前缀和/前缀扫描；scan 类算子的组织方式与平台边界行为 |
| 10 | Cat / 数据搬运 | [ops/10-cat.md](ops/10-cat.md) | concat/stack 家族；纯访存算子的组织方式与指针分支坑 |

> 注：量化类算子（fp8/int8 quant）当前工具链尚未支持，本教程暂不包含。

## 代码参照仓库

教程代码取材自真实仓库，正文会标注来源路径：

- `spine-triton/python/examples/` — 可运行示例（mm_block_ptr.py、test_smt_mm.py、test_layernorm.py、test_softmax.py…）
- `spine-triton/python/tests/` — 单元测试（test_blas_ops.py、test_norm_ops.py、test_reduction_ops.py…）
- `spine-triton/python/tests/raw/` — spine_raw eDSL 测试（test_raw_layernorm.py、test_raw_mv_svector.py…）
- `FlagGems`（`_spacemit/ops/`）— 生产级 K3 优化算子（mm.py、softmax.py、layernorm.py、flash_attention.py…）

## 全局约定

- **“K3”**：SpacemiT K3 芯片（arch id `0xA064`，向量寄存器 vlen=1024 bit，
  f16 下一个向量寄存器装 64 个元素，矩阵引擎 f16 MICRO tile = M/K/N 16/8/32）。
- **dtype 军规**：f16 输入、f32 累加、输出前才 cast 回 f16。全书通用，不再重复解释。
- **运行环境**：Python 代码默认在 K3 板卡上直接运行（basics/02 装好环境即可），
  除程序开头的 driver 激活两行外，**不需要设置任何环境变量**。
- 编译器后端工具与运行时库随 wheel 一起安装在
  `triton/backends/spine_triton/{bin,lib}/` 下，平时不需要手动调用；排查编译
  问题时用 `SPINE_TRITON_DUMP_PATH` 把中间产物落盘查看（basics/04）。

## License

MIT（与 spine-triton 一致）。
