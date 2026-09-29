# 01 — spine-triton 是什么

## 一句话

**spine-triton** 是 SpacemiT 基于 [microsoft/triton-shared](https://github.com/microsoft/triton-shared)
的 Triton CPU/RISC-V 后端：用 `@triton.jit` 写 kernel，编译成 RISC-V 共享库
（RVV 向量指令 + K3 矩阵引擎指令），通过 spine-runtime 在 K3 板卡上并行执行。

```
@triton.jit Python
      │  Triton 前端（vendored triton）
      ▼
    TTIR (tt dialect)
      │  spine-triton-opt：ptr 分析 / 结构化分解
      │  （TritonToPtr → PtrToStructured → StructuredToMemref → TritonToLinalg）
      ▼
  linalg + memref MLIR
      │  spine-opt（spine-mlir 发行包）：--spine-triton-e2e-pipeline
      │  bufferize → ConvertToScalableVector（RVV）→ 矩阵引擎 lowering
      ▼
    LLVM IR (.ll)
      │  llc -mtriple=riscv64
      ▼
   kernel .so  ──►  spine-runtime (libspert) 执行
                    ├─ native：K3 板上直接跑
                    └─ RPC：x86 host 编译，QEMU guest 执行
```

## 硬件目标：SpacemiT K3

- RISC-V，vlen=1024，RVV 1.0 scalable vector（f16 一个 scalable 寄存器 = 64 元素）。
- **矩阵引擎**：`vfwmadot` / `vfwmacc` 指令族（f16 乘、f32 累加），由 spine-mlir 的
  `vector_ext.matmul` / `vector_ext.batch_macc` 等 op lower 而来。
- **TCM**（tightly-coupled memory）：每 worker 256KB scratch，大 tile 中间量会
  spill 到这里；**超过 256KB 会分配失败**（详见 basics/04 的坑清单）。
- arch id：`0xA064`（K3/A100）。注意 QEMU RPC 模式下 host 侧探测到的 arch 与
  编译 target 的 arch 是两条路径（后者来自 `SPACEMIT_EP_QEMU_SET_CORE_ARCH`）。

## 三层语言接口

spine-triton 提供三种写 kernel 的方式，能力递增、通用性递减：

### 1. 标准 `tl.*`（默认，优先使用）

与 GPU Triton 写法一致：`tl.load/store`、`tl.dot`、`tl.sum`、`tl.make_block_ptr`…
绝大多数算子应该先用这条路——编译器自动做向量化和矩阵引擎映射。

### 2. `smt` 扩展（矩阵引擎 / TCM 显式控制）

`triton.language.extra.smt`：`smt.alloc`（TCM 分配）、`smt.view`（subview/pack 视图）、
`smt.descriptor_load`、`smt.dot`、`smt.parallel`（带 `bind_sub_block` 语义的循环）。
用于显式控制 MICRO tile 切分和矩阵引擎数据通路。见 basics/06。

### 3. `spine_raw` eDSL（向量级 / LLVM 级）

`triton.language.extra.spine_raw`：Python AST 直接翻译成 linalg/vector MLIR
（`tle.vload/vstore/vmacc/vreduce_sum/vexp/...`），或 LLVM-dialect 直发
（`tle.llvm_*`，可写 `llvm.riscv.*` RVV intrinsic）。用于标准路径表达不了的
融合、超越函数、特殊内存布局。见 basics/05。

## 相关仓库

| 仓库 | 角色 |
|---|---|
| `spine-triton` | 本教程主角：Triton 前端 + triton-shared 中层 + spacemit/spine_triton backend |
| `spine-mlir` | MLIR 编译器后端：`spine-opt`（e2e pipeline）、`llc`、`mlir-translate`；矩阵引擎/RVV lowering 都在这里 |
| `spine-runtime` | 执行引擎 `libspert`：tile 调度、协程 worker、TCM 池、grid 分块 dispatch（≥0.6.3） |
| `spine-triton-rpc-runtime` | QEMU RPC server：x86 编译产物送到 QEMU guest 内执行 |
| `FlagGems` / `FlagTree` | 上层算子库：`_spacemit/ops/` 下是生产级 K3 优化算子（mm、softmax、layernorm、flash_attention…），按注册路径接入 torch |

三者的版本耦合很紧（LLVM 版本、libspert 版本、spine-opt 代际），排错时先确认
整条链的二进制版本，见 basics/04。

## 两种运行形态

**Native（K3 板卡）**：riscv64 wheel 装到板上，libspert 在进程内，kernel 直接在
板上 JIT 编译 + 执行。这是生产形态。

**QEMU RPC（x86 开发机）**：kernel 在 x86 上交叉编译成 riscv64 `.so`，通过
TCP（默认端口 9999）发到 QEMU 里的 rpc server 执行，结果回读。开发迭代快
（无需上板），但注意 RPC 的三大限制：上传 2GB 上限、client 非线程安全、
QEMU guest 内存上限。

Python 侧通过 `SPINE_TRITON_RPC_HOST` 环境变量切换（设了 = RPC 模式，
host 进程不加载 libspert）。

## 编译产物与缓存

每个 kernel 编译后落在 `TRITON_CACHE_DIR`（默认 `~/.triton/cache`）：
`*.ttir`、`*.linalgdir`（喂给 spine-opt 的 linalg MLIR）、`*.ll`、`*.so`。
这个目录是排错的第一现场——`linalgdir` 可以直接拿 `spine-opt` 离线复现，
详见 basics/04。

下一篇：[02-installation.md](02-installation.md)
