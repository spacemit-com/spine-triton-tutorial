# 01 — spine-triton 是什么（从零讲起）

本章不涉及任何操作，目标是把“我们在做什么、为什么这么做”讲清楚。
读完你应该能回答：kernel 是什么？Triton 解决什么问题？spine-triton 和
Triton 是什么关系？一段 Python 是怎么变成 K3 芯片上跑的机器码的？

## 1.1 从 PyTorch 的一个性能问题说起

先看一段再普通不过的 PyTorch 代码：

```python
import torch

x = torch.randn(4096, 4096, dtype=torch.float16)
y = torch.nn.functional.layer_norm(x, (4096,))   # 内部: 减均值、除标准差
z = y * 2 + 1                                     # 再来一步仿射变换
```

这段代码在底层实际发生了什么？

1. `layer_norm` 把 x（32MB）**从内存读进来**，算完把结果**写回内存**；
2. `* 2` 把上一步的结果（32MB）**再读进来**，乘 2，**写回内存**；
3. `+ 1` 又**读一遍、写一遍**。

一共 6 趟 32MB 的内存读写，而真正的“计算”（乘法加法）少得可怜。现代处理器
算得快、内存搬得慢——这类算子的耗时几乎全部花在**搬数据**上，术语叫
**memory-bound（访存受限）**。

显而易见的优化：把三步**合并成一个 kernel**，数据读进来一次，在寄存器里
算完 layernorm、乘 2、加 1，写出去一次。内存流量从 6 趟变 2 趟，速度接近 3 倍。
这种合并叫**算子融合（fusion）**。

问题是：PyTorch 内置算子是预编译好的，你没法让它们融合。要融合就得**自己写
kernel**—— traditionally 用 C++/汇编，门槛极高。这就是 Triton 出场的理由。

## 1.2 Triton：用 Python 写 kernel

[Triton](https://github.com/triton-lang/triton) 是 OpenAI 开源的 kernel 编程语言
+ 编译器。它的核心思想：**你仍然写 Python，但写的是“一个数据块怎么算”，
编译器负责把它变成高效机器码**。

一个最小的 Triton kernel 长这样（下一章会逐行讲）：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                     # 我是第几号工作块
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # 我负责哪些下标
    mask = offs < n                            # 越界保护
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

和 PyTorch 写法对比，思维方式有一个关键转变：

| PyTorch | Triton |
|---|---|
| 对整个 tensor 写运算：`out = x + y` | 对**一个数据块（block/tile）**写运算，然后声明“这样的块一共有多少个并行跑” |
| 框架决定怎么并行 | 你决定怎么切块（`BLOCK`），框架负责调度 |

这种“每块代码相同、各块处理不同数据”的模式叫 **SPMD**（Single Program
Multiple Data）。“一共有多少个块”由 **grid** 指定，每个块叫一个
**program**，块编号用 `tl.program_id` 取。

原版 Triton 面向 NVIDIA GPU。而我们的目标是 **RISC-V CPU（SpacemiT K3）**——
这就轮到 spine-triton 了。

## 1.3 spine-triton：把 Triton 带到 K3 上

**spine-triton** 是 SpacemiT 维护的 Triton 后端，fork 自微软的
[triton-shared](https://github.com/microsoft/triton-shared)（一个把 Triton 降到
MLIR 的共享中间层）。它做三件事：

1. **vendored Triton 前端**：仓库里带一份打过补丁的 Triton，`@triton.jit`、
   `tl.*` 用起来和 GPU 版一致；
2. **中层转换**：把 Triton 的 tensor 语义 IR（TTIR）经过指针分析、结构化
   分解，转成 MLIR 的 linalg/memref 方言；
3. **后端对接**：把 linalg IR 继续降到 RISC-V 向量指令（RVV）与矩阵引擎
   指令，生成目标码并链接成 `.so` 共享库，由随 wheel 分发的运行时加载
   执行。这些工具与库全部打包在 wheel 里，装完即用（basics/02）。

对用户来说：**写法不变，`import` 换个 driver，tensor 不放 GPU 放 CPU 内存，
底下自动编译成 RISC-V 代码**。

## 1.4 一段 Python 到机器码的完整旅程

这张图值得看两遍——后面所有排错（basics/04）都是在问“旅程在哪一站出的事”。

```
你的 Python 函数 (@triton.jit)
   │ ① Triton 前端 “tracing”：执行一遍你的函数（用符号值），
   │    记录每个操作 → 生成 TTIR（Triton 自己的 MLIR 方言）
   ▼
TTIR  (.ttir 文件)                    ← tt.load / tt.store / tt.dot / tt.splat ...
   │ ② spine-triton-opt：指针分析（哪些 load/store 是规则的块访问）、
   │    结构化分解、TritonToLinalg 转换
   ▼
linalg + memref MLIR  (.linalgdir 文件)  ← linalg.generic / affine / vector ...
   │ ③ 后端降级（wheel 自带的编译器后端工具）：
   │    bufferize（tensor→内存缓冲）、ConvertToScalableVector（→RVV）、
   │    矩阵引擎 lowering（tl.dot → vfwmadot/vfwmacc 指令）
   ▼
LLVM IR  (.ll 文件)                    ← llvm.riscv.vle / vse / vfwmacc ...
   │ ④ llc -mtriple=riscv64 + 链接
   ▼
kernel.so （RISC-V 共享库）
   │ ⑤ 运行时（wheel 自带的执行引擎库）加载执行：
   │    grid 里的每个 program 变成一个 “tile”，调度到 worker 协程上
   ▼
结果写回你的 torch.Tensor（就是普通 CPU 内存）
```

名词解释（第一次见不用背，混个脸熟，basics/04 会实操）：

- **MLIR**：一个“多层 IR 框架”。可以理解为翻译工作中的**多级中间语言**——
  英语→中文很少直译，往往英语→法语→中文。编译器也一样：Triton 语义先降到
  更通用的 linalg（线性代数）方言，再降到向量方言，再到 LLVM IR，每层只做
  一类优化。**方言（dialect）**就是“某一层的词汇表”，比如 `tt` 方言的词汇是
  `tt.load/tt.dot`，`linalg` 方言的词汇是 `linalg.generic/linalg.matmul`。
- **IR**：Intermediate Representation，中间表示——程序的“结构化文本形态”，
  上面每层箭头之间的产物都是 IR 文件，全部可以 dump 出来看。
- **RVV**：RISC-V Vector Extension，RISC-V 的向量指令集（类比 x86 的 AVX、
  ARM 的 NEON）。
- **运行时（runtime）**：随 wheel 分发的执行引擎库，负责加载 `kernel.so`、
  把 grid 里的 program 调度到 worker 核上、管理 TCM。你不需要单独安装或
  配置它。

## 1.5 K3 芯片 60 秒速览

写 kernel 之前只需要知道 K3 的四件事：

1. **它是 RISC-V CPU，有多核 worker**。grid 里的 program 由运行时调度到
   worker 核上并行执行（协程模型）。
2. **RVV scalable vector，vlen=1024 bit**。一条向量指令一次处理：
   f32 → 32 个元素，f16 → 64 个元素。“scalable”指向量长度是硬件参数化的，
   编译器用 `vscale` 表达（你写 `BLOCK=1024`，编译器拆成若干个向量寄存器宽）。
3. **矩阵引擎**：专门算“小矩阵乘累加”的硬件单元，指令是 `vfwmacc` /
   `vfwmadot` 家族，**f16 相乘、f32 累加**（这就是“f16 输入 f32 累加”军规
   的硬件依据）。GEMM/attention 的性能形态全靠它（basics/06）。
4. **TCM**：每 worker 256KB 的紧耦合存储（类比 GPU 的 shared memory），
   编译器会把大 tile 的中间量放这里；**超过 256KB 会分配失败并段错误**——
   这是后面很多“tile 不能开太大”约束的来源。

arch id：K3 = `0xA064`。在板卡上运行时这个值会被自动探测，不需要你设置。

## 1.6 运行环境：K3 板卡直接跑

本教程的默认形态只有一种：**K3 板卡上 pip 安装官方 wheel，在板上 JIT
编译 + 板上执行**（安装见 basics/02）。Python 侧除了程序开头两行激活
后端 driver——

```python
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
```

——不需要任何环境变量、不需要任何额外进程：芯片型号自动探测，运行时在
进程内加载，结果写回的 tensor 就是普通 CPU 内存。

没有板卡、或者想改编译器源码做实验时，basics/02 的进阶节讲了怎么在
x86 开发机上**交叉编译出 riscv64 wheel**——产物同样安装到板卡上使用，
用法与官方 wheel 完全一致。

## 1.7 三层写 kernel 的方式

spine-triton 提供三层接口，能力递增、通用性递减。**新手只用第一层**：

1. **标准 `tl.*`**（本教程 Part 2 的主线）：和 GPU Triton 写法一致，编译器
   自动向量化、自动映射矩阵引擎。90% 的算子应该写在这层。
2. **`smt` 扩展**（basics/06）：显式控制矩阵引擎和 TCM——`smt.alloc`、
   `smt.view`、`smt.dot`、`smt.parallel`。当你要压榨 GEMM 的最后几倍性能、
   或标准路径映射失败时才用。
3. **`spine_raw` eDSL**（basics/05）：Python AST 直接翻译成向量级 MLIR
   （`tle.vload/vstore/vmacc/vexp/...`），甚至直发 LLVM/RVV intrinsic。
   标准路径表达不了的融合、超越函数、特殊内存布局用它。

三层的产物走同一条 ③④⑤ 流水线，排错方法一致。

## 1.8 相关仓库地图

| 仓库 | 一句话角色 | 你会直接用到吗 |
|---|---|---|
| `spine-triton` | 本教程主角：Triton 前端补丁 + 中层转换 + backend | 是（examples/tests 都在这） |
| `FlagGems` / `FlagTree` | 上层算子库：`_spacemit/ops/` 有生产级 K3 优化算子 | 参考（教程会引用其实现） |

编译器后端工具与运行时库已经**打包在 spine-triton 的 wheel 里**（basics/02），
不需要你单独获取。wheel 里的部件来自同一次构建、版本互相咬合——不要单独
替换其中任何一个，混搭二进制是排错地狱的头号来源；初学阶段永远用同一
wheel 里带的整套。

## 1.9 术语表（可随时回来查）

| 术语 | 含义 |
|---|---|
| kernel | 在加速器/目标设备上执行的函数，处理一个数据块 |
| host | 发起 kernel 调用的一侧（你的 Python 主程序） |
| grid | 一次 launch 的 program 总数（几维元组） |
| program | grid 中的一个执行实例，用 `program_id` 区分 |
| block / tile | program 一次处理的数据块，大小由 `BLOCK*` constexpr 决定 |
| mask | 布尔向量，load/store 时屏蔽越界元素 |
| constexpr | 编译期常量参数，改了会触发重新编译 |
| TTIR / linalg / LLVM IR | 编译流水线三层 IR（见 1.4 图） |
| dialect（方言） | MLIR 里某一层的 op 词汇表 |
| RVV | RISC-V 向量指令集 |
| scalable vector | 长度=vscale×固定倍数 的向量类型（硬件参数化） |
| vfwmacc / vfwmadot | K3 矩阵引擎指令（f16 乘 f32 累加） |
| TCM | 每 worker 256KB 紧耦合存储 |
| MICRO tile | 矩阵引擎一次原子计算的块（K3 f16: 16/8/32） |
| 运行时（runtime） | wheel 自带的执行引擎库：加载 kernel.so、把 program 调度成 tile |
| pass | 编译器里一步 IR→IR 的变换 |

下一篇：[02-installation.md](02-installation.md) —— 把环境装起来。
