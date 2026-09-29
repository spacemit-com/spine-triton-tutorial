# 03 — 第一个 kernel：向量加法逐行精讲

本章我们把一个完整的向量加法程序写出来、跑起来，然后**逐行拆解**每个概念。
这是全书最重要的一章：grid/program/block/mask 这四个概念贯穿之后所有算子。

（前提：basics/02 的 `verify_install.py` 已经跑通。）

## 3.1 完整程序

保存为 `vec_add.py`：

```python
# vec_add.py — 第一个 spine-triton kernel
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

# spine-triton 是 CPU 后端。PyTorch 默认假设 Triton 面向 CUDA，
# 所以必须显式激活 CPU driver（每个程序入口做一次即可）。
triton.runtime.driver.set_active(CPUDriver())


# ───────────────────────── kernel（在“设备”上执行）────────────────────────
@triton.jit
def add_kernel(
    x_ptr,                       # 输入向量 x 的首地址（指针）
    y_ptr,                       # 输入向量 y 的首地址
    out_ptr,                     # 输出向量的首地址
    n_elements,                  # 向量长度（普通运行期参数）
    BLOCK_SIZE: tl.constexpr,    # 每个 program 处理多少个元素（编译期常量）
):
    # 我是几号 program？（grid 是一维的，所以 axis=0）
    pid = tl.program_id(axis=0)

    # 我负责的元素下标：一个长度为 BLOCK_SIZE 的向量
    #   pid=0 → [0, 1, ..., BLOCK-1]
    #   pid=1 → [BLOCK, ..., 2*BLOCK-1]  ...
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 最后一个 program 可能分到超出 n_elements 的下标 → 用 mask 屏蔽
    mask = offsets < n_elements

    # 按块加载（mask=False 的位置不访问内存）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    # 按块写回（mask=False 的位置不写）
    tl.store(out_ptr + offsets, output, mask=mask)


# ───────────────────────── host（在你的 Python 里执行）─────────────────────
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)              # 1) 预分配输出（kernel 不会帮你分配）
    n_elements = out.numel()
    # 2) grid：需要多少个 program。写成 lambda，Triton 会把 constexpr 元参数传进来
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # 3) launch：kernel[grid](位置参数..., constexpr 用关键字传)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 98432                              # 故意不是 1024 的倍数，验证 mask
    x = torch.rand(n)
    y = torch.rand(n)

    out_triton = add(x, y)
    out_torch = x + y

    torch.testing.assert_close(out_triton, out_torch)
    print(f"PASS: n={n}, 最大误差 = {(out_triton - out_torch).abs().max().item()}")
```

运行：

```bash
python vec_add.py
# 第一次：板上 JIT 编译，十几秒到几十秒；之后缓存命中，秒回
# PASS: n=98432, 最大误差 = 0.0
```

## 3.2 概念拆解：grid / program / block

Triton 的执行模型一句话：**你写“一个块怎么算”，声明“有多少块”，硬件并行跑**。

以 `n=98432, BLOCK_SIZE=1024` 为例：

```
grid = (cdiv(98432, 1024),) = (97,)        # 97 个 program

数据:   [0 .......... 1023][1024 ........ 2047] ... [98304 ... 98431 空 空 空]
              pid=0              pid=1                    pid=96
                                                          └─ mask 在这里生效
```

- **program**：一个执行实例。97 个 program 拿到**同一份代码**，只有
  `tl.program_id(0)` 的返回值不同（0~96），因此各自处理不同的数据段。
- **grid**：program 的组织方式，可以是 1/2/3 维元组。二维 grid 常见于矩阵
  算子（ops/01 里 `grid=(pid_m 维, pid_n 维)`）。
- **block/tile**：一个 program 一次处理的数据块。块内的运算是**向量化**的：
  `offsets`、`x`、`y`、`mask` 都是长度 1024 的“张量值”，`x + y` 是一条块级
  加法——编译器把它拆成若干个 RVV 向量指令（f32 一个寄存器 32 元素，
  1024 元素 ≈ 32 条向量加法的量级，还会被进一步调度优化）。

在 K3 上，每个 program 由运行时变成一个 **tile**，调度到 worker
协程上执行；grid 里的 tile 之间**没有执行顺序保证，也不能互相通信**
（需要跨块归约时，要么两段式 kernel，要么单 program 串行——ops/06 细讲）。

## 3.3 概念拆解：mask 为什么必须有

`tl.load(ptr + offsets)` 会真的去访问 `offsets` 里每个下标对应的内存。
最后一个 program 的 offsets 是 `[99328, ..., 100351]`，而数据只有 98432 个
元素——**多出来的下标指向别人的内存**。轻则读到垃圾，重则越界段错误，
更阴险的是 `tl.store` 越界会**踩坏堆**（症状在很晚的另一个操作里爆发）。

`mask=offsets < n_elements` 让 mask=False 的 lane 不访问内存：

- `tl.load(..., mask=m, other=v)`：False 的位置读成 `v`（不写 `other` 则未定义，
  归约前要填单位元：求和填 0、求 max 填 `-inf`）；
- `tl.store(..., mask=m)`：False 的位置不写。

**规则：任何 `pid * BLOCK + arange` 形式的下标，只要不能整除保证不越界，
load/store 必须带 mask。** 生产 kernel 里漏 mask 的 store 是最常见的一类
堆损坏 bug（真实案例：FlagGems kron 的无 mask store 越界写堆，症状是
“之后某个不相关操作 abort”）。

## 3.4 概念拆解：指针与 tensor 参数

- host 侧传入 `torch.Tensor`，kernel 侧收到的是**它首元素的指针**
  （隐式转换）。kernel 里 `x_ptr + offsets` 是指针向量运算。
- 数据必须在**连续内存**里（`torch.empty_like` 天然连续）。如果传非连续
  视图（如转置、切片），要么先 `.contiguous()`，要么显式传 stride 参数
  用 `ptr + row * stride_r + col * stride_c` 寻址（ops/02 起大量使用）。
- 标量参数（`n_elements`）按值传入。

## 3.5 概念拆解：constexpr 与“值特化”

`BLOCK_SIZE: tl.constexpr` 表示**编译期常量**：

- 它参与形状计算（`tl.arange(0, BLOCK_SIZE)` 的长度必须编译期已知）；
- 换一个 BLOCK_SIZE 值 = 触发一次重新编译（缓存里各存一份）；
- launch 时必须用关键字传（`BLOCK_SIZE=1024`）。

反过来，**普通标量参数默认也会被“值特化”**：Triton 会把值为 1 的 i32 参数
当常量折叠（还有 16 整除性等 hint）。这通常是优化，但有个反复坑人的场景：
host 侧算好 `NK = M // 32` 传进去，当 `M=32` 时 `NK=1` 被特化成常量，
kernel 里依赖 NK 的循环直接从 IR 里消失。要运行期真值时显式声明：

```python
@triton.jit(do_not_specialize=["M", "NK"])
def kernel(..., M, NK, ...): ...
```

**经验法则：凡是 host 算出来传进去的 size/块数/循环次数，一律加
`do_not_specialize`。**

## 3.6 dtype 三条军规（K3 上必须内化）

1. **f16 输入、f32 累加、最后才 cast 回 f16。**
   浮点加法不满足结合律，长归约用 f16 累加误差会滚雪球（实测 f16 部分和
   超容差、甚至 12% 元素失败）。写法：

   ```python
   acc = tl.zeros([BLOCK], dtype=tl.float32)         # f32 累加器
   x = tl.load(...).to(tl.float32)                   # load 后立刻提升
   acc += x
   tl.store(..., acc.to(tl.float16))                 # 只在写回时降精度
   ```

   硬件依据：矩阵引擎 vfwmacc 本身就是 f16 乘 f32 累加。
2. **`tl.exp/log/sqrt/...` 等 math 函数只接受 fp32/fp64。**
   对 fp16 tensor 直接调用会 `ValueError: Expected dtype ['fp32','fp64']
   but got fp16`。这是 Triton 上游的**故意设计**（GPU 同样报错），不是
   spine 后端 bug。写法：`tl.exp(x.to(tl.float32))`。
3. **bf16 目前不可用向量路径。**工具链缺 Zvfbfmin/Zvfbfma 扩展时 bf16 的
   向量 load/store 会让 llc 直接 abort。测试环境 FLOAT_DTYPES 一般是
   `[fp16, fp32]`；看到 bf16 失败先归为平台限制。

## 3.7 动手练习

先跑通 `PASS`，再做：

1. **改造成 `out = alpha*x + y`**：加一个 float 参数 `alpha`。思考：它需要
   `do_not_specialize` 吗？（试传 `alpha=1.0` 和不传时的编译行为。）
2. **BLOCK_SIZE 实验**：分别用 256/1024/4096 跑，确认结果都正确；打开
   `SPINE_TRITON_DUMP_PATH=./dumps_256`（每个 BLOCK 一个目录）看三份 `.ttir`
   的差异——体会“constexpr 变了 = 另一个 kernel”。
3. **故意去掉 mask**：把 `n` 改成 100000（不是 1024 倍数），删掉所有
   `mask=`，观察现象（可能 PASS 也可能错/崩——这正是越界的可怕之处：
   **行为取决于堆布局，不是确定性的**）。看完改回来。
4. **二维 grid 热身**：写一个 kernel 把 `(M, N)` 矩阵每个元素乘以自己的行号，
   `grid=(M, cdiv(N, BLOCK))`，`row = tl.program_id(0)`。这是 ops/02 所有
   行级算子的骨架。

## 3.8 跑仓库自带示例

spine-triton 仓库的 `python/examples/` 有一批同款示例（多数是 pytest 函数
形态，也可以直接 python 跑）：

```bash
cd spine-triton
python3 python/examples/test_vec_add.py
python3 python/examples/test_softmax.py
python3 python/examples/test_layernorm.py
python3 python/examples/mm_block_ptr.py        # GEMM（ops/01 主角）
python3 python/examples/test_smt_mm.py         # smt 矩阵引擎 GEMM（basics/06）
```

板上直接 `python3` 跑即可，不需要任何环境变量。想给单发探针最干净的隔离：

```bash
TRITON_CACHE_DIR=/tmp/probe_cache_$$ python3 my_probe.py
```

## 3.9 报错急救（新手版）

| 报错关键词 | 八成是 | 先做什么 |
|---|---|---|
| `CompilationError` + 指到你源码某行 | 前端 tracing 层（dtype 军规 / constexpr / API 误用） | 读它附的源码定位；对照 3.5/3.6 |
| `cannot open shared object file`（某个 .so 加载失败） | 系统依赖缺了 | 按报错缺的库补 basics/02 §2.1 的 apt 包 |
| rc=134 / `Assertion failed`、没有 Python 栈 | 编译器后端崩了 | 记下完整输出与最小复现，按 basics/04 §4.4/§4.6 走：换新 cache → 升级 wheel → 仍复现再上报；这不是你 kernel 的语法问题 |
| kernel .so 加载失败 / `undefined symbol` | 运行时加载层（多为二进制混装） | 整套重装 wheel（basics/02 §2.2 的纪律），别单独换文件 |
| 结果全零 / 99.9% 垃圾值 | 旧版 wheel 的已知限制（如旧运行时对超大 grid 有上限），或输出没初始化 | 升级 wheel + 换新 cache 重跑（basics/02 §2.9、basics/04 §4.5） |
| 数值差一点点 | 精度问题 | 查 dtype 军规三件事：提升点、mask 的 other 值、归约长度 |

分层判别的完整版在 [basics/04](04-running-and-debugging.md)。

下一篇：[04-running-and-debugging.md](04-running-and-debugging.md)
