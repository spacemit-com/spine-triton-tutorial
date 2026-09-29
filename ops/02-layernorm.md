# Op 02 — LayerNorm

**重要性：★★★★☆**。Transformer 每层至少两次 LayerNorm（attention 前、
MLP 前）。它是典型的 **“归约 + 逐元素”复合算子**——学会它，你就掌握了
一大类算子（RMSNorm、GroupNorm、softmax、各种 norm）的通用骨架：
**每行独立 → 每 program 一行 → 行内多趟扫描**。

## 2.1 这个算子在算什么

对每一行 x（长度 N）独立地做：

```
mean = E[x]                          # 行均值
var  = E[(x - mean)²]                # 行方差
y    = (x - mean) / sqrt(var + eps) * w + b     # 归一化 + 仿射
```

`eps`（如 1e-5）防止除以 0；`w`、`b` 是可学习的逐列缩放/平移参数
（长度都是 N）。

一个能手算的例子——`x = [1, 2, 3, 4]`，`w=[1,1,1,1]`，`b=[0,0,0,0]`：

```
mean = 2.5
var  = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²)/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
y    = (x - 2.5) / sqrt(1.25 + 1e-5) ≈ [-1.342, -0.447, 0.447, 1.342]
```

性质：输出行均值≈0、方差≈1。**关键点：mean/var 依赖整行数据**——所以
一个 program 必须看到完整的一行才能算，这决定了并行方式。

## 2.2 为什么它值得手写 kernel

回忆 basics/01 §1.1 的访存账：PyTorch 的 `mean`、`var`、减、除、乘、加
是六个独立算子，每个都要完整读写一遍内存（6+ 趟）。而 layernorm 的算术
量极少（每元素几次乘加）——**纯 memory-bound**。融合成一个 kernel 后只需
2~3 趟访存，理论加速 2~3 倍。这正是 Triton 官方教程把它列为第 5 课的原因。

## 2.3 并行方案：每 program 一行

```
X: (M 行, N 列)                 grid = (M,)
┌────────────────┐
│ row 0          │ ← program 0：独立算出 mean0/var0，归一化 row 0
│ row 1          │ ← program 1
│ ...            │
│ row M-1        │ ← program M-1
└────────────────┘
```

行间完全独立、无通信——完美匹配 SPMD 模型。行内呢？N 可能远大于一个
向量寄存器（甚至大于 BLOCK_SIZE），所以行内要**分块循环 + 多趟扫描**：

```
趟1: 扫全行累加 x        → mean
趟2: 扫全行累加 (x-mean)² → var → rstd = 1/sqrt(var+eps)
趟3: 扫全行，(x-mean)*rstd*w+b 写出
```

为什么不能一趟搞定？mean 要全行扫完才知道，第二趟的 `(x-mean)` 依赖它。
（当 N ≤ BLOCK_SIZE 且允许时，可以一趟把整行 load 进寄存器复用，省两趟
访存——见 2.7。）

## 2.4 完整可运行程序

（kernel 来源：`python/examples/test_layernorm.py`，即 Triton 官方
05-layer-norm 教程；host 部分重写为独立脚本，golden 可自动对照。）

```python
# layernorm.py — 每 program 一行的 fused LayerNorm forward
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def _layer_norm_fwd_fused(
    X,          # 输入 (M, N)，按行连续
    Y,          # 输出，同形状
    W, B,       # weight/bias，各 (N,)
    Mean, Rstd, # 每行的 mean / 1-std（f32），缓存给 backward 用
    stride,     # 行 stride（元素数）
    N,          # 列数
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)          # 我是第几行
    X += row * stride               # 指针移到本行开头
    Y += row * stride

    # ── 趟1: mean（f32 累加器）─────────────────────────────
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean, axis=0) / N        # 向量 → 标量

    # ── 趟2: var ──────────────────────────────────────────
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)   # 越界 lane 强制贡献 0
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # 缓存 mean/rstd（backward 要用，避免重算）
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # ── 趟3: 归一化 + 仿射 + 写出 ─────────────────────────
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)      # 自动 cast 回 Y 的 dtype


def layer_norm(x, weight, bias, eps=1e-5):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32)
    rstd = torch.empty((M,), dtype=torch.float32)

    # 官方教程的 BLOCK 启发式：单 feature 不超 64KB
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    _layer_norm_fwd_fused[(M,)](
        x_arg, y, weight, bias, mean, rstd,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, mean, rstd


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, eps = 128, 1024, 1e-5
    x = torch.randn(M, N, dtype=torch.float16)
    w = torch.rand(N, dtype=torch.float16)
    b = torch.rand(N, dtype=torch.float16)

    y, mean, rstd = layer_norm(x, w, b, eps)

    # golden：torch CPU 的 layer_norm 不吃 fp16，用 f32 算再降回来
    y_ref = torch.nn.functional.layer_norm(
        x.float(), (N,), w.float(), b.float(), eps).to(torch.float16)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("PASS  y max diff =", (y - y_ref).abs().max().item())
```

```bash
python layernorm.py
# PASS  y max diff = 0.00xx
```

### 逐段精讲

- **`X += row * stride`**：kernel 参数是指针，可以直接做算术。移到行首后，
  后面所有 `X + cols` 都是行内偏移——这比每次写 `X + row*stride + cols`
  干净，也是行级算子的惯用开头。
- **向量累加器 `tl.zeros([BLOCK_SIZE], f32)`**：注意趟1 不是标量累加，而是
  **BLOCK_SIZE 个 lane 各自累加**（`_mean += x` 是向量加），最后
  `tl.sum(..., axis=0)` 一次水平归约。这比“每块先归约再加”少做很多次
  归约，是块内归约的标准形态（ops/06 详述）。
- **`.to(tl.float32)` 的位置**：load 后立刻提升。**`tl.sum` 不会自动提升
  dtype**——fp16 tile 直接 sum 得到的是 fp16 部分和，长行下误差超容差
  （实测 abs 误差可到 0.03）。这是 K3 端口层修过的真 bug（2.6 坑 1）。
- **`tl.where(cols < N, x - mean, 0.)`**：mask 已经让越界 lane 读成 0，但
  `(0 - mean)²` 不是 0——会虚增方差！所以第二趟要再显式把越界 lane 的贡献
  归零。**凡是“mask 填的值参与非线性运算”的场合都要多想一步**（求和填 0
  没事，平方/exp 就有事）。
- **Mean/Rstd 缓存**：forward 多花 2M 个 f32 的写入，backward 就不用重扫
  两趟数据算 mean/var。这是训练场景的标准交易（纯推理可以省掉这两个
  参数）。
- **`tl.store(Y + cols, y, mask=mask)`**：普通指针 store 会把 f32 的 `y`
  **隐式 cast** 到 Y 的元素类型（f16）——与 block_ptr 的严格检查相反
  （2.6 坑 3）。

## 2.5 backward 的形状（读懂即可）

backward 要算三个量：dx（同形状）、dw/db（各 (N,)，**跨所有行归约**）。

- dx 用缓存的 mean/rstd 单趟完成：`dx = rstd*(dy*w) - mean项 - rstd²项`
  （三块组合，都是行内运算，仍每 program 一行）；
- dw = Σ_rows(dy * x_hat)、db = Σ_rows(dy) 是**列方向的跨行归约**——
  并行方向反过来了：grid 按**列块**分，每 program 沿行循环累加。

一个算子里出现两种正交的归约方向，这是 norm 类 backward 的通用结构
（GroupNorm/InstanceNorm 同理，只是多一个分组维度）。

> 已知编译器问题：conv/norm 家族 backward 的“内层循环 iter_arg = 外层
> iter_arg”跨层携带累加器结构，在 bufferization（tensor→内存缓冲）阶段
> 会触发读写冲突报错。碰到 `'scf.for' op not bufferizable: cannot avoid
> RaW conflict` **不是你的 kernel 写错了**，是编译器对该结构的已知限制
> （已定位）；能绕则把跨层累加改写为单层循环内完成，或关注新版 wheel。

## 2.6 K3 端口层四连坑（全部实战修过）

FlagGems 的 `_spacemit/ops/layernorm.py` 端口曾 24F/16P，修复后 40/40。
四个坑全部是**编译期 bug**（15/24 个失败），不是精度问题——教训：表里
“fp16 数值精度类”的摘要会掩盖真实构成，要逐例拆日志。

1. **`tl.sum` 不提升 dtype**：见 2.4 精讲。load 后必须 `.to(tl.float32)`。
2. **`make_block_ptr` 的 base 不能是 None**：惯例写法 `if W is None: w=1`
   的内联守卫只护住 load 那一行；如果你在**循环外**预建 block desc，
   `tl.make_block_ptr(base=W)` 在 `W is None` 时直接前端崩
   （`'constexpr_type' object has no attribute 'is_ptr'`）。可选 weight 的
   kernel 要么两条路径分开建 desc，要么 host 侧保证非 None（比如传全 1
   tensor）。
3. **block-ptr store 无隐式 cast**：`tl.store(block_ptr, f32_val)` 对 f16
   block 报 “Block element type(fp16) and value element type(fp32)
   mismatch”——必须显式 `.to(Y.type.element_ty)`。（普通指针 store 有隐式
   cast，两种 store 行为不一致，务必分清。）
4. **constexpr 整数没有 `.to()`**：`(2).to(tl.float32)` 前端报
   `'int' object has no attribute 'to'`——Python int 不是 tensor。写
   `a * a` 代替 `a.pow(2)` 之类的场合，或用 `tl.full([], 2.0, tl.float32)`。

## 2.7 spine_raw 版本（进阶）

无 weight/bias 的 1D layernorm raw 三趟版全文见
[basics/05 §5.5](../basics/05-spine-raw-edsl.md)（来源
`python/tests/raw/test_raw_layernorm.py`）。raw 版的三个独有价值：

- 方差用 `E[x²] − mean²` 单趟形式，padding 0 的贡献天然是 0（不需要
  `tl.where` 归零那步）；
- f16 load → f32 向量/标量混合算术 → f32 输出，**没有中间 cast 截断点**；
- 想做 fusion（layernorm+residual、layernorm+后续逐元素）时不受 tl 层
  API 限制。

带 weight/bias、多行组织的 raw 版本参考 `test_raw_group_norm.py` /
`test_raw_instance_norm.py` / `test_raw_batch_norm.py`（默认 path 无
program_id，grid=(1,) + 外层串行循环遍历行）。

单 tile 优化：当 `N ≤ BLOCK_SIZE` 时三趟循环各只执行一次——数据整行留在
寄存器里，访存从 5 趟（x 三读 + w/b 一读 + y 一写按块计）降到理论最小。
host 侧启发式 `BLOCK_SIZE = next_power_of_2(N)`（封顶 64KB/元素大小）就是
为了让常见行宽命中这个形态。

## 2.8 验证

```bash
python layernorm.py                                        # 本章程序
python3 python/examples/test_layernorm.py                  # 仓库示例（M=1151, N=8192, f16）
python3 -m pytest python/tests/test_norm_ops.py -q         # norm 测试套
python3 -m pytest python/tests/raw/test_raw_layernorm.py -q  # raw 版
```

容差惯例：golden 用 f32 计算再 cast（torch CPU layer_norm 不吃 fp16——
示例文件里对照被注释掉就是这个原因）；fp16 结果 atol/rtol 1e-2 量级、
fp32 用 1e-4~1e-5。

引用历史测试数字时注意批次的 wheel 版本：旧批次的 fp32/backward“数值
失败”可能是旧版运行时丢弃超大 grid 的假象（basics/04 §4.7）。

## 2.9 练习

1. 跑通 2.4，然后故意删掉所有 `.to(tl.float32)`，把 N 改成 8192 重跑——
   观察 fp16 累加误差如何随行长滚大（对照 2.6 坑 1）。改回来。
2. 删掉趟2 的 `tl.where(cols < N, ...)`，取 N=1000、BLOCK_SIZE=1024
   （越界 lane 有 24 个）重跑。预测方差会偏大还是偏小？验证。
3. 实现 `ONE_TILE` 快速路径：host 侧当 `N <= 1024` 时走“单趟 load 全行进
   寄存器”的 kernel 变体（提示：三个循环体合并，x 只 load 一次存局部变量）。
   数值必须与三趟版一致。
4. 把 `w`/`b` 改成可选（None 时跳过仿射）。分别试“host 传全 1/全 0
   tensor”和“kernel 内 `if W_IS_NONE: tl.constexpr` 分支”两种方案，想想
   哪种不会踩 2.6 坑 2。

下一篇：[03-rmsnorm.md](03-rmsnorm.md) —— LayerNorm 的减法。
