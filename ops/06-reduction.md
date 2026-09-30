# Op 06 — Reduction（sum / mean / max / argmax）

**重要性：★★★★☆**。归约 = 把一串数变成一个数（或每行变成一个数）。
它是前面三章的公共底座——layernorm 的 mean/var、softmax 的 max/sum、
GEMV 的点积，全是“逐元素 + 归约”的组合。学透这一章，你就掌握了读任何
loss/norm/stat 类 kernel 的钥匙；同时 K3 的 scalable vector lowering 在
**归约长度与寄存器宽度的关系**上有一系列边界行为，是排错知识的高发地。

## 6.1 这个算子在算什么：单位元框架

```
sum:   out = x[0] + x[1] + ... + x[N-1]
max:   out = 最大的 x[i]
mean:  out = sum / N
argmax: out = 最大值的下标 i
```

手算例子——`x = [3, 1, 4, 1, 5]`：`sum=14`、`max=5`、`mean=2.8`、
`argmax=4`。

所有归约都遵循同一个模板：**从单位元出发，逐个吸收元素**。单位元表
（回忆 basics/03 §3.3，这决定 mask 的 `other=` 填什么）：

| 归约 | 单位元 | mask 越界 lane 填 |
|---|---|---|
| sum | 0 | 0 |
| max | -inf | -inf（或归约后 where 剔除） |
| min | +inf | +inf |
| mul | 1 | 1 |
| exp 后 sum（softmax 分母） | 0 | 填 -inf 使 exp 后为 0 |

**填错单位元是归约 kernel 的第一大逻辑 bug**——症状通常不是崩溃而是
“部分输入下结果偏一点”（全负行的 max 填 0、含越界 lane 的 sum 多算
几个 0 之后的 exp(0)=1，等等）。

性能性格：归约把 N 个元素读一遍只产出 1 个数——**极端 memory-bound**。
优化目标只有一个：数据只从内存过一趟（单 pass），中间留在寄存器。

## 6.2 两种并行形态

**形态 A：行归约（每 program 一行）**。输出还有 M 个数（每行一个），
行间独立 → `grid=(M,)`。layernorm/softmax/GEMV 都是这个形态的变体。

**形态 B：全归约（N 个元素 → 1 个数）**。一个 program 串行扫全部 N 太慢，
标准解法是**两段式**：

```
段1: grid=(cdiv(N,BLOCK),)  每 program 归约自己的 BLOCK 段 → partial[pid]
段2: 对 partial 再归约（小数组：单 program kernel / torch.sum / atomic）
```

两段式的好处：段1 完全并行；段2 的输入只有 cdiv(N,BLOCK) 个数，怎么算都
快。代价是多一次 kernel launch（K3 上 dispatch 开销不可忽略——partial 数组
不大时直接 `torch.sum(partial)` 往往就是最优解）。

## 6.3 完整可运行程序（两种形态各一个）

```python
# reduction.py — 行归约 + 两段式全归约
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


# ── 形态 A：每 program 一行，输出 (M,) ──────────────────────
@triton.jit(do_not_specialize=["M", "N"])
def row_sum_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * N                        # 行首（X 连续，行 stride = N）

    acc = tl.zeros([BLOCK], dtype=tl.float32)   # 向量累加器（lane 各自累）
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        acc += x
    tl.store(Y + row, tl.sum(acc, axis=0))      # 最后一次水平归约 → 标量


# ── 形态 B 段1：每 program 一个 BLOCK 段，输出 partial ──────
@triton.jit(do_not_specialize=["N"])
def partial_sum_kernel(X, P, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.).to(tl.float32)
    tl.store(P + pid, tl.sum(x, axis=0))


def row_sum(x):
    M, N = x.shape
    y = torch.empty(M, dtype=torch.float32)
    BLOCK = min(1024, triton.next_power_of_2(N))
    row_sum_kernel[(M,)](x, y, N, BLOCK=BLOCK)
    return y


def full_sum(x):
    N = x.numel()
    BLOCK = 1024
    n_parts = triton.cdiv(N, BLOCK)
    partial = torch.empty(n_parts, dtype=torch.float32)
    partial_sum_kernel[(n_parts,)](x, partial, N, BLOCK=BLOCK)
    return partial.sum()                # 段2：小数组，torch 直接收


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 1000, dtype=torch.float16)      # 非 2 幂行长
    torch.testing.assert_close(row_sum(x), x.float().sum(dim=1),
                               atol=1e-2, rtol=1e-2)
    print("PASS row_sum 128x1000")

    big = torch.randn(1_000_000, dtype=torch.float16)     # grid=977 段
    got = full_sum(big)
    ref = big.float().sum()
    assert torch.allclose(got, ref, atol=1e-1, rtol=1e-3), f"{got} vs {ref}"
    print("PASS full_sum 1M  (triton", got.item(), "vs torch", ref.item(), ")")
```

```bash
python reduction.py
# PASS row_sum 128x1000
# PASS full_sum 1M  (triton ... vs torch ...)
```

### 逐段精讲

- **向量累加器 + 单次水平归约**（形态 A）：`acc` 是 BLOCK 个 lane，循环里
  `acc += x` 是纯向量加，`tl.sum(acc, axis=0)` 只在最后做一次。若每块先
  `tl.sum` 再累加标量，归约次数多 N/BLOCK 倍——**“lane 内累加、最后
  归约”是块内归约的标准形态**（layernorm 趟1 同款）。
- **`.to(tl.float32)` 军规再登场**：`tl.sum` 不提升 dtype，f16 直接归约
  100 万个元素误差会大到肉眼可见。上面 1M 求和的容差特意放到
  atol=1e-1——即便 f32 累加，百万级顺序求和的舍入也在 1e-2~1e-1 量级
  （容差要按**归约长度**缩放，不是按输出元素数，见 6.6）。
- **grid=977**：旧版 wheel 会把超大 grid 静默丢弃，partial 里出现未初始
  化垃圾（basics/04 §4.5）。新版运行时自动分块，无碍；如果你的 wheel
  较旧，把 BLOCK 调到 2048（grid=489）即可绕过。
- **段2 用 torch 不丢人**：partial 只有 977 个数，torch.sum 微秒级。
  写单 program 收尾 kernel 或 atomic 都只在“launch 开销被证明是瓶颈”后
  才值得。
- `do_not_specialize`：同 ops/05 §5.2——避免 M/N 特定值触发特化路径。

## 6.4 spine_raw 版本

（来源：`python/tests/raw/test_raw_sum.py`——注释原话：“sum 是唯一不需要
标量算术的 reduce 算子”，是 raw 归约能力的最小验证。）

```python
@tle.raw_kernel
def sum_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):          # 主循环：满 tile
        vx = tle.vload(X, i)
        acc = acc + tle.cast(vx, f32)
    for i in tle.range(Nfloor, N, nvl):          # 尾循环：0 或 1 次
        tle.vconfig(N - i, 1)
        tx = tle.vload(X, i)
        acc = acc + tle.cast(tx, f32)
    tle.sstore(out, 0, tle.vreduce_sum(acc))     # 水平归约 → 标量 store
```

行归约版（`sum_2d_dim1_kernel`）就是 basics/05 §5.5 的"host pid 传参"
并行形态：`row_idx` 是 host tl kernel 的 `tl.program_id(0)`，load 偏移
变成 `row_idx * N + i`，store 到 `out[row_idx]`，launch `[(M,)]` 每行
一个 program；host 里还有 `if row_idx < M:` 守卫（tl 层的标量 if 是
允许的，default path 的"无 if"限制在 raw kernel 内部）。

对照 tl 版：结构完全同型（主/尾循环 = 分块循环、`vreduce_sum` =
`tl.sum(axis=0)`、`sstore` 标量 = store 单值），只是 raw 层把 strip-mine
显式写出来了。**tl 版里编译器替你做的“BLOCK 切循环 + 尾块 mask”，raw 版
要自己写**——这正是 raw 层的透明度代价与自由。

归约家族其余成员：`tle.vreduce_max/min/mul` 同构；带标量算术的
mean/var/rms 链见 ops/03 §3.5；argmax 见 `test_raw_argmax.py`。

## 6.5 K3 关键点：归约长度 vs 向量寄存器宽度

K3 的向量是 scalable 的（f32/i32 一个寄存器 32 元素、f16 64 元素，
vlen=1024）。**归约维长度（或 BLOCK）与寄存器宽度的三种边界关系，历史上
都出过编译器级 bug**——新版大多已修复，但症状要认识，否则会把工具链问题
当成自己 kernel 写错：

| 边界 | 症状（旧版 wheel） | 状态 |
|---|---|---|
| BLOCK < 寄存器宽（如 BLOCK=16 的 int32） | 边界误判把越界垃圾 lane 折叠进归约：`tl.sum` 恒差 -16、f32 出 NaN | 新版已修复 |
| 行长不是寄存器宽整倍数（如 cumsum 的某些 2D shape） | 编译器断言崩溃（rc=134、无 Python 栈） | 旧版限制（scan 族，见 ops/09） |
| 行长恰好等于寄存器宽（len 15/16） | 编译器断言崩溃（rc=134） | 新版已修复 |

两个配套纪律：

- **cache 会掩盖这类编译 bug**：崩溃/错值 shape“一直复现”可能只是缓存
  .so 命中。先 `mv` 换 TRITON_CACHE_DIR 再下结论（basics/04 §4.2）。
- **rc=134 且无 Python 栈 = 编译器层**：不要在 kernel 源码里瞎改，按
  basics/04 §4.4 的流程走（新 cache → 最小化 → 升级 wheel → 报告）。

其余通用约束：f16 大归约必须 f32 累加（顺序累加误差实测可达 rel 0.6%
——vector_norm 24F 的教训）；归约展开的中间 buffer 同样吃 TCM 256KB
预算（basics/06 §6.3）。

## 6.6 容差怎么定：按归约长度缩放

f32 顺序累加 N 个 ~O(1) 的数，舍入误差按 ~√N·ε 增长。实操参考：

| 归约长度 | f32 累加的合理 atol | f16 累加（错误示范）|
|---|---|---|
| ≤1K | 1e-4 | 1e-2 边缘 |
| ~100K | 1e-2 | 必然超差 |
| ~1M | 1e-1 | 灾难 |

**测试失败先问“容差按什么定的”**——只按输出元素数、不按归约长度定容差
会把正确的 kernel 判死（历史案例：addmm 1F，atol 只按 N 不按 K，边界
舍入 1.05e-4 vs 容差 1e-4，数据依赖 flaky）。golden 也讲究：用
`x.float().sum()` 而不是 `x.sum()`（后者对 f16 输入是 f16 累加，golden
自己就不准）。

## 6.7 mean / var / argmax 的常用变形

- **mean** = sum/N。raw 层的“reduce → 标量除 → （rsqrt）→ 广播回向量”
  标量算术链是 mean/rms/layernorm/var 家族共同底座（ops/03 §3.5）。
- **var** 两种写法：两趟 `(x-mean)²`（稳）与单趟 `E[x²] − mean²`（省一趟
  访存，但**大均值小方差时有消去误差**——两个大数相减）。生产 kernel
  （var/var_mean）选稳的；raw 教学例选省的。知道 trade-off 再选。
- **argmax**：`tl.argmax(x, axis)` 一步到位；或手动
  `m = tl.max(x); idx = tl.sum(tl.where(x == m, offs, 0), axis)`（利用
  “只有最大值的 lane 贡献下标”）。注意索引 dtype：i64 下标与 i32 值混算
  前先统一类型。带 tie-break 语义（torch.argmax 返回首个最大下标）时用
  tl.argmax 更省心。
- **scan 类**（cumsum/cummax，前缀归约，输出仍是 N 个数）是另一族，
  见 ops/09。

## 6.8 验证

```bash
python reduction.py                                     # 本章程序
python3 -m pytest python/tests/test_reduction_ops.py -q # tl 测试套
python3 -m pytest python/tests/raw/test_raw_sum.py -q   # raw 版
python3 -m pytest python/tests/raw/test_raw_argmax.py python/tests/raw/test_raw_max_dim.py -q
```

golden：`torch.sum/mean/max/argmax`（f16 输入一律 `.float()` 后算）。

数值排错的利器——**NaN-guard 切片探针**（basics/04 §4.9）：

```python
buf = torch.full((numel + guard,), float('nan'))
x = buf[:numel].view(shape)          # 保持精确 stride/specialization
```

任何越界读都会把 NaN 带进输出，逐块扫描即可定位 OOB 区间。归约 kernel
的“某 shape 区间数据依赖错值”（如旧版编译器只在特定 K 区间触发的越界读
bug）用这招一次定位。

## 6.9 练习

1. 跑通 6.3。把 row_sum 的 `other=0.` 改成 `other=1.`，N=1000、
   BLOCK=1024 下预测每行会多算多少（24 个越界 lane × 1），验证。
2. 把 `acc` 改成 f16（`tl.zeros([BLOCK], dtype=tl.float16)`），行归约
   N=100000 重跑，记录误差量级——对照 6.6 的表。改回来。
3. 实现 `row_max`（形态 A 改 max）：单位元、`tl.max(acc, axis=0)`，
   越界 lane 想清楚填什么。golden `x.float().max(dim=1).values`。
4. 实现两段式 argmax（段1 每 BLOCK 出局部 (max, idx)，段2 收尾——提示：
   段2 用 torch 做 `partial_max.argmax()` 再还原全局下标）。
5. 思考题：形态 B 里 BLOCK=1024 → grid=977。如果把 BLOCK 改成 64，
   grid=15625，段2 的 torch.sum 输入变 15625 个数——两头开销怎么变？
   在 K3 上实测 partial_sum 两种 BLOCK 的墙钟（记得各自独立 cache）。

下一篇：[07-elementwise-fused.md](07-elementwise-fused.md) —— 逐元素与融合。
