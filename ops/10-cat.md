# Op 10 — Cat / Concat（拼接）

**重要性：★★★☆☆**。`C = cat([A, B], dim=0)`——把几个张量沿某一维
首尾相接。它是全书最简单也最"诚实"的算子：**没有任何计算，只有搬运**
——性能上限就是内存带宽，kernel 的全部难点在"输出位置 ← 输入位置"的
映射与控制流组织。收官章选它有两个理由：一是搬运类 kernel（cat/copy/
stack/concatenate/roll/pad…）在实际工程里出现频率极高；二是它的"多输入
选一"控制流恰好踩过一个非常有代表性的编译器坑，适合作为全书分层排查
方法论（basics/04 §4.5）的最后一道练习题。

## 10.1 这个算子在算什么：手算一遍

```
A = [[1,2],[3,4]]        (2×2)
B = [[5,6]]              (1×2)
C = cat([A,B], dim=0) = [[1,2],
                         [3,4],
                         [5,6]]           (3×2)
```

dim=0 拼接的行映射规则一句话：**输出第 r 行 = A 的第 r 行（r < M_A）
或 B 的第 r-M_A 行（r ≥ M_A）**。除拼接维外所有维度必须相等；输出在
拼接维的长度 = 各输入之和。

cat 的本质是 N 段 `memcpy`（每段带 stride）。它的两个变体在生产实现里
殊途同归：

- **stack** = 先 unsqueeze 出新维再 cat（`stack([A,B])` =
  `cat([A[None], B[None]])`）；
- **hstack/vstack** = 特定维度约定的 cat（1D hstack 是 dim=0，2D 是
  dim=1；vstack 是 dim=0）。

所以吃透 dim=0 的二维 cat，整个家族都通了。

## 10.2 并行方案：输出位置驱动

搬运类 kernel 的标准并行组织是**按输出切块**（不是按输入）：输出的
每个位置恰好写一次，天然无冲突。二维 cat（dim=0）的切法：

```
grid = (M_A + M_B,  cdiv(N, BLOCK_N))     # 每 program 一个输出行×列块
```

行号 r 决定数据来自 A 还是 B——这是全 kernel 唯一的"分支"，怎么处理它
有两种写法，10.3 是无分支版（可直接跑），10.5 是分支版（生产常见，
也是那个坑的现场）。

生产实现的另一种组织（FlagGems）：**根本不在 kernel 里选输入**——
host 侧循环每个输入张量，算好它在输出里的偏移（`itertools.accumulate`
累加 `size[dim] * stride[dim]`），每段 launch 一次纯 copy kernel
（spacemit 版是 `make_block_ptr + boundary_check` 的 grid-stride 平铺
copy，固定 8 个 program；非连续切片则 fallback `copy_`）。分支消失了，
代价是 K 个输入 K 次 launch——输入个数少时这是最稳的写法。

## 10.3 完整可运行程序：无分支 mask-select

```python
# cat.py — C = cat([A, B], dim=0)，每 program 一个输出行×列块
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M_A", "M_B", "N"])
def cat0_kernel(A, B, C, M_A, M_B, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)                 # 输出行号，0 .. M_A+M_B-1
    col = tl.program_id(1)                 # 列块号
    offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # 两路都 load，用 mask 选路：
    #   row < M_A  → 只有 A 路有效（B 路 mask 全 False，读成 0）
    #   row >= M_A → 只有 B 路有效（A 路 mask 全 False）
    va = tl.load(A + row * N + offs_n,
                 mask=n_mask & (row < M_A), other=0.)
    vb = tl.load(B + (row - M_A) * N + offs_n,
                 mask=n_mask & (row >= M_A), other=0.)

    # 任意一路被 mask 掉时该路为 0，相加即"选择"
    tl.store(C + row * N + offs_n, va + vb, mask=n_mask)


def cat0(a, b):
    M_A, N = a.shape
    M_B, _ = b.shape
    c = torch.empty((M_A + M_B, N), dtype=a.dtype, device=a.device)
    BLOCK_N = min(256, triton.next_power_of_2(N))
    cat0_kernel[(M_A + M_B, triton.cdiv(N, BLOCK_N))](
        a, b, c, M_A, M_B, N, BLOCK_N=BLOCK_N)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    for M_A, M_B, N in [(2, 1, 2), (100, 37, 256), (64, 64, 1000)]:
        a = torch.randn(M_A, N, dtype=torch.float16)
        b = torch.randn(M_B, N, dtype=torch.float16)
        got = cat0(a, b)
        ref = torch.cat([a, b], dim=0)
        assert torch.equal(got, ref), f"mismatch at {(M_A, M_B, N)}"
        print(f"PASS cat0 {M_A}x{N} + {M_B}x{N}  (exact)")
```

```bash
python cat.py
# PASS cat0 2x2 + 1x2  (exact)
# PASS cat0 100x256 + 37x256  (exact)
# PASS cat0 64x1000 + 64x1000  (exact)
```

纯搬运没有算术，结果必须**逐位相等**（`torch.equal`）——这是搬运类
kernel 测试的默认判据，任何非零 diff 都是 bug（对照归约/attention 的
容差测试）。10.1 的 2×2+1×2 手算例子就在第一组用例里。

### 逐段精讲

- **mask 掉的 lane 绝不访存**——这是本写法的灵魂。`row < M_A` 时 B 路
  的 mask 全 False，`(row - M_A)` 这个**负偏移根本不会被解引用**
  （Triton 的 mask load 语义：被 mask 的 lane 直接取 `other`，不发
  访存）。初学者常以为要先 `if row >= M_A` 保护偏移计算——不需要，
  mask 就是保护。
- **`va + vb` 为什么等于"选择"**：任何时刻恰有一路是真实值、另一路是
  `other=0.`，相加即选择。搬运类 kernel 的标准无分支技巧（对
  int/float 都成立；若是 bool 要换 `|`）。
- **`grid=(M_A+M_B, cdiv(N,BLOCK_N))`**：行数 128+64=192、列块若干，
  grid 总量安全（旧版 wheel 的 grid 上限见 basics/04 §4.5；新版自动
  分块）。行多列窄时也可以退化成 `grid=(M_A+M_B,)` 每行内循环列块。
- **`do_not_specialize=["M_A","M_B","N"]`**：老规矩（ops/05 §5.2），
  M_B=1 之类的值不该被折成 constexpr。
- **多维推广**：dim=0 的 3D/4D cat 只要把行号解码成多维下标
  （`r → (b, m)` 等）、偏移按 stride 算即可——映射规则不变。非拼接维
  不连续时，host 侧先 `contiguous()` 或走 stride 版偏移（生产实现
  传全套 stride 参数，10.2）。

## 10.4 生产实现的取舍

对照 10.2 说的 FlagGems 两种实现，把"要不要在 kernel 里分支"的账
算清楚：

| 方案 | launch 次数 | kernel 内分支 | 适用 |
|---|---|---|---|
| host 循环 + 每输入一次 copy kernel | K（输入个数） | 无 | 通用、最稳；K 小 |
| 单 kernel + mask-select（本章） | 1 | 无 | 2~3 个输入、形状规整 |
| 单 kernel + 指针 if/elif 选路 | 1 | 有 | K 大且想省 launch（见 10.5 的坑） |

K3 上 dispatch 开销不可忽略（ops/05 §5.5 实测过"grid 小反而快"），
所以 K 大时第三种方案有真实吸引力——但它踩出了一个非常典型的编译器
坑，值得作为本章的进阶案例。

## 10.5 进阶案例：指针 if/elif 与 `failed to legalize 'cf.br'`

历史版本的生产 cat kernel 是这样组织的（每 program 按行号从 K 个输入
指针里选一个）：

```python
if pid_y < size0:
    src = ptr0 + (pid_y - 0) * N
elif pid_y < size0 + size1:
    src = ptr1 + (pid_y - size0) * N
else:
    src = ptr2 + ...
x = tl.load(src + offs)          # if/elif 的"结果"是一个指针
```

问题：**if/elif 的结果是张量指针**。Triton 前端只把最内层的 if/else
对转成 `select`（指针 select 是合法的），外层保留成嵌套的条件控制流，
其分支产出是 `!tt.ptr` 类型的值——旧版编译器对"控制流块参数携带指针
类型"的支持不完整，一路 lower 到控制流图后报：

```
failed to legalize 'cf.br'
```

**判读**（basics/04 §4.5 分层表的实战应用）：报错发生在编译期、
带 `legalize` 字样、且你的 kernel 语法完全合法——这是编译器层，不是
kernel 写错。该 bug 已在新版编译器修复（修复方式：把"then/else 汇合
出指针"的菱形控制流改写成指针 select 链——语义等价，select 是全程
合法的）。修复后实测：**cat 111/111、concatenate 110/110、stack
48/48、hstack 20/20、vstack 16/16 全过**。

**旧版 wheel 的规避写法**就是 10.3：mask-select 无分支，两路 load 相加
——不产生"控制流携带指针"，任何版本都能编译。这也是一个通用的工程
直觉：**能用 mask/select 表达的"选择"，就不要用 if/elif**——无分支
写法对编译器更友好，对 SIMT/SIMD 硬件也更友好。

## 10.6 家族一览

| 算子 | 与 cat 的关系 | kernel 侧差异 |
|---|---|---|
| `concatenate` | torch.cat 的别名路径 | 同 cat |
| `stack` | 新增一维再 cat | host 侧 view/unsqueeze，kernel 同 copy |
| `hstack` | 1D: dim=0；2D+: dim=1 | dim=1 时映射换成列号分段 |
| `vstack` | dim=0（1D 先升 2D） | 同 cat |
| `torch.empty + copy_` | cat 的"退化实现" | 非连续输入的 fallback |

搬运家族的共同纪律：结果**逐位相等**才算过；mask 保护偏移；grid 按
输出切；输入非连续时要么 host 侧 `contiguous()`、要么 kernel 传
stride——**永远不要在 kernel 里假设输入连续**（生产实现连"输出切片
是否连续"都要检查后才走 kernel，否则 fallback `copy_`，见 10.2）。

## 10.7 验证

```bash
python cat.py                                   # 本章程序（3 组形状，exact）
# FlagGems: tests/test_cat.py / test_stack.py / test_hstack_vstack.py
```

golden：`torch.cat` / `torch.stack`，判据 `torch.equal`（f16/f32/int
一律逐位）。必测矩阵：

- 形状：拼接维含 0 的输入（空张量 cat 合法！）、单输入、K>2、
  非拼接维为 1；
- dtype：全家族 dtype 各一遍（纯搬运与 dtype 无关，但 int64 偏移
  计算容易写错——大张量时 `row * N` 会溢出 int32，host 侧确认
  numel < 2^31 或 kernel 内 `.to(tl.int64)`）；
- 非连续输入：`a[:, ::2]` 这类（验证 fallback 路径或 stride 版）。

## 10.8 练习

1. 跑通 10.3。把 B 路的 `other=0.` 改成 `other=1.`，预测哪一半输出会
   错、错成什么样，验证后改回。
2. 实现 dim=1 的 cat（`C[M, N_A+N_B]`）：grid 怎么切？列号 c 到
   (输入, 输入内列号) 的映射怎么写？golden `torch.cat([a,b], dim=1)`。
3. 实现 K=3 的无分支版：三路 load 三个 mask 相加。然后算一算：K=10
   时这个写法的代价是什么？（提示：每路 load 的 mask 判定和 other 填充
   都要执行，访存指令数 ×K——这就是 K 大时为什么生产走 host 循环或
   指针分支。）
4. 大数偏移实验：构造 numel 接近 2^31 的两个 f16 张量（如
   2×(1<<30) 元素需要 4GB——若内存不够，把 N 调大 M 调小构造
   `row * N` 单项超 2^31 的形状），观察 int32 偏移溢出的症状（错行
   而不是崩溃），然后给 kernel 加 `.to(tl.int64)` 修复。
5. 思考题（收官题）：本章 10.5 的排错路径——合法 kernel + 编译期
   `failed to legalize` → 编译器层 → 升级/规避——与 ops/09 §9.5 的
   三层边界、ops/06 §6.5 的归约边界表是同一套方法论。回忆 basics/04
   §4.5 的分层表，把"症状 → 层 → 动作"三列自己默写一遍，写不出的
   格子回去翻对应章节。

---

## 全书完

十章走下来，你应该已经具备：

- **写**：从 GEMM/attention 到 norm/softmax/scan/cat，覆盖 LLM 推理
  的全部算子形态（矩阵引擎、向量、串行搬运三条通路都见过）；
- **调**：TCM 预算、dtype 军规、grid 组织、autotune——每个性能决策
  都有实测数字背书；
- **排**：分层排查（kernel/编译器/运行时）、cache 纪律、A/B 对照、
  最小复现报告——失败时知道该干什么，而不是随机改代码。

工具链在快速演进：本章和前面各章标注的"旧版限制"会随新版 wheel 逐个
消失，**遇到任何"教程说会崩但实际过了"的地方，以你手上的版本实测为准**
——这也是 basics/04 §4.7 历史数字纪律的最后一课。

回到目录：[README](../README.md)
