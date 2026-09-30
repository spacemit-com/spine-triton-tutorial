# Op 09 — Cumsum / Scan（前缀和）

**重要性：★★★☆☆**。`y[i] = Σ_{j≤i} x[j]`——前缀扫描（scan）。它与
归约（ops/06）是一对：归约把 N 个数变 1 个，scan 把 N 个数变 N 个
（每个位置是到它为止的归约值）。看着比 attention 简单得多，但它是
**编译链上坑密度最高的算子族**：scan 的 lowering 会产生大量中间 buffer，
行长与向量寄存器宽度的边界关系、运行时的栈空间，都出过问题。学这一章
一半是学 scan 本身，一半是学"怎么判断一个失败是编译器/运行时问题而不是
自己的 kernel 写错了"。

cumsum/cumprod/cummax/cummin，以及内部复用 scan 的
repeat_interleave/index_put/nonzero/randperm，都属本章范围。

## 9.1 这个算子在算什么：手算一遍

```
x = [3, 1, 4, 1, 5]
cumsum(x) = [3, 4, 8, 9, 14]      # y[i] = x[0]+...+x[i]
```

三个关键性质：

- **输出长度 = 输入长度**（对照归约的 N→1）；
- **顺序依赖**：y[i] 依赖前面所有 x[j]——这是并行的天敌，也是 scan
  存在专属算法（Hillis-Steele：log N 轮"每元素加上 2^r 位之前的元素"）
  和专属坑（每轮都要中间 buffer）的原因；
- **块内好办，跨块难办**：`tl.cumsum` 一步完成一个 BLOCK 内的 scan，
  但 BLOCK 之间的前缀传递需要额外组织——9.3 的两趟法是标准答案。

dtype 语义与 torch 对齐：`cumsum(int32)` 输出仍 int32（**包括溢出
行为**——torch 的 int32 cumsum 溢出会回绕，你的 kernel 也应如此；
测试 golden 别随手 `.float()`，会掩盖溢出语义差异）。int64/fp64 保持
原 dtype；其余整数升 int32、其余浮点升 f32（9.3 会看到生产 kernel
正是这个提升表）。

## 9.2 并行方案：scan-then-fan 两趟法

长序列沿 scan 维切成 part_num 段，两段 kernel：

```
趟1（并行）: grid=(part_num,)
  每 program 对自己那段做块内 tl.cumsum，写进 out；
  同时把该段的总和写进 partial_sum[pid]。
趟2（并行）: grid=(part_num,)
  pid=0 的段什么都不做；
  pid>0 的段把 partial_sum[pid-1]（前一段的总和）加到自己的每个元素上。
```

趟2 隐含一个约定：partial_sum 里存的是**每段总和**，而每段需要的
base 是"前面所有段的总和"。生产实现（FlagGems `ops/cumsum.py`）的
解法是**递归**：对 partial_sum 这个小数组再做一次同样的 scan_then_fan
（in-place 前缀化），之后 add_base 读 `partial_sum[pid-1]` 就是严格
前缀。数组只有 part_num 个元素，递归一层就到底——这正是"段2 的输入
小到怎么算都快"（ops/06 §6.2 形态 B）的思想。本章 9.3 的教学版取
part_num=2 的捷径：只有一段前驱，直接读 `partial_sum[pid-1]` 即严格
前缀，不需要递归（练习 1 会让你亲手撞破这个限制并修复）。

行 scan（`tl.cumsum(x_2d, axis=1)`，每行独立）则是
`grid=(A, part_num, C)` 的三维组织：A/C 两维是独立的行/列，中间维
切段——同一套两趟法逐行并行。

## 9.3 完整可运行程序

（kernel 结构来源：FlagGems 主线 `ops/cumsum.py` 的
`scan_part_sum_kernel` + `add_base_sum_kernel`，host 重写为独立脚本。）

```python
# cumsum.py — 两趟 scan-then-fan（1D，part_num=2）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_sum_kernel(inp, out, partial_sum, n_elements, part_num,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask, other=0.)
    # dtype 提升表（与 torch.cumsum 语义对齐，见 9.1）：
    #   int64/uint64/fp64 保持；其余整数 → int32；其余浮点 → f32
    if tl.constexpr(inp_vals.dtype.is_int64() or inp_vals.dtype.is_uint64()
                    or inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)

    result = tl.cumsum(inp_vals, axis=0)        # 块内 scan，一步到位
    tl.store(out + offset, result, mask=mask)
    # 本段总和（用 sum 而不是 result 的最后一个元素——mask 下更安全）
    tl.store(partial_sum + pid, tl.sum(inp_vals))


@triton.jit(do_not_specialize=["n_elements", "part_num"])
def add_base_sum_kernel(out, partial_sum, n_elements, part_num,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    if pid > 0:                                  # 标量条件：第 0 段无 base
        base = tl.load(partial_sum + pid - 1)
        out_vals = tl.load(out + offset, mask=mask, other=0.)
        tl.store(out + offset, (out_vals + base).to(out_vals.dtype),
                 mask=mask)


def cumsum(x, part_num=2):
    # 注意：本教学版仅对 part_num=2 正确——partial_sum[pid-1] 恰好是
    # 严格前缀。part_num>2 需要先把 partial_sum 前缀化（练习 1）。
    n = x.numel()
    BLOCK_SIZE = triton.cdiv(n, part_num)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
    out = torch.empty_like(x)
    partial_sum = torch.empty(part_num, dtype=x.dtype)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scan_part_sum_kernel[grid](x, out, partial_sum, n, part_num,
                               BLOCK_SIZE=BLOCK_SIZE)
    add_base_sum_kernel[grid](out, partial_sum, n, part_num,
                              BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in [5, 64, 1000, 4096]:
        x = torch.randn(n, dtype=torch.float32)
        got = cumsum(x)
        ref = torch.cumsum(x, dim=0)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
        print(f"PASS f32 n={n}  max diff={(got-ref).abs().max().item():.2e}")

    xi = torch.randint(-100, 100, (1000,), dtype=torch.int32)
    got, ref = cumsum(xi), torch.cumsum(xi, dim=0)
    assert torch.equal(got, ref)                 # 整数 scan 必须逐位相等
    print("PASS int32 n=1000 (exact)")
```

```bash
python cumsum.py
# PASS f32 n=5 ... PASS f32 n=4096 ...
# PASS int32 n=1000 (exact)
```

### 逐段精讲

- **`tl.cumsum(inp_vals, axis=0)`**：块内 scan 是一个原语调用，编译器
  内部展开成 Hillis-Steele（log BLOCK 轮移位加）。你写的是"什么"，
  展开的"怎么"正是 9.6 那些坑的来源。
- **`if pid > 0:` 标量条件**：注意这是**标量** if（条件不含向量），
  整个 store 段被条件包裹。历史上编译器曾把这类标量条件的 load/store
  **静默丢掉 mask**（输出看起来"算过了"但第 0 段也被加了 base，或
  干脆没写）——新版已修复。见到"标量条件下的 store 行为诡异"，先
  升级 wheel 换新 cache 复测（basics/04 §4.2），再怀疑逻辑。
- **段总和用 `tl.sum(inp_vals)` 而不是 `result` 的尾元素**：尾元素
  在 `n % BLOCK != 0` 时落在 mask 外，需要额外 where 选取；直接对
  输入求和天然免疫（越界 lane 已是 0）。**`other=0.` 在这里不是
  风格问题**——masked load 不写 `other` 时越界 lane 是未定义值，会
  直接污染 partial_sum（basics/03 §3.3 的 mask 军规）。
- **`(out_vals + base).to(out_vals.dtype)`**：base 被提升过（int→
  int32），加完要 cast 回 out 的原 dtype——int16 cumsum 的输出仍是
  int16，靠这一步。
- **grid = cdiv(n, BLOCK)**，part_num=2 时只有 2 个 program——离任何
  grid 上限都很远。但生产 kernel 的 BLOCK 是 autotune 出来的，小
  BLOCK × 大 n 会到几百上千 program：旧版 wheel 对超大 grid 有上限
  （basics/04 §4.5），新版运行时自动分块。
- **两趟 = 两次 launch**：K3 上 dispatch 开销不可忽略（ops/05 §5.5），
  n 很小时（如 <4096）单 program 串行 scan 反而可能更快——老规矩，
  实测说了算。

## 9.4 spine_raw 版本：从串行最小形到三阶段并行

### 串行最小形（语义演示）

（来源：`python/tests/raw/test_raw_cumsum.py`——文件头注释原话："Unlike
the reduce family, scan emits one output per position. The simplest correct
form is a scalar scf.for with a running f32 accumulator."）

```python
@tle.raw_kernel
def cumsum_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True),
                     N: tle.index):
    tle.vconfig(-1, 1)                    # 设 VL（vzero 需要）
    acc = tle.vreduce_sum(tle.vzero(f32)) # 0.0 作为 f32 标量（scan 种子）
    for i in tle.range(0, N, 1):
        xi = tle.sload(X, i, dtype=f32)   # 标量 load
        acc = acc + xi                    # 运行累加器
        tle.sstore(out, i, acc)           # 每个位置都 store
```

O(N) 纯串行、grid=(1,) 单 program——但它是**语义上最直白的 cumsum**：
一个累加器、逐元素加、逐位置写。测试 N=16/64/100/257 全过（golden
`torch.cumsum`，atol/rtol 1e-4）。

它用 grid=(1,) 是因为 **raw kernel 内部没有 `program_id`**——但这不等于
"raw 只能串行"。标准解法是 basics/05 §5.5 的**host pid 传参**并行形态：
host `@triton.jit` kernel 持有真正的 grid，把 `tl.program_id(0)` 当普通
标量经 `_sr_call` 传进 raw kernel。scan 也能这样并行——把 N 切成块，
块间用 9.2 的 scan-then-fan 组织。下面的三阶段版就是现成的真实测试。

### 并行版：三阶段 block-scan（完整可运行）

（来源：`python/tests/raw/test_raw_cumsum_vec.py`。把 N 切成 P = N/VL 块，
VL=64：**Phase1** grid=(P,) 每 program 向量归约自己块的总和；**Phase2**
grid=(1,) 对 P 个块和做排他前缀和；**Phase3** grid=(P,) 每 program 块内
串行 scan 再加块基值。正是 9.2 两趟法 + "段2 前缀化"的 raw 实现。）

```python
# cumsum_raw_parallel.py — 三阶段 block-scan cumsum（host pid 传参并行）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f32 = tle.f32
VL = 64          # 块宽：必须与 kernel 内 vconfig(-1,1) 的实际返回一致


# ── Phase 1: 每块求自己的总和（grid=(P,)，向量归约）──────────────
@tle.raw_kernel
def block_sum_kernel(X: tle.mem(f32), block_sums: tle.mem(f32, out=True),
                     N: tle.index, p: tle.index):
    nvl = tle.vconfig(-1, 1)
    base = p * nvl                          # ← 我是第几块：host 传来的 p
    vx = tle.vload(X, base)                 # 整块一次读进向量寄存器
    tle.sstore(block_sums, p, tle.vreduce_sum(vx))


@triton.jit
def block_sum_host(X, block_sums, N, P):
    p = tl.program_id(0)                    # ← 并行度在 tl 层
    if p < P:                               # tl 层标量 if，合法
        _sr_call(block_sum_kernel, outputs=[], inputs=[X, block_sums, N, p])


# ── Phase 2: P 个块和的排他前缀和（grid=(1,)，合理的小串行）──────
@tle.raw_kernel
def prefix_offset_kernel(block_sums: tle.mem(f32),
                         offsets: tle.mem(f32, out=True), P: tle.index):
    tle.vconfig(-1, 1)
    acc = tle.vreduce_sum(tle.vzero(f32))   # f32 标量 0.0（scan 种子）
    for i in tle.range(0, P, 1):
        tle.sstore(offsets, i, acc)         # 排他前缀：先写（不含自己）
        acc = acc + tle.sload(block_sums, i, dtype=f32)   # 再加上自己


@triton.jit
def prefix_offset_host(block_sums, offsets, P):
    _sr_call(prefix_offset_kernel, outputs=[], inputs=[block_sums, offsets, P])


# ── Phase 3: 块内串行 scan + 加块基值（grid=(P,)）────────────────
@tle.raw_kernel
def apply_prefix_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True),
                        offsets: tle.mem(f32), N: tle.index, p: tle.index):
    nvl = tle.vconfig(-1, 1)
    base = p * nvl
    offset = tle.sload(offsets, p, dtype=f32)   # 我前面所有块的总和
    acc = tle.vreduce_sum(tle.vzero(f32))
    for j in tle.range(0, nvl, 1):              # 块内 64 步串行 scan
        xi = tle.sload(X, base + j, dtype=f32)
        acc = acc + xi
        tle.sstore(out, base + j, acc + offset)


@triton.jit
def apply_prefix_host(X, out, offsets, N, P):
    p = tl.program_id(0)
    if p < P:
        _sr_call(apply_prefix_kernel, outputs=[], inputs=[X, out, offsets, N, p])


def cumsum_vectorized(X: torch.Tensor) -> torch.Tensor:
    N = X.numel()
    P = N // VL
    Nfloor = P * VL
    out = torch.zeros(N, dtype=torch.float32)
    if P > 0:
        bs = torch.zeros(P, dtype=torch.float32)
        offs = torch.zeros(P, dtype=torch.float32)
        Xf = X[:Nfloor].contiguous().reshape(-1)
        block_sum_host[(P,)](Xf, bs, Nfloor, P)      # Phase 1
        prefix_offset_host[(1,)](bs, offs, P)        # Phase 2
        apply_prefix_host[(P,)](Xf, out[:Nfloor], offs, Nfloor, P)  # Phase 3
    if Nfloor < N:                       # 尾巴（<VL 个）：host 侧 torch 收尾
        last = out[Nfloor - 1].item() if Nfloor > 0 else 0.0
        out[Nfloor:] = torch.cumsum(X[Nfloor:].float(), dim=0) + last
    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    for N in [64, 512, 8192, 100, 513]:    # 前三个 VL 对齐，后两个带尾巴
        X = torch.randn(N, dtype=torch.float32)
        got, ref = cumsum_vectorized(X), torch.cumsum(X, dim=0)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
        print(f"PASS N={N}")
```

```bash
python cumsum_raw_parallel.py
# PASS N=64 ... PASS N=513
python3 -m pytest python/tests/raw/test_raw_cumsum_vec.py -q   # 原测试
```

### 逐段精讲

- **host 三件套**：`p = tl.program_id(0)` + `if p < P:` +
  `_sr_call(..., inputs=[..., p])`。raw kernel 不知道自己是第 p 个
  program，它只收到一个标量 `p`——并行语义全在 tl 层，raw 层保持
  "无 program_id、无 if"（basics/05 §5.5 并行形态；ops/06 §6.4 的行
  归约、ops/05 §5.4 的 GEMV 同款）。
- **提速在哪：关键路径**。串行版关键路径 = N 次加法；三阶段版 ≈
  VL（Phase3 块内）+ P（Phase2）+ 两次额外 launch。N=8192 时
  8192 步 → 64+128 步，且 Phase1 的块归约是向量化的（一条
  `vreduce_sum` 收掉 64 个元素）。注意 Phase3 块内仍是 64 步串行
  标量循环——scan 的块内串行无法消除，能消除的只有块间串行。
- **Phase 2 的 grid=(1,) 是合理的**：P = N/64，N=8192 也才 128——
  128 步串行前缀和远比再来一次并行 launch + 同步便宜（ops/06 §6.2
  形态 B："段2 的输入小到怎么算都快"；9.2 生产实现的递归前缀化同理）。
  循环体内**先 `sstore` 再累加**是排他前缀（exclusive）的关键：
  `offsets[p]` 必须是"前面所有块的总和"，不含自己——写反顺序整个输出
  平移一个块和（练习 3 会让你亲手验证）。
- **与 9.3 的 tl 两趟法对照**：Phase1+2 = `scan_part_sum` 的"段总和"
  加 partial_sum 前缀化，Phase3 = 块内 `tl.cumsum` + `add_base` 合一。
  tl 版块内 scan 是原语一步到位，raw 版把 64 步显式写出来——又是
  "raw 把编译器替你做的展开给你看"（ops/06 §6.4）。
- **尾巴放 host 侧**：`N % VL != 0` 时剩不到 64 个元素，default path
  没有 if 保护越界，host 直接 `torch.cumsum(tail) + last` 收尾最省事
  （kernel 内做也可以——`vconfig` 收窄 + 尾循环，ops/05 §5.4 的写法；
  这里选择不做，因为最多 63 个元素）。
- **三阶段 = 三次 launch**：N 很小时 dispatch 开销可能超过并行收益
  （9.3 精讲最后一条同款）——原测试的对齐用例从 N=64（P=1！退化成
  串行 + 两次多余 launch）到 N=8192 全过，正确性与性能是两回事。

对照读两个版本（串行 vs 三阶段），你能看清 scan 并行化的全部要素：
分块、块归约、小块前缀、块内 scan + 基值——以及每一段该放 tl 层还是
raw 层。

## 9.5 K3 关键点：三层边界

scan 族的失败历史上来自三个不同的层，症状和处置完全不同——这是
basics/04 §4.5 分层排查的最好教材。

### 层 1：运行时栈空间（执行期段错误）

Hillis-Steele 每轮 r 分配 `(TILE + 2^r) × elemSize` 的中间量，kernel
在运行时的协程栈上跑。**旧版 wheel 的协程栈偏小**（16KB 量级），
int64 + TILE=1024 的 scan 中间量要 ~128KB——栈越界，执行期 SIGSEGV。
新版 wheel 已把栈扩大（512KB 量级），cumsum/multinomial/
repeat_interleave/index_put/cat 一大家族的执行期崩溃随之消失。

**签名**：kernel 编译成功（.so 正常生成）、执行时段错误；TILE 越大、
元素越宽（int64）越容易触发。处置：升级 wheel；仍崩则按 basics/04
§4.6 的段错误排查顺序。

### 层 2：行长 vs 向量寄存器宽度（编译期断言崩溃）

与归约同源（ops/06 §6.5 表），scan 是重灾区，因为 `tl.cumsum` 的
展开对行长更敏感：

| 边界 | 症状 | 状态 |
|---|---|---|
| 行长恰好等于寄存器宽（len 15/16 类） | rc=134 断言崩溃、无 Python 栈 | 新版已修复 |
| 行长不是寄存器宽整倍数（部分 2D scan shape） | rc=134 断言崩溃 | 旧版限制，部分 shape 仍残留 |
| 行长超大（1041 → BLOCK 2048 类） | 指令生成阶段崩溃 | 旧版限制 |

**这些崩溃只在 fresh 编译时出现——cache 命中会掩盖**（"清了缓存就
必现"正是它们的签名，反过来"一直没崩"可能只是 .so 缓存命中，
basics/04 §4.2）。处置：rc=134 无 Python 栈 = 编译器层，不要在 kernel
源码里瞎改，走 basics/04 §4.4 流程（新 cache → 最小化 → 升级 wheel →
报告）。能绕的先绕：调整 BLOCK 让行长避开边界。

### 层 3：数值假象（结果错但不是 kernel 错）

- **旧版 wheel 丢弃超大 grid**：cumsum 输出 99.9% 垃圾（未初始化
  内存）——先升级 wheel（basics/04 §4.5/§4.7）；
- **旧版编译器的边界推断 bug**：BLOCK=16 的 int32 `tl.sum` 恒差 -16、
  f32 出 NaN——越界垃圾 lane 被折叠进归约，新版已修复（ops/06 §6.5
  表第一行同源）。

**引用任何 scan 族的历史失败数字前，先确认批次的 wheel 版本**
（basics/04 §4.7）——大量"scan 数值错误"的旧记录其实是这两类假象。

## 9.6 已知残留（旧版限制，判读用）

- **cumprod 在部分 shape 仍编译期崩溃**（与 cumsum 同族的 scan
  lowering，行清缓存后必现）；
- **int64 含负值的 scan 路径**：曾有 repeat_interleave 的结果错值
   traced 到 cumsum 的 int64 负值处理——整数 scan 出错先用小数组
  含负值单独复现；
- **整数 dtype 的 scan/归约断言崩溃家族**：判据非常清晰——**同一
  kernel，fp16/fp32 全过、int dtype（i1/i16/i32/i64）全崩**（rc=134）。
  这是旧版编译器对整型元素的处理限制；碰到先升级 wheel，仍崩则按
  §4.4 报告，附上"浮点过、整数崩"这个判据（能帮维护者省一半时间）。

## 9.7 复合算子的分段探针法（重要方法论）

scan 很少单独出现——repeat_interleave、index_put、nonzero、randperm
内部都有 cumsum。复合算子失败时，**第一反应往往是怀疑最深的那层
（scan lowering），而这通常是错的**。

真实案例：repeat_interleave 数值失败曾长期被归因于 scan。正确做法是
**分段探针**——把它拆成三段独立跑：

```
repeat_interleave(x, repeats) = cumsum(repeats)        # 段1：算输出偏移
                              + repeat_kernel(x, ...)  # 段2：搬运
                              + index_select(...)      # 段3：收集
```

三段分开测（各写最小 golden），结果：段1、段2 对 len 15~320 全对，
失败全在**段3 index_select**（它自己的 grid 组织问题）。scan 是清白的。

方法论提炼：复合算子失败 → 按 kernel 边界拆段 → 每段独立最小复现 →
定位到段后再进该段的分层排查（§9.5）。不要直接怀疑最深的那层。

## 9.8 验证

```bash
python cumsum.py                                                    # 本章程序
python3 -m pytest python/tests/raw/test_raw_cumsum.py python/tests/raw/test_raw_cumsum_vec.py -q
# FlagGems: tests/test_scan_ops.py（cumsum/cumprod/cummax/cummin）
```

golden：`torch.cumsum`（dim 对齐）。三件事必须覆盖：

- **dtype 矩阵**：f32/f16/int32/int64——int 结果要求**逐位相等**
  （`torch.equal`），浮点按归约长度放容差（ops/06 §6.6 同款，cumsum
  的误差随位置增长，尾部最松）；
- **长度矩阵**：n < BLOCK、n = BLOCK、n % BLOCK != 0、行长 15/16/17
  这类寄存器宽边界（§9.5 层 2 的触发点）；
- **每次改编译器相关代码后换新 TRITON_CACHE_DIR**（§9.5 层 2 的崩溃
  会被缓存掩盖，basics/04 §4.2）。

> 跑 pytest 的一个环境细节：在仓库测试目录直接跑时，conftest 可能尝试
> 往当前目录写结果文件——没有写权限时会以 rc=1 结束但**用例其实全过**
> （摘要行在日志中间不在尾部）。换个可写目录跑（如 /tmp）最干净。

## 9.9 练习

1. 跑通 9.3。把 part_num 改成 4 和 8，注意 `add_base_sum_kernel` 读的
   是 `partial_sum[pid-1]`——part_num>2 时这样对吗？动手验证你的猜想
   （提示：int32 输入、逐位相等判据），不对就修（两种修法：先对
   partial_sum 做 scan，或 base 改成 `tl.sum(partial_sum[:pid])` 式的
   循环加载）。
2. 实现 `cummax`：把 `tl.cumsum` 换成 `tl.cummax`，两趟法的"段总和"
   换成什么？（提示：max 的结合律——段 base 是前缀 max，合并操作是
   `tl.maximum`。）golden `torch.cummax`。
3. 跑 `python3 -m pytest python/tests/raw/test_raw_cumsum_vec.py -q`，
   对照 9.4 读源码并回答：(a) 把 Phase 2 的 `sstore` 挪到 `acc +
   sload` 之后，输出会错成什么样？（inclusive vs exclusive 前缀——先
   预测每块偏多少，再动手验证。）(b) N=8192 时算一算三阶段版的关键
   路径步数和串行版的（9.4 精讲第 2 条），解释 Phase 3 块内仍是 64 步
   串行、整体为什么还能快。
4. 边界实验（在 K3 上，每换 shape 记得新 cache）：构造行长为 15、16、
   17 的二维 cumsum（`tl.cumsum(x_2d, axis=1)`，grid=(n_rows,)），
   记录哪些能过、哪些崩。若崩，按 §9.5 层 2 的判读写一段 30 字以内的
   问题描述（这就是给维护者的最小有效报告，basics/04 §4.4）。
5. 思考题：9.3 的两趟法对 n=100 是杀鸡用牛刀（两次 launch）。写一个
   单 program 串行版（raw 版 9.4 的 tl 翻译：`for off in range(0, n,
   BLOCK)` + 运行 base），实测两版在 n=100/1000/10000 的墙钟交叉点。

下一篇：[10-cat.md](10-cat.md) —— 拼接与搬运，全书收官。
