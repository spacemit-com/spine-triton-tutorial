# Op 04 — Softmax

**重要性：★★★★☆**。`y[i] = exp(x[i]) / Σ exp(x[j])`，把一行实数变成
一行和为 1 的概率。attention 的核心非线性步骤（QK^T 分数 → 权重），也是
**数值稳定性**的入门必修课：naive 写法在 f16/f32 下都会溢出，正确写法
（减最大值）是必须内化的肌肉记忆。它还是 K3 上 fp16 + `tl.exp` 第一大坑
的现场。

## 4.1 这个算子在算什么，为什么会溢出

定义（按行，dim=-1）：

```
y[i] = exp(x[i]) / Σ_j exp(x[j])
```

手算例子——`x = [1, 2, 3]`：

```
exp(x)      = [2.718, 7.389, 20.086]
Σ           = 30.193
y           = [0.090, 0.245, 0.665]      # 和 = 1 ✓
```

**溢出问题**：`exp(89) ≈ 4.5e38` 已是 f32 上限（3.4e38 附近就 inf），
f16 更夸张——`exp(11.1)` 就超了 f16 最大值 65504。神经网络里 logits 上百
很常见，naive 写法直接得到 `inf/inf = nan`。

**数值稳定版**（数学上严格等价，因为分子分母同乘 `exp(-max)`）：

```
m    = max(x)
y[i] = exp(x[i] - m) / Σ_j exp(x[j] - m)
```

减掉最大值后，指数参数 ≤0，`exp ∈ (0, 1]`——永不溢出；分母至少有一项
是 `exp(0)=1`——永不为 0。**这就是三趟结构：max → sum → 归一化。**

## 4.2 并行方案与趟数

和 norm 家族一样：**行间独立 → grid=(n_rows,)，每 program 一行**。
行内三趟扫数据。

当一行能装进一个 block（`BLOCK_SIZE = next_power_of_2(n_cols)`）时，三趟
都在寄存器里完成——数据只从内存读一次、写一次，是最优访存形态。本章
主例子就是这种 **ONE_TILE** 形态。

行装不下单 tile 时走 **multi-tile** 路径：max/sum 用分块循环 + f32 累加器
扫两趟，第三趟归一化。生产实现（FlagGems general kernel）还有个细节：
归一化趟**从后往前走 tile**，用 `prev_multiple_of(N, TILE_N) = cdiv(a,b)*b - b`
（严格小于 a 的最大 b 倍数）定位尾块起点——`N == TILE_N` 时得 0，恰好
全覆盖。

## 4.3 完整可运行程序（ONE_TILE 版）

（kernel 来源：`python/examples/test_softmax.py`，即 Triton 官方
03-softmax 教程；host 重写为独立脚本，f16 验证按 K3 军规处理。）

```python
# softmax.py — 每 program 一行的 softmax（行装得下单 block）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def softmax_kernel(output_ptr, input_ptr, in_stride, out_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)                    # 每 program 一行
    row_start_ptr = input_ptr + row_idx * in_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 整行 load 进寄存器。关键：other=-inf ——
    # 越界 lane 必须是 max 的“吸收元”（-inf 不可能是最大值），
    # 且 exp(-inf)=0 对分母也无贡献。填 0 就全错了。
    row = tl.load(row_start_ptr + col_offsets,
                  mask=col_offsets < n_cols, other=-float('inf'))

    row_f32 = row.to(tl.float32)                  # ← 军规：exp 前先升 f32！
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)   # 数值稳定
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    out_ptrs = output_ptr + row_idx * out_stride + col_offsets
    tl.store(out_ptrs, softmax_output.to(output_ptr.dtype.element_ty),
             mask=col_offsets < n_cols)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)   # 一行装进一个 block
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                              BLOCK_SIZE=BLOCK_SIZE)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    # f32 场景（仓库示例同款，1823×781 非 2 幂行长）
    x = torch.randn(1823, 781)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    torch.testing.assert_close(y_triton, y_torch)
    print("PASS f32 1823x781")

    # f16 场景：kernel 内先 .to(tl.float32) 再 exp，才过得了 dtype 检查
    x16 = torch.randn(128, 256, dtype=torch.float16) * 5   # ×5 制造大 logits
    y16 = softmax(x16)
    ref16 = torch.softmax(x16.float(), dim=-1).half()
    torch.testing.assert_close(y16, ref16, atol=1e-3, rtol=1e-3)
    print("PASS f16 128x256 (large logits)")
```

```bash
python softmax.py
# PASS f32 1823×781
# PASS f16 128x256 (large logits)
```

### 逐段精讲

- **`other=-float('inf')` 是本 kernel 的灵魂**。回忆 basics/03 §3.3 的
  “归约前填单位元”规则，softmax 一趟里有两个归约、单位元不同：
  - `tl.max`：单位元是 **-inf**（任何真实值都比它大）；
  - `tl.sum(exp(...))`：需要 `exp(填充值) = 0`，即填充值 = **-inf**。
  一个 -inf 同时满足两者。若填 0：max 在“全负行”上会错取 0，exp(0)=1
  还会给分母凭空加 1——两处都错。
- **`.to(tl.float32)` 不是可选优化，是硬性要求**：`tl.exp` 只收
  fp32/fp64，f16 直接 `ValueError`（4.5 坑 1）。f32 输入时这行是无害
  no-op，写上让同一 kernel 两种 dtype 通吃。
- **`numerator / denominator`**：向量 ÷ 标量，自动广播。此时 numerator
  的越界 lane 是 `exp(-inf)=0`，除完还是 0，但 store 有 mask，写不出去。
- **输出 cast**：普通指针 store 其实会隐式 cast，这里显式
  `.to(output_ptr.dtype.element_ty)` 是为了和 block-ptr 版行为一致、
  意图明确。
- **grid=(1823,)**：每行一个 program。行数超 512 时旧 libspert（<0.6.3）
  会静默丢弃（basics/04 §4.5），0.6.3+ 无碍。
- `num_warps` 参数（仓库示例里有）是 GPU 概念，CPU 后端忽略即可。

## 4.4 multi-tile 路径的形状（读懂即可）

行长超过单 block 容量（或 BLOCK 想控制在小值）时：

```python
# 趟1: 分块扫 max —— f32 累加器，初值 -inf
m = tl.full([BLOCK_N], float('-inf'), tl.float32)
for off in range(0, N, BLOCK_N):
    x = tl.load(..., mask=..., other=float('-inf')).to(tl.float32)
    m = tl.maximum(m, x)
m_row = tl.max(m, axis=0)                     # 行 max，标量

# 趟2: 分块扫 sum(exp(x - m_row)) —— 初值 0
l = tl.zeros([BLOCK_N], tl.float32)
for off in range(0, N, BLOCK_N):
    x = tl.load(...).to(tl.float32)
    l += tl.exp(x - m_row)                    # 越界 lane 要 -inf 或 where 归零
l_row = tl.sum(l, axis=0)

# 趟3: 分块写 exp(x - m_row) / l_row（生产实现从尾块往前走）
```

结构与 norm 家族的多趟扫描一模一样，只是累加算子换成了 `maximum` 和
`sum(exp(·))`。**“行内多趟 + f32 累加器 + 单位元填充”是这一整类算子的
通用模板**——你已经见过它三次了（layernorm、rmsnorm、softmax）。

attention 里的 online softmax（flash 风格，max/sum 单趟在线更新）是它的
进阶变体，ops/08 再见。

## 4.5 K3 关键点与坑

1. **fp16 撞 `tl.exp` dtype 检查（第一大坑）**。
   `tl.exp/log/sqrt/...` 只接受 fp32/fp64（`triton/language/math.py` 的
   `_check_dtype`，**上游故意设计**——注释明确“要求用户显式 cast”，GPU 上
   同样报错）。fp16 tile 不 cast 直接喂 exp = CompilationError。
   生产环境的三种解法（当 kernel 源码不许改时怎么办，真实案例）：
   - kernel 内显式 `.to(tl.float32)`——自己的 kernel 直接这么写；
   - **heuristic 路由**（FlagGems 生产采用）：`softmax_heur_one_tile_per_cta`
     检查 launch args 里 tensor 的 dtype，fp16/bf16 返回 False → 强制走
     multi-tile 路径（其累加器本来就是 f32，天然安全且数值更稳）；
     backward 无 exp，拆独立 heuristic 维持 ONE_TILE 语义。修复后
     test_softmax 112/112 PASS；
   - 给 math.py 加全局自动提升——被否决（改上游语义，影响面不可控），
     别走这条路。
2. **归约 dtype**：`tl.max/tl.sum` 对 fp16 输入返回 fp16，长行累加超容差。
   ONE_TILE 路径全程 f32（4.3 的写法）或 multi-tile 的 f32 累加器。
3. **旧二进制的行宽边界问题**（已在 spine-mlir `c63415b`/`d04296f` 代际
   修复）：行长非 vscale 整倍数时的 scalable 转换、in_bounds 误推断 +
   poison init 垃圾 lane——症状如“BLOCK=16 的 int32 sum 恒差 -16”、f32
   NaN。碰到先在**现役二进制**上复测，再怀疑 kernel。
4. **改了 heuristic/语言层 Python 后必须换 TRITON_CACHE_DIR**——cache key
   不含 language 模块改动，旧 .so 会掩盖你的修改（basics/04 §4.2）。

## 4.6 spine_raw 版本与两盆冷水

默认 path 有 `tle.vexp`（无 dtype 检查的向量 exp），可以做向量级
softmax：`python/tests/raw/test_raw_softmax.py`、`test_raw_log_softmax.py`。
组织方式：grid=(1,)、外层循环串行遍历行、每行三趟（max/sum/scale），
tail 用空 range loop（basics/05 §5.4）。历史上 2-pass softmax 曾是
loop-env 泄露 dominance 崩溃的触发器（已修，basics/05 坑 4）。

但实测结论要泼冷水：

- **decode 场景小 K 的 fused softmax 跑不过 `torch.softmax`**——K=33~52
  时慢 1.4x。原因：default path 无 program_id 没并行、tiny data 下 per-op
  开销 >> 计算、torch.softmax 是优化过的 C++ op。decode softmax 22% 的
  开销是 **Python dispatch**（f32 intermediate + 类型转换 + tensor alloc），
  kernel fusion 消不掉 dispatch 本身。
- **fuse QKT+softmax+AV 成单 kernel 已被实测否决**：Python
  softmax+mask+scale 仅占 prefill 墙钟 ~0.8%（0.9ms/layer × 28 层 =
  25.4ms），收益 << 实现成本（LLVM-direct 没有 exp，要多项式近似）。
  详见 ops/08。

**教训：融合决策要先 profile。**“理论上省访存”不等于“实测更快”，
tiny shape + dispatch 主导的场景尤其如此。

## 4.7 验证

```bash
python softmax.py
python3 python/examples/test_softmax.py
python3 -m pytest python/tests/raw/test_raw_softmax.py -q
```

golden：`torch.softmax(x.float(), dim=-1)`；fp16 结果 atol/rtol 1e-3~1e-2
（概率值都在 [0,1]，atol 可以收紧）。极端值测试：加一组 `x` 含 ±100 大
logits 的用例，确认无 nan/inf（验证减 max 真的生效）。

## 4.8 练习

1. 跑通 4.3。然后把 `other=-float('inf')` 改成 `other=0.`，用**全负数行**
   （如 `x = -torch.rand(64, 100) - 5`）重跑——预测输出哪里错、为什么，
   再验证。
2. 把 `.to(tl.float32)` 那行删掉，f16 输入重跑，抄下报错原文（这就是
   `_check_dtype` 的标准签名，以后一眼认得）。
3. 实现 multi-tile 版（4.4 骨架补全），用 `BLOCK_N=256` 跑 n_cols=1000，
   与 ONE_TILE 版结果对齐（atol 1e-6，同为 f32 时应几乎逐位一致）。
4. 因果 mask 热身（为 ops/08 做准备）：`n_cols=64` 的下三角 mask——
   `tl.load` 后加 `x = tl.where(col_offsets[None,:] <= row_offsets[:,None],
   x, -inf)` 再走三趟。golden：
   `torch.softmax(x.masked_fill(~mask, -inf), -1)`。

下一篇：[05-gemv.md](05-gemv.md) —— LLM decode 的命脉。
