# Op 07 — Elementwise 与融合（add / silu / gelu）

逐元素算子单看没有技术含量，但它们 (a) 数量最多、(b) 是 **fusion 的载体**
（bias+激活、mul+add 融进 GEMM epilogue），(c) 把 libdevice/math 函数链路的
坑全踩了一遍。本章以 add、silu、gelu 为代表。

## 标准写法与 grid 安全性

vec add 见 basics/03。FlagGems 里大部分二元逐元素算子走 **pointwise_dynamic**
codegen：`num_ctas = min(max_grid, num_tiles)` + kernel 内循环 → **天然
grid-safe**（不受旧 runtime grid≤512 限制）。但注意其 **in-place/out 变体走
手写平坦 kernel**，不享受该保护——历史上 le/maximum/where/clamp 的 in-place
变体失败而 out-of-place 通过，就是这个差别。

## silu / gelu：math 函数链路

```python
# silu: x * sigmoid(x)
y = x * tl.sigmoid(x)                      # sigmoid 内部 f32

# gelu (tanh 近似 / erf 精确两版，FlagGems 均有)
# 关键：COMPUTE_FP32 门控 —— fp16 输入先提升 f32 算完再 cast 回
x = x.to(tl.float32)
y = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
y = y.to(out.dtype.element_ty)
```

K3 关键点：

1. **`tl.exp/log/tanh/...` 只收 fp32/fp64**（同 softmax 军规）：fp16 输入必须
   先 cast。审计方法：全量 grep 调用点，确认每处都有前置 cast / f32 提升 /
   COMPUTE_FP32 门控。
2. **libdevice 来自 cpu shim**：`triton.language.extra.libdevice`（spine-triton
   维护 `language/cpu/libdevice.py`）。正确形态是 `@core.extern` +
   `extern_elementwise(..., {(dtype,): ("math.X", dtype)}, ...)` 映射到 MLIR
   math dialect。曾经 15 个 shim（acos/atan/tan/cosh/exp2/log2/…）调不存在的
   `_semantic.create_X`，已全部修复；C++ 侧 `ConvertExternSpecialMath` 白名单
   （isUnaryMathSymbol）必须与 Python 表同步扩。
3. **constexpr 解包坑**：JIT 内 `pow(x, 2)` 的 `2` 是 `constexpr[2]`；直调
   `binary_op_type_checking_impl` 前必须 `isinstance(x, core.constexpr)` 解包
   `.value`，否则 `'constexpr_type' object has no attribute 'scalar'`
   （gelu_backward 24 例编译崩的根因）。
4. **spine-opt 可 lower 的 math op 清单**（e2e 无残留）：acos/asin/atan/cos/
   sin/tan/cosh/sinh/tanh/exp2/expm1/log2/log10/log1p/cttz；
   **acosh/asinh/atanh/cbrt 残留不 lower** → mlir-translate 报
   `Dialect 'math' not found`（当前无算子调用，非阻塞）。
5. `ffs` 用 `math.cttz` + select 实现 CUDA 语义（1-based，0 返回 0）；
   op 注册名是 `math.cttz/ctlz`，没有 "math.ctz"。

## 融合模式

### epilogue 融合（GEMM + bias + 激活）

```python
acc += tl.dot(a, b)                # K-loop 后
c = acc + bias[None, :]            # epilogue 直接在 f32 acc 上做
c = c * tl.sigmoid(c)              # silu
tl.store(..., c.to(tl.float16))
```

参考 `python/examples/test_mm_silu.py`。融合收益 = 省一次全量中间 tensor 的
读写，K3（memory-bound 为主）上非常可观。

### spine_raw 融合

默认 path 的向量原语（`vmin/vmax/abs/select/cast/vexp/vlog/sqrt/rsqrt`）可以
把任意逐元素链塞进一个 kernel：`test_raw_activations.py`（sigmoid/tanh/relu
族）、`test_raw_silu.py`。LLVM-direct path 能进一步用 `llvm.riscv.*` intrinsic
控制 vl/ta/tu 策略，但没有 exp（多项式近似才行，见 basics/05）。

### 什么时候不要融合

decode 场景小数据量下，**Python dispatch overhead 才是大头**（softmax 案例：
22% 开销是 intermediate 创建与类型转换，fusion 消不掉反而叠加自己的 dispatch）。
融合前先 proton/profile 确认目标 kernel 真的占 wall time。

## 验证

```bash
python3 -m pytest python/tests/test_unary_pointwise_ops.py -q
python3 -m pytest python/tests/test_binary_pointwise_ops.py -q
python3 -m pytest python/tests/raw/test_raw_activations.py python/tests/raw/test_raw_silu.py -q
```

golden `torch.nn.functional.silu/gelu`；fp16 rtol/atol 1e-2。
逐元素算子数值错时先查三件事：dtype 提升点、mask 的 `other=` 值
（0 vs -inf 语义不同）、in-place 变体是否踩了非 grid-safe 路径。

下一篇：[08-attention.md](08-attention.md)
