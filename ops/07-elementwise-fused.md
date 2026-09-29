# Op 07 — Elementwise 与融合（add / silu / SwiGLU / gelu）

**重要性：★★★☆☆**。逐元素算子单看没有技术含量——`y[i] = f(x[i])`，
无归约、无通信。但它们 (a) 数量最多（FlagGems 测试套里 pointwise 文件
占大头），(b) 是 **fusion 的载体**：LLM 里真正的性能收益来自把逐元素链
融进相邻的大算子（bias+激活融进 GEMM epilogue、silu·mul 融成一个
SwiGLU），(c) 把 math/libdevice 函数链路的坑全踩了一遍。本章以
add → silu → SwiGLU → gelu 递进。

## 7.1 这个算子在算什么，为什么“融合”才是正题

三个主角的数学：

```
silu(x)    = x · sigmoid(x) = x / (1 + e^(-x))
SwiGLU(g,u)= silu(g) · u                    # LLM MLP 的核心
gelu(x)    ≈ 0.5x(1 + tanh(0.79788456·(x + 0.044715x³)))   # tanh 近似版
```

手算：`silu(1) = 1/(1+e^-1) ≈ 0.731`；`silu(0) = 0`；`silu(-100) ≈ 0`
（负半轴被 sigmoid 压灭）。Qwen3 的 MLP 就是
`down_proj( silu(gate_proj(x)) · up_proj(x) )`——gate 和 up 是两个并行的
GEMM，中间的 silu·mul 是纯逐元素。

**为什么逐元素算子是 memory-bound 的极端**：`y = silu(x)` 读 N 个元素、
写 N 个元素、每个元素只算几次乘加。若 silu、mul 分开成两个 kernel：

```
不融合:  读 g,u → 写 t=silu(g) → 读 t,u → 写 y     # 6 趟 N 元素访存
融合:    读 g,u → 写 y                              # 3 趟
```

访存直接减半。K3 上算力远富余于带宽，**融合省下的每一趟访存都是墙钟**。
这就是“逐元素算子不值得单独优化，但值得拼命融合”的原因。

## 7.2 完整可运行程序：SwiGLU

（vec_add 的骨架在 basics/03 §3.4 已精讲过，这里直接上有业务意义的
二元融合版。）

```python
# swiglu.py — y = silu(g) * u（LLM MLP 中间层）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["N"])
def swiglu_kernel(G, U, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # 军规：进 math 函数（sigmoid→exp）前升 f32
    g = tl.load(G + offs, mask=mask, other=0.).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.).to(tl.float32)

    y = (g * tl.sigmoid(g)) * u          # silu(g) * u，全程 f32

    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def swiglu(g, u):
    N = g.numel()
    y = torch.empty_like(g)
    BLOCK = 1024
    swiglu_kernel[(triton.cdiv(N, BLOCK),)](g, u, y, N, BLOCK=BLOCK)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    N = 3072 * 1024                       # Qwen3 MLP 中间维量级
    g = torch.randn(N, dtype=torch.float16)
    u = torch.randn(N, dtype=torch.float16)

    y = swiglu(g, u)
    y_ref = (torch.nn.functional.silu(g.float()) * u.float()).half()
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("PASS  max diff =", (y - y_ref).abs().max().item())
```

```bash
python swiglu.py
# PASS  max diff = 0.00xx
```

### 逐段精讲

- **逐元素 kernel 的 mask 语义与归约不同**：这里 `other=0.` 会参与计算
  （silu(0)·0=0），但**结果被 store 的 mask 挡住了，写不出去**——越界
  lane 算什么值都无所谓。归约 kernel 里填充值会被吸收进结果（单位元
  框架，ops/06 §6.1），逐元素不会。这是两类算子在 mask 上的本质区别。
- **`.to(tl.float32)` 在 sigmoid 之前**：`tl.sigmoid` 内部走 exp，只收
  fp32/fp64（`math.py:_check_dtype`，上游故意设计，GPU 同败）。f16 直接
  喂 = CompilationError。这个检查是 K3 逐元素算子的第一大坑（7.4）。
- **一维扁平化**：`N = g.numel()`，不关心原始 shape——逐元素算子对
  layout 无感知，扁平处理最简单也最快（连续访存）。
- **grid = cdiv(3M, 1024) = 3072**：新版运行时无压力；旧版 wheel 对超大
  grid 有上限，超限会静默出错（basics/04 §4.5）——升级 wheel，或把 BLOCK
  调大 / 加 kernel 内循环（7.5 的 pointwise_dynamic 模式）。

## 7.3 epilogue 融合：逐元素链挂到 GEMM 尾巴上

更进一步：SwiGLU 的两个输入来自 gate/up 两个 GEMM。生产优化是把
“GEMM + bias + 激活”融成一个 kernel——K-loop 结束后，直接在 **f32
累加器**上做 epilogue，中间结果不落内存：

```python
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, K, BK):
    acc += tl.dot(a, b)                  # ops/01 的 K-loop
# ── epilogue：全在寄存器里的 f32 acc 上做 ──
c = acc + tl.load(BIAS + offs_n)[None, :]     # + bias（广播到每行）
c = c * tl.sigmoid(c)                         # silu
tl.store(C_ptrs, c.to(tl.float16), mask=...)  # 只在最后 cast + 写出
```

完整可跑参考：`python/examples/test_mm_silu.py`（mm+silu 融合）、
`test_gelu_proton.py`（gelu + proton 计时）。两个要点：

1. epilogue 运算**天然全程 f32**（acc 是 f32），dtype 军规自动满足；
2. bias 用 `[None, :]` 广播——(BN,) 向量加到 (BM, BN) 块的每一行，
   与 ops/01 K-loop 的下标广播同款语法。

融合的账（BM=BN=32、N=3072 量级）：省掉的是一个 (M, N) f16 中间 tensor
的完整读写——对 memory-bound 的 K3，这通常比 epilogue 里多算的几次
sigmoid 贵几个数量级。**“中间结果不落内存”是融合的全部要义。**

## 7.4 math / libdevice 链路：K3 的坑位图

逐元素算子的实现一半在 math 函数上。K3 工具链的 math 链路分层：

```
tl.sigmoid/tanh/... ─► triton/language/math.py（_check_dtype: 只收 fp32/fp64）
libdevice.acos/...  ─► language/cpu/libdevice.py（@core.extern → extern_elementwise
                       → {(dtype,): ("math.X", dtype)} 映射到 MLIR math dialect）
                       ─► 编译器白名单（哪些 math 函数被支持）
                       ─► 后端降级（math → LLVM）→ 代码生成
```

| 症状 | 层 | 根因与处置 |
|---|---|---|
| `Expected dtype ['fp32','fp64'] but got fp16` | 前端 | `_check_dtype`，故意设计；kernel 里先 `.to(tl.float32)`。**别想着改 math.py 全局提升**（上游 FlagGems 曾两次尝试修复又都 revert——影响面不可控） |
| `'constexpr_type' object has no attribute 'scalar'` | 前端 | JIT 内字面量 `pow(x, 2)` 的 `2` 是 constexpr；直调类型提升函数前须 `isinstance(x, core.constexpr)` 解包 `.value`（gelu_backward 24 例编译崩的根因） |
| `Dialect 'math' not found` | 编译器后端 | math op 没被降级到底。可 lower 清单：acos/asin/atan/cos/sin/tan/cosh/sinh/tanh/exp2/expm1/log2/log10/log1p/cttz；**acosh/asinh/atanh/cbrt 残留不 lower**（当前无算子用到；新 kernel 避开或先确认） |
| 新 libdevice 函数“不存在” | 前端 shim / 后端白名单 | cpu shim 必须是 `extern_elementwise` 形态（历史上 15 个 shim 调不存在的函数，已修）；后端白名单须同步扩（`linalg.rint`→`math::RoundEvenOp` 就是这么修的；rint=round-half-to-**even**，不能映射 math.round） |
| `ffs` 语义不对 | shim | CUDA 语义 1-based、0 返回 0，用 `math.cttz`+select 实现；op 注册名是 `math.cttz/ctlz`，**没有 "math.ctz"** |

审计方法（接手一批逐元素算子时）：全量 grep `tl.exp|tl.log|tl.sqrt|
libdevice.` 调用点，确认每处都有前置 cast / f32 标量提升 / COMPUTE_FP32
门控之一（FlagGems 约 40 个调用点审计的结论：有这四类保护的都安全）。

## 7.5 grid 安全性：pointwise_dynamic vs 手写平坦 kernel

FlagGems 里大部分二元逐元素算子走 **pointwise_dynamic** codegen：
`num_ctas = min(max_grid, num_tiles)` + kernel 内循环处理多 tile——
**天然 grid-safe**（grid 永远不超上限，不受旧版 wheel 的 grid 限制影响）。
但注意它的 **in-place / out 变体走的是手写平坦 kernel**，不享受该保护——
历史上 le/maximum/where/clamp 的 in-place 变体失败而 out-of-place 通过
就是这个差别。自己写逐元素 kernel 时，两种保证 grid 安全的方式：

1. host 算 grid 时封顶 + kernel 内 `for` 循环吃掉剩余（pointwise_dynamic
   模式）；
2. 确认 wheel 版本较新（运行时会对超大 grid 自动分块），直接放开 grid。

## 7.6 spine_raw 逐元素

默认 path 的向量原语（`+ - * / %`、一元负号、比较→mask、
`vmin/vmax/sqrt/rsqrt/abs/select/cast/vexp/vlog`）可以把任意逐元素链塞进
一个 kernel：`test_raw_activations.py`（sigmoid/tanh/relu 族）、
`test_raw_silu.py`。

`test_raw_elementwise.py` 有个值得学的**测试设计**：每个 §6.4 原语一个
kernel，算完不直接写回向量，而是 `vreduce_sum` 成标量再 `sstore`——
因为“全向量 vstore”与“标量 store”是两条不同成熟度的 lowering 路径，
用已知可靠的归约-标量路径收尾，能把测试失败**隔离到被测的逐元素运算
本身**。验证新原语时值得抄这个模式。

LLVM-direct path 能进一步用 `llvm.riscv.*` intrinsic 控制 vl/ta/tu，但
**没有 exp/log/sqrt**（LLVM 23 移除 math intrinsics，需多项式近似——
basics/05 §5.7 的两路径对照表）。

## 7.7 什么时候不要融合

反面教材（ops/04 §4.6 详述）：decode 场景小数据量下 **Python dispatch
overhead 才是大头**——softmax 案例里 22% 的开销是 intermediate 创建与
类型转换，kernel fusion 消不掉 dispatch 本身，反而叠加自己的 launch。
QKT+softmax+AV 全融合被实测否决（目标只占 prefill 墙钟 0.8%）。

**纪律：融合前先 profile**（proton host-region 计时，basics/04 §4.3），
确认目标 kernel 真的占 wall time。融合的收益 = 省掉的访存趟数 × 数据量
÷ 带宽——数据量小的时候这笔账可能是负的。

## 7.8 验证

```bash
python swiglu.py                                              # 本章程序
python3 python/examples/test_vec_add.py                       # 最小逐元素
python3 python/examples/test_mm_silu.py                       # epilogue 融合
python3 -m pytest python/tests/test_unary_pointwise_ops.py -q
python3 -m pytest python/tests/test_binary_pointwise_ops.py -q
python3 -m pytest python/tests/raw/test_raw_activations.py python/tests/raw/test_raw_silu.py -q
```

golden：`torch.nn.functional.silu/gelu`（f16 输入 `.float()` 后算再降回）；
rtol/atol 1e-2。逐元素数值错时先查三件事：dtype 提升点（7.4 表）、
mask 的 `other=` 是否漏进了归约（本章没有归约，但你融合的对象可能有）、
in-place 变体是否踩了非 grid-safe 路径（7.5）。

## 7.9 练习

1. 跑通 7.2。把 `.to(tl.float32)` 删掉重跑，抄下 `_check_dtype` 报错原文
   （与 ops/04 练习 2 同款——这个签名要形成条件反射）。
2. 把 kernel 改成三元 `y = silu(g) * u + bias`（bias 是 (N,) 向量）——
   体会逐元素链加长对代码结构零影响（还是那 3 趟访存）。
3. 跑 `test_mm_silu.py`，然后 dump IR（`SPINE_TRITON_DUMP_PATH`）确认
   sigmoid 变成了 exp 相关 math op；数一数 `.linalgdir` 里
   `transfer_write` 的次数，对照“融合省一趟写”的账。
4. 用 gelu tanh 近似公式（7.1）实现 `gelu_kernel`，golden
   `torch.nn.functional.gelu(x, approximate='tanh')`。注意 `x*x*x` 与
   常量乘法都是 f32 标量/tensor 运算，无 dtype 坑；但 `tl.math.tanh`
   的输入必须已是 f32。
5. 思考题：为什么 epilogue 融合里 bias 加法要放在 f32 acc 上、cast 放
   最后？如果把 `acc.to(f16)` 提前到加 bias 之前，什么场景会出问题？
   （提示：ops/01 §1.7 的 f16 截断契约。）

下一篇：[08-attention.md](08-attention.md) —— 把所有积木拼起来。
