# 05 — spine_raw eDSL：向量级 kernel

标准 `tl.*` 之下还有一层：`triton.language.extra.spine_raw`（惯例别名 `tle`）。
它是一个 Python eDSL——`SpineMLIRCodeGenerator` 走 AST visitor 一条路，把你的
Python 函数逐语句翻译成 linalg/memref/vector MLIR，没有模板机制。适合标准路径
表达不了的融合、超越函数、特殊布局、矩阵引擎 intrinsic。

## 导入纪律

```python
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call
```

**必须从 `triton.language.extra.spine_raw` 导入**，不能 `sys.path.insert` 直接
加载源码 `language/` 目录——两者是不同模块实例，thread-local registry 不共享，
`call()` 的记录会静默丢失。

## 两条 codegen 路径

| | 默认 linalg path | LLVM-direct path |
|---|---|---|
| 产出 | `vector.transfer_read/write` + `math.fma` 等（可被 ConvertToScalableVector 转成 RVV） | LLVM dialect 直发（`llvm.fadd/fmul/load/store/getelementptr`、`llvm.riscv.*` RVV intrinsic） |
| 并行 | **无 `tle.program_id`**，grid=(1,) 单 program + 外层串行循环 | 有 `tle.program_id`，可 per-head/per-row 并行 |
| 超越函数 | 有 `tle.vexp/vlog/sqrt/rsqrt` | **没有** exp/log/sqrt（LLVM 23 移除 math intrinsic；需多项式近似） |
| 控制流 | 无 `if`；tail 用第二个空 range loop | `llvm.fcmp` + select 手工拼 |

选择原则：能用默认 path 就用（有 vexp、写法简单）；需要 program_id 并行度
才上 LLVM-direct。

## 最小完整例子：layernorm

（来源：`python/tests/raw/test_raw_layernorm.py`）

```python
import torch, triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16, f32 = tle.f16, tle.f32
EPS = 1e-5

@tle.raw_kernel
def layernorm_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)          # -1 = 取 VLMAX；返回本配置下的向量长度
    Nfloor = (N // nvl) * nvl

    # 趟1: sum(x) → mean（f32 累加）
    acc1 = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        acc1 = acc1 + tle.cast(tle.vload(X, i), f32)
    for i in tle.range(Nfloor, N, nvl):        # tail：range 为空自动跳过（无 if 可用）
        tle.vconfig(N - i, 1)                  # 收窄 vl 处理尾巴
        acc1 = acc1 + tle.cast(tle.vload(X, i), f32)
    mean = tle.vreduce_sum(acc1) / N           # f32 标量

    # 趟2: E[x²] - mean² → var（避免 padding 0 的 (0-mean)² 膨胀）
    acc2 = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vb = tle.cast(tle.vload(X, i), f32)
        acc2 = acc2 + vb * vb
    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        tb = tle.cast(tle.vload(X, i), f32)
        acc2 = acc2 + tb * tb
    var = tle.vreduce_sum(acc2) / N - mean * mean
    scale = tle.rsqrt(var + EPS)

    # 趟3: (x - mean) * scale 回写
    for i in tle.range(0, Nfloor, nvl):
        vc = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, (vc - mean) * scale)
    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        tle.vstore(out, i, (tle.cast(tle.vload(X, i), f32) - mean) * scale)

@triton.jit
def layernorm_1d_host(X, out, N):
    _sr_call(layernorm_1d_kernel, outputs=[], inputs=[X, out, N])

# launch：grid=(1,)，host 是普通 @triton.jit 函数
X = torch.randn(256, dtype=torch.float16)
out = torch.zeros(256, dtype=torch.float32)
layernorm_1d_host[(1,)](X, out, 256)
```

要点：

- `tle.mem(f16)` 标注指针参数（memref 语义），`out=True` 标只写；`tle.index`
  是 index 类型标量。类型不匹配时 InlinePass 自动插 `arith.index_cast`。
- `tle.vconfig(-1, 1)` 设 vl；tail 处理**不能用 if**，写第二个
  `tle.range(Nfloor, N, nvl)` 循环（Nfloor==N 时 range 为空自动跳过）。
- `_sr_call(raw_fn, outputs=[], inputs=[...])` 在 `@triton.jit` 体内把 raw
  kernel 挂进主 kernel：JIT tracing 时 emit `tle.dsl_region`（raw_linalg attr
  内嵌完整 linalg 文本），C++ `DSLRegionOpPattern` parse 后建
  `spine_ext.raw_region`，spine-opt 的 InlinePass 在 bufferization 前展开。
  这样 `pid * stride` 等 JIT 期计算值可以作为 operand 传进 raw kernel。

## API 概览

```
类型/注解   mem(dtype, out=)  index  f16  f32  bf16  In  InOut
装饰器      raw_kernel (= spine_raw(name="linalg"))  spine_raw
挂载        call(fn, outputs=[...], inputs=[...])     proton_mark
向量        vconfig  vzero  vload  vstore  vmacc  vpack  pack  alloc
            vreduce_sum/max/min/mul  vmin  vmax  vshape  vbroadcast  spread
            viota  cast  select  abs  sqrt  rsqrt  vexp  vlog
标量        sload  sstore
矩阵        vmadot  mma_cube(dtype) → 该 dtype 的合法 MICRO tile
2D/mmt      batch_macc  view_2d  load_2d[_at/_t]  pack_2d_t  splat_2d
            store_2d[_at]  matmul  extract_elem
LLVM-direct llvm_const  llvm_poison  llvm_base_ptr  llvm_gep  llvm_size
            call_intrinsic  program_id（仅此 path）
控制流      tle.range（两条 path 都有）
```

## 坑清单（全部实战踩过）

1. **默认 path 无 program_id / if / icmp / select**——需要 program 级并行只能
   LLVM-direct；default path kernel 必须 grid=(1,)。
2. **裸语句 `tle.vconfig(K-i, 1)` 与赋值形式 `nvl = tle.vconfig(-1, 1)` 都合法**，
   赋值形式会把 nvl 绑成编译期常量参与 `Nfloor` 计算。
3. **loop body 局部变量不泄露**（历史 bug 已修）：曾经 loop 内 `v = tle.vload(...)`
   泄露到 loop 外导致 MLIR dominance violation；多趟 reduce（layernorm/softmax）
   必崩。现版本安全，但看到 `operand #N does not dominate this use` 先怀疑此类
   问题并确认部署副本已同步。
4. **raw kernel 里产 `vector_ext` op 必须用 generic form**
   （`"vector_ext.batch_macc"(...)` 而非 assembly 语法）：raw_linalg 文本先经
   spine-triton-opt（未注册 vector_ext dialect）parse，真语法会失败。
5. **LLVM-direct 没有 exp/log/sqrt**：需要时用默认 path 的 `vexp`，或多项式
   近似（range reduction + 4-term polynomial）。负数 f32 常量必须字符串：
   `tle.llvm_const("-1.0e38", "f32")`（Python AST 会把 `-1e38` 解析成 UnaryOp）。
6. **两份副本要同步**：源码 `spine-triton/language/spine_raw/` 与安装树
   `triton/language/extra/spine_raw/`；改完 codegen/builtins 后重编或直接 cp，
   否则新原语在运行时不可见。
7. **`call()` 设计意图是在 `@triton.jit` 体内调用**（传 JIT 期计算值）；
   测试时 host 函数名需唯一以触发 AST path。

## mixed 模式

一个 raw kernel 可以带 sibling LLVM 函数（host 侧辅助例程），经 mixed bridge
（MLIR Python bindings 实现的结构化注入）合并进最终 ll.mlir。参考
`python/tests/raw/test_mixed_syntax_three_layer.py`、`test_raw_mv_three_stage.py`。

## 进一步阅读

- 向量级算子范例：`python/tests/raw/test_raw_{softmax,silu,cumsum,argmax,group_norm,...}.py`
- 矩阵引擎范例：`test_raw_mv_cbm.py`、`test_raw_mm_cbm.py`（配合 ops/05-gemv.md）
- LLVM-direct attention：ops/08-attention.md

下一篇：[06-smt-matrix-engine.md](06-smt-matrix-engine.md)
