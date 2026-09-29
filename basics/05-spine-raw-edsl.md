# 05 — spine_raw eDSL：直接控制向量指令（进阶）

> **定位**：进阶章。第一遍学习可以直接跳到 [ops/01](../ops/01-gemm.md)，
> 等标准 `tl.*` 满足不了你时再回来。本章假设你已读完 basics/01–04。

## 5.1 为什么还需要“更底层”的一层

标准 `tl.*` 写法下，编译器替你决定一切：怎么向量化、寄存器怎么分配、
归约怎么做。90% 的算子这样就够了。但有三类需求它满足不了：

1. **特殊内存布局**。比如你想“转置着读”一个行主序矩阵（每次取一列），
   标准路径没有对应原语；
2. **超越函数与融合**。`vexp`（向量 exp）这类函数，或“load-multiply-exp-store
   必须融成一条链”的场合；
3. **直接点矩阵引擎的名字**。`vfwmacc` 这类指令级操作，标准路径只能靠
   `tl.dot` 被编译器“猜”着映射过去，猜不中就没了。

`spine_raw` 就是为此存在的：**你仍然写 Python，但它不做向量化决策——
你的每一行直接对应向量级 MLIR 操作**（读一个向量寄存器、乘加、写回）。
可以把它理解成“带 Python 语法的向量汇编”。

## 5.2 它是怎么工作的（60 秒版）

```
你的 Python 函数（@tle.raw_kernel 装饰）
   │  SpineMLIRCodeGenerator：AST visitor，逐语句翻译（没有模板、没有魔法）
   ▼
linalg/memref/vector MLIR 文本（raw_linalg）
   │  经 tle.dsl_region 挂进主 kernel（见 5.6），
   │  spine-opt 的 InlinePass 在 bufferization 前展开
   ▼
之后与标准 kernel 完全同一条流水线（basics/01 §1.4 的 ③④⑤）
```

“逐语句翻译”意味着：Python 里写了什么就是什么，不会有隐式循环、
隐式广播帮你兜底；同时 Python 的很多语法（`if`、函数调用、类）它也不认识——
支持哪些语句是白名单，见 5.7 的 API 表。

## 5.3 导入纪律（第一个必踩的坑）

```python
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call
```

**必须从 `triton.language.extra.spine_raw` 导入。**不要图方便
`sys.path.insert(0, ".../spine-triton/language")` 去加载源码目录——
Python 会把它当成**另一个模块实例**（模块名不同），而 `call()` 的记录挂在
thread-local registry 上，两个实例不共享，**记录会静默丢失**（kernel 编译
“成功”但你的 raw 代码根本没进去，最难查的一类问题）。

同理，改过 `language/spine_raw/` 源码后，安装树下的副本
（`triton/language/extra/spine_raw/`）也要同步（cp 或重编），否则运行时
用的还是旧代码。

## 5.4 核心心智模型：vl、向量寄存器、mem 参数

读例子之前先建立四个概念：

**① `tle.mem(dtype)` — 内存参数。**raw kernel 的 tensor 参数用
`X: tle.mem(f16)` 标注，语义是一块连续内存（memref）。`out=True` 标只写
参数（帮助优化）。标量尺寸参数用 `N: tle.index`（MLIR 的 index 类型）。
类型对不上时（比如主 kernel 传 i32 而 raw 期望 index）InlinePass 会自动插
`arith.index_cast`，一般不用你操心。

**② `tle.vconfig(vl, ...)` — 一次处理多少个元素。**K3 的向量寄存器
vlen=1024 bit：f32 装 32 个、f16 装 64 个。`tle.vconfig(-1, 1)` 表示
“取硬件最大值（VLMAX）”，并返回这个数——所以
`nvl = tle.vconfig(-1, 1)` 之后 `nvl` 就是本 dtype 下一个寄存器的元素数。
处理“尾巴”（不足一个寄存器的部分）时用 `tle.vconfig(N - i, 1)` 收窄。

**③ 循环必须自己写。**标准 `tl.*` 里 `tl.arange(0, 4096)` 会由编译器
自动拆成向量指令；raw 层没有这种自动拆分——**你自己写
`for i in tle.range(0, N, nvl)`，每次迭代显式 load/store 一个向量**。

**④ 没有 `if`。**默认路径不支持条件语句。尾巴处理怎么办？写**第二个
range 循环**：`for i in tle.range(Nfloor, N, nvl)`——当 `Nfloor == N`
（整除）时这个 range 是空的，循环体自动一次都不执行。这是 raw 层的惯用法，
下面例子里会出现 3 次。

## 5.5 完整例子：layernorm（逐段精讲）

来源：`python/tests/raw/test_raw_layernorm.py`（真实测试，可直接跑）。
任务：对一维 f16 输入做 `out = (x - mean) / sqrt(var + eps)`，输出 f32。

先看数据流——layernorm 需要全量的 mean 和 var，所以必须**三趟**扫数据：

```
趟1: 扫一遍 x，累加 sum(x)          → mean = sum / N
趟2: 再扫一遍 x，累加 sum(x²)       → var = sum(x²)/N − mean²
趟3: 第三遍，逐元素 (x−mean)*rsqrt(var+eps) 写出
```

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
    nvl = tle.vconfig(-1, 1)          # 向量寄存器容量（f32 累加器下 = 32）
    Nfloor = (N // nvl) * nvl         # N 向下取整到 nvl 的倍数

    # ── 趟1: sum(x)，f32 累加 ────────────────────────────────
    acc1 = tle.vzero(f32)             # 全 0 的 f32 向量寄存器
    for i in tle.range(0, Nfloor, nvl):
        acc1 = acc1 + tle.cast(tle.vload(X, i), f32)   # 读 f16 向量→升 f32→累加
    for i in tle.range(Nfloor, N, nvl):                # 尾巴（整除时为空循环）
        tle.vconfig(N - i, 1)                          # 收窄 vl 到剩余元素数
        acc1 = acc1 + tle.cast(tle.vload(X, i), f32)
    mean = tle.vreduce_sum(acc1) / N  # 向量水平求和 → f32 标量

    # ── 趟2: sum(x²) → var ──────────────────────────────────
    acc2 = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vb = tle.cast(tle.vload(X, i), f32)
        acc2 = acc2 + vb * vb
    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        tb = tle.cast(tle.vload(X, i), f32)
        acc2 = acc2 + tb * tb
    var = tle.vreduce_sum(acc2) / N - mean * mean
    scale = tle.rsqrt(var + EPS)      # 1/sqrt(var+eps)，标量

    # ── 趟3: 归一化写出 ─────────────────────────────────────
    for i in tle.range(0, Nfloor, nvl):
        vc = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, (vc - mean) * scale)        # 标量自动广播到向量
    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        tle.vstore(out, i, (tle.cast(tle.vload(X, i), f32) - mean) * scale)


@triton.jit
def layernorm_1d_host(X, out, N):
    # 在 @triton.jit 函数体内把 raw kernel“挂载”进来（机制见 5.6）
    _sr_call(layernorm_1d_kernel, outputs=[], inputs=[X, out, N])


if __name__ == "__main__":
    torch.manual_seed(0)
    N = 256
    X = torch.randn(N, dtype=torch.float16)
    out = torch.zeros(N, dtype=torch.float32)
    layernorm_1d_host[(1,)](X, out, N)          # grid=(1,)：默认路径没有 program_id

    ref = torch.nn.functional.layer_norm(X.float(), (N,), eps=EPS)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASS, max diff =", (out - ref).abs().max().item())
```

逐段解释关键决策：

- **为什么每趟 load 后立刻 `tle.cast(..., f32)`**：dtype 军规第一条
  （f16 输入 f32 累加）。`acc1` 是 f32 向量寄存器，长归约下 f16 累加误差
  会滚雪球。
- **为什么趟2 用 `E[x²] − mean²` 而不是 `E[(x−mean)²]`**：后者需要第三趟
  才能算（mean 要全量扫完才知道）——这里已经是三趟了，用方差恒等式可以让
  趟2 与 mean 无关地并行推进。数值上对病态分布稍差，工程上是标准取舍。
  （另一个动机：尾巴 lane 收窄后残留的 padding 0 如果参与 `(0−mean)²`
  会虚增方差；`x²` 形式下 0 的贡献就是 0。）
- **`tle.vreduce_sum(acc1)`**：把一个向量寄存器**水平**求和成 f32 标量
  （对应 RVV 的 vredsum 指令族）。注意三趟里累加都是**向量级**的
  （lane 各加各的），只有最后一步才水平归约——这是向量归约的标准形态。
- **`(vc - mean) * scale`**：`mean`/`scale` 是标量，与向量运算时自动广播。
- **`grid=(1,)`**：默认路径没有 `program_id`，一个 kernel 只能一个 program
  串行跑完。要按行/按头并行，要么在外层 Python 循环多次 launch，要么用
  LLVM-direct 路径（5.7）。

## 5.6 `call()` 是怎么把 raw kernel 挂进主 kernel 的

`_sr_call(raw_fn, outputs=[], inputs=[...])` 写在 `@triton.jit` 函数体内。
它的设计意图：**inputs 里可以放 JIT 期才计算出来的值**（比如
`pid * stride`），这是 raw kernel 获得“我是第几号 program”信息的通道。

机制（了解即可，排错时有用）：

```
@triton.jit tracing 时
   call() → emit 一个 tle.dsl_region op
            （raw_linalg 属性里内嵌 raw kernel 翻译出的完整 linalg 文本）
   ▼
spine-triton-opt 的 C++ DSLRegionOpPattern
   parse raw_linalg → 建 spine_ext.raw_region op
   （operand 沿 ptr.to_ptr/memref.reinterpret_cast 链回溯到真实 memref）
   ▼
spine-opt 的 SpineRawRegionInlinePass
   在 bufferization 之前把 raw_region 原地展开进主函数
```

所以 raw kernel 里的错误（非法类型、不支持的语句）会在**三个不同阶段**
报出来：tracing 期（Python 异常）、DSLRegionOpPattern（parse 失败）、
InlinePass（类型不匹配）。看到报错先判断在哪一站，再对症处理。

## 5.7 API 地图

```
类型/注解   mem(dtype, out=)  index  f16  f32  bf16  In  InOut
装饰器      raw_kernel（= spine_raw(name="linalg")，默认路径）
            spine_raw（可指定 LLVM-direct 等路径）
挂载        call(fn, outputs=[...], inputs=[...])     proton_mark
─────────── 默认（linalg）路径 ───────────
向量        vconfig(vl)     设/取向量长度（见 5.4②）
            vzero(dtype)    全 0 向量寄存器
            vload(mem, off) / vstore(mem, off, v)     向量读/写
            vmacc           乘加
            vreduce_sum/max/min/mul                   水平归约 → 标量
            vmin vmax abs sqrt rsqrt vexp vlog        逐元素数学
            cast(v, dtype)  类型转换
            select viota vbroadcast spread vshape pack vpack alloc
标量        sload / sstore
矩阵        vmadot          矩阵引擎点积
            mma_cube(dtype) 返回该 dtype 在当前 arch 的合法 MICRO tile
控制流      tle.range       （唯一的控制流；没有 if！）
─────────── 2D / 矩阵引擎原语 ───────────
            batch_macc      acc[m,n] += Σ_k lhs[m,k]·rhs[k,n]（vfwmacc）
            view_2d load_2d[_at] load_2d_t（转置读！）pack_2d_t
            splat_2d store_2d[_at] matmul extract_elem
─────────── LLVM-direct 路径 ───────────
            program_id      （仅此路径有！可 per-head/per-row 并行）
            llvm_const llvm_poison llvm_base_ptr llvm_gep llvm_size
            call_intrinsic  （直发 llvm.riscv.* RVV intrinsic）
```

两条路径怎么选：

|          | 默认 linalg 路径                                                                  | LLVM-direct 路径                                                       |
| -------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 产出     | `vector.transfer_read/write` + `math.fma`（可被自动转成 RVV scalable vector） | LLVM dialect 直发（`llvm.fadd/load/store`、`llvm.riscv.*`）        |
| 并行     | **无 `program_id`** → grid=(1,) 串行                                     | **有 `program_id`** → per-head/per-row 并行                   |
| 超越函数 | 有`vexp/vlog/sqrt/rsqrt`                                                        | **没有**（LLVM 23 移除了 math intrinsic；要 exp 只能多项式近似） |
| 控制流   | 无 if；tail 用空 range 循环                                                       | `llvm.fcmp` + select 手工拼                                          |
| 适合     | 单 program 串行、需要 exp/log 的融合 kernel                                       | 需要并行度的 attention/GEMV 类 kernel                                  |

**经验法则：能用默认路径就用（有 vexp、好写）；只有确实需要 program 级
并行度时才上 LLVM-direct。**ops/08 的 attention kernel 是 LLVM-direct 的
完整实战（per-head grid）。

## 5.8 坑清单（每一条都真实炸过）

1. **导入路径错**（5.3）：`call()` 记录静默丢失，kernel “编译成功”但 raw
   代码没进去。症状：数值完全不对但无任何报错。
2. **默认路径用了 `program_id` / `if` / `icmp` / `select`**：不支持，
   前端直接报 AttributeError 或未知节点。需要它们只能换 LLVM-direct。
3. **`vconfig` 两种写法都对但语义有别**：裸语句 `tle.vconfig(N-i, 1)`
   只改 vl 不绑名字；赋值形式 `nvl = tle.vconfig(-1, 1)` 会把 `nvl` 绑成
   编译期常量（能参与 `Nfloor` 的整除计算）。
4. **多趟归约崩 `operand #N does not dominate this use`**：历史版本
   loop body 的局部变量（如 `v = tle.vload(...)`）会泄露到 loop 外，被
   下一个循环误当 iter_arg。已修复（loop 退出时恢复 pre-loop env）；若你在
   旧部署副本上见到此错，先同步两份副本（5.3）。
5. **raw kernel 里 emit `vector_ext` 系 op 必须用 generic form**：
   写 `"vector_ext.batch_macc"(%lhs, %rhs, %acc) : (...) -> ...` 这种
   引号形式，不能用真 assembly 语法——raw_linalg 文本要先经过
   spine-triton-opt（它**没注册** vector_ext dialect）parse，真语法直接
   parse 失败；generic form 任何工具都能过，最后由 spine-opt（注册了）
   正常 lower。
6. **LLVM-direct 的负数常量必须传字符串**：`tle.llvm_const("-1.0e38", "f32")`。
   写 `-1.0e38` 会被 Python AST 解析成 UnaryOp（负号是运算），codegen 报
   `'UnaryOp' object has no attribute 'value'`。
7. **`vload` 的 fill 类型**：mask/fill 值与目标 dtype 不一致时（如 f16 load
   填 -1e38 的 f32 字面量）linalg.fill 报类型不匹配——现版本已自动 truncf
   修复；见到 `expected fill value type ('f32') to match output element type ('f16')` 说明部署副本旧了。
8. **改了 spine_raw 源码不生效**：两份副本没同步，或者**没换
   TRITON_CACHE_DIR**（cache key 不含 language 模块版本，旧 .so 直接复用，
   basics/04 §4.2）。
9. **测试函数的 host 函数名要唯一**：`call()` 的 AST path 按函数名触发，
   同名函数会互相干扰。

## 5.9 mixed 模式（知道有这东西即可）

一个 raw kernel 可以携带 sibling LLVM 函数（host 侧辅助例程，比如三段式
GEMV 的中间步骤），编译后期经 mixed bridge（用 MLIR Python bindings 做的
结构化注入）合并进最终 `ll.mlir`。
参考 `python/tests/raw/test_mixed_syntax_three_layer.py`、
`test_raw_mv_three_stage.py`。初学阶段用不到，遇到 `__spine_bridge_pt`
字样知道是它就行。

## 5.10 动手练习

1. 把 5.5 的 layernorm 跑通（`N=256` 与 `N=300` 各一次——后者验证尾巴
   循环；注意 300 不是 32 的倍数）。
2. 把 `N=256` 改成 `N=2048`，确认 `do_not_specialize` 不加会怎样
   （提示：`N` 是普通参数，值特化会把它折成常量；对比编译产物）。
3. 用 `SPINE_TRITON_DUMP_PATH=./dumps` 跑一次，在 dump 里找到
   `vector.transfer_read`、`math.fma`（或 `arith`/`linalg` 对应 op）——
   确认你写的每一行真的变成了向量级操作。
4. 写一个 raw 版 `out = relu(x)`（`vmax(v, vzero)` 即可），f16 进 f16 出，
   与 `torch.relu` 对照。

## 5.11 进一步阅读

- 向量级算子范例库：`python/tests/raw/test_raw_{softmax,silu,cumsum,argmax,group_norm,...}.py`
- 矩阵引擎范例：`test_raw_mv_cbm.py`、`test_raw_mm_cbm.py`（配合 ops/05-gemv.md）
- LLVM-direct attention 实战：ops/08-attention.md
- 全套 raw 测试基线：267 用例 261P/1F/1xF/4err（4 个 error 是诊断脚本的
  pytest 签名问题，跑套件时要 `--ignore`）

下一篇：[06-smt-matrix-engine.md](06-smt-matrix-engine.md) —— 矩阵引擎与 smt 扩展。
