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
   │  编译器后端在降级早期把它原地展开
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
类型对不上时（比如主 kernel 传 i32 而 raw 期望 index）编译器会自动插
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
    layernorm_1d_host[(1,)](X, out, N)          # grid=(1,)：整个 1D 向量本就是一个 program 的活

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
- **`grid=(1,)`**：这里 grid=(1,) 是**任务性质的选择，不是 raw 的上限**
  ——整条 1D 向量的 layernorm 本来就是一个 program 的活。raw kernel 内部
  确实没有 `program_id`，但多 program 并行完全做得到：见下面的并行形态。

### grid=(1,) 不是上限：并行形态（host pid 传参，重要必读）

`_sr_call` 的 inputs 里可以放**在 host `@triton.jit` kernel 里算出来的
标量**（5.6 会讲这正是它的设计意图）——`tl.program_id(0)` 就是这样的值。
于是默认路径的标准并行形态是：

> **host tl kernel 持有真正的 grid，把 `tl.program_id(0)`（或由它算出的
> 分段范围）当普通标量参数传给 raw kernel；raw kernel 只处理自己拿到的
> 那一段。** 并行语义全部在 tl 层表达，raw 层保持"无 program_id、无 if"。

真实测试 `test_raw_group_norm.py`（按行做 layernorm，每行一个 program）：

```python
@tle.raw_kernel
def group_norm_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True),
                      G: tle.index, C: tle.index, row: tle.index):
    ...                          # 三趟扫第 row 行：偏移全是 row * C + i

@triton.jit
def group_norm_host(X, out, G, C):
    row = tl.program_id(0)       # ← 并行度在这一层
    if row < G:                  # tl 层的标量 if 是合法的（raw 层才没有 if）
        _sr_call(group_norm_kernel, outputs=[], inputs=[X, out, G, C, row])

group_norm_host[(G,)](X, out, G, C)      # grid=(G,)：每行一个 program
```

三个要点：

- raw kernel 不知道也不需要知道自己是第几号 program——它只收到一个
  `row`。上面 5.5 例子里 grid=(1,) 之所以成立，是因为那个任务本来只有
  一份工作。
- 边界守卫（`if row < G:`）写在 host tl kernel 里；grid 大小的注意项与
  普通 tl kernel 完全一样（basics/04 §4.5）。
- 这个模式在全书反复出现：ops/05 §5.4 的 GEMV（pid → row_base/row_end，
  每 program 算 BLOCK 行）、ops/06 §6.4 的行归约（row_idx）、ops/09 §9.4
  的三阶段并行 cumsum（Phase1/3 的 host 都是 `p = tl.program_id(0)` +
  `if p < P:` + 把 p 传进 raw）。**以后见到 grid=(1,) 的 raw 例子先想：
  是这个任务天然串行（整段 scan、全量归约），还是作者没写 host grid。**

只有在 raw kernel **内部**直接要 `program_id`（不想经过 tl host）时才需要
LLVM-direct 路径（5.7）——host pid 传参形态对绝大多数场景已经够用，而且
保留默认路径的全套向量原语（vexp/vlog/rsqrt…）。

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
中层转换工具（wheel 自带的 spine-triton-opt）
   parse raw_linalg → 建一个专用的 raw_region op
   （operand 沿指针 cast 链回溯到真实的 memref）
   ▼
编译器后端
   在降级早期把 raw_region 原地展开进主函数
```

所以 raw kernel 里的错误（非法类型、不支持的语句）会在**三个不同阶段**
报出来：tracing 期（Python 异常）、中层转换（parse 失败）、后端内联
（类型不匹配）。看到报错先判断在哪一站，再对症处理。

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
| 并行     | kernel 内无 `program_id`；**host tl kernel 的 pid 经 `_sr_call` 传标量** → 任意 grid（5.5 并行形态） | kernel 内**有 `program_id`** → 直接 per-head/per-row 并行      |
| 超越函数 | 有`vexp/vlog/sqrt/rsqrt`                                                        | **没有**（LLVM 23 移除了 math intrinsic；要 exp 只能多项式近似） |
| 控制流   | 无 if；tail 用空 range 循环；边界守卫写在 host tl 层                             | `llvm.fcmp` + select 手工拼                                          |
| 适合     | 绝大多数场景（host pid 传参即可并行）、需要 exp/log 的融合 kernel                | raw kernel 内部直接要 program_id 的 attention/GEMV 类               |

**经验法则：能用默认路径就用（有 vexp、好写，并行靠 host pid 传参解决）；
只有确实需要在 raw kernel 内部拿 program_id 时才上 LLVM-direct。**ops/08
§8.6 的案例研究展示了当年用 LLVM-direct
拆 per-head attention kernel 的完整实战与收益边界。

## 5.8 坑清单（每一条都真实炸过）

1. **导入路径错**（5.3）：`call()` 记录静默丢失，kernel “编译成功”但 raw
   代码没进去。症状：数值完全不对但无任何报错。
2. **默认路径用了 `program_id` / `if` / `icmp` / `select`**：不支持，
   前端直接报 AttributeError 或未知节点。**要并行度先换写法不换路径**：
   host tl kernel 拿 `tl.program_id(0)` 传标量进 `_sr_call`（5.5 并行
   形态）；边界守卫的 if 也写在 host tl 层。只有 raw kernel 内部非要
   `program_id` 不可时才换 LLVM-direct。
3. **`vconfig` 两种写法都对但语义有别**：裸语句 `tle.vconfig(N-i, 1)`
   只改 vl 不绑名字；赋值形式 `nvl = tle.vconfig(-1, 1)` 会把 `nvl` 绑成
   编译期常量（能参与 `Nfloor` 的整除计算）。
4. **多趟归约崩 `operand #N does not dominate this use`**：历史版本
   loop body 的局部变量（如 `v = tle.vload(...)`）会泄露到 loop 外，被
   下一个循环误当 iter_arg。已修复（loop 退出时恢复 pre-loop env）；若你在
   旧部署副本上见到此错，先同步两份副本（5.3）。
5. **raw kernel 里 emit `vector_ext` 系 op 必须用 generic form**：
   写 `"vector_ext.batch_macc"(%lhs, %rhs, %acc) : (...) -> ...` 这种
   引号形式，不能用真 assembly 语法——raw_linalg 文本要先经过中层转换
   工具 parse，而它**不认识** vector_ext dialect，真语法直接 parse 失败；
   generic form 任何工具都能过，最后由认识该 dialect 的编译器后端正常
   lower。
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
- LLVM-direct attention 案例研究：ops/08 §8.6
- 全套 raw 测试基线：267 用例 261P/1F/1xF/4err（4 个 error 是诊断脚本的
  pytest 签名问题，跑套件时要 `--ignore`）

下一篇：[06-smt-matrix-engine.md](06-smt-matrix-engine.md) —— 矩阵引擎与 smt 扩展。
