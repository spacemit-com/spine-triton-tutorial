# 06 — smt 扩展与 K3 矩阵引擎（进阶）

> **定位**：进阶章。ops/01（GEMM）、ops/05（GEMV）、ops/08（Attention）
> 会反复引用本章的硬件事实；第一遍可以只读 6.1–6.4（硬件背景），API 细节
> 用到再回来查。

## 6.1 向量单元 vs 矩阵引擎：为什么 GEMM 要走另一条路

K3 上有两种算力：

- **RVV 向量单元**：一条 `vfmacc` 做“标量 × 向量”的乘加，一次 32 个 f32
  （或 64 个 f16）乘加。逐元素算子和归约全靠它。
- **矩阵引擎**（`vfwmacc` / `vfwmadot` 指令族）：一条指令完成一个**小矩阵块**
  的乘累加。以 MICRO tile 16×8×32 为例，一条指令的算术量 = 16×8×32 = 4096
  次 f16 乘加——是单条向量指令的两个数量级。

对 GEMM（`C = A @ B`）这种计算密度高的算子，走不走矩阵引擎的差距是
**几倍到几十倍**。实测参照（Qwen3-0.6B prefill，K3 真机）：
proj mm 五个 shape 合计，矩阵引擎路径 3.51ms vs 纯 vfwmacc.vv 路径 21.49ms
（6 倍）；端到端 prefill ~40 tok/s vs ~21 tok/s。

好消息：**你写标准的 `tl.dot`，编译器会自动映射矩阵引擎**（经 linalg.matmul
→ mmt4d → pack/vfwmadot 编排）。`smt` 扩展是给“自动映射不够用”的场景准备的
手动挡。

## 6.2 两条矩阵引擎指令，各自的脾气

### vfwmacc（`vector_ext.batch_macc` lower 而来）

语义：

```
batch_macc(lhs: memref<m×k>, rhs: vector<k×n>, acc: vector<m×n>)
   →  acc[m,n] += Σ_k lhs[m,k] · rhs[k,n]
```

展开形态是 `for k: acc += lhs[:,k]标量 × rhs[k,:]向量`——每 k 一条
vfwmacc.vf。两个关键性质：

- **lhs 直接读内存（memref），rhs 留在寄存器**。当 n=1 或 rhs 可广播时，
  这就是 GEMV 的完美形态：**B 向量零搬运**（ops/05 的主角）。
- 约束：输出列数 n 必须 **≥64 且 %64==0**（f16 一个 scalable 寄存器 64 元素）。
  M×1 的 mv 不满足 → 要做**转置映射**：`C[N] = A[N,K] @ B[K]` 转成
  `C[1,N] = B[1,K] @ A^T[K,N]`，其中 A^T 用 `load_2d_t`（strided 转置读，
  strides=[1,M]）直接从行主序 A 读出来，**不需要 host 侧 pack**。

### vfwmadot（`vector_ext.matmul` / mmt4d lower 而来）

矩阵引擎的“点积通路”，完整 GEMM 走它。但有一个反直觉的事实：

- **单条 `vector_ext.matmul` 的 m1 形态只算出 `acc[0,0]`**（实测全 1 输入
  只有 acc[0,0]=K，其余全 0）。所以完整 GEMM 必须走 **mmt4d + pack/unpack
  编排**：把 A/B 重排成 MICRO-tile 交错布局（`linalg.pack`），算完再
  unpack 回普通布局。这套编排编译器负责——`tl.dot` 和 `smt.dot` 底下
  都是它。

**MICRO tile**：矩阵引擎一次原子计算的块尺寸。K3（arch `0xA064`）f16 的
合法配置是 **M/K/N = 16/8/32**。运行时可查：`tle.mma_cube("f16")`。
MICRO 值必须来自当前 arch 的合法表——传错值不会报“参数错误”，而是
**静默算错或直接崩**（autotune config 改写 MICRO 时要格外小心）。

## 6.3 TCM：256KB 的硬预算

TCM（紧耦合存储）是每个 worker 的私有高速存储，类比 GPU 的 shared memory。
编译器把大 tile 的中间量（pack 缓冲、spill）放这里，**单次分配超过 256KB
就失败**——而且生成的 kernel **不检查分配返回值**：NULL 直接解引用 → 段错误。

这条硬预算解释了一堆“玄学 crash”：

| 现象 | 本质 |
|---|---|
| `tl.dot`/`smt.dot` 的 BN=512 必崩、BN=256 正常 | output tile 32×512 的中间量 spill 超 256KB |
| BK=K 一次性展开 hang | BK=256、K 大时 noloop 需 608KB |
| 8192 元素 tile PASS / 16384 元素崩 | 恰好跨过 256KB 边界（实测） |
| 崩溃只依赖 config 不依赖数据（M=1 也崩） | 分配发生在 kernel 执行早期，与数据无关 |

**写 kernel 的实操规则：tile 元素数控制在 8192 以内最稳**；要更大就 K 维
分块循环。段错误且“与数据无关、只与 BLOCK 大小有关”时，第一嫌疑就是 TCM。

## 6.4 smt API 逐个讲

```python
import triton.language.extra.smt as smt
```

| API | 作用 | 必知细节 |
|---|---|---|
| `smt.alloc(shape, type=...)` | 在 TCM 分配一块缓冲 | **默认 dtype 是 f32！**存 f16 tile 必须显式 `type=b_ptr.dtype.element_ty`，否则 store 时撞 “Block element type(fp32) and value element type(fp16) mismatch”（block-ptr store 是严格类型检查，无隐式 cast） |
| `smt.view(desc, offsets, shape, micro)` | 取 subview / pack 视图 | `micro=(1,1)` 用于 unpack（见 6.5） |
| `smt.descriptor_load(block_ptr, offsets)` | 描述符块加载 | SPLIT_M 里 b tile 中转用 |
| `smt.dot(a, b)` | 矩阵引擎点积 | **返回 packed MICRO-tile 格式**，见 6.5 |
| `smt.parallel(...)` | 并行循环（range 子类） | **默认 `bind_sub_block=False`**，见 6.6 |
| `smt.compile_hint(ptr, name, val)` | 编译提示 | 少用 |
| `smt.mbarrier` 系 | 异步同步 | 见 6.8，新 kernel 尽量别用 |

## 6.5 `smt.dot` 的 packed 返回值（最容易静默算错的地方）

`smt.dot(a, b)` 的返回值是 **packed MICRO-tile 格式**——元素按矩阵引擎的
内部交错布局排列，不是你想象中的行主序 `BM×BN` 块。**直接把它加到普通
`tl.zeros` 累加器上，数值会错得离谱**（实测 diff ~175），而且不报任何错。

正确写法——先 unpack 再累加：

```python
acc = tl.zeros([BM, BN], dtype=tl.float32)
for k in range(0, K, BK):
    a = ...; b = ...
    partial = smt.view(smt.dot(a, b), (0, 0), (BM, BN), (1, 1))  # ← unpack
    acc += partial
```

`smt.view(..., micro=(1,1))` 就是把 packed 布局还原成普通 (BM, BN) 视图。
**记住这条：见 `smt.dot` 必跟 `smt.view(..., (1,1))`。**

## 6.6 `smt.parallel` 与 `bind_sub_block=False` 的由来

`smt.parallel` 设计意图是“带 sub-block 绑定的并行循环”。历史行为：

- `bind_sub_block=True` 时，前端给循环打属性，编译器把它降级成一条
  特殊的并行 lowering 链；
- 这条链在当前工具链上会**触发编译器断言崩溃**（rc=134、无 Python 栈，
  basics/04 §4.5 的编译器后端形态）。

所以**默认值已改为 `False`**：循环保持普通顺序循环（编译器最稳的常态
输入），代价是失去 sub-block 绑定语义。**写新 kernel 不要显式传
True。**SPLIT_M/SPLIT_MN 分支在 False 下已在 K3 真机端到端验证通过。

（顺带一个排错签名：如果 `smt.parallel` 在 tracing 期报 “Only range and
static_range iterators are currently supported”，说明你的 wheel 版本较旧、
前端还不认识 smt.parallel——升级 wheel 即可。）

## 6.7 SPLIT_M / SPLIT_MN / SPLIT_K：GEMM tile 的三种切法

（完整可跑来源：`python/examples/test_smt_mm.py`，三分支 512³ f16 均在
K3 真机端到端验证通过。）

一个 BLOCK_M×BLOCK_N 的 output tile，内部还要按 MICRO tile（16/8/32）切。
三种编排的区别在于**沿哪个维度切 sub-block、B 矩阵怎么中转**：

```
SPLIT_M：M 维切 sub-block
  for m_sub in smt.parallel(0, BM, MICRO_M):
      b tile 经 smt.descriptor_load + smt.view(..., (MICRO_K, MICRO_N))
      做 pack 视图复用（B 只加载一次，多个 m_sub 共享）

SPLIT_MN：M、N 双维切
  嵌套循环；smt.alloc 的 b 缓冲必须显式 type=f16（6.4 的坑）

SPLIT_K：K 维切
  走普通 tl.range 循环累加（不涉及 smt.parallel）
```

`BLOCK_SIZE_K` 受 TCM 预算约束：BK=256 是实测可用值（TCM 256KB）；
“noloop BK=K”在 K 大时需要 608KB → 溢出 hang，**K 大必须分块**。

## 6.8 mbarrier：能用则不用

`smt.mbarrier/alloc/barrier_arrive/barrier_wait` 会编译出对
`spine_mbarrier_{alloc,release,arrive,wait}` 四个符号的引用。麻烦在于：
**运行时库不保证提供 mbarrier 实现**——部分版本的运行时没有这四个符号，
kernel 编译全通、`.so` 正常生成，加载时才报 `load kernel failed` /
`undefined symbol`。

实践建议：**新 kernel 不依赖 mbarrier**。`bind_sub_block=False` 下
`smt.parallel` 就是顺序循环，同一 program 内 TCM 的 store→load 按程序序
执行，天然不需要同步——`test_smt_mm.py` 删净 mbarrier 后三分支数值仍
全对（`nm -D --undefined-only kernel.so` 可验证零 `spine_mbarrier_*`
引用）。

排错签名：编译全通但加载失败 → `nm -D --undefined-only kernel.so` 看到
`spine_mbarrier_*` 未定义 = 你环境里的运行时没有这四个符号（basics/04
§4.5 的“运行时加载层”行）。

## 6.9 什么时候需要手写 smt？

大多数情况**不需要**。决策树：

1. 先写标准 `tl.dot` + block_ptr（ops/01）——编译器自动走 mmt4d 矩阵引擎
   通路，这就是生产 GEMM 的默认形态；
2. 自动通路失败（编译崩）或次优（性能不达标）→ 才考虑手写 smt：
   显式控制 TCM 分配、pack 布局（如 SPLIT_N 的 b tile 中转）；
3. 更底层的需求（B 零搬运的 GEMV、转置读、自定义累加链）→ spine_raw 的
   矩阵引擎原语 `batch_macc`/`vmadot`（basics/05 + ops/05）。

无论哪层，验收标准一致：

```bash
# 符号干净（无意外未定义符号）
nm -D --undefined-only <cache里的 kernel.so>
# 数值 vs torch golden（f16: rtol/atol 1e-2 量级）
# 性能对照 perf.py / proton（basics/04 §4.3）
```

## 6.10 动手练习

1. 在 K3 上跑 `python3 python/examples/test_smt_mm.py`。第一次编译较慢，
   之后缓存命中。
2. 在 SPLIT_M 分支里把 `smt.view(smt.dot(...), ..., (1,1))` 的 unpack 去掉
   （直接 `acc += smt.dot(a, b)`），观察数值错成什么样——亲眼见一次
   packed 格式，比读十遍文档记得牢。改回来。
3. 把 `smt.alloc(shape=(...))` 的显式 `type=` 删掉（f16 b tile 场景），
   记录报错原文——这是 block-ptr 严格类型检查的标准签名。改回来。
4. 计算题：BM=32、BN=256、f32 累加器，output tile 占多少 TCM？
   （32×256×4B = 32KB，安全。）BN 翻倍到 512 呢？结合 6.3 的 crash 表
   想想为什么 BN=512 崩的不只是 output tile 本身。

下一篇：进入算子实战 [../ops/01-gemm.md](../ops/01-gemm.md) —— GEMM。
