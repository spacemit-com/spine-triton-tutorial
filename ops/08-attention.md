# Op 08 — Attention（QK^T + softmax + AV）

**重要性：★★★★★**。`O = softmax(Q·K^T / √D) · V`——LLM 推理里除了
矩阵乘之外的第一大计算头，也是前面所有章节知识的总汇合：两次 `tl.dot`
（ops/01）、按行 softmax（ops/04）、f16→f32 累加军规（basics/03）、
GEMV 式的访存分析（ops/05）。这一章我们先手算一个 2×2 的 attention，
再写一个完整可运行的单 tile fused causal kernel，然后读生产级
flash-attention 的骨架，最后用一个真实案例讲清楚**"优化到什么程度值得
停"**——这是全教程最重要的一课之一。

## 8.1 这个算子在算什么：手算一遍

定义（单 head，序列长 M，head 维 D）：

```
S = Q · K^T / √D        # (M,M) 分数矩阵：S[i,j] = 第 i 个 query 对第 j 个 key 的相似度
P = softmax(S, dim=-1)  # 每行归一成概率
O = P · V               # (M,D)：每行 = V 各行的加权平均
```

手算例子——`M=2, D=2`，`scale=1`：

```
Q = [[1,0],[0,1]]   K = [[1,0],[1,1]]   V = [[10,20],[30,40]]

S = Q·K^T = [[1,1],
             [0,1]]

causal mask（下三角）后：
row0: [1, -inf]  → softmax → [1, 0]        → O[0] = V[0] = [10, 20]
row1: [0, 1]     → softmax → [0.269, 0.731] → O[1] = 0.269·[10,20] + 0.731·[30,40]
                                             ≈ [24.62, 34.62]
```

**causal mask**：自回归语言模型里第 i 个 token 只能看 j≤i——上三角的分数
在 softmax 前置为 -inf（exp 后为 0，不贡献权重）。这就是 ops/04 练习 4
热身过的下三角 mask，现在它有了真实用途。

**GQA（grouped-query attention）**：现代 LLM（Qwen3、LLaMA3）的 KV head
数比 Q head 数少，多个 Q head 共享一组 KV。`GROUP_SIZE = Hq / Hkv`，
第 hq 个 Q head 用第 `hq // GROUP_SIZE` 组 KV——kernel 里一次整数除法
搞定，**不需要**把 KV 物化复制 Hq/Hkv 份（省显存也省带宽）。

## 8.2 为什么它是核心，钱都花在哪

以 Qwen3-0.6B prefill 为例（proton host-region 计时实测，方法见 8.8）：

- attention scope 占 prefill 墙钟 **~33%**，其中 **~90% 是 q/k/v/o_proj
  四个矩阵乘**（那就是 ops/01 的 GEMM），真正的 QK^T/softmax/AV 只占
  剩下的 ~10%；
- attention 内部再拆：**AV=47%、QK^T=17%**，softmax/repeat_kv/
  transpose/mask 各 <2%。

两个结论直接决定了工程优先级：

1. attention kernel 本体（QKT+AV）值得优化——它内部 AV 又占大头；
2. 但别指望端到端大收益——proj mm 主导。想提升整体，先去做 mm
   （ops/01 的 smt.dot 路径，1.91x）和 `logits_to_keep=1`（lm_head
   206ms→14ms）。8.6 有完整的实测账本。

## 8.3 并行方案

**head 之间完全独立** → 最外层的并行维度就是 (batch, head)：

```
grid = (Z * H,)          # 或把 M 也切进去（flash 风格，见 8.5）
每 program 负责一个 (batch, head) 的完整 attention
```

序列维 M 内要不要再切，取决于 M 的大小：M 小（decode，Mq=1）时一个
program 全包；M 大（prefill，几千 token）时按 BLOCK_M 切行块、K/V 沿
序列分块流式扫——这就是 flash-attention 的 tiling，8.5 展开。

K3 上的 grid 注意项：grid 大小 = Z*H*(M/BLOCK_M)，长序列大 batch 会到
几百上千。旧版 wheel 对超大 grid 有上限（basics/04 §4.5），新版运行时
自动分块；生产 kernel（8.5）本来就用 grid-stride 把 program 数压到固定
值，天然规避。

## 8.4 完整可运行程序：单 tile fused causal attention

先把 8.1 的手算例子放大成一个真 kernel：每个 (batch, head) 一个 program，
整段序列装进单个 tile（M=D=64），**QK^T、mask、softmax、AV 全部在一个
kernel 里完成**（fused，中间结果不落内存）：

```python
# attention.py — 单 tile fused causal attention（B=1, H=4, M=64, D=64）
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit(do_not_specialize=["M", "D"])
def attn_fwd_kernel(Q, K, V, O, sm_scale, M, D,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)               # 每 program 一个 (batch, head)
    base = pid * M * D                   # Q/K/V/O 都 reshape 成 (B*H, M, D)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < M                  # 行方向越界 mask
    n_mask = offs_n < M                  # 列（key）方向越界 mask

    # load Q tile (BLOCK_M, BLOCK_D)、K tile (BLOCK_N, BLOCK_D)
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                mask=m_mask[:, None], other=0.)
    k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask[:, None], other=0.)

    # S = Q·K^T * scale —— tl.dot 输出 f32（军规自动满足）
    qk = tl.dot(q, tl.trans(k)) * sm_scale

    # causal mask：只保留 j <= i，其余 -inf（softmax 后权重为 0）
    causal = offs_m[:, None] >= offs_n[None, :]
    qk = tl.where(causal, qk, float('-inf'))

    # 数值稳定 softmax（ops/04 §4.1 的三趟，在寄存器里完成）
    m_i = tl.max(qk, axis=1)             # (BLOCK_M,) 每行 max
    p = tl.exp(qk - m_i[:, None])        # f32，永不溢出
    l_i = tl.sum(p, axis=1)              # (BLOCK_M,) 每行分母

    # O = P·V / l_i
    v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask[:, None], other=0.)
    acc = tl.dot(p.to(tl.float16), v) / l_i[:, None]

    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=m_mask[:, None])


def attention(q, k, v, causal=True):
    # q/k/v: (B, H, M, D) f16 contiguous
    B, H, M, D = q.shape
    o = torch.empty_like(q)
    sm_scale = D ** -0.5
    BLOCK_M = BLOCK_N = triton.next_power_of_2(M)
    BLOCK_D = triton.next_power_of_2(D)
    attn_fwd_kernel[(B * H,)](
        q.reshape(B * H, M, D), k.reshape(B * H, M, D),
        v.reshape(B * H, M, D), o.reshape(B * H, M, D),
        sm_scale, M, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, M, D = 1, 4, 64, 64
    q = torch.randn(B, H, M, D, dtype=torch.float16)
    k = torch.randn(B, H, M, D, dtype=torch.float16)
    v = torch.randn(B, H, M, D, dtype=torch.float16)

    got = attention(q, k, v)

    # golden：manual f32 分解（不依赖任何 attention 库）
    scale = D ** -0.5
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    mask = torch.tril(torch.ones(M, M, dtype=torch.bool))
    s = s.masked_fill(~mask, float('-inf'))
    ref = torch.matmul(torch.softmax(s, dim=-1), v.float()).half()

    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
    print("PASS attention B=%d H=%d M=%d D=%d  max diff=%.4f"
          % (B, H, M, D, (got - ref).abs().max().item()))
```

```bash
python attention.py
# PASS attention B=1 H=4 M=64 D=64  max diff=0.00xx
```

### 逐段精讲

- **`grid=(B*H,)` + `base = pid * M * D`**：四个张量都当作 (B*H, M, D)
  的扁平三维数组索引，一个 program 处理一个 head。没有 stride 参数是
  因为 host 侧保证了 contiguous——生产 kernel 必须传 stride（8.5），
  教学版先减负。
- **`tl.dot(q, tl.trans(k))`**：`tl.trans` 是寄存器内转置，不产生访存。
  `tl.dot` 的输出天然 f32 累加（ops/01 §1.5），f16 输入不需要手动
  `.to(tl.float32)`——这与逐元素乘加的 GEMV（ops/05 §5.2 要手动升）
  形成对照。K3 上这个 dot 走矩阵引擎通路（basics/06）。
- **`tl.where(causal, qk, float('-inf'))`**：causal mask 用 -inf 而不是
  -1e6 之类的大负数——后者在 scale 很大或 f16 下会"漏"出非零权重。
  -inf 的 exp 严格为 0。唯一的坑：**若某行被全 mask**（causal 下不会
  发生，因为对角线恒可见；但任意 mask 可能），`m_i = -inf`、
  `qk - m_i = -inf - (-inf) = nan`。写通用 mask kernel 时要兜底
  （如 `m_i = tl.maximum(m_i, -1e30)`）。
- **`p.to(tl.float16)` 再 dot**：P 是 [0,1] 的概率值，f16 表示足够
  （生产 kernel 同款）；`tl.dot` 输出仍是 f32，最后 `/l_i` 也在 f32 里
  做，store 时才 cast 回 f16——**归一化放在最后一步**，中途不损失精度。
- **`do_not_specialize=["M", "D"]`**：同 ops/05 §5.2——M=1（decode）
  这类特殊值会被 Triton 折成 constexpr，一个二进制通吃所有 shape 要靠
  显式关特化。
- **单 tile 的适用边界**：M=64 时 qk tile 是 64×64 f32 = 16KB，加上
  q/k/v/acc，离 TCM 256KB 预算（basics/06 §6.3）很远；M=1024 时 qk 是
  1024×1024 f32 = 4MB——**装不下，必须 tiling**。这就是 8.5 的存在理由。

## 8.5 生产形态：flash 风格 tiling + online softmax

M 大了以后，K/V 沿序列切块流式扫，行块并行。难点是 softmax 的"全行
max/全行 sum"依赖整行数据——flash-attention 的解法是 **online softmax**：
边扫 K/V 块边维护运行中的 max/sum/acc，每来一块就修正一次：

```python
# 生产 kernel 骨架（FlagGems _spacemit/ops/flash_attention.py 的 _attn_fwd，节选）
m_i  = 每行当前见过的最大分数（初始 -inf）
l_i  = 每行当前的 exp 和（初始 1.0 起步的约定）
acc  = 当前的 P·V 部分和（未归一化）

for kv_block in K/V 的块:
    qk = smt.dot(q, trans_k) * sm_scale          # 本块分数
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))   # 新的行 max
    p = tl.math.exp(qk - m_ij)                   # 用新 max 重算本块概率
    alpha = tl.math.exp(m_i - m_ij)              # 旧结果的修正系数 ≤1
    l_i = l_i * alpha + tl.sum(p, axis=1)        # 分母同步修正
    acc = acc * alpha                            # ← 灵魂：旧 acc 整体缩放
    acc += smt.dot(p_cast, v)                    # 加上新块贡献
    m_i = m_ij

out = acc / l_i                                  # 扫完才归一化，一次除法
```

三个和 8.4 的结构性差异：

1. **`alpha` 修正**：max 变了，之前按旧 max 算的 exp 全部偏大
   `exp(m_old - m_new)` 倍——把 acc 和 l_i 同乘 alpha 就精确修正了。
   数学上严格等价于"先看完全部再 softmax"。
2. **grid-stride + task 解码**：`pid` 不直接是 (z, h, m_block)，而是
   `task_hz_idx = pid // NUM_BLOCKS_M; task_m_idx = pid % NUM_BLOCKS_M`，
   再用 `pid + num_ctas * block_idx` 的 grid-stride 循环遍历所有任务——
   program 总数固定（NUM_CTAS），彻底避开 grid 上限问题（8.3）。
3. **`smt.dot` 替代 `tl.dot`**：K3 矩阵引擎直通路径（basics/06），
   accumulator 是 MICRO tile 打包布局（`acc_4d`，形状按
   MICRO_M/K/N=16/8/32 组织），最后 `smt.view` 解包成普通二维再除 l_i。
   `off_hkv = off_hq // GROUP_SIZE` 就是 8.1 说的 GQA 除法。

生产 kernel 还把 causal 拆成 STAGE（全 causal 块 / 对角混合块分开处理，
对角块才逐元素判 `offs_m >= offs_n`，全 causal 块直接跳过 mask）——
纯性能优化，语义与 8.4 一致。

**backward 一句话**：`_attn_bwd_dq` 有 autotune 配置约束
（`BLOCK_M2 % BLOCK_N2 == 0` 的 static_assert，配置违反直接编译期
报错）；backward 的部分形状在旧版编译器上有崩溃家族（rc=134 无 Python
栈），处理流程同 basics/04 §4.4——新 cache、最小化、升级 wheel、报告。

## 8.6 案例研究：拆开的 QKT/AV kernel 与"收益边界"

生产 fused kernel 之外，历史上有过另一条路线：**不做 fused，把 QK^T 和
AV 写成两个独立的 spine_raw（LLVM-direct）kernel，中间 softmax 留在
Python**。这条路线的价值在于它是"vfwmacc 向量通路手写 attention"的
完整案例，也产出了全教程最重要的实测账本。

两个 kernel 同一套模式（LLVM-direct 层，basics/05 §5.7）：

```
grid = (Hq,)                        # 每 program 一个 Q head
hq = tle.program_id(0)
hk = hq // Nrep                     # GQA：整数除法，跳过 repeat_kv 物化
```

- **QKT kernel**（`O[Hq,Mq,Mk] = Q[Hq,Mq,D]·K[Hkv,Mk,D]^T`，每个输出
  元素是长度 D 的点积）：D=128 需 **2×64-chunk**——把 Q 的两个 64 元素
  chunk（qv0/qv1）hoist 出 n 循环，每 (m,n) 各 load K 的两个 chunk，
  2 次 vfwmacc.vv + 2 次向量水平归约链式累加成 dot。**必须分离 Mq/Mk**
  （decode 时 Mq=1、Mk 随 KV cache 增长，混用一个参数会直接错）。
  f16 输入 → f32 acc（vfwmacc 是 widening 乘加）→ 截断回 f16。
  实测：prefill max_diff=7.8e-3、decode 2.4e-4；**kernel 1.1ms vs
  原生 15.4ms**。
- **AV kernel**（`O[Hq,M,N] = A[Hq,M,K]·V[Hkv,K,N]`）：A 的行做标量
  广播（vfwmacc.vf——GEMV 同款指令，ops/05 §5.3），V 走向量通路；
  N=128 同样 2×64-chunk（两个独立 acc）。实测 **max_diff=0（逐位
  精确）**；kernel 0.81ms vs 原生 15.4ms。

kernel 级 ~8x 提速。但端到端呢？

| 接入路径 | prefill | decode |
|---|---|---|
| eager attention + 替换 QKT/AV | 9.79 tok/s（1.18x） | 2.15（1.01x） |
| SDPA 接口层替换（保持默认 SDPA） | 9.63（1.16x） | 2.25（1.06x） |

**kernel 8x，端到端只有 1.01~1.18x。** 为什么？回到 8.2 的账本：
attn scope 里 ~90% 是 q/k/v/o_proj 四个 mm，attention kernel 本体只动
剩下 10% 里的一部分；Amdahl 定律冷酷生效。

**fusion 也被实测否决**：把 QKT+softmax+AV 融成一个 kernel"理论上省
中间张量的访存"，但 profile 显示 Python softmax+mask+scale 只有
~0.9ms/层 × 28 层 ≈ 25.4ms，**占 prefill 墙钟 0.8%**——融合的收益上限
就这么大，而实现成本（LLVM-direct 层没有现成 exp，要手写多项式近似）
极高。不做。

这一节的三条普适教训：

1. **先 profile 再动手**：知道 AV=47%/QKT=17%/softmax<2%，才知道该写
   哪两个 kernel、不该碰 softmax；
2. **kernel 提速 ≠ 端到端提速**：算清楚你的 kernel 占墙钟百分之几，
   乘以提速倍数就是收益上限；
3. **优化的正确顺序是占比从大到小**：mm（smt.dot，端到端 1.91x）→
   lm_head（logits_to_keep，14.7x）→ attention kernel（本章）。

## 8.7 K3 关键点与坑

1. **D 从 config.json 读，不要猜**。Qwen3-0.6B 的 head_dim=128 是
   config.json 显式指定的，**不是** hidden_size/num_heads（那样算出来
   也是 128 纯属巧合，别的模型就不等了）。D=128 超过 f16 单向量寄存器
   宽（64），向量通路必须 2×64-chunk；矩阵引擎通路则按 MICRO tile
   （K 维 8）自动分块。
2. **tile 尺寸 vs TCM 预算**（basics/06 §6.3）：qk/p/acc 三个
   BLOCK_M×BLOCK_N 级 tile 同时活着，f32 acc 再翻倍。BLOCK_M=
   BLOCK_N=128、D=128 时单是 acc 就 128×128×4B=64KB——配置 BLOCK 前
   先心算总预算，256KB 超了就 rc=134 编译器崩溃（basics/04 §4.4
   流程）。
3. **`tl.exp` dtype 军规**（ops/04 §4.5 坑 1）：qk 是 f32（tl.dot 输出）
   所以本章 kernel 天然安全；但任何"先 cast f16 再 exp"的'优化'都会
   CompilationError。生产 kernel 用 `tl.math.exp` 作用在 f32 上，同规。
4. **f16→f32→f16 的截断点只在 store**：中间 m_i/l_i/acc 全程 f32。
   p cast 回 f16 喂第二个 dot 是唯一例外（概率值域 [0,1]，安全）。
5. **decode 形态单独测**：Mq=1 时 causal mask 退化成全可见、softmax
   退化成对整行 KV、GEMV 性格（memory-bound）——prefill 调好的 BLOCK
   配置在 decode 下往往不是最优，且 Mq/Mk 不分开的 kernel 直接错。
6. **bf16 attention 当前不可用**：bf16 的 matmul 通路是平台限制
   （ops/01 §1.7 同款 rc=134），f16/f32 可用。
7. **测试参数里的 dtype 陷阱**：attention 测试常用 [fp16, bf16] 双
   dtype 表——"dtype1 全挂"多半就是第 6 条的平台限制，不是你的 kernel
   （先读测试文件的 parametrize 再归因，basics/04 §4.7）。

## 8.8 profile 方法：钱花在哪一段

K3 的 wheel 自带 proton 计时（basics/04 §4.3），两个用法：

- **kernel 级**：`PROTON_KERNEL_CAPTURE=1` 编译，自动记录每个 kernel 的
  执行时间——QKT/AV/proj mm 谁大一目了然。
- **host-region 级**：从 Python 里把任意代码段包进计时区间（进入/退出
  各记一个时间戳），能拆到 `self_attn.forward`、`mlp.forward`、
  `lm_head.forward` 这一级。8.2 的 "~33% / ~90%" 账本就是这么来的，
  合计能覆盖 99.6% 墙钟。

拆 attention 内部（repeatkv/QKT/softmax/AV/transpose/mask 各步）需要
让 transformers 走 eager attention 实现（`attn_implementation="eager"`，
默认 sdpa 不经过可拆分的 Python 函数），然后对 `eager_attention_forward`
包一层计时。**这只用于 profile**；生产接入走注册路径（如 FlagGems 的
`scaled_dot_product_attention` 替换，fast-path 条件 D==128 && f16 &&
B==1 && GQA && contiguous && 无 attn_mask && dropout==0，不满足则
fallback 到 manual 分解——fallback 故意不调 aten SDPA，避免算子替换
语境下的递归）。

## 8.9 验证

```bash
python attention.py                        # 本章单 tile kernel
```

golden 三选一：

- `torch.nn.functional.scaled_dot_product_attention`（同参数，GQA 时
  `enable_gqa=True`）；
- manual f32 分解（8.4 的写法，最可控）；
- 8.1 的 2×2 手算值（写进单元测试，防"golden 和 kernel 犯同一个错"）。

必测形态：causal / 非 causal；prefill（Mq=Mk 大）/ decode（Mq=1, Mk 大）
——Mq/Mk 分离的回归点；GQA（Hq≠Hkv）；M、D 非 2 幂（mask 路径）。
f16 容差 1e-2 起步；AV 这类"权重×值"的段单独测可以要求逐位精确
（8.6 的 max_diff=0 说明做得到）。

## 8.10 练习

1. 跑通 8.4。把 causal mask 那两行注释掉，验证输出变成非因果
   attention（golden 同步去掉 masked_fill），确认 max diff 仍然很小。
2. 把 8.1 的手算例子写成单元测试：M=2, D=2，Q/K/V 用 8.1 的字面值，
   期望输出 `[[10,20],[24.62,34.62]]`（atol 1e-2）。这是你第一个
   "人肉可验证"的 attention 测试。
3. 把 M 提到 256（BLOCK_M=BLOCK_N=256），跑之前先按 8.7 第 2 条心算
   TCM 预算，预测会不会崩，再验证。若崩了，把 BLOCK_N 降到 64 并按
   8.5 的骨架加 K/V 分块循环（不需要 online softmax——非 causal 且
   max 可以先全行扫一遍，两趟法）。
4. 实现 GQA：K/V 的 head 数改成 Hq//2，kernel 里
   `off_hkv = pid_q // 2`（提示：base 的算法对 K/V 要用 Hkv）。golden
   用 `F.scaled_dot_product_attention(..., enable_gqa=True)`。
5. 思考题：8.6 的账本里，若把 proj mm 也换成 smt.dot 路径（ops/01，
   ~1.9x），attention kernel 保持 8x，端到端 prefill 大约能到多少？
   用 Amdahl 公式算，再对照 ops/01 的实测 ~40 tok/s 验证你的模型。

下一篇：[09-cumsum-scan.md](09-cumsum-scan.md) —— 前缀和与 scan，
编译链上坑密度最高的算子族。
