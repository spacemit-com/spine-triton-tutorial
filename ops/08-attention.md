# Op 08 — Attention（QK^T + softmax + AV）

LLM 推理第二大头（Qwen3-0.6B prefill 里 attn scope 占 ~33%，其中 AV=47%、
QKT=17%、softmax/repeatkv/transpose/mask 各 <2%，其余是 q/k/v/o_proj mm）。
K3 上 attention 的两个 matmul 用 vfwmacc 重写可拿到 kernel 级 8x，但端到端
收益受 proj mm 主导——本章同时给出“做到什么程度值得停”的实测数据。

## 架构：per-head GQA kernel 对

不做 fused flash-attention，而是 **QKT 与 AV 两个独立 kernel + 中间 Python
softmax**（融合已被实测否决，见文末）。两个 kernel 同一套模式：

```
grid = (Hq,)                       # 每 program 一个 Q head
hq = tl.program_id(0)
hk = hq // Nrep                    # GQA：直接除，跳过 repeat_kv 物化
```

（LLVM-direct 里是 `hq = tle.program_id(0)`、`hk = llvm.sdiv(hq, Nrep)`。）

### QKT kernel（vfwmacc.vv + reduce.fadd）

`O[Hq,Mq,Mk] = Q[Hq,Mq,D] · K[Hkv,Mk,D]^T`，per (m,n) 输出元素是一个长度 D
的点积：

- **D=128（Qwen3 head_dim，config.json 显式指定）需 2×64-chunk**：hoist
  `qv0/qv1`（Q 的两个 64-chunk）出 n 循环；每 (m,n) load `kv0/kv1`，
  2 次 vfwmacc.vv + 2 次 `llvm.vector.reduce.fadd` 链式累加到 `dot`。
- **必须分离 Mq/Mk**：decode 时 Mq=1、Mk 随 KV cache 增长。
- f16 输入 → f32 acc（widening）→ fptrunc → f16 输出。
- 精度：prefill max_diff=7.8e-3、decode 2.4e-4。kernel 1.1ms vs native 15.4ms。

### AV kernel（vfwmacc.vf）

`O[Hq,M,N] = A[Hq,M,K] · V[Hkv,K,N]`：A 的行做标量广播（vfwmacc.vf），
V 走向量通路。N=128 同样 2×64-chunk（两个独立 acc0/acc1，各 2 次 vfwmacc.vf，
最后 2 次 fptrunc+store）。精度 max_diff=0（exact）。kernel 0.81ms vs 15.4ms。

参考实现：`bench_qwen_sdpa_override.py` 的 `attn_qkt_gqa_raw` /
`attn_av_gqa_raw`（spine_raw LLVM-direct path，需要 program_id）。

## 中间 softmax

Python 侧 `torch.softmax(scores * scale + mask)`。实测该段仅 ~0.9ms/layer，
占 prefill wall 的 **0.8%**——优化优先级最低。

## 接入方式（生产 = 注册路径，不 monkey-patch）

算子必须落 FlagGems 注册路径：`_spacemit/ops/flash_attention.py` 的
`scaled_dot_product_attention` fast path 条件：

```
D == 128  &&  f16  &&  B == 1  &&  GQA  &&  contiguous
&&  attn_mask is None  &&  dropout == 0
```

命中则 `_attn_qkt_gqa_host` + Python softmax（causal for Mq>1）+
`_attn_av_gqa_host`；否则 fallback 到 manual Q@K^T + softmax + @V
（**不走 aten SDPA，避免 use_gems 下递归**）。注册进 `_FULL_CONFIG` 后
SpecOpRegistrar 自动替换到 `flag_gems` namespace。bench 脚本只允许
`flag_gems.use_gems(include=[...])` + HF 原生 API。

两条历史接入路径的成绩（Qwen3-0.6B）：

| 路径 | prefill | decode |
|---|---|---|
| eager + monkey-patch `eager_attention_forward` | 9.79 tok/s (1.18x) | 2.15 (1.01x) |
| SDPA override（保持 default SDPA） | 9.63 (1.16x) | 2.25 (1.06x) |

SDPA override 在 decode 胜（避免每 step 的 Python dispatch）。注意
`use_gems(include=[...])` 必须限定白名单——riscv64 板上全量注册会触发未优化
算子 JIT（且 x86 工具混装的老包会 Exec format error）。

## 收益边界与放弃项（重要）

- **kernel 8x ≠ 端到端 8x**：attn scope 里 ~90% 是 q/k/v/o_proj 四个 mm，
  attention kernel 只动剩下的 10%。端到端 prefill 1.16–1.18x、decode
  1.01–1.06x。要继续提升，去做 mm（ops/01 的 smt.dot，1.91x）和
  `logits_to_keep=1`（lm_head 14.7x）。
- **Fuse QKT+softmax+AV 已放弃**：softmax 段仅 0.8% wall，fusion 收益 <<
  实现成本；且 LLVM-direct path 没有 exp（LLVM 23 移除 math intrinsic），
  要多项式 exp + codegen 改动。
- **backward**：`_attn_bwd_dq` 有 autotune config 约束（`BLOCK_M2%BLOCK_N2==0`
  的 static_assert）；attn_bwd 编译链上还有 `VectorExtToLLVM.cc:112
  UNREACHABLE` 家族失败（spine-mlir 移交件）。
- **bf16 attention 不可用**：bf16 linalg.matmul 撞 MatmulConfigAnalysis
  UNREACHABLE（平台豁免）；测试参数里 dtype1 是 bf16 的 sdpa 用例全属此类。

## profile 方法

monkey-patch `eager_attention_forward` wrap 各步（repeatkv/QKT/softmax/AV/
transpose/mask）用 proton host-region 计时（`_scope()` context manager 调
`proton_enter_kernel/exit_kernel`），**必须 force `attn_implementation="eager"`**
（默认 sdpa 不走该函数，patch 不生效）。

## 验证

golden：`torch.nn.functional.scaled_dot_product_attention`（同参数、
enable_gqa=True）或 manual 分解。prefill/decode 两形态都要测（Mq/Mk 分离的
回归点）。QEMU 上大 shape attention 单 kernel 可达 20min+，复跑用
`SPINE_TRITON_RPC_TIMEOUT=7200` + 两遍法（第一遍编译、第二遍缓存执行）。

下一篇：[09-cumsum-scan.md](09-cumsum-scan.md)
