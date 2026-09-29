# Op 03 — RMSNorm

**重要性：★★★★☆**。`y = x / sqrt(mean(x²) + eps) * w`——LayerNorm 去掉
“减均值”一步的简化版，是 Qwen/LLaMA 系 LLM 的标准 norm。如果你已读懂
ops/02，本章只需要 30 分钟：它复用同一套“每 program 一行 + 多趟扫描”
骨架，但**少一趟归约**，正好让你看清两趟与三趟的差别。

## 3.1 这个算子在算什么

对每行 x（长度 N）：

```
ms   = mean(x²)                      # 均方（mean square），注意没有减均值
y    = x / sqrt(ms + eps) * w        # 缩放 + 逐列 weight（没有 bias！）
```

手算例子——`x = [1, 2, 3, 4]`，`w = [1,1,1,1]`，eps 忽略：

```
ms = (1 + 4 + 9 + 16)/4 = 7.5
y  = x / sqrt(7.5) ≈ [0.365, 0.730, 1.095, 1.461]
```

与 LayerNorm 的对照：

| | LayerNorm | RMSNorm |
|---|---|---|
| 归约趟数 | 2（mean、var） | **1**（mean square） |
| 中心化 | 减 mean | 不减 |
| bias | 有 | 无 |
| 输出均值 | ≈0 | 不保证 |

为什么 LLM 选它：实测效果与 LayerNorm 相当，但计算更省（归约少一趟、
参数少一个）。对 kernel 作者的意义：**访存趟数从 5 降到 3**（x 两读 +
w 一读 + y 一写，其中 x 两读可优化）——memory-bound 算子里这就是真金白银。

## 3.2 常见融合形态：fused_add_rmsnorm

Transformer block 里 norm 前面总有残差加：

```python
residual = residual + x          # 一趟读写
y = rmsnorm(residual) * w        # 又两趟读写
```

融合成一个 kernel（`residual += x; y = rmsnorm(residual) * w`，并把新的
residual 也写回）后，`residual + x` 的中间结果不落内存。LLM 推理框架
（vLLM 系）都提供这个融合算子，K3 的 FlagGems 端口也有
`fused_add_rms_norm`。

## 3.3 完整可运行程序

结构与 ops/02 §2.4 完全同型，删掉趟1（mean）、趟2 简化：

```python
# rmsnorm.py — 每 program 一行的 RMSNorm forward
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def rmsnorm_kernel(X, Y, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # ── 趟1: sum(x²)，f32 累加器 ──────────────────────────
    _acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        # other=0. 在这里是安全的：0² = 0，越界 lane 天然不贡献
        # （对比 layernorm 需要 tl.where 归零——平方对 0 是“吸收”的）
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _acc += x * x
    ms = tl.sum(_acc, axis=0) / N              # mean square，标量
    rstd = 1 / tl.sqrt(ms + eps)

    # ── 趟2: y = x * rstd * w ─────────────────────────────
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        tl.store(Y + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x, weight, eps=1e-5):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x)
    BLOCK = min(65536 // x.element_size(), triton.next_power_of_2(N))
    rmsnorm_kernel[(M,)](x_arg, y, weight, x_arg.stride(0), N, eps, BLOCK=BLOCK)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, eps = 128, 1024, 1e-5
    x = torch.randn(M, N, dtype=torch.float16)
    w = torch.rand(N, dtype=torch.float16)

    y = rmsnorm(x, w, eps)

    xf = x.float()
    y_ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
             * w.float()).to(torch.float16)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("PASS  max diff =", (y - y_ref).abs().max().item())
```

```bash
python rmsnorm.py
# PASS  max diff = 0.00xx
```

### 与 layernorm 版的三处差异（就这些）

1. **少一趟**：`ms = mean(x²)` 一趟出结果，不需要先算 mean。
2. **`other=0.` 不再需要 `tl.where` 补救**：layernorm 趟2 里越界 lane 的
   `(0 - mean)²` 会虚增方差，要显式归零；rmsnorm 里越界 lane 读成 0 后
   贡献 `0² = 0`，天然无害。**这就是“mask 填充值是否被后续运算吸收”的
   经典对照**——平方吸收 0，`(·-mean)²` 不吸收。
3. **没有 bias、没有 Mean/Rstd 缓存**（backward 需要 rstd 时同样可以缓存，
   结构照抄 layernorm）。

fused_add_rmsnorm 的改法：两趟里都先做 `x = tl.load(X+cols) + tl.load(R+cols)`
（并把和写回 R 输出），其余不变——残差加“免费”融进已有两趟访存。

## 3.4 K3 关键点

与 LayerNorm **完全同源**，四条军规照搬（ops/02 §2.6）：

- load 后 `.to(tl.float32)` 再归约（`tl.sum` 不提升 dtype）；
- `tl.sqrt` 只收 f32/fp64——`ms` 已是 f32 标量，安全；但若你对 f16 向量
  直接 `tl.sqrt(x)` 会撞 `_check_dtype`（军规第二条）；
- block-ptr 变体 store 要显式 `.to(...)`；
- constexpr 整数没有 `.to()`。

旧版 wheel 已知问题：大 hidden_size 的 rms_norm/fused_add_rms_norm 曾出现
在“scratch 内存溢出 / 越界读”家族失败名单（新版已修复）。遇到全零/垃圾
输出，先升级 wheel 并换新 cache 重测（basics/04 §4.2），再怀疑 kernel
（basics/04 §4.5 分层）。

## 3.5 spine_raw 版本：L0 标量算术链

（来源：`python/tests/raw/test_raw_mean_rmsnorm.py`）raw 版的核心价值是
验证一条 codegen 能力链：**向量 reduce → f32 标量 / index → rsqrt(标量) →
标量广播回向量乘**。mean/rms_norm 家族全靠这条链解锁：

```python
@tle.raw_kernel
def rms_norm_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.cast(tle.vload(X, i), f32)
        acc = acc + vx * vx
    for i in tle.range(Nfloor, N, nvl):        # tail（整除时空循环）
        tle.vconfig(N - i, 1)
        tx = tle.cast(tle.vload(X, i), f32)
        acc = acc + tx * tx
    ms = tle.vreduce_sum(acc) / N              # ← f32 标量 / index 标量
    scale = tle.rsqrt(ms + EPS)                # ← rsqrt 作用在标量上
    # 归一化循环用独立临时名（nx/mx），别复用 vx/tx——历史 bug 会把出作用域
    # 的同名变量误当 loop iter_arg
    for i in tle.range(0, Nfloor, nvl):
        nx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, nx * scale)         # ← 标量广播进向量乘
    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        mx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, mx * scale)
```

对照 ops/02 §2.7 的 raw layernorm：结构一模一样，只是少一趟。raw 层的
尾巴双循环惯用法（basics/05 §5.4④）在这里出现两次。

## 3.6 性能形态

- **纯 memory-bound**：性能上限 = 访存带宽。kernel 层的优化空间只有两个：
  减少趟数（N 塞得进单 tile 时一趟 load 复用）、融合（fused_add）。
- **不要盲目注册 override**：K3 上 Qwen3 52 tok/s（prefill）/ 3.5 tok/s
  （decode）的实测成绩**不依赖** rmsnorm override——native torch 的
  RMSNorm 已经够快，`RMSNORM_MODE` 默认就是关的。优化前先 profile
  （basics/04 §4.3 proton），确认 norm 真的在热点里。
- dispatch 开销视角：decode 场景每 step 的 rmsnorm 很小（1×hidden），
  kernel 本身微秒级，**Python/dispatch 开销反而占大头**——这类场景融合
  （减少 kernel 个数）比单 kernel 提速更有价值。

## 3.7 验证

```bash
python rmsnorm.py
python3 -m pytest python/tests/test_norm_ops.py -q -k rms
python3 -m pytest python/tests/raw/test_raw_mean_rmsnorm.py -q
```

golden（无 torch 内置时的手写参照，3.3 里已用）：

```python
y_ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()
```

fp16 rtol/atol 1e-2。引用历史批次数字先核对 wheel 版本（旧版运行时
grid 丢弃假象家族，basics/04 §4.7）。

## 3.8 练习

1. 跑通 3.3，把 N 改成 1000（BLOCK=1024，有 24 个越界 lane）确认
   `other=0.` 的平方吸收性质——结果应精确不变。再对照着把 layernorm 的
   `tl.where` 删掉跑同形状，看它怎么坏。
2. 实现 fused_add_rmsnorm：签名加 `R`（residual，读旧值、写新值）和
   `R_out`（可与 R 同 buffer）。golden：
   `r_new = r + x; y = rmsnorm(r_new) * w`。
3. 写“单 tile 一趟”变体（N ≤ BLOCK 时）：load 一次 x 存局部变量，reduce
   后直接用寄存器里的 x 算 y。对比两种版本的 `.linalgdir`，数一数
   `transfer_read`（load）出现次数。
4. 思考题：为什么 rmsnorm 不需要缓存 mean，而 layernorm 的 backward 需要
   mean 和 rstd 两个？（提示：写出各自 backward 对 forward 统计量的依赖。）

下一篇：[04-softmax.md](04-softmax.md) —— 数值稳定性入门必修。
