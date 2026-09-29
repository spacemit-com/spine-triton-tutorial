# Op 03 — RMSNorm

`y = x / sqrt(mean(x²) + eps) * w`。相比 LayerNorm 去掉了减均值一步
（少一趟归约、少一个标量），是 Qwen/LLaMA 系 LLM 的标准 norm。常见融合形态
**fused_add_rmsnorm**：`residual += x; y = rmsnorm(residual) * w`，把残差加
吸收进 norm kernel，省一次全量读写。

## 标准 Triton 写法

```python
@triton.jit
def rmsnorm_kernel(X, Y, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # 趟1: sum(x²)，f32 累加
    _acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _acc += x * x
    ms = tl.sum(_acc, axis=0) / N            # mean square
    rstd = 1 / tl.sqrt(ms + eps)

    # 趟2: y = x * rstd * w
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        tl.store(Y + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)
```

fused_add_rmsnorm 把 `x += residual` 放进两趟里（并写回 residual 输出），
其余结构不变。

K3 关键点与 LayerNorm 完全同源：f32 累加、`.to(tl.float32)` 显式提升、
block-ptr store 严格类型检查、`tl.sqrt` 只收 f32。

## spine_raw 版本

（来源：`python/tests/raw/test_raw_mean_rmsnorm.py`）核心链是
**reduce → 标量 /N → rsqrt(scalar) → 标量广播回向量 mul**：

```python
@tle.raw_kernel
def rms_norm_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.cast(tle.vload(X, i), f32)
        acc = acc + vx * vx
    for i in tle.range(Nfloor, N, nvl):        # tail
        tle.vconfig(N - i, 1)
        tx = tle.cast(tle.vload(X, i), f32)
        acc = acc + tx * tx
    scale = tle.rsqrt(tle.vreduce_sum(acc) / N + EPS)   # f32 标量
    for i in tle.range(0, Nfloor, nvl):
        tle.vstore(out, i, tle.cast(tle.vload(X, i), f32) * scale)
    for i in tle.range(Nfloor, N, nvl):
        tle.vconfig(N - i, 1)
        tle.vstore(out, i, tle.cast(tle.vload(X, i), f32) * scale)
```

这条链验证的是 codegen 的 L0 标量算术（`f32 scalar / index`、`rsqrt(scalar)`、
scalar-vector broadcast mul）——mean/rms_norm 家族全靠它解锁。

## 性能形态与已知问题

- rmsnorm 是纯 memory-bound：性能上限 = 访存带宽，kernel 优化空间在“少趟数”
  （N 能塞进单 tile 就一趟）与 fused_add 融合。
- **native RMSNORM_MODE 默认关闭**：K3 上 Qwen3 52 tok/s / 3.5 tok/s 的成绩
  不依赖 rmsnorm override（走 native torch 即可），注册它之前先 profile 确认
  有收益。
- 大 hidden_size 的 rms_norm/fused_add_rms_norm 曾出现在“TCM 溢出/负 vl”
  家族失败名单里（spine-mlir 01 族，本地 b86dd23 修复）——遇到全零/垃圾输出
  先确认 spine-opt 二进制代际与 libspert ≥0.6.3，再怀疑 kernel。

## 验证

```bash
python3 -m pytest python/tests/test_norm_ops.py -q -k rms
python3 -m pytest python/tests/raw/test_raw_mean_rmsnorm.py -q
```

golden：`x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w`。
fp16 rtol/atol 1e-2。同样注意：引用历史批次的“数值失败”先核对 runtime 版本
（0.6.0 grid 假象家族）。

下一篇：[04-softmax.md](04-softmax.md)
