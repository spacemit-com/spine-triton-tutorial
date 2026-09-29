# Op 06 — Reduction（sum / mean / max / argmax）

归约是所有 norm、softmax、loss 算子的公共底座。Triton 层就两个原语
（`tl.sum/tl.max` + axis），但 K3 的 scalable vector lowering 在
**归约长度与 vscale 的关系**上有一系列边界行为，值得单独一章。

## 标准写法

```python
@triton.jit
def sum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.).to(tl.float32)  # f32 累加
    partial = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, partial)          # 每 program 一个部分和
# host 侧第二段归约（torch.sum(partial)）或 kernel 内 atomic/单 program 收尾
```

行归约（`tl.sum(x_2d, axis=1)`）与 argmax（`tl.argmax`，或手动
max+where+index）同构。spine_raw 版本：`tle.vreduce_sum/max/min/mul` +
标量算术链（见 `test_raw_sum.py`、`test_raw_argmax.py`、`test_raw_max_dim.py`、
`test_raw_mean_dim.py`）。

## K3 关键点：行长 vs vscale

K3 vscale 下 f32/i32 一个 scalable 寄存器容纳的元素数是固定倍率。归约维长度
（或 mask 后的有效长度）与它的关系决定 lowering 路径，历史上三种边界都出过
spine-opt 级 bug（均已修/移交，写 kernel 的人要知道症状）：

| 边界 | 症状（旧二进制） | 状态 |
|---|---|---|
| BLOCK < 寄存器宽（如 BLOCK=16 int32） | in_bounds 误推断 → poison init → vredsum 折叠垃圾 lane：`tl.sum` 恒差 -16、f32 出 NaN | 已修（`d04296f`，VectorLoadConversion 强制 broadcast padding / tu 模式） |
| 行长 % vscale != 0 | `ConvertToScalableVector.cc:55 totalNumel % vscale == 0` SIGABRT | 移交（需 padding 或优雅拒绝） |
| 行长 == vscale（如 len 15/16） | `ConvertVectorToSCFPass.cc:962 rank==1` SIGABRT（scalable 单位维被误 dropDim） | 已修（`c63415b`） |

**triton cache 会掩盖这类编译 bug**（缓存 .so 不走编译）——“某 shape 一直崩/
一直错”先清 cache 复现。

其余通用约束：

- **f32 累加**：fp16 大归约的顺序累加误差可达 rel≤0.6%（vector_norm 24F 的
  教训）；容差按归约长度缩放，别只按输出元素数。
- **大 tile / 大中间量**：归约展开成的中间 buffer 也吃 TCM 256KB 预算。
- **grid**：两段式归约的第一段 grid = cdiv(N, BLOCK)，旧 runtime 注意 ≤512。

## mean / var 的标量链

mean = `sum / N`、var = `E[x²] - mean²`。spine_raw 默认 path 的
“reduce → 标量除 index → rsqrt → 广播回向量”链是 mean/rms_norm/layernorm/
var 家族共同底座（`test_raw_mean_rmsnorm.py` 是它的 L0 验证）。单趟
`E[x²]-mean²` 形式对 padding 0 友好（0²=0 无膨胀），但注意大均值小方差时的
消去误差——生产 kernel（var/var_mean）用 Welford 或两趟 `(x-mean)²` 更稳。

## argmax / 带索引归约

`tl.argmax(x, axis)` 或 `idx = tl.sum(tl.where(x == m, offs, 0), axis)` 模式。
K3 上注意 int 索引参与归约时的 dtype：i64 索引 + i32 值混算要先统一。
scan 类（cumsum/cummax）见 ops/09。

## 验证

```bash
python3 -m pytest python/tests/test_reduction_ops.py -q
python3 -m pytest python/tests/raw/test_raw_sum.py python/tests/raw/test_raw_argmax.py -q
```

golden：`torch.sum/mean/max/argmax`。数值验证的一个有效手法——**NaN-guard
buffer 切片探针**：`buf = torch.full((numel+guard,), nan); x = buf[:numel].view(...)`，
从 NaN 大 buffer 头部切出输入（保持精确 stride/specialization），任何 OOB 读
都会以 NaN 形式出现在输出里，可逐块扫描定位。

下一篇：[07-elementwise-fused.md](07-elementwise-fused.md)
