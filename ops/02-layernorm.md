# Op 02 — LayerNorm

`y = (x - E[x]) / sqrt(Var[x] + eps) * w + b`，按行（最后一维）归一化。
Transformer 每层两次，是典型的 **memory-bound 归约 + 逐元素** 复合算子。

## 标准 Triton 写法（fused forward）

（来源：`python/examples/test_layernorm.py`，即上游 05-layer-norm 教程）

```python
@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                 # 每 program 一行
    Y += row * stride
    X += row * stride

    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)   # f32 累加器
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        mean += x
    mean = tl.sum(mean, axis=0) / N

    # 二趟：方差
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += (x - mean) * (x - mean)
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # 三趟：归一化 + affine
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)      # 缓存 mean/rstd 给 backward
```

grid = `(n_rows,)`；行短且 `N ≤ BLOCK_SIZE` 时可单 tile 一趟完成
（load 全行进寄存器，省两趟访存）。backward 用缓存的 Mean/Rstd 做
dx/dw/db 三个归约，dw/db 是跨行归约（grid 按列分块 + 行循环）。

## K3 关键点（port 层四连坑，全部实战修过）

1. **`tl.sum` 不提升 dtype**：fp16 tile 的 sum 仍是 fp16 部分和，mean/var 累加
   超容差（fp16 下 abs 误差可到 0.03）。**load 后必须 `.to(tl.float32)`** 再进
   归约——f32 累加器参与时 Triton 类型提升才自动安全。
2. **`make_block_ptr` 的 base 不能是 None**：`W is None` 的内联守卫只护 load；
   循环外预建 desc 时 `tl.make_block_ptr(base=None)` 直接前端报
   `'constexpr_type' object has no attribute 'is_ptr'`。可选 weight 的 kernel
   要么两条路径分开建 desc，要么 host 侧保证非 None。
3. **block-ptr store 无隐式 cast**：普通指针 `tl.store(ptr+i, f32val)` 自动 cast，
   block_ptr 版本严格检查（"Block element type(fp16) and value element type(fp32)
   mismatch"）。f32 数学结果必须显式 `.to(Y.type.element_ty)`。
4. **constexpr 整数没有 `.to()`**：`(2).to(...)` 前端报 `'int' object has no
   attribute 'to'`；写 `a * a` 或先包成 tensor。

修复后参考成绩：FlagGems 端口 layernorm 40/40 PASS（此前 24F/16P，其中 15/24
是编译 bug 而非精度问题——**逐例拆日志才能拿到真实构成**，`re.split(r'_{5,}
(test_\w+\[[^\]]+\]) _{5,}')` 切失败块）。

## spine_raw 版本

无 weight/bias 的 1D layernorm 三趟 reduce 版见
[basics/05](../basics/05-spine-raw-edsl.md)（`python/tests/raw/test_raw_layernorm.py`）。
raw 版价值：

- 方差用 `E[x²] - mean²` 单趟形式，避免 padding 0 的 `(0-mean)²` 膨胀；
- f16 load → f32 全程标量/向量混合算术 → f32 输出，无 cast 截断点;
- 想 fuse（如 layernorm+residual、layernorm+量化）时不受 tl 层 API 限制。

带 weight/bias 与 backward 的 raw 版本参考 `test_raw_group_norm.py` /
`test_raw_instance_norm.py` / `test_raw_batch_norm.py` 的多行组织方式
（默认 path 无 program_id，grid=(1,) 外层串行循环遍历行）。

## 验证

```bash
python3 python/examples/test_layernorm.py
python3 -m pytest python/tests/test_norm_ops.py -q
python3 -m pytest python/tests/raw/test_raw_layernorm.py -q
```

golden 对照 `torch.nn.functional.layer_norm`；fp16 用 rtol/atol 1e-2 量级、
fp32 用 1e-4~1e-5。注意判别假失败：旧批次里 fp32/backward 的“数值失败”可能
是 libspert 0.6.0 grid 丢弃假象（引用旧数字前核对批次 runtime 版本）。

## 相关算子

- RMSNorm（ops/03）：去掉 mean 中心的简化版，LLM 主流。
- GroupNorm / InstanceNorm：多一个分组维度；注意其 backward 的跨层循环携带
  累加器结构曾撞 bufferization RaW（spine-mlir `SCFLoopBufferizationPreprocessing`
  家族问题，移交件）。

下一篇：[03-rmsnorm.md](03-rmsnorm.md)
