# Op 04 — Softmax

`y[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))`，按行归一化。
attention 的核心非线性步骤；数值稳定版必须减行最大值（三趟：max → sum → 归一）。

## 标准 Triton 写法（row-parallel）

（来源：`python/examples/test_softmax.py`）

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, in_stride, out_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)                        # 每 program 一行
    row_start_ptr = input_ptr + row_idx * in_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row = tl.load(row_start_ptr + col_offsets,
                  mask=col_offsets < n_cols, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)         # 数值稳定
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_idx * out_stride + col_offsets,
             softmax_output, mask=col_offsets < n_cols)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)       # 一行装进一个 block
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                              BLOCK_SIZE=BLOCK_SIZE)
    return y
```

行太长装不下单 tile 时走 **multi-tile 路径**：对 max/sum 用 f32 累加器分块扫
（`m = tl.full([BLOCK], -inf, tl.float32)`、`z/l` 在线更新），再从后往前走
tile 做归一化——FlagGems general kernel 用 `prev_multiple_of(N, TILE_N) =
cdiv(a,b)*b - b`（严格小于 a 的最大倍数）定位尾块起点，N==TILE_N 时为 0
全覆盖。

## K3 关键点

1. **fp16 撞 `tl.exp` dtype 检查（第一大坑）**：`tl.exp/log/sqrt` 等只收
   fp32/fp64（`triton/language/math.py:_check_dtype`，上游故意设计，GPU 同败）。
   ONE_TILE 快路径若保留 fp16 原样喂 exp 直接 CompilationError。三种解法：
   - kernel 内显式 `.to(tl.float32)`（**不允许改上游 FlagGems 算子源码时不可用**）；
   - **heuristic 路由**（生产采用）：`softmax_heur_one_tile_per_cta` 扫 launch
     args 里的 tensor dtype，fp16/bf16 返回 False → 走 multi-tile 路径（其 m/z
     本来就是 f32 累加器，天然安全且数值更稳）；backward 无 exp，拆独立
     heuristic 维持 ONE_TILE 语义；
   - 全局给 math.py 加自动提升——不优雅，被否决，别走这条路。
2. **归约 dtype**：`tl.max/tl.sum` 对 fp16 输入返回 fp16，长行累加超容差；
   multi-tile 路径全程 f32 累加器规避。
3. **行级并行**：grid=(n_rows,) 无 grid>512 压力（旧 runtime 除外）；行内
   BLOCK 是 2 的幂，行长非 vscale 整倍数时的 scalable 转换边界问题已在
   spine-mlir 修复（`c63415b`/`d04296f` 代际起），旧二进制上“BLOCK=16 的
   int32 sum 恒差 -16”之类症状是 in_bounds 误推断 + poison init 垃圾 lane。

修复后参考成绩：test_softmax 112/112 PASS（独立 QEMU + 独立 cache）。
回归审计方法：全量 grep `tl.exp/log/...` 调用点逐一确认有前置 cast 或 f32
提升（tensor-op-scalar 会提升到 f32、COMPUTE_FP32 门控、philox f32 源都算安全）。

## spine_raw 版本

默认 path 有 `tle.vexp`，可做无 dtype 检查的向量级 softmax
（`python/tests/raw/test_raw_softmax.py`、`test_raw_log_softmax.py`）。
组织方式：grid=(1,)、外层循环串行遍历行、每行三趟（max/sum/scale），tail 用
空 range loop。**注意 loop env 泄露类 dominance 崩溃**（2-pass softmax 是历史
触发器，现已修，见 basics/05 坑 3）。

但要泼一盆冷水（实测结论）：**decode 场景小 K 的 fused softmax kernel 跑不过
torch.softmax**——K=33~52 时比 Python 版慢 1.4x。原因：default path 无
program_id 没并行、tiny data 下 per-op 开销 >> 计算、torch.softmax 是优化过的
C++ op。decode softmax 的 22% 开销是 **Python dispatch**（f32 intermediate +
类型转换 + tensor alloc），kernel fusion 消不掉 dispatch 本身。要赢需要
LLVM-direct + program_id 并行 + 多项式 exp，成本远超收益。

同理，**fuse QKT+softmax+AV 成单 kernel 已被实测否决**：Python softmax+mask+scale
仅占 prefill wall 的 ~0.8%（0.9ms/layer × 28 = 25.4ms），见 ops/08。

## 验证

```bash
python3 python/examples/test_softmax.py
python3 -m pytest python/tests/raw/test_raw_softmax.py -q
```

golden `torch.softmax(x.float(), dim=-1)`；fp16 rtol/atol 1e-2~1e-3。
**改 heuristic/语言层 Python 后必须换 TRITON_CACHE_DIR**——cache key 不含
language 模块改动，旧 .so 会掩盖编译错误。

下一篇：[05-gemv.md](05-gemv.md)
