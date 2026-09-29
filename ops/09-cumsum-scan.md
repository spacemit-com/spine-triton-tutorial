# Op 09 — Cumsum / Scan

`y[i] = Σ_{j≤i} x[j]`。prefix scan 看着简单，在 K3 编译链上却是坑密度最高
的算子族：Hillis-Steele lowering 的中间 buffer、行长与 vscale 的边界、
runtime 协程栈大小，三层都踩过。cumsum/cumprod/cummax/cummin、以及内部复用
scan 的 repeat_interleave/index_put/nonzero/randperm 都属本章范围。

## 标准写法

```python
@triton.jit
def cumsum_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)               # 每 program 一段（或一行）
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.).to(tl.float32)
    y = tl.cumsum(x, axis=0)             # 块内 scan
    # 跨块前缀：单 program 串行，或 host 传入前段总和，或多段两趟法
    tl.store(y_ptr + offs, y, mask=offs < N)
```

行 scan（`tl.cumsum(x_2d, axis=1)`）每行独立，grid=(n_rows,)；FlagGems 的
`reduce_then_scan` 模式（先每块归约再块间 scan 再块内 scan）是长序列的通用
组织。spine_raw 版本：`test_raw_cumsum.py`、`test_raw_cumsum_vec.py`
（向量移位 + 前缀和的手工 scan）。

## K3 关键点：三层边界

### 1. runtime 协程栈（libspert 版本敏感）

Hillis-Steele scan 每 round 分配 `(TILE + 2^r) × elemSize` 的中间量；
libspert 0.6.0 协程栈默认 **16KB 不够**（int64 TILE=1024 需 ~128KB）→
sp 越界 SIGSEGV。**0.6.3 起 512KB**，cumsum/multinomial/
repeat_interleave/index_put_impl/cat 族的 guest 崩溃随之消失。板上官方 0.6.3
release 默认 64KiB，需要 `SPERT_STACK_BYTES=524288` 对齐（该变量支持 4–512K）。
**遇到 scan 族执行期段错误，先查 libspert 版本和栈配置，再怀疑编译器。**

### 2. 行长 vs vscale（编译期断言族）

与 reduction 同源（ops/06 表）：行长 % vscale != 0 →
`ConvertToScalableVector.cc:55` 断言（移交状态）；行长 == vscale →
`ConvertVectorToSCFPass.cc:962` 断言（已修 `c63415b`）。scan kernel 的
`reduce_then_scan_row` 同一 kernel 两种断言都撞过。行长 1041→BLOCK 2048 类
还会撞 llc SplitVectorResult。**这些崩溃只在 fresh 编译时出现——cache 命中
会掩盖**（行清缓存后必现是它们的签名）。

### 3. 数值假象：0.6.0 grid 丢弃

int32 BLOCK=16 的 `tl.sum` 恒差 -16、cumsum 输出 99.9% 垃圾——分别是
poison-init 垃圾 lane（已修 `d04296f`）与 grid>512 丢弃（0.6.3 修复）。
引用任何 scan 族历史失败数字前，先确认批次的 libspert 版本。

## 已知残留

- **cumprod 仍崩**：0.6.0 死@dim6 → 0.6.3 死@dim16（同
  `reduce_then_scan_root_scan_kernel_row`），spine-mlir 移交件。
- **int64 负值路径**：repeat_interleave 的 2F 是 cumsum int64 负值受害者。
- **cast560 家族**：整数 dtype（i1/i16/i32/i64）的 scan/归约 kernel 可能撞
  `ConvertSpeStructToVector` 对整型 `cast<FloatType>` 的断言（trace/cumsum
  int 用例族）——判据：fp16/fp32 全过、int dtype 全崩。

## 复合算子的分段探针法（重要方法论）

repeat_interleave 数值失败曾长期误归因 scan lowering。正确做法：
**分段探针**——`repeat_interleave_self_tensor = cumsum(scan) +
repeat_interleave_tensor_kernel + index_select`，三段分开跑，前两段 len 15~320
全对，失败全在 index_select（其 dim_compress 路径 grid>512）。
复合算子失败先拆段定位，不要直接怀疑最深的那层。

## 验证

```bash
python3 -m pytest python/tests/raw/test_raw_cumsum.py python/tests/raw/test_raw_cumsum_vec.py -q
# FlagGems: tests/test_scan_ops.py（cumsum/cumprod/...）
```

golden `torch.cumsum`；int 输入注意 dtype 保持（cumsum(int32)→int32 的溢出
语义与 torch 对齐）。板端从 /tmp 跑 pytest（conftest 写结果文件的权限坑）。

下一篇：[10-quantization.md](10-quantization.md)
