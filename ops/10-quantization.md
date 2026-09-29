# Op 10 — Quantization（per-token-group-quant-fp8）

量化算子是 LLM 推理栈的“下一层地基”（FP8 KV cache、权重量化）。以
**per_token_group_quant_fp8** 为案例：`x[M,K]` 按 (行, group_size) 分组，
每组独立 scale 量化到 fp8。它把 K3 上 dtype 生态的现状（bf16/fp8/int8）
暴露得很完整。

## 数学与 kernel 模式

```
对每个 (row, group g):
  amax   = max(|x[row, g·gs : (g+1)·gs]|)
  scale  = amax / FP8_MAX          # FP8_MAX = 448 (e4m3)
  q      = clamp(x / scale, -FP8_MAX, FP8_MAX).to(fp8)
输出: q[M,K] (fp8) + scale[M, K/gs] (f32)
```

Triton kernel 是“二维 grid + 组内归约”的标准组合：

```python
@triton.jit
def ptg_quant_kernel(x_ptr, q_ptr, s_ptr, M, K, gs,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    gid = tl.program_id(1)                 # 每 program 一个 (row, group)
    offs = gid * gs + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < gs
    x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=0)
    scale = amax / 448.0
    scale = tl.where(scale == 0, 1.0, scale)      # 全零组防除零
    q = tl.clamp(x / scale, -448.0, 448.0)
    tl.store(q_ptr + row * K + offs, q.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row * (K // gs) + gid, scale)
```

要点就是 ops/06 的归约模式 + 逐元素 cast；数值上注意 amax=0 组与
clamp 边界（±448 恰好可表示）。

## K3 dtype 生态现状（写量化算子前必读）

### bf16：向量路径不可用

- `llvm.riscv.vle/vse.nxvNbf16` 在当前 mattr（无 Zvfbfmin/Zvfbfma）下无合法
  指令 → llc `Scalarization of scalable vectors is not supported` SIGABRT。
  6 行最小复现存在（repro 7）。**启用 bf16 前必须先修工具链**（mattr 加扩展
  需硬件确认，或降为 i16 向量 + bitcast）。
- 测试环境 `bf16_is_supported=False`，通用 dtype 表 FLOAT_DTYPES=[fp16,fp32]
  ——**通用参数化测试里根本没有 bf16**。
- 但 per_token_group_quant_fp8 **自带 dtype 表 DTYPES=[bfloat16]**，直接撞上
  bf16 llc fatal（96 例失败全属此族，与 fp8 无关——spine-opt e2e 输出的
  scalable 向量只有 `vector<[N]xbf16>`（崩）与 `vector<[N]xf32>`（合法）两种，
  llc abort 的原因是 **bf16 元素类型**，fp8 只是输出 cast 目标）。

**dtype 参数判读纪律**：`dtypeN` 的映射因文件而异（自带表 vs 通用表），
判 bf16 相关性必须先读该文件的 parametrize，不能沿用其他文件的惯例
（FlagGems 板上 dtype1=fp32，FlagTree sdpa 里 dtype1=bf16）。

### fp8

fp8（e4m3/e5m2）作为**存储格式**在量化输出侧使用；K3 无 fp8 计算通路，
kernel 内的除法/clamp 都在 f32 做，最后 cast。cast 到 fp8 的 lowering 走
标量路径时可用，向量路径同受 bf16 类工具链限制——大 shape 先小批量验证。

### int8

int8/int32 整型路径可用，但注意 **cast560 家族**：整型元素类型的 linalg
在 `ConvertSpeStructToVector` 撞 `cast<FloatType>` 断言（ops/09 同源）。
int8 量化 kernel 里混 i64 索引累加时先确认二进制代际。

## 实战建议

1. 当前环境下量化算子的可行输入 dtype 是 **fp16/fp32**；bf16 输入一律先
   `.to(torch.float16/32)` 再进 kernel（host 侧转换），别指望 kernel 内
   bf16 向量 load。
2. 自带 dtype 表的测试文件（quant/reshape_and_cache/mhc_ops 等）是 bf16
   失败的重灾区，triage 时先归类到“bf16 平台豁免”，别误报成 kernel bug。
3. dequant/反量化验证：`x_recover = q.to(f32) * scale`，与原始 x 比相对误差
   （量化本身有损，rtol 按量化步长设定，不要用 1e-5 级容差）。

## 验证

```bash
# FlagGems: tests/test_quant.py -k per_token_group_quant_fp8
# 环境允许时（fp16/fp32 输入变体）：
python3 -m pytest test_quant.py -q -k "fp16 or fp32"
```

golden：参考实现的 amax/scale/clamp 逐步对照（先比 scale tensor 再比 q，
scale 全对而 q 错 → cast/clamp 层；scale 就错 → 归约层）。

---

至此 10 个算子完结。回到 [README](../README.md) 查看学习路线。
