# 03 — 快速上手：第一个 kernel

## Hello, vec add

（来源：`spine-triton/python/examples/test_vec_add.py`）

```python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

# spine-triton 是 CPU 后端，必须显式激活 CPUDriver
triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                # SPMD：每个 program 处理一段
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                # 越界保护
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(98432, dtype=torch.float32)
    y = torch.rand(98432, dtype=torch.float32)
    out = add(x, y)
    torch.testing.assert_close(out, x + y)
    print("PASS")
```

与 GPU Triton 的差别只有两点：

1. `set_active(CPUDriver())`——没有 CUDA driver，tensor 就在主机内存里。
2. tensor 不需要 `.cuda()`；`device` 用 CPU。

运行（RPC 形态需先起 QEMU server，见 basics/02；native 板上直接跑）：

```bash
export SPINE_TRITON_RPC_HOST=127.0.0.1 SPINE_TRITON_RPC_PORT=9999
export SPACEMIT_EP_QEMU_SET_CORE_ARCH=0xA064
python vec_add.py
```

## grid / program 语义

- `grid=(n,)` 启动 n 个 program；K3 上 program 由 spine-runtime 的 tile 调度器
  分发到 worker 协程。**libspert < 0.6.3 时 grid > 512 会被引擎拒绝并静默丢弃**
  （输出=未初始化内存）；≥0.6.3 已支持自动分块，无此限制。
- 每个 program 内部是“块级”编程模型：`tl.arange(0, BLOCK)` 得到向量，
  编译器负责映射到 RVV scalable vector。

## dtype 三条军规

1. **f16 输入、f32 累加**：`tl.load` 之后先 `.to(tl.float32)` 再进 reduce/dot；
   直接 f16 累加在长归约下必超容差。`tl.dot(f16, f16)` 的输出天然是 f32。
2. **`tl.exp/log/sqrt` 等 math 函数只收 fp32/fp64**：对 fp16 tile 直接调用会
   `ValueError: Expected dtype ['fp32','fp64'] but got fp16`（这是 Triton 上游
   的故意设计，GPU 同样报错）。写法：`tl.exp(x.to(tl.float32))`。
3. **bf16 暂不可用向量路径**：当前工具链 bf16 的 vle/vse 无合法指令
   （缺 Zvfbfmin/Zvfbfma），llc 会 abort。测试环境的 FLOAT_DTYPES 一般是
   `[fp16, fp32]`；遇到“dtype1”之类参数先确认它实际是哪个 dtype，别想当然。

## constexpr 与特化

- `BLOCK_SIZE: tl.constexpr` 参与编译期展开；改 constexpr 值 = 重新编译。
- 标量参数默认会被**值特化**：值为 1 的 i32 参数会被当 constexpr，导致依赖
  它的循环在 TTIR 阶段整个消失。需要运行期真值时用
  `@triton.jit(do_not_specialize=["M", "NK"])`——凡是 host 侧算好再传进去的
  size/块数（如 `NK = M // 32`）都应加上，否则 `NK=1` 这类值会被特化掉。
- `constexpr` 包装的值不是普通 Python 对象：对它调 `.to()`、直接参与
  `binary_op_type_checking` 都会炸（`'constexpr_type' object has no attribute ...`），
  先 `isinstance(x, tl.constexpr)` 解包 `.value`。

## 跑仓库自带示例

```bash
cd spine-triton
export SPINE_TRITON_DUMP_PATH=./ir_dumps   # 可选：dump 每级 IR
python3 python/examples/test_vec_add.py
python3 python/examples/test_softmax.py
python3 python/examples/mm_block_ptr.py
python3 python/examples/test_smt_mm.py     # smt 矩阵引擎 GEMM
```

pytest 套件：

```bash
python3 -m pytest python/tests/test_blas_ops.py -x -q      # mm/bmm 家族
python3 -m pytest python/tests/raw/ -x -q                  # spine_raw eDSL 套件
```

## 第一次失败时的排查顺序

1. `CompilationError` 带 Python 源码定位 → **前端 tracing 层**（dtype 军规、
   constexpr、API 误用）。
2. rc=134、无 Python 栈 → **spine-opt / llc 层**；开 `SPINE_TRITON_DUMP_PATH`
   拿 `*.linalgdir` 离线喂 `spine-opt` 复现（basics/04）。
3. `load kernel failed` / 段错误在执行期 → **runtime 层**：`nm -D --undefined-only kernel.so`
   对照 libspert 导出符号；确认 libspert 版本（grid>512、栈大小都随版本变化）。
4. 数值不对但不崩 → 先换**新 cache 目录**重跑（排除旧 `.so` 缓存），再查
   runtime 版本（0.6.0 的 grid 丢弃会产生 99.9% 错值的“未初始化内存”签名）。

下一篇：[04-running-and-debugging.md](04-running-and-debugging.md)
