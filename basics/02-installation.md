# 02 — 安装与构建

按“用现成的 → 自己编 x86/RPC 版 → 自己交叉编 riscv64 版”三档递进。

## 0. 系统依赖

```bash
apt update
apt install gcc g++ gdb libsleef-dev libnuma-dev libomp5 libgomp1 python3-dev
pip install PyYAML sympy torch opencv-python pybind11 \
    --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

Python 3.12（venv 里需要 `pybind11`，setup.py 构建时用到）。

## 1. 预编译 wheel（最快上手）

```bash
pip install triton --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

装完验证：

```bash
python -c "import triton; print(triton.__version__)"
```

> 板卡（riscv64）与 x86 主机装的是**不同架构的 wheel**，别混。
> wheel 里捆绑了 `spine-opt` / `llc` / `mlir-translate`（`triton/backends/spine_triton/bin/`）
> 和 `libspert.so`（`triton/backends/spine_triton/lib/`）——排错时先确认这几个二进制的代际。

## 2. 源码构建：x86 + QEMU RPC 版

在 x86 开发机上编译、把 kernel 送到 QEMU 里跑的开发形态。

```bash
cd spine-triton
git submodule update --init --recursive
bash scripts/build_x86_rpc.sh ${LLVM_INSTALL_DIR} ${SPINE_MLIR_INSTALL_DIR} ${SPINE_RUNTIME_INSTALL_DIR}
```

三个入参：

| 参数 | 说明 |
|---|---|
| `LLVM_INSTALL_DIR` | 预编译 LLVM/MLIR。版本必须与 vendored triton 匹配（见 `triton/cmake/llvm-hash.txt`；LLVM 版本不合会出 C++ API 编译错，如 `Resource::getName()` const 性差异） |
| `SPINE_MLIR_INSTALL_DIR` | spine-mlir 发行包（含 `bin/spine-opt`、`llc`、`mlir-translate`），GitHub releases 或内部 CI 产物 |
| `SPINE_RUNTIME_INSTALL_DIR` | spine-runtime 发行包（`include/spert.hpp` + `lib/libspert.so`），必传 |

产物在 `build-x86_64/`。构建脚本会：vendor spert 头文件 → 对 vendored triton
打 `patch/*.patch` → `setup.py install --prefix=build-x86_64`。

配套 QEMU RPC server 来自 `spine-triton-rpc-runtime` 仓：启动 `run_qemu.sh` 后
**用真实 TCP connect 等端口 9999 就绪**再跑测试（别 grep `/proc/net/tcp`）：

```python
import socket, time
while socket.socket().connect_ex(("127.0.0.1", 9999)) != 0:
    time.sleep(0.2)
```

RPC 模式运行 kernel 时设置：

```bash
export SPINE_TRITON_RPC_HOST=127.0.0.1   # 设了即 RPC 模式；x86 上 import triton 必须设，否则 dlopen RISC-V libspert 失败
export SPINE_TRITON_RPC_PORT=9999
export SPACEMIT_EP_QEMU_SET_CORE_ARCH=0xA064   # K3
```

> **x86 上任何 `import triton` 的脚本都必须设 `SPINE_TRITON_RPC_HOST`**：
> 包里的 libspert 是 RISC-V 二进制，x86 无法 dlopen；该变量让 driver 跳过加载。
> 症状是文件明明存在却报 `cannot open shared object file`（异架构 ELF 签名）。

## 3. 源码构建：riscv64 交叉编译版（上板）

需要：SpacemiT 交叉工具链（`RISCV_ROOT_PATH`）、riscv 版 Python 头文件、
riscv 静态 zlib（vendored triton 无条件链 `-lz`，工具链 sysroot 里没有）。

```bash
source <venv>/bin/activate    # pybind11 就绪的 py3.12 venv
export RISCV_ROOT_PATH=/path/to/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPINE_TRITON_EXTRA_CMAKE_ARGS="\
  -DPython3_INCLUDE_DIR=/path/to/riscv-python3.12/include/python3.12 \
  -DCMAKE_SHARED_LINKER_FLAGS=-L/path/to/riscv-deps/zlib/lib \
  -DCMAKE_EXE_LINKER_FLAGS=-L/path/to/riscv-deps/zlib/lib"
bash scripts/build.sh ${LLVM_INSTALL_DIR} riscv64 ${SPINE_MLIR_INSTALL_DIR} ${SPINE_RUNTIME_INSTALL_DIR}
```

交叉编译三大坑（都踩过，逐条对照）：

1. **Python 头**：host 解释器是 x86 Python，但 `pyconfig.h` 必须用 riscv 版的
   （Linux python module 不链 libpython，只用头文件即可）。
2. **zlib**：`triton/CMakeLists.txt` 无条件 `-lz`；用交叉编译的静态 `libz.a`
   经 `-L` 注入，静态链入后板上无运行时依赖。
3. **旧 build 目录污染**：`CMakeFiles/*/CMakeC(CXX)Compiler.cmake` 里存的旧
   工具链路径每次 reconfigure 会覆盖回 `CMAKE_AR/RANLIB`——只 sed CMakeCache
   无效，必须把 build 目录下所有含旧路径的文件一起改，且改完重跑一次构建
   （ninja 可能在重生成 rules.ninja 前就按旧规则派发任务）。

构建完成后检查产物架构没有混装 x86 工具：

```bash
file build-riscv64/triton/backends/spine_triton/bin/*   # 全部应是 RISC-V ELF
```

## 4. FlagTree 形态的 wheel（可选）

FlagTree 是 spine-triton spacemit 后端的另一载体（`FlagTree/third_party/spacemit`）。
交叉 wheel 构建 + 上板三步前处理：

```bash
bash third_party/spacemit/scripts/build_flagtree_riscv64.sh   # 全部路径 env 可覆盖
# 上板前：
python3 -m wheel tags --platform-tag linux_riscv64 --remove <wheel>   # retag
# strip（debug_info 会让 wheel 膨胀到 2GB+）：wheel unpack → strip 全部 ELF → wheel pack
# 板卡根分区紧张时先 pip uninstall 旧 triton 再装（TMPDIR=/tmp）
```

注意 wheel 捆绑的 spine-opt 代际可能落后于当前源码（CI 资产按 pipeline 落盘，
新 pipeline 不一定构建的是新代码）——上板后先用一个含 `xsmt_async.grid` 的最小
IR 探针验证 spine-opt 是否能 parse，字符串 grep 二进制不可靠。

## 5. 部署后自检清单

```bash
python - <<'EOF'
import os
os.environ.setdefault("SPINE_TRITON_RPC_HOST", "127.0.0.1")  # x86 必设
import triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
print("triton", triton.__version__)
EOF
```

再跑一个最小 kernel（见 [03-quickstart.md](03-quickstart.md) 的 vec add）。
换/升级任何二进制（spine-triton-opt、spine-opt、libspert）之后：

- **必须换 `TRITON_CACHE_DIR`**——cache key 不含编译器版本，旧 `.so` 会被直接复用，
  新修复完全不生效（用 `mv` 换目录，不要 `rm -rf`）。
- libspert 升级时把 lib 目录里旧版本**移走**只留最新——版本选择按 basename 长度
  max，等长版本号（0.6.N）平局取决于 glob 顺序。

下一篇：[03-quickstart.md](03-quickstart.md)
