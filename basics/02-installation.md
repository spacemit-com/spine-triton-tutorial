# 02 — 安装：在 K3 板卡上装好 spine-triton

本章目标：一块 SpacemiT K3 板卡（riscv64 Linux，Python 3.12），从零装到
能跑第一个 kernel。主线只有一条：**pip 安装官方预编译 wheel**。章末的
进阶节讲另一条路——在 x86 开发机上交叉编译出 wheel，再拷到板上安装
（需要改源码或官方 wheel 不满足你时才用）。

## 2.0 开始前的检查

SSH 登上板卡，先确认三件事：

```bash
uname -m                 # 期望输出: riscv64
df -h /                  # 看根分区剩余空间（wheel 解包后 1GB 上下，
                         # 加上编译缓存，建议至少留 3GB）
```

三条都满足再继续。`python3` 不是 3.12 的话，wheel 装不上（文件名里的
`cp312` 标签就是 Python 版本约束）——先解决 Python 版本，或用 venv 指定
3.12 解释器。

建议使用虚拟环境（干净、好卸载、不污染系统 Python）：

```bash
python3 -m venv ~/spine-venv
source ~/spine-venv/bin/activate
```

后文所有命令都默认在激活的 venv 里执行。

## 2.1 系统依赖

编译器链接与运行时需要几个系统库（OpenMP、NUMA、SLEEF 等）：

```bash
sudo apt update
sudo apt install gcc g++ gdb libsleef-dev libnuma-dev libomp5 libgomp1 python3-dev
```

这些是编译 kernel `.so` 时链接的库（板上 JIT 编译会用板上的 g++），缺了
会在**运行时**报 `cannot open shared object file`——报错信息里缺哪个补哪个。

## 2.2 安装 Python 依赖与 triton wheel

SpacemiT 提供私有 PyPI 源（板卡可用的 torch 等包也从这里装）：

```bash
# 1) 先装 Python 依赖（torch 等）
pip install PyYAML sympy torch opencv-python pybind11 \
    --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

# 2) 装 spine-triton 本体（包名就叫 triton）
pip install triton \
    --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

**`--index-url` 一个都不能少。** 不带它，pip 会从公共 PyPI 拉到 NVIDIA GPU
版的 triton——能装上、能 import，但没有 `spine_triton` 后端，跑 kernel 时
报 `ModuleNotFoundError: No module named 'triton.backends.spine_triton'`
（这是新手第一大坑，见 2.4）。

装完确认版本与后端都在：

```bash
pip show triton | head -3          # Name: triton，版本形如 3.x.x+spacemit...
python3 -c "import triton.backends.spine_triton; print('backend OK')"
```

wheel 里有什么（了解即可，都在 site-packages 的 `triton/` 下）：

```
triton/                          # 前端：语言、编译器框架、JIT、缓存
  backends/spine_triton/         # K3 后端
    bin/                         # 编译器后端工具（RISC-V 目标码生成链路）
    lib/                         # 运行时库（加载并执行 kernel .so）
    compiler.py driver.py ...    # 后端入口、设备驱动
  language/extra/spine_raw/      # spine_raw eDSL（basics/05）
  language/extra/smt/            # 矩阵引擎接口（basics/06）
```

`bin/` 和 `lib/` 里的工具与库是**配套**的：编译器和运行时来自同一次构建，
版本互相咬合。**永远不要单独替换其中任何一个文件**——混搭二进制是排错
地狱的头号来源（症状千奇百怪：编译崩溃、加载失败、静默错值）。要升级就
整套 wheel 一起升（2.9）。

## 2.3 验证安装

写一个最小验证脚本（vec_add 的骨架在 basics/03 会精讲，这里只求跑通）：

```python
# verify_install.py — 最小安装验证
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())   # 激活 K3 CPU 后端


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


x = torch.rand(1000, dtype=torch.float32)
y = torch.rand(1000, dtype=torch.float32)
out = torch.empty_like(x)
add_kernel[(triton.cdiv(1000, 256),)](x, y, out, 1000, BLOCK_SIZE=256)

torch.testing.assert_close(out, x + y)
print("安装验证 PASS")
```

```bash
python3 verify_install.py
```

**第一次运行会比较慢**（十几秒到几十秒）：要在板上完成一次完整的 JIT
编译（tracing → 多级 IR 变换 → 生成 RISC-V 目标码 → 链接 `.so`）。编译
产物会进缓存目录（默认 `~/.triton/cache`），第二次运行同一段代码就是
秒回。慢 ≠ 卡死，耐心等第一次。

看到 `安装验证 PASS`，环境就绪。

两个常用可选环境变量（现在设不设都行，basics/04 详述）：

```bash
export SPINE_TRITON_DUMP_PATH=./ir_dumps   # 把编译各级中间产物落盘，排错用
export TRITON_CACHE_DIR=/tmp/tcache        # 换缓存目录（升级 wheel 后必做，2.9）
```

## 2.4 验证失败排错

| 症状                                                                    | 原因                                                                            | 处置                                                                                       |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `No module named 'triton.backends.spine_triton'`                      | 装到了公共 PyPI 的 GPU 版 triton（`pip install triton` 忘了 `--index-url`） | `pip uninstall triton` 后按 2.2 重装；`pip show triton` 看版本串里有没有 spacemit 标记 |
| `pip install triton` 直接找不到包                                     | 没带`--index-url`，或板子到内网源的网络不通                                   | 检查网络；命令按 2.2 抄全                                                                  |
| wheel 装不上：`not a supported wheel on this platform`                | Python 版本不是 3.12，或 pip 太老                                               | `python3 --version`；升级 pip                                                            |
| `OSError: ... cannot open shared object file`（libgomp / libnuma 等） | 2.1 的系统依赖没装全                                                            | 按报错缺的库补 apt 包                                                                      |
| import 时报`undefined symbol`                                         | wheel 与 torch/依赖包版本不配套，或混装过二进制                                 | 整套重装：uninstall triton + 按 2.2 顺序重来                                               |
| 编译很久后失败 / 磁盘写满                                               | 根分区空间不足（编译缓存+临时文件都写根分区）                                   | `df -h`；`TRITON_CACHE_DIR` 指到大分区；`TMPDIR=/tmp`                                |
| 跑通了但结果全错 / 行为诡异                                             | 旧缓存残留（之前装过别的版本）                                                  | 换缓存目录重跑：`TRITON_CACHE_DIR=/tmp/fresh python3 verify_install.py`（2.9）           |

## 2.5 跑仓库自带示例

wheel 只有运行环境；示例与测试脚本在源码仓库里（可选，clone 到板卡或
开发机上）：

```bash
git clone <spine-triton 仓库地址>
cd spine-triton

python3 python/examples/test_smt_mm.py        # 官方 mm demo（矩阵引擎 GEMM）
python3 python/examples/test_vec_add.py       # 最小逐元素
python3 python/examples/test_softmax.py       # softmax（04 章会精讲）
```

注意示例脚本自己会做 `CPUDriver` 激活（和 2.3 的脚本同款开头），板上
直接 `python3` 跑即可，**不需要设置任何"运行模式"环境变量**。想看编译
中间产物就加 `export SPINE_TRITON_DUMP_PATH=./ir_dumps` 再跑。

## 2.6 板上工作的两条纪律

1. **pytest 从可写目录跑**。仓库测试的 conftest 会往当前目录写结果文件
   （json 等），cwd 不可写时会出现 `exit=1 但所有用例都 PASS` 的假失败
   （日志尾部是 PermissionError traceback）。在板上跑测试套先
   `cd /tmp && python3 -m pytest <路径>/tests/... `，或确认 cwd 可写。
2. **磁盘紧张时先卸旧再装新**。旧版 triton 包可能占 1GB+；升级用
   `pip uninstall triton && pip install ...`，并且大文件操作时
   `TMPDIR=/tmp`（板上 /tmp 通常是 tmpfs，不占根分区）。

## 进阶：在 x86 上交叉编译 wheel，装到 K3

什么时候需要这条路：官方 wheel 版本不满足你（比如你按 basics/01 克隆了
仓库、改了自己想试的东西），或所在环境访问不了 PyPI 源。流程 = **x86 上
构建 riscv64 wheel → 拷到板上 → pip 安装**。构建一台普通 x86 Linux 开发机
即可，全程不需要板卡参与。

### 2.7 x86 交叉编译

准备三样东西（细节都在仓库 README 的 Build 节，这里只给全景）：

1. **源码**：clone 仓库并 `git submodule update --init --recursive`。
2. **LLVM/MLIR 工具链**：版本必须与 `triton/cmake/llvm-hash.txt` 记录的
   commit 一致（README 给了构建命令；也有预编译包可用）。版本不符是
   构建失败的第一大原因——报错通常在很后期，很难联想到是 LLVM 版本问题。
3. **两个预编译依赖包**（编译器后端包 + 运行时包，README Build 节有
   下载地址）：解包后的 install 目录作为构建脚本参数传入。wheel 会把
   它们的关键二进制打包进 `backends/spine_triton/{bin,lib}/`——所以
   "不要混搭二进制"的纪律从这里就开始了。

另外需要 riscv64 交叉工具链（脚本通过 `RISCV_ROOT_PATH` 环境变量找到
`riscv64-unknown-linux-gnu-g++`）。然后一条命令：

```bash
export RISCV_ROOT_PATH=/path/to/riscv-toolchain
export MAX_JOBS=$(nproc)                     # 并行编译数，默认 20

bash scripts/build_whl.sh \
    ${LLVM_INSTALL_DIR} \        # 第 1 步准备的 LLVM install 目录
    riscv64 \                    # 目标架构（也可以是 x86_64，产开发机自用包）
    <依赖包1的install目录> \
    <依赖包2的install目录>
```

产物：`build-wheel-riscv64/` 下的 `triton-*.whl`（数百 MB——里面打包了
整套编译器工具与运行时库）。构建时间视机器核数 20 分钟到 1 小时。

常见失败：

| 症状                                          | 原因                                                                                                                                         |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `RISCV_ROOT_PATH` 相关报错 / 找不到交叉 g++ | 工具链没准备或环境变量没导出                                                                                                                 |
| LLVM 相关的编译错误（很晚才炸）               | LLVM 版本与`llvm-hash.txt` 不一致                                                                                                          |
| 依赖包路径报错                                | build_whl.sh 的第 3、4 参数必须是解包后的 install 目录（含`bin/`、`lib/`、`include/`）                                                 |
| Python 头文件相关错误                         | 交叉编译需要板卡 Python（3.12）的头文件；构建脚本支持`SPINE_TRITON_EXTRA_CMAKE_ARGS` 透传额外 cmake 参数，细节见 `scripts/build.sh` 注释 |

### 2.8 安装自编译 wheel 到板上

```bash
scp build-wheel-riscv64/triton-*.whl k3-board:/tmp/

# 板上：
pip uninstall -y triton                 # 先卸旧（磁盘紧张，2.6 纪律 2）
TMPDIR=/tmp pip install /tmp/triton-*.whl
```

wheel 文件名自带 `linux_riscv64` 平台标签（build_whl.sh 打包时指定），
pip 直接接受。装完按 2.3 验证——**并且务必换一个全新缓存目录再验证**
（2.9），否则跑的可能是旧版编译产物。

### 2.9 升级纪律：换了 wheel 必须换缓存

编译缓存的 key **不包含编译器二进制的版本信息**。这意味着：升级 wheel
后，旧缓存里的 `.so` 会被继续命中——新版的修复/改动完全不生效，你以为
在测新版，实际跑的还是旧产物。历史上大量"修了怎么还错/没修怎么好了"
的困惑都来自这里。

```bash
# 升级（官方版）：
pip install -U triton --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

# 升级后第一件事：把旧缓存挪走（用 mv，不要 rm -rf——
# mv 是原子的，正在跑的进程不受影响；rm 大缓存慢且可能删到写了一半的目录）
mv ~/.triton/cache ~/.triton/cache.old-$(date +%m%d)
```

排查疑难问题时同理：**"换一个全新 `TRITON_CACHE_DIR` 重跑"是成本最低、
信息量最大的第一步**（basics/04 §4.2 会把它列进排查军规）。

记录你环境的版本指纹（报问题、对比结果时都要带上）：

```bash
pip show triton | grep -i version
python3 -c "import triton; print(triton.__version__)"
```

下一章：[03-quickstart.md](03-quickstart.md) —— 第一个 kernel 与 Triton 思维模型。
