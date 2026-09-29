# 04 — 运行形态、调试工具与坑清单

## 两种运行形态

### Native（K3 板卡）

riscv64 wheel 装板上，`import triton` 时 driver 直接 `CDLL(libspert, RTLD_GLOBAL)`，
kernel JIT 编译 + 本机执行。板上 pytest 注意：**从 /tmp 跑**——cwd 在只读/无
写权限的 NFS 目录时，conftest 写结果文件会 PermissionError，导致 `exit=1 但用例
全 PASS` 的假失败（摘要行在日志中间，别只看尾部）。

### QEMU RPC（x86 开发机）

x86 上交叉编译出 riscv64 `.so`，经 TCP 送到 QEMU guest 里的 rpc server 执行。
启动流程（spine-triton-rpc-runtime 仓）：

```bash
bash run_qemu.sh            # 或 CI 脚本 start_rpc_server.sh
# 等端口就绪：真实 TCP connect，spine-runtime 0.6.x 约 2s 绑定
python - <<'EOF'
import socket, time
while socket.socket().connect_ex(("127.0.0.1", 9999)) != 0:
    time.sleep(0.2)
print("rpc ready")
EOF
```

两个探测误区：

- **不要 grep `/proc/net/tcp` 的 `0000270F`**——INADDR_ANY 绑定显示为
  `00000000:270F`（中间有冒号），连续 8 位 hex 永远匹配不上。
- **server stdout 写文件时被 glibc 全缓冲**，accept 循环不 flush，日志空 ≠ 没
  起来；stderr（perror/bind 失败）才是无缓冲、可信的。

RPC 三大环境限制（测试侧规避，不是编译器 bug）：

| 限制 | 症状 | 规避 |
|---|---|---|
| 单 buffer 上传 2GB 上限（`ctypes.string_at` 对 >2^31 抛 `SystemError: Negative size`） | 大输出 tensor 的算子直接抛错 | 缩小 shape |
| client 非线程安全（多线程共享单 socket） | `seq mismatch` 或直接杀 server | 单线程跑 |
| QEMU guest 内存上限（默认 128MB，启 qemu 未传 `-m`） | 大 buffer alloc_memory 失败可杀 server | 起 qemu 时加 `-m` |

RPC 超时：单 kernel 仿真执行 >600s 会 `TimeoutError`（失败时间高度集中在
~602s 是签名）。慢 kernel 用 `SPINE_TRITON_RPC_TIMEOUT=7200` 调大。QEMU 仿真
很慢（大 shape kernel 单次 20min–2h 正常），长批次用“两遍法”：第一遍负责编译
（.so 进共享 cache），第二遍缓存命中纯执行。

## 环境变量总表

| 变量 | 作用 |
|---|---|
| `SPINE_TRITON_RPC_HOST` / `_PORT` | RPC 模式开关与地址（x86 host 必设 HOST，否则 dlopen RISC-V libspert 失败） |
| `SPINE_TRITON_RPC_TIMEOUT` | RPC 单 kernel 超时秒数（默认 600） |
| `SPINE_TRITON_RPC_THREADS` | RPC guest worker 线程数（默认 8） |
| `SPACEMIT_EP_QEMU_SET_CORE_ARCH` | 编译 target 的 arch id（K3 = `0xA064`） |
| `SPINE_TRITON_DUMP_PATH` | 每级 IR dump 目录（ttir / linalgdir / ll / s） |
| `TRITON_CACHE_DIR` | 编译缓存目录（默认 `~/.triton/cache`） |
| `SPINE_MLIR_OPT_PATH` | 覆盖 wheel 捆绑的 spine-opt（A/B 二进制对照必备） |
| `PROTON_KERNEL_CAPTURE=1` | 编译期在 `_launch` 插 kernel 计时钩子（CPU proton） |
| `PROTON_OUTPUT` | proton dump 输出（`.json` Chrome Trace / `.hatchet`） |

## IR dump 与离线复现

```bash
mkdir -p ./ir_dumps
SPINE_TRITON_DUMP_PATH=./ir_dumps python my_kernel.py
ls ./ir_dumps/<kernel>/    # *.ttir  *.linalgdir  *.ll  *.s
```

`<kernel>.linalgdir` 就是 spine-opt 的输入，可离线复现编译失败：

```bash
spine-opt x.mlir --spine-triton-e2e-pipeline="enable-always-tls=1 enable-fuse-group=false" -o /dev/null
echo rc=$?    # rc 必须在下一行立刻捕获：命令替换/管道都会重置 $?
```

（两个 pipeline flag 现均为兼容性 no-op，带着与否结果相同。）

定位单 pass 崩溃：`spine-opt x.mlir --mlir-print-ir-after-all` 抓最后一个完成
的 pass，从其 dump 提取 IR（注意嵌套 pass dump 只打印 func，需手工补 module 头
和 `memref.global`/别名）做单 pass 隔离复现。

排查数值问题要 **dump 全链路逐层 grep**（linalg → ll.mlir → .ll），单层看不出
mask 丢弃这类跨层 bug。

## 缓存管理（最容易踩的坑）

**cache key 不含编译器二进制版本**。换/升级 spine-triton-opt、spine-opt、
libspert 或改 language/ 下 Python 模块后，旧 `.so` 会被直接复用，修复完全不
生效——大量“已修但持续失败”的假象都源于此。

换缓存用 `mv`，不要 `rm -rf`（原子、in-flight 编译安全；rm 大缓存可能删掉
正在写的 key 目录导致进程崩）：

```bash
mv /root/.triton/cache /root/.triton/cache.$(date +%m%d)
```

单发探针用独立 `TRITON_CACHE_DIR=/tmp/probe_cache_xxx` 最干净。

判别“结果变化是环境还是编译产物”：复跑前后对比缓存 `.so` 的 mtime
（`find $CACHE -name '*.so' -newermt ...`）——0 个重编而结果翻转 ⇒ 问题在
runtime/环境。

## 失败分层判别

| 签名 | 层 | 下一步 |
|---|---|---|
| `CompilationError` + Python 源码定位 | 前端 tracing | dtype/constexpr/API 检查 |
| rc=134、无 Python 栈 | spine-opt/llc | linalgdir 离线复现 + `--mlir-print-ir-after-all` |
| `LLVM ERROR:`（无 "error:" 前缀） | llc | 提取 .ll 单独喂 llc |
| `server error: load kernel failed` | runtime 加载 | `nm -D --undefined-only kernel.so` 对照 libspert 导出 |
| 执行期段错误、数值 99.9% 垃圾 | runtime 执行 | libspert 版本（grid>512 / 栈大小）、TCM 溢出 |
| `ConnectionRefusedError` 从第 1 例起全 F | RPC server 没起来/端口竞态 | pkill 后等端口真正释放再重启（`ss -tln` until-loop） |
| `ConnectionResetError`/connection closed 中途 | guest 被杀 | server 日志、guest 内存、堆破坏（前序 kernel OOB 延迟发作） |

## 日志分类纪律

pytest 日志按失败块（`^_{5,} name _{5,}$` 切分）逐块取**首个命中**的错误行分类；
pattern-anywhere 匹配整块会把 traceback 里 IR dump 的上下文字样误分类。
引用任何历史数字前先核对：(a) 该批次的 runtime 版本（0.6.0 时代的“数值失败”
多是 grid 丢弃假象）；(b) runner 是否真的用了你以为的二进制
（`SPINE_MLIR_OPT_PATH` 是否设置）；(c) 该文件是否真有完整跑完的记录。

## A/B 二进制对照

```bash
stat -c %s $(which spine-opt); file $(which spine-opt)   # 先确认非 0 字节、真 ELF
# bash 对 0 字节文件 execve ENOEXEC 会退化成“当脚本执行”→ rc=0 假通过！
SPINE_MLIR_OPT_PATH=/path/to/spine-opt.variant \
  TRITON_CACHE_DIR=/tmp/cache_variant \
  python -m pytest test_x.py -q
```

每个变体独立 cache + 独立 QEMU server（若用 RPC），失败用例逐例 diff。

下一篇：[05-spine-raw-edsl.md](05-spine-raw-edsl.md)
