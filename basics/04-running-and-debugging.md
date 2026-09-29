# 04 — kernel 运行时发生了什么：调试与排错手册

前几章你已能写 kernel。这一章回答三个问题：

1. 一次 launch 背后，编译产物放在哪、长什么样、怎么调出来看？
2. 每个环境变量到底控制什么？
3. 报错时怎么判断“错在哪一层”，以及每一层的标准排查动作？

这是全书最常回来翻的一章，建议第一遍通读、之后当手册用。

## 4.1 一次 launch 的完整时间线

```
kernel[grid](args..., BLOCK=1024)
   │
   ├─ ① 计算 cache key（源码哈希 + 签名 + constexpr 值 + 特化信息 + 后端配置）
   │
   ├─ ② cache 命中？ ──是──► 直接加载 .so，跳到 ⑥
   │        │否
   │        ▼
   ├─ ③ 编译（首次几秒~几十秒）：
   │      前端 tracing → TTIR
   │      spine-triton-opt → linalgdir
   │      spine-opt --spine-triton-e2e-pipeline → ll
   │      llc → .s/.o → 链接 kernel.so
   │      （全部产物写入 cache 目录）
   │
   ├─ ④ native: libspert 直接加载 .so
   │   RPC:  把 .so + 输入 tensor 字节流通过 TCP 发给 QEMU guest 的 server
   │
   ├─ ⑤ guest/板端: libspert 加载 .so，grid 的每个 program = 一个 tile，
   │      调度到 worker 协程执行（可分块：grid>capacity 时引擎顺序分块）
   │
   └─ ⑥ RPC: 按 remote buffer 逐个回读输出内存 → 写回你的 torch.Tensor
```

**cache 目录**（`TRITON_CACHE_DIR`，默认 `~/.triton/cache`）按 key 分子目录，
每个 kernel 一个，里面有：

| 文件 | 是什么 | 什么时候用 |
|---|---|---|
| `<kernel>.ttir` | 第①层 IR（Triton 方言） | 怀疑前端/tracing 问题时看 |
| `<kernel>.linalgdir` | 第②层 IR（linalg/memref），**就是 spine-opt 的输入** | 离线复现后端崩溃（4.4） |
| `<kernel>.ll` | LLVM IR | 看最终指令选择（vle/vse/vfwmacc） |
| `<kernel>.so` | RISC-V 共享库 | `nm` 查符号；执行期问题 |
| `<kernel>.json` | 编译元数据 | 查签名/特化 |

**判别技巧**：一个 cache 条目“有 linalgdir 没有 .so”= 编译失败在 spine-opt
或更后面；“连 ttir 都没有”= 前端就崩了。按失败 kernel 名从 cache 里捞
linalgdir，就是现成的复现输入。

## 4.2 缓存纪律（最容易吃亏的地方）

**cache key 不包含编译器二进制的版本。**这意味着：

- 你升级了 spine-opt / spine-triton-opt / libspert，或改了 `language/` 下的
  Python 模块——**旧 .so 会被继续命中复用，你的修复完全不生效**；
- 反过来，某个“一直失败”的 kernel 可能在修复后仍失败，仅仅因为缓存没清。

历史上大量“已修但复测仍失败”“本体全过但某 shape 一直崩”的误判都源于此。

标准动作：

```bash
# 换编译器/语言层后：换缓存目录（mv 是原子的，对正在跑的批次安全；不要 rm -rf）
mv ~/.triton/cache ~/.triton/cache.pre-upgrade

# 单发探针：独立缓存最干净，完全不碰主缓存
TRITON_CACHE_DIR=/tmp/probe_cache_demo python3 my_probe.py

# 验证“这次跑到底有没有重新编译”：
find ~/.triton/cache -name '*.so' -newermt '-10 minutes'   # 0 个 = 全走缓存
```

还有一个镜像判别法：修复前后各跑一次，对比缓存 `.so` 的 mtime——
**0 个重编而结果翻转 ⇒ 问题在 runtime/环境，不在编译产物**。

## 4.3 环境变量完整手册

### 运行模式类

| 变量 | 默认 | 说明 |
|---|---|---|
| `SPINE_TRITON_RPC_HOST` | 空 | 非空 = RPC 模式（host 不加载 libspert，kernel 发往该地址）。**x86 上必须设**，否则 import triton 时 dlopen RISC-V libspert 报 `cannot open shared object file`（文件存在也报——异架构 ELF 签名） |
| `SPINE_TRITON_RPC_PORT` | 9999 | RPC server 端口 |
| `SPINE_TRITON_RPC_TIMEOUT` | 600 | 单 kernel 执行超时（秒）。QEMU 模拟很慢，大 shape kernel 跑 20min~2h 正常；一批失败时间高度集中在 ~602s = 撞了这个超时，调大它（如 7200） |
| `SPINE_TRITON_RPC_THREADS` | 8 | guest worker 线程数 |
| `SPACEMIT_EP_QEMU_SET_CORE_ARCH` | 0xF000 | 编译 target 的 arch id，K3 = `0xA064`。它进 IR 的 dlti 属性，决定 MICRO tile 合法表等 arch 相关行为 |

### 编译调试类

| 变量 | 说明 |
|---|---|
| `SPINE_TRITON_DUMP_PATH` | 设为一个**已存在**的目录，每级 IR 都会 dump 进去（ttir/linalgdir/ll/s）。排编译问题的第一开关 |
| `TRITON_CACHE_DIR` | 缓存目录（见 4.2） |
| `SPINE_MLIR_OPT_PATH` | 覆盖 wheel 里的 spine-opt 路径——做“二进制 A/B 对照”的标准手段 |
| `SPINE_TRITON_USE_REF_PIPELINE` | =1 走参考 pipeline（对照用，一般不碰） |

### profiling 类

| 变量 | 说明 |
|---|---|
| `PROTON_KERNEL_CAPTURE=1` | 编译期在 `_launch` 插入计时钩子（rdtime） |
| `PROTON_OUTPUT` | 输出文件（`.json` = Chrome Trace 格式，可用 chrome://tracing 或 Perfetto 打开；`.hatchet` = 树状汇总） |

host 侧区间计时（拆 Python 层耗时用）：直接 ctypes 调
`libSpineTritonRuntime.so` 的 `proton_enter_kernel(name, 0,0,0)` /
`proton_exit_kernel(...)`，包住你想测的代码段。真实用例：把 Qwen prefill
拆成 lm_head 67% / attn 18% / mlp 13%，99.6% accounted。

## 4.4 IR dump 与离线复现（后端崩溃的标准流程）

**场景**：kernel 报 rc=134 / assertion，没有 Python 栈——编译器后端崩了。

第一步，拿到复现输入：

```bash
mkdir -p ./dumps && SPINE_TRITON_DUMP_PATH=./dumps python3 repro.py
ls ./dumps/<kernel_name>/
```

第二步，离线喂 spine-opt（不用起 QEMU，纯本机）：

```bash
spine-opt ./dumps/<kernel>/x.mlir \
    --spine-triton-e2e-pipeline="enable-always-tls=1 enable-fuse-group=false" \
    -o /dev/null
rc=$?          # rc 必须在下一行立刻捕获！
echo $rc       # 0=过了 1=报错 134=断言崩溃
```

> 两个 pipeline flag 现在都是兼容性 no-op，带不带结果一样，沿用惯例即可。
> **rc 捕获的两个陷阱**：`o=$(cmd); echo "$(basename f): rc=$?"` 里命令替换
> 会把 `$?` 重置成 basename 的返回值（恒 0）；`PIPESTATUS[0]` 在中间任何命令
> （包括裸 echo）之后读也会被重置。得出“全过/全挂”结论前先检查捕获方式。
> 另一个假通过：**bash 对 0 字节文件会当空脚本执行返回 rc=0**——A/B 二进制前
> 先 `stat -c %s` + `file` 确认是真 ELF。

第三步（定位到具体 pass），打印每个 pass 之后的 IR：

```bash
spine-opt x.mlir --spine-triton-e2e-pipeline=... --mlir-print-ir-after-all 2> all_passes.log
# 找最后一个成功打印的 pass = 崩溃发生在它的下一个 pass
```

要做单 pass 隔离复现时，从 dump 里提取 IR 需要**手工补 module 头和全局段**
（嵌套 pass 的 dump 只打印 func 本体，`memref.global`、别名声明都在外面）。

第四步（llc 层）：如果 spine-opt 过了但编译仍失败，`.ll` 直接喂 llc：

```bash
llc -mtriple=riscv64 ... x.ll    # LLVM ERROR: 开头的行就是 llc 的遗言
```

**数值问题也要 dump 全链路**：mask 丢弃这类 bug 跨层存在（linalg 层看着对、
ll 层 store 没了 mask 守卫），单层看不出——linalgdir → ll.mlir → .ll 逐层
grep 关键 store/load。

## 4.5 失败分层判别表（先定位层，再进对应小节）

| 症状签名 | 层 | 标准动作 |
|---|---|---|
| `CompilationError` + 指到你的 Python 源码行 | ① 前端 tracing | dtype 军规 / constexpr 解包 / API 拼写，见各 ops 章 |
| rc=134、`Assertion failed`、无 Python 栈 | ③ spine-opt/llc | 4.4 离线复现流程 |
| `LLVM ERROR:`（无 "error:" 前缀） | ④ llc | 提取 .ll 单独喂 llc |
| `server error: load kernel failed` | ⑤ runtime 加载 | `nm -D --undefined-only kernel.so` 列出未定义符号，对照 libspert 导出表（`nm -D libspert.so.0.6.x`）——通常是 emitter 引用了该 runtime 版本没有的符号 |
| 执行期段错误 / guest 被杀 | ⑤ runtime 执行 | libspert 版本（grid 分块、协程栈大小都随版本变）、TCM 溢出（tile 是否过大）、前序 kernel 堆越界的延迟发作 |
| 输出全零 / 99.9% 垃圾值 / “碰巧全对” | ⑤ runtime 丢弃 | 旧 libspert（<0.6.3）grid>512 静默丢弃 → 输出 = 未初始化内存：脏池页→99.9% 错值；fresh mmap 零页→碰巧全对（**flaky 通过勿当恢复**） |
| `ConnectionRefusedError` 从第 1 例起全 F | RPC server | server 没起或端口竞态：pkill 后端口残留监听会让新 server bind 失败——stop 脚本要 until-loop `ss -tln` 等端口真正释放 |
| `ConnectionResetError` / connection closed 中途 | RPC guest 死亡 | server 日志、guest 内存上限（qemu `-m`）、堆破坏 |
| 一批失败集中在 ~602s | RPC 超时 | `SPINE_TRITON_RPC_TIMEOUT` 调大 + 两遍法（见 4.6） |
| `exit=1` 但摘要显示全 PASS | pytest 环境 | cwd 无写权限（conftest 写结果文件 PermissionError）——板端从 /tmp 跑 |

## 4.6 QEMU RPC 实操细节

**为什么慢**：QEMU 是功能模拟器，一条 RVV 指令一条指令地模拟。大 shape
kernel 单次执行 20min~2h 正常。应对：

1. **两遍法**（省一个量级墙钟）：编译（spine-opt+llc）占慢跑的大头。第一遍
   让慢 kernel 超时被杀没关系——**.so 已进共享 TRITON_CACHE_DIR**；第二遍
   缓存命中纯执行。真实案例：3h 超时 → 12.5 分钟 PASS。多批并行共用同一
   cache 目录即可。
2. **进度判读**：别盯 pytest 输出（`-q` 管道缓冲被 kill 后只剩一两个字符），
   看 server 日志的 `execute_kernel` 计数 + QEMU 进程 cpu≈elapsed 判活。
3. **墙钟受限 ≠ 失败**：12h 上限仍 rc=124 但 server 日志显示 kernel 正常
   执行的用例，应记“需真板验证”，不是正确性失败。

**逐文件独立 server 的端口竞态**：pkill 停 server 后端口会残留监听一小段，
下一个 server bind 失败 → 整文件从第 1 例起 ConnectionRefused（假失败签名：
首个用例即连接拒绝 + 全 F）。stop 逻辑要等端口真正释放：

```bash
pkill -f spine-rpc-server || true
until ! ss -tln | grep -q ':9999 '; do sleep 0.2; done
```

（另：`pkill -f`/`pgrep -f` 的模式串会匹配到你自己的 bash -c wrapper，
按端口逐个 kill 后用完整 cmdline 复核。）

## 4.7 历史数字引用纪律（读日志/报告时）

这套环境的失败数字有强烈的“批次属性”，引用任何历史结果前核对三件事：

1. **runtime 版本**：libspert 0.6.0 时代的“数值失败”大量是 grid>512 丢弃
   假象（0.6.3 已修）。判别：复跑前后 `.so` 0 重编而结果翻转 ⇒ 环境问题。
2. **二进制 provenance**：跑批脚本里 `SPINE_MLIR_OPT_PATH` 指没指？指向谁？
   “纯净版日志”标签必须核实 runner 真的用了纯净版二进制。
3. **是否完整跑完**：rc=124（超时截断）的批次没有“全量结论”资格；
   续跑剩余用例用 `--collect-only -q` 数已完成 dot 数，`-k "(...)"` 精确
   选剩余，复用同一 cache 免重编。

pytest 日志分类要按失败块切分（`^_{5,} name _{5,}$`），逐块取**首个命中**
的错误行归类；pattern-anywhere 匹配整块会把 traceback 里 IR dump 的上下文
字样误分类（IR 文本里到处都是 expand_shape/inner_tiles 之类词）。

## 4.8 A/B 二进制对照（标准实验法）

“这个 bug 是不是 X 版本修的？”的标准答案流程：

```bash
# 0) 核实两个二进制都是真 ELF、非 0 字节
stat -c %s spine-opt.A spine-opt.B && file spine-opt.A spine-opt.B

# 1) 各自独立 cache + 独立 server（RPC 形态）
SPINE_MLIR_OPT_PATH=$PWD/spine-opt.A TRITON_CACHE_DIR=/tmp/cache_A pytest test_x.py -q > A.log
SPINE_MLIR_OPT_PATH=$PWD/spine-opt.B TRITON_CACHE_DIR=/tmp/cache_B pytest test_x.py -q > B.log

# 2) 失败用例逐例 diff（不是只看 F 数）
grep '^FAILED' A.log | sort > A.failed; grep '^FAILED' B.log | sort > B.failed
diff A.failed B.failed
```

注意跑批脚本里硬编码的 env（比如 BASE_ENV 里写死了某个
SPINE_MLIR_OPT_PATH）会让“对照组”变成假对照——跑之前先读脚本。

单变量定位回归的完整流程（编译器开发者向）：stash WIP → `git revert
--no-commit <commit>` → 增量编译 → 测 → `git revert --quit` + checkout 还原
→ 重编回一致状态 → 复现失败闭环。WIP 必须单独对照排除（先无-WIP 测过，
再 pop 重测，仍过才能锁定）。

## 4.9 数值验证工具箱

- **golden 对照**：`torch.testing.assert_close(out, ref, rtol, atol)`。
  容差按 dtype 与归约长度设：f16 逐元素 1e-2 量级；长归约（sum over N）
  的 atol 要随 N 缩放，只用固定小 atol 会出 flaky。
- **NaN-guard 切片探针**（查 OOB 读的利器）：

  ```python
  buf = torch.full((numel + 4096,), float('nan'))
  x = buf[:numel].view(M, K)          # 保持精确 stride/specialization
  ```

  从 NaN 大 buffer 头部切出输入，任何越界读都会把 NaN 带进输出；逐块扫描
  还能定位触发区间（负 vl OOB 的 K∈(64,96) 区间就是这么扫出来的）。
- **CPU 仿真三模型对拍**（查累加语义）：emuA=f16 精确乘积+f32 累加
  （=Triton 语义）、emuB=f16 舍入乘积、实测三方对比 bit 一致率——
  f16 dot 截断累加 bug 用这法定性（修复后 99.9% bit 一致）。
- **错值签名反推**：全零=写丢弃/丢弃 launch；99.9% 垃圾=未初始化内存；
  隔行错=stride/OOB；精确公式的错值可反推读偏移。

下一篇：[05-spine-raw-edsl.md](05-spine-raw-edsl.md)（进阶，可先跳过直接进 ops/01）
