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
   ├─ ② cache 命中？ ──是──► 直接加载 .so，跳到 ④
   │        │否
   │        ▼
   ├─ ③ 板上 JIT 编译（首次几秒~几十秒）：
   │      前端 tracing → TTIR
   │      中层转换（指针分析、结构化）→ linalgdir
   │      编译器后端降级（→RVV/矩阵引擎指令）→ LLVM IR → 目标码
   │      链接 kernel.so（全部产物写入 cache 目录）
   │
   ├─ ④ 运行时加载 kernel.so：grid 的每个 program = 一个 tile，
   │      调度到 worker 协程执行（超大 grid 由运行时自动分块）
   │
   └─ ⑤ kernel 把结果直接写进你传入的 torch.Tensor 内存（普通 CPU 内存，
        没有 host/device 拷贝这一步）
```

**cache 目录**（`TRITON_CACHE_DIR`，默认 `~/.triton/cache`）按 key 分子目录，
每个 kernel 一个，里面有：

| 文件 | 是什么 | 什么时候用 |
|---|---|---|
| `<kernel>.ttir` | 第①层 IR（Triton 方言） | 怀疑前端/tracing 问题时看 |
| `<kernel>.linalgdir` | 第②层 IR（linalg/memref），编译器后端的输入 | 看中层转换结果；编译崩溃时确认走到哪一层（4.4） |
| `<kernel>.ll` | LLVM IR | 看最终指令选择（vle/vse/vfwmacc） |
| `<kernel>.so` | RISC-V 共享库 | `nm` 查符号；执行期问题 |
| `<kernel>.json` | 编译元数据 | 查签名/特化 |

**判别技巧**：一个 cache 条目“有 linalgdir 没有 .so”= 编译失败在编译器
后端或更后面；“连 ttir 都没有”= 前端 tracing 就崩了。排错时先 `ls` 失败
kernel 的 cache 目录，产物断在哪一层，问题就在哪一层。

## 4.2 缓存纪律（最容易吃亏的地方）

**cache key 不包含编译器二进制的版本。**这意味着：

- 你升级了 wheel，或改了 triton 包内的 Python 模块（比如源码安装时改了
  `language/` 下的文件）——**旧 .so 会被继续命中复用，你的修复完全不生效**；
- 反过来，某个“一直失败”的 kernel 可能在新版里早已修好，仅仅因为缓存没清
  而继续失败。

历史上大量“已修但复测仍失败”“本体全过但某 shape 一直崩”的误判都源于此。

标准动作：

```bash
# 升级 wheel 后：把旧缓存挪走（mv 是原子的，对正在跑的批次安全；不要 rm -rf）
mv ~/.triton/cache ~/.triton/cache.pre-upgrade

# 单发探针：独立缓存最干净，完全不碰主缓存
TRITON_CACHE_DIR=/tmp/probe_cache_demo python3 my_probe.py

# 验证“这次跑到底有没有重新编译”：
find ~/.triton/cache -name '*.so' -newermt '-10 minutes'   # 0 个 = 全走缓存
```

还有一个镜像判别法：升级 wheel 前后各跑一次，对比缓存 `.so` 的 mtime——
**0 个重编而结果翻转 ⇒ 问题在运行时/环境，不在编译产物**；反过来，
**结果不变且 0 个重编 ⇒ 你根本没在测新版**。

## 4.3 环境变量完整手册

**先说结论：板上正常运行不需要任何环境变量**（芯片型号自动探测，运行时
进程内加载）。下面全部是可选的调试开关。

### 编译调试类

| 变量 | 说明 |
|---|---|
| `SPINE_TRITON_DUMP_PATH` | 设为一个**已存在**的目录，每级 IR 都会 dump 进去（ttir/linalgdir/ll/s）。排编译问题的第一开关 |
| `TRITON_CACHE_DIR` | 缓存目录（见 4.2）。探针、A/B 对照都用独立目录 |

### profiling 类（proton）

| 变量 | 说明 |
|---|---|
| `PROTON_KERNEL_CAPTURE=1` | 编译期在 `_launch` 插入计时钩子（读板卡的 cycle 计数），得到每个 kernel 的耗时 |
| `PROTON_OUTPUT` | 输出文件（`.json` = Chrome Trace 格式，可用 chrome://tracing 或 Perfetto 打开；`.hatchet` = 树状汇总） |

host 侧区间计时（拆 Python 层耗时用）：wheel 自带的运行时库里有
`proton_enter_kernel(name, 0, 0, 0)` / `proton_exit_kernel(...)` 两个 C 接口，
可以用 ctypes 直接调，包住你想测的任意 Python 代码段。真实用例：把一个
Qwen 模型的 prefill 拆成 lm_head 67% / attention 18% / mlp 13%，99.6% 的
墙钟都能对上号——**profile 先行，再决定优化谁**（ops/07 §7.7 的纪律）。

## 4.4 IR dump 与编译器崩溃的排查流程

**场景**：kernel 报 rc=134 / `Assertion failed`，没有 Python 栈——这是
编译器后端崩了，不是你 kernel 的语法问题。标准流程：

```bash
# 第 1 步：落盘全部中间产物
mkdir -p ./dumps && SPINE_TRITON_DUMP_PATH=./dumps python3 repro.py
ls ./dumps/<kernel_name>/     # 产物断在哪一层，问题就在哪一层
```

第 2 步：**换一个全新缓存重跑**（排除旧产物污染，4.2）：

```bash
TRITON_CACHE_DIR=/tmp/fresh_$$ python3 repro.py
```

第 3 步：**最小化复现**。把 shape 缩到最小（M=2、N=2、grid=1）、把 kernel
删到只剩出事的运算。两个判据：

- 最小 shape 也崩 ⇒ 与数据/尺寸无关，是 config（BLOCK/tile）或某类 IR 形态
  触发的编译器问题；
- 只有特定 shape/区间崩 ⇒ 记下精确触发区间（4.9 的 NaN-guard 扫描法可以
  帮你把区间扫出来），这对修复者极有价值。

第 4 步：**升级 wheel 再测**（升级后务必换缓存，4.2）。相当一部分历史
崩溃在新版已修复。

第 5 步：仍复现 ⇒ 带着这些材料报告问题：完整报错输出、dump 目录、最小
复现代码、版本指纹（`pip show triton`）。

**数值问题也要 dump 全链路**：mask 丢弃这类 bug 会跨层存在（linalgdir 层
看着对、ll 层 store 没了 mask 守卫），单层看不出——对 `.linalgdir` →
`.ll` 逐层 grep 关键的 store/load，确认语义在哪一层丢的。

## 4.5 失败分层判别表（先定位层，再进对应小节）

| 症状签名 | 层 | 标准动作 |
|---|---|---|
| `CompilationError` + 指到你的 Python 源码行 | ① 前端 tracing | dtype 军规 / constexpr 解包 / API 拼写，见各 ops 章 |
| rc=134、`Assertion failed`、无 Python 栈 | ③ 编译器后端 | 4.4 流程：dump → 新缓存 → 最小化 → 升级 → 报告 |
| `LLVM ERROR:` 开头的遗言（无 "error:" 前缀） | ③ 代码生成 | 同上；dump 里看 `.ll` 是否已生成 |
| kernel .so 加载失败 / `undefined symbol` | ④ 运行时加载 | 多为二进制混装：整套重装 wheel。诊断：`nm -D --undefined-only kernel.so` 列出缺的符号 |
| 执行期段错误（rc=139 / SIGSEGV） | ④ 运行时执行 | 4.6 排查顺序 |
| 输出全零 / 99.9% 垃圾值 / “碰巧全对” | ④ kernel 没跑或没写全 | 旧版 wheel 的运行时限制（如超大 grid 上限，新版已修）→ 升级 + 换缓存重跑；或输出 buffer 没初始化——“碰巧全对”是零页假象，**flaky 通过勿当恢复** |
| `exit=1` 但摘要显示全 PASS | pytest 环境 | cwd 无写权限（conftest 写结果文件 PermissionError）——从 /tmp 等可写目录跑 |

## 4.6 段错误与“玄学”崩溃的排查顺序

CPU 上越界访问不会像 GPU 那样立刻给你个显式错误，而是**踩坏堆**——症状
经常出现在很晚的、不相关的另一个操作里（下个 kernel 编译时 abort、GC 时
崩、甚至下一次 launch 段错误）。遇到“莫名其妙”的崩溃，按这个顺序走：

1. **换新缓存重跑**（成本最低，先排除旧产物）。
2. **看崩溃形态定层**：有 Python 栈 = host 层（读栈）；rc=134 无栈 = 编译器
   断言（4.4）；rc=139/SIGSEGV = 执行期堆损坏（继续往下）。
3. **最小化**：缩 shape、砍 kernel。注意“配置相关不数据相关”的判别——
   M=1/N=2/grid=1 也崩的话，和数据内容无关，是 tile/config 问题。
4. **查 tile 是否过大**：每 worker TCM 只有 256KB（basics/01 §1.5），大 tile
   的中间量放不下会分配失败。把 BLOCK 减半试试；经验边界：单 tile 8192
   元素（f32）量级安全，再翻倍就危险。
5. **怀疑前序 kernel**：段错误位置 ≠ 根因位置。单独只跑出事的那个 kernel
   （新进程），若单独跑不崩，逐个加回前面的 kernel 找真凶。
6. **查自己 kernel 的越界**：无 mask 的 store 是堆损坏头号来源（basics/03
   §3.3）；越界读用 NaN-guard 探针（4.9）。
7. **升级 wheel 重跑**；仍复现 ⇒ 带最小复现报告。

## 4.7 历史数字引用纪律（读日志/报告时）

这套环境的失败数字有强烈的“批次属性”，引用任何历史结果前核对三件事：

1. **wheel 版本**：旧版 wheel 时代的“数值失败”里有大量环境假象（比如
   旧运行时对超大 grid 有上限，超限的 launch 被静默丢弃，输出成了未初始化
   内存——看起来 99.9% 元素错值，其实 kernel 根本没跑）。判别：同一份缓存
   （0 重编）只换 wheel 复跑，结果翻转 ⇒ 环境问题，不是 kernel 问题。
2. **环境一致性**：对比两批数字前，确认 wheel 版本、缓存目录、跑批脚本
   （有没有硬编码环境变量指向别的安装）都核对过。“对照组”跑在了同一个
   安装上是假对照最常见的形态。
3. **是否完整跑完**：超时截断（rc=124）的批次没有“全量结论”资格；续跑
   剩余用例用 `--collect-only -q` 拿有序 ID 列表数出已完成的，`-k "(...)"`
   精确选剩余，复用同一 `TRITON_CACHE_DIR` 免重编。

pytest 日志分类要按失败块切分（`^_{5,} name _{5,}$`），逐块取**首个命中**
的错误行归类；拿关键词在整块文本里随便匹配会把 traceback 附带的 IR dump
上下文误分类（IR 文本里到处都是形似关键词的字样）。

## 4.8 A/B 对照实验（标准实验法）

“这个 bug 是不是新版修的？”“是不是某个改动引入的回归？”——不要猜，做
单变量对照：

```bash
# 两个 venv 各装一个版本的 wheel（其余依赖一致）
python3 -m venv ~/venv-A && ~/venv-A/bin/pip install <依赖> && \
    ~/venv-A/bin/pip install triton==<版本X> --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
python3 -m venv ~/venv-B && ~/venv-B/bin/pip install <依赖> && \
    ~/venv-B/bin/pip install triton==<版本Y> --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

# 各自独立缓存，跑同一份测试
TRITON_CACHE_DIR=/tmp/cache_A ~/venv-A/bin/python -m pytest test_x.py -q > A.log 2>&1
TRITON_CACHE_DIR=/tmp/cache_B ~/venv-B/bin/python -m pytest test_x.py -q > B.log 2>&1

# 失败用例逐例 diff（不是只看 F 数——数量相同名单不同是常事）
grep '^FAILED' A.log | sort > A.failed
grep '^FAILED' B.log | sort > B.failed
diff A.failed B.failed
```

三条纪律：

1. **单变量**：一次只改一样东西（wheel 版本、或缓存、或脚本），否则归因
   无效。
2. **独立缓存**：共用缓存会让一边命中另一边的产物，A/B 直接失效。
3. **跑批脚本先读一遍**：脚本里硬编码的环境变量（指向某个特定安装的
   路径等）会让对照组变成假对照。

写对照脚本时的 shell 陷阱（都真实踩过）：

- `rc=$?` 必须在目标命令的**下一行立刻捕获**——中间夹任何命令（包括
  `echo "$(basename f): rc=$?"` 里的命令替换）都会把 `$?` 重置；
- `PIPESTATUS[0]` 同理，管道后任何命令一执行就失效；
- **bash 对 0 字节文件会当空脚本执行返回 rc=0**——A/B 任何二进制前先
  `stat -c %s` + `file` 确认非 0 字节、是真 ELF，否则“全过”是假象。

## 4.9 数值验证工具箱

- **golden 对照**：`torch.testing.assert_close(out, ref, rtol, atol)`。
  容差按 dtype 与归约长度设：f16 逐元素 1e-2 量级；长归约（sum over N）
  的 atol 要随 N 缩放，只用固定小 atol 会出 flaky。
- **NaN-guard 切片探针**（查越界读的利器）：

  ```python
  buf = torch.full((numel + 4096,), float('nan'))
  x = buf[:numel].view(M, K)          # 保持精确 stride/specialization
  ```

  从 NaN 大 buffer 头部切出输入，任何越界读都会把 NaN 带进输出；改变
  shape 逐点扫描还能定位精确触发区间（历史上一个只在特定 K 区间出错的
  编译器越界 bug 就是这么扫出来的）。
- **CPU 仿真三模型对拍**（查累加语义）：emuA=f16 精确乘积+f32 累加
  （=Triton 语义）、emuB=f16 舍入乘积、实测三方对比 bit 一致率——
  “部分和截断”类精度 bug 用这法定性（修复后实测 99.9% bit 一致）。
- **错值签名反推**：全零=写丢弃或 launch 被丢弃；99.9% 垃圾=未初始化内存；
  隔行错=stride/越界；带精确公式的错值可以反推出读偏移。

下一篇：[05-spine-raw-edsl.md](05-spine-raw-edsl.md)（进阶，可先跳过直接进 ops/01）
