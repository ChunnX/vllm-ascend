# 910B4 TP=4 验证顺序

在 `vllm-ascend` 仓库根目录、已安装 NPU 依赖的 Python 环境中执行。
以下卡号仅以分配到 `0,1,2,3` 为例，请替换成自己的四张开发卡。不要并行跑。

## 0. 重新编译 D-Cut 算子（本次必做）

D-Cut Conv1D 的 host tiling 之前会忽略 `query_start_loc`：2D + runMode=1 的输入
默认按“每 request 一个 token”读形状，只有 token 数与 request 数不一致时才回退到
变长读法。`[8,0,0,0,0,0,0,0]` 恰好 8 token / 8 request，于是 qsl 被整个忽略，
第 0 行只拿到 `{start=0,len=1}`，token 1..7 落到 `cache_indices=-1` 的空行上被
整段跳过（输出留在未初始化状态）。这是算子 host 侧的问题，不是测试或 Python
侧的问题，所以必须改源码并重装算子包：

```bash
: "${SOC_VERSION:?请设置与本机910B4及CANN匹配的编译目标}"
COMPILE_CUSTOM_KERNELS=1 pip install -v -e . --no-deps --no-build-isolation
```

`CAUSAL_CONV1D_QUERY_START_LOC_DEFINES_LAYOUT` 只在 D-Cut 的编译单元里置 1，
通用 `npu_causal_conv1d_custom` 保持原有推断不变。变长读法本身不是新路径，
kernel 无改动。**不重装算子包，这个修复在设备上不生效。**

只置这个宏还不够。三个共享 tiling 头把所有 helper（包括被这个宏改动的
`GetShapeDtypeInfo`）都放在 `namespace optiling::causal_conv1d_host` 里、且全是
`inline`，而 `#define CausalConv1d DcutCausalConv1d` 只改算子名那个 token——
命名空间和函数名都没改。于是两个编译单元发出同一个 mangled symbol 而函数体不同，
`inline` 的 vague linkage 让链接器只留一份、静默丢掉另一份，留下来的通常是
stock 那份（宏=0）。所以 D-Cut 的编译单元还要 `#define causal_conv1d_host
dcut_causal_conv1d_host` 把符号隔开。头里加了 `#error` 守卫：开了 layout 策略
却没隔离命名空间会直接编译失败，不会再静默退回旧行为。

## 1. CPU UT

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest --noconftest -o addopts='' -sv \
  tests/ut/spec_decode/test_eager_survival_verification.py \
  tests/ut/ops/test_dcut_cpu_reference.py
```

共 42 项，检查两个 lane 的准入门槛、裁剪策略、请求映射、metadata、warn 日志
聚合，以及独立数学参考。

## 2. 单卡 GDN 算子

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest --noconftest -o addopts='' -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_eager_gdn_varlen.py
```

共 5 项，调用已安装的 D-Cut Conv1D/recurrent，与独立 golden 比较。
2026-09-21 在 910B4 单卡 5/5 通过（含 `[8,0,0,0,0,0,0,0]`）。这一步不全绿就
停在这里，不要继续跑模型，也不要跳过或放宽误差。

失败时先分清是哪一层，再改东西：`[8]` 和 `[3,1,4]` 的 token 数与 request 数
不相等，旧逻辑本来就会走变长路径，所以它们一直通过；`[1,1,1]` 是 T==B 但每行
恰好一个 token，两种读法一致，也一直通过。只有 `[8,0,...]` 会被旧逻辑读错。
所以「`[8,0,...]` 的 token 0 正确、token 1..7 未被写入」这个特征就等于
“运行的二进制里 layout 策略没生效”，而不是 kernel 或 golden 的问题。

## 3. TP=4 整网验证

不再用 pytest。直接跑真实引擎，一个 lane 一个进程，顺序占用同一组四张卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_TEST_QWEN36_MODEL=/实际路径/Qwen3.6-27B
export VLLM_TEST_DSPARK_MODEL=/实际路径/DSpark
python examples/dspark_eager_adaptive_verify.py
```

默认依次跑 fixed-K baseline，然后 `threshold:0.0`、`threshold:0.4`、
`threshold:1.0`、`upstream`，每个 lane 与 baseline 逐 token 对比。
只跑其中几个：

```bash
python examples/dspark_eager_adaptive_verify.py --lanes threshold:0.4 upstream
```

### 一次跑多久,以及为什么

这个脚本的成本是**启动了几个进程**,不是它们算了多少东西:27B TP=4 加载约 3 分钟,生成只有
几秒。所以:

- 噪声底默认在**同一个引擎里生成两遍**,不再起第二个进程。
- baseline 结果按 **commit + 引擎配置**缓存。同一份代码上换 lane 反复跑,只启动一个进程。

| | 进程数 | 大致耗时 |
| --- | --- | --- |
| 改动代码后第一次 | 2 | 约 10 分钟 |
| 同一 commit 再跑别的 lane | **1** | **约 5 分钟** |
| `--floor cross-process` | 3 | 约 15 分钟 |

缓存键里有 HEAD,所以 `git pull` 之后会自动重测;**tracked 文件有改动时完全不缓存**——没有
哪个键能表示「我现在手上这些改动」。`--refresh-baseline` 用于同一份代码想重测。

`--floor in-process` 读出来的噪声**不会高于**跨进程的那个:它覆盖调度与 reduce 顺序的不确
定性,但不覆盖 allocator 布局和 worker 初始化顺序。对判 lane 来说偏紧是保守方向,但也可能把
真实的参考噪声算成 lane 的缺陷,所以判决卡在边界上时用 `--floor cross-process` 复核。

判据：greedy 采样下 target 决定每一个 token，裁剪只改变每步校验多少 draft，
不改变输出。因此逐 token 相等是有效门槛，任何差异都是裁剪后布局的真实缺陷
（query 边界、GDN state 选择、logits 摆放），不是采样噪声。脚本会打印首个
分歧的 prompt 与 token 下标，据此定位。

2026-09-21 记录：四条 lane 全部 MATCH，`kept` 依次为 100% / 69.3% / 0% /
52.1%，`last_caps` 依次为 `[7,7,7]` / `[2,7,7,3]` / `[0,0,0,0]` / `[1,6,5,3]`。
中间两条是真正的 ragged 批；`threshold:1.0` 把 draft 全裁光，退化成每请求一个
token，覆盖零 draft decode 仍走 spec 路径这一条。

`threshold:0.0` 保留全部 draft，是把“新 GDN 路径”与“任何裁剪”分离开的等价性
检查，应当先过。每个 lane 的完整子进程日志保留在脚本打印的目录里。

**“相等”本身不等于证据。** 一个从未裁剪过的 lane 校验的宽度和 baseline 一样，
逐 token 相等什么都没说明。脚本因此有两道门槛：lane 只打了启动横幅、没有任何
聚合数据行 → 直接报错（横幅只证明 manager 被构造，不证明它决策过预算）；
lane 相等但全程 `kept=100%` 且不是 `threshold:0.0` → 报 `INCONCLUSIVE` 并返回
非零。看到 INCONCLUSIVE 要么提高阈值、要么加长 `--max-tokens`、要么去查
confidence 有没有真的到 manager 手里，不要当成通过。

随机采样需要另做分布验证；同 seed 逐 token 一致不作为跨裁剪策略的唯一判据。

## 功能开启方式（一个配置项）

动态校验现在由**配置驱动**，不需要任何环境变量：

```bash
vllm serve /实际路径/Qwen3.6-27B \
  --tensor-parallel-size 4 --max-num-seqs 16 --max-model-len 4096 \
  --enable-prefix-caching --async-scheduling \
  --speculative-config '{"method":"dspark","model":"/实际路径/DSpark",
    "num_speculative_tokens":7,"enable_adaptive_verification":true}'
```

`enable_adaptive_verification: true` 一项同时决定：选用带 ragged 改动的 manager、图模式取
`ragged`、GDN 请求轴钉到 `max_num_seqs`、host 侧 query 边界保持为上界（图必须如此）。
**不要加 `--enforce-eager`**，那会把图关掉。

在此之前这些都挂在环境变量上，配置项单独打开拿到的是上游未改的 manager——这是「实验通了」
和「功能通了」的差别。

剩下的环境变量都只是诊断/二分用：

| 变量 | 用途 |
| --- | --- |
| `VLLM_ASCEND_DSPARK_AV_GRAPH` | 显式指定 `none`/`uniform`/`ragged`，不设＝`ragged` |
| `VLLM_ASCEND_DSPARK_AV_ADAPT=0` | 完全退回上游 manager，用于对照 |
| `VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV=1` | 显式点名这条 lane：配置不支持时**报错**而不是退回 |
| `VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD` | lane A，二分工具，默认保持 eager |
| `VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS` | 在非 ragged 模式下单独钉住 GDN 轴 |

### 不支持的配置会退回，不会起不来

默认路径下，PP/PCP/DCP≠1、LoRA、DBO、`num_speculative_tokens` 超出 1..15 会**退回上游
manager 并打一条 warning**，而不是让引擎起不来——那是这条路径成为默认之前那些配置本来会
得到的行为。显式设了 `VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV=1` 则报错，因为那是点名要求。

## 收益测量：对齐 PR 15098 的方式

正确性用上面的整网门槛，收益用另一个脚本，因为判据不同：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python examples/dspark_adaptive_verify_throughput.py
```

它复现 15098 那张表的形状（每个 draft 数一行，定长 TPS / 接受率 对 动态 TPS / 接受率 /
加速比），并补了两件本地踩出来的事。

### 三件从 15098 表里读出来的东西

| | 说明 |
| --- | --- |
| **收益随「可裁的量」增长** | 15098 是 +12.1%（K=9）/ +6.9%（K=7）/ +2.1%（K=5）。**但 K 不是我们能调的轴**：`vllm/config/speculative.py:177` 要求 `num_speculative_tokens` **精确等于** draft checkpoint 的 `block_size`，不是 ≤。block7 的模型上 K=5/9 连引擎都起不来。15098 那三行是**三个 checkpoint**（文档里的模型名就叫 `dspark_qwen3_8b_block7`）。我们能用的等效轴是**并发**：cost table 可达 spread 在 4 并发是 12.31ms、16 并发是 29.97ms，裁剪能赚的不会超过这个 spread。 |
| **判据是 TPS 不是 TPOT** | AV 拿「每步接受更少 token」换「每步更便宜」，controller 的目标函数就是单位成本的接受 token —— 那是吞吐。单请求延迟只量到这笔交易的一边。 |
| **接受率是上下文** | 它在 15098 自己的数据里也是降的（4.10 → 4.01）。降是特性在工作；**降了而吞吐没涨**才是要担心的，那说明 cost table 高估了裁剪的收益。 |

另外 15098 两边都跑在 Target FULL 下，所以脚本把 `--cudagraph-mode` 同时施加给两条 lane，
比的是「Target FULL」对「Target FULL + Dynamics」，不是两种图配置。

### 两件本地加的

**输出长度用 `ignore_eos` 钉死。** 之前那组 specbench 里不开 AV 的那次多生成了 6.5% 的
token、输出长 7%，和被测效应同一个量级 —— 跨输出长度比 TPS 是在比不同的活儿。钉死之后每
个配置都精确发出 `concurrency × output_len` 个 token，TPS 就是纯时间比较。脚本会校验这一
点，对不上直接报错而不是给出不可读的表。

**重复跑在同一个引擎里。** 27B TP=4 加载约 3 分钟、生成几秒，所以每个进程加载一次、把同
一份 workload 计时多遍。表里带 spread，**差值小于两条 lane 自身波动的会标 `?`** —— 之前
1.3% 的结论就死在这上面（两次同类配置在 math 上差 5.9%）。

### 还会检查特性真的生效了

动态 lane 如果一条预算决策都没打，或者全程 `kept=100%`，脚本报错而不是给数字：那种情况下
它校验的宽度和定长 lane 一样，吞吐相等是同义反复。这和整网门槛里「相等不是证据」同一条
规矩。

`-k` 不传就从 draft checkpoint 的 `config.json` 里读 `block_size`，因为那是唯一能起来的值；
传多个值需要每个值一个 checkpoint。

单点多跑几遍：

```bash
python examples/dspark_adaptive_verify_throughput.py --concurrency 16 --repeats 5
```

默认 `--concurrency 4 8 16`、`--output-len 256`、`--repeats 3`，6 个进程
（3 个并发 × 2 条 lane）约 30 分钟。

### PR 15147 把方法写出来了

15147 是 D-Cut 作者自己那套 vllm-ascend 适配（不只是算子），它写了 15098 没写的东西：

> "D-Cut runs in PIECEWISE mode at concurrency 64, using 500 requests per dataset
> and a two-run average."

| 数据集 | 吞吐 | TPOT |
| --- | --- | --- |
| Math500 | **+8.5%** | −9.2% |
| Dolly | **+19.1%** | −17.4% |

从这里拿了三样，其中第二样修掉了本脚本一个真的设计错误：

1. **用具名数据集**（Math500 / Dolly），不是合成文本。收益依赖 confidence 在请求之间**有
   差异**，几条重复 prompt 会低估这个差异。

   ```bash
   # Dolly：https://huggingface.co/datasets/databricks/databricks-dolly-15k
   python examples/dspark_adaptive_verify_throughput.py --dataset /路径/databricks-dolly-15k.jsonl
   ```

   两个数据集的字段不同，loader 都认：Math500 取 `problem`，Dolly 把 `instruction` 和
   `context` **拼起来**——closed_qa / summarization 那些类目缺了 context 就没意义
   （"Summarize the following" 后面什么都没有）。

   **预期 Dolly 的收益明显高于 Math500**（15147 是 +19.1% 对 +8.5%），原因就是接受率：数学
   的 draft 本来就接受得好，可裁的更少。所以 Math500 是对 AV 最不利的那个数据集。

   workload 小于文件时按**等距取样**而不是取前 N 条：32 条请求从 Dolly 的 15011 条里取，
   取头部很可能整段是同一个类目，那会让接受率、进而让结论由文件顺序决定。
2. **workload 大小不等于并发。** 500 个请求过一个 64 宽的引擎是持续吞吐；**只发和批一样
   宽的请求数量是在测一波**。原来 `--concurrency` 被我同时当成两者用，现在拆成
   `--concurrency`（= `max_num_seqs`）和 `--num-requests`（默认 500）。
3. **重复取平均**（他们两次，这里三次，并打出 spread）。

**期望值要按比例缩。** 那些收益是在并发 64 测的，是 4×910B4 上限的四倍，而 cost table 的
可达 spread 随并发增长。

顺带一个观察：他们的 TPS 和 TPOT **不是两份独立证据**——+19.1% 与 −17.4% 是同一次测量的两
个视角（1/1.191 ≈ 0.840）。本脚本因此只报 TPS：`ignore_eos` 把输出长度钉死之后，平均单
token 时间就精确等于 `concurrency/TPS`，再列一栏只是重述。

### 上游 vLLM 那边（PR 47808）没有性能测试

`tests/v1/spec_decode/test_adaptive_verification.py` 里全是 **CPU 单测**，测的是预算算术和
不变量，没有吞吐或端到端用例：

```txt
test_budget_stops_where_marginal_drafts_stop_paying_for_themselves
test_profiled_batches_seed_cost_curves_via_consumer
test_compact_batch_preserves_totals_and_bounds
test_budget_caps_at_one_rejection_sampler_chunk
test_zero_budget_rebuilds_cpu_cu_num_logits
test_zero_budget_keeps_one_grammar_row_per_scheduled_draft
test_manager_scopes_varlen_check_without_weakening_runner_cg_mode
```

也就是说**上游 CI 里只保证正确性与不变量，性能数字只出现在 PR 描述里**（15098 的表标注
"provided by StanislavII"，**没有说用的是哪个数据集**）。所以我们这边的分工照抄这个形状是
对的：整网门槛管正确性，吞吐脚本管收益，两者都不进 CI 的性能断言。

其中有几条不变量值得我们也照着验一遍（后三条我们还没有对应用例）：预算上限是一个
rejection sampler chunk；零预算时重建 CPU 侧 `cu_num_logits`；零预算时每个被调度的 draft
仍保留一行 grammar。

## 运行时日志

两个 lane 的裁剪决策以 **warning** 打印，按步数聚合，无需调 DEBUG：

```txt
[DSPARK-EAGER-AV/threshold] active: threshold=0.4, synchronous confidence, ...
[DSPARK-EAGER-AV/threshold] 50 steps | mean reqs=3.40 scheduled_drafts=20.10 \
  admitted=8.60 verify_tokens=12.00 | kept=42.8% | trimmed_steps=47/50 | last_caps=[3, 2, 1, 2]
```

- `kept` 远低于 100% 且 `trimmed_steps` 接近满值，说明裁剪真的在发生；
  `kept=100%` 表示这一轮根本没裁，输出相等不能证明变长路径被走到。
- `last_caps` 是最近一次的每请求裁剪长度；不是全 1 就说明 batch 确实是 ragged。
- 第一条记录到的步数一定会打印一行，之后每 `interval` 步一行。短跑
  （整网门槛只有几十步）因此不会只剩一条横幅。
- 间隔用 `VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL` 调整，serve 默认 50 步；
  整网脚本自己设成 5。设成 1 会每步一行，只在定位单步问题时用。
- `upstream` lane 的每请求裁剪长度由 device 端 top-k 决定，只在会打印的那一步
  拷回 host，其余步不额外同步。
- `conf_moved=n/N` 是**窗口内 confidence 首行与上一步不同的步数**。`N/N` 表示 confidence
  op 每步都在重算；`0/N` 表示 buffer 冻结——捕获时这个 op 没被 trace 进 drafter 的图，
  之后每次回放都跳过它。这个失效模式对输出完全隐形（裁剪只是策略，拒绝采样仍然正确），
  唯一表象是「动态校验没有收益」，所以整网脚本把它算作失败。
- `graph=FULL=n` 与 `kept<100%` 同时出现，才说明**裁剪批真的回放了全图**。

## 注意事项

- 不使用 `pytest-xdist` 或 `-n`；模型验证顺序占用同一组四张卡。
- `--noconftest` 只用于上面两个自包含测试，不是完整仓库 UT 的运行方式。
- `git pull` 不更新 editable 安装中已有的算子二进制。改了算子源码就必须重装。
- 不要在现有服务使用的环境中覆盖安装。
- lane B（默认路径）用上游的异步双缓冲拷贝，稳态没有每步同步，可以用来量性能。建不出
  side stream 时会打一条 warning 并退回阻塞拷贝——看到那条 warning 时的吞吐数据不作数。
  lane A 仍然每步一次阻塞 D2H，是诊断用途。
- 回传脚本的 `==== summary ====` 段、失败 lane 的日志路径，以及其中的
  `[DSPARK-EAGER-AV/...]` 行。
