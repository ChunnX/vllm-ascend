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
