# DSpark 动态校验（adaptive verification）

Qwen3.6 + DSpark 上的自适应校验：每步按 confidence 裁剪每个请求的 draft 后缀，
用「每步接受更少 token」换「每步更便宜」。本目录记录开启方式、语义边界和验证顺序。

## 开启方式

由配置驱动，不需要环境变量：

```bash
vllm serve /path/to/Qwen3.6-27B \
  --tensor-parallel-size 4 --max-num-seqs 16 --max-model-len 4096 \
  --enable-prefix-caching --async-scheduling \
  --speculative-config '{"method":"dspark","model":"/path/to/DSpark",
    "num_speculative_tokens":7,"enable_adaptive_verification":true}'
```

`enable_adaptive_verification: true` 一项同时决定四件事：选用带 ragged 改动的
manager、图模式取 `ragged`、GDN 请求轴钉到 `max_num_seqs`、host 侧 query 边界保持为
上界（入图必须如此）。

不要加 `--enforce-eager`，那会把图关掉。

`num_speculative_tokens` 必须**精确等于** draft checkpoint 的 `block_size`，不是 ≤。
换 K 需要换 checkpoint，所以 K 不是可调的轴；可调的等效轴是并发。

### 不支持的配置会退回，不会起不来

PP/PCP/DCP≠1、LoRA、DBO、`num_speculative_tokens` 超出 1..15，都会退回上游 manager
并打一条 warning。那是这条路径成为默认之前，这些配置本来就会得到的行为。

例外是显式点名：设了 `VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV=1` 而配置不支持时**报错**，
因为那是明确要求而不是默认选择。同理，任一 lane 在没有 confidence head 时报错而不是
静默降级。

### 诊断与二分用的环境变量

默认路径不需要这些，它们只用于把一个现象拆开定位。

| 变量 | 用途 |
| --- | --- |
| `VLLM_ASCEND_DSPARK_AV_GRAPH` | 显式指定 `none`/`uniform`/`ragged`，不设＝`ragged` |
| `VLLM_ASCEND_DSPARK_AV_ADAPT=0` | 完全退回上游 manager，用于对照 |
| `VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD` | lane A（survival threshold），默认保持 eager |
| `VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV=1` | 显式点名 lane B |
| `VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS` | 在非 ragged 模式下单独钉住 GDN 请求轴 |
| `VLLM_ASCEND_DSPARK_AV_TP_BROADCAST=1` | 每步从 rank0 广播 confidence（TP=4 下是每步一次集合通信） |
| `VLLM_ASCEND_DSPARK_AV_KEEP_PIECEWISE=1` | 保留 v0.28 强加的 FULL_AND_PIECEWISE |
| `VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND=1` | eager 下去掉 host 边界回读 |
| `VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL` | 聚合 warn 行的步长间隔，默认 50 |

## 两条 lane

| | lane A | lane B |
| --- | --- | --- |
| 选择 | `..._EAGER_SURVIVAL_THRESHOLD` | `..._EAGER_UPSTREAM_AV=1`，或配置默认 |
| 策略 | survival 阈值裁剪后缀 | 上游 cost-argmax 预算 + 设备 survival top-k |
| host/device 边界 | 精确一致（每步 D2H） | 上界（上游契约） |
| 用途 | 二分：host 视图不会和 device 分歧 | 生产行为 |

lane A 刻意付每步一次 D2H 同步。精确边界的价值是让 state 算子那套管线能对着一个
不可能与 device 分歧的 host 视图来检查。入图阶段这条路走不通，所以 lane B 才是
production 路径。

## 运行时日志

两条 lane 的决策走 `logger.warning`，按 `..._LOG_INTERVAL` 步长聚合成一行。warn
级别是因为要在普通 `vllm serve` 日志里可见而不必把整个进程提到 DEBUG；聚合是因为
TP=4 下每步每 rank 一行会把要找的故障埋掉。

`EagerAVLogger.sampling()` 决定这一步是否会真的打印，调用方据此避免在不打印的步上
做阻塞拷贝 —— lane B 的 per-request capacities 在 device 上。

日志里会报 confidence 信号是否还在变化。入图后 confidence 算子只有被 trace 进捕获
的 draft 图里才会每次 replay 重算；一个在捕获时为 False 的 Python `if` 会把它整个
排除，之后每次 replay 都读一个冻结在捕获前的 buffer。**输出比对发现不了这件事**：
裁剪只是策略，输出照样有效，特性只是悄悄停止自适应。

## 正确性的十条不变量

1. 每个真实请求至少保留一个 anchor
2. 只允许裁剪 draft 后缀，不能在中间留洞
3. 上轮接受位置不能被本轮长度截断
4. 空请求行必须是零长度，并使用无效状态索引
5. padding token 不能写入真实 KV 或 GDN state
6. capture 与 replay 之间，图看到的 tensor 地址必须稳定
7. uniform target、ragged target 和 drafter descriptor 必须隔离
8. TP 所有 rank 必须使用同一份裁剪计划
9. 概率、请求集合或 metadata 不可信时，保持完整长度或回退原生路径
10. prefill、mixed 和 handoff 必须有明确边界，不能因为「看起来像 decode」就误入
    ragged 图

第 8 条默认靠「confidence head 输出已在组内 reduce 过，各 rank 本就一致」成立，而不
靠每步广播；`..._AV_TP_BROADCAST=1` 是各 rank 真的被观察到分歧时才用的保险。

## 已知的坑

- **图能捕获 ≠ 图能正确回放。** GDN 状态、query boundary、请求槽位逐轮变化。必须用
  固定地址 buffer：capture 与 replay 同一块内存，每轮 replay 前刷新，batch 收缩时
  清理整个 inactive tail，再把 active prefix 发布给图。
- **FULL replay 不重新进入 Python wrapper。** 概率若只存在普通 Python 属性里，运行时
  可能读到启动捕获阶段最后一张图留下的 buffer。概率输出必须与具体 graph descriptor
  和固定地址绑定，每轮只消费一次。
- **drafter 的旧 context 会写进新 KV。** FULL 图为较大 context bucket 准备固定 buffer，
  真实 context 较短时未使用的 tail 仍残留旧 hidden state；若其 slot mapping 仍指向真实
  KV slot，旧内容会再次写入 draft KV。处理是把未使用 tail 的 slot mapping 设为无效值。
  **症状不是报错，而是草稿质量下降、总轮数增加** —— 所以性能不能只看单步 target 时间。
- **mixed batch 不要强行进入 ragged FULL。** ragged FULL 的契约是纯 speculative decode。
  路由：纯 spec decode → ragged FULL；未裁剪的规则 decode → 原版 uniform FULL；含真实
  prefill 或不满足契约 → 退出 ragged FULL。混部的正确优化方向是把 prefill 和 decode 拆
  成独立 microbatch，而不是放宽判断。
- **padding 行不要清零。** 清零正是 `_remove_spec_graph_padding_queries` 找到它们的
  依据，而那个函数会在第一个 padding 行处截断 GDN query 视图，于是请求轴又变成跟随
  存活请求数 —— 就是那个在并发爬升时产生乱码输出的变轴故障。保留每个空行并就地让它
  失效（零长度、无效状态索引、算子跳过）才能把轴钉在 `max_num_seqs`。
- **Ascend 没有可用的 float64。** device 侧累加器一律 float32；float64 规约在那里是
  静默失败而不是报错。

## 验证顺序

在仓库根目录、已安装 NPU 依赖的环境中按顺序执行，不要并行跑。卡号以 `0,1,2,3` 为例。

### 0. 编译 D-Cut 算子

```bash
: "${SOC_VERSION:?set the 910B4/CANN build target}"
COMPILE_CUSTOM_KERNELS=1 pip install -v -e . --no-deps --no-build-isolation
```

不重装算子包，host 侧的 layout 策略在设备上不生效。

`CAUSAL_CONV1D_QUERY_START_LOC_DEFINES_LAYOUT` 只在 D-Cut 的编译单元里置 1，通用
`npu_causal_conv1d_custom` 保持原推断不变。该编译单元还必须 `#define
causal_conv1d_host dcut_causal_conv1d_host` 隔离命名空间：三个共享 tiling 头把所有
helper 都放在 `optiling::causal_conv1d_host` 里且全是 `inline`，两个编译单元发出同一
mangled symbol 而函数体不同时，vague linkage 让链接器静默只留一份。头里的 `#error`
守卫会在开了策略却没隔离命名空间时直接编译失败。

### 1. CPU UT

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest --noconftest -o addopts='' -sv \
  tests/ut/spec_decode/test_eager_survival_verification.py \
  tests/ut/ops/test_dcut_cpu_reference.py
```

检查两条 lane 的准入门槛、裁剪策略、请求映射、metadata、warn 聚合，以及独立数学参考。

### 2. 单卡 GDN 算子

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest --noconftest -o addopts='' -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_eager_gdn_varlen.py
```

这一步不全绿就停在这里，不要继续跑模型，也不要放宽误差。

失败时先分层：`[8]` 和 `[3,1,4]` 的 token 数与 request 数不等，旧逻辑本来就走变长
路径；`[1,1,1]` 是 T==B 但每行恰好一个 token，两种读法一致。只有 `[8,0,...]` 会被旧
逻辑读错。所以「`[8,0,...]` 的 token 0 正确、token 1..7 未被写入」这个特征等于
"运行的二进制里 layout 策略没生效"，不是 kernel 或 golden 的问题。

### 3. TP=4 整网验证

不用 pytest。直接跑真实引擎：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_TEST_QWEN36_MODEL=/path/to/Qwen3.6-27B
export VLLM_TEST_DSPARK_MODEL=/path/to/DSpark
python examples/dspark_eager_adaptive_verify.py
```

判据：greedy 采样下 target 决定每一个 token，裁剪只改变每步校验多少 draft，不改变
输出。因此逐 token 相等是有效门槛，任何差异都是裁剪后布局的真实缺陷（query 边界、
GDN state 选择、logits 摆放），不是采样噪声。随机采样需要另做分布验证。

**「相等」本身不是证据。** 一个从未裁剪过的 lane 校验的宽度和 baseline 一样，逐 token
相等什么都没说明。因此有两道门槛：lane 只打了启动横幅而没有任何聚合数据行 → 报错
（横幅只证明 manager 被构造，不证明它决策过预算）；lane 相等但全程 `kept=100%` 且不是
`threshold:0.0` → 报 `INCONCLUSIVE` 并返回非零。

`threshold:0.0` 保留全部 draft，是把「新 GDN 路径」与「任何裁剪」分离开的等价性检查，
应当先过。

baseline 按 commit + 引擎配置缓存，所以同一份代码上换 lane 反复跑只启动一个进程。
tracked 文件有改动时完全不缓存 —— 没有哪个键能表示「我手上这些未提交的改动」。

### 4. 收益测量

正确性和收益判据不同，所以是另一个脚本：

```bash
python examples/dspark_adaptive_verify_throughput.py --concurrency 16 --repeats 5
```

- **判据是 TPS 不是 TPOT。** AV 拿「每步接受更少 token」换「每步更便宜」，controller
  的目标函数就是单位成本的接受 token。单请求延迟只量到这笔交易的一边。
- **接受率下降是特性在工作**；降了而吞吐没涨才是要担心的，那说明 cost table 高估了
  裁剪的收益。
- **收益上限是 cost table 的可达 spread。** 4 并发 12.31ms、16 并发 29.97ms，裁剪能
  赚的不会超过这个。
- **输出长度用 `ignore_eos` 钉死。** 跨输出长度比 TPS 是在比不同的活儿；曾经一次不开
  AV 的对照多生成了 6.5% 的 token，和被测效应同一个量级。
- 表里带 spread，差值小于两条 lane 自身波动的会标 `?`。
