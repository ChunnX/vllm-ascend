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
重装算子包之后 `[8,0,0,0,0,0,0,0]` 应当转为通过；仍然失败就先停在这里，
不要继续跑模型，也不要跳过或放宽误差。

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

判据：greedy 采样下 target 决定每一个 token，裁剪只改变每步校验多少 draft，
不改变输出。因此逐 token 相等是有效门槛，任何差异都是裁剪后布局的真实缺陷
（query 边界、GDN state 选择、logits 摆放），不是采样噪声。脚本会打印首个
分歧的 prompt 与 token 下标，据此定位。

`threshold:0.0` 保留全部 draft，是把“新 GDN 路径”与“任何裁剪”分离开的等价性
检查，应当先过。lane 没有真正生效时脚本会直接报错，不会把“baseline 等于自己”
算成通过。每个 lane 的完整子进程日志保留在脚本打印的目录里。

随机采样需要另做分布验证；同 seed 逐 token 一致不作为跨裁剪策略的唯一判据。

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
- 间隔用 `VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL` 调整，默认 50 步。
  设成 1 会每步一行，只在定位单步问题时用。
- `upstream` lane 的每请求裁剪长度由 device 端 top-k 决定，只在会打印的那一步
  拷回 host，其余步不额外同步。

## 注意事项

- 不使用 `pytest-xdist` 或 `-n`；模型验证顺序占用同一组四张卡。
- `--noconftest` 只用于上面两个自包含测试，不是完整仓库 UT 的运行方式。
- `git pull` 不更新 editable 安装中已有的算子二进制。改了算子源码就必须重装。
- 不要在现有服务使用的环境中覆盖安装。
- 两个 lane 各自每步都付一次阻塞 D2H，都是诊断用途，不要拿来量性能。
- 回传脚本的 `==== summary ====` 段、失败 lane 的日志路径，以及其中的
  `[DSPARK-EAGER-AV/...]` 行。
