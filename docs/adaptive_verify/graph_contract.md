# 入图契约与阶段划分

依据 D-Cut 作者在 DFlash 侧走完同一条路的复盘。那份工作面向 DFlash，本分支面向
DSpark，算子与状态语义相同，probability 来源与部分系统边界不同，差异在末节列出。

## 为什么先 PIECEWISE，而不是直接 FULL

原计划是 eager → FULL。改成 eager → **PIECEWISE** → ragged FULL，理由是直接进
FULL 会让「算子问题」和「图回放问题」混在一起，失去归因能力；PIECEWISE 提供一条
能长期对照的正确性基线。

更硬的理由是 cost table。同一模型上实测（Qwen3.3-9B, TP=2, target cost）：

| capture_size | eager (ms) | PIECEWISE (ms) |
| --- | --- | --- |
| 16 | 84.25 | 33.88 |
| 64 | 80.18 | 39.76 |
| 256 | 79.31 | 43.98 |
| 512 | 79.88 | 59.06 |

**eager 下 cost 在 76~84ms 之间几乎水平**——Q 从 16 到 512 没有可区分的代价差异，
controller 选哪个 Q 单步时间都一样，裁剪决策在性能上毫无意义。这解释了为什么
lane B 必须注入合成曲线，也解释了为什么本分支的 eager 结论只能是布局信号。
图外固定开销（算子启动、TP 通信、边界同步）在 eager 中占主导；PIECEWISE 把它压
下来之后曲线才开始随 Q 单调。ragged FULL 之后进一步拉开到 25.87 → 103.09ms。

结论：**在 PIECEWISE 之前不要用真实 cost table 替换合成曲线，也不要谈收益。**

## 六个不同的 batch size

ragged FULL 下把它们当成同一个数字是错误来源：

| 名称 | 含义 |
| --- | --- |
| `B_max` | 服务启动时允许的最大请求数（`max_num_seqs`） |
| `B_live` | 当前这一轮真实存在的请求数 |
| `Q` | verifier 本轮实际处理的总 query token 数 |
| `B_graph` | 某张图描述符能容纳的请求数 = `min(Q, B_max)` |
| `B_fia` | Full Attention 实际看到的请求行数 = `B_live`，有 token padding 时 `B_live + 1` |
| `B_gdn` | GDN 状态算子的固定请求轴 = **`B_max`，所有 bucket 统一** |

`B_graph = min(Q, B_max)`：每个真实请求至少一个 anchor，所以一张 Q-token 图最多
容纳 Q 个真实请求，同时不超过服务最大并发。`Q > B_max` 时把 Q 尽量平均分给
`B_max` 个请求（B_max=96、Q=256 → `64×3 + 32×2`），使每个 Q 都对应一张真实存在
的捕获图。

### B_gdn 为什么必须固定为 B_max

早期让 GDN 请求轴跟随图尺寸（Q=32 用 32 行、Q=64 用 64 行）。单机稳定并发下正常，
PD 分离联调时暴露：每轮测试开始、并发从小到大爬升时，不同图的 stateful tiling 和
请求轴反复变化，前面一批请求短输出或乱码，并发稳定后恢复正常。

统一为 `B_gdn = B_max` 后解决。即使当前只有 32 个请求，后 64 行也作为零长度、
无效状态索引的空行保留，多一点 metadata 清理成本换所有图共用同一套状态契约。

> 对无状态算子，缩小 shape 往往只是性能问题；**对有状态算子，shape 变化还会改变
> 状态读写契约。**

**本分支现状：已改。** `gdn_attn_builder.py` 的 FULL speculative 分支在
`self.ragged_spec_decode`（即 `enable_adaptive_verification`）为真时用
`self.gdn_request_axis = max_num_seqs`，否则保持原来的 `m.num_reqs`。

范围是有意收窄的：固定轴是 **ragged 契约**的属性。未裁剪的固定 K 批里 Q 和请求数
同步变化，per-bucket 轴没有歧义，没必要让它为每个 bucket 多付 `B_max` 的 metadata
清理；文章 §4.5 也把 ragged FULL 和 uniform FULL 分开路由。

另有一条拒绝路径：基类的图 buffer 按 `decode_cudagraph_max_bs = max_num_seqs ×
(num_spec+1)` 再被 `max_cudagraph_capture_size` 截断分配。如果这个上限低于
`max_num_seqs`，buffer 装不下固定轴——此时**拒绝进 FULL metadata 路径**，而不是
悄悄退回 per-bucket 轴，因为那正是这条规则要消除的形状。

`spec_batch_size` 之外还加了 `num_spec_decodes <= spec_batch_size` 的断言：按固定
宽度切片时，批更宽会静默丢掉活跃行，那是这个改动唯一能损坏状态的方式。

> 这段代码在 eager 下不可达（FULL 分支 gate 在 `use_full_cuda_graph` 上，而两条
> eager lane 强制 `CUDAGraphMode.NONE`），所以整网门槛不会碰到它。三个 UT 在
> `tests/ut/ops/test_gdn_attn_builder.py`，需要装了插件的环境：
>
> ```bash
> pytest -sv tests/ut/ops/test_gdn_attn_builder.py -k "gdn_request_axis or per_bucket_request_axis"
> ```
>
> 构造 pure-spec 批时 block table 必须显式给 `num_spec + 1` 列——speculative 状态
> 索引按 `block_table[:, :num_spec+1]` 选每请求的候选状态行，列数不足会在
> `copy_` 处报 shape 不匹配，而且固定 K 那条路径同样会报，与本改动无关。

## 长期正确性的十条不变量

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

本分支已满足的：1、2、3（`accepted-1` 选历史状态）、4（`[8,0,...]` golden）、
8（TP rank0 广播 confidence）、9（不可信 confidence 回退保留 drafts）。
入图阶段才生效的：5、6、7、10。

## 已知的踩坑点，预先登记而不是重新发现

- **图能捕获 ≠ 图能正确回放。** GDN 状态、query boundary 和请求槽位逐轮变化。必须
  用固定地址 buffer：capture 与 replay 同一块内存；每轮 replay 前刷新；batch 收缩
  时清理整个 inactive tail；再把 active prefix 发布给图。
- **FULL replay 不重新进入 Python wrapper。** 概率若只存在普通 Python 属性里，运行
  时可能读到启动捕获阶段最后一张图留下的 buffer。概率输出必须与具体 graph
  descriptor 和固定地址绑定，每轮只消费一次；请求集合变化、概率缺失或 descriptor
  不一致时宁可不裁剪。
- **drafter 的旧 context 会写进新 KV。** FULL 图为较大 context bucket 准备固定
  buffer，真实 context 较短时未使用的 tail 仍残留旧 hidden state；若其 slot mapping
  仍指向真实 KV slot，旧内容会再次写入 draft KV。处理是把未使用 tail 的 slot
  mapping 设为无效值。**症状不是报错，而是草稿质量下降、总轮数增加**——所以性能
  不能只看单步 target 时间。
- **mixed batch 不要强行进入 ragged FULL。** ragged FULL 的契约是纯 speculative
  decode。路由：纯 spec decode → ragged FULL；未裁剪的规则 decode → 原版 uniform
  FULL；含真实 prefill 或不满足契约 → 退出 ragged FULL。后续优化混部的正确方向是
  把 prefill 和 decode 拆成独立 microbatch，而不是放宽判断。

## Cost table 的唯一形状来源

只能是 ACLGraph 实际注册的 capture sizes。配置文件只保留 warmup 长度、warmup 次数
和测量次数；BS 来自 `max_num_seqs`，Q 来自真实图描述符。否则会测量运行时不存在的
Q，或让 controller 选中的 Q 与真正回放的图不一致。

净收益的正确算法：

```text
净收益 = cost_table[完整 Q] - cost_table[选中的 Q]
       - cost table 测量边界之外的新增控制开销
         （概率计算、CPU 决策、TP 广播、ragged metadata）
```

所以不能用「裁掉多少 token」估算收益。接受率很高、controller 经常选完整 Q 时，
新增开销可能抵消节省。

## 从 main_dspark_adaptive_verify_dev 复用了什么

那条分支直接冲 FULL，代价记录在它自己的历史里：2 次 revert、约 10 个纯
instrumentation commit、以及一次**撤回的结论**——一个 `max_num_seqs=1` 的实验
"否证"了几何假设，但那次运行带着 block-table 缺陷，所以「任何配置都是错的，这个
实验不可能否证任何东西」，同一轮的其他排除也建立在同样被污染的运行上。它最终到达
的是 **uniform FULL**（`cap=99`，不裁剪）；ragged FULL 被显式推迟，padding 行的
场景从当期验收里退出。

**已经独立有了的，不用再拿：**

| 那边的修复 | 本分支的对应 |
| --- | --- |
| 重复 `build()` 让两个前处理成死代码 | `a32e38e87`，独立发现同一缺陷 |
| 零 draft 行塌回普通 decode，把非投机算子烤进图 | `mamba_hybrid.py` 的 `spec_decode_mask = is_decode & ~is_prefilling`；且 `threshold:1.0`（`kept=0%`）在 eager 下端到端验过 |
| batch memo 把第一个 group 的 block table 发给所有 Mamba group | `_derive_group_state_indices` + per-group block table |
| — | conv host tiling 的 qsl 推断、`inline` 跨编译单元的 ODR |
| — | `B_gdn = B_max`（那边退回 uniform 而非固定轴） |

**已复用：** uniform decode 描述符（阶段 A 的核心）。

**尚未复用，阶段 B 需要：** 概率烤进/代理进捕获的 FULL draft 图、`B_fia` 固定为
服务最大值 +1 行、清持久 seq_lens 镜像里的 padding 行、padding 行在列表满时的正
KV 长度、FIA replay line 存活于 capture。

## DSpark 与 DFlash 的差异

- **probability 来源不同，对我们有利。** DFlash 原本只需 argmax，D-Cut 额外要算
  `p(selected) = exp(selected_logit - logsumexp(logits))`，要扫词表，是新增固定
  开销；文章 §8 展望的第一条就是降低它。DSpark 有专门的 confidence head
  （`torch.sigmoid(confidence_head(...))`），是模型输出，这块开销基本不存在。
- **代价是另一个问题。** confidence head 在 prefill 期间会吐出非有限行，logsumexp
  路径不会有这个现象。见 [eager_survival.md](eager_survival.md) 的不可信行回退。
- **算子构建陷阱是本分支独有的发现。** 文章记录的 Conv1D 问题是「直接报错」；本分支
  遇到的是 token 数恰好等于 request 数时 host tiling **静默忽略 qsl**，以及
  `inline` 函数跨编译单元的 ODR/COMDAT 丢弃。两者都值得同步回去。

## 修正后的阶段划分

| 阶段 | 内容 | 判据 | 状态 |
| --- | --- | --- | --- |
| 1 | eager 下把状态算对 | 算子 golden 5/5；整网四条 lane 逐 token 相等 | 2026-09-21 通过 |
| 1.5 | 放弃精确 host 视图（`VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND`） | `+ub` lane 仍然 MATCH | 2026-09-21 通过 |
| **A** | uniform 图：`VLLM_ASCEND_DSPARK_AV_GRAPH=uniform`，按 uniform verify 宽度捕获 | `+graph` lane 输出相等**且 `graph=` 显示真的进过图** | 2026-09-21 通过 |
| **A′** | 测量真实 cost table | 可达范围内 spread 足够大 | 2026-09-21：`max_num_seqs=4` 下 spread=0.00ms |
| **A″** | 在服务规模并发下重测 | 图内可达 spread 明显 | 2026-09-21 通过：bs=16 spread=144ms，bs=32 spread=263ms |
| **A‴** | 弄清 bs≥16 的 MISMATCH | 相对 baseline 自身噪声不更差 | 2026-09-21：判据本身失效，已改用噪声底 |
| **B** | ragged 图（清单见下） | 接受率不降且吞吐/TPOT 改善 | A″ 已提供依据，待做 |
| C | 翻 capability，让上游 factory 直接接纳 GDN，撤掉自建 manager | 不设 env 也能跑；可上游 | 未开始 |

### 阶段 A 的实测结果：裁剪批落到 PIECEWISE，不是 eager

2026-09-21 第二次跑（带 `graph=` 仪器）：

| lane | `graph=` | `kept` | 实际路径 |
| --- | --- | --- | --- |
| `threshold:0.0+graph` | `FULL=5` | 100% | 未裁剪 → **FULL** |
| `threshold:0.4+graph` | `PIECEWISE=5` | 67.1% | 裁剪后 → **PIECEWISE** |

两条都 MATCH。第一条证明图**真的进去了**，阶段 A 成立——范围是 bs=4，即噪声底为 0
的并发（见下）。

第二条比预期好，也修正了此前的说法（原以为裁剪批回退到 eager）：上游在 AV 生效时把
`cudagraph_mode` 设为 `FULL_AND_PIECEWISE`，dispatcher 先试 FULL，批不 uniform 就落到
PIECEWISE。所以**文章里的阶段 2 已经免费拿到了**——而那正是 cost table 从平变单调的
那一站。旁证：`0.0+graph` 首个 prompt 是 `1.48s/it, 32.40 toks/s`，eager 同位置是
`3.80s/it`；总耗时 256s 仍高于 baseline 177s，差值是一次性的 capture 开销。

这改变了 B 的成本收益：问题不再是「能不能入图」，而是**「ragged FULL 比现在这套
FULL + PIECEWISE 还能多拿多少」**。只有真实 cost table 能回答，所以插入阶段 A′。

### 「相等」在图模式下同样不是证据

2026-09-21 第一次跑 `+graph`：三条 lane 全部 MATCH，但**没有任何证据说明图被回放过**。
耗时也回答不了：baseline(eager) 183s，三条 `+graph` 分别 249/250/247s——如果
`threshold:0.0+graph` 真在回放图，它每步该比回退到 eager 的 `0.4+graph` 便宜、总耗时
更低，而三条在 1 秒内。两种解释都成立：图从来没进过，多出的约 67s 全是 capture 开销；
或者图进了，但整个生成只有约 20 个 decode 步，每步省几毫秒在 250s（模型加载 + 捕获主导）
里完全看不见。

算术上图是可达的：`cudagraph_capture_sizes=[1,2,4,8,16,24,32]`、`max_num_seqs=4`、K=7，
纯投机批 = 4×8 = 32 token，uniform 描述符带 `32//8 = 4` 个请求 × 8 token，和真实批一致，
`decode_cudagraph_max_bs = 32` 也够。而 `threshold:0.4` 裁剪后是 `[3,8,8,4]`=23 token，
请求几何对不上任何 uniform 描述符，**应当回退**。

这两个「应当」都需要观测。聚合行现在带 `graph=` 字段，按窗口统计每步被派到哪个
cudagraph 模式：

```txt
... | last_caps=[7, 7, 7] | graph=FULL=5
... | last_caps=[2, 7, 7, 3] | graph=NONE=5
```

判据：`threshold:0.0+graph` 必须出现 `FULL`（否则图根本没进过，MATCH 只证明回退路径
正确）；`threshold:0.4+graph` 出现 `NONE` 是符合设计的回退，不是缺陷。

### 阶段 B 的一条风险，来自阶段 A 的代码

`build_attn_metadata` 不给 `build_for_cudagraph_capture` 传 `batch_shared_cache`，所以
capture 时每个 KV cache group 各自重算 GDN-local 修正视图，而
`_remove_spec_graph_padding_queries` 每次调用都新分配一个 device `query_start_loc`。
当前无害——FULL 路径把值拷进 builder 自己的持久 buffer，图看到的是持久 buffer 而不是
那个临时张量。但这正是文章 §4.1「capture 和 replay 必须同一块内存」要盯的形状，阶段 B
引入更多随批变化的张量时要重新核。

### 为什么不需要单独做 PIECEWISE

原因比预想的更直接：**它已经在跑了**。`FULL_AND_PIECEWISE` 让未裁剪批走 FULL、
裁剪批走 PIECEWISE，两者都在阶段 A 的同一次运行里验证过输出相等。所以文章里
「先 PIECEWISE 建基线」这一站不需要单独实施。

### 阶段 A′：测量 cost table

lane B 原先把 `batches_to_profile` 和 `set_initial_cost_curves` stub 掉，因为 eager
没有 capture 可以计时。图模式下改为委托给基类——基类已经把活干完了：

- `batches_to_profile` 产出 capture sizes，**并额外产出超出捕获上限的尾部尺寸**，
  注释写明"真实步会在那里跑 piecewise/eager，只从捕获尺寸线性外推会严重低估"。
  正好覆盖我们裁剪批走 PIECEWISE 的现实。
- `set_initial_cost_curves` 只用 graph-replay 样本给 **draft** 曲线定价（eager target
  步会抬高 drafter 计时，而请求数不像 token 数那样能区分执行模式），verify 曲线用
  全部样本。

测完打一行：

```txt
[DSPARK-EAGER-AV/upstream] cost table (N samples, graph_limit=32): Q=8:...ms, Q=16:...ms, ... | spread=...ms
```

判据来自文章的对照——eager 是 76~84ms 跨 Q=16..512（平的，controller 无从选择）；
PIECEWISE 是 33.88→59.06ms；ragged FULL 是 25.87→103.09ms。

#### 实测结果（2026-09-21，`max_num_seqs=4`，K=7，TP=4，27B）

```txt
graph_limit=32
Q=8:116.80  Q=16:116.80  Q=24:116.80  Q=32:116.80      ← 捕获范围
Q=48:228.20 Q=64:228.20  Q=128:229.28 ... Q=1024:233.65 ← 出图
Q=2048:293.25  Q=4096:532.68  Q=8192:1017.04
```

**可达范围内 spread = 0.00ms。** Q 从 32 裁到 8 省下的时间精确为零，所以 controller
选 `kept=100%`（不裁）是正确决策——裁剪要付概率计算、TP 广播和 ragged metadata 的
开销，换回零。这不是缺陷。

原因是规模：`max_num_seqs=4`、K=7 → 最大 Q = **32 token**，对 TP=4 的 27B 模型而言
每步 116.80ms 全被固定开销吃掉。文章的收益数据来自 32/64 并发、Q 到 1536，**token
量差两三个数量级**。

`Q=32→48` 那个近 2× 的跳变是出图的代价，方向和我们要的相反：它说「无论如何待在图
里」，不是「裁到更小的图」。把这张表按不同 `max_num_seqs` 重算可达 spread：

| `max_num_seqs` | 可达 Q 上限 | spread |
| --- | --- | --- |
| 4 | 32 | **0.00ms** |
| 32 | 256 | 112.48ms（含出图跳变，非纯图内梯度） |

所以**结论不是「裁剪没用」，而是「这个配置测不出裁剪有没有用」**。

#### 平的原因不是 PIECEWISE

一个自然的猜测是「现在没收益是因为裁剪批走了 PIECEWISE，直接入 FULL 会好」。表本身
否掉了这一点：`Q=8..32` 那四个 116.80ms 是 `captured_token_counts()` 里的尺寸，也就是
**在 FULL 图里测的**；Q≥48 才是出图后的 piecewise/eager。所以平的那一段是 FULL 对
FULL，PIECEWISE 不是它平的原因。

旁证：`threshold:0.0+graph`（全程 FULL）256s，`threshold:0.4+graph`（全程 PIECEWISE）
250s——**PIECEWISE 那条反而略快**。若出图有 2× 的稳态惩罚，总耗时会明显拉开。
`Q=32→48` 的跳变更像 shape/padding 特性，而非「piecewise 慢一倍」。`_PROFILE_REPLAYS=5`
且取中位数，所以它也不是 JIT 预热污染。

因此 ragged FULL 的价值是**让裁剪后的批不掉出图，即消除一个惩罚，而不是创造一个
梯度**。在这个规模下那个惩罚本身也很小。

#### 勘误：profiling context 不是压平曲线的原因

这里原先写：`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` 上游默认 **8192**，而门槛
跑 `max_model_len=2048`、吞吐跑 4096，于是「长 context 计时、短 context 服务」把 Q 的梯度
压平了。**这是错的，两个方向都错。**

v0.28.0 的 `set_dummy_context`（`vllm/v1/worker/gpu/input_batch.py:223`）第一步就是
`context_len = max(min(context_len, max_model_len - query_len), 0)`，所以请求值**只在
`max_model_len - query_len` 以下生效**。两个脚本的 `max_model_len` 都低于 8192，profiling
context 一直等于 `max_model_len - query_len`，也就是现在显式设进去的那个值。

因此两个脚本里的 `setdefault` 在**当前配置下是 no-op**；保留它的理由只是与上游文档的用法
一致（上游示例就是按部署 context 显式 export），并且在 `max_model_len > 8192` 时才真正起
作用——而那种情况下偏差方向恰好相反，是**计时短、服务长**，低估 attention 成本。

结论：**2026-09-23 那组 457.3 vs 441.3 TPS 不受这一项影响，不需要因此重测。**

#### A″ 实测：服务规模下曲线确实有梯度

```txt
bs=16, graph_limit=128, reachable spread=144.26ms
Q=8:74.09  Q=16:75.10  Q=24..56:130.48  Q=64:136.84  Q=72..128:218.34

bs=32, graph_limit=256, reachable spread=263.00ms
Q=8:72.97  Q=16..32:85.87  Q=40:134.05  Q=56:160.27  Q=64:204.73  Q=72..152:246.43  Q=160..256:335.98
```

`Q=64:136.84 → Q=72:218.34`：多 8 个 token 贵 81ms。**裁剪确实能落进更便宜的
bucket**，所以阶段 B 的收益论证成立，不再靠类比。

#### A‴：bs≥16 出现 MISMATCH，先于 B

| bs | `graph=` | `mean reqs` | 结果 |
| --- | --- | --- | --- |
| 4 | `FULL=5`（纯） | 3.80 | MATCH |
| 16 | `FULL=2,PIECEWISE=3` | 2.60 | **MISMATCH**（prompt 6 / token 7） |
| 32 | `FULL=3,PIECEWISE=2` | 4.00 | **MISMATCH**（prompt 3 / token 61） |

**首要嫌疑是本分支自己的 `B_gdn = B_max`。** 它在 `max_num_seqs == 活跃请求数` 时是
空操作——bs=4 恰好如此，所以当初看不出问题；bs=16/32 时活跃行只有 2~4 而轴被钉到
16/32，第一次真正产生大量 padding 行。而文章里这条契约是和 **`B_fia`** 以及**持久
seq_lens 镜像的 padding 清理**一起成立的（隔壁分支为此有 `39e454375`、`275c0df09`、
`564c7ecd9` 三个独立修复）。**只搬一半不是更安全，而是半成品。**

所以它改成 opt-in：`VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS`，默认 0（per-bucket 轴，即
已验证过的形状），阶段 B 再打开。门槛脚本加了 `+axis` 后缀用于二分。

#### A‴ 的第一步：先验 baseline 自己可不可复现

`upstream` 在 eager、不入图、固定轴关闭的情况下也 MISMATCH，所以图和固定轴都被
排除，问题在高并发本身。

但在继续追之前要先堵一个方法论漏洞：**bs=4 时 4 个 prompt 并发稳定；bs=16 时 16 个
请求先后结束、批不断收缩。如果有任何东西依赖批组成，baseline 自己跑两次都可能不
一样。** 而门槛是拿一次 baseline 比一次 lane——那样任何 MISMATCH 都归因不了。

脚本加了 `baseline2` lane：再跑一次 baseline 并与第一次比较。它先于一切 lane 结论：

**实测：`baseline2` MISMATCH。** 所以 bs≥16 的三次 MISMATCH（`upstream`、
`upstream+graph` 在 bs=16 与 bs=32）**全部作废**，归因不到任何 lane。

根因不是缺陷，是判据用错了地方。bs=16 下请求先后结束、批不断收缩并被
`sort_batch_req_ids` 重排；TP=4 的 all-reduce 顺序与不同批形状下 kernel 选的 tiling
都会变，末位 rounding 随之变，greedy 的 argmax 在接近平手处就翻。**这是数值不确定
性，不是可修的 bug。**

所以逐 token 相等只在 baseline 确定的并发下是有效判据。改法是**用 baseline 自己当
噪声底**：门槛现在每次都跑一遍 `baseline2`，量出 `(发散的 prompt 数, 最早发散的 token
下标)`，一条 lane 只有在**发散更多的 prompt 或更早发散**时才判失败。

| 噪声底 | 判据强度 |
| --- | --- |
| 0（如 bs=4） | 逐 token 精确相等，最强 |
| 非 0 | 只能证明「lane 没有引入额外漂移」，不能证明精确相等 |

因此**正确性结论要在噪声底为 0 的并发下取得**（目前是 bs=4），而 cost table 这类性能
测量不需要确定性，可以在高并发下做。别让一个工具干两件事。

### 逐 token 判据的天花板，以及正确性证据真正在哪

`threshold:0.0` 在 bs=16 判为 worse than noise。它**完全不裁剪**，所以和裁剪无关。
但它与 baseline 之间有一条更根本的差别：

| | conv / recurrent 算子 |
| --- | --- |
| baseline（`AV=false`） | `npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule` |
| 任何 AV lane | **`npu_dcut_causal_conv1d` / `npu_dcut_recurrent_gated_delta_rule`** |

两套不同 kernel，rounding 必然不同。而噪声底是 `baseline vs baseline`——**同一套
kernel 重跑的噪声**。拿「不同实现」去比「同实现重跑」的底，本来就不对等。所以这个
判定**不是新缺陷，是逐 token 判据的天花板**。

**正确性证据不在端到端比较里。** `tests/e2e/.../test_eager_gdn_varlen.py` 5/5 用独立
NumPy 参考、state 以 `rtol=0, atol=0` 精确比对——那是对 dcut 算子**比任何端到端 token
比较都强**的陈述，因为它对的是独立参考而不是另一个实现。端到端比较的作用是发现
*集成*层面的错误（query 边界、state 选择、logits 摆放），它在噪声底为 0 的并发下有效，
再往上就失去分辨力。

### 只保留 upstream 作为主路径

`threshold` lane 从一开始是诊断脚手架：同步 confidence、精确 host 边界、无成本模型，
为的是在 eager 下隔离正确性。**任务已完成**——它证明了 dcut 路径可用、ragged 布局正确。
而它的「精确 host 边界」恰恰是图不能要的东西。

所以门槛的默认 lane 改为 `upstream / upstream+ub / upstream+graph`：真正会上线的路径。
`threshold` 的代码保留（自带 env 开关，不设即不生效），作为二分工具，不删除已验证过
并已记录结果的实现。

### 高并发下的判据：接受率，不是「有没有乱码」

2026-09-21 观察：`vllm serve` 在 16 并发、`THRESHOLD=0.4` 下输出正常，无乱码、无请求
间串扰。这**基本排除了结构性状态缺陷**——16 并发比 4 并发更容易暴露这类问题（真实
padding 行、批收缩、槽位复用），而文章列举的症状正是乱码与串扰。它和门槛那边「MISMATCH
归因不了」的结论独立且一致，两者共同支持「那些 MISMATCH 是噪声」。

但**流畅 ≠ 相同**，而且文章特意警告状态错误"不一定立即崩溃…可能先被保存下来再逐轮
放大"——**先劣化的是接受率，乱码是后期表现**。所以高并发下的判据用接受率：

| | 逐 token 相等 | 接受率 |
| --- | --- | --- |
| 对数值噪声 | 极敏感（argmax 翻一次即失败） | 鲁棒（多步上的比率） |
| 对状态缺陷 | 敏感 | 敏感，且是**最早**的指标 |
| 额外价值 | 无 | **就是收益判据本身** |

vLLM 自己就在打这些数，不需要新仪器：

```bash
grep "SpecDecoding metrics" <serve 日志> | tail -3
```

对比 fixed-K 与 `THRESHOLD=0.4`（同并发、同数据集）：

- **`Mean acceptance length`**（= 1 + accepted/drafts）是唯一可直接对比的那个。裁剪
  正确时它只**略降**（放弃了一些本来能拿到的尾部接受），而单步时间降得更多，净收益
  为正；状态损坏时它会**明显下降**。
- **`Avg Draft acceptance rate`**（= accepted/verified）裁剪后应当**上升**——分母变小
  而分子基本不变，因为被裁掉的是 survival 低、本来大概率会被拒的位置。

> **一个容易误读的陷阱**：`Per-position acceptance rate` 的分母是 `num_drafts`（草稿
> 轮数）而不是该位置实际发出的草稿数。裁剪减少的正是靠后位置的草稿数，所以靠后位置
> 的 rate 会**因为裁剪本身而下降**，与状态是否损坏无关。不要把它当作损坏的证据。

净收益仍按文章 §5.3：`cost_table[完整 Q] - cost_table[选中 Q] - 测量边界外的控制开销`
（概率计算、CPU 决策、TP 广播、ragged metadata）。本分支每步三次阻塞 D2H 全在边界外。

#### 判据

- **可达 spread 明显** → 现在这套已经能支撑裁剪决策，B 的增量收益要单独论证
- **可达 spread 趋平** → 先换规模再谈 B。在测不出差异的配置上做几何工作，是把
  精力花在无法验证的地方

> 报告 spread 时只算**可达范围**（`min(max_num_seqs×(K+1), graph_limit)`）。
> 首版打的是全部 profiled 点的跨度，含这个配置永远不会出现的 Q，把一条平的可达
> 曲线显示成 900ms 的陡坡。

还有一个不在文章里的信号，读日志时一起看：**真实曲线换掉合成曲线之后
controller 还愿不愿意裁**。合成曲线是刻意造成凸的，为的是让 argmax 落在中间、
产出 ragged 布局来验证正确性；真实曲线没有这个义务。2026-09-21 首次带真实测量的
`upstream+graph` 跑出 `trimmed in 4/16 reported windows`、末尾窗口 `kept=100%`，
而同一 lane 用合成曲线时是 `16/16`、`kept=52.1%`。如果这是真实曲线的判断，那它说的
是"在这个配置下多数步不值得裁"——**这是关于收益的结论，不是缺陷**，但必须和 cost
table 的数字一起读才站得住。

> 脚本的 echo 过滤器一度只打 `" steps | "` 的数据行，把 cost table 那行漏掉了。
> 现在改成打所有非数据行（横幅、cost table，按消息去重跨 rank）加最后几个窗口。
> 旧日志里那行仍然在文件里：`grep "cost table" <日志目录>/upstream+graph.log`。

两个 capability flag（`supports_device_cpu_query_lens_mismatch`、`ALWAYS`）属于
阶段 C，是为了让上游 factory 直接接纳 GDN，**不是入图的前置条件**——本分支的两条
lane 自己替换了 factory，所以阶段 A/B 不经过那两道 gate。

## 阶段 B 的清单

「直接入全图」要分清两件事：

- **uniform FULL 已经有了**（阶段 A：`graph=FULL` 实测命中，未裁剪批走 FULL、裁剪批
  走 PIECEWISE，输出在噪声底为 0 的并发下相等）。
- **ragged FULL** 是让裁剪后的批也能命中图，需要下面六项。它们不是探索性的，来源是
  文章与 `main_dspark_adaptive_verify_dev` 的踩坑记录，逐项都有出处。

| # | 内容 | 出处 | 状态 |
| --- | --- | --- | --- |
| 1 | `B_graph = min(Q, B_max)` 描述符，`Q > B_max` 时均分 | 文章 §3.4.2/3.4.3 | 见下，**零新代码** |
| 2 | `B_fia = B_live(+1)`，或固定为服务最大值 +1 行 | 文章 §3.4.4；`39e454375` | **已在树上（15098）** |
| 3 | `B_gdn = B_max` 固定轴 | 文章 §3.4.5；本分支 `5dfe59be8` | 已实现，ragged 模式自动开启 |
| 4 | 固定地址 buffer + 批收缩时清理整个 inactive tail | 文章 §4.1；`275c0df09`、`564c7ecd9` | **已在树上（15098）** |
| 5 | 概率绑定 graph descriptor，每轮只消费一次 | 文章 §4.2；`f9542068c`、`577480c5b` | 未做，只影响随机采样 |
| 6 | drafter 未使用 KV tail 的 slot mapping 置无效 | 文章 §4.3 | 未做 |

### 第 2、4 项本来就在分支里

清单最初把六项都当成待做，这是错的。逐一对照代码后：

- **第 2 项**就是 `model_runner.py` 的 `_pad_adaptive_query_start_loc_for_fia()`：它把裁剪
  后的 `query_start_loc` 补到 `batch_desc.num_reqs`，并且把 padding token 在 padding 请求
  之间**均分**（`np.arange(1, n+1) * pad_tokens // n`），这就是文章 §3.4.3/3.4.4 的写法。
  触发条件是 `use_fia and adaptive_verification_manager and cg_mode == FULL`。
- **第 4 项**就是 `self.input_buffers.seq_lens_np[num_reqs_padded:] = 0`。

也就是说 15098 已经把 FIA 侧的请求轴和 padding 生命周期做完了，缺的只有状态算子那一侧。

### 第 1 项不需要新代码：它是一句 `varlen_decode = False` 的移除

`_is_compatible` 的规则是

```python
(desc.num_reqs is None or desc.num_reqs >= num_reqs)
and desc.num_tokens >= num_tokens
and (desc.max_query_len is None or desc.max_query_len >= max_query_len)
```

而 varlen 捕获出来的描述符是 `num_reqs = min(Q, max_num_reqs)`、`max_query_len =
decode_query_len`，注释原文就是「takes any mix of 1..decode_query_len tokens per
request」。**一个裁剪批正好满足这三条**：请求数不多、token 数不多、单请求 query 不长。
所以 ragged 回放的机制上游已经写好了，`B_graph` 也已经是 `min(Q, B_max)`——
`patch_cudagraph.py` 在批不 uniform 时走 `num_reqs = min(num_tokens_padded, max_num_seqs)`。

真正关掉它的是阶段 A 自己加的那句 `varlen_decode = False`。所以 ragged 模式在这一侧的
全部改动，就是不再执行那句；`uniform` 模式保留它。

隔壁分支 `0d6dd7559` 当初关掉 varlen 捕获，原因是状态算子看到的几何与回放几何不一致——
而文章对这个不一致的答案正是第 3 项（固定 GDN 轴）。所以阶段 B = **第 1 项 + 第 3 项**，
两者是一件事的两半，`VLLM_ASCEND_DSPARK_AV_GRAPH=ragged` 因此直接把第 3 项打开，不再
由第二个开关决定：ragged + 按桶轴是已知会坏的那个组合，不应该可达。
`VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS` 保留下来只为了在别的模式下单独二分固定轴。

### 上设备前唯一没想清楚的一点

`num_reqs_padded` 取自 **dispatcher 的** `batch_desc.num_reqs`，而图是按 **manager 的**
`BatchExecutionDescriptor.num_reqs` 捕获的，两者不一定相等：未裁剪批 dispatcher 给
`num_tokens_padded // (K+1)`，manager 捕获的是 `min(num_tokens, max_num_reqs)`（更大）。
`_is_compatible` 只要求 `>=`，所以能命中；但如果 FIA 的请求轴是**烧进图里**的，那么
padding 应该补到**捕获轴**而不是派发轴。uniform 模式下两者恰好相等（都是
`num_tokens // decode_query_len`），所以阶段 A 没暴露这一点。

这是 ragged 第一次上设备要看的第一件事。症状是局部的——FIA/attention 输出异常，而不是
状态漂移——所以可分辨；真出问题就把 padding 目标改成捕获描述符的 `num_reqs`。

第 5 项只影响随机采样（greedy 没有随机数），所以不阻塞按逐 token 判据的验收，但**上游
之前必须补**。第 6 项同理独立。

配齐后的验收判据按上面两节：正确性用算子 golden + 噪声底为 0 并发下的端到端相等，
收益用接受率与吞吐/TPOT。

## 阶段 B 实测（2026-09-21，`max_num_seqs=4`，K=7，TP=4，27B）

`upstream+ragged` **MATCH**，噪声底 identical（baseline 两遍逐 token 相同），
所以这次的相等是有效证据。判据那一行：

```txt
5 steps | mean reqs=3.40 scheduled_drafts=23.80 admitted=17.40 verify_tokens=20.80
  | kept=73.1% | trimmed_steps=4/5 | last_caps=[7] | graph=FULL=5
```

**`graph=FULL=5` 与 `kept=73.1%` 同时出现**：那个窗口 5 步全部派到 FULL，其中 4 步是裁剪
过的，所以**裁剪批确实回放了捕获的全图**。uniform 模式下这个组合不可能出现（裁剪批必然落
PIECEWISE）。全程 16/20 个窗口有裁剪。

§「上设备前唯一没想清楚的一点」里的风险（padding 补到派发轴而非捕获轴）**没有发生**：
输出在噪声底为 0 的并发上相等，说明 FIA 请求轴不是烧进图里的那种依赖。

### cost table 从平变单调，这是阶段 B 的真正回报

同一配置（bs=4）两个阶段对比：

| | 阶段 A（uniform） | 阶段 B（ragged） |
| --- | --- | --- |
| Q=8 / 16 / 24 / 32 | 116.80 / 116.80 / 116.80 / 116.80 | 33.37 / 35.89 / 39.14 / 42.13 |
| 可达范围 spread | **0.00ms** | **12.31ms**，单调 |
| controller 实际决策 | `kept=100%`（不裁） | `kept=73.1%`，16/20 窗口裁 |

因果链闭合：**曲线平 → `argmax(accepted/cost)` 永远选最大 Q → 裁剪买不到任何东西；曲线有
梯度 → 小预算真的落进更便宜的桶 → controller 开始裁。** 阶段 A 的 `kept=100%` 不是缺陷，是
对一张平表的正确决策；ragged 把表变成有梯度的，决策才跟着变。

每步 116.80 → 33–42ms（约 3×）的来源是同一件事：被 profile 的那些批在 uniform 下匹配不到
描述符、全落 PIECEWISE，ragged 下直接回放图。

这也修正了 §A″ 的一个判断。当时结论是「bs=4 测不出裁剪有没有用，要到服务规模曲线才有梯度」
——**那是 uniform 下的结论**。裁剪批一旦能回放图，bs=4 就有梯度了。服务规模的梯度依然更大，
但不再是能不能看到收益的前提。

### 这一轮几乎没有测到第 3 项（固定 GDN 轴）

`mean reqs=3.40`、`max_num_seqs=4` → 轴宽 4、活跃 3~4 行，**最多一行 padding**。而 bs=16
下批部分排空时会有多达 15 行。所以这次证明的是第 1 项（描述符几何），第 3 项只是顺带走了
一两行，padding 行处理基本没被压到。

下一轮要单独把它逼出来，而且要保住有效判据：

```bash
python examples/dspark_eager_adaptive_verify.py --lanes upstream+ragged   --max-num-seqs 16 --num-prompts 4 --max-tokens 64
```

活跃请求仍是 4（批组成稳定 → baseline 可复现 → 逐 token 判据仍然有效），而 GDN 轴被钉到
16，**凭空多出 12 行 padding**。这把「高并发本身不可复现」和「宽 padding 轴破坏状态」两件事
解耦开——这是 bs=16 直接跑做不到的，那里 baseline 自己就不相等。

### 宽 padding 轴实测通过（2026-09-22，`max_num_seqs=16 --num-prompts 4`）

`upstream+ragged` **MATCH**，噪声底 identical——`--num-prompts 4` 让活跃请求只有 4，批组成
和 bs=4 一样稳定，所以逐 token 判据在这里仍然有效。

```txt
5 steps | mean reqs=3.00 scheduled_drafts=21.00 admitted=18.60 verify_tokens=21.60
  | kept=88.6% | trimmed_steps=2/5 | last_caps=[3] | graph=FULL=5
trimmed in 20/20 reported windows
```

`mean reqs=3.00` 而轴宽 16 → **每步 13 行 padding**，同时裁剪批仍在回放 FULL，输出与定长
baseline 完全相等。第 3 项到这里才算验过，而它正是隔壁分支 `0d6dd7559` 当初坏掉的地方。

这一轮的设计意图是**解耦**：直接跑 bs=16 的话 baseline 自己就不相等，真出问题也归因不了；
把并发压在 4、只把轴放宽到 16，就把「宽 padding 轴破坏状态」从「高并发不可复现」里单独拿了
出来。

### cost table 换了量级

| | bs=4 | bs=16（4 活跃） |
| --- | --- | --- |
| `graph_limit` | 32 | **128**（16×8） |
| 图内范围 | Q=8..32 | **Q=1..128 全在图内**，单调 |
| 可达 spread | 12.31ms | **29.97ms** |
| 出图悬崖 | Q=48:229.99 | Q=192:233.59（约 4×） |

controller 现在有约 30ms 的**图内**动态范围可以优化，`trimmed` 从 16/20 升到 **20/20**。
这已经是文章里收益数据所在的那个形态——曲线在图内单调、出图是悬崖，所以「待在图里、但落进
更便宜的桶」这个决策第一次有了实际空间。

## 收益还没有被测出来（2026-09-22，specbench_80，bs=4）

修完同步拷贝、对齐图模式、少捕获 7 张图、ArgSort 回 AiCore 之后:

| | TPOT | vs 关闭 | decode token | 平均输出长度 |
| --- | --- | --- | --- | --- |
| ragged（混合批→eager） | 16.086ms | **+1.31%** | 48739（−6.5%） | 609.2 |
| ragged（混合批→piecewise） | 16.130ms | **+1.04%** | 49764（−4.5%） | 622.0 |
| 关闭 | 16.299ms | — | 52102 | 651.3 |

**回归确实修掉了**：18.10 → 16.09ms。但 1.3% 不足以称为收益，有两个独立的理由。

### 三次跑的活儿不一样

关 AV 多生成 6.5% 的 token、平均输出长 7%。TPOT 随上下文增长，所以整体差值里长度占多少
分不开。挑出长度差在 2% 以内的类目：

| 类目 | 长度 | TPOT（eager/piecewise/关） | 结果 |
| --- | --- | --- | --- |
| extraction | 82.3/81.1/81.2 | 11.89/12.23/**10.66** | AV **差 11.5%** |
| math | 462.0/459.6/460.0 | 10.43/**9.85**/10.15 | AV 好 3.0% |
| roleplay | 489.1/490.9/483.4 | **19.87**/20.24/20.45 | AV 好 2.8% |

两胜一负，负的幅度最大。

### 这组数据自带噪声估计

两次 AV 跑配置差别很小，却在 math 上差 **5.9%**、extraction 上差 2.9%、roleplay 上差 1.9%。
类目级噪声是几个百分点，整体 1.3% 落在里面。**没有重复跑就没有方差，单次跑分辨不出 1%。**

### bs=4 的天花板本来就只有约 17%

用这次的数反推：整步时间 ≈ TPOT × AvgSpecLen = 16.3 × 4.33 ≈ **70ms**；cost table 的 verify
部分 Q=32 是 **42ms**，占约 60%（其余是 K=7 次串行 drafting、采样、调度）；verify 里只有
**spread 12.31ms**（42ms 的 29%）随 Q 变化。

所以裁剪能影响的上限约是步时间的 **17%**，而且要裁到 Q=8 最狠才能拿满，代价是接受 token
大幅减少。实测 AvgSpecLen 只从 4.33 降到 4.27（−1.4%），裁得很温和。**约 1% 在预期量级内，
不是缺陷。** bs=16 的 spread 是 29.97ms、`graph_limit=128`，drafting 摊到更多请求上，头寸大
得多——文章的收益数据也在那个规模。

### 定论需要的两件事

1. **固定输出长度。** `ignore_eos=True` + 固定 `max_tokens`，三次跑生成完全相同数量的
   token，长度这个最大的混淆项直接消失。一个开关换掉它，比任何事后归一化都可靠。
2. **重复跑，并且去 bs=16。** 1% 的效应配几个百分点的噪声，单次跑无法判定。

### 降级 FULL_DECODE_ONLY 在性能上是平的

16.086 vs 16.130 差 0.3%；E2E 差 2% 但 token 数也差 2%，所以 TPOT 才是公平的数。保留它的
理由是尊重配置、少捕获 7 张图，**不是性能**。这一条不要写成收益。

## 打开这个特性为什么会变慢（2026-09-22，specbench_80，bs=4）

| | TPOT | SpecAcc | AvgSpecLen | E2E | decode token |
| --- | --- | --- | --- | --- | --- |
| `enable_adaptive_verification: true` | **18.10ms** | 46.4% | 4.25 | 229.4s | 48774 |
| 关闭 | **16.30ms** | 47.6% | 4.33 | 220.6s | 52102 |

少收 2% 的 token 却多花 11% 的时间，说明多出来的时间**不在校验宽度上**。

根因是这条 lane 自己的实现，不是动态校验。`record_confidences` 原来做
`current.cpu().numpy()`——**阻塞等这一步 drafter 刚产出的 confidence**。上游不是这样：它用
双缓冲 + 独立 stream + event，只 `synchronize()` 两步以前那次已经落地的拷贝，从不等刚
launch 出去的 draft forward。同步版每步把 host 和 device 串成一条，异步调度就完全不重叠了。

当初跳过 `super().__init__` 的理由是 `torch.cuda.Stream` 在 v2 Ascend 路径上不保证是 NPU
stream——作为 eager 预演这个取舍是对的，作为出货路径它就是「打开特性的主要代价」。

**改法**：按上游的方式建 side stream 与 event，建不出来就退回阻塞拷贝并打 warning（正确性
不受影响，只是慢）。NaN 修复挪到**设备侧** `torch.nan_to_num`——它从来不需要 host 上的值，
在 host 上修才是那次阻塞拷贝的唯一理由。

### 两个探针也不能每步付一次拷贝

`untrusted_rows` 和 confidence 活性检查都改成**设备侧累加进一个 2 元素张量**，每个日志窗口
读一次。否则为了测量这个停顿，反而会重新引入同一个停顿。

活性检查同时换了更好的形式：不再哈希 host 上的首行，而是设备侧比较首行与上一步是否相等，
累加「变化过的步数」。`conf_moved=50/50` 表示 op 每步都在重算；`conf_moved=0/50` 表示
buffer 冻结。这比指纹精确（逐步，不抽样），而且不需要任何同步。

## 已知缺陷:FIA padding 在请求槽位全满时无处可放(2026-09-23)

设备上 dynamics lane 推了约 107/500 条请求后崩:

```txt
RuntimeError: Worker failed with error 'The size of tensor a (16) must match
the size of tensor b (17) at non-singleton dimension 0'
```

现场:`num_running_reqs=16`(= `max_num_seqs`)、`total_num_scheduled_tokens=127`、
15 个 spec decode(每个 8 token)+ 1 个新请求 prefill(7 token)。

### 为什么是 17

`_pad_adaptive_query_start_loc_for_fia` 在**没有 padding 请求、只有 padding token** 时:

```python
if num_padding_reqs == 0:
    if num_padding_tokens > 0:
        query_start_loc_np[num_reqs + 1] = num_tokens_padded
        num_reqs_padded += 1          # 16 -> 17
```

而缓冲宽度是不对称的:`query_start_loc` 是 **`max_num_reqs + 2`**(放得下第 17 行),
`seq_lens` 等**只有 `max_num_reqs`**。于是 query 边界描述 17 个请求、长度只有 16 个,
几帧之后炸成 `16 vs 17`。

**这正是文章 `B_fia = B_live(+1)`(比服务上限宽一行)要解决的事。** 清单第 2 项当时记作
「15098 已做」——**只做了 `query_start_loc` 那一半**,其余每请求缓冲没有跟着加宽。

### 两种 padding,互相独立

| | 什么时候出现 |
| --- | --- |
| **请求 padding**(`num_padding_reqs`) | 描述符的请求行数 > 活跃请求数。**槽位满了就没有** |
| **token padding**(`num_padding_tokens`) | 捕获图的 token 数 > 批的实际 token 数。**和槽位满不满无关** |

文章说「有 token padding 时 `B_fia = B_live + 1`」,直觉上会以为槽位满了就没有 padding——
那是把两者混为一谈了。token padding 来自**裁剪**:16 个请求、K=7,不裁就是 `16×8 = 128`,
正好是捕获尺寸;一裁变成比如 112,dispatcher 挑最小的 ≥112 的桶(120),于是有 8 个 padding
token,却**一个 padding 请求都没有**。

**槽位满恰恰是让 token padding 无处可放的原因。** 所以 `B_live = B_max` 时仍然需要第
`B_max + 1` 行,缓冲必须按 `max_num_reqs + 1` 分配。

### 为什么 bs=4 那轮没踩到

2026-09-21 那轮 `mean reqs=3.40`、`max_num_seqs=4`:**多数步只有 3 个活跃请求**,于是
`batch_desc.num_reqs = 4 > 3`,有 1 个 padding 请求,走的是「均分到 padding 行」那一支,
根本不会 +1。

只有**正好 4 个活跃 + 恰好有 token padding** 的步才触发。那 5 步里 `trimmed_steps=4/5`,
4 请求的那步很可能正是没裁的那一个(32 token 正好是捕获尺寸,无 token padding)。

**短跑里是巧合躲过,长跑里必然撞上**——这次 500 条请求跑了几千步。这比此前那条已被推翻的
FIA 解释合理,不过仍是推断:直接证据要等护栏打出的 `num_reqs / num_tokens_padded / last_loc`。

### 完整调用链(设备日志佐证)

补充的 traceback 把链路走通了,两处改动各对应链上一环:

```txt
model_runner.py:672    seq_lens_np = self.input_buffers.seq_lens_np       ← 16 宽
mamba_hybrid.py:324    build_attn_metadata(seq_lens_np=...)              ← 不传 upper_bound
attn_utils.py:274      upper_bound = torch.from_numpy(seq_lens_np)[:17]  ← 只得 16
gdn_attn_builder.py:82 (seq_lens[:17] == 0) & (draft_tokens < 0)         ← 16 vs 17
```

`draft_tokens` 按 `num_reqs_padded` 建,是 17;`seq_lens_cpu_upper_bound` 在 model_runner
里确实是按 `num_reqs_padded` 新建的 17 长数组,**但 GDN 这条路没把它传下去**——
`attn_utils.py:274` 于是回退到 `torch.from_numpy(seq_lens_np)[:num_reqs]`,而那个 buffer
只有 `max_num_reqs` 宽。

### 只修宽度会把崩溃变成静默错误

出错那一行不是偶然的形状检查,它是契约本身。`_remove_spec_graph_padding_queries` 的
docstring 写着:

> FIA represents every padded graph request with a full K+1 query span.
> **Those rows must be zero-length for the target's recurrent GDN state update.**

也就是 GDN **靠 `seq_lens == 0` 认出 FIA padding 行**,再把 query 视图在第一个 inactive 行
处截断。而 `prepare_pos_seq_lens` 只写活跃行,padding 行留着上一个占用该槽位的请求的长度。

所以如果只把 buffer 加宽:切片不再报错,`inactive` 对那一行判成 **False**,FIA 的 dummy 行
就被当成真实请求送进 GDN 的循环状态更新——**不报错,只出错**。宽度和清零必须一起改。

### 勘误:修复方向是错的(2026-09-23 下午)

重读文章 §4.5 之后,上面那套修法是**反的**。

> **4.5 mixed batch 为什么没有强行进入 ragged FULL**
>
> ragged FULL 图的契约是纯 speculative decode。真实 prefill、普通 decode 和 speculative
> decode 混在一起时,Full Attention、GDN metadata 和 cost model 都不再是同一种几何。
>
> - 纯 speculative decode:允许 ragged FULL
> - **含真实 prefill 或不满足契约的 mixed batch:退出 ragged FULL**
>
> 这会让混部场景损失一部分性能,但**它是有意保留的正确性边界。不能为了命中图,强行把一个
> 不符合捕获契约的 batch 塞进去。**

两次设备失败的批都是 **15 decode + 1 prefill**。加宽缓冲是**让一个不合契约的批活下来**,
正是这段明确反对的做法。

而清零更糟。文章 §3.4.5 要求**保留空行**:

> 即使当前只有 32 个请求,前 32 行保存真实请求,**后 64 行也作为零长度、无效状态索引的空行
> 保留**。

`_remove_spec_graph_padding_queries` 是**截掉**它们。清零之前 padding 行带着残留长度,
`inactive` 恒为假,这个函数**从未真正执行过**;清零让它第一次生效,GDN 的请求轴于是从 B_max
变成跟着活跃数走。这正是 §4.4 记录的故障:

> 最终定位到:**不同 Q 使用了不同的 GDN 请求轴和 tiling**。把所有 bucket 的 GDN 轴固定为
> B_max,并让空行成为确定性的零长度状态后,问题得到解决。

所以整网门槛报 worse than noise 是这次清零造成的,**已撤销**。GDN 侧本来就实现了文章的契约
(`spec_state_indices_tensor[spec_batch_size:].fill_(NULL_BLOCK_ID)`),空行保留、状态索引无效,
不需要也不应该截断。

### 修复(2026-09-23)

按文章 §4.5 做正确的那件事:**ragged 模式拒绝非纯 speculative decode 的批**。

判据来自 scheduler output:正在 prefill 的请求出现在 `num_scheduled_tokens` 里、却不在
`scheduled_spec_decode_tokens` 里,所以两者长度不等**恰好**等价于批是混合的——新请求和分块
prefill 都覆盖到。崩溃那次是 16 对 15。

`_is_compatible` 看不到这件事:它只比请求数、token 数和 query 长度,而带 prefill 的批对着
捕获的 decode 图这三项全部满足。所以 ragged 模式下的 manager 覆写 `dispatch`,批不纯就直接
返回 `cg_mode=NONE` 的描述符,让它 eager 跑。

保留的是**加宽**(`B_fia = B_live + 1` 是独立且正确的,纯 spec decode 批在槽位全满且有
token padding 时同样需要那一行);撤销的是**清零**。

文章也说了后续方向:「优化混部的正确方向不是放宽判断,而是把 prefill 和 decode 拆成独立
microbatch,让 decode 子批继续回放 ragged FULL。」

按文章 §3.4.4 做了三件事:

1. **每请求缓冲加宽一行。** `AscendInputBuffers` 早就为这件事把 `query_start_loc` 加宽到
   `max_num_reqs + 2`(注释直接指向 `_pad_query_start_loc_for_fia`),但 `seq_lens`、
   `seq_lens_cpu/np`、`dcp_local_seq_lens` 都还是 `max_num_reqs`。现在都是 `+1`,dummy 行
   始终存在。
2. **padding 行置为惰性。** `prepare_pos_seq_lens` 只写活跃行,所以 padding 行会留着上一个
   占用该槽位的请求的长度,full attention 会把它当成真实序列读。现在显式把
   `[num_reqs, num_reqs_padded)` 的 device 和 host 两侧都清零——这是文章说的「零长度、
   无效状态索引」,清单第 4 项当初只看到 numpy 尾部清零就记成了「已在树上」,和第 2 项同一个
   错误。
3. **护栏保留但降级。** 现在要求第 `max_num_reqs + 2` 行才会触发,而 `B_fia ≤ B_live + 1
   ≤ B_max + 1`,所以它不该再发生;真发生了就是描述符和缓冲又对不上了。

三条 CPU 用例覆盖:槽位全满 + token padding 得到第 17 行、正好落在捕获尺寸上不加行、
有 padding 请求时按均分走。

### 一个所有 MATCH 都排除不了的失效模式

隔壁分支的 `f9542068c` 记录了这个:`compute_confidence` 被一个 Python
`if self.enable_adaptive_verification` 挡着,**捕获时若该 flag 为 False,这个 op 根本没被
trace 进 drafter 的图**,之后每次回放都跳过它,confidence buffer 冻结在捕获前的值。

对我们来说这个模式是开放的:脚本第 96 行 `"enforce_eager": not args.graph` 在
`speculative_config` 里,所以 `+graph` / `+ragged` 下 **drafter 也在图里**。
(`577480c5b` 的另一半——`ModelWithContext` 缺 `compute_confidence` 代理——已经在树上,
`aclgraph_utils.py:324`。)

**而它对逐 token 判据完全隐形。** 裁剪只是策略,拒绝采样仍然正确,输出与 baseline 逐字节
相同——只是每一步的预算都建立在停止变化的数字上。唯一的表象是「动态校验没有收益」。
`kept` 和 `last_caps` 会变也不能反驳:`scheduled_drafts` 每步不同,冻结的 confidence 配上
变化的 valid mask 同样给出变化的预算。

所以聚合行加了 `conf_moved=n/N`:窗口内 confidence 首行与上一步不同的步数。

| 现象 | 含义 |
| --- | --- |
| `conf_moved=50/50` | op 活着,每步重算 |
| `conf_moved=0/50` | **buffer 冻结**,这是 `f9542068c` |
| `conf_moved=1/1` | 单步窗口,判不了(脚本会忽略) |

探针取**固定行**而不是整批:buffer 冻结时首行每步都不变,无论坐在那一行的是哪个请求;而整批
比较会随 `num_reqs` 变化,把冻结的 buffer 报成活的。冻结的 buffer 在第一步仍会算一次「变化」
(初值是 NaN),所以脚本把 `n<=1` 都算冻结。

脚本把它列为**失败**而不是警告,理由和「只有横幅没有数据行」同一条:那样的 MATCH 不是证据。

### PIECEWISE 是另一套图,但它「有没有洞」取决于配置时的 splitting_ops

**勘误。** 此前这一节写的是「PIECEWISE 下全注意力和 GDN 都在图外 eager 跑」。那只在
`splitting_ops` 真的被设过时成立,而实际运行的配置里往往不是。

关键是**时序**:

1. **配置时**,`vllm/config/vllm.py:1741` 和 vllm-ascend `platform.py` 按**配置的**
   `cudagraph_mode` 决定 `splitting_ops`。显式配了 `FULL_DECODE_ONLY` 就走
   `has_full_cudagraphs()` 那一支,`splitting_ops = []`。
2. **之后**,`vllm/v1/worker/gpu/model_runner.py:676` 在有 AV manager 时**无条件**把
   `cudagraph_mode` 改成 `FULL_AND_PIECEWISE`——不看配置、不打 warning。
3. `resolve_cudagraph_mode_and_sizes` 的降级分支要求 `mixed_mode() == FULL`,而
   `FULL_AND_PIECEWISE.mixed_mode()` 是 PIECEWISE,所以这个覆盖**原封不动活下来**。

于是显式配 `FULL_DECODE_ONLY` 时:

| | 配置时 `splitting_ops` | 最终 mode | 那 7 张 PIECEWISE |
| --- | --- | --- | --- |
| AV 关 | `[]` | `FULL_DECODE_ONLY` | 不存在,prefill 走 eager |
| AV 开 | **`[]`,同样** | `FULL_AND_PIECEWISE` | **没切过的整模型图**,挂在混合批派发键上 |

所以那种配置下 GDN 在两套图里都在图内,而且 `enable_npugraph_ex` / static kernel 在两次
运行里相同(它们只在 piecewise 那一支被强制关掉,而那一支没走到)。**此前怀疑的编译配置
污染不存在。**

只有在真的配了 piecewise 编译时,默认 splitting ops 才会生效,而它**包含 GDN**:

```txt
vllm::unified_attention_with_output              ← 全注意力
vllm::qwen_gdn_attention_core                    ← GDN
vllm::qwen_gdn_attention_core_fused_norm_packed
vllm::mamba_mixer / vllm::linear_attention / ...
```

那种配置下 GDN 才真的在图外,而 `FULL` 那一支 `splitting_ops = []` 完全不切。这也是固定
GDN 请求轴之所以是 ragged 前提的原因:进了图的 GDN 拿不到 replay 期更新。

### ragged 不需要 piecewise 兜底,所以不接受那次覆盖

上游强制 `FULL_AND_PIECEWISE` 的理由很明确:裁剪批匹配不上 FULL 描述符,需要 piecewise
接住。**ragged 把这个理由去掉了**——裁剪批直接回放 decode 图,设备上验过两次。

所以 ragged 模式现在会把它降回 `FULL_DECODE_ONLY`,但**只在它本来就不可能是 piecewise 时**
(`splitting_ops_contain_attention()` 为假)。判据用的是上游自己的那个谓词——
`resolve_cudagraph_mode_and_sizes` 就是按它在两个模式之间选。真配了切图的运行不受影响。

`FULL_DECODE_ONLY = (FULL, NONE)` 的 `separate_routine()` 仍为 True,所以 **varlen 捕获
照样成立**,ragged 唯一依赖的东西没丢。

代价要说清楚:混合 prefill+decode 批从 PIECEWISE 掉到 eager。但**关 AV 的基线本来就是这样
跑的**,所以在这条轴上不会比基线更差。

### 耗时不要当性能读

284s vs baseline 184s（阶段 A 约 250s）。整个生成只有约 20 个 decode 步，模型加载 + 捕获
主导总耗时，ragged 捕获的桶还更多。稳态收益要用接受率与 TPOT 在长跑上量，不是这张表。

## 混合批要落到 PIECEWISE,不是 eager(2026-09-23)

第一次完整的吞吐对照跑出来是 **457.3 对 441.3 TPS,+3.6%**,噪声底 0.6%,这是决定性的。
但和崩溃那版持续 500 toks/s 相比掉了约 **9%**,而那 9% 是我只抄了文章一半造成的。

§4.5 的原文是:

> 含真实 prefill 或不满足契约的 mixed batch:退出 ragged FULL,**按场景使用 eager 或原生
> 安全组合路径**

我实现了 "eager",没实现"原生安全组合路径",而且还做了另一件事把那条路堵死:

| 改动 | 后果 |
| --- | --- |
| 拒绝混合批(`606c00e8f`) | 混合批退出 FULL |
| `FULL_AND_PIECEWISE` 降级成 `FULL_DECODE_ONLY`(`2efffaa40`) | **PIECEWISE 那一族图没了** |

两个叠加,16.8% 的步从图内掉到 eager。

### PIECEWISE 恰恰是安全的

0.28.0 自己写着:

```txt
num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
# for PIECEWISE graphs there is no limit on requests when replaying
```

**没有请求 padding、没有请求数上限**——所以混合批走 PIECEWISE 时 `num_reqs_padded =
num_reqs`,而 FIA padding 只在 `cg_mode == FULL` 时才跑。dummy 行那一整类问题根本不出现。

所以拒绝逻辑从「不是 NONE 就降级」改成 **只拒绝 FULL**。

### 但不能让 AV 自己去改图模式(勘误)

一度把 `KEEP_PIECEWISE` 默认翻成 1 来保住那一族图。**那是错的。**

关 AV 时配置是 `FULL_DECODE_ONLY = (FULL, NONE)`,没有 AV manager 就没有强制升级,
`mixed_mode()` 是 **NONE** —— 混合批走 **eager**。如果开 AV 就留在 `FULL_AND_PIECEWISE`,
混合批走 **PIECEWISE**,而 piecewise 比 eager 快。**两条 lane 对同一种批的处理就不一样了**,
dynamics 会因为一个与动态校验无关的原因显得更快——**把收益测高**。

457 对 441 那次 `KEEP_PIECEWISE` 是 0、降级生效,两边混合批都走 eager,**那个 +3.6% 是干净
的**。所以默认改回 0:开 AV 不得改变引擎对非投机批的处理。

### 项目决定:不使用 piecewise(2026-09-23)

技术上可以给两条 lane 都配 `FULL_AND_PIECEWISE` 让混合批落到 piecewise,但**本项目不这么
做**:那要多捕获一整族图,显存压力在 4×910B4 32G 上不可接受。

所以配置定死为 `FULL_DECODE_ONLY`,**混合批落 eager**,开关 AV 两边一致。

这同时确定了那 9% 的性质。此前以为是「piecewise → eager」,**错的**——崩溃那版配的也是
`FULL_DECODE_ONLY`,piecewise 从头到尾没参与过。真实的账是:

| | 混合批去哪 | TPS |
| --- | --- | --- |
| 崩溃版(无拒绝逻辑) | **ragged FULL 图(错误地)** | ~500 |
| 现在 | **eager** | 457 |

**是 FULL → eager,不是 piecewise → eager。** 那 9% 全部是把混合批请出它不该进的图的代价,
在不用 piecewise 的前提下无法靠配置回收。

唯一的回收途径是文章列为**将来方向**的那条——「后续若要优化混部……把 prefill 和 decode 拆成
独立 microbatch,让 decode 子批继续回放 ragged FULL」。注意措辞是「后续若要」:**参考实现
自己也没有做**,所以那不是抄作业,是新工作。

**「只拒绝 FULL」这一半仍然保留**:配置里有 piecewise 就用它,没有就落 eager——和关 AV 时
一致。这条改动不再决定任何事,它只是不越权。

### 一条方法论

这次是「照着文章做了一半」。那一半单独看都对(拒绝是对的、降级当时也是中性的),**合起来才
出问题**。文章把四条路径列在一起是有原因的,只取其中一条就会留下没人接的批。

## 读 vLLM 源码必须用 `git show v0.28.0:`(2026-09-23)

一天之内两次因为抄错 vLLM 签名而让设备白跑,根因是同一个:**本地 checkout 不是部署钉死的
那个版本**。

```txt
本地 checkout: v0.28.1rc0-679-gce08bb5b34
部署钉死:      v0.28.0                      ← 差 679 个提交
```

`CudaGraphManager.dispatch` 在本地有 `num_ubatches`,在 0.28.0 **没有**——照本地写的调用在
设备上就是 `takes from 5 to 6 positional arguments but 7 were given`。

**规矩:查 vLLM 的任何签名、字段、默认值,用 `git show v0.28.0:<path>`,不读工作区。**
tag 在本地仓库里,不用改动工作区就能读。

### 更好的做法是不抄

签名会变,抄了就得跟着变。所以拒绝混合批的判断放在
`_dispatch_pcp_and_sync_dp`——**本仓库自己的**拦截函数,`num_reqs` / `num_tokens` 是具名
参数、其余 `*args, **kwargs` 原样透传;要改的两个字段用 `dataclasses.replace` 按名字改,
其余字段原封不动。这样既不复述签名,也不复述字段表。

同一条规矩也适用于构造:覆写一个方法却顺手重写构造调用,是这天第一次设备崩溃的原因。

### 0.28.0 其实已经想挡住这件事

`BatchExecutionDescriptor` 的字段注释:

> `# Upper bound on per-request query length. Varlen decode graphs leave`
> `# uniform_token_count unset, so this is what keeps a prefill batch out of one.`

varlen decode 图按 decode 宽度捕获,批里最长 query 超过它就匹配不上。**但 prefill 比
decode 宽度还短时会漏**:两次设备失败的 prefill 是 6 和 7 个 token,而 decode 是 8,批的
最长 query 仍是 8,于是匹配成功。我们加的判断是**补上这个漏洞**,不是另起一套。
