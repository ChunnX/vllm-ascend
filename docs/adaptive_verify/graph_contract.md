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
| **A″** | 在服务规模并发下重测 | 可达 spread 明显且随 Q 单调 | 待上机 |
| **B** | ragged 图：`B_graph=min(Q,B_max)`、`B_fia`、固定地址、descriptor 绑定概率 | 输出相等；cost table 相邻 bucket 可区分 | 取决于 A″ |
| C | 翻 capability，让上游 factory 直接接纳 GDN，撤掉自建 manager | 不设 env 也能跑；可上游 | 未开始 |

### 阶段 A 的实测结果：裁剪批落到 PIECEWISE，不是 eager

2026-09-21 第二次跑（带 `graph=` 仪器）：

| lane | `graph=` | `kept` | 实际路径 |
| --- | --- | --- | --- |
| `threshold:0.0+graph` | `FULL=5` | 100% | 未裁剪 → **FULL** |
| `threshold:0.4+graph` | `PIECEWISE=5` | 67.1% | 裁剪后 → **PIECEWISE** |

两条都 MATCH。第一条证明图**真的进去了**，阶段 A 成立。

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
