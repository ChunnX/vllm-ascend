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

**本分支现状：`gdn_attn_builder.py` 里 `spec_batch_size = m.num_reqs`，即请求轴
跟随实时批。这正是上面被推翻的设计，进 ragged FULL 前必须改成 `B_max`。**

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

| 阶段 | 内容 | 判据 |
| --- | --- | --- |
| 1（已完成） | eager 下把状态算对 | 算子 golden 5/5；整网四条 lane 逐 token 相等 |
| 1.5 | 放弃精确 host 视图（`VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND`） | `+ub` lane 仍然 MATCH |
| 2a | PIECEWISE：稳定区域入图，GDN 与部分 attention 保留为图边界 | 输出相等；**真实 cost table 随 Q 单调** |
| 2b | dcut GDN 算子也进 PIECEWISE | 输出相等；cost 进一步下降 |
| 3 | ragged FULL：`B_gdn=B_max`、`B_graph=min(Q,B_max)`、`B_fia=B_live(+1)`、固定地址、descriptor 绑定概率、mixed 路由契约 | 输出相等；cost table 相邻 bucket 可区分 |
| 4 | 翻 capability，让上游 factory 直接接纳 GDN，撤掉自建 manager | 不设 env 也能跑；可上游 |

阶段 2a 有一个重要简化：**PIECEWISE 第一步不需要 GDN 声明 `ALWAYS`**，因为 GDN 仍
是图边界。需要的只是不再强制 `CUDAGraphMode.NONE`。两个 capability flag 是阶段 4
为了让上游 factory 接纳 GDN 才需要的，不是入图的前置条件。
