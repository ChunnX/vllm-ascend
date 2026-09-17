# D-Cut GDN 接入设计（阶段 B0/B：图契约 + 人工 cap 打通真实变长执行）

> 目标：Qwen3.6-27B（GDN）用 vendor 的 `npu_dcut_*` 算子完成**连续多轮人工变长验证**，
> 输出 + 有效 GDN/KV 状态对照参考通过。与 confidence/预算解耦——先用人工 cap 打通并钉死状态
> 正确性，再谈真实 AV manager（阶段 C）。
>
> 原设计与历史行号基准：`76780b1a1`（Ascend）/ v0.28.0（vLLM）。
> 本次契约复核基准：`2bb8a9c5c`（Ascend）。开发基线为 vLLM `v0.28.0` tag +
> vllm-ascend main 上保留既有 DSpark 优化的开发分支；每次验证固定运行端两个仓库的 SHA。
>
> **2026-09-15 修订**：上一版把"替换两个算子"和"模型级裁剪"混为一谈，且有多处接口语义错误
> （`actual_seq_lengths` vs `query_start_loc`、`num_accepted` 被误裁、state index 布局、注入层）。
> 本版先把**接口契约**定死——契约错了就是"算子能调、单轮对、跨轮状态已错"。
>
> **2026-09-16 修订**：算子契约（§1.1–1.4）已由 UT + eager 冒烟印证；新暴露的是**图层契约**
> （§1.6、§1.7）。cap=99 在 FULL 图下仍乱码、接受率全 0，而 eager 下 cap=99/2 精度正确，
> 排查重点是捕获/回放的分支与缓冲区契约；人工 cap 不依赖 confidence 决策，但 eager 冒烟
> 不能排除算子入图后的问题。`2bb8a9c5c` 已修复重复 `build()` 与零 draft 捕获分支，
> 尚不能据此宣称 FULL 精度问题已解决。本次新增六个轴、F3 观测和待验证的固定请求轴方案。

> **2026-09-17 修订（阶段 B0 图契约通过）**：cap=99 + FULL_DECODE_ONLY 在 `max_num_seqs=8`
> 下 1/2/4/8 个请求精度全部正确。根因是两个缺陷叠加，都与算子契约无关：① `2bb8a9c5c` 的
> batch 级记忆化缓存了整个 metadata 对象，而该对象携带 per-group 的 `block_table_tensor`，
> 导致 10 个 Mamba KV cache group 里 9 组按第 0 组的 block table 推导状态索引（`e41230fc7`
> 修复）；② 开启 AV 会把 decode descriptor 切成 varlen 形态，`num_tokens <= max_num_seqs`
> 的 bucket 一律按 1 token/请求编图，而真实 spec batch 回放的是 1 请求/校验宽度——GDN 层没有
> replay 期参数刷新，捕获几何就是唯一几何（`0d6dd7559` 改回 uniform 描述符）。§1.6 已解决，
> §1.7 在 cap=99 + uniform 描述符下不再被触发，进入裁剪阶段才重新上线。

## 1. 接口契约（先定死，再实现；每条都要在动手前明确）

### 1.0 六个轴与请求身份

这些概念必须分别表达；模型级 compaction 仍复用上游 AV 链路，GDN builder 只消费其结果。

| 名称 | 定义 | 数据来源与约束 |
|---|---|---|
| `B_max` | 服务允许的最大请求数 | `max_num_seqs`，是容量上限，不是当前真实 batch size |
| `B_live` | 本轮真实请求数 | 裁剪后的请求映射；cap=0 仍计为真实请求，不含 padding |
| `Q` | 本轮真实 query token 总数 | 纯 spec decode 时为 `sum(1 + cap_i)`，不含图 token padding |
| `B_graph` | 所选图 descriptor 的请求容量 | 从实际 descriptor 读取；当前 MRV2 varlen 策略为 `min(Q_bucket, B_max)` |
| `B_fia` | FIA 实际消费的请求行数 | 从传入 FIA 的 query/KV 边界及 block-table 视图读取，不从 `B_graph` 猜测 |
| `B_gdn` | GDN 算子实际消费的请求轴长度 | 从 conv/recurrent 的 qsl、状态索引表和 accepted 张量形状核对 |

`Q_bucket` 另记为实际图的 token 容量，必须与真实 `Q` 分开。上述 `Q` 不用于表示 capture size。
捕获没有真实服务请求时，日志保留 `B_live=null` 并记录 `B_dummy`、`is_dummy=true`；不能把
dummy 请求数写成线上真实并发。pure-spec 回放要求 `B_live <= B_graph`、`B_live <= B_gdn`、
`Q <= Q_bucket`，各局部视图中的行到请求映射一致。

请求身份由显式真实请求数或 mask 传递，独立于 KV 长度、本轮 query 长度和上一轮 accepted。
FIA 可以有自己的 dummy query/KV 行；GDN 的 inactive 行必须零 query，不得因此更新状态。
`B_fia = B_live + 至多一行` 是作者的一种实现选择，不作为 FIA 的通用行数限制。

### 1.1 recurrent 传**累计** `query_start_loc`，不是旧 `actual_seq_lengths`
现有 `_build_actual_seq_lengths()`（`ops/gdn_attn_builder.py:162`）存的是 `[起始, 逐请求长度]`：
长度 2、3 → `[0, 2, 3]`。而 dcut recurrent 要**累计偏移** `query_start_loc = [0, 2, 5]`。
**误传会把第二个请求当成长度 1。** → dcut recurrent 传对应 spec 子批次的 cumsum，不能沿用旧入参名/值。

### 1.2 `ssm_state_indices` 是 `[B, S]` 二维表，**不 flatten、不随 token 数压缩**
- 旧调用 `spec_state_indices_tensor.flatten()`（`ops/gdn.py:521`）是给旧算子的 1D 布局。
- dcut host 要求 `ssm_state_indices` 为 `[B, S]`；内核用 `batch_i * S + accepted - 1` 读起始状态、
  在该请求行的对应列存本轮候选状态。
- **契约**：token 可按当前长度紧凑排布，但**状态索引表必须保持 `[B, S]`**，仍能定位"上一轮接受
  位置"对应的状态；去掉 `flatten()`；**请求重排时整行 `[.,S]` 一起重排**。
- **S 与容量（代码已定，照抄即可）**：`S = num_spec + 1`；预分配行容量
  `decode_cudagraph_max_bs = max_num_seqs * (num_spec + 1)`，再被
  `compilation_config.max_cudagraph_capture_size` 截（vLLM `gdn_attn.py:118-128`）。
  **要区分"缓冲区容量"与"本次调用的 B"**：当前 pure-spec FULL 路径用 `m.num_reqs`
  切出本次 GDN 请求轴。固定 `B_gdn = B_max` 是 §1.8 的待验证选择；改切片前必须确认
  所有底层 buffer 容量足够，不能假定 `decode_cudagraph_max_bs >= B_max`。

### 1.3 人工 cap **不截断** `num_accepted_tokens`（两个独立量）
- **本轮 cap**：决定本轮处理几个 query。
- **上一轮 accepted**：决定本轮从哪个历史状态起步（内核按固定状态行宽用它索引）。
- **契约**：**禁止** `num_accepted = min(previous_accepted, current_query_len)`。上一轮选 6，本轮只处理
  2 个 query，仍必须读第 6 个状态。并写清 `num_accepted` 是否含 anchor（与"接受的 draft 数"区分）。

### 1.4 人工 cap 是**模型级 compaction**，且要注入**逐请求 capacities**（不是只给总预算）
只把累计偏移改成 `[0,2,5]` 而不重排真实输入，请求 2 会读到 `A2 A3 B0`（错）。
- **契约**：cap 必须**统一作用于真实输入及其全部派生布局**：input_ids、positions、attention/GDN 的
  query 边界、KV slot mapping、验证采样的 draft/logits 索引与计数。GDN builder **消费**已经一致的
  裁剪结果，不能单独决定裁剪。
- **关键**：`compact_batch()`（vLLM `adaptive_verification.py:337`）只按**总预算在 CPU 侧均分**
  draft 数；真正的**逐请求容量**在 `reallocate_drafts()`（`:377`）里由 `_assign_draft_token_budget`
  (confidence) 产出 `capacities`。**只喂"总预算"得不到你指定的逐请求 cap**（人工 [0,3] 只设总预算 3
  不保证还是 [0,3]，也没真解耦 confidence）。
- **人工注入方案**：
  ```
  request_id → 本轮保留 draft 数 (manual capacities，按最终请求顺序)
      ↓  跳过 confidence 排名，直接提供合法 capacities
  capacities[i] = min(manual_caps[i % len], scheduled_drafts[i])   # 受实际 scheduled draft 约束
      ↓
  总预算 = sum(capacities)；保留原有非 draft token 数；query = 1 + cap（仅普通 decode 请求）
      ↓  复用后续累计偏移 / 输入构造 / 采样布局（与真实 AV 同一条路）
  ```
  人工模式跳过 confidence、直接给 capacities；真实 AV 仍由原策略生成 capacities。**共用下游执行路径，
  不维护第二套 compaction。** → 独立算子测试只证算子对，不算"模型连续多轮验证"完成。

### 1.5 cap 语义 + cap=0 分支（2026-09-16 更正）
- **定义**：`cap = 保留的 draft token 数`；普通 decode 请求 target query 数 = `1 + cap`。
  → 有效请求 `cap=0` 仍有 1 个 anchor query，**不等于 padding 空请求**。
- **更正**：原先写"整批无 draft 则进普通 decode"——**在裁剪下不成立**。builder 的 spec 分类
  （`ops/gdn_attn_builder.py:632-643`）消费的是 `model_states/mamba_hybrid.py:74-77` 用**裁剪前**
  `num_draft_tokens_per_req` 算出的 `num_decode_draft_tokens_np`；cap=0 时裁剪前 draft 数仍 > 0，
  mask 仍为真，所以**全批 cap=0 也还在 spec 分支，每请求 1 个 query**。
- 因此测试用例改为：① 混合 batch 里部分 cap=0；② 全批 cap=0 **仍在 spec 分支**；③
  **上一轮多 query spec → 本轮全零 cap → 下一轮恢复 spec 的状态衔接**。
  原先"全批 cap=0 → 普通 decode → 恢复 spec"那条测的是裁剪永远不会走的路径，删除。
- 反过来，`num_decode_draft_tokens == 0`（不是 `-1`）这种行**只由 dummy 捕获产生**
  （vLLM `gdn_attn.py:546` 用 `diff(query_start_loc) - 1`），这一点是 §1.6 的基础。

### 1.6 捕获与回放必须走**同一条算子路径**，且**同一几何**（2026-09-17 已解决）

**结论**：分支一致（`2bb8a9c5c`）是必要条件但不充分。真正让 cap=99 FULL 乱码的是**几何**：
`varlen_decode` 让 descriptor 变成 `min(num_tokens, max_num_reqs)` 行、dummy 均分 token，
于是 `num_tokens <= max_num_seqs` 的每个 bucket 都按 **1 token/请求**编图，而真实 spec batch
回放的是 **1 请求/校验宽度**。满足 `min(Q_bucket, B_max) == Q_bucket/(K+1)` 的只有
`Q_bucket = B_max*(K+1)` 一个解，所以除了恰好满并发，**每个 bucket 的几何都是错的**；
`max_num_seqs=1` 之所以正确，是 varlen 在那里退化成了 uniform 几何，不是并发低。

**为什么 GDN 不能像 full attention 那样容忍**：full attention 每步用 `graph_task_update`
以刷新后的主机侧长度重发 kernel；linear_attn 层没有任何 replay 期刷新——`attention_v1.py:802`
的注释称它们"由 `update_conv1d_graph_params` 单独更新"，但**该函数在全仓不存在**，只出现在
这条注释里。`zip` 过滤是真的，承诺的另一条路径是空的。所以捕获进 conv/recurrent task 的
几何就是它们唯一会跑的几何。

**修法（`0d6dd7559`）**：D-Cut 路径保留 uniform 描述符（`round_up(Q, K+1)/(K+1)` 行 × `K+1`
宽），即不带 D-Cut 时已验证通过的那批图。裁剪后的 batch 不是 uniform，匹配不到 FULL
descriptor 而回落——这比回放一张为别的请求布局编的图要好，也是裁剪未验证阶段的正确分层。
可用 `VLLM_ASCEND_DSPARK_DCUT_UNIFORM_DECODE_GRAPH=0` 恢复旧行为做对照。

**以下为历史缺陷链（`76780b1a1`），保留作分支一致性的依据：**
FULL 图的 Python 分支在捕获那一刻定形，回放不重选分支。所以 GDN 的 `spec` / 普通 decode 分支
选择必须在捕获和回放之间**恒等**。以下为 `76780b1a1` 的缺陷链，`2bb8a9c5c` 已修复
分支选择；它是否足以解决模型故障，仍需运行端 FULL 验证：

1. AV manager 非空 → base 覆盖 `cudagraph_mode = FULL_AND_PIECEWISE` 并传 `varlen_decode=True`
   （vLLM `model_runner.py:574-587`）。
2. varlen descriptor 的请求数 = `min(num_tokens, max_num_reqs)`（vLLM `cudagraph_utils.py:227`），
   `make_dummy` 均分 token → **`num_tokens <= max_num_seqs` 的每个 bucket 都是 1 token/请求**。
3. 捕获期 draft 数 = `diff(query_start_loc) - 1` = 全 0。
4. builder 见"runtime draft 数之和为 0"就把 spec mask 清零转普通 decode
   （`ops/gdn_attn_builder.py:632-643`）→ **图里捕获的是普通 decode 的 conv/recurrent**。
5. 回放时真实 batch 有 draft → 构造 spec metadata，但执行的仍是捕获的普通 decode 算子及其
   捕获期缓冲区地址，形成精度风险。这能解释异常的一种机制，不能证明是唯一根因。

- **契约**：`num_decode_draft_tokens == 0` 表示"spec 分支里的 1-query 行"，**不表示"转普通
  decode"**。语义上二者等价（`num_accepted = 1` 时读写的都是 `ssm_state_indices[b, 0]`
  = `block_table[b, 0]`），所以统一到 spec 路径是安全的、且让捕获与回放同路。
- **范围限制**：这个改动只能影响捕获路径。真实 batch 里 0 draft 的行在
  `model_states/mamba_hybrid.py:77` 已被写成 `-1`（非 spec），因此 draft 数恰为 0 的行只由 dummy
  捕获产生；改动必须 gate 在 D-Cut 开关上，不能改变非 D-Cut 配置（含 MRV1 dynamic spec 里
  `decode_query_len == 1` 的 uniform decode 图，那种图本来就该捕获普通 decode）。
- **bucket 差异要记住**：`num_tokens > max_num_seqs` 的 bucket 捕获时**本来就走 spec**
  （例：40-token 图 = 24 行长度 1 + 8 行长度 2，存在正 draft 计数），所以"修掉小 bucket"
  **不足以**覆盖 §1.7。

### 1.7 padding 请求行必须是**零长度**（2026-09-16 新增；cap=99 下当前不触发）

> **2026-09-17 状态**：uniform 描述符下 `B_graph == B_live`，cap=99 一个 padding 行都没有，
> 因此本节的执行者 `_remove_spec_graph_padding_queries` 在当前能跑通的配置里**不被执行**，
> 也就没有实跑覆盖。进入裁剪（ragged replay）阶段会全部重新上线，届时必须补图级用例，
> 不能只靠单测。
- **目标契约**：FIA padding 行可以有正 query span，但 GDN recurrent 要求 inactive 行零长。
  执行者是 `_remove_spec_graph_padding_queries`（`ops/gdn_attn_builder.py:57`）。
- **仍需验证的布局**：`num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)`
  （`worker/v2/model_runner.py:351`）里的 `num_tokens` 是裁剪后的值，所以裁剪后 token 数一旦
  不正好落在 capture size 上，`_pad_adaptive_query_start_loc_for_fia`（`:1112`）就把正长度
  均摊给 padding 请求。复现布局：11 请求 / cap=2 / 33 token / 40-token 图 / 32 请求槽 → 21 个
  padding 行里 7 个拿到长度 1。
- **归一化未生效时的后果链**：这些行 draft 分类是 `-1` → 非 spec → 与真实 spec 行混合后计入 `num_prefills`
  → **跳过 `:1041` 的 pure-spec 持久缓冲区块**（该块要求 `num_prefills == 0 and num_decodes == 0`）
  → spec metadata 退回每步新建的临时张量 → FULL 回放读捕获期地址。
- **修复状态**：`2bb8a9c5c` 已通过 batch 级共享局部视图恢复执行者，见 §5 步骤 ⑤。
  但当前识别式仍是 `(seq_lens_cpu_upper_bound == 0) & (draft_tokens < 0)`，必须由 F2 改成
  显式请求身份，不能把 KV 长度为零当成 padding 的可靠来源。
- **CPU/device 缺口**：MRV2 当前只刷新 CPU `seq_lens_np[:B_live]`、清理 `[B_graph:]`，中间
  padding 区间可能残留；vLLM `v0.28.0` 的 `prepare_pos_seq_lens` 则有额外 Triton program
  将 device 未使用行清零。不是“两侧均残留”。FIA 的 dummy KV 长度修正只作用于其局部视图，
  不得全局把共享 CPU/device padding 填 1，使 GDN 再次丢失 inactive 身份。
- **关于"污染"**：padding 行写的 `NULL_BLOCK_ID = 0` 是 vLLM 保留的 null block，**这一个动作
  本身是设计意图**；但不等于整条混合路径安全，仍需按有效状态索引与有效输出验收，不提前给
  安全结论。
- **接回时要验**：`_remove_spec_graph_padding_queries` 会把 `num_actual_tokens` 改成真实
  （未 padding）token 数。需确认 FULL 捕获期（bucket 大小）与回放期（真实值）之间没有被捕获
  进图的形状假设被破坏——`ops/gdn.py:305` 的 `mixed_qkv[:num_actual_tokens]`；SPEC_ONLY 分支
  不使用 `spec_token_indx`（`ops/gdn.py:312-314`），所以它的长度变化在该分支无影响。

### 1.8 状态不变量与固定请求轴实验

以下是不依赖具体图实现的验收契约：

- 每个真实 decode 请求至少保留一个 anchor；只裁 draft 后缀，不从中间挖洞。
- 上轮 accepted 决定历史状态选择，不被本轮 cap 或 query 长度截断；请求重排时状态索引整行同行。
- GDN 空行的 query 长度为零、状态索引无效；算子必须在读取状态前跳过空行。
  累计 qsl 的 inactive 尾部重复最后一个有效累计边界，不是将非空 batch 的 qsl 尾部直接填零。
- padding token 不得写入真实 KV 或 conv/SSM 状态；batch 收缩时刷新图可读到的整个 inactive tail。
- 同一捕获图的输入地址、shape、stride、storage offset 符合捕获契约，replay 前内容刷新完成；
  不要求不同图必须共用地址，共用 storage 时必须验证不同视图间的刷新和执行顺序。
- uniform target、ragged target、drafter 的图资源不可混用。descriptor 隔离是行为要求；
  分表已满足时不重复建表，只有发现键冲突或资源别名错误时才改变索引方式。
- TP 各 rank 使用同一请求映射、cap、query 边界与裁剪计划，以实际比较验收。
- confidence 不可信时保持完整验证长度；图 metadata 不可信时退出该图，重建合法 metadata 后
  走已验证路径。不能拿坏 metadata 仅关闭裁剪继续 replay。
- prefill、mixed、PD handoff 有明确路由边界；纯 spec decode 才进入本阶段 ragged FULL。
  未裁剪 batch 也必须匹配所选图的契约，不能仅凭“不裁”认定可用 uniform 图。
- `NULL_BLOCK_ID=0` 与 `PAD_SLOT_ID=-1` 语义不同。D-Cut GDN inactive 行按算子契约使用无效
  状态索引，FIA dummy block row 与原生 decode 路径分别核对，不做全局 sentinel 替换。

**`B_gdn = B_max`：待验证的实现选择（来源：上游作者经验，我们尚无同根因证据）。**
用户于 2026-09-16 提供的文章 §3.4.5/§4.4 称固定请求轴解决了跨 bucket 精度异常，用户说明
文章对应 [PR #15207](https://github.com/vllm-project/vllm-ascend/pull/15207)。已审阅的该 PR
算子版本不等同于文章完整 FULL 接线；尚无对应固定轴接线 commit 可供核对。

F1′ 只在 D-Cut pure-spec FULL 内实验：qsl 为 `[B_max+1]`、state indices 为 `[B_max,S]`、
accepted 为 `[B_max]`，mask 和派生 metadata 同步调整。先审计分配容量，再发布固定形状视图；
有效行与 inactive tail 一并刷新，各 KV cache group 保留各自状态归属。普通 decode 路径不顺手
改轴。当前轴方案与固定轴方案做 F4 对照；不同图具有不同请求轴本身不是错误证据。

### 1.9 F3 观测契约：记录实际轴，而非由配置推断

观测位置固定为 **capture 准备 / replay metadata 刷新 / 图 dispatch**。先采未修基线，再进行
固定轴实验；不依赖 FULL replay 重新执行模型内 Python 分支来打印日志。

每条记录包含 `phase`、capture/step ID、TP rank、target/draft 身份、KV group/layer、实际 backend、
完整 descriptor、原生/spec 分支及路由结果。必须带齐 **`B_max / B_live / Q / Q_bucket /
B_graph / B_fia / B_gdn`**。准备阶段尚未构造的轴填 `null` 并说明原因，通过同一 ID 关联完成态，
不得从另一条轴猜测；capture 的 `B_live` 按 §1.0 标注 dummy。

对应张量记录 `shape + data_ptr`，并补充 dtype/device、stride、storage offset 和有效刷新范围：
query 边界、FIA CPU/device KV 长度及 block table、GDN 状态索引 `[B,S]`、accepted、请求 mask、
KV slot mapping。CPU numpy 视图记录自己的地址，明确区分 host/device 指针。
在有界诊断样本中记录 qsl、KV 长度、accepted、请求映射及 inactive tail 的值或一致性检查结果，
使“轴是否随 bucket 变化”与“地址相同但内容是否陈旧”都可核对。

日志默认关闭、限量采集；device 数值回读放在捕获区域外，避免在 capture 内新增同步或改变图。
记录 stream/event 刷新完成关系；诊断开销不计入阶段 D 的性能数据。cap=99 通过仅算冒烟，
必须继续覆盖单图/多图捕获、切图、收缩/增长和状态不变性。

### 1.10 文章经验的后续审计项与 DSpark 边界

| 经验来源 | 本项目审计内容 | 证据与适用边界 |
|---|---|---|
| 文章 §4.2：概率绑定图与固定输出地址 | 核对 confidence 输出对应的图/执行、请求集合和轮次；每份结果只消费一次，失效则完整验证 | 我们复用 DSpark confidence head 与 AV 管线，不照搬 DFlash selected-token softmax；人工 cap 不依赖这些概率 |
| 文章 §4.3：DFlash 未使用 KV tail | 查清 DSpark 自己的 hidden-state/context padding 与 slot mapping，若存在无效写入则屏蔽对应 slot | 这是 DFlash 的已述故障，尚不是 DSpark 的已证实缺陷；保留真实 token 的 KV 写入 |
| 文章 §4.1：清理 inactive tail | 核对每张图能读取的全部尾部和跨图共享视图，覆盖 batch 收缩、增长及请求槽位复用 | §1.8 是目标契约，审计尚待完成；不得用一次 cap=99 冒烟代替 |

文章认为 D-Cut 与 DSpark 更适合组合，这是作者判断，不替代本模型的收益验证。
通用状态与图资源契约可以采纳，DFlash 的故障与开销必须先映射到 DSpark 实际调用链。

## 2. 两个 hook 点（GDN spec 路径）

`ops/gdn.py::_forward_core` 的 spec 分支，已由 `VLLM_ASCEND_DSPARK_ENABLE_DCUT` 门控：

| 位置 | 原算子 | 换成 | 注意 |
|---|---|---|---|
| `ops/gdn.py:332` conv1d | `npu_causal_conv1d_custom` | `npu_dcut_causal_conv1d` | 传裁剪后 `query_start_loc`(cumsum)+`cache_indices`+`num_accepted`(按 §1.3 不截断)；无 `run_mode`/`initial_state_mode` |
| `ops/gdn.py:498` recurrent | `npu_recurrent_gated_delta_rule` | `npu_dcut_recurrent_gated_delta_rule` | 传 cumsum `query_start_loc`(§1.1)、`[B,S]` `ssm_state_indices`(§1.2)、`num_accepted`(§1.3)、`zero_padded_output=False` |

**不是"改个算子名"**：recurrent 的 `actual_seq_lengths`→`query_start_loc`（值和语义都变）、state index
去 flatten 保 `[B,S]`，都是入参契约变化。

## 3. 首个里程碑

Qwen3.6-27B 连续多轮人工变长验证：**已提交 token 序列正确** + **有效 conv/SSM/KV 状态逐轮对齐参考**。
前置是 §1.6/§1.7 的图契约——**在 FULL 图产出正确输出之前，模型级状态正确性无法验收**。

## 4. 测试参考点（避免比错对象）

- **算子级**：相同初始 state / 输入 / 人工长度，用**独立顺序参考**算，检查输出与候选状态；明确
  **数值容差 vs bitwise**（dcut 内核与参考实现不必 bit 相等，定 tol）。
- **模型级**：按**同一请求、同一已提交 token 前缀**对齐，比有效 conv/SSM/KV 状态；**未提交的候选槽位
  不要求与不裁剪运行相同**（裁剪与不裁剪同一轮结束可能到不同 token 前缀，不能直接比整体状态）。
  这条口径是唯一口径，**不按迭代轮次硬对齐**。
- **图级（新增）**：同一 bucket 下比 eager 与 FULL 的输出与有效状态；覆盖
  **8/16/32/40-token 图**、**bucket 内部落点**（token 数 ≠ capture size，制造正长度 padding）、
  1/2/4 个真实请求、不同 padding 数量、cap=99/2/0。
- **跨图 F4**：只捕获单图 vs 捕获全部 bucket 后回放同一张 8-token 图；小图→大图→小图；
  batch 收缩/增长、请求重排/槽位复用；逐 bucket 请求轴 vs 固定 `B_max`。
  conv + recurrent 输出和有效状态均对照，包含 `num_accepted > 本轮 query 长度` 与 padding
  状态不变。capture/warmup 会修改状态，每组比较前恢复相同初始状态，跨轮组内保持真实连续性。
- **TP**：人工 cap 阶段就检查**各 rank 对同一请求的 cap 和布局一致**（不必等真实 AV）；人工 cap 是
  确定性的，应作为可验证条件写出来，而不是"天然一致"。已跑通的 TP4 只能记为**功能冒烟**，
  要逐项比**总预算 / 逐请求 capacities / 请求映射 / query 边界**。

## 5. 实施顺序（2026-09-16 重排：图契约插到最前）

| 步骤 | 工作 | 状态 / 通过标准 |
|---|---|---|
| ① | GDN spec 两处切 D-Cut，纠正累计偏移(§1.1) + 二维状态表(§1.2)，**保持原验证长度(不裁)** | ✅ NPU 回归过：开/关 `ENABLE_DCUT` 输出逐词一致 |
| ② | 跨轮算子参考 + NPU 对照（`_dcut_recurrent_golden` 扩多轮 + conv 多轮） | ✅ 910B 4 个 UT 过；**待补**见下 |
| ③ | 人工 capacities 接进现有 MRV2 裁剪链路（§1.4：`reallocate_drafts` 注入，跳过 confidence） | ✅ 6 个 CPU 策略 UT 过；eager cap=99/2 精度正确 |
| ⑤ | **修 `build()` 重复定义**，把两个前置处理接回生效路径（§1.7） | `2bb8a9c5c` 已提交，图回归未验收 |
| ⑥ | **统一捕获/回放算子路径**（§1.6） | `2bb8a9c5c` 已提交，图回归未验收 |
| ④ | 连续多轮模型验证 | 待 F0/F3 基线、F4 与 F1′/F2 实验：缩短→恢复、部分/全部 cap=0、重排、TP 布局一致性 |

当前顺序：**F0 → F3 → F4 最小测试 → F1′ + 必要 F2 → 模型回归 → F5/F6 → F7**。
F6 的支持边界在实验开始前明确，最终能力声明在验证后落地；FULL_DECODE_ONLY 是目标，
PIECEWISE 是诊断与性能对照。各 F 项见 [开发计划](dcut_development_plan.md) 阶段 B0。

- **⑤⑥ 要一起验，不要分开宣称修好**：单请求 cap=99 命中 8-token 图时无 token padding，
  主要覆盖分支修复；cap=99 在其他 batch/bucket 组合下仍可能有 padding。必须另测 bucket
  内部落点来覆盖 ⑤，不能只按 cap 大小判断是否需要 padding 归一化。
- 模型状态按**同一已提交前缀**比较；浮点算子**明确容差**，不把"逐元素一致"默认 bitwise。

#### ⑤ 的具体落地（历史缺陷，已由 `2bb8a9c5c` 修复）
- `AscendGDNAttentionMetadataBuilder` 里 `build` 定义在 `ops/gdn_attn_builder.py:591` 和 `:979`，
  **Python 生效的是 `:979`**。`:591` 来自上游合入（`fa50f6ae8`，2026-09-15），`:979` 来自本地
  shared-metadata 性能重构（`cff680f36`，2026-08-18）。
- 后果：`_remove_spec_graph_padding_queries`（`:57`）与
  `_treat_single_token_prefills_with_state_as_decodes`（`:89`）的**唯一调用点都在死代码里**
  （`:600` / `:604`），两个前置处理全部不可达。
- **不能只删前一个定义**：两者都会重建 `query_start_loc`，而 shared plan 的缓存键按 tensor
  **地址**取（`:859` `_shared_batch_plan_key`）。若在生效的 `build()` 里逐 KV cache 组各做一次，
  每组会得到不同的 `query_start_loc` → 缓存全 miss、各组按组内视图刷图缓冲区。必须
  **每次 `build_attn_metadata` 只做一次**，挂在已有的 batch 级 `batch_shared_cache` 上
  （`worker/v2/attn_utils.py:282` 建、`:349-354` 只在非捕获路径传入）。

#### ③ 的具体落地（人工 capacities 注入，跳过 confidence）
- **策略函数**（`worker/v2/spec_decode/dspark/dcut_manual_cap.py`，numpy-only、可 CPU UT）：
  `compute_manual_capacities(scheduled_drafts, manual_caps)`——`manual_caps` 是**逐位置 cap pattern**，
  batch 位置 `i` 用 `manual_caps[i % len]`，负数表示该位置不裁；结果逐元素 `<= scheduled`（合法
  capacity）；`manual_batch_budget()` 出 `draft_budget=Σcap` **以及按 req_id 索引的逐请求 capacity**。
  **不吃 confidence 参数——结构上无法依赖 confidence。**
- **pattern 而非单值,是 ③ 能验变长的前提**：单个全局 cap 把每个请求裁到同一宽度,稳态下
  batch 是 `cap+1` 的**均匀**批次——在图这一层等价于 `num_speculative_tokens = cap`,
  **根本造不出 D-Cut 要的参差布局**。`7,0,3,1` 这样的 pattern 才会真参差。
- **manager 子类** `DcutManualCapVerificationManager`（同文件，惰性工厂建）：只重写两处 confidence 入口
  `get_num_tokens`（预算=Σ人工 cap，不查 cost table）+ `reallocate_drafts`（`capacities` 直接来自
  策略函数，替掉 `_assign_draft_token_budget` 排名），其余 `cu_num_logits`/`query_start_loc` cumsum 与
  上游**逐行相同**；并把 `batches_to_profile`/`set_initial_cost_curves`/`record_confidences` 置空
  （人工预算不需要 cost model，也不碰 confidence head）。
- **两入口的顺序陷阱**：`get_num_tokens` 拿到的是 scheduler dict 序，`reallocate_drafts` 拿到的是
  `sort_batch_req_ids` 之后的 batch 序。位置索引的 pattern **不是顺序无关的**，所以 capacity 必须在
  `get_num_tokens` 里算完、按 req_id 存进 `_batch_budget`，由 `reallocate_drafts` **查表**而非重算——
  否则两边会把 cap 配到不同请求上、`Σcapacities ≠ draft_budget`，而 batch 尺寸已经按后者定了。
  全局标量时代靠"min 顺序无关"侥幸成立的不变式，在 pattern 下变成结构保证。
- **接线**（`patch/worker/patch_v2/patch_adaptive_verification.py:41`）：GDN 被上游工厂
  `maybe_create_adaptive_verification_manager` 拒绝（varlen backend 检查 + 要求 `ALWAYS`）→ 原状
  `adaptive_verification=None`，裁剪链路根本不跑。patch 工厂：`ENABLE_DCUT=1 且 MANUAL_CAP>=0` 时
  返回人工 manager，否则原样交回上游。这样 `initialize_kv_cache`（vLLM `model_runner.py:540`）
  拿到非 None manager，base 的 `cudagraph_mode=FULL_AND_PIECEWISE`（`:573`）等 setup 照跑。
- **env**：`VLLM_ASCEND_DSPARK_DCUT_MANUAL_CAP`（单个 int 或逗号分隔 pattern，默认 `-1` 关）；仅
  `VLLM_ASCEND_DSPARK_ENABLE_DCUT=1` 且 pattern 里至少有一个非负项时生效。`cap=0` 全裁到只剩
  anchor；`cap` 很大 = ① 的不裁；`7,0,3,1` 出参差批次。spec 写错直接抛 `ValueError`（启动即失败，
  不静默退回不裁）。
- **验收分层**：策略函数的 cap→capacity/budget 与 confidence-无关性由 CPU UT
  （`tests/ut/spec_decode/test_dcut_manual_cap.py`）钉死；device 侧 `reallocate_drafts` 复用上游布局，
  由步骤 ④ 的 NPU 连续多轮模型验证覆盖。工厂绕过 `ALWAYS` 检查等于在赌"GDN 整个 forward 在裁剪
  下能跑变长 decode"——§1.6/§1.7 就是这个赌注的账单。
- **顺序性脆弱点**：`init_attn_backend`（vLLM `model_runner.py:533`）在 AV 覆盖 `cudagraph_mode`
  （`:573`）**之前**执行，所以 builder 的 `use_full_cuda_graph` 取启动时的模式。若启动只给
  PIECEWISE 而由 AV 升到 FULL_AND_PIECEWISE，持久缓冲区块与 padding 处理会全部跳过。当前启动
  指定 `FULL_DECODE_ONLY`，标志为真；仍应显式断言或在覆盖后重建 builder。

### UT 覆盖现状（诚实记录）
- 两个算子 UT 已在 910B pass。`_dcut_recurrent_golden` 已用
  **二维 `ssm_state_indices[req, accepted-1]` + 累计 `query_start_loc` + `num_accepted`**
  ——**印证 §1.1/1.2 契约**；跨轮用例每轮从 NPU 携带态重算 golden（避免 bf16 复合漂移，已在
  docstring 写明）。
- **待补**：① 随机物理状态行（真实索引来自 block table，不是 `num_requests * state_len` 连续行）；
  ② 跨 cache block 的 pre-copy/postprocess；③ **零长度请求行及其状态不变性**——现有 UT 测的是
  尾部**输出** padding（`zero_padded_output`），与"零长请求行"不是一回事；④ conv 的独立顺序参考
  （当前 `test_dcut_causal_conv1d_matches_update_mode` 仍与旧算子 update 路径对比）。
- **UT pass ≠ 模型接入回归过。**

**不需要再扩影子验证。** 要补的是"本轮长度 / 上一轮状态选择 / 模型整体输入布局 / 图捕获分支"
四者的契约（§1）。

## 6. 依赖
- dcut 算子已 vendor（cherry-pick `645e05ac71960cc6bf01faca8aba7037dd752002`）、910B 编译通过 +
  自带 UT；需跟踪上游 delta。
- 步骤 ④（连续多轮模型验证）依赖 ⑤⑥ 的图契约先落地。
- 阶段 C 的真实 manager 依赖 ④，且需要在修好的执行路径上**重建 cost table**（人工 manager 把
  profiling 置空，早期 smoke 也未建 varlen 图，而上游 `get_num_tokens` 第一行即
  `assert self.cost_tables is not None`）。

## 7. 变更历史

| 日期 | 内容 | 验证状态 |
|---|---|---|
| 2026-09-16 | 固化六轴、状态与观测契约；固定 `B_gdn` 降为待验证选择；区分 DFlash 经验与 DSpark 审计；同步分支修复状态 | 文档与 `2bb8a9c5c`、vLLM `v0.28.0` 接口复核；本次未实施 F1′/F3，也未新增 NPU 结果 |
