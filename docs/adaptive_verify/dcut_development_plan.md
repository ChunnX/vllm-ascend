# D-Cut 自适应验证开发计划（Qwen3.6-27B / GDN / 910B）

> 目标：在 vLLM-Ascend（MRV2 / worker v2）上，为 **Qwen3.6-27B（GDN 混合模型）** 落地
> DSpark 置信度调度的**动态验证（D-Cut）**——按每请求置信度 + Ascend 代价表，动态裁掉
> 后位低价值 draft，而不是恒定验证全部 draft。主场景：**16/32 中高并发 + 长上下文**。
> 主芯片：**910B**（310P 不在本计划范围）。
>
> 行号基准：`76780b1a1`。跨仓引用的 vLLM 行号基准：v0.28.0。

## 0. 背景与结论（为什么做、怎么做）

- 上游已有实证：DSpark 置信度调度验证在 **attention 目标**（Qwen3-8B +12.1%、
  DeepSeek-V4-Flash +22.1%，draft=9）上有收益。
- 已确认可用的既有成果：① DSpark 置信头在 **FULL_DECODE_ONLY** 下**不再冻结**、能产出逐步变化的
  置信度（图捕获冻结已修，逐元素 eager/replay 对照待补）；② AV cost-table profiling 链路**能执行**
  （`[AV-PROFILE-SMOKE] OK`）。**注意**：`monotonic=True` 是建表时 `np.maximum.accumulate`
  造出来的、不是测量稳定性证明；`set_initial_cost_curves` 也已按 shape 取 median。
- 关键经济学（上游 RFC 亲述）："**removing a few logical tokens is useful only when it
  reaches a cheaper execution shape**"。**推论（待验证，非结论）**：低并发 + 短上下文时若
  verify 代价近似平坦，裁剪到不了更便宜的图形状 → 收益有限；中高并发 / 长上下文可能进入陡增
  区才有价值。**但此推论依赖代价表能代表真实 AV varlen verify 路径**——当前 smoke 用的是临时
  manager、未建真正 varlen 图，且 dummy batch 可能跨 eager / uniform-FULL 路径（`tok256≈89ms`
  vs 邻居 ~220ms 就是执行模式差异的信号），所以**在代价表可信前，不能据平坦性止损**。
- **这条经济学与图路线直接耦合**：裁剪要变便宜，必须落到更小的执行形状。因此"走哪条图路线"
  （§3.1）不是实现细节，它部分地预先决定了阶段 D 的收益上限。

**策略：跟随上游 RFC，能采纳就采纳，不重复造轮子。** 只有 GDN×AV 的胶水是必须自研的部分。

### 0.1 工作副本纪律（2026-09-16 新增）

曾出现**两个同名分支的工作副本内容分叉**（merge-base 之后 69 / 45 个 commit，最近几条 commit
subject 完全相同），其中只有一份含上游 GQA/MRV2 那次合入、也只有那份含 GDN spec 图 padding
处理。这导致"代码 review 的树"和"NPU 上跑的树"不是同一棵，故障归因反复。

- **主线只保留一份工作副本**；其它副本只读或删除。
- 每次上机运行必须记录：**运行端 commit SHA + CANN / torch_npu 版本 + 启动命令**，不得用本地
  checkout 的 SHA 代替。
- 文档中引用行号时**标注基准 SHA**（本文件见标题下方）。

## 1. 上游参考（权威来源）

| 编号 | 标题 | 状态 | 作用 | 缺口 |
|---|---|---|---|---|
| **RFC**（adaptive speculative / Dcut） | 总设计 | OPEN | 点名 Qwen3.5/3.6 的 GDN 变长 state 正确性是核心难点；给出"更便宜的执行形状"这条经济学 | — |
| **MRV2 DSpark confidence-scheduled verification** | AV"大脑" | **MERGED**（`990243c4c5b0`，2026-09-14 14:15 UTC）| `model_runner` 接 `compact_batch`/`reallocate_drafts`/`get_num_tokens`/`record_confidences`；attention 后端（dsa/sfa/mla）变长 | **无 GDN 分支**（只对 `use_fia` 后端 pad）；每步仍有 2 次条件 D2H 回读 |
| **GQA DSpark + MRV2 spec decode on Ascend** | GDN 图 padding 契约 | **MERGED**（`fa50f6ae8`，2026-09-15 06:56 UTC）| 引入 `_remove_spec_graph_padding_queries`：把 FIA 的 padding 图行压成零长，供 GDN recurrent 状态更新 | 合入时与本地 shared-metadata 重构撞车，**处理函数被架空**（见 §3.1 第 1 项） |
| **variable-length GDN operators** | GDN 变长算子 | 上游 OPEN；**本地已 vendor** | `npu_dcut_causal_conv1d` / `npu_dcut_recurrent_gated_delta_rule`（吃 query_start_loc + cache_indices + num_accepted_tokens），注册 910B/910_93/950 | 已 cherry-pick 自 `645e05ac71960cc6bf01faca8aba7037dd752002`，910B 编译 + UT 已过；**需跟踪上游 delta** |
| 上游 vllm（AV manager 本体） | `adaptive_verification.py` | merged (v0.28.0) | 预算选择、compaction、cost table、`set_cost_curves` 已跨 TP 广播 | — |
| 参考 | 早期 Dcut（CLOSED）、mamba cache group、GDN prefill aclnn | — | 相关上下文 | — |

> 旧的 `pr15098_..._feasibility.md`：**§0–§6 已过时**（评的是合入前的 head，结论"不要
> cherry-pick"已被合入版推翻：合入版带了 UT/e2e、删了调试断言、放宽了 padding assert）。
> 仅 **§7 的 record-only 观测记录**仍有效。不要再用它做决策依据。

### 1.1 可复用 vs 必须自研

| 可以复用 | 我们必须完成 |
|---|---|
| 上游的预算选择、confidence 流水线、MRV2 接线 | Qwen3.6 hybrid metadata 与裁剪后请求布局一致 |
| D-Cut 的变长 conv/GDN 算子（已 vendor） | 算子接入、state 索引、跨轮接受状态与回退 |
| 上游图管理框架 | **GDN 的 varlen 捕获/replay 分支一致性**（当前阻塞项） |
| 现有定长 DSpark 优化 | FIA Sink、GDN metadata 复用、KV grouping 与 AV 的兼容验证 |

**注意**：结合是正确路线，但工作量**不是"合入两个 PR 就完成"**。

## 2. AV 决策与执行（上游 `AdaptiveVerificationManager` 机理）

- 决策信号 = **survival = 各位置 confidence 的累积乘积**（`get_num_tokens`）。
- `draft_budget = argmax(estimated_accepted / cost)`，只有 verify 代价随 budget 变；
  代价表由启动时 profiling（graph-bucket 感知）建成，**代价平坦时 argmax 选满 budget → 不裁**。
- 执行 = `compact_batch` + `reallocate_drafts` 在 `prepare_inputs` 里裁掉低价值 draft 后缀，
  保留每请求的 anchor，且保持 prefix 有序。注意 `compact_batch` 只在 CPU 侧**按总预算均分**，
  逐请求容量由 `reallocate_drafts` 产出；CPU 侧 `query_start_loc_np` 随后由
  `model_runner.py:448` 从 device 回读修正。
- **calibration（口径已勘误）**：confidence[i] 是**条件**每位接受概率，上游 cumprod 后才是
  survival；而 observer 的 acc[i] 是**前缀 survival** P(accepted≥i+1)。正确标定必须比
  **逐样本 `cumprod(confidence)` vs acc**（且逐样本累乘再平均，不能对均值累乘）。之前"尾部
  过自信→少裁"是拿条件比边际的错误口径，**已撤回**。上游用 raw confidence、无按数据集重标定
  ——标定是 checkpoint（训练）的责任，**不做下游 affine 拟合**。

## 3. 开发阶段（2026-09-16 重排：图契约一致性提为当前阻塞项）

**决策**：shadow / observe / smoke 全部为调试工具（默认关、非闸门）；**收益问题由阶段 D 的
三臂对照回答，不预设**。观测围绕**真实执行链**：选了什么预算、实际跑了什么形状、耗时多少。

> **当前阻塞**：FULL 图下人工 manager + cap=99（不裁）仍然增量乱码、7 个 draft 位置接受率全 0，
> 在 `debda6c68`（旧副本）和 `76780b1a1`（含上游 padding 处理）上现象相同。eager 下 cap=99/2
> 精度正确。所以问题定位在**图捕获/回放的分支与缓冲区契约**，不在 confidence、不在算子。

### 阶段 B0 —— 合并缺陷 + GDN 图契约统一（当前阻塞，先做）

原先这块被摊在阶段 C 的"图捕获"一句里，实际是独立工作量，提出来单列。

**1. 修合并缺陷：`build()` 被重复定义，前一个整体失效。**
`AscendGDNAttentionMetadataBuilder` 里 `build` 定义在 `ops/gdn_attn_builder.py:591` 和 `:979`，
Python 生效的是 `:979`。`:591` 那个来自上游合入（`fa50f6ae8`），`:979` 那个来自本地
shared-metadata 性能重构（`cff680f36`，2026-08-18）。后果是**两个前置处理全部不可达**：

- `_remove_spec_graph_padding_queries`（`:57`，唯一调用点在 `:600`）——把 FIA 的 inactive 图行
  压成零长，正是 GDN recurrent 状态更新要求的 padding 契约；
- `_treat_single_token_prefills_with_state_as_decodes`（`:89`，唯一调用点在 `:604`）——让单 token
  有状态 prompt chunk 复用 decode metadata。

**不能只删前一个定义**：两个函数都会重建 `query_start_loc`，而 shared plan 的缓存键按
**tensor 地址**取（`:859` `_shared_batch_plan_key`），若在生效的 `build()` 里逐 KV cache 组各做
一次前置处理，每组会拿到不同的 `query_start_loc` → 缓存全 miss、各组按组内视图刷图缓冲区。
必须**每次 `build_attn_metadata` 调用只做一次**，并挂在同一个 batch 级缓存上。
另外要验：`_remove_spec_graph_padding_queries` 会把 `num_actual_tokens` 改成真实（未 padding）
token 数，需确认 FULL 捕获期（bucket 大小）与 replay 期（真实值）之间没有被捕获进图的形状假设
被破坏（`ops/gdn.py:305` `mixed_qkv[:num_actual_tokens]`、SPEC_ONLY 分支不用 `spec_token_indx`）。

**2. 捕获/回放分支必须同一条。** 这是 cap=99 仍乱码的直接原因：

- AV manager 一旦非空，base 就把 `cudagraph_mode` 覆盖成 `FULL_AND_PIECEWISE` 并给 graph
  manager 传 `varlen_decode=True`（vLLM `model_runner.py:574-587`）。
- varlen descriptor 的请求数是 `min(num_tokens, max_num_reqs)`（vLLM `cudagraph_utils.py:227`），
  `make_dummy` 把 token 均分 → **`num_tokens <= max_num_seqs` 的每个 bucket 都是 1 token/请求**。
- 捕获用 `num_decode_draft_tokens_cpu = diff(query_start_loc) - 1`（vLLM `gdn_attn.py:546`）→ 全 0。
- GDN builder 见"所有 runtime draft 数之和为 0"就把 spec mask 清零、转普通 decode
  （`ops/gdn_attn_builder.py:632-643`）→ **捕获进图的是普通 decode 的 conv/recurrent**。
- replay 时真实 batch 有 draft → Python 构造 spec metadata，但 FULL replay 不重跑 Python 分支，
  执行的仍是捕获的普通 decode 算子及其捕获期缓冲区地址 → 乱码、接受率全 0。

修法：**让 decode 图只有一条算子路径**。draft 数为 0 的行表达成"spec 分支里的 1-query 行"
（语义上与普通 decode 单行更新等价：`num_accepted=1` 时读写的都是 `ssm_state_indices[b, 0]`
= `block_table[b, 0]`），而不是切换到普通 decode 分支。注意这个改动只能影响捕获路径：真实
batch 里 0 draft 的行在 `model_states/mamba_hybrid.py:77` 已被写成 `-1`（非 spec），因此
"draft 数恰为 0"这种行**只由 dummy 捕获产生**。

**3. 正长度 padding 行（cap<full 时才触发，尚未上机复现）。**
`num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)`
（`worker/v2/model_runner.py:351`），而 `num_tokens` 是裁剪后的值，所以**裁剪后 token 数一旦
不正好落在 capture size 上，`_pad_adaptive_query_start_loc_for_fia`（`:1112`）就会把正长度
均摊给 padding 请求**。复现布局（11 请求 / cap=2 / 33 token / 40-token 图 / 32 请求槽 → 21 个
padding 行里 7 个拿到长度 1）：这些行 draft 分类是 `-1` → 非 spec → 与真实 spec 行混合后计入
`num_prefills` → **跳过 `:1041` 的 pure-spec 持久缓冲区块**（该块要求
`num_prefills == 0 and num_decodes == 0`）→ spec metadata 退回每步新建的临时张量 → FULL replay
读捕获期地址。

- 40-token 图**捕获时已经走 spec**（24 行长度 1 + 8 行长度 2，存在正 draft 计数），所以
  "修掉 ≤32-token bucket 的普通 decode 捕获"**不足以**覆盖这条路径。
- 第 1 项修好后，`_remove_spec_graph_padding_queries` 会把这些行压回零长（其前置条件在 AV
  padding 下成立：`seq_lens_cpu_upper_bound` 对 `[num_reqs, num_reqs_padded)` 为 0、draft 为 -1、
  且是连续后缀）→ 恢复 SPEC_ONLY。**两项要一起验，不要分开宣称修好。**
- 关于"污染"：padding 行写的 `NULL_BLOCK_ID = 0` 是 vLLM 保留的 null block，**这一个动作本身
  是设计意图**；但这不等于整条 SPEC_MIXED + 缓冲区失效的路径安全，仍需按有效状态索引和有效
  输出验收，不提前给安全结论。

**4. 顺序性检查项**：`init_attn_backend` 在 AV 覆盖 `cudagraph_mode` **之前**执行
（vLLM `model_runner.py:533` vs `:573`），所以 builder 的 `use_full_cuda_graph`
（= `has_full_cudagraphs()`）取的是**启动时**的模式。若启动只给 PIECEWISE 而由 AV 升到
FULL_AND_PIECEWISE，builder 会以为没有 FULL 图 → 持久缓冲区块与 padding 处理全部跳过。
当前启动指定 `FULL_DECODE_ONLY`，该标志为真；但这是一处真实的脆弱点，要么显式断言，要么
在 AV 覆盖后重建 builder。

**通过标准**：cap=99 在 FULL 图下与 eager 输出一致（先复现旧故障已消失），再 cap=2；覆盖
8/16/32/40-token 图、bucket 内部落点（token 数不等于 capture size）、1/2/4 个真实请求、不同
padding 数量。

### 阶段 A —— 基线与合入版审计（**并行轨，不阻塞 B0**）

- 审计合入版：不只是 AV 那个 PR，**还包括引入 GDN 图 padding 契约的那次合入**（`fa50f6ae8`）
  ——B0 第 1 项就是它的合入事故。
- 钉运行端版本（§0.1）；确认 910B dcut csrc 编译路径。
- 建**保留现有优化的定长基线**并逐项验**无回归**：64K head、FIA Sink、GDN metadata 复用、
  KV grouping、sampling。
- **定位与归因的边界**：A 的交付物确实缺，但"不装人工 manager 正常 / FULL 失败 / eager cap=99
  与 cap=2 正常"这组对照 + B0 的具体代码缺陷，已足以先修图。**不以补完整套 A 作为继续修图的
  前置条件**；但 A 必须在阶段 D 之前闭合，否则性能对照没有干净基线。

### 阶段 B —— 人工 cap 打通 GDN 变长状态正确性（核心难点）

- 算子：910B 编译 + UT 已过（含跨轮：缩短/恢复、上轮 accepted > 本轮 query、请求重排、cap=0）。
  **待补**：随机物理状态行（真实来自 block table，不是连续行）、跨 cache block、**零长度请求行
  及其状态不变性**（现有 UT 测的是尾部**输出** padding，两者不同）、conv 的独立顺序参考
  （当前 conv 测试仍与旧算子 update 路径对比）。
- **人工 cap（与 confidence 解耦）**：cap=99/2 的 eager 冒烟已过；待验 ① 异构 cap、cap=0；
  ② 跨轮缩短（上轮接受 > 本轮验证长度）；③ 请求重排 + slot 复用；④ 跨 cache block
  pre-copy/postprocess；⑤ conv 与 recurrent state 同步提交；⑥ padding 不污染有效状态。
- **接入点（落点已更正）**：真正要改的不是 `model_runner` 的 compaction（它模型无关），而是
  **`model_states/mamba_hybrid.py:74-85` 喂给 GDN builder 的那几个字段**：目前
  `num_decode_draft_tokens_np` / `is_decode` 全部由**裁剪前**的
  `num_draft_tokens_per_req` 与 `num_scheduled_tokens` 算出，而 `query_start_loc_np` 是裁剪后的。
  - **不能简单全换成裁后值**：上一轮 accepted 大、本轮 query 只有 1 的请求必须留在 spec 路径
    才能读到候选状态（设计文档 §1.3）。要把四个量拆开表达：**原调度类型 / 本轮实际 query 长度 /
    上一轮 accepted / padding 身份**。
  - 由此更正设计文档 §1.5 的一处说法：cap=0 时裁剪前 draft 数仍 > 0，mask 仍为真，**全批 cap=0
    也不会走普通 decode**。原先"全 batch cap=0 → 普通 decode → 恢复 spec 的衔接"这条用例测的是
    裁剪永远不会走的路径，改为"全批 cap=0 仍在 spec 分支、每请求 1 query"。
  - padding 行的 spec 判定 `padded_query_lens == num_spec + 1`（`:84`）在 AV padding 下永不成立。
- **产出**：连续多轮人工变长验证，输出 + 逐轮 GDN/KV 状态对照参考实现通过。

### 阶段 C —— 路线决策 + 真实 AV manager + TP（B0/B 之后）

**C0. 图路线决策（新增，必须显式做，不能默认走第一条）。** 三选一，用同一组测量选：

1. **varlen FULL**：真正实现 GDN 的变长图契约（当前 builder 声明的是
   `AttentionCGSupport.UNIFORM_BATCH`，`ops/gdn_attn_builder.py:303`，而 AV 的
   `varlen_decode=True` 让 graph manager 无条件发 varlen descriptor——图模式解析根本不看它）。
   **注意 uniform decode FULL 图在结构上无法表达裁剪**（uniform 把 num_tokens 向上取整到
   `decode_query_len` 的倍数再推请求数，参差布局对不上），所以"保留 UNIFORM_BATCH 声明 +
   FULL 图"不是选项；能力声明只能在图能力真的验证通过之后再提升。
2. **PIECEWISE-only AV**：不让 AV 强制 `FULL_AND_PIECEWISE`，只跑 PIECEWISE。绕开全部 varlen
   FULL 工作量，代价是 §0 的经济学更不利（没有 bucket 悬崖可跳，只剩 kernel 少算 token 那部分
   收益）。工时差一个量级，值得作为正式选项评估。
3. **退回静态位置 cap**（阶段 D 的第二臂），不依赖 confidence。

**C1. 真实 cost table 必须在修好的执行路径上重建。** 人工 manager 把
`batches_to_profile` / `set_initial_cost_curves` 置空（`spec_decode/dspark/dcut_manual_cap.py:92-100`），
早期 smoke 用的是临时 manager 且没建 varlen 图，而上游 `get_num_tokens` 第一行就是
`assert self.cost_tables is not None`。调用链上游已有（vLLM `model_runner.py:893`），不需要重建
架构，**但必须在 B0 修好之后用真实 manager 重跑**：看 `raw_median_monotonic`、每 shape 的
`cg=T/F/mix` 与 `num_reqs` 分解。raw median 非单调时**先查执行模式与测量稳定性**，不要立刻切
阈值算法。

**C2. 把人工 cap 换成真实 `get_num_tokens` 决策**（confidence + cost table）。

**C3. TP 一致性（TP4）。** 已有的 TP4 跑只能记为**功能冒烟通过**，不能替代逐 rank 一致性验证。
cost curves 上游已跨 TP 广播（vLLM `adaptive_verification.py:221`），所以要逐项比：
**总预算 / 逐请求 capacities / 请求映射 / query 边界**；confidence 路径还要看各 rank 的 stale
confidence 是否逐元素一致（draft 侧有 allreduce 时 `argmax` 可能翻转）。人工 cap 路径的确定性
是结构性的（纯函数 of scheduled counts），风险更低，但**低不等于已验**。

### 阶段 D —— 三臂对照（收益判定）

- 三组配置（现有优化全一致）：**① 关 AV；② 静态 cap=k（人工 manager，不依赖 confidence）；
  ③ 真实 AV**。对照真实送验长度、每轮接受长度、target/draft 耗时、TPOT、吞吐、尾延；
  16/32 并发 × 长上下文。第二臂回答"confidence 相对静态位置裁剪到底带来多少增量"。
  人工 manager 目前**只是可复用基础**，图正确性/状态测试/性能都未通过，还不能称为可交付保底。
- **正确性**（不是"接受长度不变"——裁剪本就会降接受长度）：① temp=0 greedy 输出逐 token 与参考
  一致；② 多轮 KV/SSM 态**按同一已提交前缀**对齐比较（**不按迭代轮次硬对齐**，与设计文档
  §4/§5 统一；浮点算子明确容差，不默认 bitwise）；③ 随机采样验分布/采样语义。
  **允许接受长度下降。**
- **同步开销必须单列**：合入版每步仍有 2 次条件 D2H（`worker/v2/model_runner.py:448` 回读
  device 算好的 `query_start_loc`、`:494` 回读 `seq_lens`；两者都在 `use_fia` 分支内，不是每步
  固定发生）。不量化它，就分不清"AV 不值"和"同步吃掉了"。
- 收益不足则用真实执行观测（预算 / 形状 / 耗时）定位：代价表 / confidence / 图分档 / 同步开销。

## 4. 贯穿项
- 环境变量：`VLLM_ASCEND_DSPARK_AV_OBSERVE`、`VLLM_ASCEND_DSPARK_AV_PROFILE_SMOKE`、
  `VLLM_ASCEND_DSPARK_AV_SHADOW`、`VLLM_ASCEND_DSPARK_ENABLE_DCUT`、
  `VLLM_ASCEND_DSPARK_DCUT_MANUAL_CAP`——**默认关的调试/集成开关，非阶段闸门**。
- **observe / shadow 收口**：observer 只保留**一次有界的标定检查**（按正确口径：逐样本
  `cumprod(conf)` vs 前缀 acc、slot 对齐、NaN/unmatched 计数），**不为它再建完整校准系统**；
  **shadow 不再投入**，也不重新成为进入真实 AV 的闸门。confidence 有无增量价值，最终由阶段 D
  的第二臂（静态 cap）与第三臂（真实 AV）对照回答。
- **真实执行观测**（阶段 C/D 加）：记录"选了什么预算、实际执行了什么图形状、耗时多少"。
- 每步优先 cherry-pick 上游 > 自研；GDN×AV 胶水是必须原创。
- commit 规则：仅 `Signed-off-by: ChunnX`，**且不带任何 PR/issue 引用**。

## 5. 风险与工时
- **风险排序（2026-09-16 重排）**：
  1. **GDN 图捕获/回放契约**（当前阻塞）：捕获分支按 bucket 翻转、正长度 padding 破坏 SPEC_ONLY、
     合并导致 padding 处理失效、builder `use_full_cuda_graph` 早于 AV 覆盖模式。
  2. **完整模型状态正确性仍未验收**——算子 UT 与少量 eager 冒烟通过**不能**划掉这一项。
  3. GDN 变长 state 跨轮正确性（"上轮接受 > 本轮验证长度"）：算子级与 eager 已有通过反馈，
     模型级逐轮状态对照未做。
  4. TP 逐 rank 决策/布局一致性（阶段 C3）。
  5. vendor 的算子跟踪上游 delta。
- **工时**：旧的"合计 ~5–6 周"未包含 B0，已失效。B0 的规模取决于 §3.1 C0 的路线选择
  （varlen FULL 与 PIECEWISE-only 差一个量级），**现有信息不足以给出可靠的新总工时**；
  待 B0 第 1、2 项修完并拿到 FULL cap=99/cap=2 的结果后再估。
- **收益不确定性**：上游收益在 attention/高 TPS 上验证，GDN hybrid 未验——**由阶段 D 三臂
  对照判定**，不预设。
