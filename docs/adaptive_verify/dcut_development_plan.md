# D-Cut 自适应验证开发计划（Qwen3.6-27B / GDN / 910B）

> 目标：在 vLLM-Ascend（MRV2 / worker v2）上，为 **Qwen3.6-27B（GDN 混合模型）** 落地
> DSpark 置信度调度的**动态验证（D-Cut）**——按每请求置信度 + Ascend 代价表，动态裁掉
> 后位低价值 draft，而不是恒定验证全部 draft。主场景：**16/32 中高并发 + 长上下文**。
> 主芯片：**910B**（310P 不在本计划范围）。
>
> 历史行号基准：`76780b1a1`；本次状态复核：`2bb8a9c5c`。跨仓引用基准：vLLM `v0.28.0`。
> 开发采用 vLLM `v0.28.0` tag + vllm-ascend main 上保留既有 DSpark 优化的分支，运行固定具体 SHA。

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
- **经济学与实际执行路径耦合**：跨到更便宜的图 bucket 是主要收益来源之一；同 bucket 内变长
  kernel 是否少算、减少多少开销也需实测。FULL_DECODE_ONLY 是开发目标，不能预设一定比 PIECEWISE 快。

**策略：跟随上游 RFC，能采纳就采纳，不重复造轮子。** 只有 GDN×AV 的胶水是必须自研的部分。

### 0.1 工作副本纪律（2026-09-16 新增）

曾出现**两个同名分支的工作副本内容分叉**（merge-base 之后 69 / 45 个 commit，最近几条 commit
subject 完全相同），其中只有一份含上游 GQA/MRV2 那次合入、也只有那份含 GDN spec 图 padding
处理。这导致"代码 review 的树"和"NPU 上跑的树"不是同一棵，故障归因反复。

- **主线只保留一份工作副本**；其它副本只读或删除。
- 每次上机运行记录：**运行端 vLLM/Ascend 两个 SHA、实际 import 路径、未提交 diff、CANN /
  torch_npu 版本、算子包构建版本与加载路径、完整启动命令**，不得用本地 checkout 的 SHA 代替。
- 2026-09-16 本次复核：`vllm-workspace/vllm-ascend` 已到 `2bb8a9c5c`，包含 `ba4853209` 和
  `2bb8a9c5c`；不能继续沿用“该副本停在 `76780b1a1`”的描述。此记录不代替 NPU 部署版本。
- 文档中引用行号时**标注基准 SHA**（本文件见标题下方）。

## 1. 上游参考与证据边界

| 编号 | 标题 | 状态 | 作用 | 缺口 |
|---|---|---|---|---|
| **RFC**（adaptive speculative / Dcut） | 总设计 | OPEN | 点名 Qwen3.5/3.6 的 GDN 变长 state 正确性是核心难点；给出"更便宜的执行形状"这条经济学 | — |
| **MRV2 DSpark confidence-scheduled verification** | AV"大脑" | **MERGED**（`990243c4c5b0`，2026-09-14 14:15 UTC）| `model_runner` 接 `compact_batch`/`reallocate_drafts`/`get_num_tokens`/`record_confidences`；attention 后端（dsa/sfa/mla）变长 | **无 GDN 分支**（只对 `use_fia` 后端 pad）；每步仍有 2 次条件 D2H 回读 |
| **GQA DSpark + MRV2 spec decode on Ascend** | GDN 图 padding 契约 | **MERGED**（`fa50f6ae8`，2026-09-15 06:56 UTC）| 引入 `_remove_spec_graph_padding_queries`：把 FIA 的 padding 图行压成零长，供 GDN recurrent 状态更新 | 历史上与 shared-metadata 重构冲突；`2bb8a9c5c` 已恢复调用，FULL 回归未验收 |
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

### 1.2 作者文章与本项目契约

用户于 2026-09-16 提供的文章（§3.4.5、§4.1–4.5、§7）对应
[PR #15207](https://github.com/vllm-project/vllm-ascend/pull/15207)，但已审阅的算子提交不包含
文章全部 FULL 接线。固定 `B_gdn = B_max` 记为**待验证的实现选择（来源：上游作者经验，
我们尚无同根因证据）**，不能据文章直接宣称它是当前乱码的原因。

六个轴、状态不变量、F3 字段和 DFlash/DSpark 审计边界统一以
[接入设计 §1](dcut_gdn_integration_design.md) 为准。本计划只安排实现与验证顺序。

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
> 精度正确。优先排查**图捕获/回放的分支与缓冲区契约**；人工 cap 不依赖 confidence 决策，
> 但 eager 冒烟不能排除算子入图问题。`2bb8a9c5c` 已修复下述两处分支/调用缺陷，
> 尚无本计划验收过的 FULL 修复结果，不能把代码提交等同于模型精度通过。

### 阶段 B0 —— 合并缺陷 + GDN 图契约统一（当前阻塞，先做）

**已有修复**：`2bb8a9c5c` 移除重复 `build()`，将 GDN 局部 metadata 归一化挂到 batch 级共享
缓存，并在 D-Cut 开启时保留零 draft 捕获行的 spec 分支。这些修改保留；算子路径一致只是
必要契约，提交中的结构/metadata 测试不证明 FULL 正确。历史机制见接入设计 §1.6/§1.7。

**仍可确认的缺口**：CPU `seq_lens_np[B_live:B_graph]` 未完整刷新；device 未使用行则由 vLLM
`v0.28.0` 的额外 Triton program 清零，并非两侧都保留旧值。GDN 当前仍依赖 KV 长度为零识别
padding，因此不能直接把共享 padding 长度填成 1。8-token、8 请求的捕获 dummy 行长度是 1，
不能用回放时的 `B_live=1` 去解释捕获期的清零范围。

**执行顺序保持为 F0 → F3 → F4 最小测试 → F1′ + 必要 F2 → 模型回归 → F5/F6 → F7。**
F6 的支持边界先明确，最终声明在验证后落地；固定轴不作为已经证实的根因。

| 编号 | 工作 | 验收与范围 |
|---|---|---|
| F0 | 统一工作副本与运行身份 | 执行 §0.1；保留原失败命令与实际部署信息 |
| F3 | 补完独立观测，先采未修基线 | capture 准备 / replay metadata 刷新 / 图 dispatch 三处；每条含六轴、`Q_bucket`、相关 tensor 的 shape+ptr，细节见设计 §1.9 |
| F4 | 最小 conv + recurrent 图测试台 | 单图不同 qsl、重排、accepted 大于本轮 query、padding 状态不变、跨轮；恢复相同初始状态后比较；增加多图捕获和往返切图 |
| F1′ | 对照实验：固定 `B_gdn = B_max` | 先核对分配容量，再调整 qsl/状态索引/accepted/mask 与派生视图；限 D-Cut pure-spec FULL；不顺手改普通 decode，不全局替换 sentinel |
| F2 | 分离 FIA/GDN padding 语义 | 显式请求数/mask 标识身份；FIA 局部生成合法 dummy KV 长度与 block row；GDN 零长+无效状态索引；修正 CPU/device 元数据缺口 |
| F5 | 审计 descriptor 资源隔离 | 记录实际冲突组合；确认 uniform target / ragged target / drafter 不混用参数。现有分表足够也算完成，不预设新增字典键 |
| F6 | MRV2 图选择与能力边界 | 纯 spec decode 才进本阶段 ragged FULL；prefill/mixed/handoff 未满足契约则走已验证路径；不全局翻 `ALWAYS` |
| F7 | 基于实际图重建 cost table | 见阶段 C1；图正确性通过后执行，不拿临时 manager 的 smoke 表作收益依据 |

F3 必须让 `B_live / Q / B_graph / B_fia / B_gdn` 的捕获/回放值可关联，记录 `B_max`，
不能从 config 推算后当作实测。具体形状、地址、刷新范围及有界数值样本用于回答：
**GDN 请求轴是否随 bucket 变化，变为多少；地址稳定时内容是否已经刷新。**

F4 必含：只捕获一张图 vs 捕获全部图后反复回放同一张 8-token 图；小图→大图→小图；
batch 收缩→增长与 slot 复用；逐 bucket 轴 vs 固定 `B_max`。这样才能区分同图变长、
后续 capture 污染、切图刷新和请求身份错误，而不是一次修改多个量后直接归因。

**顺序性检查项**：`init_attn_backend` 早于 AV 覆盖 `cudagraph_mode`，builder 的
`use_full_cuda_graph` 可能来自覆盖前的模式。PIECEWISE 诊断需确认实际执行模式及 GDN 是否入图，
不能只看启动参数；FULL 配置也要核对 builder 标志和实际 descriptor。

**模型通过标准**：先 FULL cap=99，再 cap=2、cap=0；覆盖 8/16/32/40-token 图、bucket 内部
落点、1/2/4 个真实请求、不同 padding 数量、跨 bucket 和 TP 布局一致性。比较有效输出及
同一已提交前缀的状态；cap=99 冒烟通过不等于整个图契约通过。

### 阶段 A —— 基线与合入版审计（**并行轨，不阻塞 B0**）

- 审计合入版：不只是 AV 那个 PR，**还包括引入 GDN 图 padding 契约的那次合入**（`fa50f6ae8`）
  ——重复 `build()` 就是相关合入缺陷，现已提交修复。
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

### 阶段 C —— FULL 支持边界 + 真实 AV manager + TP（B0/B 之后）

**C0. FULL_DECODE_ONLY 是目标；PIECEWISE 为诊断与性能对照。** GDN 当前声明
`AttentionCGSupport.UNIFORM_BATCH`，而 MRV2 的 `varlen_decode` 会产生变长 descriptor，
需在 F6 明确支持检查。`ALWAYS` 还承诺 mixed prefill/decode 图能力，不能因纯 spec 变长
通过就全局提升声明。FIA/GDN 局部视图、图资源隔离与回退路由必须满足接入设计 §1 契约。

**C1 / F7. 在已验证执行路径上重建真实 cost table。** 人工 manager 将 profiling 置空，早期
smoke 没有覆盖最终 varlen 图；复用上游 profiling/建表接口，不另造控制器。

- 可选择的图形状来自 ACLGraph **实际注册的 descriptor/capture sizes**，不从额外 JSON
  维护第二套 Q/BS 规则。配置只保留 warmup/测量控制；仍记录服务配置、上下文等测量条件。
- `B_max` 来自 `max_num_seqs`，是上限；样本必须记录实际 `B_live`、`Q`、`Q_bucket`、query
  分布、descriptor 与执行模式。不能把所有样本的真实 batch size 都设成 `B_max`。
- 保留上游 target/draft 成本建模，不退化成一张忽略请求数的通用 Q 表；若同 Q 不同布局差异
  显著，需要验证采样代表性与模型适用范围。测量、控制器选择、最终 dispatch 必须相互对应。
- 看原始重复样本、median、波动与 `cg=T/F/mix`，再看单调化后的表。非单调不直接证明表错误，
  平坦也可能是真实代价；不能据此预设必须裁剪或 FULL 一定更快。
- 清楚标记测量边界，边界外的 confidence、CPU 决策、TP 通信和 metadata 开销单列；最终用
  每秒有效输出、生成轮数、TPOT 和吞吐评价，不只算单步 target 节省。

**C2. 把人工 cap 换成真实 `get_num_tokens` 决策**（confidence + cost table）。

**C3. TP 一致性（TP4）。** 已有的 TP4 跑只能记为**功能冒烟通过**，不能替代逐 rank 一致性验证。
cost curves 上游已跨 TP 广播（vLLM `adaptive_verification.py:221`），所以要逐项比：
**总预算 / 逐请求 capacities / 请求映射 / query 边界**；confidence 路径还要看各 rank 的 stale
confidence 是否逐元素一致（draft 侧有 allreduce 时 `argmax` 可能翻转）。人工 cap 路径的确定性
是结构性的（纯函数 of scheduled counts），风险更低，但**低不等于已验**。

### 阶段 D —— 三臂对照（收益判定）

图模式与裁剪策略是两个实验维度：比较 FULL/PIECEWISE 时固定裁剪策略，比较下面三臂时固定
图模式与其余优化。PIECEWISE 对照必须确认 GDN 实际执行位置，不把静默 eager 回退算作入图。

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
- **真实执行观测**：先完成 F3 未修基线，再在阶段 C/D 增加预算与耗时；字段契约见设计 §1.9。
- **文章后续审计项**：概率与对应图/固定输出地址、请求集合及轮次绑定且只消费一次；检查 DSpark
  是否存在文章所述 DFlash KV tail 无效写入；batch 收缩清理整个可读 inactive tail。详见设计
  §1.10。DFlash 的故障不能直接记为 DSpark 缺陷，人工 cap 也不依赖 confidence 决策。
- 每步优先 cherry-pick 上游 > 自研；GDN×AV 胶水是必须原创。
- commit 规则：仅 `Signed-off-by: ChunnX`，**且不带任何 PR/issue 引用**。

## 5. 风险与工时
- **风险排序（2026-09-16 重排）**：
  1. **GDN 图捕获/回放契约**（当前阻塞）：分支修复已提交但未验收；CPU padding
     残留与身份推断、跨 bucket 请求轴/共享视图刷新、builder 标志早于 AV 覆盖模式仍需验证。
  2. **完整模型状态正确性仍未验收**——算子 UT 与少量 eager 冒烟通过**不能**划掉这一项。
  3. GDN 变长 state 跨轮正确性（"上轮接受 > 本轮验证长度"）：算子级与 eager 已有通过反馈，
     模型级逐轮状态对照未做。
  4. TP 逐 rank 决策/布局一致性（阶段 C3）。
  5. vendor 的算子跟踪上游 delta。
- **工时**：旧的“合计 ~5–6 周”未覆盖完整图契约工作，已失效。FULL 目标已确定，不能再以
  “选 PIECEWISE 可以省多少”估本项目工时；待 F3/F4 和固定轴对照取得证据后再估。
- **收益不确定性**：上游收益在 attention/高 TPS 上验证，GDN hybrid 未验——**由阶段 D 三臂
  对照判定**，不预设。

## 6. 变更历史

| 日期 | 内容 | 验证状态 |
|---|---|---|
| 2026-09-16 | 同步 `2bb8a9c5c` 修复状态；固化 F0/F3/F4 优先顺序、固定轴实验与 F2 配套；明确 FULL 目标及图模式/裁剪策略两个对照维度 | 文档修订；F1′/F3 未实施，未新增运行端精度或性能结果 |
