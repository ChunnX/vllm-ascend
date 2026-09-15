# D-Cut 自适应验证开发计划（Qwen3.6-27B / GDN / 910B）

> 目标：在 vLLM-Ascend（MRV2 / worker v2）上，为 **Qwen3.6-27B（GDN 混合模型）** 落地
> DSpark 置信度调度的**动态验证（D-Cut）**——按每请求置信度 + Ascend 代价表，动态裁掉
> 后位低价值 draft，而不是恒定验证全部 draft。主场景：**16/32 中高并发 + 长上下文**。
> 主芯片：**910B**。

## 0. 背景与结论（为什么做、怎么做）

- 上游已有实证：DSpark 置信度调度验证在 **attention 目标**（Qwen3-8B +12.1%、
  DeepSeek-V4-Flash +22.1%，draft=9）上有收益（PR #15098）。
- 我们已在本分支验证：DSpark 置信头在 **FULL_DECODE_ONLY** 下**不再冻结**、能产出逐步变化的
  置信度（此前因图捕获冻结，已修）；AV cost-table profiling 链路**能执行**
  （`[AV-PROFILE-SMOKE] OK`）。**注意**：`monotonic=True` 是建表时 `np.maximum.accumulate`
  造出来的、不是测量稳定性证明；`set_initial_cost_curves` 也已按 shape 取 median。详见
  `docs/../pr15098_..._feasibility.md §7`。
- 关键经济学（RFC #15149 亲述）："**removing a few logical tokens is useful only when it
  reaches a cheaper execution shape**"。**推论（待验证，非结论）**：低并发 + 短上下文时若
  verify 代价近似平坦，裁剪到不了更便宜的图形状 → 收益有限；中高并发 / 长上下文可能进入陡增
  区才有价值。**但此推论依赖代价表能代表真实 AV varlen verify 路径**——当前 smoke 用的是临时
  manager、未建真正 varlen 图，且 dummy batch 可能跨 eager / uniform-FULL 路径（`tok256≈89ms`
  vs 邻居 ~220ms 就是执行模式差异的信号），所以**在代价表可信前,不能据平坦性止损**。

**策略：跟随上游 RFC #15149，能采纳就采纳，不重复造轮子。** 只有 GDN×AV 的胶水是必须
自研的部分。

## 1. 上游参考（权威来源）

| 编号 | 标题 | 状态 | 作用 | 缺口 |
|---|---|---|---|---|
| **RFC #15149** | add adaptive speculative (Dcut) verification | OPEN | 总设计。点名 Qwen3.5/3.6 的 GDN 变长 state 正确性是核心难点 | — |
| **#15098** | [MRV2] DSpark confidence-scheduled verification | **MERGED**（`990243c4c5b0`，2026-09-14 14:15 UTC）| AV“大脑”：`model_runner` 接 `compact_batch`/`reallocate_drafts`/`get_num_tokens`/`record_confidences`；attention 后端（dsa/sfa/mla）变长 | **无 GDN 分支**（只对 `use_fia` 后端 pad） |
| **#15207** | [Ops] variable-length GDN operators | OPEN | GDN 变长算子 `npu_dcut_causal_conv1d` / `npu_dcut_recurrent_gated_delta_rule`（吃 query_start_loc + cache_indices + num_accepted_tokens），注册 910B/910_93/950 | **只加算子 + UT，没接进 `gdn.py`/`gdn_attn_builder`/AV 路径** |
| 上游 vllm #47808 | DSpark confidence-scheduled adaptive verification | merged (v0.28.0) | AV manager 本体（`adaptive_verification.py`） | — |
| 参考 | #15147(CLOSED 早期 Dcut)、#16417/#16258(mamba cache group)、#15778(GDN prefill aclnn) | — | 相关上下文 | — |

**两个 PR 各做一半，缺的正是把 #15207 算子接进 GDN 前向 + 让 GDN 元数据按 compact 后变长
重建——这块两个 PR 都没落地，是本计划的核心工作量。**

### 1.1 可复用 vs 必须自研

| 可以复用 | 我们必须完成 |
|---|---|
| #15098 的预算选择、confidence 流水线、MRV2 接线 | Qwen3.6 hybrid metadata 与裁剪后请求布局一致 |
| D-Cut 的变长 conv/GDN 算子（#15207） | 算子接入、state 索引、跨轮接受状态与回退 |
| 上游图管理框架 | GDN 的 varlen 捕获、padding、replay 参数更新 |
| 现有定长 DSpark 优化 | FIA Sink、GDN metadata 复用、KV grouping 与 AV 的兼容验证 |

**注意**：结合是正确路线，但工作量**不是"合入两个 PR 就完成"**。最值得尽早验证的是
**"上一轮接受长度大、下一轮验证长度变短"时 GDN 状态是否仍正确**——这不会因 confidence /
预算算法成熟而自动解决。

## 2. AV 决策与执行（上游 `AdaptiveVerificationManager` 机理）

- 决策信号 = **survival = 各位置 confidence 的累积乘积**（`get_num_tokens`）。
- `draft_budget = argmax(estimated_accepted / cost)`，只有 verify 代价随 budget 变；
  代价表由启动时 profiling（graph-bucket 感知）建成，**代价平坦时 argmax 选满 budget → 不裁**。
- 执行 = `compact_batch` + `reallocate_drafts` 在 `prepare_inputs` 里裁掉低价值 draft 后缀，
  保留每请求的 anchor，且保持 prefix 有序。
- **calibration（口径已勘误）**：confidence[i] 是**条件**每位接受概率,上游 cumprod 后才是
  survival；而 observer 的 acc[i] 是**前缀 survival** P(accepted≥i+1)。正确标定必须比
  **逐样本 `cumprod(confidence)` vs acc**（且逐样本累乘再平均,不能对均值累乘）。之前"尾部
  过自信→少裁"是拿条件比边际的错误口径,**已撤回**,按正确口径重采后再下结论。上游用 raw
  confidence、无按数据集重标定——标定是 checkpoint（训练）的责任,**不做下游 affine 拟合**。

## 3. 开发阶段（2026-09-15 调整：正确性优先，不再以 shadow 为前置闸门）

**决策**：#15098 已合入（`990243c4c5b0`，2026-09-14 14:15 UTC），attention 目标上验证有收益；
把 shadow 完善到"能决定项目去留"要再造一套 stale/ragged 调度模拟（≈ 直接做真实 manager），
ROI 低。故 **shadow / observe / smoke 全部降级为调试工具（默认关、非闸门）**；**收益问题由阶段 D
的真实 AV 开/关 A/B 回答，不预设**（#15098 的收益在 attention/高 TPS 上验证，GDN hybrid 未验、
DeepSeek draft=7 甚至 −7%）。观测改为围绕**真实执行链**：选了什么预算、实际跑了什么形状、耗时多少。

> **首个目标**：Qwen3.6-27B 在 MRV2 上，用 D-Cut GDN 算子完成**连续多轮人工变长验证**，
> 输出与 GDN/KV 状态对照通过。这消除当前最大的工程不确定性——"上一轮接受长度大、下一轮验证
> 长度变短"时 GDN 状态是否仍正确（不会因 confidence/预算成熟而自动解决）。

### 阶段 A —— 对齐合入版 #15098 + 定长基线（保留现有优化，~1 周）
- 审计合入版（逐行看 model_runner / aclgraph_utils / patch / speculator 改动），迁到本分支，
  与 fia_sink / observe / capture 栈调和；钉基线 SHA；确认 910B dcut csrc 编译路径。
- 建**保留现有优化的定长基线**并逐项验**无回归**：64K head、FIA Sink、GDN metadata 复用、
  KV grouping、sampling（AV 接入后，尤其 **shared metadata 必须按裁剪后布局构造**）。

### 阶段 B —— D-Cut 算子 + 人工 cap 打通最小 GDN 变长链路（核心难点，~2–3 周）
- 编 910B 的 `npu_dcut_causal_conv1d` / `npu_dcut_recurrent_gated_delta_rule`；退化用例（全定长
  num_spec）须与现有定长 GDN 算子逐元素一致。
- **人工 cap（与 confidence 解耦）**：先固定 cap 打通链路，再测 ① 异构 cap、cap=0；② 跨轮缩短
  （上轮接受 > 本轮验证长度）；③ 请求重排 + slot 复用；④ 跨 cache block pre-copy/postprocess；
  ⑤ conv 与 recurrent state 同步提交；⑥ padding 不污染有效状态。
- 接入点：`gdn_attn_builder` 按裁剪后布局重建 spec 元数据（`spec_query_start_loc` /
  `spec_state_indices`·`cache_indices` / `num_accepted_tokens`）；`gdn.py` 走 dcut 算子，传请求
  边界 + 接受位置 + state-slot 索引；`model_runner` 压缩路径覆盖 **mamba/GDN 层组**（#15098 只覆盖
  `use_fia` attention）。
- **产出 = 首个目标**：连续多轮人工变长验证，输出 + 逐轮 GDN/KV 状态对照参考实现通过。因与
  confidence 解耦，出问题可排除决策干扰。**这是当前最该先啃的一步。**

### 阶段 C —— 接真实 AV manager + 图模式 + TP4（~1 周）
- 状态链正确后，把人工 cap 换成真实 `get_num_tokens` 决策（confidence + cost table）。
- 图捕获：裁出的参差形状 vs 固定捕获——捕 compact 后 bucket，或**明确区分诊断入口 vs 正式 AV
  执行**（上游 manager 仍查 target builder 的 `ALWAYS` 能力并配 varlen 图，不是"原样接入"就走
  eager/piecewise）。
- **TP 一致性从首次多卡裁剪就验（TP4）**：裁剪决策各 rank 完全一致（rank0 决策 + 广播，或输入
  完全一致），不拖到最后。

### 阶段 D —— 真实 AV 开/关对照（收益判定，~1 周）
- 两组配置（现有优化全一致）对照：真实送验长度、每轮接受长度、target/draft 耗时、TPOT、吞吐、
  尾延；16/32 并发 × 长上下文。
- **正确性**（不是"接受长度不变"——裁剪本就会降接受长度）：① temp=0 greedy 输出逐 token 与参考
  一致；② 多轮 KV/SSM 态逐轮一致；③ 随机采样验分布/采样语义。**允许接受长度下降。**
- 收益不足则用真实执行观测（预算 / 形状 / 耗时）定位：代价表 / confidence / 图分档 / 同步开销。

## 4. 贯穿项
- 环境变量：`VLLM_ASCEND_DSPARK_AV_OBSERVE`、`VLLM_ASCEND_DSPARK_AV_PROFILE_SMOKE`、
  `VLLM_ASCEND_DSPARK_AV_SHADOW`——**默认关的调试工具，非阶段闸门**；后续加真正的 enable-trim 开关。
  observe 的口径勘误（逐样本 cumprod survival、slot 对齐、NaN/unmatched 计数）已修完，但**不为它
  再建完整校准系统**；shadow 保留为调试参考，不再据其 trim% 决定去留。
- **真实执行观测**（阶段 C/D 加）：记录"选了什么预算、实际执行了什么图形状、耗时多少"，围绕真实
  执行链，比复刻影子 manager 更有价值。
- 每步优先 cherry-pick 上游 > 自研；GDN×AV 胶水（阶段 B/C）是必须原创。
- commit 规则：仅 `Signed-off-by: ChunnX`，**且不带任何 PR/issue 引用**。

## 5. 粗估与风险
- 阶段 A ~1 周 → **B ~2–3 周（核心难点）** → C ~1 周 → D ~1 周，**合计 ~5–6 周**，需固定 owner +
  910B 机器 + NPU 侧 review。
- 核心不确定性（按风险排序）：**① GDN 变长 state 正确性**（阶段 B 首验，"上轮接受 > 本轮验证
  长度"是重灾区）；② varlen 图捕获 / padding / replay 参数；③ TP 决策一致性（阶段 C 首个 TP4
  就验）；④ #15207 仍未合入（会漂）。
- **收益不确定性**：#15098 收益在 attention/高 TPS 上验证，GDN hybrid 未验——**由阶段 D 真实
  开/关 A/B 判定**，不预设；不足时用真实执行观测（预算/形状/耗时）定位代价表 / confidence /
  图分档 / 同步开销。
