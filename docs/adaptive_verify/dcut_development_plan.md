# D-Cut 自适应验证开发计划（Qwen3.6-27B / GDN / 910B）

> 目标：在 vLLM-Ascend（MRV2 / worker v2）上，为 **Qwen3.6-27B（GDN 混合模型）** 落地
> DSpark 置信度调度的**动态验证（D-Cut）**——按每请求置信度 + Ascend 代价表，动态裁掉
> 后位低价值 draft，而不是恒定验证全部 draft。主场景：**16/32 中高并发 + 长上下文**。
> 主芯片：**910B**。

## 0. 背景与结论（为什么做、怎么做）

- 上游已有实证：DSpark 置信度调度验证在 **attention 目标**（Qwen3-8B +12.1%、
  DeepSeek-V4-Flash +22.1%，draft=9）上有收益（PR #15098）。
- 我们已在本分支验证：DSpark 置信头在 **FULL_DECODE_ONLY** 下能产出**逐步变化、随位置
  衰减**的置信度（此前因图捕获被冻结，已修）；AV **cost-table profiling 在 Ascend 跑通**
  （`[AV-PROFILE-SMOKE] OK, monotonic=True`）。详见 `docs/../pr15098_..._feasibility.md §7`。
- 关键经济学（RFC #15149 亲述）："**removing a few logical tokens is useful only when it
  reaches a cheaper execution shape**"。低并发 + 短上下文时 verify 代价近似平坦，裁剪到不了
  更便宜的图形状 → 收益有限；**16/32 并发 + 长上下文**才是 D-Cut 的价值区间（长上下文下
  每个 draft query token 对长 KV 做 attention，裁一个省的 FLOPs 远大于短上下文）。

**策略：跟随上游 RFC #15149，能采纳就采纳，不重复造轮子。** 只有 GDN×AV 的胶水是必须
自研的部分。

## 1. 上游参考（权威来源）

| 编号 | 标题 | 状态 | 作用 | 缺口 |
|---|---|---|---|---|
| **RFC #15149** | add adaptive speculative (Dcut) verification | OPEN | 总设计。点名 Qwen3.5/3.6 的 GDN 变长 state 正确性是核心难点 | — |
| **#15098** | [MRV2] DSpark confidence-scheduled verification | OPEN | AV“大脑”：`model_runner` 接 `compact_batch`/`reallocate_drafts`/`get_num_tokens`/`record_confidences`；attention 后端（dsa/sfa/mla）变长 | **无 GDN 分支**（只对 `use_fia` 后端 pad） |
| **#15207** | [Ops] variable-length GDN operators | OPEN | GDN 变长算子 `npu_dcut_causal_conv1d` / `npu_dcut_recurrent_gated_delta_rule`（吃 query_start_loc + cache_indices + num_accepted_tokens），注册 910B/910_93/950 | **只加算子 + UT，没接进 `gdn.py`/`gdn_attn_builder`/AV 路径** |
| 上游 vllm #47808 | DSpark confidence-scheduled adaptive verification | merged (v0.28.0) | AV manager 本体（`adaptive_verification.py`） | — |
| 参考 | #15147(CLOSED 早期 Dcut)、#16417/#16258(mamba cache group)、#15778(GDN prefill aclnn) | — | 相关上下文 | — |

**两个 PR 各做一半，缺的正是把 #15207 算子接进 GDN 前向 + 让 GDN 元数据按 compact 后变长
重建——这块两个 PR 都没落地，是本计划的核心工作量。**

## 2. AV 决策与执行（上游 `AdaptiveVerificationManager` 机理）

- 决策信号 = **survival = 各位置 confidence 的累积乘积**（`get_num_tokens`）。
- `draft_budget = argmax(estimated_accepted / cost)`，只有 verify 代价随 budget 变；
  代价表由启动时 profiling（graph-bucket 感知）建成，**代价平坦时 argmax 选满 budget → 不裁**。
- 执行 = `compact_batch` + `reallocate_drafts` 在 `prepare_inputs` 里裁掉低价值 draft 后缀，
  保留每请求的 anchor，且保持 prefix 有序。
- **calibration**：上游用 **raw confidence，无任何按数据集重标定**——标定是 checkpoint（训练）
  的责任。我们观测到的尾部过自信当作该 checkpoint 的体检结论反馈，**不做下游 affine 拟合**
  （不具通用性）。

## 3. 开发阶段

### 阶段 0 —— 对齐与止损（~1–2 天）
- 盯 RFC #15149 / #15098 / #15207 进度；确认上游有没有正在出“GDN×AV 胶水”PR，**若上游即将
  落地则直接采纳**。
- 钉死要 cherry-pick 的 #15098 / #15207 基线 SHA（都在动）。
- 确认 910B 上 #15207 dcut csrc 的编译路径。
- 产出：采纳清单 + 基线 SHA + 编译验证。

### 阶段 1 —— 影子验证（安全、无状态风险）✅ 本次实现
- 用 profiling 建好的**真实 cost table** + 现在活的 confidence，每步 record-only 地复算
  与上游 `get_num_tokens` **相同的决策数学**（survival=cumprod、cost argmax），只打日志：
  `会选的 draft_budget vs 满 budget`、trim%、以及预估“接受 token 损失”。**不改任何 GDN 态。**
- 在真实操作点跑：**16 / 32 并发 × 短/长上下文**。
- 闸门：若各 regime 下 ≈ 满 budget → 收益有限，止步；若明显裁 → 进入阶段 2+。
- 环境变量：`VLLM_ASCEND_DSPARK_AV_SHADOW=1`（需 confidence 活，故与 observe 共用捕获修复）。
- 日志：`[DSPARK-AV-SHADOW]`。实现镜像上游 `get_num_tokens`，阶段 3 换成真 manager 调用。

### 阶段 2 —— 落 #15207 GDN 变长算子并验精度（910B，~1 周）
- 把 dcut csrc 接进构建，编 910B。
- 跑自带 UT（`test_dcut_causal_conv1d.py` / `test_dcut_recurrent_gated_delta_rule.py`）。
- **加退化用例**：所有请求定长 num_spec 时，dcut 结果须与现有定长 GDN 算子逐元素一致。
- 产出：两个 `npu_dcut_*` 算子在 910B 可调、精度过关。

### 阶段 3 —— 采纳 #15098 的 AV 大脑（先验 attention 路，~1 周）
- cherry-pick/适配 #15098 的 `model_runner` 接线 + `aclgraph_utils` + patch + speculator，
  与本分支 fia_sink / observe / capture 栈调和。
- **先在纯 attention 路径验证**：能裁 + **greedy 输出 == AV 关闭**。
- 产出：attention 层 AV 裁剪在本分支正确。

### 阶段 4 —— GDN 胶水（唯一原创、RFC 核心难点，~2–3 周）
- **4a** `gdn_attn_builder`：按 compact 后每请求变长 draft 数重建 spec 元数据
  （`spec_query_start_loc` / `spec_state_indices`·`cache_indices` / `num_accepted_tokens`）。
- **4b** `gdn.py` 前向：AV 生效时把 conv/ssm 走 `npu_dcut_*`，传请求边界 + 接受位置 + state-slot 索引。
- **4c** `model_runner`：把 AV 压缩路径从“只 `use_fia` attention”扩到 **mamba/GDN 层组**。
- **4d** 图捕获：裁出的参差形状 vs 固定捕获形状——捕 compact 后 bucket，或 GDN verify 走
  piecewise/eager（复用已摸透的 FULL-graph 捕获经验）。
- **最大风险**：变长 draft 前缀下 GDN state slot/position 正确性（RFC 反复警告）。

### 阶段 5 —— 端到端正确性 + 性能 + TP（~1 周）
- **正确性铁律**：temp=0 greedy 下 **AV 开 == AV 关，逐 token 相同**；mean accept length 一致。
- **TP 一致性**：裁剪决策各 rank 完全一致（rank0 决策 + 广播，或输入完全一致），TP4 验。
- **性能**：16/32 并发 × 长上下文，AV 开/关 TPS + 接受率，验证阶段 1 预估兑现。
- 产出：Qwen3.6-27B on 910B 的 AV 验证 + 性能/精度报告。

## 4. 贯穿项
- 环境变量：`VLLM_ASCEND_DSPARK_AV_OBSERVE`（观测）、`VLLM_ASCEND_DSPARK_AV_PROFILE_SMOKE`
  （代价表 smoke）、`VLLM_ASCEND_DSPARK_AV_SHADOW`（影子决策）；后续加真正的 enable-trim 开关。
  observe / shadow 保留为调试工具。
- 每步优先 cherry-pick 上游 > 自研；只有阶段 4 是必须原创。
- commit 规则：仅 `Signed-off-by: ChunnX`。

## 5. 粗估与风险
- 阶段 1 几天 → 2/3 各 ~1 周 → 4 ~2–3 周 → 5 ~1 周，**合计 ~5–7 周**，需固定 owner + 910B
  机器 + NPU 侧 review。
- 核心不确定性：两个上游 PR 未合入（会漂）、GDN 变长 state 正确性、图捕获、TP 一致性、
  cost table 的上下文代表性（当前只在长上下文 profile 一次）。
