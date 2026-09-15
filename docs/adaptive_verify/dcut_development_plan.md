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

## 3. 开发阶段

### 阶段 0 —— 对齐与止损（~1–2 天）
- **#15098 已合入 `990243c4c5b0`（2026-09-14 14:15 UTC）** → 改为**审计合入版**(而非等待)：
  逐行看它的 `model_runner`/`aclgraph_utils`/patch/speculator 改动,确认与本分支冲突面。
- 盯 RFC #15149 / #15207 进度；确认上游有没有正在出“GDN×AV 胶水”PR，**若上游即将落地则直接采纳**。
- 钉死要 cherry-pick 的 #15207 基线 SHA（仍 OPEN、在动）。确认 910B dcut csrc 编译路径。
- **补版本/实验配置 + 已有优化回归清单**：64K head、FIA Sink、GDN shared plan、KV grouping、
  sampling 的启用状态与回归要求（AV 接入后逐项回归,尤其 **shared metadata 必须按裁剪后布局构造**）。
- 产出：合入版审计 + 采纳清单 + 基线 SHA + 编译验证 + 回归清单。

### 阶段 1 —— 影子验证（安全、无状态风险）✅ 本次实现
- 用 profiling 建好的**真实 cost table** + 现在活的 confidence，每步 record-only 地复算
  与上游 `get_num_tokens` **相同的决策数学**（survival=cumprod、cost argmax），只打日志：
  `会选的 draft_budget vs 满 budget`、trim%、以及预估“接受 token 损失”。**不改任何 GDN 态。**
- 在真实操作点跑：**16 / 32 并发 × 短/长上下文**。
- **⚠ 受限稳态模拟器,非闸门**：shadow 硬编码 full-n draft / 单 anchor / 全 sampling / live（非
  stale）confidence,真实调用不保证满足这些,故 **trim≈0 不能据以止损、明显裁也不能据以拍板**。
  它是方向性探针,须在 (1) observer 口径勘误、(2) 代价表可信、(3) 加适用条件检查 + 与真实
  `get_num_tokens` 对照测试 之后才有决策力。已加 NaN/非有限行丢弃与计数。
- 环境变量：`VLLM_ASCEND_DSPARK_AV_SHADOW=1`（需 confidence 活，故与 observe 共用捕获修复）。
- 日志：`[DSPARK-AV-SHADOW]`。实现镜像上游 `get_num_tokens`，阶段 3 换成真 manager 调用。

### 阶段 2 —— 落 #15207 GDN 变长算子并验精度（910B，~1 周）
- 把 dcut csrc 接进构建，编 910B。
- 跑自带 UT（`test_dcut_causal_conv1d.py` / `test_dcut_recurrent_gated_delta_rule.py`）。
- **加退化用例**：所有请求定长 num_spec 时，dcut 结果须与现有定长 GDN 算子逐元素一致。
- **补 GDN 跨轮关键用例**（RFC 反复警告的正确性点）：① 上轮接受计数 > 本轮验证长度；
  ② 异构 cap、cap=0、请求重排与 slot 复用；③ 跨 cache block 的 pre-copy/postprocess；
  ④ conv 与 recurrent state 同步提交；⑤ padding 不污染有效状态。
- 产出：两个 `npu_dcut_*` 算子在 910B 可调、精度过关 + 上述跨轮用例通过。

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
  piecewise/eager。**注意**：不是"原样接上游 AV"就能走 piecewise/eager——上游 manager 仍会
  检查 target builder 的 `ALWAYS` 能力并配置 varlen 图,须**明确区分诊断测试入口 vs 正式 AV
  执行方案**。
- **4c/4d 起就上 TP4**：裁剪决策的 rank 一致性(见阶段5)是 GDN 态正确性的一部分,**首个 TP4
  裁剪测试就要验**,不能拖到最后一周。
- **最大风险**：变长 draft 前缀下 GDN state slot/position 正确性（RFC 反复警告）。

### 阶段 5 —— 端到端正确性 + 性能 + TP（~1 周）
- **正确性**（不是"接受长度不变"——裁剪本就会降接受长度）：① temp=0 greedy 下输出 token 与
  参考一致；② 多轮 KV/SSM 状态与参考逐轮一致；③ **随机采样**下验分布/采样语义,不只测 greedy。
- **性能**：**允许平均接受长度下降**,比最终有效输出吞吐、TPOT、尾延迟（16/32 并发 × 长上下文,
  AV 开/关）。
- **TP 一致性**：裁剪决策各 rank 完全一致（rank0 决策 + 广播,或输入完全一致）——**已在阶段
  4c/4d 的首个 TP4 测试引入**,此处做完整回归。
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
- 核心不确定性：#15207 仍未合入（会漂）、GDN 变长 state 正确性、图捕获、TP 一致性；以及
  **cost table 的可信度**——须记录每样本 `num_reqs` / query 布局 / context length / 实际 graph
  descriptor / `full_cudagraph`,确认比较的是**目标执行路径**(而非 eager/uniform 混采),才能
  用于投资判断。observer 口径(逐样本 cumprod survival)与请求/slot 对齐也待补齐后才能下标定结论。
