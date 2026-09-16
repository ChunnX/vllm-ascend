# D-Cut GDN 接入设计（阶段 B0/B：图契约 + 人工 cap 打通真实变长执行）

> 目标：Qwen3.6-27B（GDN）用 vendor 的 `npu_dcut_*` 算子完成**连续多轮人工变长验证**，
> 输出 + 有效 GDN/KV 状态对照参考通过。与 confidence/预算解耦——先用人工 cap 打通并钉死状态
> 正确性，再谈真实 AV manager（阶段 C）。
>
> 行号基准：`76780b1a1`（Ascend）/ v0.28.0（vLLM）。
>
> **2026-09-15 修订**：上一版把"替换两个算子"和"模型级裁剪"混为一谈，且有多处接口语义错误
> （`actual_seq_lengths` vs `query_start_loc`、`num_accepted` 被误裁、state index 布局、注入层）。
> 本版先把**接口契约**定死——契约错了就是"算子能调、单轮对、跨轮状态已错"。
>
> **2026-09-16 修订**：算子契约（§1.1–1.4）已由 UT + eager 冒烟印证；新暴露的是**图层契约**
> （§1.6、§1.7）。cap=99 在 FULL 图下仍乱码、接受率全 0，而 eager 下 cap=99/2 精度正确，
> 所以当前阻塞在捕获/回放的分支与缓冲区契约，不在算子、不在 confidence。

## 1. 接口契约（先定死，再实现；每条都要在动手前明确）

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
  **要区分"缓冲区容量"与"本次调用的 B"**：FULL 图下本次调用的 B 是 `m.num_reqs`
  （= descriptor 的请求槽数），有效行在前、其后为 padding 行。

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
  capacities[i] = min(manual_cap[i], scheduled_drafts[i])   # 受实际 scheduled draft 约束
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

### 1.6 捕获与回放必须走**同一条算子路径**（2026-09-16 新增，当前阻塞）
FULL 图的 Python 分支在捕获那一刻定形，回放不重选分支。所以 GDN 的 `spec` / 普通 decode 分支
选择必须在捕获和回放之间**恒等**。当前不恒等：

1. AV manager 非空 → base 覆盖 `cudagraph_mode = FULL_AND_PIECEWISE` 并传 `varlen_decode=True`
   （vLLM `model_runner.py:574-587`）。
2. varlen descriptor 的请求数 = `min(num_tokens, max_num_reqs)`（vLLM `cudagraph_utils.py:227`），
   `make_dummy` 均分 token → **`num_tokens <= max_num_seqs` 的每个 bucket 都是 1 token/请求**。
3. 捕获期 draft 数 = `diff(query_start_loc) - 1` = 全 0。
4. builder 见"runtime draft 数之和为 0"就把 spec mask 清零转普通 decode
   （`ops/gdn_attn_builder.py:632-643`）→ **图里捕获的是普通 decode 的 conv/recurrent**。
5. 回放时真实 batch 有 draft → 构造 spec metadata，但执行的仍是捕获的普通 decode 算子及其
   捕获期缓冲区地址 → 增量乱码、逐位接受率全 0。

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

### 1.7 padding 请求行必须是**零长度**（2026-09-16 新增）
- **目标契约**：FIA 给每个 padding 图行一个完整 K+1 span，但 GDN recurrent 要求这些行零长。
  执行者是 `_remove_spec_graph_padding_queries`（`ops/gdn_attn_builder.py:57`）。
- **当前不满足**：`num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)`
  （`worker/v2/model_runner.py:351`）里的 `num_tokens` 是裁剪后的值，所以裁剪后 token 数一旦
  不正好落在 capture size 上，`_pad_adaptive_query_start_loc_for_fia`（`:1112`）就把正长度
  均摊给 padding 请求。复现布局：11 请求 / cap=2 / 33 token / 40-token 图 / 32 请求槽 → 21 个
  padding 行里 7 个拿到长度 1。
- **后果链**：这些行 draft 分类是 `-1` → 非 spec → 与真实 spec 行混合后计入 `num_prefills`
  → **跳过 `:1041` 的 pure-spec 持久缓冲区块**（该块要求 `num_prefills == 0 and num_decodes == 0`）
  → spec metadata 退回每步新建的临时张量 → FULL 回放读捕获期地址。
- **更严重的是执行者当前不可达**：见 §5 步骤 ⑤（重复 `build()`）。
- **关于"污染"**：padding 行写的 `NULL_BLOCK_ID = 0` 是 vLLM 保留的 null block，**这一个动作
  本身是设计意图**；但不等于整条混合路径安全，仍需按有效状态索引与有效输出验收，不提前给
  安全结论。
- **接回时要验**：`_remove_spec_graph_padding_queries` 会把 `num_actual_tokens` 改成真实
  （未 padding）token 数。需确认 FULL 捕获期（bucket 大小）与回放期（真实值）之间没有被捕获
  进图的形状假设被破坏——`ops/gdn.py:305` 的 `mixed_qkv[:num_actual_tokens]`；SPEC_ONLY 分支
  不使用 `spec_token_indx`（`ops/gdn.py:312-314`），所以它的长度变化在该分支无影响。

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
- **TP**：人工 cap 阶段就检查**各 rank 对同一请求的 cap 和布局一致**（不必等真实 AV）；人工 cap 是
  确定性的，应作为可验证条件写出来，而不是"天然一致"。已跑通的 TP4 只能记为**功能冒烟**，
  要逐项比**总预算 / 逐请求 capacities / 请求映射 / query 边界**。

## 5. 实施顺序（2026-09-16 重排：图契约插到最前）

| 步骤 | 工作 | 状态 / 通过标准 |
|---|---|---|
| ① | GDN spec 两处切 D-Cut，纠正累计偏移(§1.1) + 二维状态表(§1.2)，**保持原验证长度(不裁)** | ✅ NPU 回归过：开/关 `ENABLE_DCUT` 输出逐词一致 |
| ② | 跨轮算子参考 + NPU 对照（`_dcut_recurrent_golden` 扩多轮 + conv 多轮） | ✅ 910B 4 个 UT 过；**待补**见下 |
| ③ | 人工 capacities 接进现有 MRV2 裁剪链路（§1.4：`reallocate_drafts` 注入，跳过 confidence） | ✅ 6 个 CPU 策略 UT 过；eager cap=99/2 精度正确 |
| ⑤ | **修 `build()` 重复定义**，把两个前置处理接回生效路径（§1.7） | 🔴 待做，见下 |
| ⑥ | **统一捕获/回放算子路径**（§1.6） | 🔴 待做，当前阻塞项 |
| ④ | 连续多轮模型验证 | ⛔ 被 ⑤⑥ 阻塞：缩短→恢复、部分/全部 cap=0、重排、TP 布局一致性 |

- **⑤⑥ 要一起验，不要分开宣称修好**：cap=99 只受 ⑥ 影响（padding token 数为 0），
  cap<full 才会触发 §1.7 所述的正长度 padding，需要 ⑤。
- 模型状态按**同一已提交前缀**比较；浮点算子**明确容差**，不把"逐元素一致"默认 bitwise。

#### ⑤ 的具体落地（重复 `build()`）
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
  `compute_manual_capacities(scheduled_drafts, cap)` = `min(scheduled, cap)`（`cap<0` 不裁），
  结果逐元素 `<= scheduled`（合法 capacity）；`manual_batch_budget()` 出 `draft_budget=Σcap`。
  **不吃 confidence 参数——结构上无法依赖 confidence。**
- **manager 子类** `DcutManualCapVerificationManager`（同文件，惰性工厂建）：只重写两处 confidence 入口
  `get_num_tokens`（预算=Σ人工 cap，不查 cost table）+ `reallocate_drafts`（`capacities` 直接来自
  策略函数，替掉 `_assign_draft_token_budget` 排名），其余 `cu_num_logits`/`query_start_loc` cumsum 与
  上游**逐行相同**；并把 `batches_to_profile`/`set_initial_cost_curves`/`record_confidences` 置空
  （人工预算不需要 cost model，也不碰 confidence head）。
- **接线**（`patch/worker/patch_v2/patch_adaptive_verification.py:41`）：GDN 被上游工厂
  `maybe_create_adaptive_verification_manager` 拒绝（varlen backend 检查 + 要求 `ALWAYS`）→ 原状
  `adaptive_verification=None`，裁剪链路根本不跑。patch 工厂：`ENABLE_DCUT=1 且 MANUAL_CAP>=0` 时
  返回人工 manager，否则原样交回上游。这样 `initialize_kv_cache`（vLLM `model_runner.py:540`）
  拿到非 None manager，base 的 `cudagraph_mode=FULL_AND_PIECEWISE`（`:573`）等 setup 照跑。
- **env**：`VLLM_ASCEND_DSPARK_DCUT_MANUAL_CAP`（int，默认 -1 关）；仅
  `VLLM_ASCEND_DSPARK_ENABLE_DCUT=1` 时生效。`cap=0` 全裁到只剩 anchor；`cap` 很大 = ① 的不裁。
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
