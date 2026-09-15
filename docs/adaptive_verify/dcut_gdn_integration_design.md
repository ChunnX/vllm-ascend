# D-Cut GDN 接入设计（阶段 B：人工 cap 打通真实变长执行）

> 目标：Qwen3.6-27B（GDN）用 #15207 的 `npu_dcut_*` 算子完成**连续多轮人工变长验证**，
> 输出 + 有效 GDN/KV 状态对照参考通过。与 confidence/预算解耦——先用人工 cap 打通并钉死状态
> 正确性，再谈真实 AV manager（阶段 C）。
>
> **2026-09-15 修订（review 后）**：上一版把"替换两个算子"和"模型级裁剪"混为一谈，且有多处接口
> 语义错误（`actual_seq_lengths` vs `query_start_loc`、`num_accepted` 被误裁、state index 布局、
> 注入层）。本版先把**接口契约**定死——契约错了就是"算子能调、单轮对、跨轮状态已错"。

## 1. 接口契约（先定死，再实现；每条都要在动手前明确）

### 1.1 recurrent 传**累计** `query_start_loc`，不是旧 `actual_seq_lengths`
现有 `_build_actual_seq_lengths()`（`gdn_attn_builder.py:130`）存的是 `[起始, 逐请求长度]`：
长度 2、3 → `[0, 2, 3]`。而 dcut recurrent 要**累计偏移** `query_start_loc = [0, 2, 5]`。
**误传会把第二个请求当成长度 1。** → dcut recurrent 传对应 spec 子批次的 cumsum，不能沿用旧入参名/值。

### 1.2 `ssm_state_indices` 是 `[B, S]` 二维表，**不 flatten、不随 token 数压缩**
- 旧调用 `spec_state_indices_tensor.flatten()`（`gdn.py:478`）是给旧算子的 1D 布局。
- dcut host 要求 `ssm_state_indices` 为 `[B, S]`；内核用 `batch_i * S + accepted - 1` 读起始状态、
  在该请求行的对应列存本轮候选状态。
- **契约**：token 可按当前长度紧凑排布，但**状态索引表必须保持 `[B, S]`**，仍能定位"上一轮接受
  位置"对应的状态。要写清：`S` 的来源与容量（固定状态行宽，通常 = num_spec+1 一类）、去掉
  `flatten()`、**请求重排时整行 `[.,S]` 一起重排**。

### 1.3 人工 cap **不截断** `num_accepted_tokens`（两个独立量）
- **本轮 cap**：决定本轮处理几个 query。
- **上一轮 accepted**：决定本轮从哪个历史状态起步（内核按固定状态行宽用它索引）。
- **契约**：**禁止** `num_accepted = min(previous_accepted, current_query_len)`。上一轮选 6，本轮只处理
  2 个 query，仍必须读第 6 个状态。并写清 `num_accepted` 是否含 anchor（与"接受的 draft 数"区分）。

### 1.4 人工 cap 是**模型级 compaction**，不是只改 GDN 元数据
只把累计偏移改成 `[0,2,5]` 而不重排真实输入,请求 2 会读到 `A2 A3 B0`（错）。
- **契约**：cap 必须**统一作用于真实输入及其全部派生布局**：input_ids、positions、attention/GDN 的
  query 边界、KV slot mapping、验证采样的 draft/logits 索引与计数。GDN builder **消费**已经一致的
  裁剪结果,不能单独决定裁剪。
- 落点：复用 base 里 #15098 的 `compact_batch`/`reallocate_drafts`（prepare_inputs），用**人工预算**
  驱动（而非 confidence 的 `get_num_tokens`）。→ 独立算子测试只能证算子对,不能算"模型连续多轮验证"完成。

### 1.5 cap 语义 + cap=0 分支
- **定义**：`cap = 保留的 draft token 数`；普通 decode 请求 target query 数 = `1 + cap`。
  → 有效请求 `cap=0` 仍有 1 个 anchor query,**不等于 padding 空请求**。
- builder 分类（`gdn_attn_builder.py:595`）：batch 里还有正 draft 时零-draft decode 请求**进 spec 分支**;
  整批无 draft 则**进普通 decode**。→ 必须分别验"混合 batch 里部分 cap=0"与"全 batch cap=0",尤其
  **上一轮多 query spec → 本轮全零 cap → 下一轮恢复 spec 的状态衔接**。仅换 spec 分支两处调用无法证明。

## 2. 两个 hook 点（GDN spec 路径）

`gdn.py::_forward_core` 的 spec 分支:
| 位置 | 现有算子 | 换成 | 注意 |
|---|---|---|---|
| `gdn.py:324` conv1d | `npu_causal_conv1d_custom` | `npu_dcut_causal_conv1d` | 传裁剪后 `query_start_loc`(cumsum)+`cache_indices`+`num_accepted`(按 1.3 不截断) |
| `gdn.py:469` recurrent | `npu_recurrent_gated_delta_rule` | `npu_dcut_recurrent_gated_delta_rule` | 传 cumsum `query_start_loc`(1.1)、`[B,S]` `ssm_state_indices`(1.2)、`num_accepted`(1.3) |

**不是"改个算子名"**：recurrent 的 `actual_seq_lengths`→`query_start_loc`(值和语义都变)、state index
去 flatten 保 `[B,S]`,都是入参契约变化。

## 3. 首个里程碑

Qwen3.6-27B 连续多轮人工变长验证:**已提交 token 序列正确** + **有效 conv/SSM/KV 状态逐轮对齐参考**。

## 4. 测试参考点（避免比错对象）

- **算子级**：相同初始 state / 输入 / 人工长度,用**独立顺序参考**算,检查输出与候选状态;明确
  **数值容差 vs bitwise**(dcut 内核与参考实现不必 bit 相等,定 tol)。
- **模型级**：按**同一请求、同一已提交 token 前缀**对齐,比有效 conv/SSM/KV 状态;**未提交的候选槽位
  不要求与不裁剪运行相同**(裁剪与不裁剪同一轮结束可能到不同 token 前缀,不能直接比整体状态)。
- **TP**：人工 cap 阶段就检查**各 rank 对同一请求的 cap 和布局一致**(不必等真实 AV);人工 cap 是
  确定性的,应作为可验证条件写出来,而不是"天然一致"。

## 5. 实施顺序（review 修订）

1. **接口契约确定**（本文 §1，动手前写死：本轮长度 / 上一轮状态选择 / 模型整体输入布局 三者的契约）。
2. **D-Cut 不裁剪回归**：`cap=num_spec`,dcut 路径与现有算子路径逐元素一致(定 tol)。
3. **独立跨轮算子验证**：算子级对照跑 §1.5 的跨轮用例(上轮接受>本轮长度、异构 cap、cap=0、slot 复用、
   padding、状态衔接)——证算子对。
4. **MRV2 统一人工裁剪**：把人工 cap 接进模型级 compaction(§1.4),让 input_ids/positions/query 边界/
   slot mapping/draft·logits 索引与 GDN 元数据一致。
5. **模型连续多轮验证**：§3 里程碑 + §4 两类对照 + TP 一致性。

**不需要继续扩影子验证。** 要补的是"本轮长度 / 上一轮状态选择 / 模型整体输入布局"三者的契约——尤其
§1 的前五条,动手前写清,否则很容易"算子能调、单轮对、跨轮状态已错"。

## 6. 依赖
#15207 算子在 910B 编译通过 + 自带 UT + 第 2 步不裁剪回归过,才进第 3 步跨轮;第 4 步依赖 base 的
#15098 compaction 已在(已 rebase 进来)。
