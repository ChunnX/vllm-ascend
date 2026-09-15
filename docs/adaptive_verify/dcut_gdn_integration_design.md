# D-Cut GDN 接入设计（阶段 B：人工 cap 打通最小变长链路）

> 目标：在 Qwen3.6-27B（GDN）上，用 #15207 的 `npu_dcut_*` 算子完成**连续多轮人工变长验证**，
> 输出 + 逐轮 GDN/KV 状态对照参考实现通过。**与 confidence/预算解耦**——先用人工 cap 把变长链路
> 打通、把状态正确性钉死，再谈真实 AV manager（阶段 C）。

## 1. 两个 hook 点（GDN 前向的 spec 路径）

`vllm_ascend/ops/gdn.py::_forward_core` 的 spec（多 query 草稿验证）分支里两处算子调用：

| 位置 | 现有算子 | 换成 | 关键入参（现有已具备） |
|---|---|---|---|
| [gdn.py:324](../../vllm_ascend/ops/gdn.py) conv1d | `torch.ops._C_ascend.npu_causal_conv1d_custom` | `npu_dcut_causal_conv1d` | `query_start_loc_opt`、`cache_indices_opt`、`num_accepted_tokens_opt` |
| [gdn.py:469](../../vllm_ascend/ops/gdn.py) recurrent | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` | `npu_dcut_recurrent_gated_delta_rule` | `actual_seq_lengths`(=query_start_loc)、`ssm_state_indices`、`num_accepted_tokens` |

**关键观察**：现有 spec 路径**已经**用 `query_start_loc` / `actual_seq_lengths`(每请求，ragged 可表达)
+ `num_accepted_tokens` + `ssm_state_indices` 驱动 conv/ssm。定长 DSpark 下这些是**均匀**的
（每 spec 请求 num_spec）。dcut 算子吃的是**同一批入参**，差别在**内核**能正确处理"每请求验证长度
不同 + 上一轮接受数与本轮长度不一致"的变长情况（RFC #15149 反复警告的正确性点）。

dcut 算子签名（已并入 `csrc/torch_binding.cpp`）：
```
npu_dcut_causal_conv1d(output, x, weight, conv_state, bias?, query_start_loc?,
                       cache_indices?, num_accepted_tokens?, activation_mode=0, pad_slot_id=-1) -> output
npu_dcut_recurrent_gated_delta_rule(query, key, value, state, beta?, scale?,
                       query_start_loc?, ssm_state_indices?, num_accepted_tokens?, g?, gk?,
                       zero_padded_output=False) -> output
```

## 2. 元数据（谁产出变长布局）

spec 元数据由 `vllm_ascend/ops/gdn_attn_builder.py::AscendGDNAttentionMetadataBuilder` 建：
`spec_query_start_loc` / `spec_state_indices_tensor` / `num_accepted_tokens` /
`spec_decode_metadata.spec_causal_conv1d.{query_start_loc,cache_indices,num_accepted_tokens}` /
`spec_decode_metadata.actual_seq_lengths`。

定长下这些按 num_spec 均匀构造。**变长要点**：让上面这些按**裁剪后的每请求验证长度**构造——
`spec_query_start_loc` 是 cumsum（天然支持 ragged），`actual_seq_lengths` 逐请求，`num_accepted_tokens`
逐请求；`spec_state_indices` 的 state-slot 布局要和裁剪后的每请求 token 数对齐。

## 3. 人工 cap（与 confidence 解耦）

阶段 B 不接真实 AV manager。用**人工 cap**产出每请求的裁剪验证长度，喂给上面的元数据：
- 环境变量（拟）：
  - `VLLM_ASCEND_DSPARK_ENABLE_DCUT`（0/1）：spec 路径的 conv/recurrent 切到 `npu_dcut_*`。
  - `VLLM_ASCEND_DSPARK_MANUAL_CAP`：人工 cap 策略。先支持**固定 k**（每请求验证 min(k, num_spec)）；
    再支持**异构**（如按 req_index 轮换 k，制造每请求不同长度）、**cap=0**。
- 注入点：在 spec 元数据构造前，把每请求 draft/verify 计数按 cap 截断（改 `actual_seq_lengths` /
  `spec_query_start_loc` / `num_accepted_tokens` 的来源计数），保持 prefix 有序、保留 anchor。
- **好处**：出问题时和 confidence 决策无关，可单独排查 GDN 变长 state 正确性。

## 4. 首个里程碑 + 测试（RFC 的跨轮正确性点）

**里程碑**：Qwen3.6-27B 连续多轮人工变长验证，**输出 token 逐轮对照 + conv/ssm state 逐轮对照**
参考实现通过。

分层测试：
1. **退化/回归**：`cap=num_spec`（不裁）时，dcut 路径必须与现有 `npu_causal_conv1d_custom` /
   `npu_recurrent_gated_delta_rule` 路径**逐元素一致**（先证 dcut 不引入回归）。
2. **跨轮变长**（逐个打开）：① 上一轮接受数 > 本轮验证长度（重灾区）；② 异构 cap、cap=0；
   ③ 请求重排 + slot 复用；④ 跨 cache block 的 pre-copy/postprocess；⑤ conv 与 recurrent state
   同步提交；⑥ padding 不污染有效状态。
3. **greedy 一致性**：temp=0 下，人工 cap（会降接受长度）不改变**已接受 token 序列的正确性**——
   输出 token 应与不裁时一致（只是每轮接受更少、轮数更多）。

## 5. 待实现清单（阶段 B）

- [ ] envs：`VLLM_ASCEND_DSPARK_ENABLE_DCUT`、`VLLM_ASCEND_DSPARK_MANUAL_CAP`。
- [ ] `gdn.py`：spec conv/recurrent 两处按 flag 切 `npu_dcut_*`（入参基本不变）。
- [ ] `gdn_attn_builder.py`：spec 元数据按人工 cap 的每请求长度构造。
- [ ] 人工 cap 注入（metadata 侧或 model_runner 侧，二选一，取决于哪层最干净）。
- [ ] 图捕获：变长参差形状 vs 固定捕获——**先走 eager/piecewise 做诊断**（明确区分诊断入口 vs
      正式 AV 执行；不是"原样接上游 AV"就能走 eager，manager 仍查 target builder 的 ALWAYS 能力）。
- [ ] 对照测试脚手架（上面 3 层）。

## 6. 开放问题（实现时定）

- `spec_state_indices` 在变长下的 state-slot 布局：dcut recurrent 用 `ssm_state_indices` 定位每请求
  state；裁剪后 token 数变了，slot 映射是否需要重排/重分配，要对着 dcut 内核对 `ssm_state_indices` /
  `num_accepted_tokens` 的期望语义确认。
- conv 与 recurrent 两步的 `num_accepted_tokens` 是否同一份、同一 int 类型（现有 recurrent 用
  `.to(torch.int32)`）。
- TP：裁剪长度各 rank 必须一致（人工 cap 是确定性的，天然一致；接真实 AV 时再验）。

依赖：#15207 算子在 910B 编译通过 + 自带 UT + `cap=num_spec` 退化用例过，才进第 2 层跨轮测试。
