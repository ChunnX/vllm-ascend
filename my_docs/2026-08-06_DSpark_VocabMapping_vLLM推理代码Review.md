# DSpark Vocab Mapping vLLM Ascend 推理代码复审

日期：2026-08-06  
Review 分支：`dspark-vocab-mapping-inference`  
Review HEAD：`eb042af60`  
Review 范围：`dea3bc377..eb042af60`（包含 `1d9a70c65` 与后续修复 `eb042af60`）  
运行路径范围：**仅 model runner v1**；按需求不评审 model runner v2

改动文件：

- `vllm_ascend/spec_decode/llm_base_proposer.py`
- `tests/ut/spec_decode/test_dspark_vocab_mapping.py`
- `docs/source/user_guide/feature_guide/speculative_decoding.md`

本次重新从零核对的参考实现：

- SpecForge：`/Users/wuyidong/projects/github_projects/job/specforge_sgl0514/SpecForge-dspark-vocab`
- DeepSpec：`/Users/wuyidong/projects/github_projects/job/specforge_sgl0514/DeepSpec`，分支 `dspark-vocab-eval`，提交 `02b8155`
- 配套 vLLM：`/Users/wuyidong/projects/github_projects/job/vllm-dflash/vllm-workspace-dflash/vllm`

## 结论

建议 **修改后再合入**。

与上一版 review 相比，分支新增的 `eb042af60` 已经修掉了此前指出的几个关键问题：

- `target TP > 1 + draft TP = 1` 不再因为 target/draft head 的 group 对象不同而被错误拒绝；
- target head 是否量化改为根据 quant method 判断，FP8 不再绕过检查；
- 无自有 head 的派生路径会拒绝缺失、越界或非严格递增的 `d2t`；
- Qwen3 DSpark 的 ownership 默认值由真实 MRO 中的 `SupportsEagleBase` 提供，新增测试能够守住 `has_own_lm_head=False` 和 `has_own_embed_tokens=False`。

按用户确认，模型 config 已自行调整，并且正确 DeepSpec 仓库中的真实 vocab-mapping 测试能够跑通；本次不评审 SpecForge export/config normalization。对当前这类 **vanilla Markov、checkpoint 不带 `lm_head`、带非平凡 `d2t`** 的模型，核心推理算法现在看是正确的：

1. backbone logits 与 Markov `W2` bias 都停留在 K 维 draft vocab；
2. 每一步 argmax 后立即通过 `d2t` 映射为 target token id；
3. target id 被写入 proposal，同时回喂下一步 target-vocab `markov_w1`；
4. 无自有 `lm_head` 时，从 target head 选择 mapping 保留的 K 行；
5. target verifier 收到的始终是 target token id。

代码的通用 reduced-vocab 支持仍有 **1 个高优先级问题和 1 个中优先级问题**：

1. `d2t` 校验只在“checkpoint 不带 lm_head、需要派生 head”分支执行；文档明确支持的“checkpoint 自带裁剪 head”分支仍可在缺失 mapping 时静默按 identity mapping 推理。
2. 用 `d2t.any()` 判断 mapping 是否加载会错误拒绝一个合法映射：保留 target ids `0..K-1` 时 `d2t` 本来就全零。

## Findings

### [P1] checkpoint 自带裁剪 lm_head 时，缺失或损坏的 d2t 没有被校验

相关位置：

- `vllm_ascend/spec_decode/llm_base_proposer.py:467-490`
- `vllm_ascend/spec_decode/llm_base_proposer.py:593-628`
- `tests/ut/spec_decode/test_dspark_vocab_mapping.py:417-424`
- 配套 vLLM `vllm/model_executor/models/qwen3_dspark.py:149-184`

当前 mapping 校验位于 `_build_pruned_lm_head_from_target()` 内，而该函数只在下面条件成立时调用：

```python
draft_id_to_target_id is not None and not self.model.has_own_lm_head
```

如果 reduced-vocab checkpoint 带自己的 K 行 `lm_head`，routing 会进入“Keeping separate lm_head”分支，不执行以下任何校验：

- checkpoint 是否真的加载了 `d2t`；
- `i + d2t[i]` 是否落在 target vocab；
- kept target ids 是否严格递增且无重复。

配套 vLLM loader 在 config 声明 reduced vocab 时先创建一个全零 `draft_id_to_target_id` 参数。checkpoint 缺少 `d2t` 时，loader 只是把该参数加入 skip list；如果同一 checkpoint 带 `lm_head`，`has_own_lm_head=True`，vLLM Ascend 就会保留自己的 head 并继续启动。运行时：

```text
target_id = draft_id + 0
```

即裁剪 head 的第 i 行会被静默解释成 target token i。shape 全部合法，但 token 语义错误，接受率会接近失效。

这不是纯理论分支，因为本 PR 文档明确承诺：“A checkpoint that does ship its own `lm_head` (already sliced) is used as-is.”

建议修复：

- 把 `d2t` presence/content 校验提取成独立方法；
- 只要 `draft_id_to_target_id is not None` 就在 `_maybe_share_lm_head()` routing 前调用，无论 head 是自带还是派生；
- loader 应显式保存“mapping key 是否加载”的状态，不能仅靠参数内容推断；
- 将现有 `test_pruned_checkpoint_with_its_own_lm_head_is_left_alone` 扩成两个失败测试：自带 head 但缺 mapping、自带 head 但 mapping 越界/重复。

### [P2] `d2t` 全零既可能表示未加载，也可能是合法的 first-K identity offset

相关位置：

- `vllm_ascend/spec_decode/llm_base_proposer.py:600-615`
- `tests/ut/spec_decode/test_dspark_vocab_mapping.py:324-333`
- SpecForge `specforge/data/preprocessing.py:695-741`
- SpecForge `specforge/core/compact_teacher.py:217-231`
- SpecForge `specforge/modeling/draft/vocab_mixin.py:100-114`

当前代码把下面条件定义为“checkpoint 没有 mapping”：

```python
if not bool(d2t_cpu.any()):
    raise ValueError(...)
```

但 SpecForge 的真实 mapping 公式是：

```text
d2t[i] = kept_target_ids[i] - i
```

如果某个合法训练映射恰好保留 target ids `0, 1, ..., K-1`，那么每个 offset 都是 0，`d2t` 必然全零。SpecForge 的一致性校验会接受它，因为对应 `t2d` 有前 K 个 True，且满足：

```text
nonzero(t2d) == d2t + arange(K)
```

SpecForge 用于判断“是否加载”的无歧义状态是 **all-False `t2d`**，不是 all-zero `d2t`。vLLM loader 丢弃了 `t2d`，因此单看 `d2t` 内容无法区分“合法 first-K mapping”和“参数保持零初始化”。现有单测把这个模糊状态固定成了必须失败，会拒绝合法 checkpoint。

建议修复：

- 在 Qwen3 DSpark loader 中记录 `includes_draft_id_mapping`，例如 `has_draft_id_mapping`；
- proposer 根据该 presence flag 报缺失，而不是根据 `d2t.any()`；
- 如果希望 serving 同时加载 `t2d`，也可保留 t2d 并用它做完整一致性校验；
- 将当前 all-zero failure test 改成两条：mapping key 缺失必须失败、mapping key 存在且 d2t 全零必须成功并选择 target head 前 K 行。

## 与 SpecForge / DeepSpec 的算法对齐

### 1. head 的 vocab 维度一致

三份实现都满足：

```text
base lm_head: hidden -> draft_vocab_size
markov_w1: target_vocab_size -> rank
markov_w2: rank -> draft_vocab_size
```

因此 backbone logits 和 Markov bias 可以直接相加，不需要先 scatter 到 target vocab。

### 2. d2t 边界放置正确

本 PR 的 `_dspark_sample_block()`：

```text
draft logits[K] + markov bias[K]
    -> argmax draft_id
    -> map_draft_to_target(draft_id)
    -> target_id
```

target id 随后同时用于：

- 写入 proposal buffer，交给 target verifier；
- 下一步 `markov_w1[target_id]`。

这与 SpecForge `DSparkDraftModel._sample_draft_tokens()` 和正确 DeepSpec `VanillaMarkov.sample_block_tokens()` 一致。

### 3. 派生 head 的行顺序正确

SpecForge/DeepSpec 通过 `target_head.weight[t2d]` 获得升序 target ids 对应的 K 行。本 PR 使用：

```text
kept_target_ids[i] = i + d2t[i]
```

再按 draft id 顺序组装 target rows，二者等价。

### 4. TP 派生逻辑在当前 Ascend 实现中成立

`eb042af60` 改为使用 target head 自身的 `comm_group` 做 all-reduce。Ascend 的 `AscendParallelLMHead` / `AscendVocabParallelEmbedding` 在构造时确实保存 `comm_group`、`tp_size` 和 `shard_indices`，所以：

- target TP ranks 各自填充拥有的 kept rows；
- target group all-reduce 后每个 rank 得到完整 K 行；
- draft head 的 `weight_loader` 再按自己的 TP metadata 本地取 shard；
- target TP4 + rank-local draft TP1 不需要两个 head 共用 group。

这一点与当前真实 Ascend layer contract 对得上，上一版 review 对 ownership/group 的疑问已不再作为 finding。

## 测试评价

新增 UT 对纯函数和 routing 覆盖较好：

- draft id 映射成 target id后再回喂 Markov W1；
- 不经过 target-vocab scatter；
- full-vocab fallback；
- 单 rank 与模拟 target TP 的 head row selection；
- target TP2 + draft TP1；
- quantized/FP8 target head 拒绝；
- 越界、重复及全零 mapping；
- ownership 默认值；
- 三种 lm_head routing。

仍缺少 loader 与多卡运行时的 integration coverage：

- “自带 head”测试只断言 head 被保留，没有校验 mapping；
- 没有真实 `Qwen3DSparkForCausalLM.load_weights()` 的最小 state-dict 测试；
- 没有 NPU 双卡 collective 或真实 checkpoint 的首 block logits 对齐。

## 本地验证

已执行：

```text
git diff --check dea3bc377..HEAD
结果：通过

PYTHONPYCACHEPREFIX=/tmp/dspark-review-pycache python3 -m py_compile \
  vllm_ascend/spec_decode/llm_base_proposer.py \
  tests/ut/spec_decode/test_dspark_vocab_mapping.py
结果：通过
```

未执行 pytest：当前 macOS Python 没有安装 `pytest` 和 `vllm`，且没有 Ascend NPU；因此未运行真实 loader、HCCL collective、ACL graph 或 acceptance 回归。

## 建议修复与验证顺序

1. **把 mapping 校验移出派生-head 专属分支。** reduced vocab 无论是否自带 head，都必须确认 mapping key 实际加载且内容合法。
2. **用 loader presence flag 替换 `d2t.any()`。** 缺 key 失败，合法的全零 offset mapping 成功。
3. **补最小真实 loader integration test。** 用已调整好的 config 和最小 state dict，断言共享 target embedding、派生 K 行 head并完成一个 Markov block。
4. **做 v1 NPU 首 block 对齐。** 保存 base logits、每步 Markov-corrected logits、draft id、映射后的 target id，与 `/specforge_sgl0514/DeepSpec/scripts/eval_real_dspark.py` 同输入逐项比较。
5. **最后比较 acceptance。** 先保证首 block token/logits 完全一致，再看 `acceptance_per_pos` 和端到端吞吐。

对用户当前已调整 config、checkpoint 不带 `lm_head`、`d2t` 为正常非全零映射的模型，本 PR 的核心 v1 推理路径已经具备上机验证条件。剩余 P1 位于“checkpoint 自带裁剪 head”的通用支持分支，不阻塞当前模型，但建议在合入前处理，避免实现与文档承诺不一致。
