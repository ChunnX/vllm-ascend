# DFlash/DSpark 在 Ascend 310P 上的 Qwen3-8B MRV1 eager 最终适配计划

> **文档权威性：** 本文是当前 Qwen3-8B/310P eager MVP 的唯一规范性实施与验收依据。
> 本文只覆盖 Qwen3-8B dense target、Qwen3-8B DFlash/DSpark drafter、Ascend 310P、
> Model Runner V1 和 eager 模式。更完整的背景研究见
> `docs/source/developer_guide/Design_Documents/dflash_dspark_310p_adaptation_analysis.md`；
> 该文档不定义本期范围、文件清单或完成标准。两份文档如有冲突，以本文为准。
>
> 本文的 `Phase 0` 仅指开发前硬件/ABI 门禁，不等同于完整 P0。完整 MVP P0 包括
> Phase 0、Task 1–4，以及 Task 5 和第 7 节规定的回归与验收。

**目标：** 在 Ascend 310P 上，用 FP16、TP=2、MRV1、eager 和 greedy sampling，
正确运行 Qwen3-8B + DFlash K=8，以及 Qwen3-8B + DSpark K=7。

**核心方案：**

1. 用 310P 可执行的 PyTorch NPU 输入展开替代 DFlash/DSpark 的 Triton kernel；
2. 用 `adn_custom_ops.adn_fused_infer_attention` 执行 drafter 的 non-causal paged attention；
3. 修复 Qwen3 DFlash/DSpark 五层 context KV 预写时的 310P out-of-place RoPE；
4. 只在精确识别到 DFlash/DSpark draft forward 时进入 ADN；所有 causal 路径与 pooling 保持不变，
   未支持的 non-causal 路径主动报错而不是静默回退。

---

## 0. 代码基线

本文所有行号引用都对以下基线核对过（2026-07-23 重核）：

| 仓库 | 分支 / commit |
| --- | --- |
| vllm-ascend | `dflash_dspark_310p_adapt_20260723`，基于 `e4c88fb0b` |
| vllm | `752a3a504` |
| Ascend_Ops | `914fa8d9d18e87d2b26031292537a136150eb413` |

**行号会随上游漂移，遇到对不上时以符号名为准，并回来更新本文。** 上一轮重核发现的实质变化
（不只是行号）记在这里，因为它们改变了实现方式：

| 变化 | 影响 |
| --- | --- |
| `patch_qwen3_dspark.py` 已删除，但**上游并没有完全吸收它**（见下） | Task 2.1 只剩一个 worker patch 要移；mask-token 回归测试改为盯 platform patch，不能删 |
| 类名由 `AscendDsparkProposer` 改为 **`AscendDSparkProposer`**（大写 S） | 全文与代码中统一 |
| DSpark 新增 `initialize_attn_backend`（`dspark_proposer.py:101`），按 KV cache group 组织 draft 层 | 引入多 group 机器（为 DeepSeek-V4 准备），本期必须显式断言只有 1 个 group |
| DSpark 的 `set_inputs_first_pass` 改为**按 group 循环调用 kernel**，用 per-group buffers（`dspark_proposer.py:230-264`） | Task 1.3 的 DSpark 调用点整段重写，不再与 DFlash 同形 |
| DSpark kernel 的 `block_size` 来自 `attn_group.kv_cache_spec.block_size`，DFlash 来自 `self.kernel_block_size`（`dflash_proposer.py:116`） | §3.1 权威来源表新增一行 |
| DSpark 额外设置 `cad.positions`、`cad.num_input_tokens` | 1.4 的 metadata 断言按方法分别写 |
| vLLM `qwen3_dflash.py` 的 `midlayer.` 重命名从 `WeightsMapper` 改为 `load_weights` 内联（`:628-629`） | 只是机制变化，不影响本期；但不要再拿它推断层数 |

drafter 层数以 checkpoint 的 `num_hidden_layers` 为准（模型侧在
`vllm/model_executor/models/qwen3_dflash.py:374` 按它建层），两个指定 checkpoint 都是 5。

### 0.1 DSpark mask token 的实际解析链（承重，不要简化）

指定的 DSpark checkpoint 把 `mask_token_id=151669` 放在 config **顶层**，而上游
`vllm/v1/spec_decode/llm_base_proposer.py:352-362` 只认三个位置：

```python
dflash_config = getattr(model_hf_config, "dflash_config", None)
if dflash_config and "mask_token_id" in dflash_config: ...
elif hasattr(model_hf_config, "pard_token"): ...
elif hasattr(model_hf_config, "ptd_token_id"): ...
else: raise ValueError(...)
```

**顶层 `mask_token_id` 不在其中。** 让它能工作的是 vllm-ascend 自己的 platform patch
`vllm_ascend/patch/platform/patch_speculative_config.py:148-149`：

```python
if getattr(draft_hf_config, "ptd_token_id", None) is None:
    draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)
```

它在 `vllm_ascend/patch/platform/__init__.py:44` 被加载。所以：

- **不需要**恢复已删除的 `patch_qwen3_dspark.py`；
- 但 `patch_speculative_config` 是本期承重依赖，**必须保留并有回归测试**。它一旦失效，DSpark 会在
  proposer 构造阶段直接 `ValueError`，而不是悄悄用错 token——属于会响的故障，但仍要测，因为
  这是本期唯一把顶层 `mask_token_id` 接进上游的环节。

---

## 1. 最终支持范围

| 维度 | 本期唯一支持值 |
| --- | --- |
| Target | `Qwen/Qwen3-8B`，dense `Qwen3ForCausalLM` |
| DFlash drafter | `z-lab/Qwen3-8B-DFlash-b16`，5 层，`num_speculative_tokens=8` |
| DSpark drafter | `deepseek-ai/dspark_qwen3_8b_block7`，5 层，`num_speculative_tokens=7` |
| 芯片 | Ascend 310P |
| Runner | Model Runner V1；启动时显式 `VLLM_USE_V2_MODEL_RUNNER=0` |
| 执行模式 | `enforce_eager=True`；不捕获、不回放 ACLGraph |
| dtype | target、draft 和 KV cache 全部为 FP16 |
| 并行 | TP 不锁死；每 rank head 布局满足结构约束即可（TP=2→16/4，TP=4→8/2）。本轮实机用 **TP=4**，`distributed_executor_backend="mp"` |
| Attention layout | query 为 TND；K/V 为 310P NZ paged cache |
| KV block size | vLLM 运行参数和实际 kernel block size 均为 128 |
| Sampling | greedy，`temperature=0` |
| Prefix cache | 本期关闭，`enable_prefix_caching=False` |

两个指定 checkpoint 的 `num_hidden_layers` 都是 5，不能按单层 drafter 设计：

- [Qwen3-8B-DFlash-b16 config](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/main/config.json)
- [dspark_qwen3_8b_block7 config](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/blob/main/config.json)

checkpoint 配置里的 DFlash/DSpark algorithm block 长度，不等同于 vLLM paged KV cache 的
`block_size=128`，实现和测试中必须区分两者。

### 明确不做

- Qwen3.6、hybrid attention、多 KV cache group 的通用适配；
- Qwen3 MoE、VL、MLA、GDN、pooling 或 cross-attention；
- MRV2；
- ACLGraph、TorchAir 图融合、`fuse_qknorm_rope`；
- BF16、量化、prefix caching、随机采样；
- DFlash/DSpark 其他 K 值、其他 drafter checkpoint；
- block size 64、head dim 256；
- 性能优化、零临时 tensor、通用 AscendC 输入展开算子。

本期不得为了“顺手支持”上述场景改 shared graph manager、Qwen3.6 runner plumbing 或
多 group block-size API。

---

## 2. 对原计划的 review 结论

| 原任务 | 处理 | Review 结论 |
| --- | --- | --- |
| Task 0 ADN NZ readback | 保留并重写 | 原脚本的 5D allocation + slice 不是 vLLM 生产分配；必须分别分配两个 rank-4 NZ cache，并增加 3 页和 causal discriminator 用例 |
| Task 1 无 Triton 输入展开 | 保留并加强 | 默认使用 NPU 向量化实现；CPU 只做 golden；必须先在真机验证 advanced indexing、`gather` 和原地赋值 |
| Task 2 proposer dispatch | 保留并加强 | 测试必须调用两个真实 `set_inputs_first_pass`，证明 310P 没有 launch Triton，而不只是直接测 helper |
| Task 3 DFlash RoPE | 重写 | 两个 drafter 均为 5 层；310P 必须逐层 RoPE，每层只处理 `num_ctx`，并接住 out-of-place 返回值 |
| Task 4 patch import | 与 RoPE 合并 | 只需在 310P 加载 `patch_qwen3_dflash`（`patch_qwen3_dspark` 已被删除）；测试 monkey-patch identity，不能用本来就为真的 `hasattr` |
| Task 5 block-size assertion | 删除独立任务 | Qwen3-8B 本期实际值固定为 128；输入展开从实际 `BlockTable` 读取，ADN 再校验 cache shape；不做通用多 group 修复 |
| Task 6 ADN adapter | 重写 | q-len 必须来自 310 builder 已生成的 raw host q-lens，不能使用 cumulative endpoints，也不能用 `max_query_len` 重造 |
| Task 7 attention routing | 收紧 | `causal=False` 单条件过宽；必须同时满足 draft context、method、ChunkedPrefill 和 non-causal |
| Task 8 graph fusion guard | 删除 | eager 模式不会运行该 pass，本期改它只会扩大范围 |
| Task 9 E2E | 重写 | 显式 `block_size=128`；使用精确 token IDs 构造页边界；token equality 之外还要防止全 reject |

原计划最关键的 P0 问题是五层 context RoPE。把 `[L * num_ctx]` 一次送入 310P RoPE，
再用容量 guard 报错，会让正常 Qwen3-8B 上下文直接失败；它不是有效的适配方案。

补充一条本轮新发现、三份文档此前都没有的 P0：**profile/dummy path 上 cos/sin 不按 drafter 的
positions 刷新**。两个 proposer 的 profile 分支形状相同
（`dflash_proposer.py:229-233`、`dspark_proposer.py:344-348`）：

```python
if is_profile:
    self.model.precompute_and_store_context_kv(context_states, context_positions)
    self.model(input_ids=..., positions=self._get_positions(num_query_total), inputs_embeds=None)
```

它不经过 `_runnable` / `_run_merged_draft`，而 `AscendRotaryEmbedding310._is_drafting_update_enabled`
只在 `_run_merged_draft` 里被打开。所以整个 profile 分支里 flag 都是 `False`，
`update_cos_sin` 不执行。

**故障形态是"用错值"而不是"崩"。** 全局 cos/sin 在此时都已就绪：`_cos` / `_sin` 由
`set_cos_and_sin` 在 `worker/model_runner_v1.py:477`（runner `__init__`）分配，
`_cos_sin_cache` 由 `_record_cos_sin_cache` 在 `AscendRotaryEmbedding.__init__`
（`ops/rotary_embedding.py:231`）于**模型构造期**记录，且 target 的 dummy run 已在
`model_runner_v1.py:2299` / `:3624` 刷新过 slice。因此 drafter profile 读到的是
**target 遗留的 slice**，非 `None`。

注意这里有两个 forward 都受影响：context KV 预写用 `context_positions`（长度
`num_input_tokens`），紧接着的 `self.model(...)` 用 query positions（长度
`num_query_total`），两者长度和取值都不同。只修前者不够。详见 Task 2.2。

---

## 3. 必须冻结的运行契约

### 3.1 DFlash/DSpark 输入展开契约

被替换的 Triton kernel 是
`vllm_ascend/ops/triton/spec_decode/utils.py::copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid`。
310P helper 必须逐字段等价：

1. context position 和 context slot mapping 对整个 scheduled segment 原样复制；
2. rejected tail 不物理压缩，只通过 `valid_ctx_end` 和 `effective_seq_len` 排除；
3. query position 从最后一个有效 context position 的下一位开始；
4. query slot 使用 `effective_seq_len + q_idx` 和 paged block table 计算；
5. 每请求第一个 query token 是 `next_token_ids[req]`，其余是 mask token；
6. DFlash 的 sampling indices 跳过 anchor；DSpark 从 anchor 开始；
7. DFlash 每请求 q-len 为 `K + 1 = 9`；DSpark 为 `K = 7`。

以下两个长度不能混淆：

| 字段 | 语义 |
| --- | --- |
| `query_start_loc` | 本轮送给 drafter 的 target token segment 边界，可能每请求不同 |
| `seq_lens` | 当前请求 KV cache 总长度，不等于本轮 scheduled segment 长度 |

真实 block size 必须从 draft KV group 的最终 BlockTable 读取：

```python
block_size = self.runner.input_batch.block_table[self.kv_cache_gid].block_size
```

本期该值必须为 128。不得把 backend 的候选列表、`kernel_block_sizes[0]`，或 checkpoint
algorithm block size 当成这个标量。

**权威来源声明。** 310P 上同时存在三个名字相近的字段，类型和语义都不同，不能互相直接比较：

| 字段 | 类型 | 来源 | 本期处理 |
| --- | --- | --- | --- |
| `block_table[gid].block_size` | 标量 | `worker/block_table.py:81`，"第一个能整除 physical_block_size 的候选" | **唯一权威**，断言 `== 128`；新 helper 与 ADN 都以它为准 |
| `proposer.kernel_block_size` | 标量 | `llm_base_proposer.py:336`，`get_supported_kernel_block_sizes()[0]`，未经 `head_size` 过滤 | DFlash 现在就用它（`dflash_proposer.py:116`）；断言 `== 128` |
| `attn_group.kv_cache_spec.block_size` | 标量 | DSpark 用它（`dspark_proposer.py:236`）；是 **allocation** block，不是 kernel block | 断言 `== 128`；两者本期同值是配置的结果，不是恒等式 |
| `runner.kernel_block_sizes[gid]` | **候选列表** | `_310p/model_runner_310p.py:987-996`，当前通常是 `[128, 64]` | 只检查首选为 128 或列表包含 128，**不能与标量比相等** |

第三行是本轮重核新发现的：DFlash 和 DSpark 在当前基线上用的是**两个不同来源**。`kv_cache_spec.block_size`
是 KV cache 的分配块大小，`BlockTable.block_size` 是 kernel 实际使用的块大小——只有当分配块恰好
被 backend 支持时两者才相等。本期 E2E 强制 `block_size=128`、310P backend 支持 128，所以三者同值；
这是**配置带来的巧合**，guard 的作用就是在配置变化时立刻响，而不是让 slot mapping 悄悄错位。

关于 `proposer.kernel_block_size` 的现状要说准确：它**目前确实参与 slot 计算**——两个 proposer
的 `set_inputs_first_pass` 都把 `block_size=self.kernel_block_size` 传给 Triton kernel。
Task 1 完成、310P 切到从 BlockTable 读之后，它才不再参与**新 helper** 的计算；非 310P 的
Triton 分支仍然用它。所以本期不修改它的赋值，但要用 guard 把"它恰好也等于 128"这个前提钉住。

（`llm_base_proposer.py:1662` 那处 `if self.has_gdn: block_size = self.kernel_block_size` 是另一条
无关路径，Qwen3-8B 无 GDN 不会走到，不作为本期依据。）

### 3.2 ADN ABI 契约

`adn_fused_infer_attention` 在本期的固定调用契约如下：

| 参数 | 本期值或语义 |
| --- | --- |
| `query` | FP16 `[T, Nq, 128]`，TND |
| `key` / `value` | FP16 rank-4 NZ cache `[P, Nkv*128/16, 128, 16]` |
| `attn_mask` | `None`，对应算子 `NO_MASK`，即 non-causal full attention |
| `actual_seq_lengths_q` | Python `list[int]`，每请求 raw q-len `[9, ...]` 或 `[7, ...]` |
| `actual_seq_lengths_kv` | Python `list[int]`，每请求包含当前 query 的总 KV 长度 |
| `block_table` | NPU int32 rank-2 physical page IDs |
| `num_heads` / `num_key_value_heads` | 每 rank local 值；TP=2→16/4，TP=4→8/2，由结构约束校验 |
| `block_size` | 128 |
| `input_layout` | `"TND"` |
| `scale_value` | `128 ** -0.5` |
| `inner_precise` | 2 |
| `force_call` | `False` |
| 返回值 | 新 Tensor；无 `out=`，需要 copy 到 vLLM 的 output buffer |

`attn_mask=None` 在当前 ADN host tiling 中映射为 `NO_MASK`，kernel 也只在
`maskType != 0` 时加载并加 mask。因此本期不构造伪 mask。

### 3.3 raw q-len 契约

通用 `AscendAttentionMetadataBuilder` 会把 `actual_seq_lengths_q` 写成 cumulative endpoints，
不能直接传给 ADN。310P builder 已经在 forward 外用 CPU endpoints 做差，并把 raw q-len
挂在 metadata 上：

```python
raw_q_lens = get_query_lens_cpu(attn_metadata)
q_lens = raw_q_lens.tolist()
```

该 tensor 位于 host/pinned memory，`.tolist()` 不产生 NPU D2H。缺失时必须报错，不能用
`[max_query_len] * batch_size` 猜测。

### 3.4 Qwen3 context RoPE 契约

`DFlashQwen3Model` 按 draft config 创建 5 层；`Qwen3DSparkModel` 继承该实现。310P RoPE：

- 是 out-of-place，必须使用返回的新 tensor；
- 全局 cos/sin slice 容量按 `max_num_batched_tokens` 分配；
- 正常 draft forward 由 `_run_merged_draft` 打开更新 flag；profile/dummy path 绕过该 wrapper，
  所以必须由 `_profile_rope_context` 把 context KV 和随后 query 的两个 forward 整体包住；
- flag 打开后，每层 rotary 根据自己收到的 positions 刷新全局 slice；context KV helper
  **不得**再手工准备 cos/sin，否则只能覆盖第一个 forward，并与真实 draft 路径形成两套协议；
- 310P 每层单独旋转 `[num_ctx, Nkv, D]`，不能旋转 `[L * num_ctx, ...]`。

非 310P 路径继续保留 fused `[L * num_ctx]` RoPE，避免回退 910 性能，但也应接住返回值。

---

## 4. 最终文件范围

### 生产代码

| 文件 | 变更 |
| --- | --- |
| `vllm_ascend/_310p/spec_decode/parallel_drafting_inputs.py` | 新增 PyTorch NPU 输入展开 helper |
| `vllm_ascend/spec_decode/dflash_proposer.py` | 新增共享 dispatch seam 与 `_profile_rope_context`；310P lazy import helper；使用实际 BlockTable block size；`is_profile` 分支加 flag 包装 |
| `vllm_ascend/spec_decode/dspark_proposer.py` | 复用 DFlash dispatch 与 `_profile_rope_context`；只传 DSpark 参数；`is_profile` 分支加 flag 包装；删除重复 Triton 调用/import |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | 310P 五层逐层 context RoPE、接住 out-of-place 返回值 |
| `vllm_ascend/patch/worker/__init__.py` | 310P 也加载 `patch_qwen3_dflash`（DSpark 无需 patch，见 §0） |
| `vllm_ascend/_310p/attention/adn_fused_infer_attention.py` | 新增 ADN lazy loader、scope/capability validation 和 forward adapter |
| `vllm_ascend/_310p/attention/attention_v1.py` | 精确 route DFlash/DSpark draft non-causal attention 到 ADN |

### 测试与证据

| 文件 | 用途 |
| --- | --- |
| `tests/ut/_310p/spec_decode/test_parallel_drafting_inputs_310p.py` | CPU golden、NPU helper 语义、真实 proposer dispatch |
| `tests/ut/_310p/test_qwen3_parallel_drafting_patches_310p.py` | 五层 RoPE、profile path、patch identity、DSpark mask token |
| `tests/ut/_310p/attention/test_adn_fused_infer_attention_310p.py` | adapter ABI、长度、guard、output copy |
| `tests/ut/_310p/attention/test_parallel_draft_routing_310p.py` | 精确路由和 fail-loud 行为 |
| `tests/e2e/_310p/spec_decode/smoke_parallel_drafting_inputs.py` | 310P NPU input-op 手工门禁 |
| `tests/e2e/_310p/adn/smoke_adn_nz_readback.py` | 生产格式 NZ writer-to-ADN 手工门禁 |
| `tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_eager_310p.py` | TP=2 Qwen3-8B eager 正确性与 acceptance |

两个 `smoke_*.py` 默认是明确的手工硬件门禁，不假装会被 pytest CI 自动收集。如果确认官方
310P CI 镜像已经预装对应 ADN custom OPP/PTA，则把 NZ readback 提升为
`tests/e2e/pull_request/one_card/_310p/test_adn_nz_readback.py`，并加入 CI 时间配置。

### 本期不得修改

- `vllm_ascend/compilation/graph_fusion_pass_manager.py`；
- Qwen3.6 patch 和 runner；
- MRV2 DFlash speculator；
- shared multi-group block-size plumbing；
- ADN kernel/tiling 源码，除非 Phase 0 证明当前 operator ABI 本身有缺陷。

---

## 5. 实施任务

### Phase 0：硬件与 ABI 门禁

这是一组 hard gate。任何一项不通过，都不能靠 fallback 到 causal splitfuse 继续集成。

#### 0.1 记录可复现版本矩阵

- [ ] 记录 `vllm`、`vllm-ascend`、`Ascend_Ops` 三个仓库的完整 commit SHA；
- [ ] 记录 Python、PyTorch、torch-npu、TorchAir、CANN、driver/firmware 和镜像 digest；
- [ ] 记录 custom OPP 与 PTA wheel 的 build/version 信息；
- [ ] 解析并记录 target、DFlash、DSpark 三个 Hugging Face snapshot revision；
- [ ] 后续 E2E 显式传 revision，不依赖可移动的 `main`。

revision 必须解析成不可移动的 commit SHA，不能停留在符号名：

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
for repo in ['Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16', 'deepseek-ai/dspark_qwen3_8b_block7']:
    print(repo, api.model_info(repo).sha)
"
```

把三个 SHA 直接写进 Task 4 测试文件顶部的 `TARGET_REVISION` / `DFLASH_REVISION` /
`DSPARK_REVISION` 常量。**这一步没做完之前 E2E 不算可执行**——普通 UT 不受影响，可以先写先跑，
但 Task 4 在常量仍是占位符时不得标记为完成。

完成标准是“记录已验证版本矩阵”，不是笼统声称“镜像已固定”。

#### 0.2 先验证 ADN 自身的目标 case

Ascend_Ops 已把 ATK 用例替换成直接可跑的 pytest 风格脚本，`atk_test/` 目录不再存在。
在已安装 ADN 的 310P 环境执行：

```bash
python Ascend_Ops/tests/test_adn_fia.py
```

它覆盖 4 组 layout × head_dim × block_size，每组 10 个随机用例，`MAX_SEQ=8192`、
`MAX_BATCH=32`，MHA 与 GQA 都在内；不传 `attn_mask`，走的正是本期用的 no-mask 路径。
（`tests/test_adn_fia_tnd_compress_mask.py` 测的是 compressed-mask 即 causal 路径，本期不用。）

- [ ] 该脚本通过，其中 TND / head_dim=128 / block_size=128 这组是本期的目标组合。

**判据从这里取，不要另立一套。** `test_adn_fia.py:172` 定义：

```python
atol = 1e-4
...
passed = diff_flatten_mean <= atol
```

即**按平均绝对误差判定，阈值 1e-4**；最大误差只打印不参与判定。这个选择是合理的：
fp16 在 O(1) 输出上单元素误差本就接近格式分辨率（~1e-3），拿 max 卡会变成看运气。
Phase 0.4 的 NZ readback 沿用同一判据。

**scale 已与生产一致。** `test_adn_fia.py:173` 用 `scale = head_dim ** -0.5`，
和 vLLM 运行时的 `self.scale` 相同。（旧 `atk_test/fia_common.py` 曾用 `1/head_dim`，
自洽但验的不是生产数值区间；该文件已随 ATK 一并移除，这条历史坑不再存在。）

**ADN 的 Python ABI 已精简。** 新签名移除了全部 quant/dequant/antiquant 参数和
`kv_padding_size`，保留：

```text
query, key, value, attn_mask, actual_seq_lengths_q, actual_seq_lengths_kv,
block_table, num_heads, scale_value, input_layout, num_key_value_heads,
block_size, inner_precise, force_call
```

本期 adapter 全程只用关键字实参且从未传过被移除的那些参数，**无需改动**。

#### 0.3 验证 PyTorch NPU 输入展开所需算子

`smoke_parallel_drafting_inputs.py` 必须在 310P 上执行与正式 helper 相同的操作：

- slice `copy_`；
- int32/int64 conversion；
- 用 `valid_ctx_end - 1` advanced-index `target_positions`；
- 对 rank-2 block table 做 `gather`；
- `arange`、broadcast、flatten/view；
- fill 和列赋值；
- 把结果与 CPU 逐行 golden 比较。

用例至少包括：

- DFlash K=8、q-len=9；
- DSpark K=7、q-len=7；
- `ctx_lens=[1, 4, 2]`、`seq_lens=[257, 134, 66]`；
- `rejected=[0, 3, 1]`，并保证每请求仍至少有一个有效 context token；
- query slot 跨 127/128/129 边界；
- 非连续、乱序 physical block IDs。

如果某个 exact op 在 310P 不支持，停止 NPU helper 集成。优先实现小型 AscendC helper；
CPU/NumPy + 每步 D2H/H2D 只能作为显式 debug fallback，不能静默进入最终热路径。

#### 0.4 验证 vLLM NZ writer 与 ADN reader 直通

原计划的 5D `zeros -> npu_format_cast -> cache[0]/cache[1]` 必须改掉。测试应复刻
`_310p/model_runner_310p.py`：

```python
cache_shape = AscendAttentionBackend310.get_kv_cache_shape(
    num_blocks,
    128,
    4,
    128,
)[1:]

key_cache = torch_npu.empty_with_format(
    size=cache_shape,
    dtype=torch.float16,
    device="npu",
    acl_format=ACL_FORMAT_FRACTAL_NZ,
)
value_cache = torch_npu.empty_with_format(
    size=cache_shape,
    dtype=torch.float16,
    device="npu",
    acl_format=ACL_FORMAT_FRACTAL_NZ,
)
```

然后：

1. 用 `DeviceOperator.reshape_and_cache` 写 cache，保证走 vLLM 当前 310P writer dispatch；
2. 用 CPU block-table mirror 计算 slot，避免循环 `int(npu_tensor)` 产生同步；
3. 用 ADN 读取同一对 rank-4 NZ cache；
4. 分别测试 q-len 9 和 7；
5. ragged KV 长度覆盖 1、2、3 页，例如 `[65, 133, 257]`；
6. physical page IDs 乱序且不连续；
7. 对比 FP32 full-attention golden；
8. 增加 future-token-dominance case，使 causal 与 non-causal 输出明显不同；
9. **scale 用 `128 ** -0.5`**，kernel 与 golden 一致（与 `Ascend_Ops/tests/test_adn_fia.py` 相同）；
10. 判据沿用 0.2 的 **mean_abs <= 1e-4**，同时打印 max/mean absolute error；不得另立一套或拍脑袋固定
   `max_abs < 5e-3`。

Phase 0 完成标准：两个 smoke 均在目标 310P 环境通过，且版本、命令和误差结果进入 PR 证据。

---

### Task 1：实现并接入无 Triton 输入展开

#### 1.1 先写 CPU 逐行 golden UT

golden 是原 Triton kernel 的逐行转写，不是"另一种实现"——这样任何差异都能定位到具体哪一行。

写在 `tests/ut/_310p/spec_decode/test_parallel_drafting_inputs_310p.py`：

```python
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.spec_decode.parallel_drafting_inputs import expand_parallel_drafting_inputs

MASK_ID = 151666
BLOCK_SIZE = 128
GUARD = -99  # sentinel written past the used region to catch overruns


def reference_expand(
    *,
    next_token_ids,
    target_positions,
    context_slot_mapping,
    out_input_ids,
    out_context_positions,
    out_query_positions,
    out_context_slot_mapping,
    out_query_slot_mapping,
    out_token_indices,
    block_table,
    query_start_loc,
    seq_lens,
    num_rejected_tokens,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    batch_size,
    sample_from_anchor,
):
    """Line-by-line transcription of
    ops/triton/spec_decode/utils.py::copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid.
    Golden only -- never optimize this into a vectorized form.
    """
    for req_idx in range(batch_size):
        ctx_start = int(query_start_loc[req_idx])
        ctx_end = int(query_start_loc[req_idx + 1])

        for j in range(ctx_start, ctx_end):
            out_context_positions[j] = int(target_positions[j])
            out_context_slot_mapping[j] = int(context_slot_mapping[j])

        num_rejected = int(num_rejected_tokens[req_idx]) if num_rejected_tokens is not None else 0
        valid_ctx_end = ctx_end - num_rejected
        effective_seq_len = int(seq_lens[req_idx]) - num_rejected
        last_pos = int(target_positions[valid_ctx_end - 1])

        for q_idx in range(num_query_per_req):
            out_idx = req_idx * num_query_per_req + q_idx
            out_query_positions[out_idx] = last_pos + 1 + q_idx

            cache_pos = effective_seq_len + q_idx
            block_id = int(block_table[req_idx, cache_pos // block_size])
            out_query_slot_mapping[out_idx] = block_id * block_size + cache_pos % block_size

            if q_idx == 0:
                out_input_ids[out_idx] = int(next_token_ids[req_idx])
            else:
                out_input_ids[out_idx] = parallel_drafting_token_id

            if sample_from_anchor:
                out_token_indices[req_idx * num_speculative_tokens + q_idx] = out_idx
            elif q_idx > 0:
                out_token_indices[req_idx * num_speculative_tokens + q_idx - 1] = out_idx


def make_case(*, ctx_lens, seq_lens, rejected, num_query_per_req, num_spec):
    """Build one input set.

    ctx_lens is this round's scheduled segment per request; seq_lens is the total
    KV length per request. They are NOT the same thing -- conflating them is the
    single easiest way to write a test that passes against a broken helper.
    Positions are absolute: a request whose total KV is 257 and which scheduled
    4 tokens this round carries positions [253, 254, 255, 256].
    """
    batch_size = len(ctx_lens)
    assert len(seq_lens) == batch_size
    for n_ctx, n_seq in zip(ctx_lens, seq_lens):
        assert n_seq >= n_ctx, "total KV length cannot be shorter than this round's segment"
    total = sum(ctx_lens)

    qsl = torch.zeros(batch_size + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(ctx_lens, dtype=torch.int32).cumsum(0)

    positions = torch.cat(
        [
            torch.arange(n_seq - n_ctx, n_seq, dtype=torch.int32)
            for n_ctx, n_seq in zip(ctx_lens, seq_lens)
        ]
    )
    ctx_slots = torch.arange(total, dtype=torch.int32) * 3 + 7

    max_blocks = (max(seq_lens) + num_query_per_req) // BLOCK_SIZE + 2
    # Non-contiguous, descending physical page ids so a bug that ignores the
    # block table cannot accidentally produce the right slot.
    block_table = (
        torch.arange(batch_size * max_blocks, dtype=torch.int32).flip(0).reshape(batch_size, max_blocks)
    )

    return dict(
        next_token_ids=torch.arange(batch_size, dtype=torch.int32) + 1000,
        target_positions=positions,
        context_slot_mapping=ctx_slots,
        block_table=block_table,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_rejected_tokens=(torch.tensor(rejected, dtype=torch.int32) if rejected is not None else None),
        parallel_drafting_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_spec,
        total_input_tokens=total,
        batch_size=batch_size,
    )


def run_both(case, sample_from_anchor):
    b, q, k = case["batch_size"], case["num_query_per_req"], case["num_speculative_tokens"]
    total = case["total_input_tokens"]
    slack = 8  # extra room so an overrun lands on GUARD instead of out of bounds

    def fresh():
        return dict(
            out_input_ids=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_context_positions=torch.full((total + slack,), GUARD, dtype=torch.int32),
            out_query_positions=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_context_slot_mapping=torch.full((total + slack,), GUARD, dtype=torch.int32),
            out_query_slot_mapping=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_token_indices=torch.full((b * k + slack,), GUARD, dtype=torch.int32),
        )

    got, want = fresh(), fresh()
    expand_parallel_drafting_inputs(**case, **got, sample_from_anchor=sample_from_anchor)
    reference_expand(**case, **want, sample_from_anchor=sample_from_anchor)
    return got, want, slack


class TestExpandParallelDraftingInputs(TestBase):
    def _assert_matches(self, case, sample_from_anchor):
        got, want, slack = run_both(case, sample_from_anchor)
        for name in want:
            torch.testing.assert_close(got[name], want[name], msg=f"{name} mismatch")
            tail = got[name][-slack:]
            self.assertTrue(
                bool((tail == GUARD).all()),
                f"{name}: helper wrote past its used region (tail={tail.tolist()})",
            )

    def test_dflash_single_request(self):
        self._assert_matches(
            make_case(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=9, num_spec=8),
            sample_from_anchor=False,
        )

    def test_dspark_single_request(self):
        self._assert_matches(
            make_case(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=7, num_spec=7),
            sample_from_anchor=True,
        )

    def test_ragged_segment_with_distinct_seq_lens(self):
        case = make_case(
            ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=None, num_query_per_req=9, num_spec=8
        )
        self._assert_matches(case, sample_from_anchor=False)

    def test_rejected_tail_excluded_from_query_but_kept_in_context(self):
        # Every request keeps at least one valid context token after rejection.
        case = make_case(
            ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1], num_query_per_req=9, num_spec=8
        )
        self._assert_matches(case, sample_from_anchor=False)
        self._assert_matches(
            make_case(
                ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1],
                num_query_per_req=7, num_spec=7,
            ),
            sample_from_anchor=True,
        )

    def test_query_slots_cross_page_boundary(self):
        # effective_seq_len lands at 127/128/129 so the K+1 query slots straddle
        # a kernel page and must follow the block table into a new physical page.
        for seq_len in (127, 128, 129):
            self._assert_matches(
                make_case(
                    ctx_lens=[2], seq_lens=[seq_len], rejected=None, num_query_per_req=9, num_spec=8
                ),
                sample_from_anchor=False,
            )

    def test_context_buffer_untouched_beyond_scheduled_segment(self):
        case = make_case(
            ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=None, num_query_per_req=9, num_spec=8
        )
        got, _, slack = run_both(case, sample_from_anchor=False)
        total = case["total_input_tokens"]
        self.assertTrue(bool((got["out_context_positions"][total:] == GUARD).all()))
        self.assertTrue(bool((got["out_context_slot_mapping"][total:] == GUARD).all()))
```

不要添加 block size 64 用例；它不属于当前 Qwen3-8B 范围。

验证它先失败：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/spec_decode/test_parallel_drafting_inputs_310p.py
```

Expected: FAIL，`ModuleNotFoundError: No module named 'vllm_ascend._310p.spec_decode.parallel_drafting_inputs'`

#### 1.2 实现 310P PyTorch NPU helper

实现前提是 Phase 0.3 已证明 exact ops 在 310P 可执行。

`vllm_ascend/_310p/spec_decode/parallel_drafting_inputs.py`：

```python
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import torch


def expand_parallel_drafting_inputs(
    *,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    out_input_ids: torch.Tensor,
    out_context_positions: torch.Tensor,
    out_query_positions: torch.Tensor,
    out_context_slot_mapping: torch.Tensor,
    out_query_slot_mapping: torch.Tensor,
    out_token_indices: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_rejected_tokens: torch.Tensor | None,
    parallel_drafting_token_id: int,
    block_size: int,
    num_query_per_req: int,
    num_speculative_tokens: int,
    total_input_tokens: int,
    batch_size: int,
    sample_from_anchor: bool = False,
) -> None:
    """Triton-free DFlash/DSpark input expansion for 310P.

    Field-for-field equivalent of
    ``vllm_ascend.ops.triton.spec_decode.utils.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid``.
    Writes in place into the caller's persistent buffers and returns nothing.

    ``sample_from_anchor`` selects DSpark's sampling layout (take K positions
    starting at the anchor) instead of DFlash's (skip the anchor, take the next K).

    No ``.cpu()`` / ``.item()`` / ``.tolist()`` anywhere: every value stays on
    device so this adds no synchronization to the draft step.
    """
    device = target_positions.device
    b, q, k = batch_size, num_query_per_req, num_speculative_tokens

    # The scheduled context is copied verbatim over the whole ragged range. The
    # rejected tail deliberately stays in place: it is excluded downstream by the
    # shortened seq_lens, and the new query slots overwrite those cache positions.
    # ponytail: assumes query_start_loc[batch_size] == total_input_tokens, which
    # every caller guarantees (total_input_tokens is target_token_ids.shape[0]).
    # Checking it here would cost a D2H on the draft hot path; the golden UT
    # covers it instead.
    out_context_positions[:total_input_tokens] = target_positions[:total_input_tokens]
    out_context_slot_mapping[:total_input_tokens] = context_slot_mapping[:total_input_tokens]

    ctx_end = query_start_loc[1 : b + 1].to(torch.int64)
    if num_rejected_tokens is None:
        num_rejected = torch.zeros(b, dtype=torch.int64, device=device)
    else:
        num_rejected = num_rejected_tokens[:b].to(torch.int64)

    # Absolute position of the last still-valid context token per request.
    last_pos = target_positions[ctx_end - num_rejected - 1].to(torch.int64)
    # Total KV length after dropping the rejected tail -- this is where this
    # round's query block starts in the cache, and it is derived from seq_lens
    # (total KV), not from the scheduled segment length.
    effective_seq_len = seq_lens[:b].to(torch.int64) - num_rejected

    q_arange = torch.arange(q, dtype=torch.int64, device=device)
    out_query_positions[: b * q] = (last_pos[:, None] + 1 + q_arange).flatten()

    # int64 throughout so `block_id * block_size` cannot overflow before it is
    # narrowed back into the int32 destination buffer.
    cache_pos = effective_seq_len[:, None] + q_arange
    block_id = block_table[:b].gather(1, cache_pos // block_size).to(torch.int64)
    out_query_slot_mapping[: b * q] = (block_id * block_size + cache_pos % block_size).flatten()

    ids = out_input_ids[: b * q].view(b, q)
    ids.fill_(parallel_drafting_token_id)
    ids[:, 0] = next_token_ids[:b]

    anchor_offset = 0 if sample_from_anchor else 1
    req_base = torch.arange(b, dtype=torch.int64, device=device)[:, None] * q
    k_arange = torch.arange(k, dtype=torch.int64, device=device)
    out_token_indices[: b * k] = (req_base + k_arange + anchor_offset).flatten()
```

写完再跑一次 1.1 的命令，应当全绿。允许 eager MVP 中创建小型 `arange` 临时 tensor，性能优化后置。

#### 1.3 在 proposer 打开最小 dispatch seam

在 DFlash proposer 提供共享方法，DSpark 继承：

```text
310P   -> lazy import _310p helper -> NPU PyTorch expansion
非310P -> 原 Triton launcher
```

**block size 的读取和 guard 必须放在 310P 分支内部**，不能放在 dispatch 外面。否则非 310P 的
Triton launcher 会收到 selected BlockTable size 而不是原来的 `self.kernel_block_size`，
A2/A3 上换个 cache block 配置就会被那条 `!= 128` 的 guard 直接打挂——这违反了本期
"非 310P 一行不改"的约束。

全期只认一个常量。放在 `vllm_ascend/_310p/spec_decode/parallel_drafting_inputs.py` 顶部：

```python
# The only kernel/allocation block size this scope supports. Everything that
# claims to be "the block size" is checked against this constant rather than
# against each other -- cross-checking three sources only proves they agree,
# not that they equal what ADN was validated at.
ADN_BLOCK_SIZE = 128


def resolve_310p_block_size(proposer) -> int:
    """Read the selected kernel block size and pin it to ADN_BLOCK_SIZE.

    310P only. Never call this on the Triton path: that path must keep using
    proposer.kernel_block_size so A2/A3 behaviour is bit-for-bit unchanged.
    """
    gid = proposer.kv_cache_gid
    selected = proposer.runner.input_batch.block_table[gid].block_size
    if selected != ADN_BLOCK_SIZE:
        raise RuntimeError(
            f"this scope only covers kernel block size {ADN_BLOCK_SIZE}, but KV cache "
            f"group {gid} selected {selected}; re-scope before changing anything"
        )
    if proposer.kernel_block_size != ADN_BLOCK_SIZE:
        raise RuntimeError(
            f"proposer.kernel_block_size is {proposer.kernel_block_size}, expected "
            f"{ADN_BLOCK_SIZE}. See the authority table in section 3.1 -- this needs a "
            f"scope decision, not a local fix."
        )
    # runner.kernel_block_sizes[gid] is a *candidate list* (typically [128, 64]),
    # never a scalar, so check membership rather than equality.
    candidates = proposer.runner.kernel_block_sizes[gid]
    if ADN_BLOCK_SIZE not in candidates:
        raise RuntimeError(
            f"{ADN_BLOCK_SIZE} is not in the runner's candidate list {candidates} "
            f"for KV cache group {gid}"
        )
    return ADN_BLOCK_SIZE
```

在 `AscendDflashProposer` 上加共享 dispatch 方法（`AscendDSparkProposer` 继承，不重复定义）。
参数全部显式列出，不用 `**kwargs` 转发——转发写法依赖求值顺序，且让两边的参数名漂移不会被发现：

```python
    def _expand_drafting_inputs(
        self,
        *,
        next_token_ids,
        target_positions,
        context_slot_mapping,
        out_input_ids,
        out_context_positions,
        out_query_positions,
        out_context_slot_mapping,
        out_query_slot_mapping,
        out_token_indices,
        block_table,
        query_start_loc,
        seq_lens,
        num_rejected_tokens,
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_tokens,
        total_input_tokens,
        batch_size,
        sample_from_anchor,
    ) -> None:
        """Expand DFlash/DSpark drafting inputs.

        310P has no Triton and uses the vectorized PyTorch equivalent; every other
        device keeps the original kernel launch. Both write in place into the same
        persistent buffers.
        """
        if is_310p():
            from vllm_ascend._310p.spec_decode.parallel_drafting_inputs import (
                expand_parallel_drafting_inputs,
                resolve_310p_block_size,
            )

            # 310P reads the selected BlockTable size and pins it to ADN_BLOCK_SIZE.
            # The caller's `block_size` argument is deliberately ignored here and
            # left untouched for the Triton path below, so A2/A3 keeps its exact
            # current source (proposer.kernel_block_size / kv_cache_spec.block_size).
            expand_parallel_drafting_inputs(
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                context_slot_mapping=context_slot_mapping,
                out_input_ids=out_input_ids,
                out_context_positions=out_context_positions,
                out_query_positions=out_query_positions,
                out_context_slot_mapping=out_context_slot_mapping,
                out_query_slot_mapping=out_query_slot_mapping,
                out_token_indices=out_token_indices,
                block_table=block_table,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                num_rejected_tokens=num_rejected_tokens,
                parallel_drafting_token_id=parallel_drafting_token_id,
                block_size=resolve_310p_block_size(self),
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=num_speculative_tokens,
                total_input_tokens=total_input_tokens,
                batch_size=batch_size,
                sample_from_anchor=sample_from_anchor,
            )
            return

        copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            context_slot_mapping_ptr=context_slot_mapping,
            out_input_ids_ptr=out_input_ids,
            out_context_positions_ptr=out_context_positions,
            out_query_positions_ptr=out_query_positions,
            out_context_slot_mapping_ptr=out_context_slot_mapping,
            out_query_slot_mapping_ptr=out_query_slot_mapping,
            out_token_indices_ptr=out_token_indices,
            block_table_ptr=block_table,
            block_table_stride=block_table.stride(0),
            query_start_loc_ptr=query_start_loc,
            seq_lens_ptr=seq_lens,
            num_rejected_tokens_ptr=(num_rejected_tokens if num_rejected_tokens is not None else 0),
            parallel_drafting_token_id=parallel_drafting_token_id,
            block_size=block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=num_speculative_tokens,
            total_input_tokens=total_input_tokens,
            batch_size=batch_size,
            HAS_NUM_REJECTED=num_rejected_tokens is not None,
            SAMPLE_FROM_ANCHOR=sample_from_anchor,
        )
```

`dflash_proposer.py` 顶部加 `from vllm_ascend.utils import is_310p`。

**DFlash 调用点**（替换 `dflash_proposer.py` 里原来的 kernel launch）：

```python
        self._expand_drafting_inputs(
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            context_slot_mapping=cad.slot_mapping,
            out_input_ids=self.input_ids,
            out_context_positions=self._context_positions_buffer,
            out_query_positions=self.positions,
            out_context_slot_mapping=self._context_slot_mapping_buffers,
            out_query_slot_mapping=self._slot_mapping_buffer,
            out_token_indices=token_indices_to_sample,
            block_table=cad.block_table_tensor,
            query_start_loc=cad.query_start_loc,
            seq_lens=cad.seq_lens,
            num_rejected_tokens=(num_rejected_tokens_gpu if has_num_rejected else None),
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            # Unchanged from today. The dispatch swaps in the selected BlockTable
            # size on 310P only; the Triton path must keep this exact source.
            block_size=self.kernel_block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            batch_size=batch_size,
            sample_from_anchor=False,
        )
```

**DSpark 调用点**（替换 `dspark_proposer.py:237-264` 里原来的 kernel launch）。

⚠️ **DSpark 与 DFlash 已经不同形了**，不要照抄上面那段。当前基线上 DSpark 是**按 KV cache group
循环**调用 kernel，每个 group 有自己的 block table、slot mapping 和 context slot mapping buffer，
而且 `block_size` 参数取自 `attn_group.kv_cache_spec.block_size`（不是 `self.kernel_block_size`）：

```python
        draft_attn_groups = getattr(self, "draft_attn_groups", [])
        # Scope guard: Qwen3-8B DSpark has 5 identical attention layers, which vLLM
        # merges into a single KV cache group. The per-group machinery below exists
        # for DeepSeek-V4 DSpark; this plan does not cover multi-group draft caches,
        # so refuse rather than silently expanding only the first one.
        active_gids = [
            g.kv_cache_group_id
            for g in draft_attn_groups
            if self._per_group_block_table_buffers.get(g.kv_cache_group_id) is not None
        ]
        if len(active_gids) != 1:
            raise RuntimeError(
                f"this plan only covers a single draft KV cache group, got {active_gids}. "
                f"Multi-group DSpark (DeepSeek-V4) is explicitly out of scope."
            )

        for attn_group in draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers.get(gid)
            if gid_block_table is None:
                continue
            # Unchanged from today, and deliberately NOT guarded here: this line
            # runs on every device, so pinning it to 128 would break A2/A3 configs
            # that legitimately use another allocation block size. The 310P guard
            # lives in resolve_310p_block_size, inside the dispatch.
            kv_block_size = int(attn_group.kv_cache_spec.block_size)
            self._expand_drafting_inputs(
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                context_slot_mapping=self._per_group_slot_mappings[gid],
                out_input_ids=self.input_ids,
                out_context_positions=self._context_positions_buffer,
                out_query_positions=self.positions,
                out_context_slot_mapping=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices=token_indices_to_sample,
                block_table=gid_block_table,
                query_start_loc=cad.query_start_loc,
                seq_lens=cad.seq_lens,
                num_rejected_tokens=(num_rejected_tokens_gpu if has_num_rejected else None),
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kv_block_size,
                num_query_per_req=block_size,
                num_speculative_tokens=block_size,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                sample_from_anchor=True,
            )
```

这里有**两个都叫 block size 的东西**，写错不会报错只会让 slot mapping 错位：

| 变量 | 含义 | 值 |
| --- | --- | --- |
| `block_size`（局部变量，`dspark_proposer.py:208` 的 `self.num_speculative_tokens`） | DSpark 的**算法** block，即每请求 query 数 K | 7 |
| `kv_block_size` | **paged KV cache** 的 block | 128 |

kernel 的 `num_query_per_req` / `num_speculative_tokens` 收前者，`block_size` 收后者。

DSpark 与 DFlash 的语义差别仍是那两条：每请求查 K 个位置（不是 K+1），且
`sample_from_anchor=True` 从 anchor 起取 K 个。不要把 DSpark 简化成"query 数不同的 DFlash"。

同时删掉 `dspark_proposer.py:15` 那行不再使用的 Triton import。

metadata 更新继续保持现有语义：

- `cad.query_start_loc` 改成均匀 9 或 7 的 endpoints；
- `cad.seq_lens = effective_seq_lens + q_len`；
- `cad.causal = False`；
- `cad.attn_mask = None`；
- `cad.attn_state = ChunkedPrefill`。

#### 1.4 测真实 caller，不测假 dispatch

UT 分别调用两个真实 `set_inputs_first_pass`，而不是直接测 helper——只测 helper 证明不了
proposer 真的改用了它。

不要用"新增 import 前后是否出现 triton module"作为验收；placeholder 可能早已被 package import。
真正的要求是 **310P 不 launch Triton**，所以把 launcher 换成一个"被下标访问就炸"的哨兵：

```python
class _ExplodingLauncher:
    """Stand-in for the Triton kernel: subscripting it (kernel[grid]) raises."""

    def __getitem__(self, _grid):
        raise AssertionError("310P path launched the Triton kernel")


class TestProposerDispatch(TestBase):
    def _make_proposer(self, cls, num_spec, q_per_req):
        proposer = cls.__new__(cls)
        proposer.num_speculative_tokens = num_spec
        proposer.parallel_drafting_token_id = MASK_ID
        proposer.kernel_block_size = BLOCK_SIZE
        proposer.device = torch.device("cpu")
        proposer.input_ids = torch.zeros(64, dtype=torch.int32)
        proposer.positions = torch.zeros(64, dtype=torch.int32)
        proposer._context_positions_buffer = torch.zeros(64, dtype=torch.int32)
        proposer._context_slot_mapping_buffers = torch.zeros(64, dtype=torch.int32)
        proposer._slot_mapping_buffer = torch.zeros(64, dtype=torch.int32)
        proposer._dflash_hidden_states = torch.zeros(64, 8)
        proposer.arange_dflash = torch.arange(65, dtype=torch.int32)
        proposer.token_arange_np = np.arange(65, dtype=np.int32)
        proposer.runner = MagicMock()
        proposer.runner.input_batch.block_table = {0: SimpleNamespace(block_size=BLOCK_SIZE)}
        proposer.kv_cache_gid = 0
        proposer.runner.kernel_block_sizes = {0: [128, 64]}
        return proposer

    def _make_cad(self, batch_size, ctx_lens, seq_lens):
        qsl = torch.zeros(batch_size + 1, dtype=torch.int32)
        qsl[1:] = torch.tensor(ctx_lens, dtype=torch.int32).cumsum(0)
        total = sum(ctx_lens)
        return SimpleNamespace(
            num_reqs=batch_size,
            slot_mapping=torch.arange(total, dtype=torch.int32),
            block_table_tensor=torch.arange(batch_size * 4, dtype=torch.int32).reshape(batch_size, 4),
            query_start_loc=qsl,
            query_start_loc_cpu=qsl.clone(),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            max_seq_len=max(seq_lens),
            num_actual_tokens=total,
            max_query_len=0,
            causal=True,
            attn_mask=object(),
            attn_state=None,
            actual_seq_lengths_q=[],
            decode_token_per_req=0,
        )

    def _run(self, cls, num_spec, q_per_req, on_310p):
        from unittest.mock import patch as mock_patch

        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)

        proposer = self._make_proposer(cls, num_spec, q_per_req)
        cad = self._make_cad(2, [1, 4], [257, 134])
        target_positions = torch.cat([torch.arange(256, 257), torch.arange(130, 134)]).to(torch.int32)

        # `_expand_drafting_inputs` is defined in dflash_proposer, so the names it
        # closes over (`is_310p`, the Triton launcher, the lazy helper import)
        # ALWAYS resolve from that module -- even when `self` is a DSpark proposer.
        # Patching dspark_proposer.* would silently miss.
        dflash_mod = "vllm_ascend.spec_decode.dflash_proposer"
        with (
            mock_patch(f"{dflash_mod}.is_310p", return_value=on_310p),
            mock_patch(
                "vllm_ascend._310p.spec_decode.parallel_drafting_inputs"
                ".expand_parallel_drafting_inputs",
                spy,
            ),
            mock_patch(
                "vllm_ascend._310p.spec_decode.parallel_drafting_inputs"
                ".resolve_310p_block_size",
                lambda proposer: BLOCK_SIZE,
            ),
            mock_patch(
                f"{dflash_mod}.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
                _ExplodingLauncher(),
            ),
        ):
            result = proposer.set_inputs_first_pass(
                target_token_ids=torch.zeros(5, dtype=torch.int32),
                next_token_ids=torch.tensor([11, 22], dtype=torch.int32),
                target_positions=target_positions,
                target_hidden_states=torch.zeros(5, 8),
                token_indices_to_sample=None,
                cad=cad,
                num_rejected_tokens_gpu=None,
            )
        return seen, result, cad

    def test_310p_dflash_uses_helper_with_skip_anchor(self):
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        seen, _, _ = self._run(
            AscendDflashProposer, 8, 9, on_310p=True
        )
        self.assertEqual(seen["num_query_per_req"], 9)
        self.assertEqual(seen["num_speculative_tokens"], 8)
        self.assertIs(seen["sample_from_anchor"], False)
        self.assertEqual(seen["block_size"], BLOCK_SIZE)

    def test_310p_dspark_uses_same_helper_with_anchor_sampling(self):
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

        seen, _, _ = self._run(
            AscendDSparkProposer, 7, 7, on_310p=True
        )
        self.assertEqual(seen["num_query_per_req"], 7)
        self.assertEqual(seen["num_speculative_tokens"], 7)
        self.assertIs(seen["sample_from_anchor"], True)

    def test_non_310p_still_launches_triton(self):
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        # The exploding launcher is the assertion: on non-310P it MUST be reached.
        with self.assertRaisesRegex(AssertionError, "launched the Triton kernel"):
            self._run(
                AscendDflashProposer, 8, 9, on_310p=False
            )

    def test_metadata_is_updated_for_parallel_drafting(self):
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata  # noqa: F401

        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        _, result, cad = self._run(
            AscendDflashProposer, 8, 9, on_310p=True
        )
        num_query_total, _, out_cad, _ = result
        self.assertEqual(num_query_total, 2 * 9)
        self.assertEqual(out_cad.num_actual_tokens, 2 * 9)
        self.assertEqual(out_cad.max_query_len, 9)
        self.assertIs(out_cad.causal, False)
        self.assertIsNone(out_cad.attn_mask)
        self.assertEqual(out_cad.attn_state, AscendAttentionState.ChunkedPrefill)
        self.assertEqual(out_cad.actual_seq_lengths_q, [9, 9])
        # seq_lens must become effective history + this round's query block.
        torch.testing.assert_close(
            out_cad.seq_lens, torch.tensor([257 + 9, 134 + 9], dtype=torch.int32)
        )
```

测试文件顶部需要 `import numpy as np`、`from types import SimpleNamespace`、
`from unittest.mock import MagicMock`。stub 字段清单以实施时 `set_inputs_first_pass` 实际读到的
属性为准，跑起来缺什么补什么。

**DSpark 的 stub 比 DFlash 多。** 它的 `set_inputs_first_pass` 现在还会读
`self.draft_attn_groups`、`self._per_group_block_tables`、`self._per_group_slot_mappings`、
`self._per_group_context_slot_mapping_buffers`、`self._per_group_query_slot_mapping_buffers`、
`self._layer_group_idx`、`self._dspark_seed_buffer`。给一个 group 的 stub 即可：

```python
    def _add_dspark_group_stubs(self, proposer, batch_size, num_query_total):
        gid = 0
        proposer.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_group_id=gid,
                kv_cache_spec=SimpleNamespace(block_size=BLOCK_SIZE),
            )
        ]
        proposer._per_group_block_tables = {
            gid: torch.arange(batch_size * 4, dtype=torch.int32).reshape(batch_size, 4)
        }
        proposer._per_group_slot_mappings = {gid: torch.arange(64, dtype=torch.int32)}
        proposer._per_group_context_slot_mapping_buffers = {gid: torch.zeros(64, dtype=torch.int32)}
        proposer._per_group_query_slot_mapping_buffers = {gid: torch.zeros(64, dtype=torch.int32)}
        proposer._layer_group_idx = [gid]
        proposer._dspark_seed_buffer = torch.zeros(64, dtype=torch.int64)
```

并额外加一条断言 helper **只被调用一次**（单 group）：

```python
    def test_dspark_single_group_calls_helper_once(self):
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

        calls = []
        # ... same wiring as _run, but the spy appends instead of overwriting ...
        self.assertEqual(len(calls), 1, "Qwen3-8B DSpark must have exactly one draft KV group")
```

以及一条"两个 group 时必须报错"的用例，覆盖 §1.3 里那条 scope guard。

DSpark 的 metadata 断言比 DFlash 多三项（它在 `set_inputs_first_pass` 末尾额外设了这些）：

```python
    def test_dspark_sets_positions_and_input_tokens(self):
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

        _, result, _ = self._run(AscendDSparkProposer, 7, 7, on_310p=True)
        num_query_total, _, out_cad, _ = result
        self.assertEqual(num_query_total, 2 * 7)
        self.assertEqual(out_cad.num_input_tokens, 2 * 7)
        self.assertEqual(out_cad.positions.shape[0], 2 * 7)
        # slot_mapping must come from the PRIMARY group's query buffer, not from
        # cad.slot_mapping and not from a non-primary group.
        primary_gid = 0
        self.assertEqual(
            out_cad.slot_mapping.data_ptr(),
            proposer._per_group_query_slot_mapping_buffers[primary_gid][: 2 * 7].data_ptr(),
        )
```

（`proposer` 需要从 `_run` 一并返回，或把 `_run` 改成也交出 proposer 实例。）

#### 1.5 验证命令

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/spec_decode/test_parallel_drafting_inputs_310p.py
python tests/e2e/_310p/spec_decode/smoke_parallel_drafting_inputs.py
```

---

### Task 2：启用 Qwen3 patch 并修复五层 context RoPE

#### 2.1 调整 patch gate

当前基线上 `if not is_310p()` 里只剩三个 import：`patch_qwen3_5`、`patch_qwen3_dflash`、
`patch_qwen3vl`。**只需把 `patch_qwen3_dflash` 移出去**，`patch_qwen3_5` 和 `patch_qwen3vl`
继续保持 310P gate，不扩大模型范围。

`patch_qwen3_dspark.py` 已经被删除，但**不要据此认为 DSpark 不再需要任何 patch**——顶层
`mask_token_id` 仍然靠 platform 层的 `patch_speculative_config` 接进上游，详见 §0.1。本期不需要
新增或恢复任何 worker patch，但那个 platform patch 必须保留。

DSpark 的模型侧（`Qwen3DSparkModel(DFlashQwen3Model)`）继承 DFlash 的
`precompute_and_store_context_kv`，所以 Task 2.2 的逐层 RoPE 修复对两者同时生效。

#### 2.2 310P 使用逐层 RoPE

保留 fused KV projection 和 K norm。RoPE 部分做 device dispatch：

- 310P：遍历 5 层，每层只把 `[num_ctx, Nkv, D]` reshape 后送入该层 rotary，接住返回的第一个
  tensor 作为 rotated K，cache writer 收到 rotated K 和原 V；
- 非 310P：保留 `[L * num_ctx]` fused RoPE，但也接住返回 tensor。

cos/sin 不需要在这里手工准备——drafting flag 一开，每层 rotary 自己会用它收到的
`context_positions` 刷新 slice。正常路径由 `_run_merged_draft` 开 flag，profile 路径由本 Task 2.2
下半段的 `_profile_rope_context` 开，两条路都覆盖到了。

把 `patch_qwen3_dflash.py` 原来这段：

```python
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
```

替换为：

```python
    all_k_flat = apply_context_rope(
        layers=self.layers,
        all_k_normed=all_k_normed,
        context_positions=context_positions,
        num_layers=L,
        num_ctx=num_ctx,
        kv_size=kv,
    )
```

并在模块里加：

```python
from vllm_ascend.utils import is_310p


def apply_context_rope(*, layers, all_k_normed, context_positions, num_layers, num_ctx, kv_size):
    """Rotate the fused context K and return the rotated tensor.

    Ascend rotary embeddings are out-of-place on 310P (they return new tensors),
    so the return value must be used on every device -- the A2/A3 path mutates in
    place and returns the same storage, so taking the result is correct there too.

    310P rotates one layer at a time. The fused [L * num_ctx] form cannot be used
    there: the drafting-time RoPE fills a module-global cos/sin buffer sized
    max_num_batched_tokens, and L * num_ctx overflows it for any realistic context
    with a 5-layer drafter.
    """
    if not is_310p():
        all_k_flat = all_k_normed.view(num_layers * num_ctx, kv_size)
        positions_repeated = context_positions.repeat(num_layers)
        rotated, _ = layers[0].self_attn.rotary_emb(
            positions_repeated, all_k_flat, all_k_flat.clone()
        )
        return rotated

    all_k_flat = all_k_normed.view(num_layers, num_ctx, kv_size)
    rotated_layers = []
    for i in range(num_layers):
        layer_k = all_k_flat[i]
        rotated, _ = layers[i].self_attn.rotary_emb(
            context_positions, layer_k, layer_k.clone()
        )
        rotated_layers.append(rotated)
    return torch.stack(rotated_layers).view(num_layers * num_ctx, kv_size)
```

注意 310P 分支用的是 `layers[i].self_attn.rotary_emb` 而不是 `layers[0]` 的——上游 fused 版本
之所以能用第 0 层，是因为 `_build_fused_kv_buffers` 校验过所有层的 RoPE 参数一致
（`qwen3_dflash.py:450-456`）。逐层写法保持同样语义但不依赖那个假设，代价为零。

不再 `context_positions.repeat(L)` 后送入 310P RoPE，也不添加“超过全局 buffer 就报错”的
容量 guard。逐层处理本身消除了 `L * num_ctx` 容量问题。

正常 draft wrapper 可能让每层 rotary 再次刷新同一份 cos/sin；eager correctness MVP 可以先容忍
这部分重复工作，后续性能任务再将其收敛为严格一次更新。

##### profile 分支必须整体包在 drafting flag 里

不要在 context KV 之前手工刷一次 cos/sin。那样只覆盖了两个 forward 中的第一个，紧随其后的
`self.model(positions=query_positions)` 仍会读到 context 的 slice；而且"刷完检查非 `None` 和长度"
这种校验识别不了**旧 slice 长度恰好相同**的情况，属于假的安全感。

正确做法是让 profile 分支和正常 draft 路径行为一致。正常路径靠
`AscendSpecDecodeBaseProposer310._run_merged_draft` 打开 flag，于是每次 rotary 都用**自己的**
positions 调 `update_cos_sin`。profile 分支照抄这个模式即可：

在 `AscendDflashProposer` 上加一个共享 context manager，`AscendDSparkProposer` 继承：

```python
from contextlib import contextmanager

from vllm_ascend.utils import is_310p


    @contextmanager
    def _profile_rope_context(self):
        """Keep the 310P drafting RoPE flag on for the whole profile branch.

        dummy_run(is_profile=True) calls precompute_and_store_context_kv and then
        model(...) directly, bypassing _run_merged_draft, which is what normally
        turns this flag on. Without it neither forward refreshes the global
        cos/sin slice, so both silently reuse whatever the target's dummy run
        left behind -- and the two forwards need different positions anyway
        (context_positions vs query positions, different lengths).
        """
        if not is_310p():
            yield
            return

        from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310

        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            yield
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
```

然后在**两个** proposer 的 `dummy_run` 里各包一次（`dflash_proposer.py:229`、
`dspark_proposer.py:344`）：

```python
            if is_profile:
                with self._profile_rope_context():
                    self.model.precompute_and_store_context_kv(context_states, context_positions)
                    self.model(
                        input_ids=self.input_ids[:num_query_total],
                        positions=self._get_positions(num_query_total),
                        inputs_embeds=None,
                    )
```

这样两个 forward 各自按自己的 positions 刷新 slice，无需任何手工准备函数，也不需要在
patch 里加校验。**不要再引入单独的 cos/sin 准备 helper** —— flag 包装已经完全覆盖它的职责，
两套机制并存只会让下一个人不知道该信哪个。

代价是每层 rotary 都会用同一份 `context_positions` 重复刷新一次 slice（5 层就是 5 次）。
这与正常 draft 路径的行为完全一致，eager correctness MVP 接受这部分重复，后续性能任务再收敛。

#### 2.3 UT

- [ ] 构造 L=5 的模型 stub；
- [ ] 310P 下 rotary 被调用 5 次，每次 positions 长度都是 `num_ctx`；
- [ ] rotary mock 返回与输入明显不同的新 tensor；
- [ ] cache writer 收到返回后的 rotated K，而不是原 K；
- [ ] 非 310P 仍是一次 fused RoPE；
- [ ] 在 subprocess 中断言 `DFlashQwen3Model.precompute_and_store_context_kv` identity 指向 patch
      实现（**不能用 `hasattr` 验**，上游类本来就定义了同名方法）；
- [ ] 断言 `Qwen3DSparkModel` 继承到的是同一个 patch 实现（即 DSpark 也被覆盖到）。

原计划里"DSpark `_init_parallel_drafting_params` patch identity"一条已随
`patch_qwen3_dspark.py` 的删除失去目标，去掉。

**但 mask-token 回归测试不能删**，只是要换靶子：顶层 `mask_token_id` → `ptd_token_id` 的转换现在
由 platform patch 承担（§0.1）。加这一条：

```python
class TestDSparkMaskTokenPlumbing(TestBase):
    def test_top_level_mask_token_is_promoted_to_ptd_token_id(self):
        """The指定 DSpark checkpoint puts mask_token_id at the top level, which
        upstream does not read. patch_speculative_config promotes it; without
        that promotion the proposer raises ValueError at construction."""
        from types import SimpleNamespace
        from unittest.mock import patch as mock_patch

        from vllm_ascend.patch.platform.patch_speculative_config import _dspark_post_init

        draft_hf = SimpleNamespace(mask_token_id=151669)
        cfg = SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft_hf),
            use_dspark=lambda: True,
        )
        with mock_patch(
            "vllm_ascend.patch.platform.patch_speculative_config._orig_post_init",
            lambda self: None,
        ):
            _dspark_post_init(cfg)

        self.assertEqual(draft_hf.ptd_token_id, 151669)

    def test_existing_ptd_token_id_is_not_overwritten(self):
        from types import SimpleNamespace
        from unittest.mock import patch as mock_patch

        from vllm_ascend.patch.platform.patch_speculative_config import _dspark_post_init

        draft_hf = SimpleNamespace(ptd_token_id=42, mask_token_id=151669)
        cfg = SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft_hf),
            use_dspark=lambda: True,
        )
        with mock_patch(
            "vllm_ascend.patch.platform.patch_speculative_config._orig_post_init",
            lambda self: None,
        ):
            _dspark_post_init(cfg)

        self.assertEqual(draft_hf.ptd_token_id, 42)
```

上游怎么消费 `ptd_token_id` 不重复测；端到端是否真的读对由 Task 4 的 DSpark E2E 覆盖。

profile 分支的 flag 包装另用一组测试，**必须调用两个真实的 `dummy_run(is_profile=True)`**，
不能只测 context manager 本身，也不要去 mock `ops/rotary_embedding` 的私有全局——那些全局在
真实时序下不会是 `None`，mock 成 `None` 测的是不存在的场景。

要点是让 context 与 query 的 positions **长度不同**，这样"flag 只覆盖了第一个 forward"这种
半修复会被抓住：

```python
class TestProfileRopeFlagCoverage(TestBase):
    """dummy_run(is_profile=True) must keep the 310P drafting RoPE flag on for
    both the context KV precompute and the query forward that follows it."""

    def _fake_model(self, seen):
        class FakeModel:
            def precompute_and_store_context_kv(_self, states, positions):
                seen.append(("context", AscendRotaryEmbedding310._is_drafting_update_enabled,
                             positions.shape[0]))

            def __call__(_self, *, input_ids, positions, inputs_embeds):
                seen.append(("query", AscendRotaryEmbedding310._is_drafting_update_enabled,
                             positions.shape[0]))

        return FakeModel()

    def _run_profile(self, proposer_cls, num_spec, q_per_req, caller_module):
        from unittest.mock import patch as mock_patch

        seen = []
        proposer = proposer_cls.__new__(proposer_cls)
        # Minimal state dummy_run touches on the is_profile path.
        proposer.model = self._fake_model(seen)
        proposer.num_speculative_tokens = num_spec
        proposer.max_query_tokens = 64
        proposer.use_cuda_graph = False
        proposer.input_ids = torch.zeros(64, dtype=torch.int32)
        proposer.hidden_states = torch.zeros(64, 8)
        proposer._context_positions_buffer = torch.arange(64, dtype=torch.int32)
        proposer.token_indices_to_sample = torch.zeros(64, dtype=torch.int32)
        proposer.positions = torch.arange(64, dtype=torch.int32)
        proposer.runner = MagicMock()
        proposer.runner._sync_metadata_across_dp.return_value = (12, None, None)
        proposer.vllm_config = MagicMock()

        # `set_ascend_forward_context` / `get_forward_context` are imported directly
        # into each proposer module (dflash_proposer.py:8, dspark_proposer.py:13),
        # so they must be patched in the module whose dummy_run is running --
        # patching llm_base_proposer.* would not be seen.
        # `is_310p` is patched in dflash_proposer because that is where
        # `_profile_rope_context` is defined, whichever subclass calls it.
        with (
            mock_patch("vllm_ascend.spec_decode.dflash_proposer.is_310p", return_value=True),
            mock_patch(f"{caller_module}.set_ascend_forward_context"),
            mock_patch(f"{caller_module}.get_forward_context"),
        ):
            proposer.dummy_run(num_tokens=12, num_reqs=2, is_profile=True)

        return seen

    def _assert_both_forwards_covered(self, seen):
        self.assertEqual([kind for kind, _, _ in seen], ["context", "query"])
        for kind, flag, _ in seen:
            self.assertTrue(flag, f"{kind} forward ran with the drafting RoPE flag off")
        ctx_len = seen[0][2]
        q_len = seen[1][2]
        self.assertNotEqual(
            ctx_len, q_len, "test is not discriminating: make context and query lengths differ"
        )

    def test_dflash_profile_covers_context_and_query(self):
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        self._assert_both_forwards_covered(
            self._run_profile(
                AscendDflashProposer, 8, 9, "vllm_ascend.spec_decode.dflash_proposer"
            )
        )

    def test_dspark_profile_covers_context_and_query(self):
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

        self._assert_both_forwards_covered(
            self._run_profile(
                AscendDSparkProposer, 7, 7, "vllm_ascend.spec_decode.dspark_proposer"
            )
        )

    def test_flag_restored_after_profile_even_on_error(self):
        from unittest.mock import patch as mock_patch

        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        proposer = AscendDflashProposer.__new__(AscendDflashProposer)
        # Without forcing is_310p the context manager yields immediately without
        # ever touching the flag, and this test passes vacuously on CPU/Mac.
        with mock_patch("vllm_ascend.spec_decode.dflash_proposer.is_310p", return_value=True):
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
            with self.assertRaises(ValueError):
                with proposer._profile_rope_context():
                    self.assertTrue(
                        AscendRotaryEmbedding310._is_drafting_update_enabled,
                        "flag was never set, so the restore assertion below is vacuous",
                    )
                    raise ValueError("boom")
        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)
```

测试文件顶部需要 `from unittest.mock import MagicMock` 和
`from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310`。

上面的 stub 字段清单以实施时 `dummy_run` 在 `is_profile` 路径实际读到的属性为准；跑起来缺什么补
什么，但**不要**为了让测试跑通而绕过 `dummy_run` 直接调 `_profile_rope_context` —— 那样就测不到
"包装是否真的套住了两个 forward"，而这正是本条要防的回归。

不能用 `hasattr(DFlashQwen3Model, "precompute_and_store_context_kv")` 验证 patch，因为上游类本来
就定义了同名方法。

#### 2.4 验证命令

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/test_qwen3_parallel_drafting_patches_310p.py
```

---

### Task 3：实现 ADN adapter 和精确 attention routing

#### 3.1 lazy loader 与 fail-loud

`vllm_ascend/_310p/attention/adn_fused_infer_attention.py` 在真正进入 DFlash/DSpark draft path 时
才 import `adn_custom_ops`，并缓存成功结果。

ImportError、动态库错误或 ABI 不匹配时，错误信息必须说明：

- 需要安装 310P 对应 custom OPP 和 PTA wheel；
- 当前请求是 DFlash/DSpark non-causal draft attention；
- 系统不会 fallback 到 causal splitfuse。

#### 3.2 静态能力校验

每个 attention impl 首次调用校验一次：

- 当前 scope 是 Qwen3-8B dense + 指定 DFlash/DSpark architecture；
- MRV1、eager、FP16、TP=2；
- query 是 FP16 TND `[T, 16, 128]`；
- K/V 是同 shape、同 device 的 FP16 rank-4 cache；
- `torch_npu.get_npu_format(cache) == ACL_FORMAT_FRACTAL_NZ`；
- cache last dim 为 16，dim1 为 `4 * 128 / 16 = 32`；
- cache `shape[-2] == block_size == 128`；
- GQA ratio 和 ADN constraints 满足。

检查通过后才设置 static-validated flag。

#### 3.3 每步动态校验

每次 forward 都检查：

- `get_query_lens_cpu(attn_metadata)` 存在；
- DFlash raw q-lens 全为 9，DSpark 全为 7；
- `sum(q_lens) == num_actual_tokens == query_slice.shape[0]`；
- `seq_lens_list` 长度与 q-lens、block-table rows 一致；
- 每个 `0 < q_len <= kv_len <= block_table_cols * 128`；
- block table 为 NPU int32 rank-2；
- output slice 的 shape/dtype 与 ADN 返回一致。

active physical ID bounds 不放入热路径，因为读取 NPU IDs 会产生同步；allocator invariant 由
Phase 0 的乱序 physical-page readback 验证。

#### 3.4 固定 ADN 调用

adapter：

1. 截取 `query[:num_actual_tokens]` 和对应 output slice；
2. 从 raw host q-len 生成 Python list；
3. 使用 `attn_metadata.seq_lens_list` 作为总 KV lengths；
4. 传 `attn_mask=None`、`inner_precise=2`、`force_call=False`；
5. 检查返回 shape/dtype；
6. copy 到 caller output slice；
7. 返回原 caller output buffer。

不得使用 base metadata 的 cumulative `actual_seq_lengths_q`。

#### 3.1–3.4 的完整实现

`vllm_ascend/_310p/attention/adn_fused_infer_attention.py`：

```python
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import torch
import torch_npu

from vllm_ascend._310p.attention.metadata_builder import get_query_lens_cpu
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

# NOTE (2026-07-24 amendment): the TP=2 / hardcoded-head-count version below is
# superseded. The implemented validator (validate_adn_scope in
# _310p/attention/adn_fused_infer_attention.py) does NOT pin TP or a fixed head
# count. TP only shards heads across ranks without changing the numerics, and the
# drafter's exact head layout comes from its checkpoint, so the per-rank layout is
# constrained structurally instead: 0 < head_dim <= 256, head_dim * block <= 16384
# (which caps head_dim at 128 given block=128), Nq % Nkv == 0, Nq/Nkv <= 64, and
# Nkv*head_dim 16-aligned. This lets TP=2 (16/4) and TP=4 (8/2) both run. The
# source is authoritative; the snapshot below is kept only for narrative context.
ADN_BLOCK_SIZE = 128
ADN_HEAD_DIM = 128
ADN_LOCAL_NUM_HEADS = 16  # Qwen3-8B: 32 Q heads / TP=2
ADN_LOCAL_NUM_KV_HEADS = 4  # Qwen3-8B: 8 KV heads / TP=2
ADN_SUPPORTED_METHODS = {"dflash": 8, "dspark": 7}  # method -> num_speculative_tokens

_adn_module = None


def _load_adn():
    """Import adn_custom_ops on first real use and cache it.

    Its package __init__ imports torchair at module scope, so importing it
    eagerly would make torchair a hard dependency of every 310P run.
    """
    global _adn_module
    if _adn_module is not None:
        return _adn_module
    try:
        import adn_custom_ops
    except Exception as exc:  # ImportError, OSError from the .so, ABI mismatch
        raise RuntimeError(
            "DFlash/DSpark non-causal draft attention on 310P requires the ADN "
            "custom op package (adn_custom_ops + adn_custom_ops_lib + torchair). "
            "Install the Ascend_Ops custom_opp and PTA wheels for this device. "
            "There is no fallback: routing this path to the causal split-fuse "
            "kernel would silently produce wrong results, so startup fails instead."
        ) from exc
    _adn_module = adn_custom_ops
    return _adn_module


def validate_adn_capability(*, vllm_config, query, key_cache, value_cache, num_heads, num_kv_heads, head_size):
    """Static, scope-level checks. Run once per attention impl, then cached.

    These pin the exact configuration this plan validated on hardware -- not ADN's
    generic envelope. Running an unvalidated shape would silently produce numbers
    nobody has checked, so anything outside the scope fails at startup.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or spec_config.method not in ADN_SUPPORTED_METHODS:
        raise RuntimeError(
            f"ADN draft attention only covers {sorted(ADN_SUPPORTED_METHODS)}, got "
            f"{getattr(spec_config, 'method', None)}"
        )
    expected_k = ADN_SUPPORTED_METHODS[spec_config.method]
    if spec_config.num_speculative_tokens != expected_k:
        raise RuntimeError(
            f"{spec_config.method} is only validated at K={expected_k}, got "
            f"{spec_config.num_speculative_tokens}"
        )

    arch = getattr(spec_config.draft_model_config.hf_config, "architectures", [None])[0]
    if arch not in {"DFlashQwen3ForCausalLM", "Qwen3DSparkForCausalLM"}:
        raise RuntimeError(f"unsupported draft architecture {arch} for this scope")

    if not vllm_config.model_config.enforce_eager:
        raise RuntimeError("this scope is eager-only; ACLGraph is validated separately")
    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp != 2:
        raise RuntimeError(f"this scope is validated at TP=2 only, got TP={tp}")

    if (num_heads, num_kv_heads, head_size) != (
        ADN_LOCAL_NUM_HEADS,
        ADN_LOCAL_NUM_KV_HEADS,
        ADN_HEAD_DIM,
    ):
        raise RuntimeError(
            f"this scope only covers local Nq={ADN_LOCAL_NUM_HEADS}, "
            f"Nkv={ADN_LOCAL_NUM_KV_HEADS}, D={ADN_HEAD_DIM} (Qwen3-8B at TP=2), got "
            f"Nq={num_heads}, Nkv={num_kv_heads}, D={head_size}"
        )

    for name, t in (("query", query), ("key_cache", key_cache), ("value_cache", value_cache)):
        if t.dtype != torch.float16:
            raise RuntimeError(
                f"ADN on 310P only supports float16 in this scope, but {name} is "
                f"{t.dtype}. Start the engine with dtype=float16."
            )

    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(f"ADN needs rank-4 NZ K/V caches, got {key_cache.ndim}/{value_cache.ndim}")
    if key_cache.shape != value_cache.shape:
        raise RuntimeError(f"K/V cache shapes differ: {key_cache.shape} vs {value_cache.shape}")
    if key_cache.device != value_cache.device:
        raise RuntimeError(f"K/V caches are on different devices: {key_cache.device} vs {value_cache.device}")

    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        fmt = torch_npu.get_npu_format(cache)
        if fmt != ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError(
                f"{name} is in acl format {fmt}, expected ACL_FORMAT_FRACTAL_NZ "
                f"({ACL_FORMAT_FRACTAL_NZ}). ADN reads the NZ layout directly."
            )

    if key_cache.shape[-1] != 16:
        raise RuntimeError(f"NZ cache last dim must be 16, got {key_cache.shape[-1]}")
    # Compare the cache's physical block size against the scope constant, not
    # against a value derived from the cache itself -- that would be tautological.
    if key_cache.shape[-2] != ADN_BLOCK_SIZE:
        raise RuntimeError(
            f"cache physical block size is {key_cache.shape[-2]}, this scope only "
            f"covers {ADN_BLOCK_SIZE}"
        )
    expected_dim1 = num_kv_heads * head_size // 16
    if key_cache.shape[1] != expected_dim1:
        raise RuntimeError(
            f"NZ cache dim1 is {key_cache.shape[1]}, expected "
            f"num_kv_heads*head_size/16 = {expected_dim1}"
        )
    if (num_kv_heads * head_size) % 16:
        raise RuntimeError(f"num_kv_heads*head_dim = {num_kv_heads * head_size} is not 16-aligned")


def _expected_q_per_req(spec_config):
    """DFlash queries K+1 positions per request; DSpark queries K."""
    k = spec_config.num_speculative_tokens
    return k + 1 if spec_config.method == "dflash" else k


def forward_parallel_draft_adn(self, query, attn_metadata, output):
    """Non-causal parallel-draft attention via ADN.

    `attn_mask=None` is what makes ADN non-causal: its host tiling maps an empty
    mask to NO_MASK and the kernel neither loads nor applies one, so every query
    row sees the full [0, actual_seq_lengths_kv[b]) range -- context plus this
    round's entire query block. Never pass the 310P compressed split-fuse mask
    here, and never synthesize an all-zero causal mask to fake it.
    """
    adn = _load_adn()

    num_tokens = int(attn_metadata.num_actual_tokens)
    query_slice = query[:num_tokens]
    output_slice = output[:num_tokens]

    # Raw per-request q-lens come from the 310P builder, which already diffed the
    # CPU endpoints outside the forward. The tensor is host/pinned, so .tolist()
    # costs no NPU sync. Never rebuild these from the base metadata's
    # actual_seq_lengths_q (cumulative endpoints) or from max_query_len.
    raw_q_lens = get_query_lens_cpu(attn_metadata)
    if raw_q_lens is None:
        raise RuntimeError(
            "310P parallel draft attention needs raw per-request query lengths on "
            "attn_metadata, but query_lens_cpu is missing. It is set by "
            "AscendAttentionMetadataBuilder310.build() for ChunkedPrefill/SpecDecoding; "
            "check that the draft metadata went through that builder."
        )
    q_lens = raw_q_lens.tolist()
    kv_lens = attn_metadata.seq_lens_list

    spec_config = self.vllm_config.speculative_config
    expected_q = _expected_q_per_req(spec_config)
    if any(q != expected_q for q in q_lens):
        raise RuntimeError(
            f"{spec_config.method} expects every request to query {expected_q} "
            f"positions, got {q_lens}. A cumulative-endpoint tensor was most likely "
            f"passed instead of raw per-request lengths."
        )
    if sum(q_lens) != num_tokens or query_slice.shape[0] != num_tokens:
        raise RuntimeError(
            f"sum(q_lens)={sum(q_lens)}, num_actual_tokens={num_tokens}, "
            f"query rows={query_slice.shape[0]} must all agree"
        )

    block_table = attn_metadata.block_tables[: len(q_lens)]
    if len(kv_lens) != len(q_lens) or block_table.shape[0] != len(q_lens):
        raise RuntimeError(
            f"batch size disagreement: {len(q_lens)} q-lens, {len(kv_lens)} kv-lens, "
            f"{block_table.shape[0]} block-table rows"
        )
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise RuntimeError(
            f"block table must be a rank-2 int32 NPU tensor, got ndim="
            f"{block_table.ndim} dtype={block_table.dtype}"
        )

    key_cache = self.key_cache
    value_cache = self.value_cache
    # Fixed by scope, not read back from the cache. validate_adn_capability checks
    # the cache's physical block size against this same constant.
    capacity = block_table.shape[1] * ADN_BLOCK_SIZE
    for b, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        if not 0 < ql <= kl:
            raise RuntimeError(f"request {b}: need 0 < q_len({ql}) <= kv_len({kl})")
        if kl > capacity:
            raise RuntimeError(
                f"request {b}: kv_len {kl} exceeds what its block table can address "
                f"({block_table.shape[1]} pages x {ADN_BLOCK_SIZE})"
            )

    query_tnd = query_slice.reshape(num_tokens, self.num_heads, self.head_size)

    if not self._adn_validated:
        validate_adn_capability(
            vllm_config=self.vllm_config,
            query=query_tnd,
            key_cache=key_cache,
            value_cache=value_cache,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
        )
        self._adn_validated = True

    adn_out = adn.adn_fused_infer_attention(
        query=query_tnd,
        key=key_cache,
        value=value_cache,
        attn_mask=None,
        actual_seq_lengths_q=q_lens,
        actual_seq_lengths_kv=kv_lens,
        block_table=block_table,
        num_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        block_size=ADN_BLOCK_SIZE,
        input_layout="TND",
        scale_value=self.scale,
        inner_precise=2,
        force_call=False,
    )

    # ADN's contract is "output has the same shape as query". Check that exactly,
    # not just numel against the (possibly flat [T, Nq*D]) output slice -- a numel
    # match would accept a transposed or mis-headed result.
    if adn_out.shape != query_tnd.shape or adn_out.dtype != query_tnd.dtype:
        raise RuntimeError(
            f"ADN returned {tuple(adn_out.shape)}/{adn_out.dtype}, expected the query "
            f"shape {tuple(query_tnd.shape)}/{query_tnd.dtype}"
        )
    if adn_out.numel() != output_slice.numel():
        raise RuntimeError(
            f"ADN output has {adn_out.numel()} elements but the output slice holds "
            f"{output_slice.numel()}"
        )

    # ADN allocates its own output and has no `out=`, so the MVP eats one copy.
    # Do not change the operator ABI before correctness closes.
    output_slice.copy_(adn_out.reshape_as(output_slice))
    return output
```

`self._adn_validated` 在 `AscendAttentionBackendImpl310.__init__` 末尾初始化为 `False`。

上面两个 import 路径已核对过：`ACL_FORMAT_FRACTAL_NZ = 29` 定义在 `vllm_ascend/utils.py:55`
（`_310p/model_runner_310p.py:58` 用的就是它），`torch_npu.get_npu_format` 在
`xlite/xlite.py:198` 和 `model_executor/offloader/prefetch.py:40` 有现成用法。

#### 3.5 精确路由

在 `_310p/attention/attention_v1.py::forward_impl` 里，`state = attn_metadata.attn_state` 之后、
`if state == AscendAttentionState.PrefillNoCache:` 之前插入：

```python
        # self.vllm_config is bound in AscendAttentionBackendImpl.__init__ via
        # get_current_vllm_config(); there is no bare `speculative_config` here.
        spec_config = self.vllm_config.speculative_config
        is_parallel_draft_adn = (
            _EXTRA_CTX.is_draft_model
            and spec_config is not None
            and spec_config.method in {"dflash", "dspark"}
            and state == AscendAttentionState.ChunkedPrefill
            and not attn_metadata.causal
        )
        if is_parallel_draft_adn:
            return forward_parallel_draft_adn(self, query, attn_metadata, output)
        if not attn_metadata.causal:
            # Everything below this point is causal. Falling through would hand a
            # non-causal request to the split-fuse kernel and return a plausible
            # but wrong result, so refuse instead.
            raise NotImplementedError(
                f"310P has no non-causal attention path for attn_state={state}, "
                f"is_draft_model={_EXTRA_CTX.is_draft_model}, "
                f"method={getattr(spec_config, 'method', None)}. Only DFlash/DSpark "
                f"draft ChunkedPrefill is routed to ADN in this scope."
            )
```

文件顶部加：

```python
from vllm_ascend._310p.attention.adn_fused_infer_attention import forward_parallel_draft_adn
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
```

并在 `AscendAttentionBackendImpl310.__init__` 末尾加 `self._adn_validated = False`。

- `is_parallel_draft_adn=True`：走 ADN adapter；
- 其他 `causal=False`：抛 `NotImplementedError`，避免静默进入 causal splitfuse；
- causal Prefill/Decode/ChunkedPrefill/SpecDecoding：保持当前 310P routing，一行不改。

pooling 已在 base `forward` 提前返回，不会到达这里，不需要纳入本期 ADN route。

这条 fail-loud guard **只能放在 310P 的 `forward_impl`**，不得上移到通用
`AscendAttentionBackendImpl` 或 base `forward`——A2/A3 上的 non-causal 路径（含 pooling 和已支持
的 910 FIA 分支）不在本期范围内，在共享层加 guard 会直接把它们打挂。

与文首“核心方案”第 4 条的措辞对齐：**保持不变的是 causal 与 pooling 路径**，未支持的 non-causal
是主动报错，两者不冲突。

#### 3.6 UT

adapter 测试：

- [ ] 使用 raw `[9, 9, 9]` / `[7, 7, 7]`，并故意把 cumulative field 设成错误值；
- [ ] ragged KV lengths；
- [ ] q-len 缺失、长度不匹配、错误 method q-len 均 fail loud；
- [ ] FP16、NZ format、rank、shape、block size、block table dtype/capacity guard；
- [ ] ADN missing 或抛错时没有 causal fallback；
- [ ] adapter 参数中 mask 为 `None`；
- [ ] 返回新 tensor 被 copy，函数返回原 output buffer。

routing 测试：

- [ ] draft + dflash + ChunkedPrefill + non-causal -> ADN；
- [ ] draft + dspark + ChunkedPrefill + non-causal -> ADN；
- [ ] target/non-draft non-causal -> fail loud；
- [ ] wrong method non-causal -> fail loud；
- [ ] causal DFlash/DSpark -> 原 causal route；
- [ ] PrefillCacheHit、SpecDecoding、DecodeOnly 的现有 causal route 不变。

测试必须 patch `_EXTRA_CTX.is_draft_model` 和 `impl.vllm_config.speculative_config.method`
（注意是 impl 上的 `vllm_config`，由 `get_current_vllm_config()` 在 `__init__` 里绑定），
不能只构造 `causal=False` metadata。

#### 3.7 验证命令

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/attention/test_adn_fused_infer_attention_310p.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/attention/test_parallel_draft_routing_310p.py
```

---

### Task 4：Qwen3-8B eager E2E

> **NOTE (2026-07-24 amendment):** 实机只有 `deepseek-ai/dspark_qwen3_8b_block7`，且要用 **TP=4**。
> 因此本轮 E2E **只测 DSpark（K=7）**，`tensor_parallel_size=4`、`draft_tensor_parallel_size=4`。
> DFlash 没有 checkpoint，端到端**延后**——但它并非零覆盖：DFlash 的 q=9 / skip-anchor 布局有
> CPU 单测，且 Phase 0.4 的 NZ 门禁包含 DFlash q=9 用例，attention 算子层已验证。下面 4.1 里
> DFlash 的 runner 本轮跳过，其余（baseline vs spec 一致性、acceptance 防全拒绝、精确页边界）
> 照常，只是 runner 从三个减为两个（baseline + DSpark）。

测试放在 `four_card/_310p` 是刻意的：四卡 runner 正好申请到 TP=4。完成后更新
`.github/workflows/scripts/test_config.yaml` 的 estimated time。

#### 4.1 固定 runner 配置

三种 runner 使用完全相同的 target 配置，只有 speculative config 不同：

测试文件顶部先写死 Phase 0.1 解析出来的三个 SHA（示例格式，实际值以 Phase 0 记录为准）：

```python
# Resolved in Phase 0.1; never use a movable ref like "main" here.
TARGET_REVISION = "<sha from HfApi().model_info('Qwen/Qwen3-8B').sha>"
DFLASH_REVISION = "<sha from HfApi().model_info('z-lab/Qwen3-8B-DFlash-b16').sha>"
DSPARK_REVISION = "<sha from HfApi().model_info('deepseek-ai/dspark_qwen3_8b_block7').sha>"
```

```python
common = dict(
    model_name="Qwen/Qwen3-8B",
    revision=TARGET_REVISION,
    dtype="float16",
    tensor_parallel_size=2,
    block_size=128,
    enforce_eager=True,
    distributed_executor_backend="mp",
    enable_prefix_caching=False,
    disable_log_stats=False,
    max_model_len=512,
)
```

DFlash：

```python
speculative_config = {
    "method": "dflash",
    "model": "z-lab/Qwen3-8B-DFlash-b16",
    "revision": DFLASH_REVISION,
    "num_speculative_tokens": 8,
    "draft_tensor_parallel_size": 2,
}
```

DSpark 同理，使用 K=7、对应 revision 和 `draft_tensor_parallel_size=2`。

测试命令和测试进程都要确保：

```bash
VLLM_USE_V2_MODEL_RUNNER=0
```

不得依赖 `VllmRunner` 默认的 `block_size=16`，否则 310P D=128 的 kernel block selection 会失败。

#### 4.2 只加载三次大模型

1. baseline Qwen3-8B runner 加载一次，生成所有普通、ragged 和 boundary prompts 的 greedy golden；
2. DFlash runner 加载一次，运行正确性 + acceptance；
3. DSpark runner 加载一次，运行正确性 + acceptance。

每个 runner 退出后使用仓库现有 NPU memory cleanup/wait helper，避免连续 load OOM。

#### 4.3 正确性矩阵

- [ ] 单请求普通文本；
- [ ] 三请求 ragged batch；
- [ ] 长生成，跨多个 draft/verify round，并实际产生 rejection；
- [ ] DFlash 页边界；
- [ ] DSpark 页边界；
- [ ] 每个 spec 输出 token IDs 与 non-spec baseline 完全一致。

不能用 `"word " * n` 猜 token 长度。使用 `prompt_token_ids=list[int]`，先由 target tokenizer 找到
一个非 special 的单 token，再精确重复。若验证 total KV 边界 127/128/129：

- DFlash q=9，构造 history 长度 118/119/120；
- DSpark q=7，构造 history 长度 120/121/122。

测试中要打印并断言实际 prompt/history 长度和预期 total KV，避免 tokenizer 或 BOS 行为改变边界。
边界用例也必须与 baseline token-identical，不能只断言输出非空。

#### 4.4 Acceptance 防退化

greedy token equality 即使所有 draft 都被 reject 也能通过，因此必须读取 metrics：

- `vllm:spec_decode_num_drafts > 0`；
- 总 accepted tokens > 0；
- **总 accepted tokens < 总 draft tokens**——4.3 要求"实际产生 rejection"，只排除全拒绝证明不了
  这一点：全接受同样会让前两条通过。两侧都卡住才说明 draft/verify 循环真的在做取舍；
- acceptance per position 可计算且不是全 0；
- 首次 310P bring-up 记录两种方法的 eager baseline；
- 复核稳定性后，把 `BASELINES_310P_EAGER` 固化到新测试，逐位置容差初始为 0.1。

现有 A2/A3 one-card baseline 只能作为参考，不能直接作为 310P pass/fail golden；现有测试是 TP=1、
dtype auto 和 graph 模式，也不能作为本期验收命令。ADN length/mask/RoPE 错误同样可能表现为低
acceptance，排障时不能先假定 attention 无关。

#### 4.5 E2E 命令

```bash
VLLM_USE_V2_MODEL_RUNNER=0 pytest -sv \
  tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_eager_310p.py
```

---

### Task 5：回归、文档和提交拆分

#### 5.1 Focused CPU UT

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/spec_decode/test_parallel_drafting_inputs_310p.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/test_qwen3_parallel_drafting_patches_310p.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/attention/test_adn_fused_infer_attention_310p.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/attention/test_parallel_draft_routing_310p.py
```

#### 5.2 Shared-path 回归

- [ ] 运行现有 310P attention metadata/splitfuse UT；
- [ ] 运行现有 DFlash/DSpark proposer UT；
- [ ] 运行现有 patch worker import UT；
- [ ] 验证非 310P dispatch 仍走 Triton；
- [ ] 验证 normal causal Qwen3-8B 310P eager E2E 未回归。

#### 5.3 代码质量

```bash
pre-commit run
```

记录所有真机命令、芯片、版本和结果。硬件未执行的项目不能写成 passed。

#### 5.4 建议提交边界

1. `[Test][310P] Add parallel draft hardware gates`
2. `[Feature][310P] Add Triton-free DFlash and DSpark input expansion`
3. `[Bugfix][310P] Fix Qwen3 parallel draft context RoPE`
4. `[Feature][310P] Route Qwen3 parallel draft attention to ADN`
5. `[Test][310P] Add Qwen3-8B eager parallel drafting E2E`

每个 commit 使用 `-s`，并按仓库要求添加 AI attribution。不要在中间提交里混入 graph、Qwen3.6
或通用 block-size 重构。

---

## 6. 排障顺序

| 现象 | 第一检查点 | 第二检查点 |
| --- | --- | --- |
| profile 阶段 RoPE shape 错，或起服后 draft 一直全 reject | `_profile_rope_context` 是否套住了 `is_profile` 分支里**两个** forward（context KV 和随后的 `self.model(...)`） | 每层是否只传 `num_ctx` positions |
| 只有第一次 draft 正常、之后异常 | flag 是否在 `finally` 里恢复成 `False` | 是否有另一处提前把 flag 关掉 |
| cache writer 正常但 ADN 数值错 | Phase 0 NZ readback、format 29、shape[-2]=128 | raw q-len、总 KV len、local heads |
| 输出像 causal attention | route 是否进入 ADN | `attn_mask` 是否确实为 `None`，future-token discriminator |
| 第二轮开始 slot 错 | rejected tail 与 `effective_seq_len` | 是否用了真实 BlockTable block size |
| DFlash 对、DSpark 错 | anchor sampling indices | q=7、mask token patch identity |
| token equality 通过但 acceptance 全 0 | input expansion/position/RoPE | ADN q/KV lengths 与 no-mask 语义 |
| 310P 报 GatherV2/advanced-index 不支持 | Phase 0 input-op smoke | 转小型 AscendC helper，不静默 CPU sync |
| 只有非 310P 回归 | device dispatch 或 fused RoPE 分支 | Triton fallback caller test |

不得用以下方式掩盖错误：

- non-causal 失败后调用 causal splitfuse；
- 把 raw q-len 改成 cumulative endpoints 迎合 shape；
- 把 NZ cache repack 成 ND 放入每层热路径；
- 把 FP16 改回 BF16；
- 开启 graph 绕过 eager bug；
- 降低边界/acceptance 检查到“输出非空”。

---

## 7. 最终完成标准

只有同时满足以下条件，Qwen3-8B 310P eager 适配才算完成：

- [ ] scope 仍严格限定于本文支持矩阵；
- [ ] `Ascend_Ops/tests/test_adn_fia.py` 通过（含 TND/no-mask/D128/B128 这组）；
- [ ] vLLM rank-4 NZ writer-to-ADN readback 通过 q=9、q=7、ragged、3 页和 causal discriminator；
- [ ] PyTorch NPU 输入展开 exact-op smoke 在 310P 通过；
- [ ] 五层逐层 context RoPE 生效；
- [ ] `_profile_rope_context` 覆盖 DFlash 与 DSpark `is_profile` 分支里的 context KV 与 query
      两个 forward，并有调用**真实** `dummy_run(is_profile=True)` 的 UT（context 与 query
      长度不同，能区分半修复）；
- [ ] 引擎能完成显存 profile run，且起服后 acceptance 不是全 0（这条单独列出，因为 profile 路径
      不经过 `_run_merged_draft`，Phase 0 的两个 smoke 都测不到它；且它的故障形态是"用错 cos/sin"
      而非崩溃，只看能否起服会漏掉）；
- [ ] Phase 0.1 的三个 model revision 已解析成 SHA 并写进 E2E 测试常量；
- [ ] focused CPU UT 全绿；
- [ ] 310P 路径没有 launch Triton；
- [ ] Qwen3-8B baseline、DFlash、DSpark 都使用 FP16、TP=2、block size 128、MRV1、eager；
- [ ] DFlash 与 DSpark 的普通、ragged、rejection、精确页边界输出均与 baseline token-identical；
- [ ] 两种方法 `num_drafts > 0`、accepted tokens > 0，且通过已校准的 310P eager acceptance baseline；
- [ ] normal causal 310P attention 和非 310P Triton path 无回归；
- [ ] PR 记录三个仓库 SHA、依赖/固件/镜像版本、模型 revisions、真机命令与误差结果；
- [ ] 没有加入 Qwen3.6、ACLGraph、MRV2、BF16、量化或通用多 group 改动。

Qwen3.6、图模式和性能优化在该 MVP 完成并稳定后另立计划，不作为本期隐藏验收项。
