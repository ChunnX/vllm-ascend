# 交付件 2：仅在 DSpark 草稿模型（GQA / 非因果 / headdim128）的 attention 中使能 `npu_fused_infer_attention_sink`

## 1. 背景与根因

在 model runner v2 的 DSpark 投机推理里，草稿模型（draft）在 t-1 步并行产出一批草稿 token，
t 步由主模型校验并拒绝采样。每个请求被拒绝的 token 数 `num_rejected` **只存在于 device 侧**。

草稿模型准备输入的 Triton kernel（`_prepare_dflash_inputs_kernel_ascend`，
`vllm_ascend/worker/v2/spec_decode/dflash/speculator.py`）在 **device 上**写入草稿侧的
`input_buffers.seq_lens`：

```
tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
```

其中 `last_valid_pos` 依赖 `num_rejected`。随后
`DraftModelSpeculator._build_draft_attn_metadata` 把这个 device tensor 直接塞进
`AscendCommonAttentionMetadata.seq_lens`（`vllm_ascend/worker/v2/attn_utils.py:193`）。

`AscendAttentionMetadataBuilder.build`（`vllm_ascend/attention/attention_v1.py:291`）里，
对于 `speculative_config.parallel_drafting`（DSpark/DFlash）分支：

```python
# attention_v1.py:321-322
elif self.speculative_config and self.speculative_config.parallel_drafting:
    seq_lens = common_attn_metadata.seq_lens      # 这是 device tensor
...
# attention_v1.py:333
seq_lens_list = seq_lens.tolist()                  # ← 强制 device→host 同步，性能损耗点
```

同函数里还有 `query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)`（:330）与
`actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()`（:332）。常规 FIA V1 算子
（`npu_fused_infer_attention_score`）要求 `actual_seq_lengths(_kv)` 是 **host list**，
所以必须 `tolist()`。

**根因**：不是 `tolist()` 本身慢，而是它在投机热路径上每步都把依赖 `num_rejected` 的
device seq_lens 拉到 host，阻断异步调度。

## 2. 为什么用 `npu_fused_infer_attention_sink`

`sink` 算子与普通 FIA V1 的关键区别：

| | 普通 FIA V1 | Sink 算子 |
|---|---|---|
| `actual_seq_qlen/kvlen` | host list（要 `tolist()`） | **device int64 tensor** |
| tiling | host 侧 | **AICPU 侧**（`_npu_fused_infer_attention_sink_metadata` 提前算好 `meta_data`） |
| aclgraph 入图 | 支持 | 支持（文档明确「支持图模式」） |
| 分页 KV | 支持 | 支持（`block_table` + `block_size`，BBH/BNBD/NZ 三种布局） |
| GQA / 非因果 / D=128 | 支持 | 支持（`num_key_value_heads`、`sparse_mode=0`、D≤512） |

于是把 **device 侧 `seq_lens` / `query_start_loc` 直接传给算子**，tiling 在 AICPU 完成，
`build()` 里这条 `tolist()` 同步即可被消除。

### 关键入参形态（DSpark 草稿）

- `input_layout = "TND"`；`query` 形状 `(T, num_heads, 128)`。
- `key/value`：分页 KV 重排为 **BBH** `(num_block, block_size, num_kv_heads*128)`
  （与现有 `_get_fia_params` 的 `key_cache.view(num_block, block_size, -1)` 一致）。
- `sparse_mode = 0`（非因果，不传 `atten_mask`）。
- `num_query_heads` / `num_key_value_heads`（GQA）。
- `actual_seq_qlen`：TND 累加值，device int64，来自 `query_start_loc[1:]`。
- `actual_seq_kvlen`：分页场景逐 batch 真实长度，device int64，来自 `seq_lens`。
- `meta_data`：`_npu_fused_infer_attention_sink_metadata` 的产出（shape `(1024,)`，int32）。
- `sink_number = 0`（DSpark 与「sink token」无关，这里只用其 AICPU tiling 能力）。

### omniinfer 参考实现（使能范式）

`/opt/zsy/omniinfer/omni/attention/backends/attention.py`：

- `_init_cross_layer_shared_ops()`（:305-313）用 `CrossLayerSharedOp` 持有一个
  **地址稳定的** `(1024,) int32` 持久 buffer，包装
  `torch.ops.custom._npu_fused_infer_attention_sink_metadata`。
- eager 分支（:445-477）：`use_aicpu_fa_tiling` 时，把 `actual_seq_qlen/kvlen` 转 int64，
  用 `get_stream_limit` 取 cube/vector core 数，调 metadata op 生成 `meta_data`，再调
  `torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]`。
- 图捕获分支（:429-443）：走 `capture_graph_task(OP_FIA_SINK, ...)`，
  `OP_FIA_SINK` 定义在 `/opt/zsy/omniinfer/omni/compilation/utils.py:142-155`，
  `op_out_fn=torch.ops.custom.npu_fused_infer_attention_sink.out`、
  `workspace_fn=torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace`。
  （这两个 `.out` / `_get_max_workspace` 变体由 PTA 的 aclnn 适配层自动生成，
  对应 `aclnnAiInfraFusedInferAttentionSink[V3]GetMaxWorkspaceSize`。）

---

## 3. 使能方案（`vllm_ascend/attention/attention_v1.py`）

### 3.1 判定条件：只在 DSpark 草稿 attention 生效

在 `AscendAttentionMetadataBuilder.build()` 里可稳定拿到：
- `self.speculative_config.parallel_drafting`（DSpark/DFlash 草稿的标志）；
- `common_attn_metadata.causal == False`（草稿非因果，普通解码/prefill 为 True）；
- `self.kv_cache_spec.head_size`（草稿 head_dim=128）；
- `self.kv_cache_spec.num_kv_heads`（GQA 用）。

建议在 `AscendMetadata` 上增加一个布尔字段，由 builder 一次性判定，impl 侧再加形状兜底：

```python
# AscendMetadata 新增字段
use_fia_sink: bool = False

# AscendAttentionMetadataBuilder.build() 末尾、构造 metadata 之前
is_draft_gqa_noncausal = (
    self.speculative_config is not None
    and self.speculative_config.parallel_drafting
    and not common_attn_metadata.causal
    and getattr(self.kv_cache_spec, "head_size", None) == 128
)
```

impl 侧（`AscendAttentionBackendImpl.forward_impl` 或 `forward_fused_infer_attention`）再加
GQA + 形状兜底：

```python
use_sink = (
    getattr(attn_metadata, "use_fia_sink", False)
    and self.head_size == 128
    and self.num_heads != self.num_kv_heads
    and self.num_heads % self.num_kv_heads == 0
    and self.sliding_window is None
    and self.sinks is None
)
```

> 双保险理由：builder 侧的 `parallel_drafting + 非因果` 精确定位「草稿」；
> impl 侧的 `headdim128 + GQA + 无 SWA/sink` 保证只对已知良好 shape 走 sink 算子，
> 其余路径保持现状，风险最小化。

### 3.2 `build()` 里消除同步

当命中 sink 路径时，跳过 host 侧序列长度物化，改存 device tensor：

```python
if use_sink:
    # 不 tolist、不 pin_memory().to()，直接用 device tensor
    seq_lens = common_attn_metadata.seq_lens[:num_reqs].to(torch.int64)
    query_start_loc = common_attn_metadata.query_start_loc[:num_reqs + 1].to(torch.int64)
    actual_seq_lengths_q = query_start_loc[1:]          # 累加值，device int64
    seq_lens_list = None                                 # 不再需要 host list
    ...
else:
    # 现有逻辑保持不变（tolist 等）
```

`AscendMetadata` 已有 `seq_lens`（device）与 `query_start_loc`（device）字段，
把 device tensor 存进去即可；sink 分支的 `seq_lens_list` / `actual_seq_lengths_q` 置 None，
并保证只有 sink 分支去读它们（见 3.3/3.4）。

注意：`build()` 里 `common_attn_metadata.seq_lens` 在 DSpark 下就是
`self.input_buffers.seq_lens[:num_reqs]`（device int32），`query_start_loc` 同理，
地址跨 step 稳定（预分配 max-size buffer），这是后面 aclgraph 能直接引用的前提。

### 3.3 eager 路径：`forward_fused_infer_attention`

在 `forward_fused_infer_attention` 里、`_EXTRA_CTX.capturing` 为 False 时，新增 sink 分支：

```python
if getattr(attn_metadata, "use_fia_sink", False):
    key, value, block_size, block_table, _ = self._get_fia_params(
        key, value, attn_metadata, kv_cache
    )
    num_tokens = int(attn_metadata.query_start_loc[-1].item())  # 或 attn_metadata.num_actual_tokens
    query = query[:num_tokens]

    actual_seq_qlen = attn_metadata.query_start_loc[1:].to(torch.int64)
    actual_seq_kvlen = attn_metadata.seq_lens.to(torch.int64)

    stream_limit = torch.npu.get_stream_limit(torch.npu.current_stream())
    meta_data = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
        self.num_heads, self.num_kv_heads, self.head_size, self.head_size,
        actual_seq_lengths=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        batch_size=len(actual_seq_kvlen),
        sparse_mode=0,                # 非因果
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="TND",
        input_layout_kv="BnBsH",      # (blocknum, blocksize, H) == BBH
        sink_num=0,
        block_size=block_size,
        aic_core_num=stream_limit["cube_core_num"],
        aiv_core_num=stream_limit["vector_core_num"],
    )
    attn_output, _ = torch.ops.custom.npu_fused_infer_attention_sink(
        query, key, value,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        block_table=block_table,
        num_query_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        softmax_scale=self.scale,
        input_layout="TND",
        sparse_mode=0,
        block_size=block_size,
        sink_number=0,
        meta_data=meta_data,
    )
    attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
    output[:num_tokens] = attn_output[:num_tokens]
    return output
```

要点：
- `key/value` 沿用 `_get_fia_params` 的 BBH reshape（`attn_state` 为 None 时走 else 分支）。
- `num_tokens` 若担心 `.item()` 同步，可用 `attn_metadata.num_actual_tokens` 或
  `attn_metadata.query_start_loc` 的形状信息（草稿下 `num_tokens = num_reqs * num_query_per_req`）。
- 该分支完全不含 `seq_lens.tolist()` / `actual_seq_lengths_q`（host list）。

### 3.4 aclgraph 路径：新增 `full_graph_fia_sink` + 扩展 `update_graph_params`

DSpark 草稿在 v2 走 `FULL_DECODE_ONLY` aclgraph（继承 DFlash 的 `DFlashCudaGraphManager`，
Ascend 侧 patch 成 `DFlashAclGraphManager`，`vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py`）。
草稿 attention 目前经 `forward_fused_infer_attention` 的 `_EXTRA_CTX.capturing` 分支进入
`full_graph_fia`（`self.sinks is None` 时）。需要给草稿加一条 sink 捕获分支。

设计要点（对照 `full_graph_fia` / `full_graph_fia_v2`）：

1. **持久 `meta_data` buffer**：在 impl `__init__` 里分配
   `self._fia_sink_meta = torch.empty((1024,), dtype=torch.int32, device="npu")`，
   地址跨 capture/replay 稳定（不必跨层共享，draft 层数少；若要省内存可仿 omniinfer
   `CrossLayerSharedOp` 共享）。

2. **捕获期**（`_EXTRA_CTX.capturing == True`）新增 `full_graph_fia_sink`：
   - workspace：用 `torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace(...)`
     查询，并按 `num_tokens` 缓存进 `graph_params.workspaces`（草稿用 `get_draft_graph_params()`）。
   - 把 sink 算子用 `torch.npu.graph_task_group_begin/end` 包起来捕获，输出写
     `[output, softmax_lse]`，`meta_data=self._fia_sink_meta`、device seq tensors 直接引用
     `attn_metadata.query_start_loc` / `attn_metadata.seq_lens`（地址稳定）。
   - `attn_params` 里记录：query/key/value/block_table/seq_lens/query_start_loc/output/softmax_lse/
     num_kv_heads/num_heads/scale/block_size/meta buffer，供 replay 更新。

3. **replay 更新**（`update_graph_params` 里新增 sink 分支，参考现有 FIA 分支）：
   草稿 replay 时 `update_full_graph_params` → `update_graph_params`。sink 分支在
   `graph_task_update_begin/end` 内：
   - 先重算 metadata：把更新后的 device seq tensors 传入
     `torch.ops.custom._npu_fused_infer_attention_sink_metadata`，结果 `copy_` 进
     `self._fia_sink_meta`（AICPU，需放到 AICPU 流并用 event 与 aicore 流同步，见文档「图模式调用」示例）；
   - 再 `torch.ops.custom.npu_fused_infer_attention_sink.out(..., meta_data=self._fia_sink_meta, out=[...])`。

   由于 `meta_data`、seq tensors、block_table、output 地址都稳定，捕获进图的 sink 算子
   replay 时读到的就是本步更新后的值，全程无 host 同步。

4. **更简单备选（推荐先落地）**：metadata 不在图内捕获，而是 replay **前**在
   `run_fullgraph`（`DFlashAclGraphManager.run_fullgraph`，`build_draft_attn_metadatas` 之后、
   `super().run_fullgraph(desc)` 之前）eager 重算一次 `meta_data` 进持久 buffer，
   图内只捕获 sink 算子（读 buffer + 稳定地址的 device tensors）。这避开了 AICPU 算子入图
   的复杂性，代价是 metadata 每次 replay 前多一次 AICPU 调用（这正是我们想要的 AICPU tiling，
   性能远好于 host 同步）。

> 推荐：先按「备选」落地打通 eager + aclgraph（图内只含 aicore sink 算子，metadata 图外重算）；
> 验证无误后再考虑把 metadata 也 `graph_task_update` 进图，进一步省 AICPU 调度开销。

### 3.5 需要改动的文件清单

| 文件 | 改动 |
|---|---|
| `vllm_ascend/attention/attention_v1.py` | `AscendMetadata` 加 `use_fia_sink`；`build()` 判定并跳过 tolist；`forward_fused_infer_attention` 加 eager sink 分支；`full_graph_fia_sink` 捕获；`update_graph_params` 加 sink replay 分支 |
| `vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py` | （若走「备选」）`run_fullgraph` 里 replay 前重算 metadata |
| `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` | 无需改动（继承 DFlash 的 aclgraph 与 metadata 构建） |
| （可选）`vllm_ascend/envs.py` | 新增 `VLLM_ASCEND_*` 开关，默认关，便于灰度（按 AGENTS.md 要求集中登记 env） |

---

## 4. aclgraph 适配的关键约束与坑

1. **地址稳定性**：图捕获会把 tensor 地址固化。`meta_data` buffer 必须是持久成员变量；
   `query_start_loc`/`seq_lens` 必须用 `input_buffers` 里预分配的 max-size buffer
   （DSpark 已满足），不能在 replay 期间重分配。block_table 在 replay 更新时用
   `attn_metadata[draft_step][key].block_tables`（现有草稿更新逻辑已处理）。

2. **AICPU 流同步**：metadata 算子跑在 AICPU，主算子跑在 AICORE。图外重算时用
   `torch.npu.Stream()` + `Event`（参考 omni-ops sink doc「图模式调用」示例里的
   `aicpu_stream` / `aicore_stream.wait_event`）。

3. **workspace**：图捕获需要 `.out` + `_get_max_workspace` 变体；eager 不需要 workspace。
   若 `_npu_fused_infer_attention_sink_get_max_workspace` 未随 PTA 生成，需在
   `torch_ops_extension` 里确认该 aclnn 适配是否开启（`aclnnAiInfraFusedInferAttentionSinkV3GetMaxWorkspaceSize`
   已存在）。

4. **布局字符串**：`input_layout="TND"`；`input_layout_kv` 对 BBH 用 `"BnBsH"`
   （`(blocknum, blocksize, H)`），与 omniinfer sink 路径一致；落地前用单算子脚本核对
   AICPU 对该字符串的实际解析（见交付件 1 的验证脚本）。

5. **dtype**：sink/metadata 的 seq len 入参要求 `int64`；`input_buffers.seq_lens`/
   `query_start_loc` 默认 int32，必须 `.to(torch.int64)`（在 device 上做，无同步）。

6. **判定边界**：`parallel_drafting + 非因果` 也覆盖 DFlash；若只想 DSpark，可进一步用
   `speculative_config.method == "mtp"` 或 speculator 类型收窄（DSpark 走 `method=="mtp"`
   的配置）。当前双保险（builder 判定 + impl 侧 `headdim128/GQA`）已足够安全。

---

## 5. 验证计划

1. **单算子正确性**：跑交付件 1 的脚本，与 `torch.nn.functional.scaled_dot_product_attention`
   （非因果）比对误差 < 1e-2（fp16）。
2. **端到端正确性**：开启 DSpark（`method="mtp"` + draft 模型），对比使能前后生成结果一致；
   用现有 `tests/e2e/` 下 speculative decoding 用例回归。
3. **性能**：用 profile 确认 `build()` 里 `seq_lens.tolist()` 的 host 同步消失，
   AsyncScheduler 不再在该点阻塞；对比每步 draft 时延。
4. **aclgraph**：`FULL_DECODE_ONLY` 下确认草稿图能捕获、replay 正确（debug 日志核对输入地址一致）。

---

## 6. 参考

- 现有 FIA 实现与图捕获：`vllm_ascend/attention/attention_v1.py`（`full_graph_fia` :849、
  `full_graph_fia_v2` :1020、`update_graph_params` :475、`_get_fia_params` :1212）
- 草稿 aclgraph 管理：`vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py`
- 草稿 metadata 构建：`vllm_ascend/worker/v2/attn_utils.py`（`build_attn_metadata` / `build_attn_metadata_wrapper`）
- 上游 DSpark/DFlash 草稿：`/opt/zsy/vllm-0.26.0/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`、
  `.../dflash/speculator.py`（`_build_draft_attn_metadata`、`_prepare_dflash_inputs_kernel` 写 device seq_lens）
- omniinfer 使能范式：`/opt/zsy/omniinfer/omni/attention/backends/attention.py`、
  `/opt/zsy/omniinfer/omni/compilation/utils.py`（`OP_FIA_SINK`、`capture_graph_task`）、
  `/opt/zsy/omniinfer/omni/attention/backends/utils.py`（`CrossLayerSharedOp`）
- 算子接口文档：`/opt/zsy/omni-ops/inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink/docs/`、
  `.../ai_infra_fused_infer_attention_sink_metadata/docs/`
