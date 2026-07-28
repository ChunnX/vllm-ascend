# Custom fused infer attention for Ascend 310P

This directory vendors the 310P `CustomFusedInferAttention` OPP from
`Ascend_Ops` commit `1d29869908ba0b4d0464bf5a9abd20b71366ac00` and exposes it
through the vllm-ascend `_C_ascend` extension. It replaces the separately built
`adn_custom_ops` PTA wheel; no `adn_custom_ops_lib` or TorchAir import is needed
for the eager vLLM path.

Upstream it is still named `AdnFusedInferAttention` / `adn_fused_infer_attention`
(`custom_opp/src/adn_opp/adn_fused_infer_attention`). Only the names were changed
on the way in -- `Adn` -> `Custom` throughout, including the generated
`aclnnCustomFusedInferAttention` symbol. When re-syncing, diff against the
upstream path and re-apply that substitution; the kernel bodies must stay
byte-identical apart from it.

## Layout

- `op_kernel/`: the AscendC entry, paged-attention implementation, and the exact
  utility headers used by the source kernel.
- `op_host/`: OpDef, shape/type inference, tiling, checks, and tiling-key
  registration.
- `custom_fused_infer_attention_torch_adpt.h`: ACLNN-to-Torch functional and
  caller-provided-output bindings.

The utility headers are intentionally operator-private. The source and
vllm-ascend copies of `mem.h` construct `AsdopsBuffer` differently; changing
the shared vllm-ascend headers would affect unrelated operators, while silently
using them would no longer reproduce the known source kernel.

## Torch APIs

The functional API preserves the source TensorList ABI:

```python
out = torch.ops._C_ascend.npu_custom_fused_infer_attention(
    query,
    [key_cache],
    [value_cache],
    attn_mask=None,
    actual_seq_lengths_q=q_lens,
    actual_seq_lengths_kv=kv_lens,
    block_table=block_table,
    num_heads=num_heads,
    scale_value=scale,
    input_layout="TND",
    num_key_value_heads=num_kv_heads,
    block_size=128,
    inner_precise=2,
)
```

The vLLM hot path uses the out variant, which avoids the old PTA allocation and
the subsequent Python `copy_`:

```python
torch.ops._C_ascend.npu_custom_fused_infer_attention_out(
    query,
    [key_cache],
    [value_cache],
    output,
    attn_mask=None,
    actual_seq_lengths_q=q_lens,
    actual_seq_lengths_kv=kv_lens,
    block_table=block_table,
    num_heads=num_heads,
    scale_value=scale,
    input_layout="TND",
    num_key_value_heads=num_kv_heads,
    block_size=128,
    inner_precise=2,
)
```

Call `vllm_ascend.utils.enable_custom_op()` before using either API directly.

## Supported production scope

The current DFlash/DSpark path is intentionally narrow:

- Ascend 310P, eager drafter execution;
- FP16 query/K/V/output;
- TND query and rank-4 FRACTAL_NZ paged K/V caches;
- one dynamic key tensor and one dynamic value tensor;
- raw per-request q lengths and total per-request KV lengths;
- INT32 block table, block size 128, `inner_precise=2`;
- `attn_mask=None`, which is full/non-causal attention.

The Python attention adapter still owns vLLM-specific metadata, cache-format,
scope, and ACLGraph-capture checks. Moving the OPP into `csrc` replaces the
external PTA wrapper; it does not make Python length lists graph-replay safe.

The kernel owns pipeline synchronization, so
`op_host/CMakeLists.txt` must retain `--cce-auto-sync=off`.
