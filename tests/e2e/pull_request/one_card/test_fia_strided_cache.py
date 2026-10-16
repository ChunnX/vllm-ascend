# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu

import vllm_ascend.attention.attention_v1 as attn_module
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState, AttentionType
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph


@pytest.mark.parametrize("factor", [2, 32])
@pytest.mark.parametrize("causal", [False, True])
def test_fia_shared_pages_graph_replay(factor, causal, monkeypatch):
    """Exercise the backend, including runtime tables with changing addresses."""
    monkeypatch.setattr(attn_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    torch.manual_seed(42)
    blocks, block_size, head_size = 6, 128, 256
    page = block_size * head_size
    raw = torch.randn(blocks * factor * page, dtype=torch.bfloat16, device="npu")
    key = raw.as_strided((blocks, block_size, 1, head_size), (factor * page, head_size, head_size, 1))
    value = raw.as_strided(key.shape, key.stride(), storage_offset=page)
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 1, head_size
    impl.scale = head_size**-0.5
    impl.attn_type = AttentionType.DECODER
    impl.use_bnsd_kv_cache = False
    impl.key_cache, impl.value_cache = key, value
    impl.sliding_window = impl.sinks = None
    impl.enable_c8_quant = impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "layer"
    query = torch.randn(8, 6, head_size, dtype=key.dtype, device="npu")
    output = torch.empty_like(query)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device="npu"), diagonal=1)
    if not causal:
        mask = None
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding,
        actual_seq_lengths_q=[4, 8],
        seq_lens_list=[137, 191],
        block_tables=torch.tensor([[0, 3], [5, 1]], dtype=torch.int32, device="npu"),
        num_actual_tokens=8,
        causal=causal,
        attn_mask=mask,
    )
    impl.full_graph_fia(query, query, query, metadata, output)
    torch.npu.synchronize()
    stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    graph = UpdatableGraph()
    with torch.npu.graph(graph, stream=stream):
        impl.full_graph_fia(query, query, query, metadata, output)
    torch.npu.synchronize()
    for pages, lengths in [([[0, 3], [5, 1]], [137, 191]), ([[2, 4], [1, 0]], [191, 137])]:
        metadata.block_tables = torch.tensor(pages, dtype=torch.int32, device="npu")
        metadata.seq_lens_list = lengths
        query.add_(0.01)
        key[pages[0][0]].add_(0.125)
        torch.npu.synchronize()
        tasks = graph.resolve_tasks(ContextSource({"layer": metadata}))
        graph.replay()
        graph.update(update_stream, tasks)
        torch.npu.synchronize()
        reference, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key.contiguous().flatten(2),
            value=value.contiguous().flatten(2),
            block_table=metadata.block_tables,
            num_heads=6,
            num_key_value_heads=1,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=[4, 8],
            actual_seq_lengths_kv=lengths,
            scale=impl.scale,
            atten_mask=mask,
            sparse_mode=3 if causal else 0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )
        torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
