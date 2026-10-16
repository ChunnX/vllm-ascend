# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.triton.mamba.gdn_state_indices import fused_gdn_spec_metadata, fused_gdn_state_indices
from vllm_ascend.ops.triton.mamba.state_index import (
    STATE_IO_BLOCK_SIZE,
    gather_ssm_states,
    scatter_ssm_states_,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Mamba state index tests require an Ascend NPU",
)


@pytest.mark.parametrize(
    "num_rows,num_active,graph_rows,num_slots",
    [(1, 1, 1, 8), (1, 1, 4, 8), (5, 3, 8, 8), (5, 0, 8, 8), (5, 5, 5, 1), (32, 19, 32, 8), (3, 2, 5, 3), (5, 0, 0, 8)],
)
@pytest.mark.parametrize("strided", [False, True])
def test_fused_gdn_state_indices(num_rows, num_active, graph_rows, num_slots, strided):
    """Match align-state selection across block boundaries and graph reuse."""
    block_size, num_columns = 64, 44
    storage = torch.arange(num_rows * num_columns * 2, dtype=torch.int32).view(num_rows, -1).to("npu")
    table = storage[:, ::2] if strided else storage[:, :num_columns].contiguous()
    lengths_cpu = torch.tensor(([0, 1, 64, 65, 128] * num_rows)[:num_rows], dtype=torch.int32)
    selected_cpu = torch.arange(num_rows - 1, -1, -1, dtype=torch.int64)[:num_active]
    lengths = torch.stack((lengths_cpu, lengths_cpu), 1).to("npu")[:, 0] if strided else lengths_cpu.to("npu")
    selected = torch.stack((selected_cpu, selected_cpu), 1).to("npu")[:, 0] if strided else selected_cpu.to("npu")
    output = torch.empty((graph_rows, num_slots), dtype=torch.int32, device="npu")
    address = output.data_ptr()
    for iteration in range(2):
        fused_gdn_state_indices(table, lengths, selected, output, block_size, NULL_BLOCK_ID)
        expected = torch.full((graph_rows, num_slots), NULL_BLOCK_ID, dtype=torch.int32)
        table_cpu = table.cpu()
        for row, source_row in enumerate(selected_cpu.tolist()):
            start = max((int(lengths_cpu[source_row]) - 1) // block_size, 0)
            expected[row] = table_cpu[source_row, start : start + num_slots]
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        assert output.data_ptr() == address
        if iteration == 0:
            lengths_cpu.fill_(130)
            lengths.fill_(130)
            table.add_(1)


def test_fused_gdn_state_indices_graph_replay():
    table = torch.arange(88, dtype=torch.int32, device="npu").view(2, 44)
    lengths = torch.tensor([64, 65], dtype=torch.int32, device="npu")
    selected = torch.tensor([1, 0], dtype=torch.int64, device="npu")
    output = torch.empty((4, 8), dtype=torch.int32, device="npu")
    fused_gdn_state_indices(table, lengths, selected, output, 64, NULL_BLOCK_ID)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream):
        fused_gdn_state_indices(table, lengths, selected, output, 64, NULL_BLOCK_ID)
    for length in (65, 128, 129):
        lengths.fill_(length)
        graph.replay()
        expected = torch.full((4, 8), NULL_BLOCK_ID, dtype=torch.int32)
        start = (length - 1) // 64
        expected[:2] = table.cpu()[[1, 0], start : start + 8]
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "live,graph_rows,slots", [(1, 1, 8), (1, 4, 8), (3, 8, 8), (0, 8, 8), (0, 0, 8), (19, 32, 8), (3, 5, 3)]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
def test_fused_gdn_spec_metadata(live, graph_rows, slots, strided, input_dtype):
    """All outputs retain padding semantics when device inputs change in place."""
    table_storage = torch.arange(max(1, live) * 88, dtype=torch.int32).view(-1, 88).to("npu")
    table = table_storage[:live, ::2] if strided else table_storage[:live, :44].contiguous()

    def to_device(values):
        return torch.stack((values, values), 1).to("npu")[:, 0] if strided else values.to("npu")

    seq = to_device(torch.arange(live, dtype=torch.int32) * 64)
    query_lengths = torch.tensor(([slots, 1, 0] * live)[:live], dtype=torch.int32)
    query_cpu = torch.cat((torch.zeros(1, dtype=torch.int32), query_lengths.cumsum(0).to(torch.int32)))
    query = to_device(query_cpu.to(input_dtype))
    accepted = to_device(torch.arange(live, dtype=input_dtype))
    tokens = int(query_cpu[-1])
    outputs = (
        torch.empty((graph_rows, slots), dtype=torch.int32, device="npu"),
        torch.empty(graph_rows + 1, dtype=torch.int32, device="npu"),
        torch.empty(graph_rows, dtype=torch.int32, device="npu"),
        torch.empty(graph_rows, dtype=torch.bool, device="npu"),
        torch.empty(graph_rows + 1, dtype=torch.int32, device="npu"),
        torch.empty(tokens, dtype=torch.int32, device="npu"),
    )
    addresses = [x.data_ptr() for x in outputs]
    fused_gdn_spec_metadata(table, seq, query, accepted, outputs, live, tokens, 64, NULL_BLOCK_ID)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream()):
        fused_gdn_spec_metadata(table, seq, query, accepted, outputs, live, tokens, 64, NULL_BLOCK_ID)
    for iteration in range(2):
        graph.replay()
        states = torch.full((graph_rows, slots), NULL_BLOCK_ID, dtype=torch.int32)
        for row, length in enumerate(seq.cpu().tolist()):
            start = max((length - 1) // 64, 0)
            states[row] = table.cpu()[row, start : start + slots]
        boundaries = torch.full((graph_rows + 1,), int(query_cpu[-1]), dtype=torch.int32)
        boundaries[: live + 1] = query_cpu
        counts = torch.ones(graph_rows, dtype=torch.int32)
        counts[:live] = accepted.cpu().to(torch.int32)
        masks = torch.arange(graph_rows) < live
        lengths = torch.cat((boundaries[:1], torch.diff(boundaries)))
        expected = (states, boundaries, counts, masks, lengths, torch.arange(tokens, dtype=torch.int32))
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        assert [x.data_ptr() for x in outputs] == addresses
        if iteration == 0:
            seq.fill_(130)
            accepted.add_(1)
            query_cpu[1:] = query_lengths.flip(0).cumsum(0)
            query.copy_(query_cpu)
            table.add_(1)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("live,graph_rows", [(1, 1), (1, 4), (3, 4)])
def test_fused_gdn_spec_metadata_builder(adaptive, live, graph_rows):
    """Compare the real NPU builder against its original full metadata path."""
    from dataclasses import fields

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.kv_cache_interface import MambaSpec

    import vllm_ascend.ops.gdn_attn_builder as module
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    spec = SimpleNamespace(
        method="dspark", num_speculative_tokens=7, enable_adaptive_verification=adaptive, parallel_drafting=False
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048, get_num_attention_heads=lambda _: 32),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        speculative_config=spec,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, max_cudagraph_capture_size=None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=2048),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1, tensor_parallel_size=1
        ),
        additional_config=None,
        lora_config=None,
    )
    kv = MambaSpec(
        block_size=64, shapes=((1,), (1,)), dtypes=(torch.float32,), mamba_cache_mode="align", num_speculative_blocks=7
    )
    builders = [
        module.AscendGDNAttentionMetadataBuilder(kv, [f"layer{i}"], config, torch.device("npu")) for i in range(3)
    ]
    for length in (64, 65, 128, 129):
        query_lens = torch.tensor(
            ([8, 3, 1] if adaptive else [8] * 3)[:live] + [0] * (graph_rows - live), dtype=torch.int32
        )
        q_cpu = torch.cat((torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0).to(torch.int32)))
        seq_cpu = torch.tensor([length] * live + [0] * (graph_rows - live), dtype=torch.int32)
        metadata = AscendCommonAttentionMetadata(
            num_reqs=graph_rows,
            num_actual_tokens=int(q_cpu[-1]),
            max_query_len=8,
            max_seq_len=length,
            query_start_loc=q_cpu.to("npu"),
            query_start_loc_cpu=q_cpu,
            seq_lens=seq_cpu.to("npu"),
            seq_lens_cpu=seq_cpu,
            seq_lens_cpu_upper_bound=seq_cpu,
            block_table_tensor=torch.arange(graph_rows * 44, dtype=torch.int32, device="npu").view(graph_rows, 44),
            slot_mapping=torch.arange(int(q_cpu[-1]), dtype=torch.int64, device="npu"),
            is_prefilling=torch.zeros(graph_rows, dtype=torch.bool),
            num_computed_tokens_cpu=seq_cpu - query_lens,
            causal=True,
        )
        kwargs = dict(
            num_accepted_tokens=torch.arange(graph_rows, dtype=torch.int32, device="npu") % 8 + 1,
            num_decode_draft_tokens_cpu=torch.tensor([7] * live + [-1] * (graph_rows - live), dtype=torch.int32),
            num_actual_reqs=live,
        )
        cache = {}
        for group, builder in enumerate(builders):
            metadata.block_table_tensor.add_(group * 100)
            with patch.object(builder, "_try_fused_spec_metadata", return_value=None):
                reference = builder.build(0, metadata, **kwargs)
                expected = {
                    f.name: getattr(reference, f.name).clone()
                    for f in fields(reference)
                    if isinstance(getattr(reference, f.name), torch.Tensor)
                }
                expected_lengths = reference.spec_decode_metadata.actual_seq_lengths.clone()
            with (
                patch.object(builder, "_get_shared_batch_plan", side_effect=AssertionError("general plan")),
                patch.object(builder, "_pad_spec_decode_metadata", side_effect=AssertionError("separate padding")),
                patch.object(module, "fused_gdn_spec_metadata", wraps=module.fused_gdn_spec_metadata) as fused,
            ):
                actual = builder.build(0, metadata, batch_shared_cache=cache, **kwargs)
            assert fused.call_count == 1
            for name, value in expected.items():
                torch.testing.assert_close(getattr(actual, name).cpu(), value.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(actual.spec_decode_metadata.actual_seq_lengths.cpu(), expected_lengths.cpu())
            assert actual.spec_query_start_loc.data_ptr() == builder.spec_query_start_loc.data_ptr()
            assert actual.num_accepted_tokens.data_ptr() == builder.num_accepted_tokens.data_ptr()


@pytest.mark.parametrize(
    "case",
    [
        "mixed",
        "hole",
        "zero_drafts",
        "long_query",
        "eager",
        "cache_mode",
        "310p",
        "capacity",
        "device_control",
        "device_query",
    ],
)
def test_fused_gdn_spec_metadata_declines_unsupported_batches(case):
    import vllm_ascend.ops.gdn_attn_builder as module

    builder = object.__new__(module.AscendGDNAttentionMetadataBuilder)
    builder.use_full_cuda_graph = builder.use_spec_decode = True
    builder.ragged_spec_decode = builder.adaptive_verification_enabled = False
    builder.num_spec, builder.decode_cudagraph_max_bs = 7, 64
    builder.kv_cache_spec = SimpleNamespace(num_speculative_blocks=7)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    query = torch.tensor([0, 8, 16, 16], dtype=torch.int32)
    drafts = torch.tensor([7, 7, -1], dtype=torch.int32)
    if case == "mixed":
        query[-1] = 17
    elif case == "hole":
        drafts = torch.tensor([7, -1, 7], dtype=torch.int32)
    elif case == "zero_drafts":
        drafts[:2] = 0
    elif case == "long_query":
        query[1] = 9
    elif case == "eager":
        builder.use_full_cuda_graph = False
    elif case == "cache_mode":
        builder.vllm_config.cache_config.mamba_cache_mode = "none"
    elif case == "310p":
        builder._USE_COMMON_KERNEL_METADATA = False
    elif case == "capacity":
        builder.decode_cudagraph_max_bs = 1
    elif case == "device_control":
        drafts = drafts.to("npu")
    metadata = SimpleNamespace(
        block_table_tensor=torch.zeros((3, 44), dtype=torch.int32, device="npu"),
        seq_lens=torch.full((3,), 64, dtype=torch.int32, device="npu"),
        query_start_loc=query.to("npu"),
        query_start_loc_cpu=query,
        num_reqs=3,
        num_actual_tokens=int(query[-1]),
    )
    if case == "device_query":
        metadata.query_start_loc_cpu = metadata.query_start_loc
    with patch.object(module, "fused_gdn_spec_metadata") as fused:
        assert (
            builder._try_fused_spec_metadata(metadata, torch.ones(3, dtype=torch.int32, device="npu"), drafts, 3, {})
            is None
        )
    fused.assert_not_called()


@pytest.mark.parametrize("padding", [0, 13])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "state_shape",
    [
        (1, 1, STATE_IO_BLOCK_SIZE - 1),
        (1, 1, STATE_IO_BLOCK_SIZE),
        (1, 1, STATE_IO_BLOCK_SIZE + 1),
        (8, 128, 128),
    ],
)
def test_gather_and_scatter_ssm_states(
    padding: int,
    index_dtype: torch.dtype,
    state_shape: tuple[int, ...],
) -> None:
    num_states = 5
    row_size = math.prod(state_shape)
    state_stride = row_size + padding
    storage_size = (num_states - 1) * state_stride + row_size
    inner_strides = []
    running_stride = 1
    for dim in reversed(state_shape):
        inner_strides.append(running_stride)
        running_stride *= dim
    inner_strides.reverse()

    storage = torch.full(
        (storage_size,),
        -1.0,
        dtype=torch.float32,
        device="npu",
    )
    state = torch.as_strided(
        storage,
        size=(num_states, *state_shape),
        stride=(state_stride, *inner_strides),
    )
    for state_idx in range(num_states):
        state[state_idx].fill_(state_idx + 0.25)

    indices = torch.tensor([3, 0, 2], dtype=index_dtype, device="npu")
    has_initial_state = torch.tensor([True, False, True], device="npu")

    gathered = gather_ssm_states(
        state,
        indices,
        has_initial_state,
        output_dtype=torch.bfloat16,
    )
    expected_gather = torch.stack([state[3], torch.zeros_like(state[0]), state[2]]).to(torch.bfloat16)
    torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)

    source = (
        torch.arange(
            indices.numel() * row_size,
            dtype=torch.float32,
            device="npu",
        )
        .reshape(indices.numel(), *state_shape)
        .to(torch.bfloat16)
    )
    scatter_ssm_states_(state, indices, source)

    for source_idx, state_idx in enumerate((3, 0, 2)):
        torch.testing.assert_close(
            state[state_idx],
            source[source_idx].to(state.dtype),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        state[1],
        torch.full_like(state[1], 1.25),
        rtol=0,
        atol=0,
    )

    if padding:
        storage_cpu = storage.cpu()
        for state_idx in range(num_states - 1):
            padding_start = state_idx * state_stride + row_size
            padding_end = (state_idx + 1) * state_stride
            torch.testing.assert_close(
                storage_cpu[padding_start:padding_end],
                torch.full((padding,), -1.0),
                rtol=0,
                atol=0,
            )


def test_rejects_empty_state_cache() -> None:
    state = torch.empty((0, 8), dtype=torch.float32, device="npu")
    indices = torch.empty((0,), dtype=torch.int32, device="npu")
    has_initial_state = torch.empty((0,), dtype=torch.bool, device="npu")

    with pytest.raises(ValueError, match="at least one row"):
        gather_ssm_states(state, indices, has_initial_state)


def test_rejects_overlapping_state_rows() -> None:
    row_size = 8
    storage = torch.zeros((2 * row_size,), dtype=torch.float32, device="npu")
    state = torch.as_strided(
        storage,
        size=(2, row_size),
        stride=(row_size - 1, 1),
    )
    indices = torch.tensor([0], dtype=torch.int32, device="npu")
    has_initial_state = torch.tensor([True], device="npu")

    with pytest.raises(ValueError, match=r"stride\(0\)"):
        gather_ssm_states(state, indices, has_initial_state)
