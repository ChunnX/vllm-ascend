# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.triton.mamba.gdn_state_indices import fused_gdn_state_indices
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
