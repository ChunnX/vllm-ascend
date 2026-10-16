# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton

_STATE_INDEX_BLOCK_SIZE = 128


@triton.jit
def _gdn_state_indices_kernel(
    block_table,
    seq_lens,
    sequence_indices,
    output,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    NUM_ACTIVE: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // NUM_SLOTS
    slots = offsets % NUM_SLOTS
    if NUM_ACTIVE > 0:
        active = rows < NUM_ACTIVE
        source_rows = tl.load(sequence_indices + rows * INDEX_STRIDE, active, 0).to(tl.int32)
        lengths = tl.load(seq_lens + source_rows * SEQ_STRIDE, active, 0)
        start_cols = tl.maximum(lengths - 1, 0) // MAMBA_BLOCK_SIZE
        values = tl.load(
            block_table + source_rows * TABLE_ROW_STRIDE + (start_cols + slots) * TABLE_COL_STRIDE,
            active,
            PAD_SLOT_ID,
        )
    else:
        # Avoid even masked loads from empty index tensors on Ascend.
        values = tl.full((BLOCK_SIZE,), PAD_SLOT_ID, tl.int32)
    tl.store(output + offsets, values, offsets < NUM_ROWS * NUM_SLOTS)


def fused_gdn_state_indices(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sequence_indices: torch.Tensor,
    output: torch.Tensor,
    block_size: int,
    padding_value: int,
) -> torch.Tensor:
    """Gather align-mode speculative states into a stable, padded graph buffer.

    The scheduler must provide valid sequence rows and enough table columns
    for each selected sequence's final block plus the speculative state slots.
    No device values are read back to the host to check this contract.
    """
    assert block_table.ndim == output.ndim == 2
    assert seq_lens.ndim == sequence_indices.ndim == 1
    assert block_table.dtype == output.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert sequence_indices.dtype in (torch.int32, torch.int64)
    assert block_table.device == seq_lens.device == sequence_indices.device == output.device
    assert output.is_contiguous()
    assert sequence_indices.numel() <= output.shape[0]
    assert output.shape[1] > 0 and block_size > 0
    if output.numel():
        _gdn_state_indices_kernel[(triton.cdiv(output.numel(), _STATE_INDEX_BLOCK_SIZE),)](
            block_table,
            seq_lens,
            sequence_indices,
            output,
            block_table.stride(0),
            block_table.stride(1),
            seq_lens.stride(0),
            sequence_indices.stride(0),
            sequence_indices.numel(),
            output.shape[0],
            output.shape[1],
            block_size,
            padding_value,
            _STATE_INDEX_BLOCK_SIZE,
        )
    return output


@triton.jit
def _gdn_spec_metadata_kernel(
    block_table,
    seq_lens,
    query_start_loc,
    accepted_tokens,
    states_out,
    query_out,
    accepted_out,
    masks_out,
    lengths_out,
    tokens_out,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    QUERY_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    NUM_ACTIVE: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // NUM_SLOTS
    slots = offsets % NUM_SLOTS
    if NUM_ACTIVE > 0:
        lengths = tl.load(seq_lens + rows * SEQ_STRIDE, rows < NUM_ACTIVE, 0)
        start_cols = tl.maximum(lengths - 1, 0) // MAMBA_BLOCK_SIZE
        states = tl.load(
            block_table + rows * TABLE_ROW_STRIDE + (start_cols + slots) * TABLE_COL_STRIDE,
            rows < NUM_ACTIVE,
            PAD_SLOT_ID,
        )
        accepted = tl.load(accepted_tokens + offsets * ACCEPTED_STRIDE, offsets < NUM_ACTIVE, 1)
    else:
        states = tl.full((BLOCK_SIZE,), PAD_SLOT_ID, tl.int32)
        accepted = tl.full((BLOCK_SIZE,), 1, tl.int32)
    if NUM_ROWS > 0:
        tl.store(states_out + offsets, states, offsets < NUM_ROWS * NUM_SLOTS)
        tl.store(accepted_out + offsets, accepted, offsets < NUM_ROWS)
        tl.store(masks_out + offsets, offsets < NUM_ACTIVE, offsets < NUM_ROWS)

    # Dummy requests repeat the final boundary and have zero recurrent length.
    query_indices = tl.minimum(offsets, NUM_ACTIVE)
    query = tl.load(query_start_loc + query_indices * QUERY_STRIDE, offsets <= NUM_ROWS, 0)
    previous_indices = tl.maximum(query_indices - 1, 0)
    previous = tl.load(
        query_start_loc + previous_indices * QUERY_STRIDE,
        (offsets > 0) & (offsets <= NUM_ACTIVE),
        0,
    )
    actual_lengths = tl.where(offsets == 0, query, tl.where(offsets <= NUM_ACTIVE, query - previous, 0))
    tl.store(query_out + offsets, query, offsets <= NUM_ROWS)
    tl.store(lengths_out + offsets, actual_lengths, offsets <= NUM_ROWS)
    if NUM_TOKENS > 0:
        tl.store(tokens_out + offsets, offsets, offsets < NUM_TOKENS)


def fused_gdn_spec_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_tokens: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    num_active: int,
    num_tokens: int,
    block_size: int,
    padding_value: int,
) -> None:
    """Prepare all full-graph metadata for a contiguous speculative prefix.

    Outputs are stable, contiguous buffers for states, query boundaries,
    accepted counts, sequence masks, actual lengths and token indices.
    Counts come from existing CPU metadata; no device value is read back.
    """
    states, query, accepted, masks, lengths, tokens = outputs
    num_rows, num_slots = states.shape
    assert 0 <= num_active <= num_rows and 0 <= num_tokens <= tokens.numel()
    assert seq_lens.numel() >= num_active and accepted_tokens.numel() >= num_active
    assert query_start_loc.numel() >= num_active + 1
    assert query.numel() >= num_rows + 1 and lengths.numel() >= num_rows + 1
    assert accepted.numel() >= num_rows and masks.numel() >= num_rows
    assert block_size > 0 and num_slots > 0
    assert all(t.is_contiguous() for t in outputs)
    grid_size = max(states.numel(), num_rows + 1, num_tokens)
    _gdn_spec_metadata_kernel[(triton.cdiv(grid_size, _STATE_INDEX_BLOCK_SIZE),)](
        block_table,
        seq_lens,
        query_start_loc,
        accepted_tokens,
        states,
        query,
        accepted,
        masks,
        lengths,
        tokens,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens.stride(0),
        query_start_loc.stride(0),
        accepted_tokens.stride(0),
        num_active,
        num_rows,
        num_slots,
        num_tokens,
        block_size,
        padding_value,
        _STATE_INDEX_BLOCK_SIZE,
    )
