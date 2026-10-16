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
