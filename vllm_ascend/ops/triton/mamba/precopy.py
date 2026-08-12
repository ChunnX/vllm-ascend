# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton-Ascend Mamba align pre-copy kernel.

Same contract as upstream's ``precopy_mamba_align_fused_kernel``: before the
forward pass, migrate each request's conv/temporal state from its previous
block column into the new window block column, so the mamba kernels read the
initial state from the write-side block (V1 align semantics).

The only change is where the ``int -> pointer`` cast happens. Upstream's
conv branches build the pointer *inside* the copy loop, from a vector:

    curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))

triton-ascend's pointer-offset analysis aborts on that form -- an MLIR
``PtrOffsetInfo::AxisInfo`` assertion, i.e. a hard abort() with no Python
traceback. The working form casts the *scalar* base address once, outside the
loop, and offsets the resulting pointer:

    src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
    ...
    tl.load(src_ptr + i + offsets, mask=mask)

vllm-ascend already carries exactly this rewrite for the two sibling kernels
that share the same copy body (``batch_memcpy_kernel`` and
``postprocess_mamba_fused_kernel``); this is the third and last entry point
into it, and the only one on the model-runner-v2 align path.

Upstream's temporal branch already casts the scalar base (``src_u64``), so it
is reproduced unchanged; only the two conv branches move.

The copy body is inlined rather than shared with a ``@triton.jit`` device
function, matching the existing Ascend postprocess kernel: the point is not to
introduce another Triton helper for the Ascend backend to lower.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def precopy_mamba_align_fused_kernel(
    # Per-request-slot inputs (indexed by req_idx via idx_mapping), produced by
    # the V2 fused align preprocess kernel for the current step:
    mamba_state_idx_ptr,  # post-advance dst block column
    src_col_ptr,  # pre-advance src block column (-1 = fresh)
    token_bias_ptr,  # accepted-token bias = num_accepted - 1 (pre-reset)
    # Flattened state-layout metadata, indexed by state_idx
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(mamba_state_idx_ptr + req_idx)
    # Fresh state, or still writing the same block: the mamba kernels locate the
    # initial state in-block via num_accepted, so there is nothing to copy.
    if src_col < 0 or src_col == dst_col:
        return

    token_bias = tl.load(token_bias_ptr + req_idx)

    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    # Each mamba group allocates its own physical blocks, so resolve this
    # state's group before indexing the block table. Block ids are int32.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + batch_idx * block_table_stride_req

    # Widen block ids to int64 before `block_id * state_block_stride`:
    # state_block_stride can exceed 2**31 bytes for large mamba caches, and
    # Triton would otherwise do the multiply in int32 and wrap.
    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dest_block_id * state_block_stride

    is_conv_state = conv_width > 0

    if CONV_STATE_DIM_FIRST and is_conv_state:
        # DS conv layout: state_len is the slide axis; copy per dim row.
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        per_row_bytes = (conv_width - token_bias).to(tl.int64) * state_elem_size
        bias_bytes = token_bias.to(tl.int64) * state_elem_size
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        for d in range(0, dim_rows):
            row_src = src_block_addr + d * row_stride + bias_bytes
            row_dst = dst_addr + d * row_stride
            # Cast the scalar row base once, outside the copy loop.
            row_src_ptr = row_src.to(tl.pointer_type(tl.uint8))
            row_dst_ptr = row_dst.to(tl.pointer_type(tl.uint8))
            for i in range(0, per_row_bytes, COPY_BLOCK_SIZE):
                mask = (i + offsets) < per_row_bytes
                data = tl.load(row_src_ptr + i + offsets, mask=mask)
                tl.store(row_dst_ptr + i + offsets, data, mask=mask)
        return

    if is_conv_state:
        # SD conv: state[bt[src_col], token_bias:] ->
        #          state[bt[dst_col], :conv_width - token_bias]
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        num_elems_to_copy = (conv_width - token_bias).to(tl.int64) * state_inner_size
        copy_size = num_elems_to_copy * state_elem_size
        # Cast the scalar base once, outside the copy loop.
        src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
        dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
        offsets = tl.arange(0, COPY_BLOCK_SIZE)
        for i in range(0, copy_size, COPY_BLOCK_SIZE):
            mask = (i + offsets) < copy_size
            data = tl.load(src_ptr + i + offsets, mask=mask)
            tl.store(dst_ptr + i + offsets, data, mask=mask)
        return

    # Temporal state: state[bt[src_col + token_bias]] -> state[bt[dst_col]]
    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
    src_addr = state_base_addr + actual_src_block_id * state_block_stride
    # Natural block data size (inner_size * elem_size), NOT state_block_stride,
    # which is the page stride and can exceed the actual data when the state
    # tensor uses as_strided page padding.
    copy_size = state_inner_size * state_elem_size

    # Vectorize via uint64 (8B per thread): both temporal and SD conv produce
    # addresses aligned to a full token slice (inner_size * elem_size) and a
    # copy_size that is a multiple of it, which is 8B-aligned for every state
    # dtype in use. A masked byte tail covers the remaining 0-7 bytes, only
    # reachable for sub-8B slices. Both casts are already on scalars upstream.
    copy_size_u64 = copy_size // 8
    src_u64 = src_addr.to(tl.pointer_type(tl.uint64))
    dst_u64 = dst_addr.to(tl.pointer_type(tl.uint64))
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for i in range(0, copy_size_u64, COPY_BLOCK_SIZE):
        mask = (i + offsets) < copy_size_u64
        data = tl.load(src_u64 + i + offsets, mask=mask)
        tl.store(dst_u64 + i + offsets, data, mask=mask)

    tail_start = copy_size_u64 * 8
    tail_bytes = copy_size - tail_start
    tail_off = tl.arange(0, 8)
    tail_src = (src_addr + tail_start).to(tl.pointer_type(tl.uint8))
    tail_dst = (dst_addr + tail_start).to(tl.pointer_type(tl.uint8))
    tail_mask = tail_off < tail_bytes
    tail_data = tl.load(tail_src + tail_off, mask=tail_mask)
    tl.store(tail_dst + tail_off, tail_data, mask=tail_mask)
