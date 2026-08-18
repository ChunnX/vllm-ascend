# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import logging
from dataclasses import dataclass, fields

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.logger import logger
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_final_chunk_indices,
    prepare_update_chunk_offsets,
)

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18


def _stable_argsort_for_npu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.int32)
    return torch.argsort(tensor, stable=True)


def _treat_single_token_prefills_with_state_as_decodes(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """Use decode metadata for one-token stateful prompt chunks.

    A final one-token prompt chunk can replay the same fixed graph as an
    ordinary decode.  Once recurrent state exists, both paths must construct
    identical GDN metadata so the graph consumes the current state indices.
    First-token prefills remain on the prefill path because they have no state
    to update.
    """
    is_prefilling = common_attn_metadata.is_prefilling
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    if is_prefilling is None or seq_lens_cpu is None:
        return common_attn_metadata

    query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
    prefill_to_decode = is_prefilling & (query_lens_cpu == 1) & (seq_lens_cpu > 1)
    if not torch.any(prefill_to_decode).item():
        return common_attn_metadata

    is_prefilling = is_prefilling.clone()
    is_prefilling[prefill_to_decode] = False
    return common_attn_metadata.replace(is_prefilling=is_prefilling)


@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    num_decodes: int
    cu_seqlens_kern: tuple[int, ...] | None = None
    keep_meta: torch.Tensor | None = None


@dataclass
class GDNCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    initial_state_mode: torch.Tensor | None


@dataclass
class GDNSpecCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor


@dataclass
class GDNPrefillMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    chunk: GDNChunkedPrefillMetadata


@dataclass
class GDNDecodeMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    actual_seq_lengths: torch.Tensor


@dataclass
class GDNSpecDecodeMetadata:
    spec_causal_conv1d: GDNSpecCausalConv1dMetadata
    actual_seq_lengths: torch.Tensor


def _build_actual_seq_lengths(
    query_start_loc: torch.Tensor,
    num_sequences: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    actual_seq_lengths = (
        torch.empty_like(query_start_loc[: num_sequences + 1]) if out is None else out[: num_sequences + 1]
    )
    actual_seq_lengths[:1].copy_(query_start_loc[:1])
    torch.sub(
        query_start_loc[1 : num_sequences + 1],
        query_start_loc[:num_sequences],
        out=actual_seq_lengths[1:],
    )
    return actual_seq_lengths


def _compact_empty_segments(cu_seqlens_host, initial_state, device=None):
    """Drop zero-length segments so AscendC fwd_h/fwd_o indexing lines up.

    Returns ``(cu_seqlens_kern, initial_state_kern, keep_meta)``:
    cu_seqlens / initial_state with empty segments removed, plus a bool
    mask (None when nothing was removed).  The compacted ``final_state``
    must be scattered back via ``keep_meta`` (empty segments keep their
    initial state).

    When *device* is given, ``keep_meta`` is moved to that device so that
    callers can index NPU tensors without an extra host→device sync.
    """
    if cu_seqlens_host is None:
        return None, initial_state, None
    cu = torch.tensor(cu_seqlens_host, dtype=torch.int64)
    keep = (cu[1:] - cu[:-1]) > 0
    if bool(keep.all()):
        return cu_seqlens_host, initial_state, None
    # Compute compact cu_seqlens while keep is still on CPU (cu is CPU-only).
    cu_kern = torch.cat([cu[:1], cu[1:][keep]]).tolist()
    # Move keep to device only for indexing device-side tensors.
    if device is not None:
        keep = keep.to(device)
    st_kern = initial_state[keep] if initial_state is not None else None
    return cu_kern, st_kern, keep


def _build_non_spec_chunked_prefill_metadata(
    builder,
    cu_seqlens_cpu: torch.Tensor,
    device: torch.device,
) -> GDNChunkedPrefillMetadata:
    hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
    linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("num_heads") is not None:
        gdn_num_heads = linear_attn_config["num_heads"] // builder.vllm_config.parallel_config.tensor_parallel_size
    elif hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
        gdn_num_heads = (
            hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
        )
    else:
        gdn_num_heads = builder.vllm_config.model_config.get_num_attention_heads(builder.vllm_config.parallel_config)
    cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * _GDN_CHUNK_SIZE))
    cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()

    chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    chunk_offsets_chunk64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    update_chunk_offsets_chunk64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    final_chunk_indices_chunk64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    chunk_indices_large_block = prepare_chunk_indices(
        cu_seqlens_cpu,
        _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
    )
    block_indices_cumsum = prepare_chunk_indices(cu_seqlens_cpu, cumsum_chunk_size)

    cu_seqlens_host = tuple(cu_seqlens_cpu.to(torch.int64).reshape(-1).tolist())
    num_decodes = sum(1 for seq_start, seq_end in zip(cu_seqlens_host, cu_seqlens_host[1:]) if seq_end - seq_start == 1)
    # Pre-compute compact cu_seqlens for AscendC kernels so each layer
    # can reuse them instead of calling _compact_empty_segments again.
    cu_seqlens_kern, _, keep_meta = _compact_empty_segments(cu_seqlens_host, None, device=device)
    if keep_meta is None:
        cu_seqlens_kern = None
    else:
        cu_seqlens_kern = tuple(cu_seqlens_kern)

    return GDNChunkedPrefillMetadata(
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_chunk64_host=tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist()),
        chunk_indices_chunk64=chunk_indices_chunk64.to(device=device, non_blocking=True),
        chunk_offsets_chunk64=chunk_offsets_chunk64.to(device=device, non_blocking=True),
        update_chunk_offsets_chunk64=update_chunk_offsets_chunk64.to(device=device, non_blocking=True),
        final_chunk_indices_chunk64=final_chunk_indices_chunk64.to(device=device, non_blocking=True),
        chunk_indices_large_block=chunk_indices_large_block.to(device=device, non_blocking=True),
        block_indices_cumsum=block_indices_cumsum.to(device=device, non_blocking=True),
        num_decodes=num_decodes,
        cu_seqlens_kern=cu_seqlens_kern,
        keep_meta=keep_meta,
    )


# How ``_derive_group_state_indices`` turns this group's block table into state
# indices. Recorded on the plan so later groups reproduce the branch the first
# group took without redoing the batch-shape work that selected it.
_STATE_INDEX_NON_SPEC = "non_spec"
_STATE_INDEX_SPEC_ONLY = "spec_only"
_STATE_INDEX_SPEC_MIXED = "spec_mixed"


@dataclass
class _GDNSharedBatchPlan:
    """One batch's GDN metadata, minus anything derived from a block table.

    A hybrid model spreads its Mamba layers over several KV cache groups, and
    ``build_attn_metadata`` calls the GDN builder once per group with metadata
    that differs only in ``block_table_tensor`` and ``slot_mapping``. Every
    field here is a function of the query/sequence lengths alone, so it is
    computed for the first group and reused by the rest.
    """

    state_index_mode: str
    num_prefills: int
    num_decodes: int
    num_decode_tokens: int
    num_prefill_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    spec_sequence_masks: torch.Tensor | None
    spec_sequence_indices: torch.Tensor | None
    non_spec_sequence_indices: torch.Tensor | None
    spec_token_indx: torch.Tensor | None
    non_spec_token_indx: torch.Tensor | None
    spec_query_start_loc: torch.Tensor | None
    non_spec_query_start_loc: torch.Tensor | None
    num_accepted_tokens: torch.Tensor | None
    has_initial_state: torch.Tensor | None
    prefill_has_initial_state: torch.Tensor | None
    prefill_query_start_loc: torch.Tensor | None
    chunk_indices: torch.Tensor | None
    chunk_offsets: torch.Tensor | None
    non_spec_chunked_prefill_metadata: GDNChunkedPrefillMetadata | None
    nums_dict: dict | None
    batch_ptr: torch.Tensor | None
    token_chunk_offset_ptr: torch.Tensor | None


class AscendGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        sequence_index_capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.decode_cudagraph_max_bs,
        )

        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (sequence_index_capacity,), dtype=torch.bool, device=device
        )

        self.spec_sequence_masks_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        )

        self.spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.non_spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.non_spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.spec_actual_seq_lengths: torch.Tensor = torch.empty(
            (sequence_index_capacity + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_actual_seq_lengths: torch.Tensor = torch.empty(
            (sequence_index_capacity + 1,),
            dtype=torch.int32,
            device=device,
        )

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        super()._init_reorder_batch_threshold(
            reorder_batch_threshold,
            supports_spec_as_decode,
            True,
        )
        if self.reorder_batch_threshold != 1:  # type: ignore
            speculative_config = self.vllm_config.speculative_config
            method = getattr(speculative_config, "method", None)
            num_spec = getattr(speculative_config, "num_speculative_tokens", None)
            if num_spec is not None:
                # dflash counts the base token in addition to the N speculative
                # tokens; dspark's threshold is just N by design.
                if method == "dflash":
                    self.reorder_batch_threshold = 1 + num_spec
                elif method == "dspark":
                    self.reorder_batch_threshold = num_spec

    def _copy_sequence_indices_to_device(
        self,
        spec_sequence_masks_cpu: torch.Tensor,
        num_spec_decodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = spec_sequence_masks_cpu.numel()
        num_non_spec_decodes = num_reqs - num_spec_decodes

        spec_indices_cpu = self.spec_sequence_indices_cpu[:num_spec_decodes]
        spec_indices_cpu.copy_(
            torch.nonzero(spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        spec_indices = self.spec_sequence_indices[:num_spec_decodes]
        spec_indices.copy_(spec_indices_cpu, non_blocking=True)

        non_spec_indices_cpu = self.non_spec_sequence_indices_cpu[:num_non_spec_decodes]
        non_spec_indices_cpu.copy_(
            torch.nonzero(~spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        non_spec_indices = self.non_spec_sequence_indices[:num_non_spec_decodes]
        non_spec_indices.copy_(non_spec_indices_cpu, non_blocking=True)

        return spec_indices, non_spec_indices

    def _pad_non_spec_decode_graph_inputs(
        self,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        num_decode_tokens: int,
        graph_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh fixed buffers consumed by a non-spec decode graph.

        ``num_decodes`` includes graph padding rows, while each real non-spec
        decode contributes exactly one token.  Therefore
        ``num_decode_tokens`` is the number of live rows to copy.  All other
        state rows must be null and their repeated terminal offsets must
        describe zero-length sequences.
        """
        assert num_decode_tokens <= graph_batch_size

        padded_state_indices = self.non_spec_state_indices_tensor[:graph_batch_size]
        padded_state_indices[num_decode_tokens:].fill_(NULL_BLOCK_ID)
        padded_state_indices[:num_decode_tokens].copy_(
            state_indices[:num_decode_tokens],
            non_blocking=True,
        )

        padded_query_start_loc = self.non_spec_query_start_loc[: graph_batch_size + 1]
        padded_query_start_loc[: num_decode_tokens + 1].copy_(
            query_start_loc[: num_decode_tokens + 1],
            non_blocking=True,
        )
        query_padding = padded_query_start_loc[num_decode_tokens + 1 :]
        if query_padding.numel() > 0:
            query_padding.copy_(
                padded_query_start_loc[num_decode_tokens].expand_as(query_padding),
                non_blocking=True,
            )

        return padded_state_indices, padded_query_start_loc

    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        """Make a captured speculative branch a no-op for this replay."""
        self.spec_state_indices_tensor[:graph_batch_size].fill_(PAD_SLOT_ID)
        self.spec_query_start_loc[: graph_batch_size + 1].zero_()
        self.num_accepted_tokens[:graph_batch_size].zero_()
        self.spec_actual_seq_lengths[: graph_batch_size + 1].zero_()

    def _attach_non_spec_prefill_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        chunk_metadata: GDNChunkedPrefillMetadata | None,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_prefill_metadata = None
        if attn_metadata.num_prefills <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.non_spec_query_start_loc for Ascend GDN non-spec prefill path.")
        if attn_metadata.prefill_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.prefill_query_start_loc for Ascend GDN non-spec prefill path.")
        if chunk_metadata is None:
            raise RuntimeError("Expected chunk metadata for Ascend GDN non-spec prefill path.")

        initial_state_mode = attn_metadata.has_initial_state
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN prefill conv1d path.")
        prefill_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        pcp_size = getattr(self.vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        if pcp_rank > 0 and attn_metadata.num_prefills > 0:
            prefill_seq_offset = max(0, prefill_num_rows - attn_metadata.num_prefills)
            initial_state_mode = initial_state_mode.clone()
            initial_state_mode[prefill_seq_offset:] = True
        attn_metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:prefill_num_rows],
                initial_state_mode=initial_state_mode,
            ),
            chunk=chunk_metadata,
        )
        return attn_metadata

    def _attach_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
    ) -> GDNAttentionMetadata:
        attn_metadata.spec_decode_metadata = None
        if attn_metadata.spec_sequence_masks is None:
            return attn_metadata

        if attn_metadata.spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.spec_query_start_loc for Ascend GDN speculative path.")
        if attn_metadata.spec_state_indices_tensor is None:
            raise RuntimeError("Expected spec_state_indices_tensor for Ascend GDN speculative conv1d path.")
        if attn_metadata.num_accepted_tokens is None:
            raise RuntimeError("Expected num_accepted_tokens for Ascend GDN speculative conv1d path.")

        num_sequences = attn_metadata.num_spec_decodes
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            num_sequences = attn_metadata.spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.spec_actual_seq_lengths
        spec_num_rows = attn_metadata.spec_query_start_loc.size(0) - 1

        attn_metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
            spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
                query_start_loc=attn_metadata.spec_query_start_loc,
                cache_indices=attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                num_accepted_tokens=attn_metadata.num_accepted_tokens[:spec_num_rows],
            ),
            actual_seq_lengths=_build_actual_seq_lengths(
                attn_metadata.spec_query_start_loc,
                num_sequences,
                actual_seq_lengths_buffer,
            ),
        )
        return attn_metadata

    def _attach_non_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_decode_metadata = None
        if attn_metadata.num_decodes <= 0 and attn_metadata.num_prefills <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected non-spec query_start_loc for Ascend GDN non-spec decode path.")
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN decode conv1d path.")

        num_sequences = attn_metadata.num_decodes
        non_spec_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_spec_decodes == 0:
            num_sequences = attn_metadata.non_spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.non_spec_actual_seq_lengths

        attn_metadata.non_spec_decode_metadata = GDNDecodeMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:non_spec_num_rows],
                initial_state_mode=None,
            ),
            actual_seq_lengths=_build_actual_seq_lengths(
                attn_metadata.non_spec_query_start_loc,
                num_sequences,
                actual_seq_lengths_buffer,
            ),
        )
        return attn_metadata

    def _fold_spec_sized_prefill_chunks_into_spec(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_sequence_masks_cpu: torch.Tensor,
        num_accepted_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance stateful spec-width prompt chunks through live spec inputs.

        A stateful prompt chunk of exactly ``num_spec + 1`` tokens is
        shape-identical to a speculative decode row and can replay that graph.
        Mark it speculative and accept the complete chunk so conv1d and
        recurrent state advance over every token.  First prompt chunks stay on
        the prefill path because they have no prior state.
        """
        is_prefilling = common_attn_metadata.is_prefilling
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if is_prefilling is None or seq_lens_cpu is None or num_accepted_tokens is None:
            return spec_sequence_masks_cpu, num_accepted_tokens

        # Common metadata tensors can include graph padding; only leading rows
        # correspond to requests represented by the runtime spec mask.
        num_reqs = min(
            spec_sequence_masks_cpu.numel(),
            is_prefilling.numel(),
            seq_lens_cpu.numel(),
        )
        is_prefilling = is_prefilling[:num_reqs]
        seq_lens_cpu = seq_lens_cpu[:num_reqs]
        query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)[:num_reqs]
        fold = (
            is_prefilling
            & ~spec_sequence_masks_cpu
            & (query_lens_cpu == self.num_spec + 1)
            & (seq_lens_cpu > query_lens_cpu)
        )
        fold_indices = fold.nonzero(as_tuple=True)[0]
        if fold_indices.numel() == 0:
            return spec_sequence_masks_cpu, num_accepted_tokens

        spec_sequence_masks_cpu = spec_sequence_masks_cpu.clone()
        spec_sequence_masks_cpu[fold_indices] = True
        num_accepted_tokens = num_accepted_tokens.clone()
        num_accepted_tokens[fold_indices.to(num_accepted_tokens.device)] = self.num_spec + 1
        return spec_sequence_masks_cpu, num_accepted_tokens

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = _treat_single_token_prefills_with_state_as_decodes(common_attn_metadata)

    def _compute_shared_batch_plan(
        self,
        m: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> _GDNSharedBatchPlan:
        """The batch-shape half of ``build``, with no block table involved.

        This is the original ``build`` body up to the point where state indices
        are derived; the block-table work moved to
        ``_derive_group_state_indices`` so it can stay per-group.
        """
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_sequence_indices: torch.Tensor | None = None
        non_spec_sequence_indices: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_reqs = num_decode_draft_tokens_cpu.numel()
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_reqs]
            torch.ge(
                num_decode_draft_tokens_cpu,
                0,
                out=spec_sequence_masks_cpu,
            )
            spec_sequence_masks_cpu, num_accepted_tokens = self._fold_spec_sized_prefill_chunks_into_spec(
                m,
                spec_sequence_masks_cpu,
                num_accepted_tokens,
            )
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.spec_sequence_masks[:num_reqs]
                spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)
                spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                    spec_sequence_masks_cpu,
                    num_spec_decodes,
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
                m,
                decode_threshold=1,
                treat_short_extends_as_decodes=False,
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            state_index_mode = _STATE_INDEX_NON_SPEC
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            assert spec_sequence_masks_cpu is not None
            assert spec_sequence_indices is not None
            assert non_spec_sequence_indices is not None

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            num_spec_decode_tokens = query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens

            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                state_index_mode = _STATE_INDEX_SPEC_ONLY
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = _stable_argsort_for_npu(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]
                state_index_mode = _STATE_INDEX_SPEC_MIXED

                spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    spec_sequence_indices,
                )
                non_spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    non_spec_sequence_indices,
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    spec_query_lens,
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    non_spec_query_lens,
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = torch.index_select(
                num_accepted_tokens,
                0,
                spec_sequence_indices,
            )

        # A FULL graph may retain a captured speculative conv/recurrent task
        # even when the current replay contains only non-spec prefill work.
        # Refresh its stable inputs for every no-spec replay, not only for the
        # pure non-spec decode branch below. Otherwise a DP idle dummy can
        # replay the preceding request's state indices and mutate live Mamba
        # checkpoints after that request has finished.
        if self.use_full_cuda_graph and self.use_spec_decode and num_spec_decodes == 0:
            self._reset_spec_decode_graph_inputs(m.num_reqs)

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_query_start_loc_cpu: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        non_spec_chunked_prefill_metadata: GDNChunkedPrefillMetadata | None = None
        has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                prefill_query_start_loc = non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu

            assert prefill_query_start_loc_cpu is not None
            non_spec_chunked_prefill_metadata = _build_non_spec_chunked_prefill_metadata(
                self,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )
            # Preserve upstream GDNAttentionMetadata fields for callers that
            # still use the chunk_gated_delta_rule API directly.
            chunk_indices = non_spec_chunked_prefill_metadata.chunk_indices_chunk64
            chunk_offsets = non_spec_chunked_prefill_metadata.chunk_offsets_chunk64

            (
                has_initial_state,
                nums_dict,
                batch_ptr,
                token_chunk_offset_ptr,
            ) = self._build_prefill_has_initial_state_and_causal_conv1d_meta(
                common_attn_metadata=m,
                context_lens_tensor=context_lens_tensor,
                num_prefills=num_prefills,
                spec_sequence_masks_cpu=spec_sequence_masks_cpu,
                non_spec_sequence_indices=non_spec_sequence_indices,
                non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
                query_start_loc=query_start_loc,
            )
            assert has_initial_state is not None
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state

        return _GDNSharedBatchPlan(
            state_index_mode=state_index_mode,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_indices=spec_sequence_indices,
            non_spec_sequence_indices=non_spec_sequence_indices,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            num_accepted_tokens=num_accepted_tokens,
            has_initial_state=has_initial_state,
            prefill_has_initial_state=prefill_has_initial_state,
            prefill_query_start_loc=prefill_query_start_loc,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            non_spec_chunked_prefill_metadata=non_spec_chunked_prefill_metadata,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    def _shared_batch_plan_key(
        self,
        m: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> tuple:
        """Identify one batch's group-independent inputs.

        Deliberately excludes ``self``: the point is for every GDN group in one
        invocation to share a plan. Also excludes ``block_table_tensor`` and
        ``slot_mapping``, the only two fields ``build_attn_metadata`` varies
        per group.

        ``data_ptr()`` is safe here only because the cache dict lives for a
        single ``build_attn_metadata`` call; addresses are recycled across
        steps, so a longer-lived dict would alias unrelated batches.
        """
        ptr = lambda t: None if t is None else t.data_ptr()  # noqa: E731
        return (
            m.num_reqs,
            m.num_actual_tokens,
            ptr(m.query_start_loc),
            ptr(m.query_start_loc_cpu),
            ptr(m.seq_lens),
            ptr(num_accepted_tokens),
            ptr(num_decode_draft_tokens_cpu),
            self.num_spec,
            self.use_spec_decode,
        )

    def _get_shared_batch_plan(
        self,
        m: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
        batch_shared_cache: dict | None,
    ) -> _GDNSharedBatchPlan:
        """Compute the batch-shape work once per invocation, not once per group.

        A hybrid model splits its Mamba layers over several KV cache groups
        (Qwen3.6 + DSpark: 10), and ``build_attn_metadata`` calls this builder
        once per group. Everything below depends on query/sequence lengths
        alone -- only the state indices derive from the group's block table --
        so without this the same batch-shape computation runs once per group.

        Mirrors the DSA ``*_ratio_to_sas_metadata`` dicts that
        ``build_attn_metadata`` already threads through for the same reason.
        """
        if batch_shared_cache is None:
            return self._compute_shared_batch_plan(m, num_accepted_tokens, num_decode_draft_tokens_cpu)

        key = self._shared_batch_plan_key(m, num_accepted_tokens, num_decode_draft_tokens_cpu)
        plan = batch_shared_cache.get(key)
        if plan is None:
            plan = self._compute_shared_batch_plan(m, num_accepted_tokens, num_decode_draft_tokens_cpu)
            batch_shared_cache[key] = plan
        elif logger.isEnabledFor(logging.DEBUG):
            # Recompute and compare: proves the cached fields really are
            # group-independent. A field that secretly varies per group would
            # otherwise corrupt results silently, which is far more expensive
            # to find later than an assertion here.
            self._assert_plan_is_group_independent(
                plan,
                self._compute_shared_batch_plan(m, num_accepted_tokens, num_decode_draft_tokens_cpu),
            )
        return plan

    @staticmethod
    def _assert_plan_is_group_independent(
        cached: _GDNSharedBatchPlan,
        fresh: _GDNSharedBatchPlan,
    ) -> None:
        for field in fields(_GDNSharedBatchPlan):
            lhs = getattr(cached, field.name)
            rhs = getattr(fresh, field.name)
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                same = lhs.shape == rhs.shape and bool(torch.equal(lhs, rhs))
            elif isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
                same = False
            elif field.name in ("nums_dict", "non_spec_chunked_prefill_metadata"):
                same = True  # opaque payloads; covered by the tensors they wrap
            else:
                same = lhs == rhs
            if not same:
                raise AssertionError(
                    f"GDN shared batch plan field {field.name!r} differs between KV cache "
                    "groups; it is not group-independent and must move out of the plan"
                )

    def _derive_group_state_indices(
        self,
        plan: _GDNSharedBatchPlan,
        block_table_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """The only part of ``build`` that depends on this group's block table."""
        if plan.state_index_mode == _STATE_INDEX_NON_SPEC:
            return None, block_table_tensor[:, 0], block_table_tensor

        assert plan.spec_sequence_indices is not None
        spec_state_indices_tensor = torch.index_select(
            block_table_tensor[:, : self.num_spec + 1],
            0,
            plan.spec_sequence_indices,
        )
        if plan.state_index_mode == _STATE_INDEX_SPEC_ONLY:
            return spec_state_indices_tensor, None, None

        assert plan.non_spec_sequence_indices is not None
        non_spec_state_indices_tensor = torch.index_select(
            block_table_tensor[:, 0],
            0,
            plan.non_spec_sequence_indices,
        )
        return (
            spec_state_indices_tensor,
            non_spec_state_indices_tensor,
            non_spec_state_indices_tensor,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        batch_shared_cache: dict | None = None,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata
        plan = self._get_shared_batch_plan(
            m, num_accepted_tokens, num_decode_draft_tokens_cpu, batch_shared_cache
        )

        num_prefills = plan.num_prefills
        num_decodes = plan.num_decodes
        num_decode_tokens = plan.num_decode_tokens
        num_prefill_tokens = plan.num_prefill_tokens
        num_spec_decodes = plan.num_spec_decodes
        num_spec_decode_tokens = plan.num_spec_decode_tokens
        spec_sequence_masks = plan.spec_sequence_masks
        spec_token_indx = plan.spec_token_indx
        non_spec_token_indx = plan.non_spec_token_indx
        spec_query_start_loc = plan.spec_query_start_loc
        non_spec_query_start_loc = plan.non_spec_query_start_loc
        num_accepted_tokens = plan.num_accepted_tokens
        has_initial_state = plan.has_initial_state
        chunk_indices = plan.chunk_indices
        chunk_offsets = plan.chunk_offsets
        prefill_query_start_loc = plan.prefill_query_start_loc
        prefill_has_initial_state = plan.prefill_has_initial_state
        non_spec_chunked_prefill_metadata = plan.non_spec_chunked_prefill_metadata
        nums_dict = plan.nums_dict
        batch_ptr = plan.batch_ptr
        token_chunk_offset_ptr = plan.token_chunk_offset_ptr

        # Everything below is this group's own: the block table it addresses,
        # the state indices derived from it, and the per-builder graph buffers.
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        (
            spec_state_indices_tensor,
            non_spec_state_indices_tensor,
            non_spec_conv1d_cache_indices,
        ) = self._derive_group_state_indices(plan, block_table_tensor)

        prefill_state_indices: torch.Tensor | None = None
        if num_prefills > 0:
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_state_indices_tensor is not None
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_state_indices = non_spec_state_indices_tensor

        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            # Spec decode has multiple tokens per request. Keep the metadata
            # passed to conv1d/recurrent kernels at request granularity; padding
            # it to the token count makes the conv1d update kernel treat every
            # token as an independent decode sequence.
            #
            # These buffers belong to this builder, so each KV cache group
            # keeps its own graph-stable addresses even though the values
            # copied in come from the shared plan.
            spec_batch_size = m.num_reqs

            self.spec_state_indices_tensor[spec_batch_size:].fill_(NULL_BLOCK_ID)
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor,
                non_blocking=True,
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:spec_batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes],
                non_blocking=True,
            )
            spec_sequence_masks = self.spec_sequence_masks[:spec_batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx,
                non_blocking=True,
            )
            non_spec_token_indx = self.non_spec_token_indx[: non_spec_token_indx.size(0)]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx,
                non_blocking=True,
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc,
                non_blocking=True,
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore
            spec_query_start_loc = self.spec_query_start_loc[: spec_batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens,
                non_blocking=True,
            )
            num_accepted_tokens = self.num_accepted_tokens[:spec_batch_size]
            # Zero-length rows are skipped before the recurrent kernel reads
            # this value. Keep the padding sentinel within the public
            # accepted-token range [1, query_width].
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            graph_batch_size = m.num_reqs
            (
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
            ) = self._pad_non_spec_decode_graph_inputs(
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
                num_decode_tokens=num_decode_tokens,
                graph_batch_size=graph_batch_size,
            )
            non_spec_conv1d_cache_indices = non_spec_state_indices_tensor

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        attn_metadata = self._attach_non_spec_prefill_metadata(
            attn_metadata,
            non_spec_chunked_prefill_metadata,
            non_spec_conv1d_cache_indices,
        )
        attn_metadata = self._attach_spec_decode_metadata(
            attn_metadata,
        )
        return self._attach_non_spec_decode_metadata(
            attn_metadata,
            non_spec_conv1d_cache_indices,
        )

    def _build_prefill_has_initial_state_and_causal_conv1d_meta(
        self,
        *,
        common_attn_metadata: CommonAttentionMetadata,
        context_lens_tensor: torch.Tensor,
        num_prefills: int,
        spec_sequence_masks_cpu: torch.Tensor | None,
        non_spec_sequence_indices: torch.Tensor | None,
        non_spec_query_start_loc_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        dict[int, dict[str, object]] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del num_prefills
        has_initial_state = context_lens_tensor > 0
        if spec_sequence_masks_cpu is not None:
            assert non_spec_sequence_indices is not None
            has_initial_state = torch.index_select(
                has_initial_state,
                0,
                non_spec_sequence_indices,
            )
            assert non_spec_query_start_loc_cpu is not None
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            non_spec_query_start_loc_cpu,
            device=query_start_loc.device,
        )
        return (
            has_initial_state,
            nums_dict,
            batch_ptr,
            token_chunk_offset_ptr,
        )


class AscendGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        return AscendGDNAttentionMetadataBuilder
