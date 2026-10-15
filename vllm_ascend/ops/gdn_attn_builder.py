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
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
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
# Accepted-token counts are one-based. The zero-length query segment makes
# graph-padding rows no-ops, while 1 keeps their count valid for GDN kernels.
_GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT = 1


def _stable_argsort_for_npu(tensor: torch.Tensor) -> torch.Tensor:
    """Stable argsort with a sort key AiCore can actually take.

    ArgSort has no AiCore implementation for int32/int64 and silently falls back
    to AiCPU, which the runtime warns about and recommends float32 for. The keys
    here are boolean masks, so float32 represents them exactly and a stable sort
    gives the identical permutation -- the dtype is free to choose and int32 was
    the one choice that leaves AiCore.
    """
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.float32)
    return torch.argsort(tensor, stable=True)


def _remove_spec_graph_padding_queries(
    common_attn_metadata: CommonAttentionMetadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> CommonAttentionMetadata:
    """Remove inactive FIA graph rows from the GDN-local query view.

    FIA represents every padded graph request with a full K+1 query span.
    Those rows must be zero-length for the target's recurrent GDN state update.
    The shared MLA metadata remains unchanged.
    """
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound
    if seq_lens is None or num_decode_draft_tokens_cpu is None:
        return common_attn_metadata
    num_reqs = common_attn_metadata.num_reqs
    draft_tokens = num_decode_draft_tokens_cpu[:num_reqs]
    if draft_tokens.numel() != num_reqs or not torch.any(draft_tokens > 0).item():
        return common_attn_metadata
    inactive = (seq_lens[:num_reqs] == 0) & (draft_tokens < 0)
    if not torch.any(inactive).item():
        return common_attn_metadata
    first_inactive = int(torch.nonzero(inactive, as_tuple=True)[0][0])
    if not torch.all(inactive[first_inactive:]).item():
        raise ValueError("GDN speculative graph padding must be a request suffix")
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu.clone()
    query_start_loc_cpu[first_inactive + 1 :] = query_start_loc_cpu[first_inactive]
    return common_attn_metadata.replace(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(common_attn_metadata.query_start_loc.device),
        num_actual_tokens=int(query_start_loc_cpu[-1]),
    )


def _treat_single_token_prefills_with_state_as_decodes(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """Use decode metadata for one-token stateful prompt chunks.

    A final one-token prompt chunk can replay the same fixed graph as an
    ordinary decode. Once recurrent state exists, both paths must construct
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
    chunk_indices_chunk64: torch.Tensor | None
    chunk_offsets_chunk64: torch.Tensor | None
    update_chunk_offsets_chunk64: torch.Tensor | None
    final_chunk_indices_chunk64: torch.Tensor | None
    chunk_indices_large_block: torch.Tensor | None
    block_indices_cumsum: torch.Tensor | None
    num_decodes: int = 0
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
    # 310P derives lengths in its recurrent wrapper and only consumes conv metadata.
    actual_seq_lengths: torch.Tensor | None


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


def _materialize_graph_request_tensor(
    buffer: torch.Tensor,
    values: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
    padding_value: int | bool,
) -> torch.Tensor:
    """Write request-level values into a stable full-graph buffer."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count > buffer.size(0):
        raise ValueError(f"graph_request_count ({graph_request_count}) exceeds buffer capacity ({buffer.size(0)})")
    if actual_request_count > values.size(0):
        raise ValueError(f"actual_request_count ({actual_request_count}) exceeds source rows ({values.size(0)})")

    materialized = buffer[:graph_request_count]
    materialized[:actual_request_count].copy_(
        values[:actual_request_count],
        non_blocking=True,
    )
    materialized[actual_request_count:].fill_(padding_value)
    return materialized


def _materialize_graph_query_start_loc(
    buffer: torch.Tensor,
    query_start_loc: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
) -> torch.Tensor:
    """Pad request boundaries with zero-length dummy request segments."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count + 1 > buffer.size(0):
        raise ValueError(
            f"graph request boundaries ({graph_request_count + 1}) exceed buffer capacity ({buffer.size(0)})"
        )
    if actual_request_count + 1 > query_start_loc.size(0):
        raise ValueError(
            f"actual request boundaries ({actual_request_count + 1}) exceed source rows ({query_start_loc.size(0)})"
        )
    materialized = buffer[: graph_request_count + 1]
    materialized[: actual_request_count + 1].copy_(
        query_start_loc[: actual_request_count + 1],
        non_blocking=True,
    )
    padding = materialized[actual_request_count + 1 :]
    if padding.numel() > 0:
        padding.copy_(
            materialized[actual_request_count : actual_request_count + 1].expand_as(padding),
            non_blocking=True,
        )
    return materialized


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
    build_device_metadata: bool = True,
) -> GDNChunkedPrefillMetadata:
    chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    if build_device_metadata:
        hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
        linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
        if isinstance(linear_attn_config, dict) and linear_attn_config.get("num_heads") is not None:
            gdn_num_heads = linear_attn_config["num_heads"] // builder.vllm_config.parallel_config.tensor_parallel_size
        elif hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
            gdn_num_heads = (
                hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
            )
        else:
            gdn_num_heads = builder.vllm_config.model_config.get_num_attention_heads(
                builder.vllm_config.parallel_config
            )
        cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * _GDN_CHUNK_SIZE))
        cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()
        chunk_offsets_chunk64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        update_chunk_offsets_chunk64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        final_chunk_indices_chunk64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        chunk_indices_large_block = prepare_chunk_indices(
            cu_seqlens_cpu,
            _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        )
        block_indices_cumsum = prepare_chunk_indices(cu_seqlens_cpu, cumsum_chunk_size)

    cu_seqlens_host = tuple(cu_seqlens_cpu.to(torch.int64).reshape(-1).tolist())
    # Pre-compute compact cu_seqlens for AscendC kernels so each layer
    # can reuse them instead of calling _compact_empty_segments again.
    cu_seqlens_kern, _, keep_meta = _compact_empty_segments(cu_seqlens_host, None, device=device)
    if keep_meta is None:
        cu_seqlens_kern = None
    else:
        cu_seqlens_kern = tuple(cu_seqlens_kern)

    if build_device_metadata:
        device_chunk_metadata = {
            "chunk_indices_chunk64": chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_offsets_chunk64": chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "update_chunk_offsets_chunk64": update_chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "final_chunk_indices_chunk64": final_chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_indices_large_block": chunk_indices_large_block.to(device=device, non_blocking=True),
            "block_indices_cumsum": block_indices_cumsum.to(device=device, non_blocking=True),
        }
    else:
        device_chunk_metadata = {
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": None,
            "update_chunk_offsets_chunk64": None,
            "final_chunk_indices_chunk64": None,
            "chunk_indices_large_block": None,
            "block_indices_cumsum": None,
        }

    return GDNChunkedPrefillMetadata(
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_chunk64_host=tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist()),
        **device_chunk_metadata,
        cu_seqlens_kern=cu_seqlens_kern,
        keep_meta=keep_meta,
    )


# How ``_derive_group_state_indices`` turns this group's block table into state
# indices. Recorded on the plan so later groups reproduce the branch the first
# group took without redoing the batch-shape work that selected it.
# Namespaces this builder's entry in the batch-scoped cache that
# ``build_attn_metadata`` threads through, so it cannot collide with the
# shared batch plan's own entry.
_GDN_LOCAL_METADATA_CACHE_TAG = "gdn_local_metadata"

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
    build_device_chunk_metadata = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        from vllm_ascend.worker.v2.spec_decode.dspark.av_config import adaptive_verification_enabled

        # Gate ragged-decode GDN metadata on the adapted AV manager.
        self.adaptive_verification_enabled = adaptive_verification_enabled(vllm_config)

        # Adaptive verification gives each request its own verification length,
        # so its batches are ragged. That is a different contract from the
        # untrimmed fixed-K one and it is what fixes the GDN request axis below,
        # independently of whether this run is eager or captured.
        spec = vllm_config.speculative_config
        self.ragged_spec_decode = bool(spec is not None and getattr(spec, "enable_adaptive_verification", False))
        # Opt-in until the rest of the fixed-axis contract lands. Pinning the
        # GDN axis alone leaves the attention request axis and the persistent
        # seq_lens mirror still following the batch, so it is a half-applied
        # contract rather than a safer one: the padding rows it creates are
        # inert for GDN and not yet accounted for on those sides. It is also
        # a no-op when max_num_seqs equals the live request count, which is
        # why it looked harmless where it was first exercised.
        from vllm_ascend.worker.v2.spec_decode.dspark.av_config import av_graph_pins_gdn_axis

        self.ragged_spec_decode = self.ragged_spec_decode and av_graph_pins_gdn_axis()

        # B_gdn -- the request axis the GDN state operators see. Under the ragged
        # contract it is the service maximum for every graph bucket, never the
        # bucket's own request count.
        #
        # Letting it follow the bucket (32 rows at Q=32, 64 at Q=64, B_max above)
        # reads as the natural choice and holds up on a single node at steady
        # concurrency. It fails while concurrency ramps: each bucket then carries
        # its own stateful tiling and request axis, and the first requests of a
        # ramp come out short or garbled until the shape settles. Shrinking a
        # shape costs a stateless operator only speed; for a stateful one it
        # changes the state read/write contract. So every bucket is given the
        # same axis, with the rows past the live ones held at zero length and an
        # invalid state index.
        self.gdn_request_axis = self.vllm_config.scheduler_config.max_num_seqs
        # The inherited graph buffers are sized by decode_cudagraph_max_bs --
        # max_num_seqs * (num_spec + 1), capped by the largest capture size. A cap
        # below max_num_seqs would leave them too narrow for the fixed axis, so
        # the full-graph path is refused rather than quietly reverting to a
        # per-bucket axis, which is the shape this is here to prevent.
        self.gdn_request_axis_fits_graph = self.gdn_request_axis <= self.decode_cudagraph_max_bs
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
        self.spec_actual_seq_lengths: torch.Tensor | None = None
        self.non_spec_actual_seq_lengths: torch.Tensor | None = None
        if self._USE_COMMON_KERNEL_METADATA:
            self.spec_actual_seq_lengths = torch.empty((sequence_index_capacity + 1,), dtype=torch.int32, device=device)
            self.non_spec_actual_seq_lengths = torch.empty(
                (sequence_index_capacity + 1,), dtype=torch.int32, device=device
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
            if num_spec is not None and method in ("dflash", "dspark"):
                # The target-model verification query always contains the
                # base token plus N speculative tokens. DSpark's
                # sample_from_anchor only makes the draft-model forward use N
                # queries; it must not change target batch reordering.
                self.reorder_batch_threshold = 1 + num_spec

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

    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        """Make a captured speculative branch a no-op for this replay."""
        self.spec_state_indices_tensor[:graph_batch_size].fill_(PAD_SLOT_ID)
        self.spec_query_start_loc[: graph_batch_size + 1].zero_()
        self.num_accepted_tokens[:graph_batch_size].zero_()
        if self.spec_actual_seq_lengths is not None:
            self.spec_actual_seq_lengths[: graph_batch_size + 1].zero_()

    def _attach_non_spec_prefill_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        chunk_metadata: GDNChunkedPrefillMetadata | None,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_prefill_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA or attn_metadata.num_prefills <= 0:
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
            actual_seq_lengths=(
                _build_actual_seq_lengths(
                    attn_metadata.spec_query_start_loc,
                    num_sequences,
                    actual_seq_lengths_buffer,
                )
                if self._USE_COMMON_KERNEL_METADATA
                else None
            ),
        )
        return attn_metadata

    def _attach_non_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_decode_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA:
            return attn_metadata
        if attn_metadata.num_decodes <= 0:
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

    _SPEC_GRAPH_PAD_SLOT_ID = NULL_BLOCK_ID
    # Common GDN/KDA kernels consume prebuilt chunks and recurrent lengths.
    # 310P consumes top-level fields and only shares the spec conv wrapper.
    _USE_COMMON_KERNEL_METADATA = True

    def _can_pad_spec_decode(self, graph_request_count: int, num_spec_decode_tokens: int) -> bool:
        return (
            graph_request_count <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _pad_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize speculative decode metadata at graph request granularity."""
        actual_request_count = attn_metadata.num_spec_decodes
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert spec_state_indices is not None
        assert spec_sequence_masks is not None
        assert spec_query_start_loc is not None
        assert num_accepted_tokens is not None

        # Request and token buffers have different capacities under MTP.
        # Check token indices separately before modifying any captured inputs.
        assert attn_metadata.non_spec_token_indx is not None
        assert attn_metadata.spec_token_indx is not None
        if attn_metadata.spec_token_indx.numel() > self.spec_token_indx.numel():
            raise ValueError("Spec token indices exceed the graph token buffer capacity")
        if attn_metadata.non_spec_token_indx.numel() > self.non_spec_token_indx.numel():
            raise ValueError("Non-spec token indices exceed the graph token buffer capacity")

        attn_metadata.spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.spec_state_indices_tensor,
            spec_state_indices,
            actual_request_count,
            graph_request_count,
            self._SPEC_GRAPH_PAD_SLOT_ID,
        )
        attn_metadata.spec_sequence_masks = _materialize_graph_request_tensor(
            self.spec_sequence_masks,
            spec_sequence_masks,
            actual_request_count,
            graph_request_count,
            False,
        )

        non_spec_token_count = attn_metadata.non_spec_token_indx.size(0)
        spec_token_count = attn_metadata.spec_token_indx.size(0)
        self.non_spec_token_indx[:non_spec_token_count].copy_(
            attn_metadata.non_spec_token_indx,
            non_blocking=True,
        )
        self.spec_token_indx[:spec_token_count].copy_(
            attn_metadata.spec_token_indx,
            non_blocking=True,
        )
        attn_metadata.non_spec_token_indx = self.non_spec_token_indx[:non_spec_token_count]
        attn_metadata.spec_token_indx = self.spec_token_indx[:spec_token_count]

        attn_metadata.spec_query_start_loc = _materialize_graph_query_start_loc(
            self.spec_query_start_loc,
            spec_query_start_loc,
            actual_request_count,
            graph_request_count,
        )
        attn_metadata.num_accepted_tokens = _materialize_graph_request_tensor(
            self.num_accepted_tokens,
            num_accepted_tokens,
            actual_request_count,
            graph_request_count,
            _GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT,
        )

    def _pad_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize non-speculative decode metadata at request granularity."""
        # Ordinary decode has one token per live request. Zero-length graph
        # rows (including idle DP batches) must not copy stale state indices.
        actual_request_count = attn_metadata.num_decode_tokens
        state_indices = attn_metadata.non_spec_state_indices_tensor
        query_start_loc = attn_metadata.non_spec_query_start_loc
        assert state_indices is not None
        assert query_start_loc is not None

        attn_metadata.non_spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.non_spec_state_indices_tensor,
            state_indices,
            actual_request_count,
            graph_request_count,
            NULL_BLOCK_ID,
        )
        attn_metadata.non_spec_query_start_loc = _materialize_graph_query_start_loc(
            self.non_spec_query_start_loc,
            query_start_loc,
            actual_request_count,
            graph_request_count,
        )

    def _compute_gdn_local_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> CommonAttentionMetadata:
        """GDN-local query view of the batch, before any plan is derived.

        Two corrections have to land before the batch shape is classified:
        inactive FIA graph rows must become zero-length for the recurrent state
        update, and a one-token stateful prompt chunk must look like an ordinary
        decode so it takes the decode path the state indices are built for.
        """
        if self.use_full_cuda_graph and self.use_spec_decode:
            common_attn_metadata = _remove_spec_graph_padding_queries(
                common_attn_metadata,
                num_decode_draft_tokens_cpu,
            )
        return _treat_single_token_prefills_with_state_as_decodes(common_attn_metadata)

    def _get_gdn_local_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
        batch_shared_cache: dict | None,
    ) -> CommonAttentionMetadata:
        """Correct the batch once per invocation, not once per KV cache group.

        Both corrections rebuild ``query_start_loc``, and
        ``_shared_batch_plan_key`` identifies a batch by tensor address, so
        running them per group would hand every group a distinct view: the plan
        cache would miss on each one and each group would refresh its graph
        buffers from its own copy. Memoize on the same batch-scoped cache the
        plan uses, keyed off the inputs the corrections actually read.

        Cache the corrected fields, never the corrected metadata. The metadata
        also carries ``block_table_tensor`` and ``slot_mapping`` -- the only two
        fields ``build_attn_metadata`` varies per group -- and ``build`` derives
        this group's conv and recurrent state indices from
        ``m.block_table_tensor``. Handing back one group's object would point
        every Mamba group at the first group's block table, so nine of the ten
        groups in a Qwen3.6 hybrid would read and write the wrong state slots.
        """
        if batch_shared_cache is None:
            return self._compute_gdn_local_metadata(common_attn_metadata, num_decode_draft_tokens_cpu)

        ptr = lambda t: None if t is None else t.data_ptr()  # noqa: E731
        key = (
            _GDN_LOCAL_METADATA_CACHE_TAG,
            common_attn_metadata.num_reqs,
            common_attn_metadata.num_actual_tokens,
            ptr(common_attn_metadata.query_start_loc),
            ptr(common_attn_metadata.query_start_loc_cpu),
            ptr(common_attn_metadata.seq_lens_cpu_upper_bound),
            ptr(common_attn_metadata.is_prefilling),
            ptr(num_decode_draft_tokens_cpu),
        )
        corrections = batch_shared_cache.get(key)
        if corrections is None:
            m = self._compute_gdn_local_metadata(common_attn_metadata, num_decode_draft_tokens_cpu)
            corrections = (
                {}
                if m is common_attn_metadata
                else {
                    "query_start_loc": m.query_start_loc,
                    "query_start_loc_cpu": m.query_start_loc_cpu,
                    "num_actual_tokens": m.num_actual_tokens,
                    "is_prefilling": m.is_prefilling,
                }
            )
            batch_shared_cache[key] = corrections
        if not corrections:
            return common_attn_metadata
        # The values are the cached tensors, so every group still presents the
        # same addresses to the plan key below and the plan is computed once.
        return common_attn_metadata.replace(**corrections)

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
            runtime_draft_tokens = num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            if runtime_draft_tokens.sum().item() > 0 or getattr(self, "adaptive_verification_enabled", False):
                torch.ge(
                    num_decode_draft_tokens_cpu,
                    0,
                    out=spec_sequence_masks_cpu,
                )
            else:
                # Dynamic speculative decoding can be enabled while this batch
                # carries no draft tokens. Keep the normal prefill/decode split.
                spec_sequence_masks_cpu.zero_()
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
            if self._USE_COMMON_KERNEL_METADATA:
                non_spec_chunked_prefill_metadata = _build_non_spec_chunked_prefill_metadata(
                    self,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                    build_device_metadata=self.build_device_chunk_metadata,
                )
                # Common GDN/KDA callers also consume the upstream chunk fields.
                chunk_indices = non_spec_chunked_prefill_metadata.chunk_indices_chunk64
                chunk_offsets = non_spec_chunked_prefill_metadata.chunk_offsets_chunk64

        if num_prefills > 0:
            has_initial_state = self._build_prefill_has_initial_state(
                context_lens_tensor=m.compute_num_computed_tokens(),
                non_spec_sequence_indices=non_spec_sequence_indices,
            )
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
        num_actual_reqs: int | None = None,
        batch_shared_cache: dict | None = None,
    ) -> GDNAttentionMetadata:
        graph_request_count = common_attn_metadata.num_reqs
        if num_actual_reqs is None:
            num_actual_reqs = graph_request_count
        elif not 0 <= num_actual_reqs <= graph_request_count:
            raise ValueError(
                f"num_actual_reqs ({num_actual_reqs}) must be between 0 and graph_request_count ({graph_request_count})"
            )
        # Normalize FIA dummy queries before dropping their request rows: the
        # padded token count can only be corrected while those rows are visible.
        m = self._get_gdn_local_metadata(
            common_attn_metadata,
            num_decode_draft_tokens_cpu,
            batch_shared_cache,
        )
        if num_actual_reqs < graph_request_count:
            m = m.unpadded(m.num_actual_tokens, num_actual_reqs)
            if num_accepted_tokens is not None:
                num_accepted_tokens = num_accepted_tokens[:num_actual_reqs]
            if num_decode_draft_tokens_cpu is not None:
                num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu[:num_actual_reqs]
        plan = self._get_shared_batch_plan(m, num_accepted_tokens, num_decode_draft_tokens_cpu, batch_shared_cache)

        num_prefills = plan.num_prefills
        num_decodes = plan.num_decodes
        num_decode_tokens = plan.num_decode_tokens
        num_prefill_tokens = plan.num_prefill_tokens
        num_spec_decodes = plan.num_spec_decodes
        num_spec_decode_tokens = plan.num_spec_decode_tokens
        # Reset each group's captured buffers even when its batch plan is shared.
        if self.use_full_cuda_graph and self.use_spec_decode and num_spec_decodes == 0:
            reset_rows = self.gdn_request_axis if self.ragged_spec_decode else graph_request_count
            self._reset_spec_decode_graph_inputs(reset_rows)
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
        )
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and self._can_pad_spec_decode(graph_request_count, num_spec_decode_tokens)
            and (not self.ragged_spec_decode or self.gdn_request_axis_fits_graph)
        ):
            spec_batch_size = self.gdn_request_axis if self.ragged_spec_decode else graph_request_count
            self._pad_spec_decode_metadata(attn_metadata, spec_batch_size)
        elif (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and graph_request_count <= self.decode_cudagraph_max_bs
        ):
            self._pad_decode_metadata(attn_metadata, graph_request_count)
            non_spec_conv1d_cache_indices = attn_metadata.non_spec_state_indices_tensor
        # Attach once, using the final graph-stable Tensor references.
        self._attach_non_spec_prefill_metadata(
            attn_metadata, non_spec_chunked_prefill_metadata, non_spec_conv1d_cache_indices
        )
        attn_metadata.use_dcut_varlen = getattr(self, "adaptive_verification_enabled", False)
        self._attach_spec_decode_metadata(attn_metadata)
        self._attach_non_spec_decode_metadata(attn_metadata, non_spec_conv1d_cache_indices)
        return attn_metadata

    def _build_prefill_has_initial_state(
        self,
        *,
        context_lens_tensor: torch.Tensor,
        non_spec_sequence_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        has_initial_state = context_lens_tensor > 0
        if non_spec_sequence_indices is not None:
            has_initial_state = torch.index_select(
                has_initial_state,
                0,
                non_spec_sequence_indices,
            )
        return has_initial_state


class AscendGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        return AscendGDNAttentionMetadataBuilder


class AscendGDNHostMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """Build only the descriptors consumed by AscendC host-metadata paths."""

    build_device_chunk_metadata = False


class AscendGDNHostMetadataBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNHostMetadataBuilder]:
        return AscendGDNHostMetadataBuilder


class AscendGDNFusedAttentionBackend(AscendGDNAttentionBackend):
    """Select host-only metadata when Qwen can use a fused prefill op."""

    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        # Import lazily to avoid the gdn <-> builder import cycle at module
        # initialization.  The probe result is cached by the attention class.
        from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

        if get_pcp_group().world_size == 1 and AscendGatedDeltaNetAttention._supports_host_metadata_prefill():
            return AscendGDNHostMetadataBuilder
        return AscendGDNAttentionMetadataBuilder
