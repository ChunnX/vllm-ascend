#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Parallel-drafting draft attention on flash_attn_npu_4.

Same problem as `fia_sink_v1.py`, different operator. A DSpark/DFlash draft
reads a KV length that only exists on device -- the scheduled length minus the
tokens this step rejected -- so serving it through an entry point that wants a
host-side list costs a device-to-host sync every metadata build.
`npu_fused_infer_attention_sink` solved that by taking the lengths as device
tensors and tiling on AICPU. It only serves head sizes 128, 192 and 512, which
leaves a model with head_dim 256 without a path.

flash-attention-npu v4 has the same property through a different door:
`get_scheduler_metadata` runs an AICPU kernel over the device-side
`cu_seqlens_q` / `cache_seqlens` and writes a tiling blob that `mha_fwd` then
consumes without touching the host. And its forward does not bin head_dim into
a kernel template -- it is a runtime field of that tiling blob -- so the whole
range up to 256 is one code path.

Neither operator dominates: the sink op reaches 512, v4 stops at 256. So both
backends stay, and `fa4_selected` claims only the layers the sink op cannot
serve. Turning the sink flag off while leaving this one on hands every
qualifying layer to v4, which is how to A/B the two on a shape both support.

Everything downstream of "the builder leaves the host-side sequence lists unset"
follows exactly as it does for the sink backend -- the metadata, the forward,
the full-graph replay path that skips layers without a `seq_lens_list`.
"""

import importlib
from collections.abc import Callable

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import AttentionSpec

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

_FA4_META_CACHE_ATTR = "_ascend_fa4_meta_cache"
_fa4_module = None

# Head sizes `npu_fused_infer_attention_sink` serves. A layer outside this set is
# what this backend exists for; a layer inside it keeps the sink operator, which
# is the path that has been run on hardware.
FIA_SINK_HEAD_SIZES = (128, 192, 512)

# `mha_fwd` checks `head_size_og <= 256`; head_dim is a tiling field rather than a
# template axis, so everything at or below that bound is the same kernel.
FA4_MAX_HEAD_SIZE = 256

_FA4_ENABLED = bool(envs_ascend.VLLM_ASCEND_ENABLE_DSPARK_FA4)
# Mirrors the sink module's own read of its flag. Held here so both halves of
# the head-size split are decided from constants fixed at import, rather than
# one at import and one per call.
_FIA_SINK_ENABLED = bool(envs_ascend.VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK)


def _load_fa4():
    """Import flash_attn_npu_4 once, failing with something actionable."""
    global _fa4_module
    if _fa4_module is not None:
        return _fa4_module

    try:
        module = importlib.import_module("flash_attn_npu_4")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_DSPARK_FA4=1 requires the flash-attn-npu wheel built "
            "with FLASH_ATTN_BUILD_VERSION=v4 for Ascend910. Install it and source the "
            "matching CANN environment."
        ) from exc

    missing = [name for name in ("flash_attn_varlen_func", "get_scheduler_metadata") if not hasattr(module, name)]
    if missing:
        # The package picks its interface from the device name at import time, so a
        # 950 build exports neither of these -- say that rather than "attribute error".
        raise RuntimeError(
            "flash_attn_npu_4 imported but does not expose "
            f"{', '.join(missing)}. The Ascend910 v4 interface provides both; a build "
            "for another device does not."
        )
    _fa4_module = module
    return module


def _get_or_compute_fa4_inputs(
    cache_key: tuple[int, ...],
    compute: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute device seq tensors and scheduler metadata once per forward/signature.

    Mirrors the sink backend's cache and for the same reason: during aclgraph
    capture the first layer records the conversion and the AICPU metadata launch,
    later layers reuse the same tensors, and replay reruns one metadata launch per
    signature with every captured address stable. Eager forwards get a fresh
    context-local cache each step. The cache is separate from the sink one because
    the payload is -- `cu_seqlens_q` here is int32 and B+1 long.
    """
    forward_context = get_forward_context()
    cache = getattr(forward_context, _FA4_META_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(forward_context, _FA4_META_CACHE_ATTR, cache)
    if cache_key not in cache:
        cache[cache_key] = compute()
    return cache[cache_key]


def _build_fa4_seq_tensors(num_tokens: int, seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build legal device-side TND lengths for uniform parallel-drafting queries.

    Same repair the sink backend makes, in v4's dtypes. FULL graph replay pads the
    request bucket: the producer leaves padded `query_start_loc` entries at the
    real-token boundary and padded KV lengths at zero. DSpark/DFlash queries are
    uniform, so cumulative Q lengths come from the static shapes instead, and dummy
    KV lengths map to 1 -- a zero-length request is not a shape v4's tiling has been
    exercised on, and one padded token of attention is discarded downstream anyway.

    v4 wants `cu_seqlens_q` with a leading zero and B+1 entries, both tensors int32
    (`get_scheduler_metadata` TORCH_CHECKs the dtype of each).
    """
    num_reqs = seq_lens.shape[0]
    if num_reqs <= 0 or num_tokens % num_reqs != 0:
        raise RuntimeError(
            "Parallel-drafting FA4 requires a non-empty uniform query batch: "
            f"num_tokens={num_tokens}, num_reqs={num_reqs}"
        )
    query_tokens_per_req = num_tokens // num_reqs
    cu_seqlens_q = (
        torch.arange(num_reqs + 1, dtype=torch.int32, device=seq_lens.device) * query_tokens_per_req
    )
    seqused_k = seq_lens.to(torch.int32).clamp_min(1)
    return cu_seqlens_q, seqused_k


def fa4_selected(attn_selector_config: object) -> bool:
    """Whether this layer's attention should be routed to flash_attn_npu_4.

    Reads only fields of ``AttentionSelectorConfig``, which is part of the key
    ``_cached_get_attn_backend`` memoizes on -- a predicate that reached for
    ``get_current_vllm_config()`` would be answered once and reused for every
    later config that hashed the same.

    The layer test is the sink backend's, for the same reasons: ``use_non_causal``
    is what upstream sets for a parallel-drafting draft, and sliding-window or
    learnable-sink layers are excluded because this call passes
    ``causal=False, window_size=(-1, -1)`` and no sink tensor -- v4 rejects
    ``learnable_sink`` outright.

    On top of that, ``head_size``. Above 256 v4's own ``mha_fwd`` refuses the
    call, and inside the sink operator's own set the sink operator keeps the
    layer: that is the path with hardware runs behind it. So this backend claims
    exactly the gap -- which is what a head_dim 256 model falls into.
    """
    if not _FA4_ENABLED:
        return False
    if not getattr(attn_selector_config, "use_non_causal", False):
        return False
    if getattr(attn_selector_config, "has_sliding_window", False):
        return False
    if getattr(attn_selector_config, "has_sink", False):
        return False
    head_size = getattr(attn_selector_config, "head_size", 0)
    if not 0 < head_size <= FA4_MAX_HEAD_SIZE:
        return False
    if head_size in FIA_SINK_HEAD_SIZES and _FIA_SINK_ENABLED:
        return False
    return True


class AscendFA4MetadataBuilder(AscendAttentionMetadataBuilder):
    """Builds draft metadata that keeps the sequence lengths on device."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # `fa4_selected` keys off `use_non_causal`, which a target model can also
        # carry (DiffusionGemma). Only a parallel-drafting draft has the uniform
        # query shape `_build_fa4_seq_tensors` derives lengths from, so refuse the
        # layer here, where the full config is in hand, rather than producing
        # quietly wrong lengths.
        speculative_config = vllm_config.speculative_config
        if not (speculative_config is not None and getattr(speculative_config, "parallel_drafting", False)):
            raise RuntimeError(
                "The Ascend FA4 backend serves parallel-drafting (DSpark / DFlash) "
                "draft attention, but these layers belong to a model without parallel "
                f"drafting: {layer_names}. Unset VLLM_ASCEND_ENABLE_DSPARK_FA4."
            )

        # Fail at construction if the wheel is missing, rather than on the first
        # forward of a served request.
        _load_fa4()

        logger.info(
            "Ascend FA4 backend selected for %d %s draft attention layer(s) "
            "(head_size=%s): %s",
            len(layer_names),
            getattr(speculative_config, "method", "parallel-drafting"),
            getattr(kv_cache_spec, "head_size", "unknown"),
            layer_names,
        )

    def _build_fia_seq_inputs(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs: int,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[int] | None, list[int] | None, torch.Tensor, torch.Tensor | None]:
        """Keep the device-side lengths; leave the host-side lists unset.

        Identical in intent to `AscendFIASinkMetadataBuilder._build_fia_seq_inputs`
        -- the base builder calls `.tolist()` on both, which for this draft is a
        device-to-host sync on a value the verify kernel has only just written, and
        `seq_lens_list` being None is also what keeps these layers out of the
        per-step `graph_task_update` loop in `update_graph_params`.

        Causality is per build, not per layer: a DFlash draft can carry a different
        flag for each KV cache group, so one backend's layers see both. This call
        passes `causal=False`, so a causal group has to keep the ordinary path --
        and `AscendFA4Impl` reads the same `causal` field to make the matching
        choice.
        """
        if common_attn_metadata.causal:
            return super()._build_fia_seq_inputs(
                common_attn_metadata,
                num_reqs,
                query_start_loc_cpu,
                seq_lens,
                block_table,
            )

        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        return query_start_loc, None, None, seq_lens, block_table


class AscendFA4Impl(AscendAttentionBackendImpl):
    """Runs draft attention through flash_attn_npu_4, in eager and in graph."""

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # A DFlash draft can mix causal and non-causal KV cache groups, so this is
        # decided per build rather than per layer. The builder made the same call
        # from the same field: a causal group kept its host-side lengths, which is
        # what the ordinary path below needs.
        if attn_metadata.causal:
            return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return self._forward_fa4(query, key, value, attn_metadata, output, kv_cache)

    def _forward_fa4(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ) -> torch.Tensor:
        """Parallel-drafting (DSpark/DFlash) attention via flash_attn_npu_4.

        `get_scheduler_metadata` runs the AICPU tiling kernel over the device-side
        `cu_seqlens_q` / `seqused_k`, so no `seq_lens.tolist()` is needed, and
        `flash_attn_varlen_func` given that blob takes the no-host-work branch of
        `mha_fwd`. Both are issued inline so aclgraph captures them together and
        replay re-reads the draft's stable device buffers.
        """
        fa4 = _load_fa4()

        if self.key_cache is None and kv_cache is not None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.key_cache is None:
            raise RuntimeError("key_cache is None in _forward_fa4")

        # v4's paged layout is (num_blocks, page_size, num_kv_heads, head_size),
        # which is the Ascend KV cache shape already -- no view needed, unlike the
        # sink operator's BnBsH.
        _, block_size, _, _ = self.key_cache.shape
        key_cache = self.key_cache
        value_cache = self.value_cache

        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens]

        num_reqs = attn_metadata.seq_lens.shape[0]
        block_table = attn_metadata.block_tables
        if block_table.shape[0] < num_reqs:
            raise RuntimeError(
                "Parallel-drafting FA4 block table has fewer rows than requests: "
                f"rows={block_table.shape[0]}, num_reqs={num_reqs}"
            )
        block_table = block_table[:num_reqs]

        # `get_scheduler_metadata` derives the block-table row stride the kernel
        # indexes with as ceil(max_seqlen_k / page_size), so max_seqlen_k has to be
        # the page capacity this block table was allocated at, not the actual
        # maximum KV length. Passing the actual max would silently mis-address
        # paged KV, and nothing on either side checks it.
        max_seqlen_k = block_table.shape[1] * block_size
        query_tokens_per_req = num_tokens // num_reqs if num_reqs else 0

        cache_key = (
            attn_metadata.seq_lens.data_ptr(),
            num_tokens,
            num_reqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            block_size,
            max_seqlen_k,
        )

        def compute_fa4_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cu_seqlens_q, seqused_k = _build_fa4_seq_tensors(num_tokens, attn_metadata.seq_lens)
            scheduler_metadata = fa4.get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=query_tokens_per_req,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=self.num_heads,
                num_heads_kv=self.num_kv_heads,
                headdim=self.head_size,
                cache_seqlens=seqused_k,
                qkv_dtype=query.dtype,
                headdim_v=self.head_size,
                cu_seqlens_q=cu_seqlens_q,
                page_size=block_size,
                causal=False,
                window_size=(-1, -1),
                softmax_scale=self.scale,
            )
            return cu_seqlens_q, seqused_k, scheduler_metadata

        cu_seqlens_q, seqused_k, scheduler_metadata = _get_or_compute_fa4_inputs(cache_key, compute_fa4_inputs)

        attn_output = fa4.flash_attn_varlen_func(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=block_table,
            max_seqlen_q=query_tokens_per_req,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            window_size=(-1, -1),
            scheduler_metadata=scheduler_metadata,
            num_splits=0,
            return_lse=False,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendFA4Backend(AscendAttentionBackend):
    """`AscendAttentionBackend` with the draft's FA4 builder and impl.

    Everything that decides KV cache layout -- `get_kv_cache_shape`,
    `get_required_kv_cache_layout`, `indexes_kv_by_block_stride` -- is inherited
    unchanged and deliberately so, exactly as for the sink backend: the draft
    shares the target's cache pool, and `get_required_kv_cache_layout` is applied
    through a process-global setter, so a second layout here would not stay on
    this backend's layers. v4 happens to want that same layout.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_FA4"

    @staticmethod
    def get_impl_cls() -> type["AscendFA4Impl"]:
        return AscendFA4Impl

    @staticmethod
    def get_builder_cls() -> type["AscendFA4MetadataBuilder"]:
        return AscendFA4MetadataBuilder
