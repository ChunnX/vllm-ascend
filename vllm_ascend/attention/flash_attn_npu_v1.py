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
"""Parallel-drafting draft attention on flash-attention-npu v4.

Same problem as the FIA sink backend, and the same shape of answer: a
DSpark/DFlash draft reads a KV length that only exists on device, so the
ordinary FIA entry point's host-side length list costs a sync in every draft
metadata build. ``flash_attn_npu_4`` takes ``cu_seqlens_q`` and ``seqused_k`` as
device tensors and runs its tiling through an AICPU kernel
(``get_scheduler_metadata``), so the draft keeps them where they are.

What this adds beyond the sink backend is head-size reach. The sink operator
serves a fixed set of head sizes; above the largest of them a head_dim 256
draft has nowhere to go. This backend claims exactly that gap, and declines any
layer the sink operator already serves while the sink is enabled -- that is the
path with hardware runs behind it.

A backend of its own, for the reasons the sink backend's docstring gives: the
builder leaves the host-side sequence lists unset and every consumer downstream
follows from that one decision, so a flag inside the shared backend would mean
keeping each of those conditions in sync by hand.

Only the v4 interface is served. The wheel picks its exported API from the
device name at import time, so a build for another device exports a different
set and is reported as such rather than failing later as an attribute error.
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

_FA_NPU_META_CACHE_ATTR = "_ascend_fa_npu_meta_cache"
_MODULE_NAME = "flash_attn_npu_4"
_GENERATION = "v4"
_REQUIRED_ATTRS = ("flash_attn_varlen_func", "get_scheduler_metadata")
_loaded_modules: dict[str, object] = {}

# Head sizes npu_fused_infer_attention_sink serves. Inside this set the sink
# operator keeps the layer whenever it is enabled, so this backend takes only
# what the sink cannot.
FIA_SINK_HEAD_SIZES = (128, 192, 512)
# Above this the forward refuses the call outright.
FA_NPU_MAX_HEAD_SIZE = 256

# Which flash-attention-npu API this process routes draft attention to, read
# once at import so the selection predicate stays a pure function of the
# selector config. Empty disables the backend.
_SELECTED = (envs_ascend.VLLM_ASCEND_DSPARK_FLASH_ATTN_NPU or "").strip().lower()
# Mirrors the sink module's own read of its flag, so both halves of the
# head-size split are decided from constants fixed at import.
_FIA_SINK_ENABLED = bool(envs_ascend.VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK)


def enabled() -> bool:
    """Whether this process routes draft attention to the wheel.

    A value that is neither empty nor the API this backend targets raises rather
    than silently disabling it: an unrecognised setting is a request that cannot
    be served, not a request to serve nothing, and a typo would otherwise look
    exactly like the flag working.
    """
    if not _SELECTED:
        return False
    if _SELECTED != _GENERATION:
        raise RuntimeError(
            f"VLLM_ASCEND_DSPARK_FLASH_ATTN_NPU={_SELECTED!r} is not an API this "
            f"backend serves. Expected {_GENERATION!r}, or unset to disable it."
        )
    return True


def _load():
    """Import the wheel once, failing with something actionable."""
    cached = _loaded_modules.get(_MODULE_NAME)
    if cached is not None:
        return cached

    try:
        module = importlib.import_module(_MODULE_NAME)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"VLLM_ASCEND_DSPARK_FLASH_ATTN_NPU={_GENERATION} requires the "
            "flash-attn-npu wheel built with "
            f"FLASH_ATTN_BUILD_VERSION={_GENERATION} for Ascend910. Install it and "
            "source the matching CANN environment."
        ) from exc

    missing = [name for name in _REQUIRED_ATTRS if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"{_MODULE_NAME} imported but does not expose {', '.join(missing)}. The "
            f"Ascend910 {_GENERATION} interface provides them; a build for another "
            "device does not."
        )
    _loaded_modules[_MODULE_NAME] = module
    return module


def _get_or_compute_inputs(
    cache_key: tuple,
    compute: Callable[[], tuple[torch.Tensor, torch.Tensor, object]],
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """Compute device seq tensors and scheduler metadata once per signature.

    Mirrors the sink backend's cache and for the same reason: during aclgraph
    capture the first layer records the conversion and the AICPU metadata
    launch, later layers reuse the same tensors, and replay reruns one metadata
    launch per signature with every captured address stable. Eager forwards get
    a fresh context-local cache each step. The cache is separate from the sink
    one because the payload is -- ``cu_seqlens_q`` here is int32 and B+1 long.
    """
    forward_context = get_forward_context()
    cache = getattr(forward_context, _FA_NPU_META_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(forward_context, _FA_NPU_META_CACHE_ATTR, cache)
    if cache_key not in cache:
        cache[cache_key] = compute()
    return cache[cache_key]


def flash_attn_npu_selected(attn_selector_config: object) -> bool:
    """Whether this layer's attention should be routed to flash-attention-npu.

    Reads only fields of ``AttentionSelectorConfig``, which is part of the key
    ``_cached_get_attn_backend`` memoizes on -- a predicate that reached for
    ``get_current_vllm_config()`` would be answered once and reused for every
    later config that hashed the same.

    The layer test is the sink backend's, for the same reasons: ``use_non_causal``
    is what upstream sets for a parallel-drafting draft, and sliding-window or
    learnable-sink layers are excluded because this call passes
    ``causal=False, window_size=(-1, -1)`` and no sink tensor.

    On top of that, ``head_size``. Above ``FA_NPU_MAX_HEAD_SIZE`` the forward
    refuses the call, and inside the sink operator's own set the sink operator
    keeps the layer. So this backend claims exactly the gap -- which is where a
    head_dim 256 draft falls.
    """
    if not enabled():
        return False
    if not getattr(attn_selector_config, "use_non_causal", False):
        return False
    if getattr(attn_selector_config, "has_sliding_window", False):
        return False
    if getattr(attn_selector_config, "has_sink", False):
        return False
    head_size = getattr(attn_selector_config, "head_size", 0)
    if not 0 < head_size <= FA_NPU_MAX_HEAD_SIZE:
        # The flag is set and the layer is otherwise the right kind, so someone
        # expects this to be used. Saying why it is not beats a silent fallback
        # that looks identical to the flag having no effect.
        logger.info_once(
            "Ascend flash-attention-npu is enabled but head_size %s is outside the "
            "(0, %d] the forward accepts; this layer keeps the ordinary backend",
            head_size,
            FA_NPU_MAX_HEAD_SIZE,
        )
        return False
    if head_size in FIA_SINK_HEAD_SIZES and _FIA_SINK_ENABLED:
        logger.info_once(
            "Ascend flash-attention-npu is enabled but head_size %d is served by the "
            "FIA sink operator, which is also enabled and keeps the layer; unset "
            "VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK to route it to the wheel instead",
            head_size,
        )
        return False
    return True


class AscendFlashAttnV4MetadataBuilder(AscendAttentionMetadataBuilder):
    """Builds draft metadata that keeps the sequence lengths on device."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # `flash_attn_npu_selected` keys off `use_non_causal`, which a target
        # model can also carry (DiffusionGemma). Only a parallel-drafting draft
        # has the uniform query shape the forward derives its partition from, so
        # refuse the layer here, where the full config is in hand, rather than
        # producing quietly wrong offsets.
        speculative_config = vllm_config.speculative_config
        if not (speculative_config is not None and getattr(speculative_config, "parallel_drafting", False)):
            raise RuntimeError(
                "The Ascend flash-attention-npu backend serves parallel-drafting "
                "(DSpark / DFlash) draft attention, but these layers belong to a model "
                f"without parallel drafting: {layer_names}. Unset "
                "VLLM_ASCEND_DSPARK_FLASH_ATTN_NPU."
            )

        # Fail at construction if the wheel is missing, rather than on the first
        # forward of a served request.
        module = _load()

        # Name the build, not just the generation. A wheel under site-packages
        # and a source tree that shadows it are different binaries, and the only
        # symptom of picking the wrong one is the operator behaving like an older
        # version of itself.
        logger.info(
            "Ascend flash-attention-npu backend (%s) selected for %d %s draft attention layer(s) (head_size=%s): %s",
            _GENERATION,
            len(layer_names),
            getattr(speculative_config, "method", "parallel-drafting"),
            getattr(kv_cache_spec, "head_size", "unknown"),
            layer_names,
        )
        logger.info(
            "Ascend flash-attention-npu %s loaded from %s",
            _GENERATION,
            getattr(module, "__file__", "<unknown>"),
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

        Identical in intent to ``AscendFIASinkMetadataBuilder._build_fia_seq_inputs``
        -- the base builder calls ``.tolist()`` on both, which for this draft is a
        device-to-host sync on a value the verify kernel has only just written,
        and ``seq_lens_list`` being None is also what keeps these layers out of
        the per-step graph task update loop.

        Causality is per build, not per layer: a DFlash draft can carry a
        different flag for each KV cache group, so one backend's layers see both.
        This call passes ``causal=False``, so a causal group has to keep the
        ordinary path -- and ``AscendFlashAttnV4Impl`` reads the same ``causal``
        field to make the matching choice.
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


class AscendFlashAttnV4Impl(AscendAttentionBackendImpl):
    """Runs draft attention through flash-attention-npu, in eager and in graph."""

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
            # Said once because it is otherwise invisible: the selection log would
            # suggest every one of these layers goes through the wheel when a
            # causal group does not. Anyone reading a run to confirm the operator
            # is in use needs to see which half they are looking at.
            logger.info_once(
                "Ascend flash-attention-npu: causal KV cache group falls back to the "
                "ordinary FIA path (the wheel is called with causal=False only)"
            )
            return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return self._forward_flash_attn_npu(query, key, value, attn_metadata, output, kv_cache)

    def _forward_flash_attn_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ) -> torch.Tensor:
        """Parallel-drafting (DSpark/DFlash) attention via flash-attention-npu.

        ``get_scheduler_metadata`` runs the AICPU tiling kernel over the
        device-side ``cu_seqlens_q`` / ``seqused_k``, so no ``seq_lens.tolist()``
        is needed, and the forward given that blob takes the no-host-work branch.
        Both are issued inline so aclgraph captures them together and replay
        re-reads the draft's stable device buffers.
        """
        module = _load()

        if self.key_cache is None and kv_cache is not None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.key_cache is None:
            raise RuntimeError("key_cache is None in _forward_flash_attn_npu")

        # The wheel's paged layout is (num_blocks, page_size, num_kv_heads,
        # head_size), which is the Ascend KV cache shape already -- no view
        # needed, unlike the sink operator's BnBsH.
        _, block_size, _, _ = self.key_cache.shape
        key_cache = self.key_cache
        value_cache = self.value_cache

        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens]

        # The request count comes from query_start_loc's length, and the offsets
        # are rebuilt rather than read from it. Both halves matter, and they come
        # from different places for different reasons.
        #
        # The length, because the builder leaves seq_lens as the whole device
        # buffer for a parallel-drafting draft, so seq_lens.shape[0] is the buffer
        # size rather than the request count.
        #
        # The offsets rebuilt, because query_start_loc's values are the producer's,
        # and the producer pads: entries past the real requests repeat the last
        # boundary. A parallel-drafting draft runs a uniform query per request, so
        # the true partition follows from the token count and the request count,
        # which is what the FIA sink backend has always done and what it is
        # correct with. Reading the values instead was wrong, and quiet, since
        # nothing downstream validates the partition against the query it slices.
        num_reqs = attn_metadata.query_start_loc.shape[0] - 1
        if num_reqs <= 0 or num_tokens % num_reqs != 0:
            raise RuntimeError(
                "Parallel-drafting flash-attention-npu requires a non-empty uniform "
                f"query batch: num_tokens={num_tokens}, num_reqs={num_reqs}"
            )
        query_tokens_per_req = num_tokens // num_reqs

        block_table = attn_metadata.block_tables
        if block_table.shape[0] < num_reqs:
            raise RuntimeError(
                "Parallel-drafting flash-attention-npu block table has fewer rows than "
                f"requests: rows={block_table.shape[0]}, num_reqs={num_reqs}"
            )
        block_table = block_table[:num_reqs]
        if not block_table.is_contiguous():
            # The kernel reaches row b at `b * maxNumBlocksPerBatch`, and
            # maxNumBlocksPerBatch is derived from block_table.shape[1] below. A
            # column-sliced view keeps that shape while its rows are further
            # apart, so every request after the first would read another one's
            # pages -- which looks like low acceptance and repeated output, with
            # no error anywhere. The ordinary FIA path survives such a view
            # because aclnn ops go through a tensor descriptor; this one does not.
            logger.warning_once(
                "Ascend flash-attention-npu: block table is not contiguous "
                "(shape=%s stride=%s); copying. The kernel indexes it by hand and "
                "cannot honour a stride that disagrees with the row length.",
                tuple(block_table.shape),
                tuple(block_table.stride()),
            )
            block_table = block_table.contiguous()

        # `get_scheduler_metadata` derives the block-table row stride the kernel
        # indexes with as ceil(max_seqlen_k / page_size), so max_seqlen_k has to
        # be the page capacity this block table was allocated at, not the actual
        # maximum KV length. Passing the actual max would silently mis-address
        # paged KV under v4, which does not check it.
        max_seqlen_k = block_table.shape[1] * block_size

        # From the same arithmetic as the offsets, so the two cannot disagree.
        max_seqlen_q = query_tokens_per_req

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

        def compute_inputs() -> tuple[torch.Tensor, torch.Tensor, object]:
            # A leading zero and B+1 entries, both int32; get_scheduler_metadata
            # checks the dtype of each. The KV lengths are sliced to the batch and
            # clamped, since a padded request carries zero and the tiling takes
            # the maximum across all of them.
            cu_seqlens_q = (
                torch.arange(num_reqs + 1, dtype=torch.int32, device=attn_metadata.seq_lens.device)
                * query_tokens_per_req
            )
            seqused_k = attn_metadata.seq_lens[:num_reqs].to(torch.int32).clamp_min(1)
            scheduler_metadata = module.get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=max_seqlen_q,
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

        cu_seqlens_q, seqused_k, scheduler_metadata = _get_or_compute_inputs(cache_key, compute_inputs)

        # The selection log says a layer chose this backend; this says the
        # operator ran, and on what. Once per process -- it is evidence, not
        # telemetry, and every draft layer of every step comes through here.
        logger.info_once(
            "Ascend flash-attention-npu %s forward: q=%s kv_cache=%s page_table=%s "
            "num_reqs=%d heads=%d/%d head_size=%d block_size=%d max_seqlen_q=%d "
            "max_seqlen_k=%d causal=False block_table_stride=%s kv_cache_contig=%s "
            "q_contig=%s",
            _GENERATION,
            tuple(query.shape),
            tuple(key_cache.shape),
            tuple(block_table.shape),
            num_reqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            block_size,
            max_seqlen_q,
            max_seqlen_k,
            tuple(block_table.stride()),
            key_cache.is_contiguous(),
            query.is_contiguous(),
        )

        attn_output = module.flash_attn_varlen_func(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=block_table,
            max_seqlen_q=max_seqlen_q,
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


class AscendFlashAttnV4Backend(AscendAttentionBackend):
    """`AscendAttentionBackend` with the draft's flash-attention-npu builder and impl.

    Everything that decides KV cache layout -- ``get_kv_cache_shape``,
    ``get_required_kv_cache_layout``, ``indexes_kv_by_block_stride`` -- is
    inherited unchanged and deliberately so, exactly as for the sink backend: the
    draft shares the target's cache pool, and ``get_required_kv_cache_layout`` is
    applied through a process-global setter, so a second layout here would not
    stay on this backend's layers. The wheel wants that same layout.
    """

    # get_name is deliberately not overridden. vLLM resolves it as an
    # AttentionBackendEnum member, so a name of this backend's own raises
    # "Unknown attention backend" before a single layer is built. It is a
    # registry key, not an identity; the identity is in the logs.

    @staticmethod
    def get_impl_cls() -> type["AscendFlashAttnV4Impl"]:
        return AscendFlashAttnV4Impl

    @staticmethod
    def get_builder_cls() -> type["AscendFlashAttnV4MetadataBuilder"]:
        return AscendFlashAttnV4MetadataBuilder
