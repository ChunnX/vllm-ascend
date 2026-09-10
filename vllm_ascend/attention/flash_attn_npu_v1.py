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
"""Parallel-drafting draft attention on the flash-attention-npu wheel.

Same problem as `fia_sink_v1.py`, different operator. A DSpark/DFlash draft reads
a KV length that only exists on device -- the scheduled length minus the tokens
this step rejected -- so serving it through an entry point that wants a host-side
list costs a device-to-host sync every metadata build.
`npu_fused_infer_attention_sink` solved that by taking the lengths as device
tensors and tiling on AICPU. It only serves head sizes 128, 192 and 512, which
leaves a model with head_dim 256 without a path.

flash-attention-npu has the same property through a different door.
`get_scheduler_metadata` runs an AICPU kernel over the device-side `cu_seqlens_q`
and `cache_seqlens` and writes a tiling blob that the forward then consumes
without touching the host. And its forward does not bin head_dim into a kernel
template -- it is a runtime field of that blob -- so everything up to its
`head_size <= 256` check is one code path.

This calls the v4 API. The repo maintains v2, v3 and v4 side by side and v4 is the
inference-shaped fork of v3, not its successor: it drops the in-kernel KV append
(`k_new`/`v_new`) a framework managing its own KV cache never uses. v3 was
implemented here too while the choice was open, and the measurements on 910B4
settled it -- see the note above `_MODULE_NAME`.

Enable with VLLM_ASCEND_DSPARK_FLASH_ATTN_NPU=v4; unset disables the backend.

Everything downstream of "the builder leaves the host-side sequence lists unset"
is the sink backend's, for the sink backend's reasons: the metadata, the forward,
the full-graph replay path that skips layers without a `seq_lens_list`, the
per-forward metadata cache that keeps captured addresses stable, and the per-build
causal fallback for a DFlash draft whose KV cache groups disagree.

The classes say V4 because that is the wheel API they call, and the module says
`_v1` because that is this package's suffix for a v1-engine backend, as in
`attention_v1.py` and `mla_v1.py`. Not to be read against `fa3_v1.py`, which is a
different backend on a different wheel entirely (`flash_attn_npu_v3`, not
`flash_attn_npu_3`); the two are unrelated and neither succeeds the other.
"""

import importlib
from collections.abc import Callable
from typing import Any

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
_loaded_modules: dict[str, Any] = {}
# Query offsets keyed by (num_reqs, query_tokens_per_req, device); see _cu_seqlens_q.
_CU_SEQLENS_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}

# Head sizes `npu_fused_infer_attention_sink` serves. A layer outside this set is
# what this backend exists for; a layer inside it keeps the sink operator, which
# is the path that has been run on hardware.
FIA_SINK_HEAD_SIZES = (128, 192, 512)

# `mha_fwd` checks `head_size_og <= 256`, and head_dim is a tiling field rather
# than a template axis, so everything at or below that bound is the same kernel.
FA_NPU_MAX_HEAD_SIZE = 256


# The wheel API this backend targets. flash-attention-npu maintains v2, v3 and v4
# side by side; v4 is the inference-shaped fork of v3, dropping the in-kernel KV
# append a framework that manages its own KV cache never uses.
#
# v3 was implemented here too while the choice was open, and measured on 910B4:
# the same accuracy, the same attention kernel within noise, a tiling op 38%
# slower (39.4us against 28.5us), and wrong results whenever the tiling turns
# flash decode on -- zeros or garbage, 360x the noise floor, at single-request
# long-context, which is the latency case a draft exists for. It was removed once
# that settled. `benchmarks/scripts/compare_draft_attention.py` still carries both,
# which is where a comparison belongs; a serving backend should do one thing.
# `docs/v3-metadata-flash-decode.md` in the flash-attention-npu tree has the
# details, and git history has the adapter if 950 ever forces the question -- v4
# has no metadata op there at all.
_MODULE_NAME = "flash_attn_npu_4"
_GENERATION = "v4"
_REQUIRED_ATTRS = ("flash_attn_varlen_func", "get_scheduler_metadata")

_SELECTED = (envs_ascend.VLLM_ASCEND_DSPARK_FLASH_ATTN_NPU or "").strip().lower()
# Mirrors the sink module's own read of its flag, so both halves of the head-size
# split are decided from constants fixed at import.
_FIA_SINK_ENABLED = bool(envs_ascend.VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK)


def enabled() -> bool:
    """Whether this process routes draft attention to the wheel.

    A value that is neither empty nor the API this backend targets raises rather
    than silently disabling it: an unrecognised setting is a request that cannot be
    served, not a request to serve nothing, and a typo would otherwise look exactly
    like the flag working.
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
        # The package picks its interface from the device name at import time, so a
        # build for another device exports a different set -- say that rather than
        # letting it fail later as an attribute error.
        raise RuntimeError(
            f"{_MODULE_NAME} imported but does not expose {', '.join(missing)}. The "
            f"Ascend910 {_GENERATION} interface provides them; a build for another "
            "device does not."
        )
    _loaded_modules[_MODULE_NAME] = module
    return module


def _get_or_compute_inputs(
    cache_key: tuple,
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
    cache = getattr(forward_context, _FA_NPU_META_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(forward_context, _FA_NPU_META_CACHE_ATTR, cache)
    if cache_key not in cache:
        cache[cache_key] = compute()
    return cache[cache_key]


def _build_seq_tensors(num_tokens: int, seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build legal device-side TND lengths for uniform parallel-drafting queries.

    Same repair the sink backend makes, in this wheel's dtypes. FULL graph replay
    pads the request bucket: the producer leaves padded `query_start_loc` entries at
    the real-token boundary and padded KV lengths at zero. DSpark/DFlash queries are
    uniform, so cumulative Q lengths come from the static shapes instead, and dummy
    KV lengths map to 1 -- a zero-length request is not a shape the tiling has been
    exercised on, and one padded token of attention is discarded downstream anyway.

    `get_scheduler_metadata` wants `cu_seqlens_q` with a leading zero and B+1
    entries, and TORCH_CHECKs the dtype of both as int32.
    """
    num_reqs = seq_lens.shape[0]
    if num_reqs <= 0 or num_tokens % num_reqs != 0:
        raise RuntimeError(
            "Parallel-drafting flash-attention-npu requires a non-empty uniform query "
            f"batch: num_tokens={num_tokens}, num_reqs={num_reqs}"
        )
    query_tokens_per_req = num_tokens // num_reqs
    cu_seqlens_q = _cu_seqlens_q(num_reqs, query_tokens_per_req, seq_lens.device)
    seqused_k = seq_lens.to(torch.int32).clamp_min(1)
    return cu_seqlens_q, seqused_k


def _cu_seqlens_q(num_reqs: int, query_tokens_per_req: int, device) -> torch.Tensor:
    """The query offsets for a uniform draft batch, built once per shape.

    Profiling put the arange and its multiply at 13.6us of device time per step --
    a third of everything this backend spends outside the attention kernel -- to
    produce a handful of int32 values. Nearly all of it is launch overhead on two
    tiny vector kernels, not arithmetic.

    They do not have to be rebuilt: a parallel-drafting batch has a uniform query
    length, so the offsets are a function of the shape and change only when the
    request count or the draft length does. Caching also gives aclgraph a stable
    address to capture, which the per-step version had only by luck of the
    allocator returning the same block.

    Never handed out for mutation -- the wheel treats cu_seqlens_q as read-only.
    `seqused_k` beside it is still derived fresh each step, because the KV lengths
    genuinely change.
    """
    key = (num_reqs, query_tokens_per_req, str(device))
    cached = _CU_SEQLENS_CACHE.get(key)
    if cached is None:
        cached = torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * query_tokens_per_req
        _CU_SEQLENS_CACHE[key] = cached
    return cached


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

    On top of that, ``head_size``. Above 256 the forward refuses the call, and
    inside the sink operator's own set the sink operator keeps the layer: that is
    the path with hardware runs behind it. So this backend claims exactly the gap
    -- which is where a head_dim 256 model falls.
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

        # `flash_attn_npu_selected` keys off `use_non_causal`, which a target model
        # can also carry (DiffusionGemma). Only a parallel-drafting draft has the
        # uniform query shape `_build_seq_tensors` derives lengths from, so refuse
        # the layer here, where the full config is in hand, rather than producing
        # quietly wrong lengths.
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

        # Name the build, not just the generation. A wheel installed under
        # site-packages and a source tree that shadows it are different binaries,
        # and the only symptom of picking the wrong one is the operator behaving
        # like an older version of itself.
        logger.info(
            "Ascend flash-attention-npu backend (%s) selected for %d %s draft "
            "attention layer(s) (head_size=%s): %s",
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

        Identical in intent to `AscendFIASinkMetadataBuilder._build_fia_seq_inputs`
        -- the base builder calls `.tolist()` on both, which for this draft is a
        device-to-host sync on a value the verify kernel has only just written, and
        `seq_lens_list` being None is also what keeps these layers out of the
        per-step `graph_task_update` loop in `update_graph_params`.

        Causality is per build, not per layer: a DFlash draft can carry a different
        flag for each KV cache group, so one backend's layers see both. This call
        passes `causal=False`, so a causal group has to keep the ordinary path --
        and `AscendFlashAttnV4Impl` reads the same `causal` field to make the
        matching choice.
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
            # Said once because it is otherwise invisible: a DFlash draft can have
            # both kinds of group, and the selection log above would suggest every
            # one of these layers goes through the wheel when a causal group does
            # not. Anyone reading a run to confirm the operator is in use needs to
            # see which half they are looking at.
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

        `get_scheduler_metadata` runs the AICPU tiling kernel over the device-side
        `cu_seqlens_q` / `seqused_k`, so no `seq_lens.tolist()` is needed, and the
        forward given that blob takes the no-host-work branch. Both are issued
        inline so aclgraph captures them together and replay re-reads the draft's
        stable device buffers.
        """
        module = _load()

        if self.key_cache is None and kv_cache is not None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.key_cache is None:
            raise RuntimeError("key_cache is None in _forward_flash_attn_npu")

        # The wheel's paged layout is (num_blocks, page_size, num_kv_heads,
        # head_size), which is the Ascend KV cache shape already -- no view needed,
        # unlike the sink operator's BnBsH.
        _, block_size, _, _ = self.key_cache.shape
        key_cache = self.key_cache
        value_cache = self.value_cache

        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens]

        num_reqs = attn_metadata.seq_lens.shape[0]
        block_table = attn_metadata.block_tables
        if block_table.shape[0] < num_reqs:
            raise RuntimeError(
                "Parallel-drafting flash-attention-npu block table has fewer rows than "
                f"requests: rows={block_table.shape[0]}, num_reqs={num_reqs}"
            )
        block_table = block_table[:num_reqs]

        # `get_scheduler_metadata` derives the block-table row stride the kernel
        # indexes with as ceil(max_seqlen_k / page_size), so max_seqlen_k has to be
        # the page capacity this block table was allocated at, not the actual
        # maximum KV length. Passing the actual max would silently mis-address paged
        # KV under v4, which does not check it; v3 rejects the mismatch.
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

        def compute_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cu_seqlens_q, seqused_k = _build_seq_tensors(num_tokens, attn_metadata.seq_lens)
            scheduler_metadata = module.get_scheduler_metadata(
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

        cu_seqlens_q, seqused_k, scheduler_metadata = _get_or_compute_inputs(cache_key, compute_inputs)

        # The selection log says a layer chose this backend; this says the operator
        # ran, and on what. Once per process -- it is evidence, not telemetry, and
        # every draft layer of every step would come through here.
        logger.info_once(
            "Ascend flash-attention-npu %s forward: q=%s kv_cache=%s page_table=%s "
            "seqused_k=%s heads=%d/%d head_size=%d block_size=%d max_seqlen_q=%d "
            "max_seqlen_k=%d causal=False",
            _GENERATION,
            tuple(query.shape),
            tuple(key_cache.shape),
            tuple(block_table.shape),
            tuple(seqused_k.shape),
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            block_size,
            query_tokens_per_req,
            max_seqlen_k,
        )

        attn_output = module.flash_attn_varlen_func(
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


class AscendFlashAttnV4Backend(AscendAttentionBackend):
    """`AscendAttentionBackend` with the draft's flash-attention-npu builder and impl.

    Everything that decides KV cache layout -- `get_kv_cache_shape`,
    `get_required_kv_cache_layout`, `indexes_kv_by_block_stride` -- is inherited
    unchanged and deliberately so, exactly as for the sink backend: the draft
    shares the target's cache pool, and `get_required_kv_cache_layout` is applied
    through a process-global setter, so a second layout here would not stay on this
    backend's layers. The wheel wants that same layout.
    """

    # get_name is deliberately not overridden. vLLM resolves it as an
    # AttentionBackendEnum member -- `Attention.__init__` does
    # `AttentionBackendEnum[self.attn_backend.get_name()]` -- so a name of this
    # backend's own raises "Unknown attention backend" before a single layer is
    # built. It is a registry key, not an identity; the identity is in the logs.

    @staticmethod
    def get_impl_cls() -> type["AscendFlashAttnV4Impl"]:
        return AscendFlashAttnV4Impl

    @staticmethod
    def get_builder_cls() -> type["AscendFlashAttnV4MetadataBuilder"]:
        return AscendFlashAttnV4MetadataBuilder
