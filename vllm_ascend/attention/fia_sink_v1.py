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
"""Parallel-drafting draft attention on npu_fused_infer_attention_sink.

A DSpark/DFlash draft reads a KV length that is only known on device: it is the
scheduled length minus the tokens this step rejected, and the rejection count is
produced by the verify kernel. The ordinary FIA entry point wants that length as
a host-side list, so serving it costs a device-to-host sync in every draft
metadata build. The sink operator takes the lengths as device tensors and does
its tiling on AICPU, so the draft keeps them where they already are.

This lives in its own backend rather than as a flag on the shared one because
the two differ in what they read, not merely in which kernel they call: the
builder here leaves the host-side sequence lists unset, and every consumer
downstream -- the metadata, the forward, the full-graph replay path that skips
layers without a `seq_lens_list` -- follows from that one decision. As a branch
inside the shared backend, each of those was a separate condition to keep in
sync, and enabling the operator was process-global, so one sliding-window layer
anywhere turned into a hard failure. Which layers get this is now settled once,
where every other Ascend backend is settled: in `NPUPlatform.get_attn_backend_cls`.

A plain subclass, not `subclass_attention_backend`: that helper builds a variant
of whatever backend was selected underneath, which is what ChunkedLocalAttention
and the encoder-only wrapper need. There is exactly one backend underneath this
one -- the sink operator rejects Ascend310P outright -- so naming it directly is
both simpler and more honest about the coupling.
"""

import importlib
from collections.abc import Callable

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.kv_cache_interface import AttentionSpec

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.attention_v1 import (
    SWA_INT_MAX,
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

_FIA_SINK_META_CACHE_ATTR = "_ascend_fia_sink_meta_cache"
_FIA_SINK_REQUIRED_OPS = (
    "_npu_fused_infer_attention_sink_metadata",
    "npu_fused_infer_attention_sink",
)
_fia_sink_ops_registered = False

# When enabled, non-causal parallel-drafting (DSpark / DFlash) attention is
# dispatched to npu_fused_infer_attention_sink, which accepts device-side
# seq_lens and computes tiling on AICPU. Tensor-shape and attention-topology
# capability checks are intentionally delegated to the custom op so this
# integration does not narrow the operator's supported domain to one validated
# model shape.
_FIA_SINK_ENABLED = bool(envs_ascend.VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK)


def _ensure_fia_sink_ops_registered() -> None:
    """Load omni_custom_ops and fail early when its sink ops are unavailable."""
    global _fia_sink_ops_registered
    if _fia_sink_ops_registered:
        return

    try:
        importlib.import_module("omni_custom_ops")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=1 requires the omni_custom_ops "
            "wheel. Install it and source the matching CANN custom OPP environment."
        ) from exc

    missing_ops = [name for name in _FIA_SINK_REQUIRED_OPS if not hasattr(torch.ops.custom, name)]
    if missing_ops:
        raise RuntimeError(
            "omni_custom_ops was imported but the required FIA sink operators "
            f"were not registered: {', '.join(missing_ops)}"
        )
    _fia_sink_ops_registered = True


def _get_or_compute_fia_sink_inputs(
    cache_key: tuple[int, ...],
    compute: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute device seq tensors and metadata once per forward/signature.

    During aclgraph capture the first layer records the conversion, metadata and
    sink dependency. Later layers reuse the same tensors. Graph replay therefore
    reruns one metadata op per signature while keeping all captured addresses
    stable. Eager forwards get a fresh context-local cache on every step.
    """
    forward_context = get_forward_context()
    cache = getattr(forward_context, _FIA_SINK_META_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(forward_context, _FIA_SINK_META_CACHE_ATTR, cache)
    if cache_key not in cache:
        cache[cache_key] = compute()
    return cache[cache_key]


def _build_fia_sink_seq_tensors(
    num_tokens: int,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build legal device-side TND lengths for uniform parallel-drafting queries.

    FULL graph replay pads the request bucket. The producer leaves padded
    query_start_loc entries at the real-token boundary and padded KV lengths at
    zero, neither of which is legal for FIA sink. DSpark/DFlash queries are
    uniform, so derive cumulative Q lengths from static shapes and map dummy KV
    lengths to 1.
    """
    num_reqs = seq_lens.shape[0]
    if num_reqs <= 0 or num_tokens % num_reqs != 0:
        raise RuntimeError(
            "Parallel-drafting FIA sink requires a non-empty uniform query batch: "
            f"num_tokens={num_tokens}, num_reqs={num_reqs}"
        )
    query_tokens_per_req = num_tokens // num_reqs
    actual_seq_qlen = (
        torch.arange(1, num_reqs + 1, dtype=torch.int64, device=seq_lens.device) * query_tokens_per_req
    )
    actual_seq_kvlen = seq_lens.to(torch.int64).clamp_min(1)
    return actual_seq_qlen, actual_seq_kvlen


def fia_sink_selected(attn_selector_config: object) -> bool:
    """Whether this layer's attention should be routed to the sink operator.

    Reads only fields of ``AttentionSelectorConfig``, which is part of the key
    ``_cached_get_attn_backend`` memoizes on. A predicate that reached for
    ``get_current_vllm_config()`` instead would be answered once and reused for
    every later config that hashed the same.

    ``use_non_causal`` is what upstream sets for a parallel-drafting draft
    (``load_dspark_model`` / ``load_dflash_model`` derive it from the draft's own
    hf_config), so it is the draft-vs-target discriminator. It is not exclusively
    that -- DiffusionGemma sets it for a target model -- which is why the builder
    re-checks ``parallel_drafting`` once it has the full config.

    Sliding-window and learnable-sink layers are excluded here rather than
    failing later: the sink call hardcodes ``sparse_mode=0`` and forwards neither
    ``atten_mask`` nor ``learnable_sink``, so it cannot express them. Both are
    already in the selector config, so the exclusion costs nothing at runtime.
    """
    if not _FIA_SINK_ENABLED:
        return False
    if not getattr(attn_selector_config, "use_non_causal", False):
        return False
    if getattr(attn_selector_config, "has_sliding_window", False):
        return False
    if getattr(attn_selector_config, "has_sink", False):
        return False
    return True


class AscendFIASinkMetadataBuilder(AscendAttentionMetadataBuilder):
    """Builds draft metadata that keeps the sequence lengths on device."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # `fia_sink_selected` keys off `use_non_causal`, which a target model can
        # also carry (DiffusionGemma). Only a parallel-drafting draft has the
        # uniform query shape `_build_fia_sink_seq_tensors` derives lengths from,
        # so refuse the layer here, where the full config is in hand, rather than
        # producing quietly wrong lengths.
        speculative_config = vllm_config.speculative_config
        if not (speculative_config is not None and getattr(speculative_config, "parallel_drafting", False)):
            raise RuntimeError(
                "The Ascend FIA sink backend serves parallel-drafting (DSpark / "
                "DFlash) draft attention, but these layers belong to a model "
                f"without parallel drafting: {layer_names}. Unset "
                "VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK."
            )

        # Fail at construction if the operator package is missing, rather than on
        # the first forward of a served request.
        _ensure_fia_sink_ops_registered()

    def _build_fia_seq_inputs(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs: int,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[int] | None, list[int] | None, torch.Tensor, torch.Tensor | None]:
        """Keep the device-side lengths; leave the host-side lists unset.

        The base builder calls `.tolist()` on both, which for this draft is a
        device-to-host sync on a value the verify kernel has only just written.
        `seq_lens_list` being None is also what keeps these layers out of the
        per-step `graph_task_update` loop in `update_graph_params`.

        The padding the base builder does here is not needed: it repairs a
        request bucket that FIA's host-side length list would otherwise
        under-count, and `_build_fia_sink_seq_tensors` derives the padded shape
        from `num_tokens` directly.

        Causality is per build, not per layer: a DFlash draft can carry a
        different flag for each KV cache group (`_group_causal`), so one
        backend's layers see both. The sink call hardcodes ``sparse_mode=0``, so
        a causal group has to keep the ordinary path -- and `AscendFIASinkImpl`
        reads the same ``causal`` field to make the matching choice.
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


class AscendFIASinkImpl(AscendAttentionBackendImpl):
    """Runs draft attention through the sink operator, in eager and in graph."""

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # A DFlash draft can mix causal and non-causal KV cache groups, so this
        # is decided per build rather than per layer. The builder made the same
        # call from the same field: a causal group kept its host-side lengths,
        # which is what the ordinary path below needs.
        if attn_metadata.causal:
            return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        # The sink op is captured inline for aclgraph and re-reads its device
        # inputs on replay, so there is no capture-vs-replay split to make here.
        return self._forward_fia_sink(query, key, value, attn_metadata, output, kv_cache)

    def _forward_fia_sink(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ) -> torch.Tensor:
        """Parallel-drafting (DSpark/DFlash) attention via the sink operator.

        The draft seq_lens depends on the device-side rejected-token count, so a
        host copy is a sync. The sink operator accepts device-side
        actual_seq_qlen / actual_seq_kvlen and computes tiling on AICPU (via
        _npu_fused_infer_attention_sink_metadata), so no seq_lens.tolist() is
        needed. It is captured inline for aclgraph: the draft's seq_lens /
        query_start_loc / block_table are stable device buffers, so the graph
        re-executes the metadata op + sink op reading fresh values at replay.

        The forward-context cache shares converted Q/KV lengths and metadata
        across all layers with the same input signature. During graph capture
        the producing metadata op is recorded once and replayed before its sink
        consumers, so graph-managed tensor addresses remain stable.
        """
        if self.key_cache is None and kv_cache is not None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.key_cache is None:
            raise RuntimeError("key_cache is None in _forward_fia_sink")

        num_block, block_size, _, _ = self.key_cache.shape
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
        block_table = attn_metadata.block_tables

        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens]

        num_reqs = attn_metadata.seq_lens.shape[0]
        if block_table.shape[0] < num_reqs:
            raise RuntimeError(
                "Parallel-drafting FIA sink block table has fewer rows than requests: "
                f"rows={block_table.shape[0]}, num_reqs={num_reqs}"
            )
        block_table = block_table[:num_reqs]

        cache_key = (
            attn_metadata.seq_lens.data_ptr(),
            num_tokens,
            num_reqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            block_size,
        )

        def compute_sink_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            actual_seq_qlen, actual_seq_kvlen = _build_fia_sink_seq_tensors(
                num_tokens,
                attn_metadata.seq_lens,
            )
            stream_limit = torch.npu.get_stream_limit(torch.npu.current_stream())
            meta_data = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                self.head_size,
                actual_seq_lengths=actual_seq_qlen,
                actual_seq_lengths_kv=actual_seq_kvlen,
                batch_size=num_reqs,
                sparse_mode=0,
                pre_tokens=SWA_INT_MAX,
                next_tokens=SWA_INT_MAX,
                input_layout="TND",
                input_layout_kv="BnBsH",
                sink_num=0,
                block_size=block_size,
                aic_core_num=stream_limit["cube_core_num"],
                aiv_core_num=stream_limit["vector_core_num"],
            )
            return actual_seq_qlen, actual_seq_kvlen, meta_data

        actual_seq_qlen, actual_seq_kvlen, meta_data = _get_or_compute_fia_sink_inputs(
            cache_key,
            compute_sink_inputs,
        )

        attn_output, _ = torch.ops.custom.npu_fused_infer_attention_sink(
            query,
            key,
            value,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=block_table,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=self.scale,
            input_layout="TND",
            sparse_mode=0,
            block_size=block_size,
            sink_number=0,
            meta_data=meta_data,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendFIASinkBackend(AscendAttentionBackend):
    """`AscendAttentionBackend` with the draft's sink builder and impl.

    Everything that decides KV cache layout -- `get_kv_cache_shape`,
    `get_required_kv_cache_layout`, `indexes_kv_by_block_stride` -- is inherited
    unchanged and deliberately so. The draft shares the target's cache pool, and
    `get_required_kv_cache_layout` is applied through a process-global setter, so
    a second layout here would not stay on this backend's layers.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_FIA_SINK"

    @staticmethod
    def get_impl_cls() -> type["AscendFIASinkImpl"]:
        return AscendFIASinkImpl

    @staticmethod
    def get_builder_cls() -> type["AscendFIASinkMetadataBuilder"]:
        return AscendFIASinkMetadataBuilder
