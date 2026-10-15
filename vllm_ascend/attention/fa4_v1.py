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
"""FA4 AICPU tiling for non-causal parallel-drafting (DSpark / DFlash).

VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2 selects this backend.
Metadata and attention are captured together so replay reads the rejected-token
adjusted KV lengths on device. Physical page strides are encoded in FA4 page
IDs without repacking the shared cache, preserving cache-group isolation.
"""

import importlib
from types import ModuleType

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

_FA4_META_CACHE_ATTR = "_ascend_fa4_meta_cache"
_FA4_ENABLED = envs_ascend.VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK == 2


def _load_fa4() -> ModuleType:
    # Lazy import: the package inspects the NPU and loads its extension. Do not
    # initialize it in the driver merely to select a backend.
    try:
        module = importlib.import_module("flash_attn_npu_4")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The Ascend FA4 backend requires flash-attention-npu with FA4 AICPU "
            "scheduler metadata support. Install its wheel on the NPU worker."
        ) from exc
    if not all(callable(getattr(module, name, None)) for name in ("get_scheduler_metadata", "flash_attn_varlen_func")):
        raise RuntimeError(
            "flash_attn_npu_4 must provide get_scheduler_metadata and flash_attn_varlen_func (Ascend910)."
        )
    return module


def fa4_selected(attn_selector_config: object) -> bool:
    """Use only selector-key fields so vLLM's memoized selection stays valid."""
    return (
        _FA4_ENABLED
        and getattr(attn_selector_config, "use_non_causal", False)
        and not getattr(attn_selector_config, "has_sliding_window", False)
        and not getattr(attn_selector_config, "has_sink", False)
    )


def _build_fa4_seq_tensors(num_tokens: int, seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = seq_lens.shape[0]
    if num_reqs <= 0 or num_tokens <= 0 or num_tokens % num_reqs:
        raise RuntimeError(
            f"Parallel-drafting FA4 requires a non-empty uniform query batch: {num_tokens=}, {num_reqs=}"
        )
    # Graph buckets include dummy requests with zero KV lengths. Give them one
    # KV token and a full uniform query; their outputs are discarded downstream.
    # FA4 wants int32 cumulative Q offsets INCLUDING the initial zero, unlike FIA.
    cu_seqlens_q = torch.arange(num_reqs + 1, dtype=torch.int32, device=seq_lens.device) * (num_tokens // num_reqs)
    return cu_seqlens_q, seq_lens.to(torch.int32).clamp_min(1)


def _fa4_cache_views(
    key: torch.Tensor, value: torch.Tensor, block_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode block strides as virtual FA4 page IDs without moving shared KV.

    FA4 ignores tensor page strides. A physical page holding [K, V, padding]
    can instead be addressed as several dense K-sized pages: block b becomes
    page b * stride_factor. K/V retain their own base pointers, and gap pages
    are never referenced. Writers keep the original block-strided views.
    """
    if key.ndim != 4 or key.shape != value.shape or key.dtype != value.dtype:
        raise RuntimeError("FA4 requires matching four-dimensional NHD K/V caches.")
    if key.is_contiguous() and value.is_contiguous():
        return key, value, block_table.contiguous()
    num_blocks, block_size, num_heads, head_size = key.shape
    page_elements = block_size * num_heads * head_size
    inner_strides = (num_heads * head_size, head_size, 1)
    if (
        num_blocks <= 0
        or page_elements <= 0
        or key.stride()[1:] != inner_strides
        or value.stride()[1:] != inner_strides
        or key.stride(0) != value.stride(0)
        or key.stride(0) < page_elements
        or key.stride(0) % page_elements
    ):
        raise RuntimeError(
            "FA4 requires packed NHD pages with a physical stride that is an integer multiple of one K/V page: "
            f"K shape={tuple(key.shape)} stride={key.stride()}, V stride={value.stride()}."
        )
    stride_factor = key.stride(0) // page_elements
    # Each view ends at the last logical page, staying inside its raw storage.
    shape = ((num_blocks - 1) * stride_factor + 1, block_size, num_heads, head_size)
    strides = (page_elements, *inner_strides)
    return key.as_strided(shape, strides), value.as_strided(shape, strides), (block_table * stride_factor).contiguous()


class AscendFA4MetadataBuilder(AscendAttentionMetadataBuilder):
    """Leave non-causal draft lengths on device, including during graph replay."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        speculative_config = vllm_config.speculative_config
        if not (speculative_config is not None and getattr(speculative_config, "parallel_drafting", False)):
            raise RuntimeError("The Ascend FA4 backend cannot serve a model without parallel drafting.")
        _load_fa4()
        logger.info("Ascend FA4 backend selected for %d draft attention layer(s): %s", len(layer_names), layer_names)

    def _build_fia_seq_inputs(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs: int,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[int] | None, list[int] | None, torch.Tensor, torch.Tensor | None]:
        # DFlash can mix causal and non-causal cache groups. Keep builder and
        # forward fallback in agreement; the ordinary path needs host lists.
        if common_attn_metadata.causal:
            return super()._build_fia_seq_inputs(
                common_attn_metadata, num_reqs, query_start_loc_cpu, seq_lens, block_table
            )
        # seq_lens_list=None also skips the ordinary FIA graph_task_update.
        return common_attn_metadata.query_start_loc[: num_reqs + 1], None, None, seq_lens, block_table


class AscendFA4Impl(AscendAttentionBackendImpl):
    """Run FA4 against the shared paged KV cache without host length reads."""

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        if attn_metadata.causal:
            return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
        if self.key_cache is None and kv_cache is not None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("FA4 draft attention requires a paged KV cache.")
        if self.use_bnsd_kv_cache:
            raise RuntimeError("FA4 draft attention requires NHD KV cache layout.")

        fa4 = _load_fa4()
        num_tokens = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.seq_lens.shape[0]
        block_table = attn_metadata.block_tables
        if block_table.shape[0] < num_reqs:
            raise RuntimeError("FA4 block table has fewer rows than requests.")
        # FA4's kernel uses a packed row stride derived from max_seqlen_k.
        key_cache, value_cache, block_table = _fa4_cache_views(self.key_cache, self.value_cache, block_table[:num_reqs])
        block_size = self.key_cache.shape[1]
        max_seqlen_k = block_table.shape[1] * block_size
        # Q may be a strided view of a fused QKV projection. FA4 uses packed TND.
        query = query[:num_tokens].reshape(num_tokens, self.num_heads, self.head_size).contiguous()

        # Share conversions and AICPU metadata within this forward only. Include
        # scale, dtype and page capacity: they affect tiling even for equal heads.
        cache_key = (
            attn_metadata.seq_lens.data_ptr(),
            attn_metadata.seq_lens.stride(),
            num_tokens,
            num_reqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            block_size,
            max_seqlen_k,
            query.dtype,
            query.device,
            self.scale,
        )
        context = get_forward_context()
        cache = getattr(context, _FA4_META_CACHE_ATTR, None)
        if cache is None:
            cache = {}
            setattr(context, _FA4_META_CACHE_ATTR, cache)
        if cache_key not in cache:
            cu_seqlens_q, cache_seqlens = _build_fa4_seq_tensors(num_tokens, attn_metadata.seq_lens)
            scheduler_metadata = fa4.get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=num_tokens // num_reqs,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=self.num_heads,
                num_heads_kv=self.num_kv_heads,
                headdim=self.head_size,
                cache_seqlens=cache_seqlens,
                qkv_dtype=query.dtype,
                cu_seqlens_q=cu_seqlens_q,
                page_size=block_size,
                causal=False,
                window_size=(-1, -1),
                softmax_scale=self.scale,
                num_splits=0,
            )
            cache[cache_key] = cu_seqlens_q, cache_seqlens, scheduler_metadata
        cu_seqlens_q, cache_seqlens, scheduler_metadata = cache[cache_key]
        attn_output = fa4.flash_attn_varlen_func(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            page_table=block_table,
            max_seqlen_q=num_tokens // num_reqs,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            window_size=(-1, -1),
            scheduler_metadata=scheduler_metadata,
            num_splits=0,
            return_lse=False,
        )
        output[:num_tokens] = attn_output.view(num_tokens, self.num_heads, self.head_size)
        return output


class AscendFA4Backend(AscendAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type["AscendFA4Impl"]:
        return AscendFA4Impl

    @staticmethod
    def get_builder_cls() -> type["AscendFA4MetadataBuilder"]:
        return AscendFA4MetadataBuilder
