#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_5_vl.py
# Copyright 2023 The vLLM team.
#
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

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from einops import rearrange
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VisionBlock, Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionTransformer, Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY


class AscendQwen2_5_VisionAttention_Without_Padding(Qwen2_5_VisionAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            projection_size,
            quant_config,
            prefix,
        )
        self.embed_dim = embed_dim
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        q, k, v = [
            rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        context_layer = torch.empty_like(q)

        # operator requires pta version >= 2.5.1.dev20250226
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer)

        context_layer = rearrange(context_layer,
                                  "(b s) h d -> s b (h d)",
                                  b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


class AscendQwen2_5_VisionBlock_Without_Padding(Qwen2_5_VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim, num_heads, mlp_hidden_dim, act_fn, norm_layer,
                         quant_config, prefix)
        self.attn = AscendQwen2_5_VisionAttention_Without_Padding(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn")

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens=cu_seqlens, cos=cos, sin=sin)

        x = x + self.mlp(self.norm2(x))
        return x


class AscendQwen2_5_VisionPatchEmbed_Without_Padding(Qwen2_5_VisionPatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(
            self.proj.weight.data.view(self.hidden_size, -1).transpose(0, 1))
        return x


class AscendQwen2_5_VisionTransformer_Without_Padding(Qwen2_5_VisionTransformer
                                                      ):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        interleaved=False,
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix)
        norm_layer = partial(RMSNorm, eps=norm_eps)
        self.interleaved = interleaved
        self.patch_embed = AscendQwen2_5_VisionPatchEmbed_Without_Padding(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        self.blocks = nn.ModuleList([
            AscendQwen2_5_VisionBlock_Without_Padding(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            self.hidden_size, self.num_heads)

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()  # [seqlen, rotary_dim / 2]
        sin = rotary_pos_emb.sin()

        if not self.interleaved:
            cos_new = torch.cat((cos, cos), dim=-1)
            sin_new = torch.cat((sin, sin), dim=-1)
        else:
            cos_new = rearrange(torch.stack((cos, cos), dim=-1),
                                "... d two -> ...(d two)",
                                two=2)
            sin_new = rearrange(torch.stack((sin, sin), dim=-1),
                                "... d two -> ...(d two)",
                                two=2)
        cos_new = cos_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        sin_new = sin_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        return cos_new, sin_new

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:,
                                                      0]).cpu().to(torch.int32)

        # patchify
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_window_seqlens = torch.diff(cu_window_seqlens).cpu().to(torch.int32)
        seq_len, _ = x.size()
        x = x.reshape(seq_len // self.spatial_merge_unit,
                      self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cos, sin = self.cal_cos_sin(rotary_pos_emb)

        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(x, cu_seqlens=cu_seqlens_now, cos=cos, sin=sin)

        # adapter
        x = self.merger(x)
        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]
        return x


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class AscendQwen2_5_VLForConditionalGeneration_Without_Padding(
        Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.visual = AscendQwen2_5_VisionTransformer_Without_Padding(
            vision_config=config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )
