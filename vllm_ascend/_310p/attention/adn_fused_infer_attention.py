#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
import torch_npu

from vllm_ascend._310p.attention.metadata_builder import get_query_lens_cpu
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

# The single configuration this path has been validated at. These are not ADN's
# generic limits: they pin Qwen3-8B at TP=2, which is the only shape whose
# numerics anyone has checked. Anything else fails at startup rather than
# running unvalidated.
ADN_BLOCK_SIZE = 128
ADN_HEAD_DIM = 128
ADN_LOCAL_NUM_HEADS = 16  # Qwen3-8B: 32 query heads / TP=2
ADN_LOCAL_NUM_KV_HEADS = 4  # Qwen3-8B: 8 KV heads / TP=2
ADN_SUPPORTED_METHODS = {"dflash": 8, "dspark": 7}  # method -> num_speculative_tokens
ADN_SUPPORTED_ARCHITECTURES = {"DFlashQwen3ForCausalLM", "Qwen3DSparkForCausalLM"}


def expected_queries_per_request(method):
    """Queries each request issues per draft step.

    DFlash prepends an anchor to the K mask tokens, DSpark does not.
    """
    num_spec = ADN_SUPPORTED_METHODS[method]
    return num_spec + 1 if method == "dflash" else num_spec

_adn_module = None


def load_adn():
    """Import adn_custom_ops on first real use and cache it.

    Its package __init__ imports torchair at module scope, so importing eagerly
    would make torchair a hard dependency of every 310P run.
    """
    global _adn_module
    if _adn_module is not None:
        return _adn_module
    try:
        import adn_custom_ops
    except Exception as exc:  # ImportError, or OSError from the .so, or ABI mismatch
        raise RuntimeError(
            "DFlash/DSpark non-causal draft attention on 310P requires the ADN custom "
            "op package (adn_custom_ops + adn_custom_ops_lib + torchair). Install the "
            "Ascend_Ops custom_opp and PTA wheels for this device. There is no "
            "fallback: sending this path to the causal split-fuse kernel would return "
            "plausible but wrong numbers, so startup fails instead."
        ) from exc
    _adn_module = adn_custom_ops
    return _adn_module


def validate_adn_scope(*, vllm_config, query, key_cache, value_cache, num_heads, num_kv_heads, head_size):
    """Check every startup invariant once, before the first ADN call."""
    spec_config = vllm_config.speculative_config
    method = getattr(spec_config, "method", None)
    if spec_config is None or method not in ADN_SUPPORTED_METHODS:
        raise RuntimeError(
            f"ADN draft attention only covers {sorted(ADN_SUPPORTED_METHODS)}, got {method}"
        )
    expected_k = ADN_SUPPORTED_METHODS[method]
    if spec_config.num_speculative_tokens != expected_k:
        raise RuntimeError(
            f"{method} is only validated at num_speculative_tokens={expected_k}, got "
            f"{spec_config.num_speculative_tokens}"
        )

    architectures = getattr(spec_config.draft_model_config.hf_config, "architectures", None) or []
    arch = architectures[0] if architectures else None
    if arch not in ADN_SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            f"draft architecture {arch} is outside this scope "
            f"({sorted(ADN_SUPPORTED_ARCHITECTURES)})"
        )

    if not vllm_config.model_config.enforce_eager:
        raise RuntimeError("this scope is eager-only; ACLGraph is validated separately")
    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp != 2:
        raise RuntimeError(f"this scope is validated at TP=2 only, got TP={tp}")

    if (num_heads, num_kv_heads, head_size) != (ADN_LOCAL_NUM_HEADS, ADN_LOCAL_NUM_KV_HEADS, ADN_HEAD_DIM):
        raise RuntimeError(
            f"this scope only covers local Nq={ADN_LOCAL_NUM_HEADS}, Nkv={ADN_LOCAL_NUM_KV_HEADS}, "
            f"D={ADN_HEAD_DIM} (Qwen3-8B at TP=2), got Nq={num_heads}, Nkv={num_kv_heads}, "
            f"D={head_size}"
        )

    for name, tensor in (("query", query), ("key_cache", key_cache), ("value_cache", value_cache)):
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"ADN on 310P only supports float16 in this scope, but {name} is "
                f"{tensor.dtype}. Start the engine with dtype=float16."
            )

    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(f"ADN needs rank-4 NZ K/V caches, got {key_cache.ndim}/{value_cache.ndim}")
    if key_cache.shape != value_cache.shape:
        raise RuntimeError(f"K/V cache shapes differ: {key_cache.shape} vs {value_cache.shape}")
    if key_cache.device != value_cache.device:
        raise RuntimeError(f"K/V caches are on different devices: {key_cache.device} vs {value_cache.device}")

    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        fmt = int(torch_npu.get_npu_format(cache))
        if fmt != ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError(
                f"{name} is in acl format {fmt}, expected ACL_FORMAT_FRACTAL_NZ "
                f"({ACL_FORMAT_FRACTAL_NZ}); ADN reads the NZ layout directly"
            )

    if key_cache.shape[-1] != 16:
        raise RuntimeError(f"NZ cache last dim must be 16, got {key_cache.shape[-1]}")
    # Compared against the scope constant, not against a value derived from the
    # cache itself -- the latter would be tautological.
    if key_cache.shape[-2] != ADN_BLOCK_SIZE:
        raise RuntimeError(
            f"cache physical block size is {key_cache.shape[-2]}, this scope only covers "
            f"{ADN_BLOCK_SIZE}"
        )
    expected_dim1 = num_kv_heads * head_size // 16
    if key_cache.shape[1] != expected_dim1:
        raise RuntimeError(
            f"NZ cache dim1 is {key_cache.shape[1]}, expected num_kv_heads*head_size/16 = {expected_dim1}"
        )


def forward_parallel_draft_adn(self, query, attn_metadata, output):
    """Non-causal parallel-draft attention via ADN.

    ``attn_mask=None`` is what makes ADN non-causal: its host tiling maps an empty
    mask to NO_MASK and the kernel neither loads nor applies one, so every query row
    sees the full ``[0, actual_seq_lengths_kv[b])`` range -- context plus this
    round's entire query block. Never pass the 310P compressed split-fuse mask here,
    and never synthesize an all-zero causal mask to imitate it.
    """
    adn = load_adn()

    num_tokens = int(attn_metadata.num_actual_tokens)
    query_slice = query[:num_tokens]
    output_slice = output[:num_tokens]

    # Raw per-request q-lens come from the 310P builder, which diffed the CPU
    # endpoints outside the forward. The tensor is host/pinned, so .tolist() costs
    # no device sync. Never rebuild these from the base metadata's
    # actual_seq_lengths_q (cumulative endpoints) or from max_query_len.
    raw_q_lens = get_query_lens_cpu(attn_metadata)
    if raw_q_lens is None:
        raise RuntimeError(
            "310P parallel draft attention needs raw per-request query lengths on "
            "attn_metadata, but query_lens_cpu is missing. It is set by "
            "AscendAttentionMetadataBuilder310.build() for ChunkedPrefill/SpecDecoding; "
            "check that the draft metadata went through that builder."
        )
    q_lens = raw_q_lens.tolist()
    kv_lens = attn_metadata.seq_lens_list

    method = getattr(self.vllm_config.speculative_config, "method", None)
    if method not in ADN_SUPPORTED_METHODS:
        raise RuntimeError(f"ADN draft attention reached with unsupported method {method}")
    expected_q = expected_queries_per_request(method)
    if any(q != expected_q for q in q_lens):
        raise RuntimeError(
            f"{method} expects every request to query {expected_q} positions, got {q_lens}. "
            f"A cumulative-endpoint tensor was most likely passed instead of raw "
            f"per-request lengths."
        )
    if sum(q_lens) != num_tokens or query_slice.shape[0] != num_tokens:
        raise RuntimeError(
            f"sum(q_lens)={sum(q_lens)}, num_actual_tokens={num_tokens}, query rows="
            f"{query_slice.shape[0]} must all agree"
        )

    block_table = attn_metadata.block_tables[: len(q_lens)]
    if len(kv_lens) != len(q_lens) or block_table.shape[0] != len(q_lens):
        raise RuntimeError(
            f"batch size disagreement: {len(q_lens)} q-lens, {len(kv_lens)} kv-lens, "
            f"{block_table.shape[0]} block-table rows"
        )
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise RuntimeError(
            f"block table must be a rank-2 int32 tensor, got ndim={block_table.ndim} "
            f"dtype={block_table.dtype}"
        )

    # Fixed by scope rather than read back from the cache; validate_adn_scope
    # checks the cache's physical block size against this same constant.
    capacity = block_table.shape[1] * ADN_BLOCK_SIZE
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        if not 0 < q_len <= kv_len:
            raise RuntimeError(f"request {b}: need 0 < q_len({q_len}) <= kv_len({kv_len})")
        if kv_len > capacity:
            raise RuntimeError(
                f"request {b}: kv_len {kv_len} exceeds what its block table can address "
                f"({block_table.shape[1]} pages x {ADN_BLOCK_SIZE})"
            )

    key_cache = self.key_cache
    value_cache = self.value_cache
    query_tnd = query_slice.reshape(num_tokens, self.num_heads, self.head_size)

    if not self._adn_scope_validated:
        validate_adn_scope(
            vllm_config=self.vllm_config,
            query=query_tnd,
            key_cache=key_cache,
            value_cache=value_cache,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
        )
        self._adn_scope_validated = True

    adn_out = adn.adn_fused_infer_attention(
        query=query_tnd,
        key=key_cache,
        value=value_cache,
        attn_mask=None,
        actual_seq_lengths_q=q_lens,
        actual_seq_lengths_kv=kv_lens,
        block_table=block_table,
        num_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        block_size=ADN_BLOCK_SIZE,
        input_layout="TND",
        scale_value=self.scale,
        inner_precise=2,
        force_call=False,
    )

    # ADN's contract is "output has the same shape as query". Check that, not just
    # numel against the possibly flat output slice: a numel match would accept a
    # transposed or mis-headed result.
    if tuple(adn_out.shape) != tuple(query_tnd.shape) or adn_out.dtype != query_tnd.dtype:
        raise RuntimeError(
            f"ADN returned {tuple(adn_out.shape)}/{adn_out.dtype}, expected the query "
            f"shape {tuple(query_tnd.shape)}/{query_tnd.dtype}"
        )
    if adn_out.numel() != output_slice.numel():
        raise RuntimeError(
            f"ADN output has {adn_out.numel()} elements but the output slice holds "
            f"{output_slice.numel()}"
        )

    # ADN allocates its own output and has no `out=`, so the MVP eats one copy.
    # Do not change the operator ABI before correctness closes.
    output_slice.copy_(adn_out.reshape(output_slice.shape))
    return output
