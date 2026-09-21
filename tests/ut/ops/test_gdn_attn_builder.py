# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.third_party.flash_linear_attention.ops import index as _fla_index
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops import gdn_attn_builder as ascend_gdn_attn_builder
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)
from vllm_ascend.ops.triton.fla import utils as fla_utils
from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices as runtime_prepare_chunk_indices,
)
from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_offsets as runtime_prepare_chunk_offsets,
)
from vllm_ascend.ops.triton.fla.utils import (
    prepare_final_chunk_indices as runtime_prepare_final_chunk_indices,
)
from vllm_ascend.ops.triton.fla.utils import (
    prepare_update_chunk_offsets as runtime_prepare_update_chunk_offsets,
)


@pytest.fixture(autouse=True)
def _patch_triton_cdiv(monkeypatch):
    if not hasattr(_fla_index.triton, "cdiv"):
        monkeypatch.setattr(
            _fla_index.triton,
            "cdiv",
            lambda a, b: (a + b - 1) // b,
            raising=False,
        )


@pytest.fixture(autouse=True)
def _no_pin_memory():
    # compute_causal_conv1d_metadata uses np_to_pinned_tensor which reads
    # PIN_MEMORY.  Without physical NPU, t.pin_memory() raises
    # "Please register PrivateUse1HooksInterface first".
    with (
        patch("vllm.utils.torch_utils.PIN_MEMORY", False),
        patch("vllm.v1.attention.backends.utils.PIN_MEMORY", False),
    ):
        yield


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]
    name: str = "unnamed"

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_lens_cpu = torch.tensor(batch_spec.query_lens, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(
        batch_spec.batch_size + 1,
        dtype=torch.int32,
    )
    query_start_loc_cpu[1:] = query_lens_cpu.cumsum(0)
    query_start_loc = query_start_loc_cpu.to(device=device)
    num_tokens = sum(batch_spec.query_lens)

    seq_lens_cpu = torch.tensor(batch_spec.seq_lens, dtype=torch.int32)
    seq_lens = seq_lens_cpu.to(device=device)
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i] for i in range(batch_spec.batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    # Mirror model_runner: is_prefilling = num_computed < num_prompt_tokens.
    # Chunked prefills still have prompt tokens beyond num_computed; decodes do not.
    num_prompt_tokens_cpu = torch.tensor(
        [
            context_lens[i] + batch_spec.query_lens[i] if batch_spec.query_lens[i] > 1 else context_lens[i]
            for i in range(batch_spec.batch_size)
        ],
        dtype=torch.int32,
    )
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
        is_prefilling=is_prefilling,
    )


def _make_vllm_config(
    *,
    max_model_len: int = 8192,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_heads: int = 32,
    num_speculative_tokens: int = 0,
    mamba_cache_mode: str = "none",
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    enable_adaptive_verification: bool = False,
    max_cudagraph_capture_size: int | None = None,
):
    speculative_config = None
    if num_speculative_tokens > 0:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
            enable_adaptive_verification=enable_adaptive_verification,
        )

    model_config = SimpleNamespace(max_model_len=max_model_len)
    model_config.get_num_attention_heads = lambda parallel_config: num_heads

    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            max_cudagraph_capture_size=max_cudagraph_capture_size,
        ),
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        model_config=model_config,
        additional_config=None,
    )


def _make_builder(
    *,
    device: torch.device,
    num_heads: int,
    num_speculative_tokens: int,
    mamba_cache_mode: str = "none",
    block_size: int = 16,
    num_speculative_blocks: int = 0,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    max_num_seqs: int = 16,
    enable_adaptive_verification: bool = False,
    max_cudagraph_capture_size: int | None = None,
):
    vllm_config = _make_vllm_config(
        num_heads=num_heads,
        max_num_seqs=max_num_seqs,
        num_speculative_tokens=num_speculative_tokens,
        mamba_cache_mode=mamba_cache_mode,
        cudagraph_mode=cudagraph_mode,
        enable_adaptive_verification=enable_adaptive_verification,
        max_cudagraph_capture_size=max_cudagraph_capture_size,
    )
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    return AscendGDNAttentionMetadataBuilder(spec, ["layer0"], vllm_config, device)


def _build_attn_metadata(
    batch_spec: BatchSpec,
    *,
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
):
    device = torch.device("cpu")
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=device,
    )
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=num_speculative_tokens,
    )
    num_accepted_tokens = None
    if num_decode_draft_tokens_cpu is not None:
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size,
            dtype=torch.int32,
        )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )
    return builder, common_attn_metadata, attn_metadata


def _assert_chunk_meta_matches_runtime(builder, chunk_meta, cu_seqlens: torch.Tensor) -> None:
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
    cumsum_chunks = max(
        1,
        ascend_gdn_attn_builder._GDN_CUMSUM_WORKING_SET // (gdn_num_heads * ascend_gdn_attn_builder._GDN_CHUNK_SIZE),
    )
    cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()

    assert torch.equal(
        chunk_meta.chunk_indices_chunk64,
        runtime_prepare_chunk_indices(cu_seqlens, ascend_gdn_attn_builder._GDN_CHUNK_SIZE),
    )
    assert torch.equal(
        chunk_meta.chunk_offsets_chunk64,
        runtime_prepare_chunk_offsets(cu_seqlens, ascend_gdn_attn_builder._GDN_CHUNK_SIZE),
    )
    assert torch.equal(
        chunk_meta.update_chunk_offsets_chunk64,
        runtime_prepare_update_chunk_offsets(
            cu_seqlens,
            ascend_gdn_attn_builder._GDN_CHUNK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.final_chunk_indices_chunk64,
        runtime_prepare_final_chunk_indices(
            cu_seqlens,
            ascend_gdn_attn_builder._GDN_CHUNK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.chunk_indices_large_block,
        runtime_prepare_chunk_indices(
            cu_seqlens,
            ascend_gdn_attn_builder._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.block_indices_cumsum,
        runtime_prepare_chunk_indices(
            cu_seqlens,
            cumsum_chunk_size,
        ),
    )


def _patch_missing_runtime_cdiv(monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(fla_utils.triton, "cdiv"):
        return
    monkeypatch.setattr(
        fla_utils.triton,
        "cdiv",
        lambda x, y: (x + y - 1) // y,
        raising=False,
    )


def test_kimi_chunk_metadata_uses_linear_attention_head_count() -> None:
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=128,
        num_speculative_tokens=0,
    )
    builder.vllm_config.model_config.hf_text_config = SimpleNamespace(
        linear_attn_config={"num_heads": 32},
    )
    cu_seqlens = torch.tensor([0, 130], dtype=torch.int32)

    chunk_meta = ascend_gdn_attn_builder._build_non_spec_chunked_prefill_metadata(
        builder,
        cu_seqlens,
        torch.device("cpu"),
    )

    _assert_chunk_meta_matches_runtime(builder, chunk_meta, cu_seqlens)


def test_ascend_gdn_attention_uses_ascend_backend():
    assert AscendGatedDeltaNetAttention.get_attn_backend(object()) is AscendGDNAttentionBackend
    assert AscendGDNAttentionBackend.get_builder_cls() is AscendGDNAttentionMetadataBuilder


def test_sequence_index_buffers_cover_spec_decode_when_cudagraph_disabled():
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
    )
    assert builder.spec_sequence_indices_cpu.numel() >= builder.vllm_config.scheduler_config.max_num_seqs

    spec_indices, non_spec_indices = builder._copy_sequence_indices_to_device(
        torch.tensor([True], dtype=torch.bool),
        num_spec_decodes=1,
    )

    assert torch.equal(spec_indices, torch.tensor([0]))
    assert non_spec_indices.numel() == 0


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_dspark_target_reorder_threshold_includes_base_token_regardless_of_anchor(
    sample_from_anchor: bool,
):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
    )
    builder.vllm_config.speculative_config.method = "dspark"
    builder.vllm_config.speculative_config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(sample_from_anchor=sample_from_anchor),
    )

    builder._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    assert builder.reorder_batch_threshold == 8


def _cache_index_first_column(cache_indices: torch.Tensor) -> torch.Tensor:
    if cache_indices.dim() == 1:
        return cache_indices
    return cache_indices[:, 0]


def _assert_non_spec_conv1d_args_match_metadata(attn_metadata) -> None:
    conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
    assert torch.equal(conv1d_meta.query_start_loc, attn_metadata.non_spec_query_start_loc)
    assert torch.equal(
        _cache_index_first_column(conv1d_meta.cache_indices),
        attn_metadata.non_spec_state_indices_tensor,
    )
    assert torch.equal(conv1d_meta.initial_state_mode, attn_metadata.has_initial_state)


@pytest.mark.parametrize(
    ("batch_spec", "num_speculative_tokens", "num_decode_draft_tokens_cpu"),
    [
        (
            BatchSpec(
                seq_lens=[8, 12],
                query_lens=[4, 8],
                name="pure_non_spec_prefill",
            ),
            0,
            None,
        ),
        (
            BatchSpec(
                seq_lens=[8, 4, 0, 12],
                query_lens=[4, 4, 0, 8],
                name="mixed_spec_non_spec_with_padding",
            ),
            3,
            torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
        ),
        (
            BatchSpec(
                seq_lens=[5, 12, 0, 9],
                query_lens=[1, 8, 0, 1],
                name="mixed_prefill_decode_without_spec",
            ),
            0,
            None,
        ),
    ],
    ids=lambda case: case.name if isinstance(case, BatchSpec) else None,
)
def test_non_spec_prefill_metadata_matches_original_inputs_and_runtime_helpers(
    batch_spec: BatchSpec,
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    builder, _, attn_metadata = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=num_speculative_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    prefill_metadata = getattr(attn_metadata, "non_spec_prefill_metadata", None)
    assert prefill_metadata is not None
    assert prefill_metadata.causal_conv1d is not None
    assert prefill_metadata.chunk is not None

    _assert_non_spec_conv1d_args_match_metadata(attn_metadata)

    _assert_chunk_meta_matches_runtime(
        builder,
        prefill_metadata.chunk,
        attn_metadata.prefill_query_start_loc,
    )


def test_non_spec_prefill_metadata_uses_prefill_tail_for_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[5, 12, 9],
        query_lens=[1, 8, 4],
        name="decode_prefill_without_spec",
    )
    builder, _, attn_metadata = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=0,
        num_decode_draft_tokens_cpu=None,
    )

    assert attn_metadata.num_decodes == 1
    assert attn_metadata.num_prefills == 2
    assert torch.equal(
        attn_metadata.non_spec_query_start_loc,
        torch.tensor([0, 1, 9, 13], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.prefill_query_start_loc,
        torch.tensor([0, 8, 12], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.prefill_state_indices,
        torch.tensor([1, 2], dtype=torch.int32),
    )

    prefill_metadata = getattr(attn_metadata, "non_spec_prefill_metadata", None)
    assert prefill_metadata is not None
    decode_metadata = getattr(attn_metadata, "non_spec_decode_metadata", None)
    assert decode_metadata is not None
    assert torch.equal(
        decode_metadata.actual_seq_lengths,
        torch.tensor([0, 1], dtype=torch.int32),
    )
    conv1d_meta = prefill_metadata.causal_conv1d
    assert torch.equal(conv1d_meta.query_start_loc, torch.tensor([0, 1, 9, 13], dtype=torch.int32))
    assert torch.equal(_cache_index_first_column(conv1d_meta.cache_indices), torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(conv1d_meta.initial_state_mode, torch.tensor([True, True, True]))
    _assert_chunk_meta_matches_runtime(
        builder,
        prefill_metadata.chunk,
        attn_metadata.prefill_query_start_loc,
    )


def test_spec_conv1d_args_use_device_cache_and_accepted_tokens():
    batch_spec = BatchSpec(
        seq_lens=[4, 4],
        query_lens=[4, 4],
        name="spec_only_device_args",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
    )
    num_accepted_tokens = torch.tensor([2, 4], dtype=torch.int32)

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )

    spec_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
    query_start_loc = spec_conv1d_meta.query_start_loc
    assert torch.equal(query_start_loc, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert torch.equal(
        spec_conv1d_meta.cache_indices,
        torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32),
    )
    assert torch.equal(spec_conv1d_meta.num_accepted_tokens, num_accepted_tokens)
    assert torch.equal(
        attn_metadata.spec_decode_metadata.actual_seq_lengths,
        torch.tensor([0, 4, 4], dtype=torch.int32),
    )


def test_full_graph_spec_conv1d_args_keep_request_granularity():
    batch_spec = BatchSpec(
        seq_lens=[4, 4, 4],
        query_lens=[4, 4, 4],
        name="full_graph_spec_only_device_args",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    num_accepted_tokens = torch.tensor([2, 4, 3], dtype=torch.int32)

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3, 3], dtype=torch.int32),
    )

    spec_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
    query_start_loc = spec_conv1d_meta.query_start_loc
    assert torch.equal(query_start_loc, torch.tensor([0, 4, 8, 12], dtype=torch.int32))
    assert query_start_loc.numel() == batch_spec.batch_size + 1
    assert spec_conv1d_meta.cache_indices.shape == (batch_spec.batch_size, 4)
    assert torch.equal(spec_conv1d_meta.cache_indices[:, 0], torch.tensor([10, 20, 30], dtype=torch.int32))
    assert torch.equal(spec_conv1d_meta.num_accepted_tokens, num_accepted_tokens)
    assert torch.equal(
        attn_metadata.spec_decode_metadata.actual_seq_lengths,
        torch.tensor([0, 4, 4, 4], dtype=torch.int32),
    )


def test_full_graph_spec_actual_seq_lengths_use_padded_builder_buffer():
    batch_spec = BatchSpec(
        seq_lens=[4, 4],
        query_lens=[4, 4],
        name="full_graph_padded_spec_actual_seq_lengths",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.num_reqs = 4
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=torch.tensor([2, 4], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )

    assert torch.equal(
        attn_metadata.spec_query_start_loc,
        torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32),
    )
    assert (
        attn_metadata.spec_decode_metadata.actual_seq_lengths.data_ptr() == builder.spec_actual_seq_lengths.data_ptr()
    )
    assert torch.equal(
        attn_metadata.spec_decode_metadata.actual_seq_lengths,
        torch.tensor([0, 4, 4, 0, 0], dtype=torch.int32),
    )


def test_causal_conv1d_cache_indices_use_device_block_table(monkeypatch: pytest.MonkeyPatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[4, 4],
        query_lens=[4, 4],
        name="device_block_table_source",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[40], [41]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
    )

    attn_metadata = builder.build(0, common_attn_metadata)

    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor([40, 41], dtype=torch.int32),
    )
    conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
    assert torch.equal(conv1d_meta.query_start_loc, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert torch.equal(_cache_index_first_column(conv1d_meta.cache_indices), torch.tensor([40, 41], dtype=torch.int32))
    assert torch.equal(conv1d_meta.initial_state_mode, torch.tensor([False, False]))


def test_mamba_align_cache_indices_follow_device_seq_lens(monkeypatch: pytest.MonkeyPatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[1, 9],
        query_lens=[1, 1],
        name="align_device_seq_lens",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=4,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.arange(20, dtype=torch.int32).view(2, 10)
    common_attn_metadata._seq_lens_cpu = torch.tensor([5, 13], dtype=torch.int32)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        mamba_cache_mode="align",
        block_size=4,
        num_speculative_blocks=2,
    )

    attn_metadata = builder.build(0, common_attn_metadata)

    conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
    assert torch.equal(
        _cache_index_first_column(conv1d_meta.cache_indices),
        torch.tensor([0, 12], dtype=torch.int32),
    )


def test_builder_builds_prebuilt_chunk_metadata_with_prefill_query_start_loc(monkeypatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[8, 4, 0, 12],
        query_lens=[4, 4, 0, 8],
        name="mixed_spec_non_spec_with_padding",
    )
    builder, common_attn_metadata, _ = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=3,
        num_decode_draft_tokens_cpu=torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
    )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=torch.ones(batch_spec.batch_size, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
    )

    chunk_meta = attn_metadata.non_spec_prefill_metadata.chunk
    assert chunk_meta.chunk_indices_chunk64 is attn_metadata.chunk_indices
    assert chunk_meta.chunk_offsets_chunk64 is attn_metadata.chunk_offsets
    _assert_chunk_meta_matches_runtime(
        builder,
        chunk_meta,
        attn_metadata.prefill_query_start_loc,
    )
    assert chunk_meta.cu_seqlens_host == tuple(attn_metadata.prefill_query_start_loc.to(torch.int64).tolist())
    expected_chunk_indices = runtime_prepare_chunk_indices(
        attn_metadata.prefill_query_start_loc,
        ascend_gdn_attn_builder._GDN_CHUNK_SIZE,
    )
    assert chunk_meta.chunk_indices_chunk64_host == tuple(expected_chunk_indices.to(torch.int64).reshape(-1).tolist())


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[1, 1, 1], query_lens=[1, 1, 1], name="decode_only"),
        BatchSpec(seq_lens=[4, 4], query_lens=[4, 4], name="spec_only"),
    ],
)
def test_builder_skips_prebuilt_meta_without_non_spec_prefill(batch_spec: BatchSpec):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3 if batch_spec.name == "spec_only" else 0,
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )

    num_accepted_tokens = None
    num_decode_draft_tokens_cpu = None
    if batch_spec.name == "spec_only":
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size,
            dtype=torch.int32,
        )
        num_decode_draft_tokens_cpu = torch.full(
            (batch_spec.batch_size,),
            3,
            dtype=torch.int32,
        )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    assert getattr(attn_metadata, "non_spec_prefill_metadata", None) is None
    if batch_spec.name == "decode_only":
        decode_metadata = getattr(attn_metadata, "non_spec_decode_metadata", None)
        assert decode_metadata is not None
        assert torch.equal(
            decode_metadata.actual_seq_lengths,
            torch.tensor([0, 1, 1, 1], dtype=torch.int32),
        )
    else:
        spec_decode_metadata = getattr(attn_metadata, "spec_decode_metadata", None)
        assert spec_decode_metadata is not None
        assert torch.equal(
            spec_decode_metadata.actual_seq_lengths,
            torch.tensor([0, 4, 4], dtype=torch.int32),
        )


def test_mixed_spec_prefill_chunk_metadata_preserves_single_token_count(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[1, 4, 8],
        query_lens=[1, 4, 8],
        name="mixed_spec_prefill_with_single_token_non_spec",
    )
    builder, _, attn_metadata = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=3,
        num_decode_draft_tokens_cpu=torch.tensor([-1, 3, -1], dtype=torch.int32),
    )

    assert attn_metadata.num_decodes == 0
    assert attn_metadata.num_prefills == 2
    assert torch.equal(
        attn_metadata.prefill_query_start_loc,
        torch.tensor([0, 1, 9], dtype=torch.int32),
    )
    chunk_metadata = attn_metadata.non_spec_prefill_metadata.chunk
    _assert_chunk_meta_matches_runtime(
        builder,
        chunk_metadata,
        attn_metadata.prefill_query_start_loc,
    )


@pytest.mark.parametrize(
    ("seq_len", "expected_decodes", "expected_prefills"),
    [
        pytest.param(1, 0, 1, id="first_token_stays_prefill"),
        pytest.param(17, 1, 0, id="block-size-plus-one-becomes-decode"),
    ],
)
def test_one_token_prefill_selection_respects_recurrent_state(
    monkeypatch: pytest.MonkeyPatch,
    seq_len: int,
    expected_decodes: int,
    expected_prefills: int,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[seq_len], query_lens=[1]),
        block_size=16,
        device=torch.device("cpu"),
    )
    # Model a prompt chunk explicitly. The helper normally classifies a
    # one-token row as decode when synthesizing test metadata.
    common_attn_metadata.is_prefilling = torch.tensor([True])
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
    )

    attn_metadata = builder.build(0, common_attn_metadata)

    assert common_attn_metadata.is_prefilling.tolist() == [True]
    assert attn_metadata.num_decodes == expected_decodes
    assert attn_metadata.num_prefills == expected_prefills


@pytest.mark.parametrize(
    "dcp_size,num_spec,context_len,mixed_spec,graph_mode",
    [
        (1, 3, 0, False, CUDAGraphMode.NONE),
        (1, 3, 384, False, CUDAGraphMode.NONE),
        (1, 5, 384, True, CUDAGraphMode.FULL_DECODE_ONLY),
        (16, 5, 0, False, CUDAGraphMode.NONE),
        (16, 5, 384, False, CUDAGraphMode.NONE),
        (16, 5, 384, False, CUDAGraphMode.FULL_DECODE_ONLY),
        (16, 5, 384, True, CUDAGraphMode.NONE),
        (16, 5, 384, True, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_spec_width_prompt_chunk_folds_only_without_dcp(
    monkeypatch: pytest.MonkeyPatch,
    dcp_size: int,
    num_spec: int,
    context_len: int,
    mixed_spec: bool,
    graph_mode: CUDAGraphMode,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    width = num_spec + 1
    query_lens = [width, width] if mixed_spec else [width]
    seq_lens = [768 + width, context_len + width] if mixed_spec else [context_len + width]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=384,
        device=torch.device("cpu"),
    )
    common.is_prefilling = torch.tensor([False, True] if mixed_spec else [True])
    common.block_table_tensor = torch.arange(len(query_lens) * 10, dtype=torch.int32).view(-1, 10)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=num_spec,
        mamba_cache_mode="align",
        block_size=384,
        num_speculative_blocks=num_spec,
        cudagraph_mode=graph_mode,
    )
    builder.vllm_config.parallel_config.decode_context_parallel_size = dcp_size
    accepted = torch.tensor([2, 1] if mixed_spec else [1], dtype=torch.int32)
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=accepted,
        num_decode_draft_tokens_cpu=torch.tensor([num_spec, -1] if mixed_spec else [-1], dtype=torch.int32),
    )

    assert accepted.tolist() == ([2, 1] if mixed_spec else [1])
    if dcp_size == 1 and context_len > 0:
        assert metadata.num_prefills == 0
        assert metadata.num_prefill_tokens == 0
        assert metadata.num_spec_decodes == 1 + int(mixed_spec)
        assert metadata.spec_sequence_masks.tolist() == ([True, True] if mixed_spec else [True])
        assert metadata.num_accepted_tokens.tolist() == ([2, width] if mixed_spec else [width])
        return

    # DCP retains prefill state semantics regardless of the prompt chunk width.
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == width
    assert metadata.num_spec_decodes == int(mixed_spec)
    assert metadata.prefill_has_initial_state.tolist() == [context_len > 0]
    expected_slot = (10 if mixed_spec else 0) + (seq_lens[-1] - 1) // 384
    assert metadata.prefill_state_indices.tolist() == [expected_slot]
    if mixed_spec:
        assert metadata.spec_sequence_masks.tolist() == [True, False]
        assert metadata.num_accepted_tokens.tolist() == [2]
    else:
        assert metadata.spec_sequence_masks is None
        assert metadata.num_accepted_tokens is None


def test_full_graph_without_runtime_spec_resets_captured_spec_inputs():
    capture_common_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[4, 4],
            query_lens=[4, 4],
            name="full_graph_spec_capture",
        ),
        block_size=16,
        device=torch.device("cpu"),
    )
    capture_common_metadata.num_reqs = 4
    capture_common_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    captured_metadata = builder.build(
        0,
        capture_common_metadata,
        num_accepted_tokens=torch.tensor([2, 4], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )
    captured_spec_metadata = captured_metadata.spec_decode_metadata
    captured_conv1d_metadata = captured_spec_metadata.spec_causal_conv1d

    assert torch.count_nonzero(captured_conv1d_metadata.query_start_loc) > 0
    assert torch.count_nonzero(captured_spec_metadata.actual_seq_lengths) > 0

    replay_common_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[1, 1, 0, 0],
            query_lens=[1, 1, 0, 0],
            name="full_graph_replay_without_spec",
        ),
        block_size=16,
        device=torch.device("cpu"),
    )
    replay_metadata = builder.build(
        0,
        replay_common_metadata,
        num_accepted_tokens=torch.ones(4, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.full((4,), -1, dtype=torch.int32),
    )

    assert replay_metadata.spec_sequence_masks is None
    assert replay_metadata.spec_decode_metadata is None
    assert torch.equal(
        captured_conv1d_metadata.cache_indices,
        torch.full((4, 4), PAD_SLOT_ID, dtype=torch.int32),
    )
    assert torch.count_nonzero(captured_conv1d_metadata.query_start_loc) == 0
    assert torch.count_nonzero(captured_conv1d_metadata.num_accepted_tokens) == 0
    assert torch.count_nonzero(captured_spec_metadata.actual_seq_lengths) == 0


def test_full_graph_idle_dummy_uses_zero_length_recurrent_metadata():
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[8, 8, 8, 8],
            query_lens=[0, 0, 0, 0],
            name="full_graph_idle_dummy",
        ),
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor[:, 0] = torch.tensor(
        [10, 11, 98, 99],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    builder.spec_state_indices_tensor.fill_(77)
    builder.spec_query_start_loc.fill_(77)
    builder.non_spec_state_indices_tensor.fill_(77)
    builder.non_spec_query_start_loc.fill_(77)

    attn_metadata = builder.build(0, common_attn_metadata)

    assert attn_metadata.num_actual_tokens == 0
    assert attn_metadata.num_decode_tokens == 0
    assert torch.count_nonzero(attn_metadata.non_spec_query_start_loc) == 0
    assert torch.all(attn_metadata.non_spec_state_indices_tensor == NULL_BLOCK_ID)
    assert torch.count_nonzero(builder.spec_query_start_loc[:5]) == 0
    assert torch.all(builder.spec_state_indices_tensor[:4] == PAD_SLOT_ID)


@pytest.mark.parametrize(
    ("num_speculative_tokens", "num_decode_draft_tokens_cpu"),
    [
        pytest.param(0, None, id="without_spec_decode"),
        pytest.param(
            3,
            torch.full((4,), -1, dtype=torch.int32),
            id="spec_decode_without_runtime_spec_requests",
        ),
    ],
)
def test_full_graph_non_spec_metadata_nulls_padded_state_indices(
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
):
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[1, 1, 0, 0],
            query_lens=[1, 1, 0, 0],
            name="full_graph_padded_non_spec_actual_seq_lengths",
        ),
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor[:, 0] = torch.tensor([10, 11, 98, 99])
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=num_speculative_tokens,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    builder.non_spec_state_indices_tensor.fill_(77)
    builder.non_spec_query_start_loc.fill_(77)
    builder.non_spec_actual_seq_lengths.fill_(77)

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    assert attn_metadata.num_decodes == 4
    assert attn_metadata.num_decode_tokens == 2
    assert torch.equal(
        attn_metadata.non_spec_query_start_loc,
        torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor(
            [10, 11, NULL_BLOCK_ID, NULL_BLOCK_ID],
            dtype=torch.int32,
        ),
    )
    decode_metadata = attn_metadata.non_spec_decode_metadata
    conv1d_metadata = decode_metadata.causal_conv1d
    assert conv1d_metadata.query_start_loc.data_ptr() == attn_metadata.non_spec_query_start_loc.data_ptr()
    assert conv1d_metadata.cache_indices.data_ptr() == attn_metadata.non_spec_state_indices_tensor.data_ptr()
    assert decode_metadata.actual_seq_lengths.data_ptr() == builder.non_spec_actual_seq_lengths.data_ptr()
    assert torch.equal(
        decode_metadata.actual_seq_lengths,
        torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32),
    )


@pytest.mark.parametrize("live_requests", [1, 2])
def test_spec_graph_fia_padding_refreshes_captured_buffers(live_requests):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    capture = create_common_attn_metadata(BatchSpec([8, 8], [8, 8]), 16, torch.device("cpu"))
    capture.block_table_tensor = torch.arange(16, dtype=torch.int32).view(2, 8) + 10
    captured = builder.build(0, capture, torch.ones(2, dtype=torch.int32), torch.tensor([7, 7]))
    stable = captured.spec_decode_metadata.spec_causal_conv1d
    pointers = [
        stable.query_start_loc.data_ptr(),
        stable.cache_indices.data_ptr(),
        stable.num_accepted_tokens.data_ptr(),
    ]

    lengths = [71, 34] if live_requests == 2 else [71, 0]
    replay = create_common_attn_metadata(BatchSpec(lengths, [8, 8]), 16, torch.device("cpu"))
    replay.block_table_tensor = torch.arange(16, dtype=torch.int32).view(2, 8) + 30
    drafts = torch.tensor([7, 7 if live_requests == 2 else -1])
    runtime = builder.build(0, replay, torch.tensor([1, 1], dtype=torch.int32), drafts)

    assert runtime.num_prefills == 0
    assert runtime.num_spec_decodes == live_requests
    actual = runtime.spec_decode_metadata.spec_causal_conv1d
    assert pointers == [
        actual.query_start_loc.data_ptr(),
        actual.cache_indices.data_ptr(),
        actual.num_accepted_tokens.data_ptr(),
    ]
    assert stable.query_start_loc.tolist() == [0, 8, 8 * live_requests]
    assert stable.cache_indices[0].tolist() == list(range(30, 38))
    if live_requests == 1:
        assert torch.all(stable.cache_indices[1] == NULL_BLOCK_ID)
    assert replay.query_start_loc_cpu.tolist() == [0, 8, 16]
    assert replay.num_actual_tokens == 16


def test_spec_graph_real_prefill_is_not_treated_as_padding():
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    common = create_common_attn_metadata(BatchSpec([71, 8], [8, 8]), 16, torch.device("cpu"))
    common.block_table_tensor = torch.arange(16, dtype=torch.int32).view(2, 8) + 10

    runtime = builder.build(0, common, torch.ones(2, dtype=torch.int32), torch.tensor([7, -1]))

    assert runtime.num_spec_decodes == 1
    assert runtime.num_prefills == 1


class TestSharedBatchPlan:
    """A hybrid model spreads its Mamba layers over several KV cache groups, so
    ``build`` runs once per group with metadata that differs only in the block
    table. The plan is the batch-shape half of that work, computed once and
    reused; anything that actually varies per group must stay outside it.
    """

    @staticmethod
    def _make_plan_builder():
        builder = AscendGDNAttentionMetadataBuilder.__new__(AscendGDNAttentionMetadataBuilder)
        builder.num_spec = 3
        builder.use_spec_decode = True
        return builder

    @staticmethod
    def _make_group_metadatas(num_groups, num_reqs=4):
        """Mirror how build_attn_metadata feeds the group loop.

        It builds one AscendCommonAttentionMetadata per KV cache group from the
        same batch tensors, varying only the block table and slot mapping. The
        plan key reads tensor addresses, so a test that rebuilt the batch
        tensors per group would miss the cache for a reason production never
        hits.
        """
        query_start_loc = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
        query_start_loc_cpu = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
        seq_lens = torch.tensor([9, 9, 9, 9], dtype=torch.int32)
        return [
            SimpleNamespace(
                num_reqs=num_reqs,
                num_actual_tokens=16,
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens[:num_reqs],
                block_table_tensor=torch.full((num_reqs, 8), gid, dtype=torch.int32),
            )
            for gid in range(num_groups)
        ]

    def test_key_ignores_the_only_per_group_field(self):
        """block_table is what build_attn_metadata varies per group, so every
        group in one batch must land on the same key."""
        builder = self._make_plan_builder()
        groups = self._make_group_metadatas(10)

        keys = {builder._shared_batch_plan_key(m, None, None) for m in groups}

        assert len(keys) == 1

    def test_key_separates_different_batches(self):
        builder = self._make_plan_builder()
        first = self._make_group_metadatas(1)[0]
        second = self._make_group_metadatas(1)[0]
        second.num_reqs = 3

        assert builder._shared_batch_plan_key(first, None, None) != builder._shared_batch_plan_key(second, None, None)

    def test_plan_is_computed_once_and_reused(self):
        builder = self._make_plan_builder()
        cache = {}
        sentinel = object()
        calls = []

        def fake_compute(m, num_accepted_tokens, num_decode_draft_tokens_cpu):
            calls.append(m)
            return sentinel

        builder._compute_shared_batch_plan = fake_compute

        # Qwen3.6 + DSpark puts 10 Mamba groups in one invocation.
        for m in self._make_group_metadatas(10):
            got = builder._get_shared_batch_plan(m, None, None, cache)
            assert got is sentinel

        assert len(calls) == 1

    def test_no_cache_means_no_reuse(self):
        """Callers that pass no cache (mrv1, direct unit tests) keep the
        original one-build-one-compute behaviour."""
        builder = self._make_plan_builder()
        calls = []
        builder._compute_shared_batch_plan = lambda *a: calls.append(1)

        for m in self._make_group_metadatas(2):
            builder._get_shared_batch_plan(m, None, None, None)

        assert len(calls) == 2

    def test_a_key_miss_only_costs_speed(self):
        """If a caller ever rebuilds the batch tensors per group the key stops
        matching. That must degrade to today's recompute-per-group, never to a
        reused plan built from someone else's batch."""
        builder = self._make_plan_builder()
        cache = {}
        calls = []
        builder._compute_shared_batch_plan = lambda *a: (calls.append(1), object())[1]

        for _ in range(3):
            fresh = self._make_group_metadatas(1)[0]
            builder._get_shared_batch_plan(fresh, None, None, cache)

        assert len(calls) == 3
        assert len(cache) == 3

    def test_state_indices_follow_this_group_block_table(self):
        """The derived indices are the whole reason each group still calls
        build: they must track the group's own block table."""
        builder = self._make_plan_builder()
        plan = SimpleNamespace(
            state_index_mode=ascend_gdn_attn_builder._STATE_INDEX_NON_SPEC,
            spec_sequence_indices=None,
            non_spec_sequence_indices=None,
        )
        block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)

        spec, non_spec, conv1d = builder._derive_group_state_indices(plan, block_table)

        assert spec is None
        assert torch.equal(non_spec, torch.tensor([10, 20], dtype=torch.int32))
        assert torch.equal(conv1d, block_table)

    def test_state_indices_spec_mixed_selects_both_sides(self):
        builder = self._make_plan_builder()
        plan = SimpleNamespace(
            state_index_mode=ascend_gdn_attn_builder._STATE_INDEX_SPEC_MIXED,
            spec_sequence_indices=torch.tensor([0], dtype=torch.int32),
            non_spec_sequence_indices=torch.tensor([1], dtype=torch.int32),
        )
        block_table = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32)

        spec, non_spec, conv1d = builder._derive_group_state_indices(plan, block_table)

        assert torch.equal(spec, torch.tensor([[10, 11, 12, 13]], dtype=torch.int32))
        assert torch.equal(non_spec, torch.tensor([20], dtype=torch.int32))
        assert conv1d is non_spec

    def test_state_indices_spec_only_has_no_non_spec_side(self):
        builder = self._make_plan_builder()
        plan = SimpleNamespace(
            state_index_mode=ascend_gdn_attn_builder._STATE_INDEX_SPEC_ONLY,
            spec_sequence_indices=torch.tensor([1], dtype=torch.int32),
            non_spec_sequence_indices=None,
        )
        block_table = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32)

        spec, non_spec, conv1d = builder._derive_group_state_indices(plan, block_table)

        assert torch.equal(spec, torch.tensor([[20, 21, 22, 23]], dtype=torch.int32))
        assert non_spec is None
        assert conv1d is None

    def test_group_independence_check_catches_a_varying_field(self):
        """The debug cross-check exists because a field that secretly varies
        per group would otherwise corrupt results silently."""
        make = lambda tokens: ascend_gdn_attn_builder._GDNSharedBatchPlan(  # noqa: E731
            state_index_mode=ascend_gdn_attn_builder._STATE_INDEX_NON_SPEC,
            num_prefills=0,
            num_decodes=4,
            num_decode_tokens=tokens,
            num_prefill_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            spec_sequence_masks=None,
            spec_sequence_indices=None,
            non_spec_sequence_indices=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            spec_query_start_loc=None,
            non_spec_query_start_loc=None,
            num_accepted_tokens=None,
            has_initial_state=None,
            prefill_has_initial_state=None,
            prefill_query_start_loc=None,
            chunk_indices=None,
            chunk_offsets=None,
            non_spec_chunked_prefill_metadata=None,
            nums_dict=None,
            batch_ptr=None,
            token_chunk_offset_ptr=None,
        )

        AscendGDNAttentionMetadataBuilder._assert_plan_is_group_independent(make(4), make(4))

        with pytest.raises(AssertionError, match="num_decode_tokens"):
            AscendGDNAttentionMetadataBuilder._assert_plan_is_group_independent(make(4), make(5))


def test_gdn_builder_defines_build_once_and_routes_through_the_local_view() -> None:
    """A second ``build`` definition silently shadowed the first one.

    The shadowed definition held the only calls to
    ``_remove_spec_graph_padding_queries`` and
    ``_treat_single_token_prefills_with_state_as_decodes``, so both corrections
    were unreachable while still reading as present. Nothing raises in that
    state, so guard the structure rather than only the behaviour.
    """
    tree = ast.parse(Path(ascend_gdn_attn_builder.__file__).read_text())
    (class_node,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendGDNAttentionMetadataBuilder"
    ]
    builds = [node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "build"]
    assert len(builds) == 1, "a shadowed build() definition is dead code"

    assert "_get_gdn_local_metadata" in {
        node.func.attr
        for node in ast.walk(builds[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    (view,) = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_gdn_local_metadata"
    ]
    assert {
        "_remove_spec_graph_padding_queries",
        "_treat_single_token_prefills_with_state_as_decodes",
    } <= {node.func.id for node in ast.walk(view) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


def test_gdn_local_view_zeroes_padding_rows_and_keeps_each_group_block_table() -> None:
    """Inactive graph rows must be zero-length; the block table must stay local.

    The FIA-padded query boundary gives padding requests a positive length
    whenever the token count lands inside a graph bucket. Those rows classify
    as non-speculative, so they fold into ``num_prefills`` and cost the batch
    its pure-spec persistent graph buffers.

    The correction is computed once per invocation, because it rebuilds
    ``query_start_loc`` and the plan cache identifies a batch by tensor address.
    What must NOT be shared is the metadata object around it: it carries the
    block table this group addresses, and ``build`` derives this group's conv
    and recurrent state indices from it, so one shared object sends every Mamba
    group to the first group's state slots.
    """
    batch_spec = BatchSpec(
        seq_lens=[64, 64, 0, 0],
        query_lens=[8, 8, 1, 1],
        name="spec_rows_plus_positive_length_padding",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    num_decode_draft_tokens_cpu = torch.tensor([7, 7, -1, -1], dtype=torch.int32)

    batch_shared_cache: dict = {}
    # build_attn_metadata varies exactly two fields per KV cache group.
    group0 = common_attn_metadata
    group1 = common_attn_metadata.replace(
        block_table_tensor=common_attn_metadata.block_table_tensor + 1000,
    )

    view0 = builder._get_gdn_local_metadata(group0, num_decode_draft_tokens_cpu, batch_shared_cache)
    view1 = builder._get_gdn_local_metadata(group1, num_decode_draft_tokens_cpu, batch_shared_cache)

    # The padding rows keep their slot but lose their tokens.
    assert torch.equal(view0.query_start_loc_cpu, torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32))
    assert view0.num_actual_tokens == 16
    assert view0.num_reqs == batch_spec.batch_size

    # One correction for the whole invocation: the second group presents the
    # same tensors, so the plan key still identifies a single batch.
    assert view1.query_start_loc_cpu is view0.query_start_loc_cpu
    assert view1.query_start_loc is view0.query_start_loc
    assert view1.num_actual_tokens == view0.num_actual_tokens

    # Each group still addresses its own block table.
    assert view0.block_table_tensor is group0.block_table_tensor
    assert view1.block_table_tensor is group1.block_table_tensor

    # Without a cache every caller recomputes, so nothing is shared.
    uncached = builder._get_gdn_local_metadata(group0, num_decode_draft_tokens_cpu, None)
    assert uncached.query_start_loc_cpu is not view0.query_start_loc_cpu


def _full_graph_spec_metadata(*, max_num_seqs: int, live_reqs: int, adaptive: bool, **builder_kwargs):
    """Build pure-speculative FULL-graph metadata for a partly filled batch."""
    batch_spec = BatchSpec(
        seq_lens=[64] * live_reqs,
        query_lens=[8] * live_reqs,
        name=f"spec_{live_reqs}of{max_num_seqs}",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        max_num_seqs=max_num_seqs,
        enable_adaptive_verification=adaptive,
        **builder_kwargs,
    )
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        num_accepted_tokens=torch.ones(live_reqs, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.full((live_reqs,), 7, dtype=torch.int32),
    )
    return builder, metadata


def test_ragged_spec_decode_pins_the_gdn_request_axis_to_max_num_seqs() -> None:
    """A ragged bucket must present the service-maximum request axis, not its own.

    With the axis following the bucket, each capture size carries its own
    stateful tiling and request axis. That survives steady concurrency and fails
    while concurrency ramps, because a shape change is only a speed question for
    a stateless operator -- for a stateful one it changes the state read/write
    contract. So every bucket is given B_max rows, and the rows past the live
    ones have to be inert: zero length and an invalid state index, or an empty
    row reads and writes state that now belongs to another request.
    """
    max_num_seqs, live = 16, 4
    builder, metadata = _full_graph_spec_metadata(max_num_seqs=max_num_seqs, live_reqs=live, adaptive=True)

    assert builder.ragged_spec_decode is True
    assert builder.gdn_request_axis == max_num_seqs
    assert metadata.spec_state_indices_tensor.shape[0] == max_num_seqs
    assert metadata.spec_sequence_masks.shape[0] == max_num_seqs
    assert metadata.num_accepted_tokens.shape[0] == max_num_seqs
    assert metadata.spec_query_start_loc.shape[0] == max_num_seqs + 1

    # The padded rows carry no tokens ...
    query_lens = torch.diff(metadata.spec_query_start_loc)
    assert torch.all(query_lens[live:] == 0)
    # ... and address no state.
    assert torch.all(metadata.spec_state_indices_tensor[live:] == NULL_BLOCK_ID)
    assert not metadata.spec_sequence_masks[live:].any()


def test_fixed_k_spec_decode_keeps_the_per_bucket_request_axis() -> None:
    # Without adaptive verification the batch is not ragged: Q and the request
    # count move together, so the per-bucket axis carries no ambiguity and is
    # left alone rather than paying B_max metadata clearing for every bucket.
    max_num_seqs, live = 16, 4
    builder, metadata = _full_graph_spec_metadata(max_num_seqs=max_num_seqs, live_reqs=live, adaptive=False)

    assert builder.ragged_spec_decode is False
    assert metadata.spec_state_indices_tensor.shape[0] == live


def test_ragged_spec_decode_refuses_full_graph_when_the_axis_cannot_fit() -> None:
    """A capture cap below max_num_seqs leaves the graph buffers too narrow.

    Falling back to a per-bucket axis there would reintroduce exactly the shape
    this pins down, so the full-graph metadata path is declined instead.
    """
    # One live request of eight tokens, so the pre-existing width gates
    # (num_spec_decodes and num_spec_decode_tokens against decode_cudagraph_max_bs)
    # both pass and the refusal can only come from the axis not fitting.
    builder, metadata = _full_graph_spec_metadata(
        max_num_seqs=16,
        live_reqs=1,
        adaptive=True,
        max_cudagraph_capture_size=8,
    )
    assert builder.decode_cudagraph_max_bs == 8
    assert builder.gdn_request_axis == 16
    assert builder.gdn_request_axis_fits_graph is False
    # Declined: the metadata keeps the live width instead of a fixed axis.
    assert metadata.spec_state_indices_tensor.shape[0] == 1
