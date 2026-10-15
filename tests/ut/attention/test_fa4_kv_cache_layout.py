# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for FA4 virtual page IDs (pytest --noconftest).

Extract the production functions to exercise cache contracts without loading
vLLM's NPU runtime or requiring accelerator hardware.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def functions_from_source(relative, names, namespace, base=None):
    source = ast.parse((ROOT / relative).read_text())
    definitions = [node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name in names]
    if base is not None:
        namespace["Base"] = base
        definitions = [
            ast.ClassDef(
                name="Probe",
                bases=[ast.Name(id="Base", ctx=ast.Load())],
                keywords=[],
                body=definitions,
                decorator_list=[],
            )
        ]
    code = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(code, relative, "exec"), namespace)
    return namespace


@pytest.fixture
def cache_functions():
    namespace = functions_from_source(
        "vllm_ascend/attention/fa4_v1.py", {"_fa4_cache_views", "_build_fa4_seq_tensors"}, dict(torch=torch)
    )
    return functions_from_source(
        "vllm_ascend/worker/v2/attn_utils.py",
        {"_reshape_combined_attention_kv_cache"},
        dict(namespace, math=math, get_dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size()),
    )


@pytest.mark.parametrize("stride_factor", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [8, 128])
def test_virtual_pages_preserve_shared_group_isolation(cache_functions, stride_factor, dtype, block_size):
    shape = (4, block_size, 2, 16)
    page_elements = math.prod(shape[1:])
    page_bytes = stride_factor * page_elements * torch.empty((), dtype=dtype).element_size()
    raw = torch.zeros(shape[0] * page_bytes, dtype=torch.uint8)
    physical_pages = raw.view(dtype).view(shape[0], -1)
    key, value = cache_functions["_reshape_combined_attention_kv_cache"](raw, (2, *shape), dtype, page_bytes)
    # Target group owns block 2 while draft owns block 1. The old dense
    # reshaping wrote outside physical block 1, corrupting the target state.
    physical_pages[2].fill_(7)
    before = physical_pages.clone()
    key[1].fill_(1)
    value[1].fill_(2)
    torch.testing.assert_close(physical_pages[2], before[2])
    changed = (physical_pages != before).any(dim=1).nonzero().flatten().tolist()
    assert changed == [1]

    # Use poisoned gaps and nonsequential block IDs, including the last block.
    for block in (0, 3):
        key[block].fill_(block + 1)
        value[block].fill_(block + 11)
    if stride_factor > 2:
        for block in (0, 1, 3):
            physical_pages.view(shape[0], stride_factor, -1)[block, 2:].fill_(float("nan"))
    table = torch.tensor([[3, 99, 1, 99, 0, 99]], dtype=torch.int32)[:, ::2]
    table_before = table.clone()
    fa_key, fa_value, fa_table = cache_functions["_fa4_cache_views"](key, value, table)
    for original, alias in ((key, fa_key), (value, fa_value)):
        assert alias.is_contiguous()
        assert alias.data_ptr() == original.data_ptr()
        assert alias.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        torch.testing.assert_close(alias[fa_table.long()], original[table.long()])
    assert fa_table.is_contiguous()
    torch.testing.assert_close(table, table_before)
    torch.testing.assert_close(fa_table, table * stride_factor)
    # Subsequent cache writes/reused block IDs remain visible without copying.
    key[1].fill_(23)
    assert fa_key[fa_table[0, 1]].eq(23).all()
    torch.testing.assert_close(physical_pages[2], before[2], equal_nan=True)


@pytest.mark.parametrize("num_blocks", [1, 4])
def test_contiguous_cache_keeps_page_ids(cache_functions, num_blocks):
    key, value = torch.randn(2, num_blocks, 128, 2, 16)
    table = torch.zeros(2, 4, dtype=torch.int32)[:, ::2]
    fa_key, fa_value, fa_table = cache_functions["_fa4_cache_views"](key, value, table)
    assert fa_key is key and fa_value is value
    torch.testing.assert_close(fa_table, table)
    assert fa_table.is_contiguous()


@pytest.mark.parametrize("failure", ["inner_stride", "fractional_page", "mismatched_stride", "shape"])
def test_unsupported_layout_fails_before_kernel(cache_functions, failure):
    key = torch.empty(4, 128, 2, 16)
    value = torch.empty_like(key)
    if failure == "inner_stride":
        key, value = key.transpose(1, 2), value.transpose(1, 2)
    elif failure == "fractional_page":
        backing = torch.empty(4 * (4096 + 1))
        key = backing.as_strided(key.shape, (4097, 32, 16, 1))
        value = key
    elif failure == "mismatched_stride":
        key = torch.empty(8, 128, 2, 16)[::2]
    else:
        value = value[:2]
    with pytest.raises(RuntimeError, match="FA4 requires"):
        cache_functions["_fa4_cache_views"](key, value, torch.zeros(1, 2, dtype=torch.int32))


def test_forward_passes_virtual_pages_to_fa4(cache_functions):
    fa4 = MagicMock()
    context = SimpleNamespace()
    namespace = functions_from_source(
        "vllm_ascend/attention/fa4_v1.py",
        {"forward_fused_infer_attention"},
        dict(
            cache_functions,
            AscendMetadata=object,
            _load_fa4=lambda: fa4,
            get_forward_context=lambda: context,
            _FA4_META_CACHE_ATTR="_ascend_fa4_meta_cache",
        ),
        object,
    )
    impl = namespace["Probe"]()
    impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, 128
    impl.scale = 0.125
    impl.use_bnsd_kv_cache = False
    backing = torch.randn(4, 2, 128, 2, 128)
    impl.key_cache, impl.value_cache = backing[:, 0], backing[:, 1]
    original_key = impl.key_cache
    table = torch.tensor([[3, 1]], dtype=torch.int32)
    metadata = SimpleNamespace(causal=False, num_actual_tokens=7, seq_lens=torch.tensor([129]), block_tables=table)
    query = torch.randn(7, 4, 128)
    output = torch.empty_like(query)
    fa4.flash_attn_varlen_func.return_value = torch.ones_like(query)
    impl.forward_fused_infer_attention(query, None, None, metadata, output)
    args = fa4.flash_attn_varlen_func.call_args
    assert args.args[1].is_contiguous() and args.args[2].is_contiguous()
    assert args.args[1].data_ptr() == original_key.data_ptr()
    torch.testing.assert_close(args.kwargs["page_table"], table * 2)
    assert args.kwargs["max_seqlen_k"] == 256
    assert fa4.get_scheduler_metadata.call_args.kwargs["page_size"] == 128
    assert impl.key_cache is original_key  # Writer keeps the original layout.
    assert torch.all(output == 1)


@pytest.mark.parametrize("seq_len", [31, 32, 33, 127, 128, 129])
@pytest.mark.parametrize("stride_factor", [2, 4])
def test_paged_attention_across_reused_blocks(cache_functions, seq_len, stride_factor):
    generator = torch.Generator().manual_seed(34)
    block_size, num_heads, head_size = 128, 2, 16
    page_elements = block_size * num_heads * head_size
    raw = torch.empty(4 * stride_factor * page_elements * 2, dtype=torch.uint8)
    raw.view(torch.bfloat16).fill_(float("nan"))
    key, value = cache_functions["_reshape_combined_attention_kv_cache"](
        raw, (2, 4, block_size, num_heads, head_size), torch.bfloat16, stride_factor * page_elements * 2
    )
    physical_pages = raw.view(4, -1)
    target_before = physical_pages[2].clone()
    query = torch.randn(7, num_heads, head_size, generator=generator)

    def attend(k, v):
        scores = torch.einsum("qhd,khd->hqk", query, k.float()) / math.sqrt(head_size)
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())

    # Reassign IDs between requests, reusing the allocation and poisoned gaps.
    for draft_block in (1, 3, 1, 3):
        for block in (0, draft_block):
            key[block].copy_(torch.randn(key[block].shape, generator=generator))
            value[block].copy_(torch.randn(value[block].shape, generator=generator))
        table = torch.tensor([[draft_block, 0]], dtype=torch.int32)
        fa_key, fa_value, fa_table = cache_functions["_fa4_cache_views"](key, value, table)
        expected_k = key[table.long()].reshape(-1, num_heads, head_size)[:seq_len]
        expected_v = value[table.long()].reshape(-1, num_heads, head_size)[:seq_len]
        actual_k = fa_key[fa_table.long()].reshape(-1, num_heads, head_size)[:seq_len]
        actual_v = fa_value[fa_table.long()].reshape(-1, num_heads, head_size)[:seq_len]
        actual = attend(actual_k, actual_v)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, attend(expected_k, expected_v))
        torch.testing.assert_close(physical_pages[2], target_before)
