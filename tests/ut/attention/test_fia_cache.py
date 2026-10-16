# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.fia_cache import call_fia_with_virtual_pages, prepare_fia_cache


@pytest.mark.parametrize("factor", [2, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_virtual_pages_preserve_offsets_and_live_cache_writes(factor, dtype):
    blocks, block_size, hidden = 5, 4, 8
    page = block_size * hidden
    raw = torch.arange(blocks * factor * page + 7, dtype=torch.float32).to(dtype)
    strides = (factor * page, hidden, 1)
    key = raw.as_strided((blocks, block_size, hidden), strides, storage_offset=7)
    value = raw.as_strided(key.shape, strides, storage_offset=7 + page)
    table = torch.tensor([[3, 0, 4], [2, 1, 0]], dtype=torch.int32)
    original_table = table.clone()
    k, v, mapped = prepare_fia_cache(key, value, table)
    assert k.is_contiguous() and v.is_contiguous()
    assert k.data_ptr() == key.data_ptr() and v.data_ptr() == value.data_ptr()
    assert k.shape[0] == (blocks - 1) * factor + 1
    for alias, cache in ((k, key), (v, value)):
        torch.testing.assert_close(alias[mapped.long()], cache[table.long()])
        cache[3].fill_(42)
        torch.testing.assert_close(alias[3 * factor], cache[3])
    torch.testing.assert_close(table, original_table)


@pytest.mark.parametrize(
    "layout", ["dense", "non_pa", "inner_stride", "fractional_page", "different_stride", "nz", "int8", "float32"]
)
def test_unsupported_or_dense_inputs_keep_original_path(layout):
    k = torch.empty_strided((3, 4, 8), (64, 8, 1), dtype=torch.bfloat16)
    v = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype)
    table = torch.zeros((1, 2), dtype=torch.int32)
    if layout == "dense":
        k, v = k.contiguous(), v.contiguous()
    elif layout == "non_pa":
        table = None
    elif layout == "inner_stride":
        k = torch.empty_strided(k.shape, (128, 16, 2), dtype=k.dtype)
    elif layout == "fractional_page":
        k = torch.empty_strided(k.shape, (65, 8, 1), dtype=k.dtype)
    elif layout == "different_stride":
        v = torch.empty_strided(k.shape, (96, 8, 1), dtype=k.dtype)
    elif layout == "nz":
        k, v = k.unsqueeze(1), v.unsqueeze(1)
    elif layout == "int8":
        k, v = k.to(torch.int8), v.to(torch.int8)
    else:
        k, v = k.to(torch.float32), v.to(torch.float32)
    actual = prepare_fia_cache(k, v, table)
    assert all(a is b for a, b in zip(actual, (k, v, table)))


def test_task_updates_remap_the_runtime_table_without_changing_cache_views():
    key = torch.empty_strided((5, 4, 8), (1024, 8, 1), dtype=torch.bfloat16)
    value = torch.empty_strided(key.shape, key.stride(), dtype=key.dtype)
    op = Mock(return_value="output")
    for table in (torch.tensor([[0, 3]], dtype=torch.int32), torch.tensor([[4, 1]], dtype=torch.int32)):
        assert call_fia_with_virtual_pages(op, key=key, value=value, block_table=table, scale=0.5) == "output"
        kwargs = op.call_args.kwargs
        torch.testing.assert_close(kwargs["block_table"], table * 32)
        assert kwargs["key"].data_ptr() == key.data_ptr()
        assert kwargs["key"].is_contiguous()
        assert key.stride(0) == 1024
        assert kwargs["scale"] == 0.5


def test_virtual_pages_preserve_invalid_block_sentinels():
    key = torch.empty_strided((5, 4, 8), (1024, 8, 1), dtype=torch.bfloat16)
    table = torch.tensor([[3, -1, -99, 0]], dtype=torch.int32)
    _, _, mapped = prepare_fia_cache(key, key, table)
    torch.testing.assert_close(mapped, torch.tensor([[96, -1, -99, 0]], dtype=torch.int32))
