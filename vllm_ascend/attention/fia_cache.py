# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense virtual pages for FIA over block-strided TND caches."""

from collections.abc import Callable
from typing import Any

import torch


def prepare_fia_cache(
    key: torch.Tensor, value: torch.Tensor, block_table: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Address packed pages in shared backing without materializing the cache.

    ACLNN may silently copy strided PA inputs. Expose the same storage as dense
    virtual pages and scale block IDs to skip the padding and other components.
    Cache writers and the cache manager retain their original views and IDs.
    Layouts that cannot be expressed this way keep the existing operator path.
    """
    if (
        block_table is None
        or key.ndim != 3
        or key.shape != value.shape
        or key.dtype != value.dtype
        or key.dtype not in (torch.bfloat16, torch.float16)
        or (key.is_contiguous() and value.is_contiguous())
    ):
        return key, value, block_table
    num_blocks, block_size, hidden_size = key.shape
    page_elements = block_size * hidden_size
    if (
        num_blocks <= 0
        or page_elements <= 0
        or key.stride()[1:] != (hidden_size, 1)
        or value.stride()[1:] != (hidden_size, 1)
        or key.stride(0) != value.stride(0)
        or key.stride(0) < page_elements
        or key.stride(0) % page_elements
    ):
        return key, value, block_table
    factor = key.stride(0) // page_elements
    shape = ((num_blocks - 1) * factor + 1, block_size, hidden_size)
    strides = (page_elements, hidden_size, 1)
    # End at the final logical page, not at the end of its physical slot.
    # This also preserves nonzero offsets of V and individual shared layers.
    mapped_table = torch.where(block_table >= 0, block_table * factor, block_table)
    return key.as_strided(shape, strides), value.as_strided(shape, strides), mapped_table


def call_fia_with_virtual_pages(operation: Callable[..., Any], **kwargs: Any) -> Any:
    """Keep page remapping inside the updatable graph task group.

    Doing this only at capture time would let the replay parameter provider
    replace mapped IDs with logical IDs. Each task update must remap the runtime
    table, including when the table's address changes between forwards.
    """
    key, value, block_table = prepare_fia_cache(kwargs["key"], kwargs["value"], kwargs.get("block_table"))
    return operation(**{**kwargs, "key": key, "value": value, "block_table": block_table})
