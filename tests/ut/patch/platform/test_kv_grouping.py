# SPDX-License-Identifier: Apache-2.0

from itertools import permutations
from types import SimpleNamespace

import pytest
import torch
import vllm.v1.core.kv_cache_utils
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

import vllm_ascend.patch.platform.patch_kv_cache_utils as kv_cache_utils_patch
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    DSPARK_KV_GROUP_MAX_PADDING_RATIO,
    _ascend_get_kv_cache_groups,
    _evaluate_kv_group_size,
    _get_default_kv_group_size,
    _get_kv_cache_groups_uniform_page_size,
    _select_kv_group_size,
)


@pytest.mark.parametrize(
    (
        "layer_counts",
        "max_padding_ratio",
        "expected_group_size",
        "expected_num_groups",
    ),
    [
        pytest.param(
            [16, 5, 48],
            DSPARK_KV_GROUP_MAX_PADDING_RATIO,
            16,
            5,
            id="dspark-performance",
        ),
        pytest.param([16, 5, 48], None, 8, 9, id="padding-safe"),
        pytest.param([17, 48], None, 17, 4, id="mtp"),
        pytest.param([12, 13], None, 13, 2, id="balanced-buckets"),
        pytest.param([10, 20], None, 10, 3, id="zero-padding-baseline"),
        pytest.param([1], DSPARK_KV_GROUP_MAX_PADDING_RATIO, 1, 1, id="single-bucket"),
        pytest.param([5, 5, 5], DSPARK_KV_GROUP_MAX_PADDING_RATIO, 5, 3, id="equal-buckets"),
    ],
)
def test_select_kv_group_size(
    layer_counts: list[int],
    max_padding_ratio: float | None,
    expected_group_size: int,
    expected_num_groups: int,
) -> None:
    selected = _select_kv_group_size(
        layer_counts,
        max_padding_ratio=max_padding_ratio,
    )

    assert selected == expected_group_size
    assert _evaluate_kv_group_size(layer_counts, selected)[0] == expected_num_groups


def test_select_kv_group_size_never_increases_baseline_padding() -> None:
    for layer_counts in permutations([16, 5, 48]):
        default_group_size = _get_default_kv_group_size(layer_counts)
        selected_group_size = _select_kv_group_size(layer_counts)
        default_groups, default_padding = _evaluate_kv_group_size(
            layer_counts,
            default_group_size,
        )
        selected_groups, selected_padding = _evaluate_kv_group_size(
            layer_counts,
            selected_group_size,
        )

        assert selected_groups <= default_groups
        assert selected_padding <= default_padding
        assert selected_group_size == 8


def test_dspark_group_size_stays_within_padding_budget() -> None:
    for layer_counts in permutations([16, 5, 48]):
        selected_group_size = _select_kv_group_size(
            layer_counts,
            max_padding_ratio=DSPARK_KV_GROUP_MAX_PADDING_RATIO,
        )
        selected_groups, selected_padding = _evaluate_kv_group_size(
            layer_counts,
            selected_group_size,
        )

        assert selected_group_size == 16
        assert selected_groups == 5
        assert selected_padding / sum(layer_counts) <= DSPARK_KV_GROUP_MAX_PADDING_RATIO


@pytest.mark.parametrize("layer_counts", [[], [0], [1, -1]])
def test_select_kv_group_size_rejects_invalid_counts(
    layer_counts: list[int],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        _select_kv_group_size(layer_counts)


def test_select_kv_group_size_rejects_negative_padding_ratio() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _select_kv_group_size([16, 5, 48], max_padding_ratio=-0.01)


def _make_dspark_kv_cache_specs() -> dict[str, FullAttentionSpec | MambaSpec]:
    base_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float16,
        indexes_kv_by_block_stride=True,
    )
    draft_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float16,
        indexes_kv_by_block_stride=True,
    )
    assert base_spec.page_size_bytes == draft_spec.page_size_bytes
    mamba_spec = MambaSpec(
        block_size=128,
        shapes=((1,),),
        dtypes=(torch.float16,),
        page_size_padded=base_spec.page_size_bytes,
    )

    return {
        **{f"base.{idx}": base_spec for idx in range(16)},
        **{f"draft.{idx}": draft_spec for idx in range(5)},
        **{f"mamba.{idx}": mamba_spec for idx in range(48)},
    }


def test_dspark_uniform_page_size_groups_are_reduced_to_five() -> None:
    kv_cache_specs = _make_dspark_kv_cache_specs()

    groups = _get_kv_cache_groups_uniform_page_size(
        kv_cache_specs,
        max_padding_ratio=DSPARK_KV_GROUP_MAX_PADDING_RATIO,
    )

    assert len(groups) == 5
    assert sorted(len(group.layer_names) for group in groups) == [5] + [16] * 4
    grouped_layer_names = [
        layer_name for group in groups for layer_name in group.layer_names
    ]
    assert len(grouped_layer_names) == len(set(grouped_layer_names)) == 69
    assert set(grouped_layer_names) == set(kv_cache_specs)
    assert {
        group.kv_cache_spec.page_size_bytes for group in groups
    } == {next(iter(kv_cache_specs.values())).page_size_bytes}


@pytest.mark.parametrize(
    ("use_dspark", "expected_padding_ratio"),
    [
        pytest.param(True, DSPARK_KV_GROUP_MAX_PADDING_RATIO, id="dspark"),
        pytest.param(False, None, id="non-dspark"),
    ],
)
def test_get_kv_cache_groups_scopes_dspark_padding_budget(
    monkeypatch,
    use_dspark: bool,
    expected_padding_ratio: float | None,
) -> None:
    observed_padding_ratios: list[float | None] = []

    def fake_get_kv_cache_groups(vllm_config, kv_cache_spec):
        observed_padding_ratios.append(
            kv_cache_utils_patch._KV_GROUP_MAX_PADDING_RATIO.get()
        )
        return []

    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_orig_get_kv_cache_groups",
        fake_get_kv_cache_groups,
    )
    speculative_config = SimpleNamespace(use_dspark=lambda: use_dspark)
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    assert _ascend_get_kv_cache_groups(vllm_config, {}) == []
    assert observed_padding_ratios == [expected_padding_ratio]
    assert kv_cache_utils_patch._KV_GROUP_MAX_PADDING_RATIO.get() is None


def test_core_uniform_page_size_grouping_uses_ascend_selector() -> None:
    assert (
        vllm.v1.core.kv_cache_utils.get_kv_cache_groups
        is _ascend_get_kv_cache_groups
    )
    assert (
        vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size
        is _get_kv_cache_groups_uniform_page_size
    )
