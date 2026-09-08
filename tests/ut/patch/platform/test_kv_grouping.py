# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import vllm.v1.core.kv_cache_utils
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

import vllm_ascend.patch.platform.patch_kv_cache_utils as kv_cache_utils_patch
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_default_kv_group_size,
    _get_kv_cache_groups_uniform_page_size,
    get_kv_cache_groups,
)


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


@pytest.mark.parametrize(
    ("layer_counts", "expected"),
    [
        pytest.param([12, 13], 13, id="balanced-buckets"),
        pytest.param([16, 5, 48], 5, id="dspark-skewed"),
        pytest.param([17, 48], 17, id="mtp"),
    ],
)
def test_get_default_kv_group_size_matches_upstream_heuristic(
    layer_counts: list[int],
    expected: int,
) -> None:
    assert _get_default_kv_group_size(layer_counts) == expected


def test_min_size_reduces_dspark_groups_to_five() -> None:
    kv_cache_specs = _make_dspark_kv_cache_specs()

    groups = _get_kv_cache_groups_uniform_page_size(kv_cache_specs, min_size=16)

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


def test_min_size_is_a_lower_bound_not_an_override() -> None:
    # A min_size below the upstream default is a no-op: the default (5 for
    # [16, 5, 48]) still wins, so grouping is unchanged at 15 groups.
    kv_cache_specs = _make_dspark_kv_cache_specs()
    groups = _get_kv_cache_groups_uniform_page_size(kv_cache_specs, min_size=1)
    assert len(groups) == 15


def test_without_min_size_delegates_to_upstream(monkeypatch) -> None:
    kv_cache_specs = _make_dspark_kv_cache_specs()
    sentinel = object()
    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_orig_get_kv_cache_groups_uniform_page_size",
        lambda kv_cache_spec: sentinel,
    )

    assert _get_kv_cache_groups_uniform_page_size(kv_cache_specs) is sentinel
    assert _get_kv_cache_groups_uniform_page_size(kv_cache_specs, min_size=0) is sentinel


def test_min_size_takes_precedence_over_the_kimi_k3_heuristic(monkeypatch) -> None:
    """An explicit width beats a shape match, or the variable silently no-ops.

    The Kimi K3 grouping matches on layer-name prefixes and spec equality rather
    than model identity, so it also claims Qwen3.6 + DSpark -- the exact shape
    VLLM_ASCEND_KV_GROUP_MIN_SIZE was written for. Running it first made the
    variable dead on those models with nothing in the log to say so.
    """
    kv_cache_specs = _make_dspark_kv_cache_specs()
    kimi_sentinel = object()
    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_get_kimi_k3_dspark_mixed_kv_cache_groups",
        lambda kv_cache_spec: kimi_sentinel,
    )

    groups = _get_kv_cache_groups_uniform_page_size(kv_cache_specs, min_size=16)

    assert groups is not kimi_sentinel
    assert len(groups) == 5


def test_kimi_k3_heuristic_still_wins_when_no_width_is_configured(monkeypatch) -> None:
    """With the variable unset, main's grouping is untouched."""
    kv_cache_specs = _make_dspark_kv_cache_specs()
    kimi_sentinel = object()
    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_get_kimi_k3_dspark_mixed_kv_cache_groups",
        lambda kv_cache_spec: kimi_sentinel,
    )

    assert _get_kv_cache_groups_uniform_page_size(kv_cache_specs) is kimi_sentinel
    assert _get_kv_cache_groups_uniform_page_size(kv_cache_specs, min_size=0) is kimi_sentinel


def test_get_kv_cache_groups_scopes_min_size_from_env(monkeypatch) -> None:
    observed: list[int | None] = []

    def fake_get_kv_cache_groups(vllm_config, kv_cache_spec):
        observed.append(kv_cache_utils_patch._KV_GROUP_MIN_SIZE.get())
        return []

    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_orig_get_kv_cache_groups",
        fake_get_kv_cache_groups,
    )
    monkeypatch.setenv("VLLM_ASCEND_KV_GROUP_MIN_SIZE", "16")

    assert get_kv_cache_groups(SimpleNamespace(), {}) == []
    assert observed == [16]
    assert kv_cache_utils_patch._KV_GROUP_MIN_SIZE.get() is None


def test_get_kv_cache_groups_defaults_to_disabled(monkeypatch) -> None:
    observed: list[int | None] = []

    def fake_get_kv_cache_groups(vllm_config, kv_cache_spec):
        observed.append(kv_cache_utils_patch._KV_GROUP_MIN_SIZE.get())
        return []

    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_orig_get_kv_cache_groups",
        fake_get_kv_cache_groups,
    )
    monkeypatch.delenv("VLLM_ASCEND_KV_GROUP_MIN_SIZE", raising=False)

    assert get_kv_cache_groups(SimpleNamespace(), {}) == []
    assert observed == [None]
    assert kv_cache_utils_patch._KV_GROUP_MIN_SIZE.get() is None


def test_core_grouping_patch_is_installed() -> None:
    assert vllm.v1.core.kv_cache_utils.get_kv_cache_groups is get_kv_cache_groups
    assert (
        vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size
        is _get_kv_cache_groups_uniform_page_size
    )
