"""CPU unit tests for the D-Cut manual-cap policy.

These pin the confidence-free mapping from scheduled draft counts + a manual cap
to per-request retained capacities and the batch draft budget. The device-side
``reallocate_drafts`` reuses the upstream layout, so it is exercised by the NPU
model validation (step 4); here we lock down the pure policy the injection rests
on -- including that confidence never participates.
"""

import numpy as np

from vllm_ascend.worker.v2.spec_decode.dspark.dcut_manual_cap import (
    compute_manual_capacities,
    manual_batch_budget,
)


def test_manual_cap_bounds_each_request_by_scheduled():
    scheduled = np.array([2, 4, 0, 3], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, manual_cap=2)
    # Requests over the cap are trimmed to 2; the request scheduling fewer drafts
    # than the cap keeps its own count; a no-draft request stays 0.
    np.testing.assert_array_equal(capacities, np.array([2, 2, 0, 2], dtype=np.int32))
    assert np.all(capacities <= scheduled)
    assert capacities.dtype == scheduled.dtype


def test_manual_cap_zero_keeps_only_anchor():
    scheduled = np.array([3, 1, 5], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, manual_cap=0)
    np.testing.assert_array_equal(capacities, np.zeros(3, dtype=np.int32))


def test_manual_cap_negative_retains_all_as_copy():
    scheduled = np.array([3, 1, 5], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, manual_cap=-1)
    np.testing.assert_array_equal(capacities, scheduled)
    # A fresh array: mutating the result must not touch the scheduled counts.
    capacities[0] = 99
    assert scheduled[0] == 3


def test_manual_cap_large_matches_no_trim():
    scheduled = np.array([3, 1, 5], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, manual_cap=1000)
    np.testing.assert_array_equal(capacities, scheduled)


def test_manual_batch_budget_sums_capped_capacities():
    num_drafts_per_req = {"a": 4, "b": 1, "c": 0, "d": 3}
    num_non_draft_tokens_per_req = {"a": 1, "b": 1, "c": 1, "d": 1}
    drafts, non_draft, draft_budget = manual_batch_budget(
        num_drafts_per_req, num_non_draft_tokens_per_req, manual_cap=2
    )
    # min(4,2)+min(1,2)+min(0,2)+min(3,2) = 2+1+0+2 = 5
    assert draft_budget == 5
    # The per-request dicts pass through untouched (order and values preserved).
    assert drafts == num_drafts_per_req
    assert non_draft == num_non_draft_tokens_per_req


def test_manual_batch_budget_is_deterministic():
    num_drafts_per_req = {"a": 4, "b": 3}
    num_non_draft_tokens_per_req = {"a": 1, "b": 1}
    first = manual_batch_budget(num_drafts_per_req, num_non_draft_tokens_per_req, 2)
    second = manual_batch_budget(num_drafts_per_req, num_non_draft_tokens_per_req, 2)
    # No confidence, no cost table: identical inputs give identical budget.
    assert first[2] == second[2] == 4
