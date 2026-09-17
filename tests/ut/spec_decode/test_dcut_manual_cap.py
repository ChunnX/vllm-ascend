"""CPU unit tests for the D-Cut manual-cap policy.

These pin the confidence-free mapping from scheduled draft counts + a manual cap
pattern to per-request retained capacities and the batch draft budget. The
device-side ``reallocate_drafts`` reuses the upstream layout, so it is exercised
by the NPU model validation (step 4); here we lock down the pure policy the
injection rests on -- including that confidence never participates, and that a
pattern's budget survives the request reordering between the two manager entry
points.
"""

import numpy as np
import pytest

from vllm_ascend.worker.v2.spec_decode.dspark.dcut_manual_cap import (
    compute_manual_capacities,
    manual_batch_budget,
    manual_cap_enabled,
    parse_manual_cap_spec,
)


def test_single_cap_bounds_each_request_by_scheduled():
    scheduled = np.array([2, 4, 0, 3], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, (2,))
    # Requests over the cap are trimmed to 2; the request scheduling fewer drafts
    # than the cap keeps its own count; a no-draft request stays 0.
    np.testing.assert_array_equal(capacities, np.array([2, 2, 0, 2], dtype=np.int32))
    assert np.all(capacities <= scheduled)
    assert capacities.dtype == scheduled.dtype


def test_single_cap_leaves_the_batch_uniform():
    """The reason a single cap cannot validate the variable-length path.

    In steady state every request schedules the full draft count, so one global
    cap gives every request the same width -- the batch is uniform at cap+1
    tokens per request, which at the graph layer is indistinguishable from
    running with num_speculative_tokens = cap.
    """
    steady_state = np.full(8, 7, dtype=np.int32)
    capacities = compute_manual_capacities(steady_state, (2,))
    assert len(set(capacities.tolist())) == 1


def test_pattern_makes_a_steady_state_batch_ragged():
    """The reason a pattern is what step 3 needs: widths actually differ."""
    steady_state = np.full(8, 7, dtype=np.int32)
    capacities = compute_manual_capacities(steady_state, (7, 0, 3, 1))
    # The pattern repeats across batch positions.
    np.testing.assert_array_equal(
        capacities, np.array([7, 0, 3, 1, 7, 0, 3, 1], dtype=np.int32)
    )
    # Per-request query widths (capacity + 1 anchor) are genuinely ragged.
    assert len(set((capacities + 1).tolist())) > 1


def test_pattern_shorter_and_longer_than_the_batch():
    scheduled = np.full(3, 7, dtype=np.int32)
    # Pattern longer than the batch: the tail is simply unused.
    np.testing.assert_array_equal(
        compute_manual_capacities(scheduled, (1, 2, 3, 4, 5)),
        np.array([1, 2, 3], dtype=np.int32),
    )
    # Pattern shorter than the batch: it cycles.
    np.testing.assert_array_equal(
        compute_manual_capacities(scheduled, (0, 4)),
        np.array([0, 4, 0], dtype=np.int32),
    )


def test_pattern_still_bounded_by_scheduled_counts():
    # A cap above a request's scheduled count cannot invent drafts.
    scheduled = np.array([7, 2, 0, 5], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, (7, 7, 7, 7))
    np.testing.assert_array_equal(capacities, scheduled)
    assert np.all(capacities <= scheduled)


def test_cap_zero_keeps_only_anchor():
    scheduled = np.array([3, 1, 5], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, (0,))
    np.testing.assert_array_equal(capacities, np.zeros(3, dtype=np.int32))


def test_negative_entry_retains_all_as_copy():
    scheduled = np.array([3, 1, 5], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, (-1,))
    np.testing.assert_array_equal(capacities, scheduled)
    # A fresh array: mutating the result must not touch the scheduled counts.
    capacities[0] = 99
    assert scheduled[0] == 3


def test_negative_entry_inside_a_pattern_is_per_position():
    scheduled = np.array([7, 7, 7], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, (-1, 0, 2))
    np.testing.assert_array_equal(capacities, np.array([7, 0, 2], dtype=np.int32))


def test_large_cap_matches_no_trim():
    scheduled = np.array([3, 1, 5], dtype=np.int32)
    capacities = compute_manual_capacities(scheduled, (1000,))
    np.testing.assert_array_equal(capacities, scheduled)


def test_parse_accepts_a_single_int_and_a_pattern():
    assert parse_manual_cap_spec("-1") == (-1,)
    assert parse_manual_cap_spec("2") == (2,)
    assert parse_manual_cap_spec("7,0,3,1") == (7, 0, 3, 1)
    # Whitespace and a trailing comma are tolerated.
    assert parse_manual_cap_spec(" 7, 0 ,3, ") == (7, 0, 3)


@pytest.mark.parametrize("spec", ["", " ", ",", "2,x", "abc"])
def test_parse_rejects_a_malformed_spec(spec):
    # A typo must fail loudly at startup, not silently disable trimming.
    with pytest.raises(ValueError):
        parse_manual_cap_spec(spec)


def test_enabled_only_when_some_position_is_capped():
    assert not manual_cap_enabled((-1,))
    assert not manual_cap_enabled((-1, -2))
    assert manual_cap_enabled((0,))
    assert manual_cap_enabled((-1, 3))


def test_manual_batch_budget_sums_capped_capacities():
    num_drafts_per_req = {"a": 4, "b": 1, "c": 0, "d": 3}
    num_non_draft_tokens_per_req = {"a": 1, "b": 1, "c": 1, "d": 1}
    drafts, non_draft, capacity_per_req, draft_budget = manual_batch_budget(
        num_drafts_per_req, num_non_draft_tokens_per_req, (2,)
    )
    # min(4,2)+min(1,2)+min(0,2)+min(3,2) = 2+1+0+2 = 5
    assert draft_budget == 5
    assert capacity_per_req == {"a": 2, "b": 1, "c": 0, "d": 2}
    assert sum(capacity_per_req.values()) == draft_budget
    # The per-request dicts pass through untouched (order and values preserved).
    assert drafts == num_drafts_per_req
    assert non_draft == num_non_draft_tokens_per_req


def test_manual_batch_budget_keys_the_pattern_to_requests():
    """Why the budget carries a per-request dict rather than just a sum.

    ``get_num_tokens`` resolves capacities in scheduler order, but
    ``reallocate_drafts`` walks requests in the ``sort_batch_req_ids`` batch
    order. A position-indexed pattern is not order-independent, so the second
    site must look these values up: re-deriving them from the batch order would
    pair caps with different requests and could sum to a different budget than
    the one the batch was already sized from.
    """
    num_drafts_per_req = {"a": 7, "b": 7, "c": 7}
    num_non_draft_tokens_per_req = dict.fromkeys(num_drafts_per_req, 1)
    _, _, capacity_per_req, draft_budget = manual_batch_budget(
        num_drafts_per_req, num_non_draft_tokens_per_req, (7, 0, 1)
    )
    assert capacity_per_req == {"a": 7, "b": 0, "c": 1}
    assert draft_budget == 8

    # Whatever order the device side walks the requests in, the looked-up
    # capacities still sum to the budget the batch was sized from.
    for order in (["a", "b", "c"], ["c", "a", "b"], ["b", "c", "a"]):
        assert sum(capacity_per_req[req_id] for req_id in order) == draft_budget

    # Re-deriving from a permuted order instead would not have been safe: the
    # capacities differ request-by-request from the ones already published.
    permuted = ["c", "a", "b"]
    rederived = compute_manual_capacities(
        np.fromiter(
            (num_drafts_per_req[req_id] for req_id in permuted), dtype=np.int32, count=3
        ),
        (7, 0, 1),
    )
    assert dict(zip(permuted, rederived.tolist())) != capacity_per_req


def test_manual_batch_budget_is_deterministic():
    num_drafts_per_req = {"a": 4, "b": 3}
    num_non_draft_tokens_per_req = {"a": 1, "b": 1}
    first = manual_batch_budget(num_drafts_per_req, num_non_draft_tokens_per_req, (2,))
    second = manual_batch_budget(num_drafts_per_req, num_non_draft_tokens_per_req, (2,))
    # No confidence, no cost table: identical inputs give identical budget.
    assert first[3] == second[3] == 4
    assert first[2] == second[2]
