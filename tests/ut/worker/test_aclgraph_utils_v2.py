# SPDX-License-Identifier: Apache-2.0

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.aclgraph_utils import (
    merge_decode_descriptors,
    trimmed_decode_descriptors,
)

_FULL = CUDAGraphMode.FULL


def _trimmed(width: int, max_num_reqs: int = 8, max_capture_size: int = 512):
    return trimmed_decode_descriptors(
        decode_mode=_FULL,
        width=width,
        max_num_reqs=max_num_reqs,
        max_decode_tokens=max_num_reqs * 8,
        max_capture_size=max_capture_size,
        lora_capture_cases=[0],
    )


def test_one_descriptor_per_request_count_so_no_replay_needs_padding() -> None:
    """Sizes follow the request count, not the captured token sizes.

    Rounding a captured size up to a multiple of the width lands between
    request counts -- width 3 gives 3, 6, 9, 18, 24, so four requests would
    round 12 up to 18 -- and the token and request padding that follows brings
    back padding rows this stage does not need to handle.
    """
    descs = _trimmed(width=3)

    assert [d.num_tokens for d in descs] == [3, 6, 9, 12, 15, 18, 21, 24]
    assert [d.num_reqs for d in descs] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert {d.uniform_token_count for d in descs} == {3}
    assert {d.cg_mode for d in descs} == {_FULL}
    # Every descriptor is exactly its batch, so nothing is padded on replay.
    assert all(d.num_tokens == d.num_reqs * 3 for d in descs)


def test_the_decode_token_budget_and_the_capture_ceiling_both_stop_the_grid() -> None:
    # 8 requests x width 8 is the whole decode budget, so width 8 stops there.
    assert [d.num_reqs for d in _trimmed(width=8)] == [1, 2, 3, 4, 5, 6, 7, 8]
    # A capture ceiling below the budget truncates instead.
    assert [d.num_tokens for d in _trimmed(width=3, max_capture_size=10)] == [3, 6, 9]


def test_a_trimmed_descriptor_outranks_the_graph_that_would_pad_it() -> None:
    """The exact-width entry has to be reached before a catch-all one.

    A descriptor whose uniform token count is None matches any batch, so an
    entry appended after one is never reached and the trimmed batch keeps
    falling back to piecewise.
    """
    full_width = BatchExecutionDescriptor(cg_mode=_FULL, num_tokens=16, num_reqs=2, uniform_token_count=8)
    catch_all = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=16, num_reqs=None)
    candidates = {(12, 0): [full_width, catch_all], (16, 0): [full_width, catch_all]}
    capture_descs: dict = {_FULL: [full_width]}

    trimmed = BatchExecutionDescriptor(cg_mode=_FULL, num_tokens=12, num_reqs=4, uniform_token_count=3)
    added = merge_decode_descriptors(candidates, capture_descs, [trimmed], _FULL)

    assert added == [trimmed]
    assert candidates[(12, 0)] == [trimmed, full_width, catch_all]
    # Untouched token counts keep their order.
    assert candidates[(16, 0)] == [full_width, catch_all]
    # Capture order stays widest-first, the order the base captures in.
    assert capture_descs[_FULL] == [full_width, trimmed]


def test_an_unreachable_token_count_is_dropped_rather_than_captured() -> None:
    """A graph the dispatch table cannot reach is only wasted memory.

    The base fills a dispatch entry for every token count up to the largest it
    captured; above that there is nothing to dispatch through.
    """
    candidates: dict = {(3, 0): []}
    capture_descs: dict = {}
    reachable = BatchExecutionDescriptor(cg_mode=_FULL, num_tokens=3, num_reqs=1, uniform_token_count=3)
    beyond = BatchExecutionDescriptor(cg_mode=_FULL, num_tokens=6, num_reqs=2, uniform_token_count=3)

    added = merge_decode_descriptors(candidates, capture_descs, [reachable, beyond], _FULL)

    assert added == [reachable]
    assert capture_descs[_FULL] == [reachable]
    assert (6, 0) not in candidates


def test_merging_the_same_descriptor_twice_changes_nothing() -> None:
    desc = BatchExecutionDescriptor(cg_mode=_FULL, num_tokens=3, num_reqs=1, uniform_token_count=3)
    candidates: dict = {(3, 0): []}
    capture_descs: dict = {}

    assert merge_decode_descriptors(candidates, capture_descs, [desc], _FULL) == [desc]
    assert merge_decode_descriptors(candidates, capture_descs, [desc], _FULL) == []
    assert candidates[(3, 0)] == [desc]
    assert capture_descs[_FULL] == [desc]
