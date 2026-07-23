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

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.spec_decode.parallel_drafting_inputs import expand_parallel_drafting_inputs

MASK_ID = 151666
BLOCK_SIZE = 128
GUARD = -99  # sentinel written past the used region to catch overruns


def reference_expand(
    *,
    next_token_ids,
    target_positions,
    context_slot_mapping,
    out_input_ids,
    out_context_positions,
    out_query_positions,
    out_context_slot_mapping,
    out_query_slot_mapping,
    out_token_indices,
    block_table,
    query_start_loc,
    seq_lens,
    num_rejected_tokens,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    batch_size,
    sample_from_anchor,
):
    """Line-by-line transcription of
    ops/triton/spec_decode/utils.py::copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid.

    This is the golden, not a second implementation. Do not vectorize it: its whole
    value is that a mismatch points at one specific line of the original kernel.
    """
    for req_idx in range(batch_size):
        ctx_start = int(query_start_loc[req_idx])
        ctx_end = int(query_start_loc[req_idx + 1])

        for j in range(ctx_start, ctx_end):
            out_context_positions[j] = int(target_positions[j])
            out_context_slot_mapping[j] = int(context_slot_mapping[j])

        num_rejected = int(num_rejected_tokens[req_idx]) if num_rejected_tokens is not None else 0
        valid_ctx_end = ctx_end - num_rejected
        effective_seq_len = int(seq_lens[req_idx]) - num_rejected
        last_pos = int(target_positions[valid_ctx_end - 1])

        for q_idx in range(num_query_per_req):
            out_idx = req_idx * num_query_per_req + q_idx
            out_query_positions[out_idx] = last_pos + 1 + q_idx

            cache_pos = effective_seq_len + q_idx
            block_id = int(block_table[req_idx, cache_pos // block_size])
            out_query_slot_mapping[out_idx] = block_id * block_size + cache_pos % block_size

            if q_idx == 0:
                out_input_ids[out_idx] = int(next_token_ids[req_idx])
            else:
                out_input_ids[out_idx] = parallel_drafting_token_id

            if sample_from_anchor:
                out_token_indices[req_idx * num_speculative_tokens + q_idx] = out_idx
            elif q_idx > 0:
                out_token_indices[req_idx * num_speculative_tokens + q_idx - 1] = out_idx


def make_case(*, ctx_lens, seq_lens, rejected, num_query_per_req, num_spec):
    """Build one input set.

    ctx_lens is this round's scheduled segment per request; seq_lens is the total KV
    length per request. They are NOT the same thing -- conflating them is the easiest
    way to write a test that passes against a broken helper. Positions are absolute:
    a request whose total KV is 257 and which scheduled 4 tokens this round carries
    positions [253, 254, 255, 256].
    """
    batch_size = len(ctx_lens)
    assert len(seq_lens) == batch_size
    for n_ctx, n_seq in zip(ctx_lens, seq_lens):
        assert n_seq >= n_ctx, "total KV length cannot be shorter than this round's segment"
    total = sum(ctx_lens)

    qsl = torch.zeros(batch_size + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(ctx_lens, dtype=torch.int32).cumsum(0)

    positions = torch.cat(
        [torch.arange(n_seq - n_ctx, n_seq, dtype=torch.int32) for n_ctx, n_seq in zip(ctx_lens, seq_lens)]
    )
    ctx_slots = torch.arange(total, dtype=torch.int32) * 3 + 7

    max_blocks = (max(seq_lens) + num_query_per_req) // BLOCK_SIZE + 2
    # Descending, non-contiguous physical page ids, shifted past the logical index
    # range so no entry can equal its own column.
    #
    # That invariant is what makes these fixtures discriminating: a query slot is
    # physical * block_size + offset while the raw cache position is
    # logical * block_size + offset, with the same offset. So physical != logical
    # is exactly the condition under which a slot differs from the position, and a
    # helper that ignored the block table entirely would be caught.
    #
    # The plain reversal used before had a fixed point at the middle column for odd
    # max_blocks (arange(3).flip(0) == [2, 1, 0], so column 1 maps to page 1). With
    # seq_lens=[129] the whole query block landed on that column and every slot
    # equalled its cache position.
    block_table = (
        torch.arange(batch_size * max_blocks, dtype=torch.int32).flip(0).reshape(batch_size, max_blocks) + max_blocks
    )
    assert bool(
        (block_table != torch.arange(max_blocks, dtype=torch.int32)).all()
    ), "block table maps some logical page to the same physical page; slots would be degenerate"

    return dict(
        next_token_ids=torch.arange(batch_size, dtype=torch.int32) + 1000,
        target_positions=positions,
        context_slot_mapping=ctx_slots,
        block_table=block_table,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_rejected_tokens=(torch.tensor(rejected, dtype=torch.int32) if rejected is not None else None),
        parallel_drafting_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_spec,
        total_input_tokens=total,
        batch_size=batch_size,
    )


def run_both(case, sample_from_anchor):
    b, q, k = case["batch_size"], case["num_query_per_req"], case["num_speculative_tokens"]
    total = case["total_input_tokens"]
    slack = 8  # extra room so an overrun lands on GUARD instead of out of bounds

    def fresh():
        return dict(
            out_input_ids=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_context_positions=torch.full((total + slack,), GUARD, dtype=torch.int32),
            out_query_positions=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_context_slot_mapping=torch.full((total + slack,), GUARD, dtype=torch.int32),
            out_query_slot_mapping=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_token_indices=torch.full((b * k + slack,), GUARD, dtype=torch.int32),
        )

    got, want = fresh(), fresh()
    expand_parallel_drafting_inputs(**case, **got, sample_from_anchor=sample_from_anchor)
    reference_expand(**case, **want, sample_from_anchor=sample_from_anchor)
    return got, want, slack


class TestExpandParallelDraftingInputs(TestBase):
    def _assert_matches(self, case, sample_from_anchor):
        got, want, slack = run_both(case, sample_from_anchor)
        for name in want:
            torch.testing.assert_close(got[name], want[name], msg=f"{name} mismatch")
            tail = got[name][-slack:]
            self.assertTrue(
                bool((tail == GUARD).all()),
                f"{name}: helper wrote past its used region (tail={tail.tolist()})",
            )

    def test_dflash_single_request(self):
        self._assert_matches(
            make_case(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=9, num_spec=8),
            sample_from_anchor=False,
        )

    def test_dspark_single_request(self):
        self._assert_matches(
            make_case(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=7, num_spec=7),
            sample_from_anchor=True,
        )

    def test_ragged_segment_with_distinct_seq_lens(self):
        self._assert_matches(
            make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=None, num_query_per_req=9, num_spec=8),
            sample_from_anchor=False,
        )

    def test_rejected_tail_excluded_from_query_but_kept_in_context(self):
        # Every request keeps at least one valid context token after rejection.
        self._assert_matches(
            make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1], num_query_per_req=9, num_spec=8),
            sample_from_anchor=False,
        )
        self._assert_matches(
            make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1], num_query_per_req=7, num_spec=7),
            sample_from_anchor=True,
        )

    def test_query_slots_cross_page_boundary(self):
        # effective_seq_len lands at 127/128/129 so the K+1 query slots straddle a
        # kernel page and must follow the block table into a new physical page.
        for seq_len in (127, 128, 129):
            self._assert_matches(
                make_case(ctx_lens=[2], seq_lens=[seq_len], rejected=None, num_query_per_req=9, num_spec=8),
                sample_from_anchor=False,
            )

    def test_context_buffer_untouched_beyond_scheduled_segment(self):
        case = make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=None, num_query_per_req=9, num_spec=8)
        got, _, _ = run_both(case, sample_from_anchor=False)
        total = case["total_input_tokens"]
        self.assertTrue(bool((got["out_context_positions"][total:] == GUARD).all()))
        self.assertTrue(bool((got["out_context_slot_mapping"][total:] == GUARD).all()))

    def test_query_slots_never_equal_raw_cache_positions(self):
        """The block table must be load-bearing for every shape the suite uses.

        This is implied by the no-fixed-point invariant asserted in make_case, but
        checking the end result across all shapes keeps the two from drifting apart
        if the fixture construction is ever changed.
        """
        shapes = [
            dict(ctx_lens=[2], seq_lens=[127], rejected=None, num_query_per_req=9, num_spec=8),
            dict(ctx_lens=[2], seq_lens=[128], rejected=None, num_query_per_req=9, num_spec=8),
            dict(ctx_lens=[2], seq_lens=[129], rejected=None, num_query_per_req=9, num_spec=8),
            dict(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=7, num_spec=7),
            dict(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1], num_query_per_req=9, num_spec=8),
        ]
        for shape in shapes:
            case = make_case(**shape)
            _, want, _ = run_both(case, sample_from_anchor=False)
            b, q = case["batch_size"], case["num_query_per_req"]
            rejected = shape["rejected"] or [0] * b
            for req in range(b):
                effective_seq_len = shape["seq_lens"][req] - rejected[req]
                slots = want["out_query_slot_mapping"][req * q : (req + 1) * q]
                raw = torch.arange(effective_seq_len, effective_seq_len + q, dtype=torch.int32)
                self.assertFalse(
                    bool(torch.equal(slots, raw)),
                    f"{shape}: request {req} slots equal raw cache positions, "
                    f"so the block table is not exercised",
                )
