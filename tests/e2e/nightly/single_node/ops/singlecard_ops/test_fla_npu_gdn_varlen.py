# SPDX-License-Identifier: Apache-2.0
"""Does fla_npu's AscendC varlen path satisfy the trimmed-batch contract?

This is a probe, not a regression gate yet. vLLM-Ascend 0.30 replaced the
in-tree causal-conv and recurrent GDN operators with the external
``flash_linear_attention_npu`` wheel, whose interface already takes
``query_start_loc`` / ``actual_seq_lengths`` and ``num_accepted_tokens`` -- the
two parameters the D-Cut operator fork existed to add. If these gates pass,
that fork is unnecessary here and ~5600 lines of csrc can retire. The wheel is
beta, so this has to be measured rather than assumed.

The question each gate asks, in order of how much it matters:

1. ``[8, 0, 0, 0, 0, 0, 0, 0]`` -- one request owning every token beside seven
   owning none. The stock operator read this wrong: its 2D decode input carries
   no boundaries of its own, so it inferred one token per request, and that
   inference is indistinguishable from the truth exactly when the token count
   equals the request count. It wrote token 0 and silently dropped tokens 1..7.
   This is the only pattern in the set that the old reading got wrong, so a
   failure here that looks like "token 0 correct, tokens 1..7 untouched" means
   the boundaries are being ignored, not that the kernel is inaccurate.
2. Ragged state across rounds -- whether per-request history selection by
   ``num_accepted_tokens`` is honoured when this round's widths differ from the
   previous round's accepted counts.

The reference is ``tests/ut/helpers/varlen_gdn_reference.py``, an independent
numpy implementation. ``tests/ut/ops/test_varlen_gdn_reference.py`` checks the
reference itself on CPU; run that first, so a failure here is never ambiguous
between "the wheel is wrong" and "the golden is wrong".

Two conventions below are assumptions read off the call sites in
``vllm_ascend/ops/gdn.py`` rather than from documentation, and are the first
things to question if a gate fails for a reason other than dropped tokens:

* ``_NULL_BLOCK_ID`` -- ``causal_conv1d_update`` is called there with
  ``null_block_id=0``, so an inert row is marked with block 0 and slot 0 is
  reserved. The golden's own convention for an inert row is -1, so the two
  index vectors are built separately and deliberately.
* ``recurrent_gated_delta_rule`` is called there with no null-block argument at
  all, so this keeps the golden's -1 for it. If that operator rejects or
  mishandles -1, try 0 and reserve slot 0 the same way.
"""

import inspect

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import causal_conv1d_update, recurrent_gated_delta_rule

from tests.ut.helpers.varlen_gdn_reference import recurrent_reference, speculative_conv_reference

# The block id fla_npu treats as "no state for this row", per gdn.py's
# causal_conv1d_update call. Slot 0 is therefore never a live request.
_NULL_BLOCK_ID = 0
# What the numpy golden uses for the same thing.
_GOLDEN_NULL_INDEX = -1


@pytest.fixture(scope="module", autouse=True)
def _eager_device_mode():
    torch.npu.set_compile_mode(jit_compile=False)


def _live_mask(lengths: list[int]) -> np.ndarray:
    return np.asarray(lengths) != 0


def _slot_table(lengths: list[int], *, null_index: int) -> np.ndarray:
    """One physical slot per request, with the empty rows marked inert.

    Slots start at 1 so that block 0 stays reserved as fla_npu's null block.
    """
    table = np.arange(1, len(lengths) + 1, dtype=np.int32)
    table[~_live_mask(lengths)] = null_index
    return table


def test_fla_npu_exposes_the_varlen_parameters():
    """Fail on a renamed parameter here, not inside a numeric comparison.

    The whole premise is that these names exist; a signature change would
    otherwise surface as a confusing TypeError halfway through a gate.
    """
    missing: list[str] = []
    for fn, expected in (
        (
            causal_conv1d_update,
            ("query_start_loc", "num_accepted_tokens", "conv_state_indices", "max_query_len", "null_block_id"),
        ),
        (recurrent_gated_delta_rule, ("actual_seq_lengths", "num_accepted_tokens", "ssm_state_indices")),
    ):
        params = inspect.signature(fn).parameters
        # A **kwargs-only wrapper cannot be checked by name; accept it.
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        missing += [f"{fn.__name__}.{name}" for name in expected if name not in params]
    assert not missing, f"fla_npu no longer takes: {missing}"


@pytest.mark.parametrize("lengths", [[8], [8, 0, 0, 0, 0, 0, 0, 0], [3, 1, 4], [1, 1, 1]])
def test_conv_varlen_matches_cpu_golden(lengths):
    torch.manual_seed(31)
    dim, width, history = 64, 4, 10
    rows = len(lengths)
    qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    x = torch.randn(int(qsl[-1]), dim, dtype=torch.bfloat16)
    weight = torch.randn(width, dim, dtype=torch.bfloat16) * 0.2
    bias = torch.randn(dim, dtype=torch.bfloat16) * 0.1
    # One block per request plus the reserved null block at index 0.
    state = torch.randn(rows + 1, history, dim, dtype=torch.bfloat16)
    # History selector deliberately exceeds some rows' current length: the
    # accepted count belongs to the previous round, not this one.
    accepted = np.full(rows, 6, dtype=np.int32)

    expected, expected_state = speculative_conv_reference(
        x.float().numpy(),
        weight.float().numpy(),
        state.float().numpy(),
        qsl,
        _slot_table(lengths, null_index=_GOLDEN_NULL_INDEX),
        accepted,
        bias=bias.float().numpy(),
    )

    device_state = state.npu()
    output = torch.empty_like(x, device="npu")
    returned = causal_conv1d_update(
        x.npu(),
        device_state,
        weight.npu(),
        bias=bias.npu(),
        activation="silu",
        conv_state_indices=torch.from_numpy(_slot_table(lengths, null_index=_NULL_BLOCK_ID)).npu(),
        num_accepted_tokens=torch.from_numpy(accepted).npu(),
        query_start_loc=torch.from_numpy(qsl).npu(),
        max_query_len=max(lengths),
        null_block_id=_NULL_BLOCK_ID,
        out=output,
    )
    torch.npu.synchronize()
    # gdn.py reads the return value rather than the out= buffer; keep both honest.
    actual = (returned if returned is not None else output).cpu().float()

    torch.testing.assert_close(actual, torch.from_numpy(expected).float(), rtol=2e-2, atol=2e-2)
    # The rolled-back history must land exactly: it is a copy, not arithmetic.
    # Rows the golden marks inert are skipped by the operator, so only the live
    # rows' slots are compared -- slot 0 belongs to neither side's contract.
    live_slots = _slot_table(lengths, null_index=_GOLDEN_NULL_INDEX)[_live_mask(lengths)]
    torch.testing.assert_close(
        device_state.cpu().float()[live_slots],
        torch.from_numpy(expected_state).float()[live_slots],
        rtol=0,
        atol=0,
    )


def test_recurrent_multiround_ragged_state_matches_cpu_golden():
    torch.manual_seed(17)
    rows, stride, nk, nv, dim = 3, 8, 2, 4, 64
    indices = np.arange(1, rows * stride + 1, dtype=np.int32).reshape(rows, stride)
    state = torch.randn(rows * stride + 1, nv, dim, dim, dtype=torch.float32) * 0.1
    device_state = state.npu()
    # accepted is the previous round's output count, independent of this round's
    # width. Round three also drops a request to zero tokens mid-sequence.
    rounds = [([8, 8, 8], [1, 1, 1]), ([3, 1, 4], [6, 8, 3]), ([1, 4, 0], [3, 1, 4])]

    for round_idx, (lengths, accepted) in enumerate(rounds):
        qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
        tokens = int(qsl[-1])
        q = torch.nn.functional.normalize(torch.randn(tokens, nk, dim), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(torch.randn(tokens, nk, dim), dim=-1).to(torch.bfloat16)
        v = torch.randn(tokens, nv, dim, dtype=torch.bfloat16) * 0.2
        beta = torch.rand(tokens, nv, dtype=torch.bfloat16)
        g = -torch.rand(tokens, nv, dtype=torch.float32) * 0.1

        table = indices.copy()
        table[~_live_mask(lengths)] = _GOLDEN_NULL_INDEX
        expected, expected_state = recurrent_reference(
            q.float().numpy(),
            k.float().numpy(),
            v.float().numpy(),
            state.numpy(),
            beta.float().numpy(),
            g.numpy(),
            qsl,
            table,
            np.array(accepted),
            scale=dim**-0.5,
        )

        # This operator takes per-request lengths, not the cumulative vector.
        actual_seq_lengths = torch.from_numpy(np.diff(qsl).astype(np.int32)).npu()
        out = recurrent_gated_delta_rule(
            q.npu(),
            k.npu(),
            v.npu(),
            device_state,
            g=g.npu(),
            beta=beta.npu(),
            scale=dim**-0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=torch.from_numpy(table).npu().flatten(),
            num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device="npu"),
        )
        torch.npu.synchronize()

        torch.testing.assert_close(
            out.cpu().float(),
            torch.from_numpy(expected).float(),
            rtol=2e-2,
            atol=2e-3,
            msg=lambda m, i=round_idx, le=lengths: f"round {i} lengths={le} output mismatch\n{m}",
        )
        torch.testing.assert_close(
            device_state.cpu(),
            torch.from_numpy(expected_state).float(),
            rtol=2e-2,
            atol=2e-3,
            msg=lambda m, i=round_idx, le=lengths: f"round {i} lengths={le} state mismatch\n{m}",
        )
        # Carry the independent golden state forward; never seed it from the DUT,
        # or a wrong round would be hidden by the next round agreeing with it.
        state = torch.from_numpy(expected_state).float()
