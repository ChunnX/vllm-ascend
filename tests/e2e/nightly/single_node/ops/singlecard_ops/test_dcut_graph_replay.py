"""Does a captured D-Cut graph honour the per-row lengths it is replayed with?

This is the gate for variable-length full-graph decode. The linear-attention
layers get no replay-time parameter update -- the helper the FIA path's comment
names for them does not exist -- so whatever a conv or recurrent task holds
after capture is what every replay runs. One token bucket has to serve many
request-count and verify-width combinations, so a full graph can only carry
trimmed batches if these operators read ``query_start_loc`` at run time rather
than baking the capture-time distribution in.

Each capture/replay pair has a control that keeps the distribution fixed and
changes only the values. The control must pass: if it does not, the harness is
wrong and the varying-width result says nothing.
"""

import pytest
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

PAD_SLOT_ID = -1

_NUM_ROWS = 8  # request axis, fixed at capture
_NUM_TOKENS = 8  # token axis, fixed at capture
_STATE_LEN = 8  # candidate state rows per request (num_spec + 1)
_NUM_KEY_HEADS = 2
_NUM_VALUE_HEADS = 4
_HEAD_DIM = 64
_CONV_WIDTH = 4
_CONV_DIM = 64
_CONV_STATE_LEN = 8

# One row of eight tokens beside seven inactive rows, and eight rows of one
# token: the two distributions a single eight-token bucket has to serve. The
# first is what a real single-request speculative batch replays; the second is
# what the variable-length dummy batch captures.
_ONE_WIDE_ROW = [8, 0, 0, 0, 0, 0, 0, 0]
_EIGHT_NARROW_ROWS = [1, 1, 1, 1, 1, 1, 1, 1]
_TWO_WIDE_ROWS = [4, 4, 0, 0, 0, 0, 0, 0]
# Six tokens on an eight-token axis. A trimmed batch rarely lands exactly on a
# capture size, so the replay occupies part of the token axis the graph holds
# and the tail slots carry whatever the previous step left.
_PARTIAL_TOKENS = [3, 3, 0, 0, 0, 0, 0, 0]


def _query_start_loc(widths: list[int]) -> torch.Tensor:
    assert len(widths) == _NUM_ROWS and sum(widths) <= _NUM_TOKENS
    starts = [0]
    for width in widths:
        starts.append(starts[-1] + width)
    return torch.tensor(starts, dtype=torch.int32)


def _state_indices(widths: list[int]) -> torch.Tensor:
    """Row ``p`` owns state rows ``[p * S, (p + 1) * S)``; inactive rows skip.

    ``PAD_SLOT_ID`` is the only value the kernels treat as "skip this row" --
    ``ResolveSeqCacheIndex`` compares against ``padSlotId`` and otherwise
    accepts any in-range line, so a zero would be processed as cache line 0.
    """
    rows = [
        [row * _STATE_LEN + column for column in range(_STATE_LEN)]
        if width > 0
        else [PAD_SLOT_ID] * _STATE_LEN
        for row, width in enumerate(widths)
    ]
    return torch.tensor(rows, dtype=torch.int32)


def _accepted(widths: list[int]) -> torch.Tensor:
    # The previous round's accepted count; 1 reads each request's first row and
    # keeps this test about the query distribution alone.
    return torch.ones(len(widths), dtype=torch.int32)


class _RecurrentInputs:
    """Persistent operator inputs: a captured graph holds these addresses."""

    def __init__(self, dtype: torch.dtype, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        normalized = lambda: torch.nn.functional.normalize(  # noqa: E731
            torch.rand(_NUM_TOKENS, _NUM_KEY_HEADS, _HEAD_DIM, generator=generator),
            p=2,
            dim=-1,
        ).to(dtype)
        self.query = normalized().npu()
        self.key = normalized().npu()
        self.value = torch.rand(_NUM_TOKENS, _NUM_VALUE_HEADS, _HEAD_DIM, generator=generator).to(dtype).npu()
        self.beta = torch.rand(_NUM_TOKENS, _NUM_VALUE_HEADS, generator=generator).to(dtype).npu()
        self.g = torch.rand(_NUM_TOKENS, _NUM_VALUE_HEADS, generator=generator, dtype=torch.float32).npu()
        self.state = (
            torch.rand(
                _NUM_ROWS * _STATE_LEN,
                _NUM_VALUE_HEADS,
                _HEAD_DIM,
                _HEAD_DIM,
                generator=generator,
                dtype=torch.float32,
            )
            .to(dtype)
            .npu()
        )
        self.pristine_state = self.state.clone()
        self.query_start_loc = _query_start_loc(_EIGHT_NARROW_ROWS).npu()
        self.state_indices = _state_indices(_EIGHT_NARROW_ROWS).npu()
        self.num_accepted = _accepted(_EIGHT_NARROW_ROWS).npu()
        self.scale = _HEAD_DIM**-0.5

    def set_widths(self, widths: list[int]) -> None:
        """Refresh the geometry in place, keeping every address."""
        self.query_start_loc.copy_(_query_start_loc(widths))
        self.state_indices.copy_(_state_indices(widths))
        self.num_accepted.copy_(_accepted(widths))

    def reseed_values(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.value.copy_(
            torch.rand(_NUM_TOKENS, _NUM_VALUE_HEADS, _HEAD_DIM, generator=generator).to(self.value.dtype)
        )
        self.beta.copy_(torch.rand(_NUM_TOKENS, _NUM_VALUE_HEADS, generator=generator).to(self.beta.dtype))

    def reset_state(self) -> None:
        self.state.copy_(self.pristine_state)

    def run(self) -> torch.Tensor:
        return torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
            self.query,
            self.key,
            self.value,
            self.state,
            beta=self.beta,
            scale=self.scale,
            query_start_loc=self.query_start_loc,
            ssm_state_indices=self.state_indices,
            num_accepted_tokens=self.num_accepted,
            g=self.g,
            zero_padded_output=False,
        )


def _eager_result(inputs: _RecurrentInputs, widths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs.set_widths(widths)
    inputs.reset_state()
    output = inputs.run()
    torch.npu.synchronize()
    return output.cpu(), inputs.state.cpu()


def _capture(inputs: _RecurrentInputs, widths: list[int]):
    inputs.set_widths(widths)
    inputs.reset_state()
    inputs.run()  # warmup outside the graph, as the attention op tests do
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output = inputs.run()
    return graph, output


@pytest.mark.parametrize("replay_widths", [_ONE_WIDE_ROW, _TWO_WIDE_ROWS])
def test_recurrent_graph_replays_the_widths_it_is_given(replay_widths: list[int]) -> None:
    """The captured task must follow ``query_start_loc``, not the capture layout.

    Capture eight rows of one token, replay the same eight rows redistributed
    into one or two wide rows with the rest skipped. Token count and request
    count are identical, so the query distribution is the only thing that
    changes. If this disagrees with eager, one token bucket cannot serve more
    than the single distribution it was captured with, and variable-length
    full-graph decode needs a replay-time refresh for these layers before
    anything else.
    """
    inputs = _RecurrentInputs(torch.bfloat16, seed=1234)
    expected_output, expected_state = _eager_result(inputs, replay_widths)

    graph, graph_output = _capture(inputs, _EIGHT_NARROW_ROWS)
    inputs.set_widths(replay_widths)
    inputs.reset_state()
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(graph_output.cpu(), expected_output, rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(inputs.state.cpu(), expected_state, rtol=3e-3, atol=1e-2)


def test_recurrent_graph_replay_control_same_widths_new_values() -> None:
    """Control for the test above: only the values move.

    A captured graph has to reread its device inputs at all. If this fails the
    harness is at fault and the varying-width result carries no information.
    """
    inputs = _RecurrentInputs(torch.bfloat16, seed=1234)
    graph, graph_output = _capture(inputs, _EIGHT_NARROW_ROWS)

    inputs.reseed_values(seed=99)
    expected_output, expected_state = _eager_result(inputs, _EIGHT_NARROW_ROWS)

    inputs.set_widths(_EIGHT_NARROW_ROWS)
    inputs.reset_state()
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(graph_output.cpu(), expected_output, rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(inputs.state.cpu(), expected_state, rtol=3e-3, atol=1e-2)


def test_recurrent_graph_replays_fewer_tokens_than_it_captured() -> None:
    """A trimmed replay occupies part of the token axis the graph holds.

    Trimming rarely lands exactly on a capture size, so the batch is padded up
    to the descriptor and ``query_start_loc`` ends below the query tensor's
    length, with the tail slots holding the previous step's values. Only the
    tokens the boundary claims are compared; the model slices the rest off.
    """
    inputs = _RecurrentInputs(torch.bfloat16, seed=1234)
    expected_output, expected_state = _eager_result(inputs, _PARTIAL_TOKENS)

    graph, graph_output = _capture(inputs, _EIGHT_NARROW_ROWS)
    inputs.set_widths(_PARTIAL_TOKENS)
    inputs.reset_state()
    graph.replay()
    torch.npu.synchronize()

    used = sum(_PARTIAL_TOKENS)
    torch.testing.assert_close(graph_output.cpu()[:used], expected_output[:used], rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(inputs.state.cpu(), expected_state, rtol=3e-3, atol=1e-2)


def test_recurrent_confines_state_writes_to_the_active_request() -> None:
    """An inactive row must not reach another request's state rows.

    The graph padding path writes ``NULL_BLOCK_ID`` (zero) into the state table
    of inactive rows, and zero is an in-range cache line, so
    ``ResolveSeqCacheIndex`` accepts it and the kernel works on the row -- only
    ``PAD_SLOT_ID`` makes it skip. With one wide row, cache line 0 belongs to
    the active request, so the reserved sentinel aims seven inactive rows at
    state the active request owns. Assert the contract on the skip sentinel and
    report what the reserved one actually does.
    """
    inputs = _RecurrentInputs(torch.bfloat16, seed=7)
    widths = _ONE_WIDE_ROW
    active_rows = slice(0, _STATE_LEN)
    other_rows = slice(_STATE_LEN, None)

    inputs.set_widths(widths)
    inputs.reset_state()
    skipped_output = inputs.run()
    torch.npu.synchronize()
    skipped_state = inputs.state.cpu()

    # Only the active request's own rows may move.
    torch.testing.assert_close(
        skipped_state[other_rows],
        inputs.pristine_state.cpu()[other_rows],
        rtol=0,
        atol=0,
    )

    # Now aim every inactive row at cache line 0 instead of the skip sentinel.
    reserved = _state_indices(widths)
    reserved[1:] = 0
    inputs.state_indices.copy_(reserved.npu())
    inputs.reset_state()
    reserved_output = inputs.run()
    torch.npu.synchronize()
    reserved_state = inputs.state.cpu()

    # Rows nobody addresses stay put either way.
    torch.testing.assert_close(
        reserved_state[other_rows],
        inputs.pristine_state.cpu()[other_rows],
        rtol=0,
        atol=0,
    )
    same_output = torch.allclose(reserved_output.cpu(), skipped_output.cpu(), rtol=3e-3, atol=1e-2)
    same_active = torch.allclose(
        reserved_state[active_rows],
        skipped_state[active_rows],
        rtol=3e-3,
        atol=1e-2,
    )
    print(
        "[D-Cut] inactive rows at cache line 0 vs the skip sentinel: "
        f"output identical={same_output} active state rows identical={same_active}"
    )


class _ConvInputs:
    """Persistent conv1d inputs, same fixed-address requirement."""

    def __init__(self, dtype: torch.dtype, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(_NUM_TOKENS, _CONV_DIM, generator=generator).to(dtype).npu()
        self.weight = torch.randn(_CONV_WIDTH, _CONV_DIM, generator=generator).to(dtype).npu()
        self.bias = torch.randn(_CONV_DIM, generator=generator).to(dtype).npu()
        # One conv state block per candidate position, addressed by the same
        # [B, S] table the model hands this operator -- the spec path passes
        # spec_state_indices_tensor, not a per-request slot id.
        self.state = (
            torch.randn(_NUM_ROWS * _STATE_LEN, _CONV_STATE_LEN, _CONV_DIM, generator=generator).to(dtype).npu()
        )
        self.pristine_state = self.state.clone()
        self.output = torch.empty_like(self.x)
        self.query_start_loc = _query_start_loc(_EIGHT_NARROW_ROWS).npu()
        self.cache_indices = _state_indices(_EIGHT_NARROW_ROWS).npu()
        self.num_accepted = _accepted(_EIGHT_NARROW_ROWS).npu()

    def set_widths(self, widths: list[int]) -> None:
        self.query_start_loc.copy_(_query_start_loc(widths))
        self.cache_indices.copy_(_state_indices(widths))
        self.num_accepted.copy_(_accepted(widths))

    def reset_state(self) -> None:
        self.state.copy_(self.pristine_state)
        self.output.zero_()

    def run(self) -> None:
        torch.ops._C_ascend.npu_dcut_causal_conv1d(
            self.output,
            self.x,
            self.weight,
            conv_state=self.state,
            bias=self.bias,
            query_start_loc=self.query_start_loc,
            cache_indices=self.cache_indices,
            num_accepted_tokens=self.num_accepted,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
        )


@pytest.mark.parametrize("replay_widths", [_ONE_WIDE_ROW, _TWO_WIDE_ROWS])
def test_conv1d_graph_replays_the_widths_it_is_given(replay_widths: list[int]) -> None:
    """Same question for the conv hook, which owns the other half of the state."""
    inputs = _ConvInputs(torch.bfloat16, seed=4321)

    inputs.set_widths(replay_widths)
    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    expected_output = inputs.output.cpu()
    expected_state = inputs.state.cpu()

    inputs.set_widths(_EIGHT_NARROW_ROWS)
    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        inputs.run()

    inputs.set_widths(replay_widths)
    inputs.reset_state()
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(inputs.output.cpu(), expected_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(inputs.state.cpu(), expected_state, rtol=1e-2, atol=1e-2)


def test_conv1d_graph_replay_control_same_widths_new_values() -> None:
    """Control for the conv test above: only the values move."""
    inputs = _ConvInputs(torch.bfloat16, seed=4321)

    inputs.set_widths(_EIGHT_NARROW_ROWS)
    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        inputs.run()

    generator = torch.Generator().manual_seed(5)
    inputs.x.copy_(torch.randn(_NUM_TOKENS, _CONV_DIM, generator=generator).to(inputs.x.dtype))

    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    expected_output = inputs.output.cpu()
    expected_state = inputs.state.cpu()

    inputs.reset_state()
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(inputs.output.cpu(), expected_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(inputs.state.cpu(), expected_state, rtol=1e-2, atol=1e-2)



def test_conv1d_graph_replays_fewer_tokens_than_it_captured() -> None:
    """Same partial token axis for the conv hook."""
    inputs = _ConvInputs(torch.bfloat16, seed=4321)

    inputs.set_widths(_PARTIAL_TOKENS)
    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    expected_output = inputs.output.cpu()
    expected_state = inputs.state.cpu()

    inputs.set_widths(_EIGHT_NARROW_ROWS)
    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        inputs.run()

    inputs.set_widths(_PARTIAL_TOKENS)
    inputs.reset_state()
    graph.replay()
    torch.npu.synchronize()

    used = sum(_PARTIAL_TOKENS)
    torch.testing.assert_close(inputs.output.cpu()[:used], expected_output[:used], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(inputs.state.cpu(), expected_state, rtol=1e-2, atol=1e-2)


def test_conv1d_confines_state_writes_to_the_active_request() -> None:
    """The conv hook's half of the empty-row contract, which had no test.

    The model tells this operator ``pad_slot_id=PAD_SLOT_ID`` and the builder
    was filling an empty row's ``cache_indices`` with ``NULL_BLOCK_ID`` -- zero,
    an in-range cache line. The operator is therefore told that -1 means skip
    and handed 0, so it treats every empty row as real cache line 0.

    Only the recurrent hook had been probed for this, and it came back
    indifferent, which is what the probe could show: it iterates per token, so a
    zero-width row does no iterations whatever its state index. The conv hook
    updates a rolling window per row, so a row it does not skip can still write.
    That asymmetry is why one measurement was not evidence about the other.

    What this pins, and what it does not: eagerly, the skip sentinel leaves the
    cache line an empty row would otherwise address untouched, and a valid line
    does get written. Both halves are measured here.

    That is not licence to switch the builder's fill to the skip sentinel --
    that was tried and measured worse on the model, taking a single request from
    an acceptance profile that decayed with position to zero acceptance and
    garbage output. This test runs the kernel eagerly, and the model's failure
    is under graph replay, which it does not cover. So the builder still fills
    NULL_BLOCK_ID, and this test stands as the record that doing so writes real
    state: both the hazard and the fact that the obvious remedy is not one.

    The remaining question for the kernel is whether its skip path is graph-safe
    -- an early return under replay would leave the padded output buffer holding
    whatever was there before, since it is not zeroed.
    """
    inputs = _ConvInputs(torch.bfloat16, seed=11)
    widths = _ONE_WIDE_ROW
    # Row 0 is the only live request; it owns conv state lines [0, _STATE_LEN).
    active_lines = slice(0, _STATE_LEN)
    idle_lines = slice(_STATE_LEN, None)

    inputs.set_widths(widths)
    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    skipped_state = inputs.state.cpu()

    # With the skip sentinel, nothing outside the live request's own lines moves.
    torch.testing.assert_close(
        skipped_state[idle_lines],
        inputs.pristine_state.cpu()[idle_lines],
        rtol=0,
        atol=0,
    )

    # Now aim all seven empty rows at cache line 0, which the live request owns,
    # exactly as NULL_BLOCK_ID did.
    reserved = _state_indices(widths)
    reserved[1:] = 0
    inputs.cache_indices.copy_(reserved.npu())
    inputs.reset_state()
    inputs.run()
    torch.npu.synchronize()
    reserved_state = inputs.state.cpu()

    # Lines nobody addresses stay put either way.
    torch.testing.assert_close(
        reserved_state[idle_lines],
        inputs.pristine_state.cpu()[idle_lines],
        rtol=0,
        atol=0,
    )
    # The live request's own state must not survive seven empty rows pointed at
    # it. If this holds, the sentinel is load bearing for the conv hook.
    assert not torch.allclose(
        reserved_state[active_lines],
        skipped_state[active_lines],
        rtol=3e-3,
        atol=1e-2,
    ), (
        "conv state at cache line 0 was unchanged by seven empty rows aimed at "
        "it, so this operator tolerates NULL_BLOCK_ID and the single-request "
        "collapse has another cause"
    )
