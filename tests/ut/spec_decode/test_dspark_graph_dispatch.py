#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Which ACLGraph a DSpark draft dispatches to.

The draft graph is *captured* under the descriptor the target model produced
(``dummy_run`` is handed it verbatim), so replay has to reproduce that same
descriptor. Two hooks have to agree for that to hold:

* ``get_graph_dispatch_num_tokens`` picks the token count the key is dispatched
  on;
* ``get_graph_num_input_tokens`` reads the draft's own width back out of the
  resulting descriptor.

The dispatcher always re-derives ``num_reqs`` as ``padded_num_tokens //
uniform_decode_query_len``. Anchor-first DSpark drafts only ``N`` tokens per
request against the target's ``1 + N``, so keying on the draft's own width
recovers ``ceil(N * num_reqs / (1 + N))`` requests -- equal to ``num_reqs``
only while ``num_reqs < 1 + N``. At ``num_reqs == 1 + N`` the draft width is
itself a capture size (7 * 8 = 56 for a block-7 draft) and the division gives
``num_reqs - 1``: a real, captured key belonging to the previous batch size.
Nothing raises, the wrong graph replays, and FIA reports the mismatch much
later as ``queryT(49) != actualSequenceLengthQ[-1](56)``.

These tests drive the real ``CudagraphDispatcher`` at ``FULL_DECODE_ONLY``,
where vLLM Ascend's ``_create_padded_batch_descriptor`` patch is behaviourally
identical to upstream's (it only diverges for plain ``FULL``), so no NPU and no
patch import are needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# The deployment the failure was reported on: Qwen3.6-27B, block-7 DSpark,
# max_num_seqs=8, cudagraph_capture_sizes=[8, 16, 24, 32, 40, 48, 56, 64].
_BLOCK7 = (7, 8)


def _capture_sizes(uniform_decode_query_len: int, max_num_seqs: int) -> list[int]:
    """Reproduce the sizes vLLM ends up with for a spec-decode FULL decode run.

    ``[1, 2, 4] + range(8, max + 1, 8)``, then every entry rounded up to a
    multiple of ``uniform_decode_query_len`` and de-duplicated -- see
    ``CompilationConfig.adjust_cudagraph_sizes_for_spec_decode``.
    """
    max_size = uniform_decode_query_len * max_num_seqs
    raw = [1, 2, 4] + list(range(8, max_size + 1, 8))
    rounded = {-(-size // uniform_decode_query_len) * uniform_decode_query_len for size in raw}
    return sorted(size for size in rounded if size <= max_size)


def _make_dispatcher(num_speculative_tokens: int, max_num_seqs: int) -> CudagraphDispatcher:
    """A dispatcher with real key initialization and no VllmConfig plumbing."""
    uniform_decode_query_len = 1 + num_speculative_tokens
    capture_sizes = _capture_sizes(uniform_decode_query_len, max_num_seqs)

    dispatcher = CudagraphDispatcher.__new__(CudagraphDispatcher)
    dispatcher.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=capture_sizes,
        max_cudagraph_capture_size=capture_sizes[-1],
        compile_sizes=None,
    )
    dispatcher.vllm_config = SimpleNamespace(
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    dispatcher.uniform_decode_query_len = uniform_decode_query_len
    dispatcher.cudagraph_keys = {
        CUDAGraphMode.PIECEWISE: set(),
        CUDAGraphMode.FULL: set(),
    }
    dispatcher.keys_initialized = False
    dispatcher.specialize_lora_count = False
    dispatcher.cudagraph_mode = CUDAGraphMode.NONE
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, uniform_decode_query_len)
    return dispatcher


def _make_proposer(num_speculative_tokens: int, *, sample_from_anchor: bool = True) -> AscendDSparkProposer:
    proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    # Anchor-first drafts N query tokens; bonus-anchor drafts the full 1 + N.
    proposer.sample_from_anchor = sample_from_anchor
    proposer.num_query_per_req = num_speculative_tokens if sample_from_anchor else 1 + num_speculative_tokens
    return proposer


def _replay_width(dispatcher, proposer, batch_size: int) -> tuple[CUDAGraphMode, int]:
    """The mode and draft width one decode step would replay at."""
    draft_tokens = batch_size * proposer.num_query_per_req
    mode, descriptor = dispatcher.dispatch(
        num_tokens=proposer.get_graph_dispatch_num_tokens(draft_tokens, batch_size, True),
        uniform_decode=True,
    )
    return mode, proposer.get_graph_num_input_tokens(descriptor)


def _capture_width(dispatcher, proposer, batch_size: int) -> int:
    """The draft width ``dummy_run`` captures at, from the target's descriptor.

    ``_dummy_run`` hands the drafter the descriptor the *target* dispatched on,
    i.e. the verification width ``(1 + N) * batch_size``.
    """
    _, target_descriptor = dispatcher.dispatch(
        num_tokens=batch_size * (1 + proposer.num_speculative_tokens),
        uniform_decode=True,
    )
    return proposer.get_graph_num_input_tokens(target_descriptor)


class TestReplayMatchesCapture:
    """The invariant: replay lands on the graph capture built, and never on a
    narrower one.

    A *wider* graph is legal -- the tail is absorbed by the virtual padding
    request ``_propose`` appends -- and does happen when the verification width
    ``(1 + N) * batch_size`` is not itself a capture size, e.g. 4 * 5 = 20 for a
    block-3 draft. A *narrower* graph silently drops whole requests, which is
    the failure this pins.
    """

    @pytest.mark.parametrize(
        ("num_speculative_tokens", "max_num_seqs"),
        [
            _BLOCK7,  # bs == 1 + N is reachable: the reported failure
            (7, 16),  # bs > 1 + N as well
            (3, 8),  # a shorter block, where the old code truncates from bs == 5
        ],
    )
    def test_every_batch_size_replays_its_own_graph(self, num_speculative_tokens, max_num_seqs):
        dispatcher = _make_dispatcher(num_speculative_tokens, max_num_seqs)
        proposer = _make_proposer(num_speculative_tokens)

        for batch_size in range(1, max_num_seqs + 1):
            draft_tokens = batch_size * proposer.num_query_per_req
            mode, replay_width = _replay_width(dispatcher, proposer, batch_size)

            assert mode == CUDAGraphMode.FULL, f"batch_size={batch_size} fell out of graph mode"
            assert replay_width >= draft_tokens, (
                f"batch_size={batch_size} would drop {draft_tokens - replay_width} draft tokens"
            )
            assert replay_width == _capture_width(dispatcher, proposer, batch_size), (
                f"batch_size={batch_size} replays a graph it was not captured as"
            )

    def test_block7_needs_no_padding_at_any_batch_size(self):
        """``1 + N == 8`` divides every capture size, so the graph is exact."""
        num_speculative_tokens, max_num_seqs = _BLOCK7
        dispatcher = _make_dispatcher(num_speculative_tokens, max_num_seqs)
        proposer = _make_proposer(num_speculative_tokens)

        for batch_size in range(1, max_num_seqs + 1):
            _, replay_width = _replay_width(dispatcher, proposer, batch_size)

            assert replay_width == batch_size * proposer.num_query_per_req

    def test_bonus_anchor_drafts_at_the_verification_width(self):
        """``num_query_per_req == 1 + N``: draft and target widths coincide."""
        num_speculative_tokens, max_num_seqs = _BLOCK7
        dispatcher = _make_dispatcher(num_speculative_tokens, max_num_seqs)
        proposer = _make_proposer(num_speculative_tokens, sample_from_anchor=False)

        for batch_size in range(1, max_num_seqs + 1):
            mode, replay_width = _replay_width(dispatcher, proposer, batch_size)

            assert mode == CUDAGraphMode.FULL
            assert replay_width == batch_size * (1 + num_speculative_tokens)


class TestTheCollisionItself:
    """Pin the mechanism, so a future change to either hook has to face it."""

    def test_keying_on_the_draft_width_lands_on_the_previous_batch_size(self):
        """What the code did before: dispatch on ``N * batch_size`` directly.

        This is the 561002 in miniature -- 56 draft tokens dispatch to the
        batch-7 key and come back 49 wide, one request short.
        """
        num_speculative_tokens, max_num_seqs = _BLOCK7
        dispatcher = _make_dispatcher(num_speculative_tokens, max_num_seqs)
        proposer = _make_proposer(num_speculative_tokens)
        batch_size = 1 + num_speculative_tokens  # 8
        assert batch_size <= max_num_seqs, "the collision has to be reachable"

        draft_tokens = batch_size * proposer.num_query_per_req
        mode, descriptor = dispatcher.dispatch(num_tokens=draft_tokens, uniform_decode=True)

        assert draft_tokens == 56
        # A real, captured key -- which is why nothing raised at dispatch time.
        assert mode == CUDAGraphMode.FULL
        assert descriptor.num_reqs == batch_size - 1
        assert proposer.get_graph_num_input_tokens(descriptor) == 49

        # And what the hook does instead.
        assert proposer.get_graph_dispatch_num_tokens(draft_tokens, batch_size, True) == 64
        assert _replay_width(dispatcher, proposer, batch_size) == (CUDAGraphMode.FULL, 56)

    def test_the_safe_range_stops_exactly_at_the_block_size(self):
        """``ceil(N * bs / (1 + N)) == bs`` iff ``bs < 1 + N``.

        Batch sizes below that were correct by coincidence, which is why 4-way
        concurrency passing said nothing about 8.
        """
        num_speculative_tokens, max_num_seqs = _BLOCK7
        dispatcher = _make_dispatcher(num_speculative_tokens, max_num_seqs)
        proposer = _make_proposer(num_speculative_tokens)

        for batch_size in range(1, num_speculative_tokens + 1):
            draft_tokens = batch_size * proposer.num_query_per_req
            _, descriptor = dispatcher.dispatch(num_tokens=draft_tokens, uniform_decode=True)
            assert descriptor.num_reqs == batch_size

        _, descriptor = dispatcher.dispatch(
            num_tokens=(1 + num_speculative_tokens) * proposer.num_query_per_req,
            uniform_decode=True,
        )
        assert descriptor.num_reqs == num_speculative_tokens


class TestBaseProposerIsUnchanged:
    """Serial methods draft at the width they dispatch at; leave them alone."""

    @pytest.mark.parametrize("num_tokens", [1, 7, 56, 64])
    @pytest.mark.parametrize("uniform_decode", [True, False])
    def test_the_base_hook_is_the_identity(self, num_tokens, uniform_decode):
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)

        assert proposer.get_graph_dispatch_num_tokens(num_tokens, 8, uniform_decode) == num_tokens

    def test_dspark_defers_to_it_for_non_uniform_batches(self):
        """A mixed prefill/decode batch has no uniform key to reproduce."""
        proposer = _make_proposer(7)

        assert proposer.get_graph_dispatch_num_tokens(56, 8, False) == 56
        assert proposer.get_graph_dispatch_num_tokens(56, 0, True) == 56

    def test_dspark_never_narrows_a_dp_inflated_count(self):
        """DP padding raises the token count; rounding it back down would
        re-introduce the same class of mismatch, so take the wider one."""
        proposer = _make_proposer(7)

        assert proposer.get_graph_dispatch_num_tokens(96, 8, True) == 96
