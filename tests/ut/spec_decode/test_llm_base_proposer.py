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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.utils import _disable_flash_comm_v1_context

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()


class TestDisableFlashCommV1Context:
    """``_disable_flash_comm_v1_context`` temporarily clears
    ``forward_context.flash_comm_v1_enabled`` while MarkovHead runs -- MarkovHead
    operates in the all-gathered full space, so SP's reduce-scatter must not
    split ``markov_emb`` -- then restores the original value on exit, including
    on exception. See commit c62ef687b ([BugFix] Fix `sp` in dspark).
    """

    @staticmethod
    def _patch_forward_context(monkeypatch, flash_comm_v1_enabled: bool):
        ctx = SimpleNamespace(flash_comm_v1_enabled=flash_comm_v1_enabled)
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.utils.get_forward_context",
            lambda: ctx,
        )
        return ctx

    def test_clears_while_inside_when_sp_on(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with _disable_flash_comm_v1_context():
            assert ctx.flash_comm_v1_enabled is False

    def test_restores_true_on_exit(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with _disable_flash_comm_v1_context():
            pass
        assert ctx.flash_comm_v1_enabled is True

    def test_restores_false_on_exit(self, monkeypatch):
        """SP already off -> clearing is a no-op, original False preserved."""
        ctx = self._patch_forward_context(monkeypatch, False)
        with _disable_flash_comm_v1_context():
            assert ctx.flash_comm_v1_enabled is False
        assert ctx.flash_comm_v1_enabled is False

    def test_restores_on_exception(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with pytest.raises(RuntimeError, match="boom"), _disable_flash_comm_v1_context():
            raise RuntimeError("boom")
        assert ctx.flash_comm_v1_enabled is True


class TestDraftSeqLensCpuMirror:
    """The dflash-family bake ``seq_lens - rejected + N`` happens on device;
    _update_draft_seq_lens_cpu_mirror must reproduce it exactly on host, or
    clear the mirrors so the attention builder falls back to a device read.
    Never a third outcome: a populated but wrong mirror is a wrong KV length.
    """

    @staticmethod
    def _make_proposer(*, event=None, counts=None, num_draft=None) -> AscendSpecDecodeBaseProposer:
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.runner = SimpleNamespace(
            valid_sampled_token_count_event=event,
            valid_sampled_token_count_cpu=counts,
        )
        proposer._num_draft_tokens_cpu = (
            torch.tensor(num_draft, dtype=torch.int32) if num_draft is not None else None
        )
        return proposer

    @staticmethod
    def _make_cad(_seq_lens_cpu=None, seq_lens_cpu=None) -> SimpleNamespace:
        return SimpleNamespace(
            _seq_lens_cpu=_seq_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_host_exact=False,
        )

    def test_no_rejection_adds_query_stretch_on_host(self):
        proposer = self._make_proposer()
        cad = self._make_cad(_seq_lens_cpu=torch.tensor([19, 30], dtype=torch.int32))

        proposer._update_draft_seq_lens_cpu_mirror(cad, 2, None, 7)

        assert cad._seq_lens_cpu.tolist() == [26, 37]
        assert cad.seq_lens_cpu.tolist() == [26, 37]
        assert cad._seq_lens_cpu.dtype == torch.int32
        assert cad.seq_lens_host_exact is True

    def test_rejection_uses_host_twin_of_device_formula(self):
        # Device formula: rejected = num_draft + 1 - valid when num_draft > 0.
        # Request 0: all 7 drafts accepted (valid=8) -> rejected 0.
        # Request 1: 2 accepted (valid=3) -> rejected 5.
        # Request 2: prefill, no drafts -> rejected 0 regardless of valid.
        event = MagicMock()
        proposer = self._make_proposer(
            event=event,
            counts=torch.tensor([8, 3, 1], dtype=torch.int64),
            num_draft=[7, 7, 0],
        )
        cad = self._make_cad(_seq_lens_cpu=torch.tensor([19, 30, 5], dtype=torch.int32))

        proposer._update_draft_seq_lens_cpu_mirror(cad, 3, torch.zeros(3), 7)

        event.synchronize.assert_called_once()
        assert cad._seq_lens_cpu.tolist() == [26, 32, 12]
        assert cad.seq_lens_cpu.tolist() == [26, 32, 12]
        assert cad.seq_lens_host_exact is True

    def test_rejection_without_host_counts_clears_mirrors(self):
        proposer = self._make_proposer(num_draft=[7])
        cad = self._make_cad(_seq_lens_cpu=torch.tensor([19], dtype=torch.int32))

        proposer._update_draft_seq_lens_cpu_mirror(cad, 1, torch.zeros(1), 7)

        assert cad._seq_lens_cpu is None
        assert cad.seq_lens_cpu is None
        assert cad.seq_lens_host_exact is False

    def test_second_call_in_same_step_clears_mirrors(self):
        """The draft counts are consume-once: a second bake without a fresh
        prepare_inputs_padded must fall back to the device read rather than
        reuse the previous step's numbers."""
        proposer = self._make_proposer(
            event=MagicMock(), counts=torch.tensor([8], dtype=torch.int64), num_draft=[7]
        )
        first_cad = self._make_cad(_seq_lens_cpu=torch.tensor([19], dtype=torch.int32))
        proposer._update_draft_seq_lens_cpu_mirror(first_cad, 1, torch.zeros(1), 7)
        assert first_cad.seq_lens_host_exact is True

        second_cad = self._make_cad(_seq_lens_cpu=torch.tensor([19], dtype=torch.int32))
        proposer._update_draft_seq_lens_cpu_mirror(second_cad, 1, torch.zeros(1), 7)

        assert second_cad._seq_lens_cpu is None
        assert second_cad.seq_lens_host_exact is False

    def test_short_num_draft_buffer_clears_mirrors(self):
        proposer = self._make_proposer(
            event=MagicMock(), counts=torch.tensor([8, 3], dtype=torch.int64), num_draft=[7]
        )
        cad = self._make_cad(_seq_lens_cpu=torch.tensor([19, 30], dtype=torch.int32))

        proposer._update_draft_seq_lens_cpu_mirror(cad, 2, torch.zeros(2), 7)

        assert cad._seq_lens_cpu is None
        assert cad.seq_lens_cpu is None
        assert cad.seq_lens_host_exact is False

    def test_missing_base_keeps_mirrors_cleared(self):
        proposer = self._make_proposer()
        cad = self._make_cad()

        proposer._update_draft_seq_lens_cpu_mirror(cad, 1, None, 7)

        assert cad._seq_lens_cpu is None
        assert cad.seq_lens_cpu is None
        assert cad.seq_lens_host_exact is False

    def test_falls_back_to_subclass_field_when_parent_mirror_absent(self):
        proposer = self._make_proposer()
        cad = self._make_cad(seq_lens_cpu=torch.tensor([19], dtype=torch.int32))

        proposer._update_draft_seq_lens_cpu_mirror(cad, 1, None, 7)

        assert cad._seq_lens_cpu.tolist() == [26]
        assert cad.seq_lens_cpu.tolist() == [26]
        assert cad.seq_lens_host_exact is True
