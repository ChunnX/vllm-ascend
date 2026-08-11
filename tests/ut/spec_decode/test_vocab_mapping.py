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
"""Reduced draft vocabularies: mapping validation and pruned-head derivation.

The draft ships no ``lm_head`` (training borrows the frozen target head and
reads only the rows the mapping keeps), so serving must reproduce that same row
slice. Sharing the target head instead is the silent failure this guards: the
shapes still line up, and the logits processor quietly slices the first
``draft_vocab_size`` target columns rather than the kept ones.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from vllm_ascend.spec_decode import vocab_mapping

# A tiny vocabulary pair: 4 draft ids selected out of 10 target ids. The kept
# ids are ascending, as t2d row selection produces them.
_KEPT_TARGET_IDS = [1, 3, 6, 9]
_DRAFT_VOCAB_SIZE = len(_KEPT_TARGET_IDS)
_TARGET_VOCAB_SIZE = 10
_HIDDEN_SIZE = 3

# d2t[i] = kept[i] - i, the offset form the checkpoint stores.
_D2T = torch.tensor([kept - i for i, kept in enumerate(_KEPT_TARGET_IDS)], dtype=torch.long)


def _target_weight() -> torch.Tensor:
    # Row v is filled with v, so a misrouted row is unmistakable.
    return torch.arange(_TARGET_VOCAB_SIZE, dtype=torch.float32).unsqueeze(-1).repeat(1, _HIDDEN_SIZE)


class _StubShardedLMHead:
    """A vocab-sharded ``ParallelLMHead`` stand-in holding one rank's rows."""

    def __init__(self, full_weight: torch.Tensor, *, tp_size: int = 1, tp_rank: int = 0, comm_group=None):
        vocab_size = full_weight.shape[0]
        shard = vocab_size // tp_size
        start, end = tp_rank * shard, (tp_rank + 1) * shard
        self.weight = full_weight[start:end].clone()
        self.tp_size = tp_size
        self.comm_group = comm_group
        # The real class, since the "is it quantized" check is an isinstance test.
        self.quant_method = UnquantizedEmbeddingMethod()
        self.shard_indices = SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=end)


class _StubDraftLMHead:
    """Records what ``weight_loader`` is handed, which is the full pruned head."""

    def __init__(self):
        self.weight = torch.zeros((_DRAFT_VOCAB_SIZE, _HIDDEN_SIZE))
        self.loaded: torch.Tensor | None = None

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        self.loaded = loaded_weight.clone()


def _make_group(*, world_size: int, all_reduce=None, name: str = "tp"):
    return SimpleNamespace(world_size=world_size, all_reduce=all_reduce, unique_name=name)


def _make_draft(d2t, *, has_draft_id_mapping=None, has_own_lm_head_weights=None, lm_head=None):
    model = SimpleNamespace(draft_id_to_target_id=d2t, lm_head=lm_head)
    if has_draft_id_mapping is not None:
        model.has_draft_id_mapping = has_draft_id_mapping
    if has_own_lm_head_weights is not None:
        model.has_own_lm_head_weights = has_own_lm_head_weights
    return model


class TestValidateDraftVocabMapping:
    """Whether the mapping is usable at all, independent of head routing."""

    def test_returns_the_kept_target_ids(self):
        draft = _make_draft(_D2T.clone(), has_draft_id_mapping=True)

        kept = vocab_mapping.validate_draft_vocab_mapping(draft, _TARGET_VOCAB_SIZE)

        assert kept.tolist() == _KEPT_TARGET_IDS

    def test_rejects_a_mapping_key_the_checkpoint_never_carried(self):
        draft = _make_draft(torch.zeros_like(_D2T), has_draft_id_mapping=False)

        with pytest.raises(ValueError, match="carries no d2t mapping"):
            vocab_mapping.validate_draft_vocab_mapping(draft, _TARGET_VOCAB_SIZE)

    def test_accepts_an_all_zero_mapping_the_checkpoint_did_carry(self):
        """d2t is all zeros for the mapping that keeps target ids 0..K-1, which
        is legal; only the loader flag separates it from an unloaded buffer, so
        the contents alone must not decide."""
        draft = _make_draft(torch.zeros_like(_D2T), has_draft_id_mapping=True)

        kept = vocab_mapping.validate_draft_vocab_mapping(draft, _TARGET_VOCAB_SIZE)

        assert kept.tolist() == list(range(_DRAFT_VOCAB_SIZE))

    def test_rejects_all_zeros_when_no_loader_flag_is_available(self):
        """Without the flag the state is ambiguous, and the conservative read is
        "unloaded": that failure is silent, the other is merely implausible."""
        draft = _make_draft(torch.zeros_like(_D2T))

        with pytest.raises(ValueError, match="carries no d2t mapping"):
            vocab_mapping.validate_draft_vocab_mapping(draft, _TARGET_VOCAB_SIZE)

    def test_rejects_a_mapping_that_is_out_of_range(self):
        draft = _make_draft(_D2T + _TARGET_VOCAB_SIZE, has_draft_id_mapping=True)

        with pytest.raises(ValueError, match="outside the target vocabulary"):
            vocab_mapping.validate_draft_vocab_mapping(draft, _TARGET_VOCAB_SIZE)

    def test_rejects_a_mapping_that_is_not_strictly_increasing(self):
        # kept ids 1, 3, 6, 9 -> 1, 3, 3, 9: a repeated target row.
        scrambled = _D2T.clone()
        scrambled[2] = 3 - 2
        draft = _make_draft(scrambled, has_draft_id_mapping=True)

        with pytest.raises(ValueError, match="not strictly increasing"):
            vocab_mapping.validate_draft_vocab_mapping(draft, _TARGET_VOCAB_SIZE)


class TestDerivePrunedLMHead:
    """The derived head must equal the target rows d2t keeps, in draft order."""

    def test_single_rank_selects_the_kept_rows_in_draft_order(self):
        draft_head = _StubDraftLMHead()

        vocab_mapping.derive_pruned_lm_head(
            draft_head, _StubShardedLMHead(_target_weight()), torch.tensor(_KEPT_TARGET_IDS), _TARGET_VOCAB_SIZE
        )

        assert draft_head.loaded is not None
        assert torch.equal(draft_head.loaded, _target_weight()[_KEPT_TARGET_IDS])

    def test_sharded_ranks_contribute_disjoint_rows_that_sum_to_the_head(self):
        """Each rank fills only the kept ids inside its own target shard.

        Their sum is the full pruned head, which is what the all-reduce
        reconstructs; double counting or a missed row would break the equality.
        """
        tp_size = 2
        full_weight = _target_weight()
        contributions = []

        for tp_rank in range(tp_size):
            captured: list[torch.Tensor] = []

            def all_reduce(tensor, captured=captured):
                captured.append(tensor.clone())
                return tensor

            target_head = _StubShardedLMHead(
                full_weight,
                tp_size=tp_size,
                tp_rank=tp_rank,
                comm_group=_make_group(world_size=tp_size, all_reduce=all_reduce),
            )
            vocab_mapping.derive_pruned_lm_head(
                _StubDraftLMHead(), target_head, torch.tensor(_KEPT_TARGET_IDS), _TARGET_VOCAB_SIZE
            )

            assert len(captured) == 1
            contributions.append(captured[0])

        assert torch.equal(contributions[0] + contributions[1], full_weight[_KEPT_TARGET_IDS])
        # kept ids 1, 3 live on rank 0 (target rows 0..4); 6, 9 on rank 1.
        assert torch.equal(contributions[0][2:], torch.zeros((2, _HIDDEN_SIZE)))
        assert torch.equal(contributions[1][:2], torch.zeros((2, _HIDDEN_SIZE)))

    def test_supports_a_rank_local_draft_head_against_a_sharded_target(self):
        """``draft_tensor_parallel_size=1`` with a TP target.

        The draft head belongs to a rank-local group of its own, yet reduction
        still happens over the target's group and each rank's singleton draft
        head receives the whole reconstructed head.
        """
        full_weight = _target_weight()
        reduced = full_weight[_KEPT_TARGET_IDS]

        for tp_rank in range(2):
            draft_head = _StubDraftLMHead()
            target_head = _StubShardedLMHead(
                full_weight,
                tp_size=2,
                tp_rank=tp_rank,
                # Stands in for the other rank's contribution: the collective
                # returns the fully reconstructed head on every participant.
                comm_group=_make_group(world_size=2, all_reduce=lambda tensor: reduced),
            )

            vocab_mapping.derive_pruned_lm_head(
                draft_head, target_head, torch.tensor(_KEPT_TARGET_IDS), _TARGET_VOCAB_SIZE
            )

            assert draft_head.loaded is not None
            assert torch.equal(draft_head.loaded, reduced)

    def test_rejects_a_quantized_target_head(self):
        target_head = _StubShardedLMHead(_target_weight())
        target_head.quant_method = SimpleNamespace()

        with pytest.raises(ValueError, match="unquantized target"):
            vocab_mapping.derive_pruned_lm_head(
                _StubDraftLMHead(), target_head, torch.tensor(_KEPT_TARGET_IDS), _TARGET_VOCAB_SIZE
            )

    def test_rejects_an_fp8_target_head_despite_a_floating_point_dtype(self):
        """The failure a dtype test misses: FP8 rows are meaningless without
        their scales, and would yield a wrong head of legal shape."""
        target_head = _StubShardedLMHead(_target_weight())
        target_head.weight = target_head.weight.to(torch.float8_e4m3fn)
        target_head.weight_scale = torch.ones(_TARGET_VOCAB_SIZE)
        target_head.quant_method = SimpleNamespace()

        assert target_head.weight.is_floating_point()
        with pytest.raises(ValueError, match="unquantized target"):
            vocab_mapping.derive_pruned_lm_head(
                _StubDraftLMHead(), target_head, torch.tensor(_KEPT_TARGET_IDS), _TARGET_VOCAB_SIZE
            )

    def test_rejects_a_sharded_target_head_with_no_identifiable_group(self):
        """Without a comm_group the reduction group is unknown, and the global TP
        group is not a safe guess while the draft's patch is in effect."""
        target_head = _StubShardedLMHead(_target_weight(), tp_size=2, tp_rank=0)

        with pytest.raises(NotImplementedError, match="no comm_group"):
            vocab_mapping.derive_pruned_lm_head(
                _StubDraftLMHead(), target_head, torch.tensor(_KEPT_TARGET_IDS), _TARGET_VOCAB_SIZE
            )

    def test_reports_a_missing_target_head(self):
        with pytest.raises(ValueError, match="no lm_head"):
            vocab_mapping.derive_pruned_lm_head(
                _StubDraftLMHead(), None, torch.tensor(_KEPT_TARGET_IDS), _TARGET_VOCAB_SIZE
            )


class TestSettleReducedVocabLMHead:
    """Which drafts get touched, and how much, after the model is loaded."""

    @staticmethod
    def _target_model():
        return SimpleNamespace(lm_head=_StubShardedLMHead(_target_weight()))

    def test_full_vocabulary_draft_is_untouched(self):
        """No d2t -> nothing to validate and nothing to derive."""
        draft = _make_draft(None, lm_head=_StubDraftLMHead())

        vocab_mapping.settle_reduced_vocab_lm_head(draft, self._target_model(), _TARGET_VOCAB_SIZE)

        assert draft.lm_head.loaded is None

    def test_checkpoint_without_lm_head_gets_a_derived_one(self):
        draft = _make_draft(
            _D2T.clone(), has_draft_id_mapping=True, has_own_lm_head_weights=False, lm_head=_StubDraftLMHead()
        )

        vocab_mapping.settle_reduced_vocab_lm_head(draft, self._target_model(), _TARGET_VOCAB_SIZE)

        assert torch.equal(draft.lm_head.loaded, _target_weight()[_KEPT_TARGET_IDS])

    def test_checkpoint_with_its_own_lm_head_is_left_alone(self):
        draft = _make_draft(
            _D2T.clone(), has_draft_id_mapping=True, has_own_lm_head_weights=True, lm_head=_StubDraftLMHead()
        )

        vocab_mapping.settle_reduced_vocab_lm_head(draft, self._target_model(), _TARGET_VOCAB_SIZE)

        assert draft.lm_head.loaded is None

    def test_own_lm_head_does_not_exempt_the_mapping_from_validation(self):
        """The head is kept as-is, but every proposed id is still read through
        d2t, so a checkpoint that shipped no mapping must not start."""
        draft = _make_draft(
            torch.zeros_like(_D2T),
            has_draft_id_mapping=False,
            has_own_lm_head_weights=True,
            lm_head=_StubDraftLMHead(),
        )

        with pytest.raises(ValueError, match="carries no d2t mapping"):
            vocab_mapping.settle_reduced_vocab_lm_head(draft, self._target_model(), _TARGET_VOCAB_SIZE)
