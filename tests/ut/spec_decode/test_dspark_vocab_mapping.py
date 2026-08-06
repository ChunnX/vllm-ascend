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
"""DSpark drafting against a reduced (pruned) draft vocabulary.

A DSpark draft may be trained over the top-K target tokens. Two invariants
follow, and both are what these tests pin:

* the Markov head lives in *draft* space (its bias is ``draft_vocab_size``
  wide), while every id written into the block buffer is a *target* id --
  ``markov_w1`` is indexed by the previous real token, so an unmapped draft id
  is a valid row pointing at the wrong token;
* the draft ships no ``lm_head`` (training borrows the frozen target head and
  reads only the rows the mapping keeps), so serving must reproduce that same
  row slice from the target head.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# A tiny vocabulary pair: 4 draft ids selected out of 10 target ids. The kept
# ids are ascending, as t2d row selection produces them.
_KEPT_TARGET_IDS = [1, 3, 6, 9]
_DRAFT_VOCAB_SIZE = len(_KEPT_TARGET_IDS)
_TARGET_VOCAB_SIZE = 10
_HIDDEN_SIZE = 3

# d2t[i] = kept[i] - i, the offset form the checkpoint stores.
_D2T = torch.tensor([kept - i for i, kept in enumerate(_KEPT_TARGET_IDS)], dtype=torch.long)


class _StubDSparkModel:
    """Minimal stand-in for ``Qwen3DSparkForCausalLM`` drafting surface.

    ``markov_bias`` makes the next draft id a pure function of the id fed back
    in: ``draft_id = prev_token_id % draft_vocab_size``. Base logits are zero,
    so the argmax is decided entirely by the bias, which is what lets the test
    tell a target id from a draft id by looking only at the output.
    """

    def __init__(self, *, num_sample: int, reduced_vocab: bool = True):
        self.reduced_vocab = reduced_vocab
        self.vocab_size = _DRAFT_VOCAB_SIZE if reduced_vocab else _TARGET_VOCAB_SIZE
        self.num_sample = num_sample
        self.draft_id_to_target_id = _D2T.clone() if reduced_vocab else None
        self.compute_logits_calls = 0
        self.compute_draft_logits_calls = 0
        self.markov_embed_inputs: list[torch.Tensor] = []

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_draft_logits_calls += 1
        return torch.zeros((hidden_states.shape[0], self.vocab_size))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The real one scatters a pruned draft's logits into the full target
        # vocabulary; the drafting path must not go through it.
        self.compute_logits_calls += 1
        return torch.zeros((hidden_states.shape[0], _TARGET_VOCAB_SIZE))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.markov_embed_inputs.append(token_ids.clone())
        return token_ids.to(torch.float32).unsqueeze(-1)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        prev = markov_embed.squeeze(-1).to(torch.long)
        bias = torch.zeros((prev.shape[0], self.vocab_size))
        bias.scatter_(1, (prev % self.vocab_size).unsqueeze(-1), 1000.0)
        return bias

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]


def _make_proposer(model, *, num_speculative_tokens: int, seed_token_id: int, num_blocks: int = 1):
    """Bypass ``__init__``; set only what the DSpark sampling path reads."""
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.model = model
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer._dspark_draft_buffer = torch.zeros((num_blocks, 1 + num_speculative_tokens), dtype=torch.int64)
    proposer._dspark_seed_buffer = torch.full((num_blocks,), seed_token_id, dtype=torch.int64)
    return proposer


@pytest.fixture(autouse=True)
def _stub_forward_context(monkeypatch):
    """``_disable_flash_comm_v1_context`` needs a forward context to toggle."""
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.utils.get_forward_context",
        lambda: SimpleNamespace(flash_comm_v1_enabled=False),
    )


class TestDSparkSampleBlockRemapsIds:
    """The block buffer must carry target ids at every step, not draft ids."""

    def test_sampled_ids_are_target_ids_and_feed_back_as_target_ids(self):
        # seed 5 -> draft 5%4=1 -> target 3 -> draft 3%4=3 -> target 9
        #        -> draft 9%4=1 -> target 3
        num_spec = 3
        model = _StubDSparkModel(num_sample=num_spec)
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        block = proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        assert block[0].tolist() == [5, 3, 9, 3]
        # Every Markov step is fed the *target* id of the previous position.
        assert [int(t[0]) for t in model.markov_embed_inputs] == [5, 3, 9]

    def test_does_not_go_through_the_target_vocab_scatter(self):
        num_spec = 2
        model = _StubDSparkModel(num_sample=num_spec)
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        assert model.compute_draft_logits_calls == 1
        assert model.compute_logits_calls == 0

    def test_full_vocabulary_draft_is_an_identity_remap(self):
        """No d2t -> the sampled draft id is already a target id."""
        num_spec = 3
        model = _StubDSparkModel(num_sample=num_spec, reduced_vocab=False)
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        block = proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        # bias picks prev % 10, and 5 % 10 == 5 is a fixed point.
        assert block[0].tolist() == [5, 5, 5, 5]

    def test_falls_back_to_compute_logits_without_compute_draft_logits(self):
        """DSV4 DSpark defines neither hook; behavior is unchanged for it."""

        class _FullVocabOnlyModel(_StubDSparkModel):
            compute_draft_logits = None
            map_draft_to_target = None

            def __getattribute__(self, name):
                if name in ("compute_draft_logits", "map_draft_to_target"):
                    raise AttributeError(name)
                return super().__getattribute__(name)

        num_spec = 2
        model = _FullVocabOnlyModel(num_sample=num_spec, reduced_vocab=False)
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        block = proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        assert model.compute_logits_calls == 1
        assert block[0].tolist() == [5, 5, 5]


class _StubShardedLMHead:
    """A vocab-sharded ``ParallelLMHead`` stand-in holding one rank's rows."""

    def __init__(self, full_weight: torch.Tensor, *, tp_size: int, tp_rank: int, comm_group=None):
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

    def __init__(self, comm_group=None):
        self.weight = torch.zeros((_DRAFT_VOCAB_SIZE, _HIDDEN_SIZE))
        self.comm_group = comm_group
        self.loaded: torch.Tensor | None = None

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        self.loaded = loaded_weight.clone()


class TestBuildPrunedLMHead:
    """The derived head must equal the target rows d2t keeps, in draft order."""

    @staticmethod
    def _target_weight() -> torch.Tensor:
        # Row v is filled with v, so a misrouted row is unmistakable.
        return torch.arange(_TARGET_VOCAB_SIZE, dtype=torch.float32).unsqueeze(-1).repeat(1, _HIDDEN_SIZE)

    @staticmethod
    def _make_proposer(draft_head):
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "dspark"
        proposer.model = SimpleNamespace(lm_head=draft_head, draft_id_to_target_id=_D2T.clone())
        proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: _TARGET_VOCAB_SIZE))
        return proposer

    @staticmethod
    def _make_group(*, world_size: int, all_reduce=None, name: str = "tp"):
        return SimpleNamespace(world_size=world_size, all_reduce=all_reduce, unique_name=name)

    def test_single_rank_selects_the_kept_rows_in_draft_order(self):
        draft_head = _StubDraftLMHead()
        proposer = self._make_proposer(draft_head)
        target_head = _StubShardedLMHead(self._target_weight(), tp_size=1, tp_rank=0)

        proposer._build_pruned_lm_head_from_target(target_head)

        assert draft_head.loaded is not None
        expected = self._target_weight()[_KEPT_TARGET_IDS]
        assert torch.equal(draft_head.loaded, expected)

    def test_sharded_ranks_contribute_disjoint_rows_that_sum_to_the_head(self):
        """Each rank fills only the kept ids inside its own target shard.

        Their sum is the full pruned head, which is what the all-reduce
        reconstructs; double counting or a missed row would break the equality.
        """
        tp_size = 2
        full_weight = self._target_weight()
        contributions = []

        for tp_rank in range(tp_size):
            captured: list[torch.Tensor] = []

            def all_reduce(tensor, captured=captured):
                captured.append(tensor.clone())
                return tensor

            target_group = self._make_group(world_size=tp_size, all_reduce=all_reduce)
            draft_head = _StubDraftLMHead(comm_group=target_group)
            proposer = self._make_proposer(draft_head)
            target_head = _StubShardedLMHead(full_weight, tp_size=tp_size, tp_rank=tp_rank, comm_group=target_group)

            proposer._build_pruned_lm_head_from_target(target_head)

            assert len(captured) == 1
            contributions.append(captured[0])

        assert torch.equal(contributions[0] + contributions[1], full_weight[_KEPT_TARGET_IDS])
        # kept ids 1, 3 live on rank 0 (target rows 0..4); 6, 9 on rank 1.
        assert torch.equal(contributions[0][2:], torch.zeros((2, _HIDDEN_SIZE)))
        assert torch.equal(contributions[1][:2], torch.zeros((2, _HIDDEN_SIZE)))

    def test_supports_a_rank_local_draft_head_against_a_sharded_target(self):
        """``draft_tensor_parallel_size=1`` with a TP target.

        The draft head is built inside ``patch_tensor_parallel_group`` and so
        belongs to a rank-local group of its own. Reduction still happens over
        the target's group, and each rank's singleton draft head receives the
        whole reconstructed head.
        """
        tp_size = 2
        full_weight = self._target_weight()

        for tp_rank in range(tp_size):
            reduced = full_weight[_KEPT_TARGET_IDS]

            def all_reduce(tensor, reduced=reduced):
                # Stand in for the other rank's contribution: the collective
                # returns the fully reconstructed head on every participant.
                return reduced

            target_group = self._make_group(world_size=tp_size, all_reduce=all_reduce)
            draft_head = _StubDraftLMHead(comm_group=self._make_group(world_size=1, name="tp_draft"))
            proposer = self._make_proposer(draft_head)
            target_head = _StubShardedLMHead(full_weight, tp_size=tp_size, tp_rank=tp_rank, comm_group=target_group)

            proposer._build_pruned_lm_head_from_target(target_head)

            assert draft_head.loaded is not None
            assert torch.equal(draft_head.loaded, full_weight[_KEPT_TARGET_IDS])

    def test_rejects_a_quantized_target_head(self):
        proposer = self._make_proposer(_StubDraftLMHead())
        target_head = _StubShardedLMHead(self._target_weight(), tp_size=1, tp_rank=0)
        target_head.quant_method = SimpleNamespace()

        with pytest.raises(ValueError, match="unquantized target"):
            proposer._build_pruned_lm_head_from_target(target_head)

    def test_rejects_an_fp8_target_head_despite_a_floating_point_dtype(self):
        """The failure mode a dtype test misses: FP8 rows are meaningless
        without their scales, and would yield a wrong head of legal shape."""
        proposer = self._make_proposer(_StubDraftLMHead())
        target_head = _StubShardedLMHead(self._target_weight(), tp_size=1, tp_rank=0)
        target_head.weight = target_head.weight.to(torch.float8_e4m3fn)
        target_head.weight_scale = torch.ones(_TARGET_VOCAB_SIZE)
        target_head.quant_method = SimpleNamespace()

        assert target_head.weight.is_floating_point()
        with pytest.raises(ValueError, match="unquantized target"):
            proposer._build_pruned_lm_head_from_target(target_head)

    def test_rejects_a_mapping_outside_the_target_vocabulary(self):
        proposer = self._make_proposer(_StubDraftLMHead())
        proposer.model.draft_id_to_target_id = _D2T + _TARGET_VOCAB_SIZE
        target_head = _StubShardedLMHead(self._target_weight(), tp_size=1, tp_rank=0)

        with pytest.raises(ValueError, match="outside the target vocabulary"):
            proposer._build_pruned_lm_head_from_target(target_head)

    def test_rejects_an_unloaded_all_zero_mapping(self):
        """The model allocates d2t zero-filled and the loader skips it when the
        checkpoint has none. All zeros reads as the in-range mapping "keep
        target ids 0..K-1", so only an explicit check catches it."""
        proposer = self._make_proposer(_StubDraftLMHead())
        proposer.model.draft_id_to_target_id = torch.zeros_like(_D2T)
        target_head = _StubShardedLMHead(self._target_weight(), tp_size=1, tp_rank=0)

        with pytest.raises(ValueError, match="carries no d2t mapping"):
            proposer._build_pruned_lm_head_from_target(target_head)

    def test_rejects_a_mapping_that_is_not_strictly_increasing(self):
        proposer = self._make_proposer(_StubDraftLMHead())
        # kept ids 1, 3, 6, 9 -> 1, 3, 3, 9: a repeated target row.
        scrambled = _D2T.clone()
        scrambled[2] = 3 - 2
        proposer.model.draft_id_to_target_id = scrambled
        target_head = _StubShardedLMHead(self._target_weight(), tp_size=1, tp_rank=0)

        with pytest.raises(ValueError, match="not strictly increasing"):
            proposer._build_pruned_lm_head_from_target(target_head)

    def test_rejects_a_sharded_target_head_with_no_identifiable_group(self):
        """Without a comm_group the reduction group is unknown, and the global
        TP group is not a safe guess while the draft's patch is in effect."""
        proposer = self._make_proposer(_StubDraftLMHead())
        target_head = _StubShardedLMHead(self._target_weight(), tp_size=2, tp_rank=0)
        target_head.comm_group = None

        with pytest.raises(NotImplementedError, match="no comm_group"):
            proposer._build_pruned_lm_head_from_target(target_head)

    def test_reports_a_missing_target_head(self):
        proposer = self._make_proposer(_StubDraftLMHead())

        with pytest.raises(ValueError, match="no lm_head"):
            proposer._build_pruned_lm_head_from_target(None)


class TestCheckpointOwnershipDefaults:
    """``has_own_*`` must default to False, or nothing below is ever reached.

    ``process_eagle_weight`` only ever sets these to True, so the "checkpoint
    ships no lm_head / embed_tokens" case rests entirely on the class-attribute
    defaults that ``SupportsEagleBase`` contributes through the MRO. If a
    refactor dropped the protocol from the bases, every DSpark draft -- reduced
    vocabulary or not -- would silently keep an unloaded head instead of
    sharing or deriving one.
    """

    def test_defaults_are_false_before_any_weight_is_seen(self):
        assert Qwen3DSparkForCausalLM.has_own_lm_head is False
        assert Qwen3DSparkForCausalLM.has_own_embed_tokens is False

    def test_the_proposer_reads_those_defaults_as_false(self):
        assert getattr(Qwen3DSparkForCausalLM, "has_own_lm_head", True) is False
        assert getattr(Qwen3DSparkForCausalLM, "has_own_embed_tokens", True) is False


class TestMaybeShareLMHeadRouting:
    """Which of the three lm_head outcomes each checkpoint shape lands on."""

    @staticmethod
    def _make_proposer(monkeypatch, *, draft_id_to_target_id, has_own_lm_head):
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "dspark"
        proposer.model = SimpleNamespace(
            lm_head="draft-head",
            draft_id_to_target_id=draft_id_to_target_id,
            has_own_lm_head=has_own_lm_head,
        )
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=False),
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)),
        )
        proposer.use_cuda_graph = False
        built: list[object] = []
        monkeypatch.setattr(
            AscendSpecDecodeBaseProposer,
            "_build_pruned_lm_head_from_target",
            lambda self, target: built.append(target),
        )
        return proposer, built

    def test_pruned_checkpoint_without_lm_head_derives_it_from_the_target(self, monkeypatch):
        proposer, built = self._make_proposer(monkeypatch, draft_id_to_target_id=_D2T.clone(), has_own_lm_head=False)
        target = SimpleNamespace(lm_head="target-head")

        proposer._maybe_share_lm_head(target)

        assert built == ["target-head"]
        assert proposer.model.lm_head == "draft-head"

    def test_pruned_checkpoint_with_its_own_lm_head_is_left_alone(self, monkeypatch):
        proposer, built = self._make_proposer(monkeypatch, draft_id_to_target_id=_D2T.clone(), has_own_lm_head=True)
        target = SimpleNamespace(lm_head="target-head")

        proposer._maybe_share_lm_head(target)

        assert built == []
        assert proposer.model.lm_head == "draft-head"

    def test_full_vocabulary_checkpoint_still_shares_the_target_head(self, monkeypatch):
        """The unpruned path, unchanged: no d2t, no own head -> share."""
        proposer, built = self._make_proposer(monkeypatch, draft_id_to_target_id=None, has_own_lm_head=False)
        target = SimpleNamespace(lm_head="target-head")

        proposer._maybe_share_lm_head(target)

        assert built == []
        assert proposer.model.lm_head == "target-head"
