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
# This file is a part of the vllm-ascend embed_tokensect.
#
"""CPU-only tests for Qwen3 DSpark weight loading."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import vllm_ascend.models.qwen3_dspark as qwen3_dspark
from vllm_ascend.models import qwen3_dflash2
from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM


class TestQwen3DSparkWeightLoading:
    """Tests for Qwen3 DSpark weight loading."""

    def test_rotates_only_fc_weights(self) -> None:
        """Post-processing rotates FC weights while preserving draft-owned vocab weights."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.fc = nn.Linear(2, 2, bias=False)
        model.model.embed_tokens = nn.Embedding(2, 2)
        model.lm_head = nn.Linear(2, 2, bias=False)
        model.has_own_embed_tokens = model.has_own_lm_head = True
        rotation_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        fc_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model.model.fc.weight.data.copy_(fc_weight)
        embed_before = model.model.embed_tokens.weight.detach().clone()
        head_before = model.lm_head.weight.detach().clone()
        config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace()))
        with (
            patch.object(qwen3_dspark, "get_rotation_path", return_value="quarot.safetensors"),
            patch.object(qwen3_dspark, "get_rotation_matrix", return_value=rotation_matrix) as rotation_loader,
        ):
            model.post_process(config)
        rotation_loader.assert_called_once_with("quarot.safetensors")
        torch.testing.assert_close(model.model.fc.weight, fc_weight @ rotation_matrix)
        torch.testing.assert_close(model.model.embed_tokens.weight, embed_before)
        torch.testing.assert_close(model.lm_head.weight, head_before)


@pytest.mark.parametrize("model_cls", [qwen3_dspark.AscendQwen3DSparkForCausalLM, DFlash2Qwen3ForCausalLM])
def test_quarot_loads_missing_target_vocab_shards(tmp_path, model_cls) -> None:
    embed_name = "language_model.model.embed_tokens.weight"
    head_name = "language_model.lm_head.weight"
    shard_name = "model-00001-of-00001.safetensors"
    target_weight = torch.arange(8, dtype=torch.float32).view(4, 2)
    save_file({embed_name: target_weight, head_name: target_weight + 10}, tmp_path / shard_name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {embed_name: shard_name, head_name: shard_name}}),
        encoding="utf-8",
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotation_path = tmp_path / "rotation.safetensors"
    save_file({"global_rotation": rotation}, rotation_path)

    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.rotation_path = rotation_path
    model.target_model_path = tmp_path
    model.enable_confidence_head = False
    model.model = SimpleNamespace(embed_tokens=nn.Linear(2, 3, bias=False), fc=nn.Linear(2, 2, bias=False))
    model.lm_head = nn.Linear(2, 3, bias=False)
    for layer in (model.model.embed_tokens, model.lm_head):
        layer.weight.data.fill_(99)
        layer.shard_indices = SimpleNamespace(org_vocab_start_index=1, org_vocab_end_index=3)

    original_embed, original_head = model.model.embed_tokens, model.lm_head

    def vocab_layer(vocab_size, hidden_size, params_dtype):
        layer = nn.Linear(hidden_size, 3, bias=False, dtype=params_dtype)
        layer.shard_indices = SimpleNamespace(org_vocab_start_index=1, org_vocab_end_index=3)
        layer.quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: None)
        return layer

    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), hf_text_config=SimpleNamespace(vocab_size=4, hidden_size=2))
    )
    with (
        patch.object(qwen3_dspark, "get_rotation_path", return_value=rotation_path),
        patch.object(qwen3_dflash2, "get_rotation_path", return_value=rotation_path),
        patch.object(qwen3_dspark, "VocabParallelEmbedding", side_effect=vocab_layer),
        patch.object(qwen3_dspark, "ParallelLMHead", side_effect=vocab_layer),
    ):
        model.post_process(config)
    assert model.model.embed_tokens is not original_embed
    assert model.lm_head is not original_head
    torch.testing.assert_close(original_embed.weight, torch.full((3, 2), 99.0))
    torch.testing.assert_close(original_head.weight, torch.full((3, 2), 99.0))

    for layer, weight in ((model.model.embed_tokens, target_weight), (model.lm_head, target_weight + 10)):
        expected = torch.cat((weight[1:3] @ rotation.T, torch.zeros(1, 2)))
        torch.testing.assert_close(layer.weight, expected)
    assert model.has_own_embed_tokens
    assert model.has_own_lm_head


class TestReducedVocabCheckpointFlags:
    """What the checkpoint carried, which its loaded contents cannot say.

    ``draft_id_to_target_id`` is allocated zero-filled and skipped by the loader
    when the checkpoint has no ``d2t``, so an unloaded buffer is
    indistinguishable from the legal mapping that keeps target ids ``0..K-1``.
    Only the weight stream knows, and only while it is passing through.
    """

    @staticmethod
    def _load(weights, *, draft_id_to_target_id=None):
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        object.__setattr__(model, "rotation_path", None)
        object.__setattr__(model, "enable_confidence_head", False)
        object.__setattr__(model, "has_own_lm_head", False)
        object.__setattr__(model, "draft_id_to_target_id", draft_id_to_target_id)
        with (
            patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent,
            patch.object(model_cls, "_load_reduced_vocab_without_head"),
        ):
            model.load_weights(weights)
            # The parent consumes the stream; the flags are set on the way through.
            if mock_parent.called:
                list(mock_parent.call_args.args[0])
        return model

    def test_flags_default_to_absent(self) -> None:
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        assert model_cls.has_draft_id_mapping is False
        assert model_cls.has_own_lm_head_weights is False

    def test_records_a_present_mapping(self) -> None:
        model = self._load([("d2t", torch.zeros(4, dtype=torch.long))])

        assert model.has_draft_id_mapping is True
        assert model.has_own_lm_head_weights is False

    def test_records_present_lm_head_weights(self) -> None:
        model = self._load([("lm_head.weight", torch.zeros(4, 2))])

        assert model.has_own_lm_head_weights is True
        assert model.has_draft_id_mapping is False

    def test_stays_absent_for_a_full_vocabulary_checkpoint(self) -> None:
        model = self._load([("model.embed_tokens.weight", torch.zeros(4, 2))])

        assert model.has_draft_id_mapping is False
        assert model.has_own_lm_head_weights is False

    def test_records_a_present_mapping_through_the_rotation_path(self) -> None:
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        object.__setattr__(model, "rotation_path", "quarot.safetensors")
        object.__setattr__(model, "enable_confidence_head", False)
        object.__setattr__(model, "draft_id_to_target_id", None)

        with (
            patch.object(qwen3_dspark, "get_rotation_matrix", return_value=torch.eye(2)),
            patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent,
        ):
            model.load_weights([("model.fc.weight", torch.zeros(2, 2)), ("d2t", torch.zeros(4, dtype=torch.long))])
            list(mock_parent.call_args.args[0])

        assert model.has_draft_id_mapping is True


class TestReducedVocabBlocksHeadSharing:
    """A reduced vocabulary must never accept the target's full-vocab LM head.

    ``has_own_lm_head`` drives exactly one decision in the draft loaders -- may
    the target head replace this model's -- and for a reduced vocabulary the
    answer is always no: the logits processor would slice the first
    ``draft_vocab_size`` target columns instead of the kept ones, proposing
    plausible but wrong tokens with nothing raised.
    """

    def test_reduced_vocabulary_blocks_the_share(self) -> None:
        model = TestReducedVocabCheckpointFlags._load(
            [("d2t", torch.zeros(4, dtype=torch.long))],
            draft_id_to_target_id=torch.zeros(4, dtype=torch.long),
        )

        assert model.has_own_lm_head is True
        # ...while still recording that the weights were not in the checkpoint.
        assert model.has_own_lm_head_weights is False

    def test_full_vocabulary_leaves_the_share_decision_alone(self) -> None:
        model = TestReducedVocabCheckpointFlags._load(
            [("model.embed_tokens.weight", torch.zeros(4, 2))], draft_id_to_target_id=None
        )

        assert model.has_own_lm_head is False


@pytest.mark.parametrize("with_own_head", [False, True])
def test_030_loads_reduced_checkpoint(with_own_head):
    model = qwen3_dspark.AscendQwen3DSparkForCausalLM.__new__(qwen3_dspark.AscendQwen3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=10, draft_vocab_size=4)
    model.target_vocab_size = 10
    model.draft_id_to_target_id = nn.Parameter(torch.zeros(4, dtype=torch.long), requires_grad=False)
    model.model = nn.Module()
    model.model.fc = nn.Linear(3, 3, bias=False)
    model.model.embed_tokens = nn.Embedding(10, 3)
    model.model.confidence_head = None
    model.lm_head = nn.Linear(3, 4, bias=False)
    original_head = model.lm_head.weight.detach().clone()
    offsets = torch.tensor([1, 2, 4, 6])
    with patch.object(model.model, "_build_fused_kv_buffers", create=True) as build_buffers:
        weights = [("d2t", offsets), ("fc.weight", torch.ones(3, 3))]
        if with_own_head:
            weights.append(("lm_head.weight", torch.full((4, 3), 7.0)))
        model.load_weights(iter(weights))
    torch.testing.assert_close(model.draft_id_to_target_id, offsets)
    torch.testing.assert_close(model.model.fc.weight, torch.ones(3, 3))
    torch.testing.assert_close(model.lm_head.weight, torch.full((4, 3), 7.0) if with_own_head else original_head)
    assert model.has_own_lm_head
    assert model.has_draft_id_mapping
    assert model.has_own_lm_head_weights == with_own_head
    build_buffers.assert_called_once()
