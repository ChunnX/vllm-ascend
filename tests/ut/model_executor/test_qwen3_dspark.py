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

from unittest.mock import patch

import torch

import vllm_ascend.models.qwen3_dspark as qwen3_dspark


class TestQwen3DSparkWeightLoading:
    """Tests for Qwen3 DSpark weight loading."""

    def test_rotates_only_fc_weights(self) -> None:
        """Rotate FC weights and preserve all other weights before delegation."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM

        # ``load_weights`` only reads ``rotation_path`` from the model. Bypass the
        # full model constructor and nn.Module attribute handling to keep this a
        # focused CPU unit test.
        model = model_cls.__new__(model_cls)
        rotation_path = "quarot.safetensors"
        object.__setattr__(model, "rotation_path", rotation_path)
        object.__setattr__(model, "draft_id_to_target_id", None)

        # Use a non-identity matrix so an unrotated FC weight fails the assertion.
        rotation_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        fc_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        non_fc_weight = torch.tensor([[5.0, 6.0]])
        weights_to_load = [("model.fc.weight", fc_weight), ("model.embed_tokens.weight", non_fc_weight)]
        expected_fc_weight = torch.matmul(fc_weight, rotation_matrix)

        # Capture the final delegation without invoking the real model loader.
        with (
            patch.object(
                qwen3_dspark, "get_rotataion_matrix", return_value=rotation_matrix
            ) as mock_get_rotation_matrix,
            patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent_load_weights,
        ):
            model.load_weights(weights_to_load)

        mock_get_rotation_matrix.assert_called_once_with(rotation_path)
        mock_parent_load_weights.assert_called_once()

        # The stream reaches the parent through the d2t-noting pass-through, so
        # it arrives lazily; the real parent drains it into a dict.
        processed_weights = list(mock_parent_load_weights.call_args.args[0])
        torch.testing.assert_close(processed_weights[0][1], expected_fc_weight)
        torch.testing.assert_close(processed_weights[1][1], non_fc_weight)


class TestReducedVocabularyCheckpointShape:
    """What the weight stream says about ``d2t`` and the draft's own lm_head.

    ``draft_id_to_target_id`` is allocated zero-filled and skipped by the loader
    when the checkpoint has none, so an unloaded buffer is indistinguishable from
    the legal mapping that keeps target ids ``0..K-1``. Only the stream knows.
    """

    @staticmethod
    def _load(weights, *, draft_id_to_target_id, has_own_lm_head=False):
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        object.__setattr__(model, "rotation_path", None)
        object.__setattr__(model, "draft_id_to_target_id", draft_id_to_target_id)
        object.__setattr__(model, "has_own_lm_head", has_own_lm_head)
        with patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent:
            model.load_weights(iter(weights))
            # The real parent consumes the stream; the flag is set on the way
            # through, so a mocked parent has to drain it explicitly.
            list(mock_parent.call_args.args[0])
        return model

    def test_defaults_are_absent_before_any_weight_is_seen(self) -> None:
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        assert model_cls.has_draft_id_mapping is False
        assert model_cls.lm_head_needs_target_rows is False

    def test_records_a_present_mapping(self) -> None:
        model = self._load(
            [("d2t", torch.zeros(4, dtype=torch.long)), ("lm_head.weight", torch.zeros(4, 2))],
            draft_id_to_target_id=torch.zeros(4, dtype=torch.long),
            has_own_lm_head=True,
        )

        assert model.has_draft_id_mapping is True
        # The checkpoint shipped its own head, so there is nothing to derive.
        assert model.lm_head_needs_target_rows is False

    def test_pruned_checkpoint_without_lm_head_claims_its_own_head(self) -> None:
        """Otherwise upstream aliases the full target head over the pruned one."""
        model = self._load(
            [("d2t", torch.zeros(4, dtype=torch.long))],
            draft_id_to_target_id=torch.zeros(4, dtype=torch.long),
        )

        assert model.has_own_lm_head is True
        assert model.lm_head_needs_target_rows is True

    def test_full_vocabulary_checkpoint_still_shares_the_target_head(self) -> None:
        model = self._load(
            [("model.embed_tokens.weight", torch.zeros(4, 2))],
            draft_id_to_target_id=None,
        )

        assert model.has_draft_id_mapping is False
        assert model.has_own_lm_head is False
        assert model.lm_head_needs_target_rows is False
