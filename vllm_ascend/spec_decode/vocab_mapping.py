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
"""Reduced draft vocabularies for the DFlash/DSpark draft families.

A draft may be trained over the top-K target tokens instead of the full
vocabulary. The checkpoint then carries a ``d2t`` offset table -- draft id ``i``
means target id ``i + d2t[i]`` -- and, because training borrows the frozen
target LM head and projects through only the rows the mapping keeps, it usually
carries no ``lm_head`` weights at all.

Serving therefore has to do two things before the draft can propose anything:
check that the mapping is real and usable, and reproduce the row slice training
used. Both run once at load time and are shared by the v1 proposer and the v2
speculator, which reach them from different places but need identical behavior.
"""

import torch
import torch.nn as nn
from vllm.logger import logger
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod


def resolve_lm_head(model: nn.Module) -> nn.Module | None:
    """The LM head of a target model, through the language-model wrapper."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "get_language_model") and hasattr(model.get_language_model(), "lm_head"):
        return model.get_language_model().lm_head
    return None


def settle_reduced_vocab_lm_head(
    draft_model: nn.Module,
    target_model: nn.Module,
    target_vocab_size: int,
) -> None:
    """Make a freshly loaded draft usable when its vocabulary is reduced.

    A no-op for a full-vocabulary draft, which is every draft without a ``d2t``
    table. Otherwise the mapping is checked -- whichever head the draft ends up
    with, every proposed id is read through it -- and the head is filled from the
    kept target rows unless the checkpoint already shipped its own.

    Callers are the draft-model load paths of both model runners, which differ in
    where they hook but not in what has to happen.
    """
    if getattr(draft_model, "draft_id_to_target_id", None) is None:
        return
    kept_target_ids = validate_draft_vocab_mapping(draft_model, target_vocab_size)
    if getattr(draft_model, "has_own_lm_head_weights", False):
        return
    derive_pruned_lm_head(
        draft_model.lm_head,
        resolve_lm_head(target_model),
        kept_target_ids,
        target_vocab_size,
    )


def validate_draft_vocab_mapping(draft_model: nn.Module, target_vocab_size: int) -> torch.Tensor:
    """Check the draft's ``d2t`` table and return the target ids it keeps.

    Draft id ``i`` means target id ``i + d2t[i]``, and every proposed id is read
    through that table -- so the table has to be checked whether the pruned head
    was derived at load time or shipped by the checkpoint. Returns the kept ids
    on CPU, where the derived head also selects its rows.
    """
    d2t = draft_model.draft_id_to_target_id
    d2t_cpu = d2t.cpu().to(torch.long)
    draft_vocab_size = d2t_cpu.shape[0]
    kept_target_ids = torch.arange(draft_vocab_size, dtype=torch.long) + d2t_cpu

    # An unloaded mapping is all zeros, and so is the legitimate mapping that
    # keeps target ids 0..K-1: the offsets alone cannot separate them, which is
    # why the loader records whether the key was present at all.
    has_draft_id_mapping = getattr(draft_model, "has_draft_id_mapping", None)
    mapping_is_missing = (
        not has_draft_id_mapping
        if has_draft_id_mapping is not None
        # No loader flag to consult. Fall back to the contents and reject all
        # zeros: an unloaded buffer proposes wrong tokens silently, whereas "the
        # K most frequent tokens are exactly ids 0..K-1" is not a vocabulary any
        # real tokenizer produces.
        else not bool(d2t_cpu.any())
    )
    if mapping_is_missing:
        raise ValueError(
            f"The draft config declares draft_vocab_size={draft_vocab_size} "
            f"(target vocabulary {target_vocab_size}) but its checkpoint carries "
            "no d2t mapping; the draft vocabulary would silently be read as the "
            "first draft_vocab_size target ids. Export the draft from a run "
            "trained with draft_vocab_size."
        )
    if int(kept_target_ids.min()) < 0 or int(kept_target_ids.max()) >= target_vocab_size:
        raise ValueError(
            "The draft's d2t mapping points outside the target vocabulary "
            f"[0, {target_vocab_size}); the checkpoint was trained against a "
            "different target model."
        )
    if draft_vocab_size > 1 and not bool(torch.all(kept_target_ids[1:] > kept_target_ids[:-1])):
        # Row selection by a t2d mask yields ascending, distinct target ids.
        # Anything else means the offsets and the head rows disagree.
        raise ValueError(
            "The draft's d2t mapping is not strictly increasing, so it does not "
            "describe an ascending selection of target vocabulary rows."
        )
    return kept_target_ids


def derive_pruned_lm_head(
    draft_lm_head: nn.Module,
    target_lm_head: nn.Module | None,
    kept_target_ids: torch.Tensor,
    target_vocab_size: int,
) -> None:
    """Fill a pruned draft LM head with the target rows its mapping keeps.

    Training borrows the frozen target head and projects through only the kept
    rows, in ascending target-id order, so reproducing that row slice here is
    what makes serving agree with training.

    Under TP the target head is vocab-sharded, so the rows a rank needs mostly
    live elsewhere. The pruned head is assembled in full on every rank with a
    single all-reduce over the group that shards the *target* head --
    ``draft_vocab_size`` rows, not the target vocabulary -- and is then handed to
    the draft head's own ``weight_loader``, which selects that head's shard
    locally and issues no collective of its own. The two heads therefore need not
    share a group: with ``draft_tensor_parallel_size=1`` against a TP target,
    every target rank reconstructs the same full head and its rank-local draft
    head loads all of it.
    """
    if target_lm_head is None:
        raise ValueError(
            "The draft model uses a reduced vocabulary (d2t) and ships no lm_head "
            "of its own, but the target model exposes no lm_head to derive it from."
        )

    # Quantization is decided by the head's quant method, not by its weight
    # dtype: an FP8 head is floating point yet its rows are meaningless without
    # the accompanying scales, so a dtype test would let it through and produce a
    # numerically wrong head that still has a legal shape.
    target_weight = getattr(target_lm_head, "weight", None)
    target_quant_method = getattr(target_lm_head, "quant_method", None)
    if target_weight is None or not isinstance(target_quant_method, UnquantizedEmbeddingMethod):
        raise ValueError(
            "Deriving a pruned draft lm_head requires an unquantized target "
            f"lm_head; got {type(target_lm_head).__name__} with quant method "
            f"{type(target_quant_method).__name__}. Export the draft checkpoint "
            "with its own lm_head instead."
        )

    # The group that shards the target head, which is the only group whose ranks
    # jointly hold every kept row. The global TP group is not a fallback: the
    # draft model is loaded inside patch_tensor_parallel_group whenever
    # draft_tensor_parallel_size differs from the target's, so the global group
    # may currently be the draft's rank-local one.
    target_group = getattr(target_lm_head, "comm_group", None)
    tp_size = getattr(target_lm_head, "tp_size", 1)
    if tp_size > 1 and target_group is None:
        raise NotImplementedError(
            f"The target lm_head is sharded {tp_size} ways but exposes no "
            "comm_group, so the group holding its rows cannot be identified. "
            "Export the draft checkpoint with its own lm_head instead."
        )

    draft_vocab_size = kept_target_ids.shape[0]
    target_vocab_start = target_lm_head.shard_indices.org_vocab_start_index
    target_vocab_end = target_lm_head.shard_indices.org_vocab_end_index

    # kept_target_ids is held on CPU so that selecting this rank's share costs no
    # device synchronization and needs no boolean row indexing, which is not
    # portable on NPU.
    local_draft_rows = torch.nonzero(
        (kept_target_ids >= target_vocab_start) & (kept_target_ids < target_vocab_end),
        as_tuple=True,
    )[0]

    pruned_weight = torch.zeros(
        (draft_vocab_size, target_weight.shape[1]),
        dtype=target_weight.dtype,
        device=target_weight.device,
    )
    if local_draft_rows.numel() > 0:
        local_target_rows = (kept_target_ids[local_draft_rows] - target_vocab_start).to(target_weight.device)
        pruned_weight.index_copy_(
            0,
            local_draft_rows.to(target_weight.device),
            target_weight.index_select(0, local_target_rows),
        )
    if tp_size > 1:
        # Every kept id lives in exactly one rank's shard, so summing the per-rank
        # contributions reconstructs the head without double counting.
        pruned_weight = target_group.all_reduce(pruned_weight)

    draft_lm_head.weight_loader(draft_lm_head.weight, pruned_weight)
    del pruned_weight

    logger.info(
        "[spec_decode/vocab_mapping] Draft uses a reduced vocabulary (%d of %d"
        " target tokens, %.2fx) and ships no lm_head; derived it from the target"
        " lm_head rows kept by d2t.",
        draft_vocab_size,
        target_vocab_size,
        target_vocab_size / draft_vocab_size,
    )
