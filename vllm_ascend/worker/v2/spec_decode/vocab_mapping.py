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
"""Reduced-vocabulary (``d2t``) support for DFlash/DSpark drafts.

A draft trained with ``draft_vocab_size`` proposes over a subset of the target
vocabulary: draft id ``i`` means target id ``i + d2t[i]``. Training borrows the
frozen target lm_head and projects through only the kept rows, in ascending
target-id order, so the served checkpoint carries ``d2t`` but no ``lm_head`` of
its own -- and serving has to reproduce that same row slice.

Upstream vLLM already handles the sampling side (``compute_draft_logits`` /
``map_draft_to_target`` / the d2t scatter in the speculator). What it does not
handle is the head: ``load_dspark_model`` aliases the *full* target lm_head
whenever the checkpoint ships none, which for a pruned draft yields target-vocab
logits against a ``draft_vocab_size`` Markov bias.
"""

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod


def validate_draft_vocab_mapping(draft_model: nn.Module, target_vocab_size: int) -> torch.Tensor:
    """Check the draft's ``d2t`` table and return the target ids it keeps.

    Every proposed id is read through this table, so it is checked whether the
    pruned head was derived here or shipped by the checkpoint. Returns the kept
    ids on CPU, where the derived head also selects its rows.
    """
    d2t = draft_model.draft_id_to_target_id.cpu().to(torch.long)
    draft_vocab_size = d2t.shape[0]
    kept_target_ids = torch.arange(draft_vocab_size, dtype=torch.long) + d2t

    # An unloaded mapping is all zeros, and so is the legitimate mapping that
    # keeps target ids 0..K-1: the offsets alone cannot separate them, which is
    # why the loader records whether the key was present at all.
    has_draft_id_mapping = getattr(draft_model, "has_draft_id_mapping", None)
    mapping_is_missing = (
        not has_draft_id_mapping
        if has_draft_id_mapping is not None
        # No loader flag to consult. Fall back to the contents and reject all
        # zeros: an unloaded buffer proposes wrong tokens silently, whereas "the
        # K kept tokens are exactly ids 0..K-1" is not a vocabulary any real
        # tokenizer produces.
        else not bool(d2t.any())
    )
    if mapping_is_missing:
        raise ValueError(
            f"The draft config declares draft_vocab_size={draft_vocab_size} (target vocabulary "
            f"{target_vocab_size}) but its checkpoint carries no d2t mapping; the draft vocabulary "
            "would silently be read as the first draft_vocab_size target ids. Export the draft from "
            "a run trained with draft_vocab_size."
        )
    if int(kept_target_ids.min()) < 0 or int(kept_target_ids.max()) >= target_vocab_size:
        raise ValueError(
            f"The draft's d2t mapping points outside the target vocabulary [0, {target_vocab_size}); "
            "the checkpoint was trained against a different target model."
        )
    if draft_vocab_size > 1 and not bool(torch.all(kept_target_ids[1:] > kept_target_ids[:-1])):
        # Row selection by a t2d mask yields ascending, distinct target ids.
        # Anything else means the offsets and the head rows disagree.
        raise ValueError(
            "The draft's d2t mapping is not strictly increasing, so it does not describe an ascending "
            "selection of target vocabulary rows."
        )
    return kept_target_ids


def build_pruned_lm_head_from_target(
    draft_model: nn.Module,
    target_lm_head: nn.Module | None,
    kept_target_ids: torch.Tensor,
) -> None:
    """Fill the draft's pruned lm_head with the target rows its ``d2t`` keeps.

    Under TP the target head is vocab-sharded, so most rows a rank needs live
    elsewhere. The pruned head is assembled in full on every rank with a single
    all-reduce -- ``draft_vocab_size`` rows, not the target vocabulary -- and is
    then handed to the draft head's own ``weight_loader``, which selects that
    head's shard locally and issues no collective of its own.
    """
    draft_lm_head = draft_model.lm_head
    if target_lm_head is None:
        raise ValueError(
            "The draft model uses a reduced vocabulary (d2t) and ships no lm_head of its own, but the "
            "target model exposes no lm_head to derive it from."
        )

    # Quantization is decided by the head's quant method, not by its weight
    # dtype: an FP8 head is floating point yet its rows are meaningless without
    # the accompanying scales, so a dtype test would let it through and produce a
    # numerically wrong head that still has a legal shape.
    target_weight = getattr(target_lm_head, "weight", None)
    target_quant_method = getattr(target_lm_head, "quant_method", None)
    if target_weight is None or not isinstance(target_quant_method, UnquantizedEmbeddingMethod):
        raise ValueError(
            "Deriving a pruned draft lm_head requires an unquantized target lm_head; got "
            f"{type(target_lm_head).__name__} with quant method {type(target_quant_method).__name__}. "
            "Export the draft checkpoint with its own lm_head instead."
        )

    tp_size = getattr(target_lm_head, "tp_size", 1)
    target_group = get_tp_group() if tp_size > 1 else None
    if target_group is not None and target_group.world_size != tp_size:
        # The draft is loaded under the target's TP group in model runner v2, so
        # the two must agree. If they ever stop agreeing, the group holding the
        # target rows is no longer identifiable from here.
        raise NotImplementedError(
            f"The target lm_head is sharded {tp_size} ways but the current tensor-parallel group has "
            f"{target_group.world_size} ranks, so the group holding its rows cannot be identified. "
            "Export the draft checkpoint with its own lm_head instead."
        )

    # Kept ids stay on CPU so that selecting this rank's share costs no device
    # synchronization and needs no boolean row indexing, which is not portable on NPU.
    target_vocab_start = target_lm_head.shard_indices.org_vocab_start_index
    target_vocab_end = target_lm_head.shard_indices.org_vocab_end_index
    local_draft_rows = torch.nonzero(
        (kept_target_ids >= target_vocab_start) & (kept_target_ids < target_vocab_end),
        as_tuple=True,
    )[0]

    pruned_weight = torch.zeros(
        (kept_target_ids.shape[0], target_weight.shape[1]),
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
    if target_group is not None:
        # Every kept id lives in exactly one rank's shard, so summing the
        # per-rank contributions reconstructs the head without double counting.
        pruned_weight = target_group.all_reduce(pruned_weight)

    draft_lm_head.weight_loader(draft_lm_head.weight, pruned_weight)
    del pruned_weight

    logger.info(
        "[spec_decode] Draft uses a reduced vocabulary (%d target tokens) and ships no lm_head;"
        " derived it from the target lm_head rows kept by d2t.",
        kept_target_ids.shape[0],
    )
