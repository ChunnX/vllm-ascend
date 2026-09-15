#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.spec_decode.vocab_mapping import settle_reduced_vocab_lm_head
from vllm_ascend.utils import (
    get_rotation_path,
    vllm_version_is,
)
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import prepare_replicated_pcp_config

logger = init_logger(__name__)


class AscendDSparkSpeculator(DSparkSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        vllm_config, self.replicated_pcp = prepare_replicated_pcp_config(vllm_config)
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None
        # Record-only adaptive-verification observation (does not trim). When on,
        # propose() flips enable_adaptive_verification True *only* around the
        # draft sampling so the upstream confidence-head path fills
        # draft_token_confidence_probs, while the model runner still sees it False
        # at init and never builds the (unadapted) trimming manager.
        self._av_observe = bool(envs_ascend.VLLM_ASCEND_DSPARK_AV_OBSERVE)
        self._av_shadow = bool(envs_ascend.VLLM_ASCEND_DSPARK_AV_SHADOW)
        # Both record-only observation and the shadow trim-decision need the
        # confidence head live (they only read draft_token_confidence_probs,
        # never trim); this single flag gates keeping it computed under capture.
        self._av_wants_confidence = self._av_observe or self._av_shadow
        # Per-confidence-row persistent slot + prefill flag from the propose that
        # last wrote draft_token_confidence_probs, so the observer realigns the
        # next verify step by request identity (see _snapshot_av_confidence_alignment).
        self._av_conf_slots: np.ndarray | None = None
        self._av_conf_prefill: np.ndarray | None = None
        if self._av_wants_confidence:
            # Seed a clean value so the first verify step reads zeros, not
            # uninitialized memory, before any propose() has run.
            self.draft_token_confidence_probs.zero_()

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        """Load the draft, then settle its LM head when the vocabulary is reduced.

        Upstream shares the target's LM head with any draft that did not ship one
        of its own, which is right for a full-vocabulary draft and wrong for a
        pruned one: the shared head spans the target vocabulary while the draft's
        logits processor is ``draft_vocab_size`` wide, so it would slice the first
        K columns instead of the K the mapping keeps -- plausible tokens, wrong
        ones, and nothing raises. The draft model blocks that share for a reduced
        vocabulary; what is left is to check the mapping and fill the head that
        the checkpoint left empty.

        The sequential Markov sampling itself already reads the mapping upstream,
        so nothing on the hot path changes here.
        """
        # Upstream replaces quant_config with None for a BF16 draft. Pass only
        # the target QuaRot path so the draft's existing load_weights can fold
        # input inverse rotation into FC (W @ R) and align fallback embedding /
        # lm_head before upstream decides weight sharing. Do not rotate again
        # after loading or replace the draft's own quantization configuration.
        draft_hf_config = self.draft_model_config.hf_config
        rotation_path = get_rotation_path(self.vllm_config)
        draft_hf_config._ascend_target_rotation_path = str(rotation_path) if rotation_path is not None else None
        model = super().load_draft_model(target_model, target_attn_layer_names)
        if hasattr(model, "configure_target_aux_hidden_capture"):
            model.configure_target_aux_hidden_capture(target_model)
        settle_reduced_vocab_lm_head(model, target_model, self.vllm_config.model_config.get_vocab_size())

        # Observation needs the confidence head; the trimming path would raise at
        # load, but record-only leaves enable_adaptive_verification False, so
        # check here and degrade gracefully instead of asserting mid-forward.
        if self._av_wants_confidence and getattr(model.model, "confidence_head", None) is None:
            logger.warning(
                "VLLM_ASCEND_DSPARK_AV_OBSERVE/_SHADOW is set but this DSpark "
                "checkpoint has no confidence head (enable_confidence_head); "
                "disabling AV observation and shadow."
            )
            self._av_observe = False
            self._av_shadow = False
            self._av_wants_confidence = False
        return model

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    def capture(self) -> None:
        """Bake the confidence-head branch into the captured FULL draft graph.

        Upstream ``_sample_sequential`` guards the confidence computation with a
        Python ``if self.enable_adaptive_verification``, and ``capture()`` traces
        ``_generate_draft`` *directly* (upstream ``DFlashSpeculator.capture``
        bypasses ``propose``). So when the flag is False at trace time the
        confidence op is never recorded into the graph: every FULL replay then
        skips it and ``draft_token_confidence_probs`` stays frozen at its
        pre-capture value -- the bit-identical per-position confidence seen under
        FULL_DECODE_ONLY, while eager (no captured graph, ``propose`` flips the
        flag) produces a live signal.

        The model runner has already read this flag (via
        ``maybe_create_adaptive_verification_manager``) at setup and built no
        trimming manager, so turning it on for capture only bakes the confidence
        branch into the FULL graph -- recomputed on every replay -- without
        enabling the (not-yet-ported) trimming path. Left True afterwards is
        harmless: every trimming code path is gated on the manager, which is
        None. Only meaningful when observing/shadowing; otherwise capture unchanged.
        """
        if self._av_wants_confidence:
            self.enable_adaptive_verification = True
        super().capture()

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        # Initialize the draft attention backend with its PCP=1 config.
        with set_current_vllm_config(self.attn_vllm_config):
            super().set_attn(
                model_state,
                kv_cache_config,
                block_tables,
                target_input_buffers,
                target_attn_groups,
            )
            self._context_slot_mappings = self._context_slot_mappings.to(torch.int32)  # type: ignore[has-type]
            # npu needs attn_backends to update full graph params in run_fullgraph.
            attn_backends: dict[str, type[AttentionBackend]] = {}
            active_layer_names = self.draft_attn_layer_names
            for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
                layer_names = kv_cache_group_spec.layer_names
                if active_layer_names is not None:
                    # Preserve cache-group order so captured graph tasks and
                    # runtime metadata stay aligned.
                    layer_names = [name for name in layer_names if name in active_layer_names]

                layer_type = cast(type[Any], AttentionLayerBase)
                attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

                for layer_name in layer_names:
                    attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

            self.attn_backends = attn_backends

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        assert self.input_batch is not None
        # The draft attention metadata is built through the generic
        # (Ascend) build_attn_metadata path; the factory forwards the draft
        # query positions that the DSA metadata builder needs for RoPE.
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                torch.from_numpy(self.input_batch.is_prefilling_np),
            ),
        ):
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        return [self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        DSpark inherits DFlash's full-graph path, and upstream
        ``Speculator._build_draft_attn_metadata`` clamps ``query_start_loc`` at
        the real ``num_reqs`` to keep the cumulative series non-decreasing, so
        when a batch is padded to a capture size (``num_reqs_padded >
        num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list
        return attn_metadata

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        # vLLM #53694 replaced num_tokens_across_dp with the DP sync state.
        dp_sync: Any = None,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        assert self.input_batch is not None
        sync_state = num_tokens_across_dp if vllm_version_is("0.28.0") else dp_sync
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
        # Record-only observation: flip the flag on just for draft sampling so
        # the upstream confidence-head path fills draft_token_confidence_probs,
        # then restore it so nothing downstream (the trimming manager) ever sees
        # it enabled.
        observe = self._av_wants_confidence and not self.enable_adaptive_verification
        if observe:
            self.enable_adaptive_verification = True
        try:
            with (
                build_attn_metadata_wrapper(),
                build_draft_attn_metadata_factory(
                    self.input_buffers.positions, self.max_num_tokens, torch.from_numpy(self.input_batch.is_prefilling_np)
                ),
            ):
                result = super().propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    sync_state,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    mm_inputs,
                    is_profile=is_profile,
                )
            # super().propose just wrote draft_token_confidence_probs[:num_reqs] in
            # THIS batch's row order; snapshot which persistent slot each row maps
            # to so the observer can realign the next verify step (which runs in a
            # different batch order) by request identity, not leading row.
            if self._av_wants_confidence and not dummy_run:
                self._snapshot_av_confidence_alignment(input_batch)
            return result
        finally:
            if observe:
                self.enable_adaptive_verification = False

    def _snapshot_av_confidence_alignment(self, input_batch: InputBatch) -> None:
        """Record per-confidence-row persistent slot + prefill flag (record-only).

        ``idx_mapping_np[row]`` is the req_states slot for batch ``row``; snapshot
        it (CPU, no D2H) while it still matches the confidence just written, since
        the InputBatch buffers are reused/overwritten by the next step's
        prepare_inputs. ``is_prefilling_np`` is kept only to help locate NaN
        confidence rows (seen during prefill bursts).
        """
        idx_np = getattr(input_batch, "idx_mapping_np", None)
        num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
        if idx_np is None or num_reqs == 0:
            self._av_conf_slots = None
            self._av_conf_prefill = None
            return
        self._av_conf_slots = np.asarray(idx_np[:num_reqs]).copy()
        pref = getattr(input_batch, "is_prefilling_np", None)
        self._av_conf_prefill = None if pref is None else np.asarray(pref[:num_reqs]).copy()
