# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/aclgraph_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import set_graph_params, update_full_graph_params
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.utils import communicator_switch


def _prepare_pcp_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    _block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
    max_query_len: int | None = None,
    *,
    pcp_manager: Any,
) -> cudagraph_utils.AttentionState:
    """Build graph inputs with the same PCP-local layout used on replay."""
    # vLLM #53515 passes PCP-local input buffers into graph capture, so the
    # dummy batch must not be partitioned a second time. vLLM #53869
    # supplies capture-only PCP metadata instead. The block tables must
    # retain the same PCP-local backing that runtime prepare_attn updates,
    # because the SFA full graph cannot rebind their captured pointer.
    # The Ascend dummy carries the seq_lens_np/attn_state views consumed
    # by Ascend metadata builders and doubles as the capture-time PCP
    # global batch (is_dummy=True).
    input_batch = AscendInputBatch.make_dummy(  # type: ignore[call-arg]
        num_reqs, num_tokens, input_buffers, max_query_len=max_query_len
    )
    input_block_tables = pcp_manager.get_dummy_block_tables(num_reqs)
    slot_mappings = pcp_manager.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = cudagraph_utils.build_slot_mappings_by_layer(slot_mappings, kv_cache_config)

    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return cudagraph_utils.AttentionState(attn_metadata, slot_mappings_by_layer)


def collect_sorted_captured_token_sizes(capture_descs: dict) -> list[int]:
    """Collect the actual per-graph token counts that will be captured.

    With speculative decoding under FULL_DECODE_ONLY, each raw
    ``cudagraph_capture_size`` is rounded up to a multiple of
    ``decode_query_len`` (see ``CudaGraphManager._init_candidates``), so the
    real graph sizes differ from ``compilation_config.cudagraph_capture_sizes``.
    The attention backend keys its per-size graph params (events/handles/...)
    by these rounded token counts, so they must be derived from the actual
    capture descriptors, not the raw config sizes.
    """
    return sorted({desc.num_tokens for descs in capture_descs.values() for desc in descs})


def _get_graph_update_backend(
    attn_groups: list[list[AttentionGroup]],
) -> type[AttentionBackend]:
    for groups in attn_groups:
        for group in groups:
            backend = group.backend
            if backend.get_impl_cls() is not None:
                return backend
    raise RuntimeError("No executable attention backend is available for full-graph parameter updates.")


def trimmed_decode_descriptors(
    *,
    decode_mode: CUDAGraphMode,
    width: int,
    max_num_reqs: int,
    max_decode_tokens: int,
    max_capture_size: int,
    lora_capture_cases: list[int],
) -> list[BatchExecutionDescriptor]:
    """Uniform decode descriptors for a trimmed verify width, one per request count.

    A global D-Cut cap trims every request to the same width, so a trimmed step
    is still a uniform decode batch -- just narrower than the verify width the
    base captures. Nothing matches it, so it falls back and the tokens D-Cut
    removed buy nothing.

    The sizes come from the request count, not from ``cudagraph_capture_sizes``.
    Rounding a captured size up to a multiple of the width lands between request
    counts -- width 3 yields 3, 6, 9, 18, 24, so four requests would round 12 up
    to 18 -- and the token and request padding that follows brings back the
    padding-row handling this stage has no need for. One descriptor per request
    count keeps every replay exact.
    """
    descs: list[BatchExecutionDescriptor] = []
    for num_reqs in range(1, max_num_reqs + 1):
        num_tokens = num_reqs * width
        if num_tokens > max_decode_tokens or num_tokens > max_capture_size:
            break
        descs.extend(
            BatchExecutionDescriptor(
                cg_mode=decode_mode,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_token_count=width,
                num_active_loras=num_active_loras,
            )
            for num_active_loras in lora_capture_cases
        )
    return descs


def merge_decode_descriptors(
    candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]],
    capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]],
    descs: list[BatchExecutionDescriptor],
    decode_mode: CUDAGraphMode,
) -> list[BatchExecutionDescriptor]:
    """Make extra descriptors reachable without demoting the ones already there.

    ``candidates`` maps a token count to a priority-ordered list, and a
    descriptor whose uniform token count is None matches any batch, so an entry
    appended after one would never be reached. Each of these goes to the front
    of its own token count instead: it is the only entry there that needs
    neither token nor request padding.

    A token count the base captured nothing at or above has no dispatch entry to
    reach a graph through, so a descriptor there is dropped rather than captured
    into memory nothing can replay. Returns what was actually added.
    """
    added: list[BatchExecutionDescriptor] = []
    for desc in descs:
        key = (desc.num_tokens, desc.num_active_loras)
        reachable = candidates.get(key)
        if reachable is None or desc in reachable:
            continue
        candidates[key] = [desc, *reachable]
        added.append(desc)
    if added:
        captured = capture_descs.setdefault(decode_mode, [])
        captured.extend(added)
        captured.sort(key=lambda d: d.num_tokens, reverse=True)
    return added


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    def __init__(  # type: ignore[misc]
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        model_runner: Any,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
        )
        self.breakable_cg_runner: BreakableACLGraphWrapper | None = None
        self.model_runner = model_runner
        self.update_stream = self.model_runner.update_stream
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        if super().needs_capture():
            set_graph_params(self.capture_sizes)

    def _dcut_trimmed_decode_width(self) -> int | None:
        """The width a global D-Cut cap trims to, when it needs its own graphs."""
        if not envs_ascend.VLLM_ASCEND_DSPARK_DCUT_TRIM_DECODE_GRAPH:
            return None
        # The manual cap only trims when the D-Cut switch is on; without it
        # these graphs would be captured for a width nothing ever presents.
        if not envs_ascend.VLLM_ASCEND_DSPARK_ENABLE_DCUT:
            return None
        cap = envs_ascend.VLLM_ASCEND_DSPARK_DCUT_MANUAL_CAP
        if cap < 0:
            return None
        width = cap + 1
        # The full verify width is what the base already captures, exactly.
        if not 1 <= width < self.decode_query_len:
            return None
        return width

    def _init_candidates(self) -> None:
        """Add the trimmed-width decode graphs on top of the base's own."""
        super()._init_candidates()
        width = self._dcut_trimmed_decode_width()
        decode_mode = self.cudagraph_mode.decode_mode()
        if width is None or self.varlen_decode or not (self.cudagraph_mode.separate_routine() and decode_mode):
            return
        added = merge_decode_descriptors(
            self._candidates,
            self._capture_descs,
            trimmed_decode_descriptors(
                decode_mode=decode_mode,
                width=width,
                max_num_reqs=self.max_num_reqs,
                max_decode_tokens=self.max_num_reqs * self.decode_query_len,
                max_capture_size=self.compilation_config.max_cudagraph_capture_size,
                lora_capture_cases=self.lora_capture_cases,
            ),
            decode_mode,
        )
        logger.info(
            "[D-Cut] added %d decode graphs at trimmed verify width %d (full width %d)",
            len(added),
            width,
            self.decode_query_len,
        )

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableACLGraphWrapper(model, self.vllm_config)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        logger.info_once("run_fullgraph with num_tokens=%s", num_tokens)
        assert self.update_stream is not None
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)

        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.model_runner.dp_size], num_tokens)
        # sfa_v1.py:AscendSFABackend.get_impl_cls reaches
        # sfa_cp.py:resolve_sfa_impl, whose SFA CP selector reads the current
        # ModelConfig. Publish the target config because set_forward_context()
        # does not update it.
        # TODO: Remove this explicit current-config scope once ACL graph replay
        # passes VllmConfig directly through the graph-update interfaces.
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                self.model_runner.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ),
        ):
            forward_context = get_forward_context()
            attn_backend = _get_graph_update_backend(self.model_runner.attn_groups)
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
            )
        return ret

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        # vLLM #53869 supplies PCP slot mappings during graph capture.
        pcp_manager: Any = None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        model = ModelWithContext(model)
        pcp_manager = getattr(self.model_runner, "pcp_manager", None)
        if pcp_manager is not None:
            cudagraph_utils.prepare_inputs_to_capture = partial(
                _prepare_pcp_inputs_to_capture,
                pcp_manager=pcp_manager,
            )
        with communicator_switch():
            # vLLM #53869 added pcp_manager to ModelCudaGraphManager.capture on
            # main; v0.28.0 still uses the older signature without that kwarg.
            if not vllm_version_is("0.28.0"):
                return super().capture(
                    model,
                    model_state,
                    input_buffers,
                    intermediate_tensors,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    pcp_manager=pcp_manager,
                    has_lora=has_lora,
                    use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                    lora_capture_hook=lora_capture_hook,
                    progress_bar_desc=progress_bar_desc,
                )
            return super().capture(
                model,
                model_state,
                input_buffers,
                intermediate_tensors,
                block_tables,
                attn_groups,
                kv_cache_config,
                has_lora=has_lora,
                use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                lora_capture_hook=lora_capture_hook,
                progress_bar_desc=progress_bar_desc,
            )


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context.
    so we can inherit vllm's CudaGraphManager._capture_full_graph.
    """

    def __init__(self, original_model, is_draft_model=False, is_draft_model_prefill=False):
        super().__init__()
        self.original_model = original_model
        self.is_draft_model = is_draft_model
        self.is_draft_model_prefill = is_draft_model_prefill

    def forward(self, *args, **kwargs):
        forward_context = get_forward_context()
        # In warmup phase, capturing=False by default.
        # when capturing, we need to set capturing=True in forward context.
        _EXTRA_CTX.capturing = (
            torch.npu.is_current_stream_capturing()
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        )
        if self.is_draft_model:
            _EXTRA_CTX.is_draft_model = True
        if self.is_draft_model_prefill:
            _EXTRA_CTX.is_draft_model_prefill = True

        return self.original_model(*args, **kwargs)

    def get_original_model(self):
        return self.original_model

    def compute_logits(self, hidden_states: torch.Tensor):
        # draft model has `compute_logits`, which is not in ModelWithContext
        return self.original_model.compute_logits(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor):
        return self.original_model.compute_draft_logits(hidden_states)

    def markov_embed(self, token_ids: torch.Tensor):
        return self.original_model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor):
        return self.original_model.markov_bias(markov_embed)

    def map_draft_to_target(self, draft_ids: torch.Tensor):
        return self.original_model.map_draft_to_target(draft_ids)

    def embed_input_ids(self, *args, **kwargs):
        return self.original_model.embed_input_ids(*args, **kwargs)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor):
        return self.original_model.compute_confidence(head_hidden, markov_embed)


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
