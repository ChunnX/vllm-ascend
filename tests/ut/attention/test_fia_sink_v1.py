#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Unit tests for the parallel-drafting FIA sink attention backend.

These cover selection, the device-side length construction and the once-per-
forward metadata cache. The operator call itself needs an NPU and is not
exercised here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.fia_sink_v1 as sink_module
from tests.ut.base import TestBase
from vllm_ascend.attention.fia_sink_v1 import (
    AscendFIASinkBackend,
    AscendFIASinkImpl,
    AscendFIASinkMetadataBuilder,
    _build_fia_sink_seq_tensors,
    _ensure_fia_sink_ops_registered,
    _get_or_compute_fia_sink_inputs,
    fia_sink_selected,
)


def _selector_config(**overrides):
    """The subset of AttentionSelectorConfig the predicate reads."""
    fields = {
        "use_non_causal": False,
        "has_sliding_window": False,
        "has_sink": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestFIASinkSelection(TestBase):
    def test_disabled_by_default(self):
        """The operator is opt-in, so nothing routes here without the env var."""
        self.assertFalse(fia_sink_selected(_selector_config(use_non_causal=True)))

    def test_selects_non_causal_layers(self):
        with patch.object(sink_module, "_FIA_SINK_ENABLED", True):
            self.assertTrue(fia_sink_selected(_selector_config(use_non_causal=True)))
            # The target model is causal, so it never reaches the sink backend.
            self.assertFalse(fia_sink_selected(_selector_config(use_non_causal=False)))

    def test_excludes_what_the_operator_call_cannot_express(self):
        """sparse_mode is hardcoded to 0 and neither mask nor sink is forwarded.

        Excluding these at selection is what replaced the old runtime raise: a
        sliding-window layer now simply picks a different backend instead of
        turning the whole process into a hard failure.
        """
        with patch.object(sink_module, "_FIA_SINK_ENABLED", True):
            self.assertFalse(
                fia_sink_selected(_selector_config(use_non_causal=True, has_sliding_window=True))
            )
            self.assertFalse(fia_sink_selected(_selector_config(use_non_causal=True, has_sink=True)))

    def test_reads_only_fields_of_the_selector_config(self):
        """The result must not depend on anything outside the memoized key.

        ``_cached_get_attn_backend`` keys on the selector config, so a predicate
        that consulted the current vllm config would be answered once and reused
        for every later config that hashed the same.
        """
        with patch.object(sink_module, "_FIA_SINK_ENABLED", True):
            with patch("vllm.config.get_current_vllm_config", side_effect=AssertionError("must not be read")):
                self.assertTrue(fia_sink_selected(_selector_config(use_non_causal=True)))


class TestFIASinkBackendWiring(TestBase):
    def test_backend_names_its_own_builder_and_impl(self):
        self.assertEqual(AscendFIASinkBackend.get_name(), "ASCEND_FIA_SINK")
        self.assertIs(AscendFIASinkBackend.get_builder_cls(), AscendFIASinkMetadataBuilder)
        self.assertIs(AscendFIASinkBackend.get_impl_cls(), AscendFIASinkImpl)

    def test_kv_cache_layout_is_inherited_unchanged(self):
        """The draft shares the target's cache pool, so the layout must match.

        ``get_required_kv_cache_layout`` is applied through a process-global
        setter, so a layout of its own here would not stay on this backend.
        """
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        self.assertIs(
            AscendFIASinkBackend.get_required_kv_cache_layout.__func__,
            AscendAttentionBackend.get_required_kv_cache_layout.__func__,
        )
        self.assertEqual(
            AscendFIASinkBackend.get_kv_cache_shape(2, 4, 8, 16),
            AscendAttentionBackend.get_kv_cache_shape(2, 4, 8, 16),
        )


class TestFIASinkMetadataBuilder(TestBase):
    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.speculative_config = SimpleNamespace(
            method="dspark",
            parallel_drafting=True,
            num_speculative_tokens=7,
        )
        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.mock_vllm_config.model_config.max_model_len = 640
        self.mock_vllm_config.model_config.hf_text_config.sliding_window = None
        self.mock_vllm_config.cache_config.block_size = 64
        self.mock_vllm_config.compilation_config.cudagraph_mode = None
        self.mock_vllm_config.scheduler_config.max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        self.mock_device = "cpu:0"
        torch.Tensor.pin_memory = lambda x: x  # noqa

    def _build(self, layer_names=None):
        with patch.object(sink_module, "_ensure_fia_sink_ops_registered") as ensure_ops:
            builder = AscendFIASinkMetadataBuilder(
                None,
                layer_names or ["layer-with-operator-defined-shape"],
                self.mock_vllm_config,
                self.mock_device,
            )
        return builder, ensure_ops

    def test_operator_package_is_checked_at_construction(self):
        """Missing omni_custom_ops should stop startup, not the first request."""
        _, ensure_ops = self._build()

        ensure_ops.assert_called_once_with()

    def test_head_topology_is_left_to_the_operator(self):
        """No head-dim or head-count gate here; the operator owns its domain."""
        builder, _ = self._build()

        self.assertIsInstance(builder, AscendFIASinkMetadataBuilder)

    def test_refuses_a_model_without_parallel_drafting(self):
        """use_non_causal is not exclusive to drafts -- DiffusionGemma sets it.

        Selection can only see the selector config, so the check that this really
        is a parallel-drafting draft happens here, where the full config is in
        hand.
        """
        self.mock_vllm_config.speculative_config = None

        with patch.object(sink_module, "_ensure_fia_sink_ops_registered"):
            with self.assertRaisesRegex(RuntimeError, "without parallel drafting"):
                AscendFIASinkMetadataBuilder(None, ["layer0"], self.mock_vllm_config, self.mock_device)

    def test_causal_group_keeps_the_ordinary_path(self):
        """A DFlash draft can carry a different causal flag per KV cache group.

        The sink call hardcodes sparse_mode=0, so a causal group must not take
        it. Backend selection is per layer and cannot see this -- causality is
        per build -- so the fallback lives here.
        """
        builder, _ = self._build()
        query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
        seq_lens = torch.tensor([19, 23], dtype=torch.int32)
        common_attn_metadata = SimpleNamespace(query_start_loc=query_start_loc, causal=True)

        _, actual_seq_lengths_q, seq_lens_list, _, _ = builder._build_fia_seq_inputs(
            common_attn_metadata,
            num_reqs=2,
            query_start_loc_cpu=query_start_loc.clone(),
            seq_lens=seq_lens,
            block_table=torch.zeros((2, 4), dtype=torch.int32),
        )

        self.assertEqual(actual_seq_lengths_q, [4, 8])
        self.assertEqual(seq_lens_list, [19, 23])

    def test_keeps_sequence_lengths_on_device(self):
        """The whole point: no .tolist(), and no host-side lists downstream."""
        builder, _ = self._build()
        query_start_loc = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
        seq_lens = torch.tensor([19, 23], dtype=torch.int32)
        block_table = torch.zeros((2, 4), dtype=torch.int32)
        common_attn_metadata = SimpleNamespace(query_start_loc=query_start_loc, causal=False)

        (
            out_query_start_loc,
            actual_seq_lengths_q,
            seq_lens_list,
            out_seq_lens,
            out_block_table,
        ) = builder._build_fia_seq_inputs(
            common_attn_metadata,
            num_reqs=2,
            query_start_loc_cpu=query_start_loc.clone(),
            seq_lens=seq_lens,
            block_table=block_table,
        )

        self.assertTrue(torch.equal(out_query_start_loc, query_start_loc[:3]))
        # seq_lens_list=None is also what keeps these layers out of the per-step
        # graph_task_update loop in update_graph_params.
        self.assertIsNone(actual_seq_lengths_q)
        self.assertIsNone(seq_lens_list)
        self.assertIs(out_seq_lens, seq_lens)
        self.assertIs(out_block_table, block_table)


class TestFIASinkSeqTensors(TestBase):
    def test_builds_legal_full_graph_padding_lengths(self):
        seq_lens = torch.tensor([19, 23, 0, 0], dtype=torch.int32)

        actual_seq_qlen, actual_seq_kvlen = _build_fia_sink_seq_tensors(
            num_tokens=32,
            seq_lens=seq_lens,
        )

        self.assertTrue(
            torch.equal(
                actual_seq_qlen,
                torch.tensor([8, 16, 24, 32], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                actual_seq_kvlen,
                torch.tensor([19, 23, 1, 1], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                seq_lens,
                torch.tensor([19, 23, 0, 0], dtype=torch.int32),
            )
        )
        self.assertEqual(actual_seq_qlen[-1].item(), 32)

    def test_rejects_non_uniform_query_batch(self):
        with self.assertRaisesRegex(RuntimeError, "uniform query batch"):
            _build_fia_sink_seq_tensors(
                num_tokens=31,
                seq_lens=torch.tensor([4, 5, 6, 7], dtype=torch.int32),
            )


class TestFIASinkForwardCache(TestBase):
    def test_metadata_is_computed_once_per_forward_signature(self):
        forward_context = SimpleNamespace()
        expected = (
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64),
            torch.empty(1024, dtype=torch.int32),
        )
        compute = MagicMock(return_value=expected)

        with patch.object(sink_module, "get_forward_context", return_value=forward_context):
            first = _get_or_compute_fia_sink_inputs((1, 2, 3), compute)
            second = _get_or_compute_fia_sink_inputs((1, 2, 3), compute)

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        compute.assert_called_once_with()

    def test_dependency_failure_is_reported_at_initialization(self):
        with (
            patch.object(sink_module, "_fia_sink_ops_registered", False),
            patch.object(sink_module.importlib, "import_module", side_effect=ImportError("missing")),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires the omni_custom_ops wheel"):
                _ensure_fia_sink_ops_registered()


class TestFIASinkImpl(TestBase):
    def test_non_causal_build_takes_the_sink_path(self):
        """No capture-vs-replay split: the op is captured inline either way."""
        impl = AscendFIASinkImpl.__new__(AscendFIASinkImpl)
        sentinel = object()
        impl._forward_fia_sink = MagicMock(return_value=sentinel)
        metadata = SimpleNamespace(causal=False)

        result = impl.forward_fused_infer_attention("q", "k", "v", metadata, "out", "kv")

        self.assertIs(result, sentinel)
        impl._forward_fia_sink.assert_called_once_with("q", "k", "v", metadata, "out", "kv")

    def test_causal_build_falls_back_to_the_ordinary_path(self):
        """Mirrors the builder: a causal KV group of a draft is not sink-able."""
        impl = AscendFIASinkImpl.__new__(AscendFIASinkImpl)
        impl._forward_fia_sink = MagicMock()
        metadata = SimpleNamespace(causal=True)
        sentinel = object()

        with patch.object(
            AscendFIASinkImpl.__mro__[1],
            "forward_fused_infer_attention",
            return_value=sentinel,
        ) as base_forward:
            result = impl.forward_fused_infer_attention("q", "k", "v", metadata, "out", "kv")

        self.assertIs(result, sentinel)
        impl._forward_fia_sink.assert_not_called()
        base_forward.assert_called_once()
