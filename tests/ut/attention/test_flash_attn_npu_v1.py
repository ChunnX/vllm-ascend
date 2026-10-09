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
"""Unit tests for the parallel-drafting flash-attention-npu v4 backend.

These cover the generation gate, selection including the head-size split
against the FIA sink operator, the device-side length handling, and the
once-per-signature metadata cache. The operator call itself needs an NPU and
the wheel, and is not exercised here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.flash_attn_npu_v1 as fa_module
from tests.ut.base import TestBase
from vllm_ascend.attention.flash_attn_npu_v1 import (
    FA_NPU_MAX_HEAD_SIZE,
    FIA_SINK_HEAD_SIZES,
    AscendFlashAttnV4Backend,
    AscendFlashAttnV4Impl,
    AscendFlashAttnV4MetadataBuilder,
    _get_or_compute_inputs,
    _load,
    enabled,
    flash_attn_npu_selected,
)


def _selector_config(**overrides):
    """The subset of AttentionSelectorConfig the predicate reads."""
    fields = {
        "use_non_causal": False,
        "has_sliding_window": False,
        "has_sink": False,
        "head_size": 256,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestGenerationGate(TestBase):
    def test_disabled_when_unset(self):
        with patch.object(fa_module, "_SELECTED", ""):
            self.assertFalse(enabled())

    def test_enabled_for_the_api_this_backend_serves(self):
        with patch.object(fa_module, "_SELECTED", "v4"):
            self.assertTrue(enabled())

    def test_an_unrecognised_value_raises_instead_of_disabling(self):
        """A typo must not look exactly like the flag working.

        Silently returning False would serve the ordinary backend and produce a
        correct run, so nothing would ever reveal that the operator under test
        was never called.
        """
        for value in ("v3", "4", "true", "yes"):
            with patch.object(fa_module, "_SELECTED", value), self.assertRaisesRegex(RuntimeError, "not an API"):
                enabled()


class TestFlashAttnNpuSelection(TestBase):
    def test_disabled_by_default(self):
        with patch.object(fa_module, "_SELECTED", ""):
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True)))

    def test_selects_non_causal_layers(self):
        with patch.object(fa_module, "_SELECTED", "v4"):
            self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True)))
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=False)))

    def test_excludes_what_the_operator_call_cannot_express(self):
        """The call passes causal=False, window_size=(-1, -1) and no sink tensor.

        A sliding-window or learnable-sink layer therefore picks a different
        backend rather than being served wrongly.
        """
        with patch.object(fa_module, "_SELECTED", "v4"):
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, has_sliding_window=True)))
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, has_sink=True)))

    def test_declines_a_head_size_the_forward_refuses(self):
        with patch.object(fa_module, "_SELECTED", "v4"):
            over = FA_NPU_MAX_HEAD_SIZE + 64
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=over)))
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=0)))

    def test_leaves_the_sink_operators_head_sizes_to_the_sink(self):
        """Where both are enabled and both could serve, the sink keeps the layer.

        That is the path with hardware runs behind it, so this backend takes only
        the gap above it.
        """
        head_size = FIA_SINK_HEAD_SIZES[0]
        with patch.object(fa_module, "_SELECTED", "v4"), patch.object(fa_module, "_FIA_SINK_ENABLED", True):
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=head_size)))
        # With the sink disabled nothing else claims it, so the wheel does.
        with patch.object(fa_module, "_SELECTED", "v4"), patch.object(fa_module, "_FIA_SINK_ENABLED", False):
            self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=head_size)))

    def test_head_dim_256_is_the_case_this_backend_exists_for(self):
        """Not served by the sink operator at any setting, and within the forward's range."""
        self.assertNotIn(256, FIA_SINK_HEAD_SIZES)
        with patch.object(fa_module, "_SELECTED", "v4"), patch.object(fa_module, "_FIA_SINK_ENABLED", True):
            self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=256)))

    def test_reads_only_fields_of_the_selector_config(self):
        """`_cached_get_attn_backend` memoizes on the selector config.

        A predicate reaching for the current vllm config would be answered once
        and that answer reused for every later config that hashed the same.
        """
        with (
            patch.object(fa_module, "_SELECTED", "v4"),
            patch("vllm.config.get_current_vllm_config", side_effect=AssertionError("must not be read")),
        ):
            self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True)))


class TestFlashAttnNpuBackendWiring(TestBase):
    def test_backend_names_its_own_builder_and_impl(self):
        self.assertIs(AscendFlashAttnV4Backend.get_impl_cls(), AscendFlashAttnV4Impl)
        self.assertIs(AscendFlashAttnV4Backend.get_builder_cls(), AscendFlashAttnV4MetadataBuilder)

    def test_name_resolves_through_the_attention_backend_enum(self):
        """`Attention.__init__` looks the name up in AttentionBackendEnum.

        A name of this backend's own would raise there, at model load, the first
        time a draft layer selected it -- which no CPU unit test constructing the
        backend directly would reach.
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        self.assertIn(AscendFlashAttnV4Backend.get_name(), AttentionBackendEnum.__members__)

    def test_kv_cache_layout_is_inherited_unchanged(self):
        """The draft shares the target's cache pool, and the wheel wants that layout."""
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        for name in ("get_kv_cache_shape", "get_required_kv_cache_layout"):
            self.assertEqual(
                getattr(AscendFlashAttnV4Backend, name).__func__,
                getattr(AscendAttentionBackend, name).__func__,
            )


class TestFlashAttnNpuBuilder(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock()
        self.vllm_config.speculative_config = SimpleNamespace(parallel_drafting=True, method="dspark")
        self.device = torch.device("cpu")

    def test_refuses_a_model_without_parallel_drafting(self):
        """`use_non_causal` is not exclusively a draft's flag (DiffusionGemma).

        Only a parallel-drafting draft has the uniform query shape the forward
        derives its partition from, so refuse here where the full config is in
        hand rather than producing quietly wrong offsets.
        """
        self.vllm_config.speculative_config = None
        with (
            patch.object(fa_module, "_load"),
            patch.object(AscendFlashAttnV4MetadataBuilder.__bases__[0], "__init__", return_value=None),
            self.assertRaisesRegex(RuntimeError, "without parallel drafting"),
        ):
            AscendFlashAttnV4MetadataBuilder(None, ["layer0"], self.vllm_config, self.device)

    def test_a_missing_wheel_is_reported_at_construction(self):
        """Not on the first forward of a served request."""
        with (
            patch.dict(fa_module._loaded_modules, {}, clear=True),
            patch("importlib.import_module", side_effect=ImportError("no module")),
            self.assertRaisesRegex(RuntimeError, "requires the flash-attn-npu wheel"),
        ):
            _load()

    def test_a_build_for_another_device_is_named_as_such(self):
        """The wheel picks its exported API from the device name at import time.

        Letting that fail later as an AttributeError inside a forward says
        nothing about the cause.
        """
        with (
            patch.dict(fa_module._loaded_modules, {}, clear=True),
            patch("importlib.import_module", return_value=SimpleNamespace()),
            self.assertRaisesRegex(RuntimeError, "does not expose"),
        ):
            _load()


class TestFlashAttnNpuSeqInputs(TestBase):
    def _builder(self):
        builder = AscendFlashAttnV4MetadataBuilder.__new__(AscendFlashAttnV4MetadataBuilder)
        return builder

    def test_keeps_sequence_lengths_on_device_for_a_non_causal_group(self):
        """The host-side lists stay unset; that is the whole point of the backend.

        `seq_lens_list` being None is also what keeps these layers out of the
        per-step graph task update loop.
        """
        builder = self._builder()
        seq_lens = torch.tensor([7, 7, 7], dtype=torch.int32)
        common = SimpleNamespace(causal=False, query_start_loc=torch.tensor([0, 7, 14, 21], dtype=torch.int32))
        qsl, lengths_q, lengths_k, out_seq_lens, block_table = builder._build_fia_seq_inputs(
            common, 3, torch.tensor([0, 7, 14, 21]), seq_lens, None
        )
        self.assertIsNone(lengths_q)
        self.assertIsNone(lengths_k)
        self.assertIs(out_seq_lens, seq_lens)
        torch.testing.assert_close(qsl, common.query_start_loc[:4])

    def test_a_causal_group_keeps_the_ordinary_path(self):
        """A DFlash draft can carry a different flag per KV cache group.

        The call passes causal=False, so a causal group has to fall back -- and
        the forward reads the same field to make the matching choice.
        """
        builder = self._builder()
        sentinel = ("ordinary", None, None, None, None)
        common = SimpleNamespace(causal=True, query_start_loc=torch.tensor([0, 1]))
        with patch.object(
            AscendFlashAttnV4MetadataBuilder.__bases__[0],
            "_build_fia_seq_inputs",
            return_value=sentinel,
        ) as base:
            result = builder._build_fia_seq_inputs(common, 1, torch.tensor([0, 1]), torch.tensor([3]), None)
        self.assertEqual(result, sentinel)
        base.assert_called_once()


class TestFlashAttnNpuMetadataCache(TestBase):
    def test_metadata_is_computed_once_per_forward_signature(self):
        """Later layers in a forward reuse the first layer's tensors.

        Under graph capture that is what keeps every captured address stable, so
        replay reruns one metadata launch per signature rather than one per layer.
        """
        calls = []

        def compute():
            calls.append(1)
            return (torch.tensor([0]), torch.tensor([1]), object())

        context = SimpleNamespace()
        with patch.object(fa_module, "get_forward_context", return_value=context):
            first = _get_or_compute_inputs((1, 2), compute)
            second = _get_or_compute_inputs((1, 2), compute)
            third = _get_or_compute_inputs((1, 3), compute)
        self.assertEqual(len(calls), 2)
        self.assertIs(first, second)
        self.assertIsNot(first, third)
