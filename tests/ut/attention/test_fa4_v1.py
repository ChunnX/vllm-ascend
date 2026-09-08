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
"""Unit tests for the parallel-drafting flash_attn_npu_4 attention backend.

These cover selection (including how it divides head sizes with the FIA sink
backend), the device-side length construction, the KV bound the paged block
table is addressed with, and the once-per-forward metadata cache. The operator
call itself needs an NPU and the flash-attn-npu wheel, so it is mocked here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.fa4_v1 as fa4_module
from tests.ut.base import TestBase
from vllm_ascend.attention.fa4_v1 import (
    AscendFA4Backend,
    AscendFA4Impl,
    AscendFA4MetadataBuilder,
    _build_fa4_seq_tensors,
    _get_or_compute_fa4_inputs,
    _load_fa4,
    fa4_selected,
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


class TestFA4Selection(TestBase):
    def test_disabled_by_default(self):
        """Opt-in, like the sink backend: nothing routes here without the env var."""
        self.assertFalse(fa4_selected(_selector_config(use_non_causal=True)))

    def test_selects_non_causal_draft_layers(self):
        with patch.object(fa4_module, "_FA4_ENABLED", True):
            self.assertTrue(fa4_selected(_selector_config(use_non_causal=True)))
            # The target model is causal, so it never reaches this backend.
            self.assertFalse(fa4_selected(_selector_config(use_non_causal=False)))

    def test_head_dim_256_is_the_gap_this_backend_exists_for(self):
        """The sink operator serves 128/192/512; v4 covers everything up to 256."""
        with patch.object(fa4_module, "_FA4_ENABLED", True):
            self.assertTrue(fa4_selected(_selector_config(use_non_causal=True, head_size=256)))

    def test_yields_the_sink_operators_head_sizes_back_to_it(self):
        """Where both can serve, the operator with hardware runs behind it wins.

        Only while the sink backend is actually enabled -- with its flag off there
        is no other backend keeping the device-side lengths, so v4 takes the layer.
        """
        with patch.object(fa4_module, "_FA4_ENABLED", True):
            with patch.object(fa4_module, "_FIA_SINK_ENABLED", True):
                for head_size in fa4_module.FIA_SINK_HEAD_SIZES:
                    with self.subTest(head_size=head_size):
                        self.assertFalse(fa4_selected(_selector_config(use_non_causal=True, head_size=head_size)))

            with patch.object(fa4_module, "_FIA_SINK_ENABLED", False):
                self.assertTrue(fa4_selected(_selector_config(use_non_causal=True, head_size=128)))

    def test_refuses_head_sizes_the_forward_rejects(self):
        """mha_fwd checks head_size <= 256, so 512 is not this backend's to take."""
        with patch.object(fa4_module, "_FA4_ENABLED", True):
            with patch.object(fa4_module, "_FIA_SINK_ENABLED", False):
                self.assertFalse(fa4_selected(_selector_config(use_non_causal=True, head_size=512)))
                self.assertFalse(fa4_selected(_selector_config(use_non_causal=True, head_size=0)))

    def test_excludes_what_the_operator_call_cannot_express(self):
        """This call passes causal=False, window_size=(-1, -1) and no sink tensor.

        v4 rejects learnable_sink outright, so those layers pick another backend
        rather than failing on the first forward.
        """
        with patch.object(fa4_module, "_FA4_ENABLED", True):
            self.assertFalse(fa4_selected(_selector_config(use_non_causal=True, has_sliding_window=True)))
            self.assertFalse(fa4_selected(_selector_config(use_non_causal=True, has_sink=True)))

    def test_reads_only_fields_of_the_selector_config(self):
        """_cached_get_attn_backend memoizes on the selector config alone.

        A predicate that consulted get_current_vllm_config() would be answered
        once and reused for every later config that hashed the same.
        """
        with patch.object(fa4_module, "_FA4_ENABLED", True):
            with patch("vllm.config.get_current_vllm_config", side_effect=AssertionError("must not be read")):
                self.assertTrue(fa4_selected(_selector_config(use_non_causal=True)))


class TestFA4BackendWiring(TestBase):
    def test_backend_names_its_own_builder_and_impl(self):
        self.assertEqual(AscendFA4Backend.get_name(), "ASCEND_FA4")
        self.assertIs(AscendFA4Backend.get_impl_cls(), AscendFA4Impl)
        self.assertIs(AscendFA4Backend.get_builder_cls(), AscendFA4MetadataBuilder)

    def test_kv_cache_layout_is_inherited_unchanged(self):
        """The draft shares the target's cache pool, so the layout must match.

        ``get_required_kv_cache_layout`` is applied through a process-global
        setter, so a layout of its own here would not stay on this backend.
        """
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        self.assertIs(
            AscendFA4Backend.get_required_kv_cache_layout.__func__,
            AscendAttentionBackend.get_required_kv_cache_layout.__func__,
        )
        self.assertEqual(
            AscendFA4Backend.get_kv_cache_shape(2, 4, 8, 16),
            AscendAttentionBackend.get_kv_cache_shape(2, 4, 8, 16),
        )


class TestFA4MetadataBuilder(TestBase):
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
        with patch.object(fa4_module, "_load_fa4") as load:
            builder = AscendFA4MetadataBuilder(
                None,
                layer_names or ["model.layers.0.self_attn.attn"],
                self.mock_vllm_config,
                self.mock_device,
            )
        return builder, load

    def test_wheel_is_checked_at_construction(self):
        """A missing flash-attn-npu should stop startup, not the first request."""
        _, load = self._build()

        load.assert_called_once_with()

    def test_logs_that_a_layer_actually_reached_this_backend(self):
        with patch.object(fa4_module.logger, "info") as mock_info:
            self._build(["model.layers.3.self_attn.attn"])

        mock_info.assert_called_once()
        self.assertIn("model.layers.3.self_attn.attn", str(mock_info.call_args.args))

    def test_refuses_a_model_without_parallel_drafting(self):
        """use_non_causal is not exclusive to drafts -- DiffusionGemma sets it."""
        self.mock_vllm_config.speculative_config = None

        with patch.object(fa4_module, "_load_fa4"):
            with self.assertRaisesRegex(RuntimeError, "without parallel drafting"):
                AscendFA4MetadataBuilder(None, ["layer0"], self.mock_vllm_config, self.mock_device)

    def test_causal_group_keeps_the_ordinary_path(self):
        """A DFlash draft can carry a different causal flag per KV cache group."""
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


class TestFA4SeqTensors(TestBase):
    def test_builds_legal_full_graph_padding_lengths(self):
        """FULL replay pads the bucket with zero KV lengths; v4 gets 1 instead."""
        seq_lens = torch.tensor([19, 23, 0, 0], dtype=torch.int32)

        cu_seqlens_q, seqused_k = _build_fa4_seq_tensors(num_tokens=16, seq_lens=seq_lens)

        # get_scheduler_metadata TORCH_CHECKs both dtypes as int32.
        self.assertEqual(cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(seqused_k.dtype, torch.int32)
        # B+1 entries with a leading zero, derived from the static shape rather
        # than from the padded query_start_loc the producer emits.
        self.assertTrue(torch.equal(cu_seqlens_q, torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)))
        self.assertTrue(torch.equal(seqused_k, torch.tensor([19, 23, 1, 1], dtype=torch.int32)))

    def test_rejects_non_uniform_query_batch(self):
        with self.assertRaisesRegex(RuntimeError, "uniform query batch"):
            _build_fa4_seq_tensors(num_tokens=7, seq_lens=torch.tensor([4, 5], dtype=torch.int32))

        with self.assertRaisesRegex(RuntimeError, "uniform query batch"):
            _build_fa4_seq_tensors(num_tokens=8, seq_lens=torch.zeros(0, dtype=torch.int32))


class TestFA4ForwardCache(TestBase):
    def test_metadata_is_computed_once_per_forward_signature(self):
        forward_context = SimpleNamespace()
        expected = (
            torch.tensor([0, 4], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
            torch.empty(1024, dtype=torch.uint8),
        )
        compute = MagicMock(return_value=expected)

        with patch.object(fa4_module, "get_forward_context", return_value=forward_context):
            first = _get_or_compute_fa4_inputs((1, 2, 3), compute)
            second = _get_or_compute_fa4_inputs((1, 2, 3), compute)

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        compute.assert_called_once_with()


class TestFA4WheelLoading(TestBase):
    def test_missing_wheel_is_reported_with_the_build_flag(self):
        with (
            patch.object(fa4_module, "_fa4_module", None),
            patch.object(fa4_module.importlib, "import_module", side_effect=ImportError("missing")),
        ):
            with self.assertRaisesRegex(RuntimeError, "FLASH_ATTN_BUILD_VERSION=v4"):
                _load_fa4()

    def test_wrong_device_build_is_named_as_such(self):
        """flash_attn_npu_4 picks its interface from the device at import time."""
        with (
            patch.object(fa4_module, "_fa4_module", None),
            patch.object(fa4_module.importlib, "import_module", return_value=SimpleNamespace()),
        ):
            with self.assertRaisesRegex(RuntimeError, "get_scheduler_metadata"):
                _load_fa4()


class TestFA4Impl(TestBase):
    def _impl(self):
        impl = AscendFA4Impl.__new__(AscendFA4Impl)
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.head_size = 256
        impl.scale = 256**-0.5
        impl.key_cache = torch.zeros((4, 64, 2, 256), dtype=torch.float16)
        impl.value_cache = torch.zeros((4, 64, 2, 256), dtype=torch.float16)
        return impl

    def test_non_causal_build_takes_the_fa4_path(self):
        impl = AscendFA4Impl.__new__(AscendFA4Impl)
        sentinel = object()
        impl._forward_fa4 = MagicMock(return_value=sentinel)
        metadata = SimpleNamespace(causal=False)

        result = impl.forward_fused_infer_attention("q", "k", "v", metadata, "out", "kv")

        self.assertIs(result, sentinel)
        impl._forward_fa4.assert_called_once_with("q", "k", "v", metadata, "out", "kv")

    def test_causal_build_falls_back_to_the_ordinary_path(self):
        """Mirrors the builder: a causal KV group of a draft is not v4-able here."""
        impl = AscendFA4Impl.__new__(AscendFA4Impl)
        impl._forward_fa4 = MagicMock()
        metadata = SimpleNamespace(causal=True)
        sentinel = object()

        with patch.object(
            AscendFA4Impl.__mro__[1],
            "forward_fused_infer_attention",
            return_value=sentinel,
        ) as base_forward:
            result = impl.forward_fused_infer_attention("q", "k", "v", metadata, "out", "kv")

        self.assertIs(result, sentinel)
        impl._forward_fa4.assert_not_called()
        base_forward.assert_called_once()

    def test_kv_bound_is_the_page_capacity_not_the_actual_max(self):
        """get_scheduler_metadata derives the block-table row stride from this.

        `maxNumBlocksPerBatch = ceil(max_seqlen_k / page_size)` is what the kernel
        offsets rows with, so the bound has to be the capacity the block table was
        allocated at. Passing the actual maximum KV length (300 here) would give a
        stride of 5 rows' worth of blocks against a 5-wide table only by accident,
        and mis-address paged KV as soon as the two diverge. Nothing on either side
        checks it, which is why it is pinned here.
        """
        impl = self._impl()
        num_reqs, q_per_req = 2, 4
        num_tokens = num_reqs * q_per_req
        metadata = SimpleNamespace(
            causal=False,
            num_actual_tokens=num_tokens,
            seq_lens=torch.tensor([300, 120], dtype=torch.int32),
            block_tables=torch.zeros((num_reqs, 5), dtype=torch.int32),
        )
        query = torch.zeros((num_tokens, impl.num_heads, impl.head_size), dtype=torch.float16)
        output = torch.zeros_like(query)

        fa4 = MagicMock()
        fa4.get_scheduler_metadata.return_value = torch.empty(8, dtype=torch.uint8)
        fa4.flash_attn_varlen_func.return_value = torch.ones_like(query)

        with (
            patch.object(fa4_module, "_load_fa4", return_value=fa4),
            patch.object(fa4_module, "get_forward_context", return_value=SimpleNamespace()),
        ):
            impl._forward_fa4(query, None, None, metadata, output, None)

        block_size = impl.key_cache.shape[1]
        expected_bound = metadata.block_tables.shape[1] * block_size
        self.assertEqual(fa4.get_scheduler_metadata.call_args.kwargs["max_seqlen_k"], expected_bound)
        self.assertEqual(fa4.flash_attn_varlen_func.call_args.kwargs["max_seqlen_k"], expected_bound)
        # And the same blob is handed to the forward, so the two agree on layout.
        self.assertIs(
            fa4.flash_attn_varlen_func.call_args.kwargs["scheduler_metadata"],
            fa4.get_scheduler_metadata.return_value,
        )

    def test_forward_is_non_causal_and_writes_the_output_buffer(self):
        impl = self._impl()
        num_reqs, q_per_req = 2, 4
        num_tokens = num_reqs * q_per_req
        metadata = SimpleNamespace(
            causal=False,
            num_actual_tokens=num_tokens,
            seq_lens=torch.tensor([300, 120], dtype=torch.int32),
            block_tables=torch.zeros((num_reqs, 5), dtype=torch.int32),
        )
        query = torch.zeros((num_tokens, impl.num_heads, impl.head_size), dtype=torch.float16)
        output = torch.zeros_like(query)

        fa4 = MagicMock()
        fa4.get_scheduler_metadata.return_value = torch.empty(8, dtype=torch.uint8)
        fa4.flash_attn_varlen_func.return_value = torch.ones_like(query)

        with (
            patch.object(fa4_module, "_load_fa4", return_value=fa4),
            patch.object(fa4_module, "get_forward_context", return_value=SimpleNamespace()),
        ):
            result = impl._forward_fa4(query, None, None, metadata, output, None)

        fwd_kwargs = fa4.flash_attn_varlen_func.call_args.kwargs
        self.assertFalse(fwd_kwargs["causal"])
        self.assertEqual(fwd_kwargs["window_size"], (-1, -1))
        self.assertEqual(fwd_kwargs["max_seqlen_q"], q_per_req)
        # The paged cache goes in with its own layout: v4 wants
        # (num_blocks, page_size, num_kv_heads, head_size), which is what Ascend
        # already allocates, so unlike the sink operator there is no view.
        self.assertIs(fa4.flash_attn_varlen_func.call_args.args[1], impl.key_cache)
        self.assertIs(result, output)
        self.assertTrue(torch.equal(output, torch.ones_like(query)))

    def test_block_table_shorter_than_the_batch_is_refused(self):
        impl = self._impl()
        metadata = SimpleNamespace(
            causal=False,
            num_actual_tokens=8,
            seq_lens=torch.tensor([300, 120], dtype=torch.int32),
            block_tables=torch.zeros((1, 5), dtype=torch.int32),
        )
        query = torch.zeros((8, impl.num_heads, impl.head_size), dtype=torch.float16)

        with patch.object(fa4_module, "_load_fa4", return_value=MagicMock()):
            with self.assertRaisesRegex(RuntimeError, "fewer rows than requests"):
                impl._forward_fa4(query, None, None, metadata, torch.zeros_like(query), None)
