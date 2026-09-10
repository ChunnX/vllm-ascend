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
"""Unit tests for the parallel-drafting flash-attention-npu attention backend.

These cover when the backend is enabled, how selection divides head sizes with the
FIA sink backend, the device-side length construction, the KV bound the paged block
table is addressed with, the once-per-forward metadata cache, and the logging a
full-network run reads to confirm the operator was used. The operator call itself needs an NPU and the
wheel, so it is mocked here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.flash_attn_npu_v1 as fa_module
from tests.ut.base import TestBase
from vllm_ascend.attention.flash_attn_npu_v1 import (
    AscendFlashAttnV4Backend,
    AscendFlashAttnV4Impl,
    AscendFlashAttnV4MetadataBuilder,
    _build_seq_tensors,
    _cu_seqlens_q,
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


class TestEnableFlag(TestBase):
    def test_unset_disables_the_backend(self):
        with patch.object(fa_module, "_SELECTED", ""):
            self.assertFalse(enabled())

    def test_the_targeted_api_enables_it(self):
        with patch.object(fa_module, "_SELECTED", "v4"):
            self.assertTrue(enabled())

    def test_unknown_value_is_refused_not_ignored(self):
        """A typo is a request that cannot be served, not a request to serve nothing.

        Silently disabling would look identical to the flag working, and the only
        symptom would be a draft that is quietly slower. v3 is refused the same way
        as nonsense: this backend no longer carries that adapter.
        """
        for value in ("v3", "v5"):
            with self.subTest(value=value):
                with patch.object(fa_module, "_SELECTED", value):
                    with self.assertRaisesRegex(RuntimeError, "not an API this backend serves"):
                        enabled()


class TestLayerSelection(TestBase):
    def test_disabled_by_default(self):
        """Opt-in, like the sink backend: nothing routes here without the env var."""
        with patch.object(fa_module, "_SELECTED", ""):
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True)))

    def test_selects_non_causal_draft_layers(self):
        with patch.object(fa_module, "_SELECTED", "v4"):
            self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True)))
            # The target model is causal, so it never reaches this backend.
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=False)))

    def test_head_dim_256_is_the_gap_this_backend_exists_for(self):
        """The sink operator serves 128/192/512; this wheel covers everything to 256."""
        with patch.object(fa_module, "_SELECTED", "v4"):
            self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=256)))

    def test_yields_the_sink_operators_head_sizes_back_to_it(self):
        """Where both can serve, the operator with hardware runs behind it wins.

        Only while the sink backend is actually enabled -- with its flag off there
        is no other backend keeping the device-side lengths, so this one takes it.
        """
        with patch.object(fa_module, "_SELECTED", "v4"):
            with patch.object(fa_module, "_FIA_SINK_ENABLED", True):
                for head_size in fa_module.FIA_SINK_HEAD_SIZES:
                    with self.subTest(head_size=head_size):
                        self.assertFalse(
                            flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=head_size))
                        )

            with patch.object(fa_module, "_FIA_SINK_ENABLED", False):
                self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=128)))

    def test_refuses_head_sizes_the_forward_rejects(self):
        """mha_fwd checks head_size <= 256, so 512 is not this backend's to take."""
        with patch.object(fa_module, "_SELECTED", "v4"), patch.object(fa_module, "_FIA_SINK_ENABLED", False):
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=512)))
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, head_size=0)))

    def test_excludes_what_the_operator_call_cannot_express(self):
        """This call passes causal=False, window_size=(-1, -1) and no sink tensor."""
        with patch.object(fa_module, "_SELECTED", "v4"):
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, has_sliding_window=True)))
            self.assertFalse(flash_attn_npu_selected(_selector_config(use_non_causal=True, has_sink=True)))

    def test_reads_only_fields_of_the_selector_config(self):
        """_cached_get_attn_backend memoizes on the selector config alone.

        A predicate that consulted get_current_vllm_config() would be answered
        once and reused for every later config that hashed the same.
        """
        with patch.object(fa_module, "_SELECTED", "v4"):
            with patch("vllm.config.get_current_vllm_config", side_effect=AssertionError("must not be read")):
                self.assertTrue(flash_attn_npu_selected(_selector_config(use_non_causal=True)))


class TestBackendWiring(TestBase):
    def test_backend_names_its_own_builder_and_impl(self):
        self.assertIs(AscendFlashAttnV4Backend.get_impl_cls(), AscendFlashAttnV4Impl)
        self.assertIs(AscendFlashAttnV4Backend.get_builder_cls(), AscendFlashAttnV4MetadataBuilder)

    def test_get_name_stays_something_vllm_can_resolve(self):
        """The name is a registry key, not an identity.

        `Attention.__init__` does `AttentionBackendEnum[attn_backend.get_name()]`,
        so a name of this backend's own raises "Unknown attention backend" before a
        single layer is built -- which is exactly what a name like
        "ASCEND_FLASH_ATTN_NPU" did. Asserting the literal string is what let that
        through unit tests and into a full-network run, so assert the contract: it
        resolves, and it is whatever the base backend answers.
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        name = AscendFlashAttnV4Backend.get_name()
        self.assertEqual(name, AscendAttentionBackend.get_name())
        AttentionBackendEnum[name]  # raises ValueError if vLLM cannot resolve it

    def test_kv_cache_layout_is_inherited_unchanged(self):
        """The draft shares the target's cache pool, so the layout must match.

        ``get_required_kv_cache_layout`` is applied through a process-global
        setter, so a layout of its own here would not stay on this backend.
        """
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        self.assertIs(
            AscendFlashAttnV4Backend.get_required_kv_cache_layout.__func__,
            AscendAttentionBackend.get_required_kv_cache_layout.__func__,
        )
        self.assertEqual(
            AscendFlashAttnV4Backend.get_kv_cache_shape(2, 4, 8, 16),
            AscendAttentionBackend.get_kv_cache_shape(2, 4, 8, 16),
        )


class TestMetadataBuilder(TestBase):
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
        with (
            patch.object(fa_module, "_SELECTED", "v4"),
            patch.object(fa_module, "_load") as load,
        ):
            builder = AscendFlashAttnV4MetadataBuilder(
                None,
                layer_names or ["model.layers.0.self_attn.attn"],
                self.mock_vllm_config,
                self.mock_device,
            )
        return builder, load

    def test_wheel_is_checked_at_construction(self):
        """A missing wheel should stop startup, not the first request."""
        _, load = self._build()

        load.assert_called_once_with()

    def test_logs_the_layers_and_the_build_they_reached(self):
        """A full-network run is read for these two lines.

        The first says which layers chose this backend. The second names the
        resolved module file, because a wheel in site-packages and a source tree
        that shadows it are different binaries and the only symptom of the wrong
        one is the operator behaving like an older version of itself.
        """
        with patch.object(fa_module.logger, "info") as mock_info:
            self._build(["model.layers.3.self_attn.attn"])

        self.assertEqual(mock_info.call_count, 2)
        selected, loaded = (str(c.args) for c in mock_info.call_args_list)
        self.assertIn("model.layers.3.self_attn.attn", selected)
        self.assertIn("v4", selected)
        self.assertIn("loaded from", loaded)

    def test_refuses_a_model_without_parallel_drafting(self):
        """use_non_causal is not exclusive to drafts -- DiffusionGemma sets it."""
        self.mock_vllm_config.speculative_config = None

        with patch.object(fa_module, "_SELECTED", "v4"), patch.object(fa_module, "_load"):
            with self.assertRaisesRegex(RuntimeError, "without parallel drafting"):
                AscendFlashAttnV4MetadataBuilder(None, ["layer0"], self.mock_vllm_config, self.mock_device)

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


class TestSeqTensors(TestBase):
    def test_builds_legal_full_graph_padding_lengths(self):
        """FULL replay pads the bucket with zero KV lengths; the wheel gets 1."""
        seq_lens = torch.tensor([19, 23, 0, 0], dtype=torch.int32)

        cu_seqlens_q, seqused_k = _build_seq_tensors(num_tokens=16, seq_lens=seq_lens)

        # get_scheduler_metadata TORCH_CHECKs both dtypes as int32.
        self.assertEqual(cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(seqused_k.dtype, torch.int32)
        # B+1 entries with a leading zero, derived from the static shape rather
        # than from the padded query_start_loc the producer emits.
        self.assertTrue(torch.equal(cu_seqlens_q, torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)))
        self.assertTrue(torch.equal(seqused_k, torch.tensor([19, 23, 1, 1], dtype=torch.int32)))

    def test_query_offsets_are_built_once_per_shape(self):
        """They are a function of the shape, and rebuilding them is not free.

        Profiling put the arange and its multiply at 13.6us of device time per step
        -- launch overhead on two tiny vector kernels -- for values that cannot
        change while the batch shape holds. Reusing the tensor also gives aclgraph a
        stable address to capture.
        """
        first = _cu_seqlens_q(4, 4, torch.device("cpu"))
        again = _cu_seqlens_q(4, 4, torch.device("cpu"))
        other = _cu_seqlens_q(4, 8, torch.device("cpu"))

        self.assertIs(first, again)
        self.assertIsNot(first, other)
        self.assertTrue(torch.equal(first, torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)))
        self.assertTrue(torch.equal(other, torch.tensor([0, 8, 16, 24, 32], dtype=torch.int32)))

    def test_rejects_non_uniform_query_batch(self):
        with self.assertRaisesRegex(RuntimeError, "uniform query batch"):
            _build_seq_tensors(num_tokens=7, seq_lens=torch.tensor([4, 5], dtype=torch.int32))

        with self.assertRaisesRegex(RuntimeError, "uniform query batch"):
            _build_seq_tensors(num_tokens=8, seq_lens=torch.zeros(0, dtype=torch.int32))


class TestForwardCache(TestBase):
    def test_metadata_is_computed_once_per_forward_signature(self):
        forward_context = SimpleNamespace()
        expected = (
            torch.tensor([0, 4], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
            torch.empty(1024, dtype=torch.uint8),
        )
        compute = MagicMock(return_value=expected)

        with patch.object(fa_module, "get_forward_context", return_value=forward_context):
            first = _get_or_compute_inputs(("v4", 1, 2, 3), compute)
            second = _get_or_compute_inputs(("v4", 1, 2, 3), compute)

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        compute.assert_called_once_with()


class TestWheelLoading(TestBase):
    def test_missing_wheel_is_reported_with_the_build_flag(self):
        with (
            patch.dict(fa_module._loaded_modules, {}, clear=True),
            patch.object(fa_module.importlib, "import_module", side_effect=ImportError("missing")),
        ):
            with self.assertRaisesRegex(RuntimeError, "FLASH_ATTN_BUILD_VERSION=v4"):
                _load()

    def test_wrong_device_build_is_named_as_such(self):
        """The wheel picks its interface from the device at import time."""
        with (
            patch.dict(fa_module._loaded_modules, {}, clear=True),
            patch.object(fa_module.importlib, "import_module", return_value=SimpleNamespace()),
        ):
            with self.assertRaisesRegex(RuntimeError, "get_scheduler_metadata"):
                _load()


class TestImpl(TestBase):
    def _impl(self):
        impl = AscendFlashAttnV4Impl.__new__(AscendFlashAttnV4Impl)
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.head_size = 256
        impl.scale = 256**-0.5
        impl.key_cache = torch.zeros((4, 64, 2, 256), dtype=torch.float16)
        impl.value_cache = torch.zeros((4, 64, 2, 256), dtype=torch.float16)
        return impl

    @staticmethod
    def _metadata(num_reqs=2, q_per_req=4, block_table_width=5):
        return SimpleNamespace(
            causal=False,
            num_actual_tokens=num_reqs * q_per_req,
            seq_lens=torch.tensor([300, 120][:num_reqs], dtype=torch.int32),
            block_tables=torch.zeros((num_reqs, block_table_width), dtype=torch.int32),
        )

    def _run(self, impl, metadata):
        num_tokens = metadata.num_actual_tokens
        query = torch.zeros((num_tokens, impl.num_heads, impl.head_size), dtype=torch.float16)
        output = torch.zeros_like(query)

        module = MagicMock()
        module.get_scheduler_metadata.return_value = torch.empty(8, dtype=torch.uint8)
        module.flash_attn_varlen_func.return_value = torch.ones_like(query)

        with (
            patch.object(fa_module, "_SELECTED", "v4"),
            patch.object(fa_module, "_load", return_value=module),
            patch.object(fa_module, "get_forward_context", return_value=SimpleNamespace()),
        ):
            result = impl._forward_flash_attn_npu(query, None, None, metadata, output, None)
        return module, query, output, result

    def test_non_causal_build_takes_the_wheel_path(self):
        impl = AscendFlashAttnV4Impl.__new__(AscendFlashAttnV4Impl)
        sentinel = object()
        impl._forward_flash_attn_npu = MagicMock(return_value=sentinel)
        metadata = SimpleNamespace(causal=False)

        result = impl.forward_fused_infer_attention("q", "k", "v", metadata, "out", "kv")

        self.assertIs(result, sentinel)
        impl._forward_flash_attn_npu.assert_called_once_with("q", "k", "v", metadata, "out", "kv")

    def test_causal_build_falls_back_to_the_ordinary_path(self):
        """Mirrors the builder: a causal KV group of a draft is not served here."""
        impl = AscendFlashAttnV4Impl.__new__(AscendFlashAttnV4Impl)
        impl._forward_flash_attn_npu = MagicMock()
        metadata = SimpleNamespace(causal=True)
        sentinel = object()

        with patch.object(
            AscendFlashAttnV4Impl.__mro__[1],
            "forward_fused_infer_attention",
            return_value=sentinel,
        ) as base_forward:
            result = impl.forward_fused_infer_attention("q", "k", "v", metadata, "out", "kv")

        self.assertIs(result, sentinel)
        impl._forward_flash_attn_npu.assert_not_called()
        base_forward.assert_called_once()

    def test_dispatches_to_flash_attn_varlen_func(self):
        impl = self._impl()
        module, _, _, _ = self._run(impl, self._metadata())

        module.flash_attn_varlen_func.assert_called_once()
        kwargs = module.flash_attn_varlen_func.call_args.kwargs
        self.assertEqual(kwargs["max_seqlen_k"], 5 * 64)
        self.assertIn("seqused_k", kwargs)

    def test_kv_bound_is_the_page_capacity_not_the_actual_max(self):
        """get_scheduler_metadata derives the block-table row stride from this.

        `maxNumBlocksPerBatch = ceil(max_seqlen_k / page_size)` is what the kernel
        offsets rows with, so the bound has to be the capacity the block table was
        allocated at. Passing the actual maximum KV length (300 here) would
        mis-address paged KV as soon as the two diverge. v4 does not check it, which
        is why it is pinned here.
        """
        impl = self._impl()
        metadata = self._metadata()
        module, _, _, _ = self._run(impl, metadata)

        expected_bound = metadata.block_tables.shape[1] * impl.key_cache.shape[1]
        self.assertEqual(module.get_scheduler_metadata.call_args.kwargs["max_seqlen_k"], expected_bound)
        self.assertEqual(module.flash_attn_varlen_func.call_args.kwargs["max_seqlen_k"], expected_bound)
        # The same blob is handed to the forward, so the two agree on layout.
        self.assertIs(
            module.flash_attn_varlen_func.call_args.kwargs["scheduler_metadata"],
            module.get_scheduler_metadata.return_value,
        )

    def test_forward_logs_that_the_operator_actually_ran(self):
        """The selection log proves a layer chose this backend, not that it ran.

        An end-to-end run needs to distinguish the two, so the forward says so once
        with the shapes it ran on.
        """
        impl = self._impl()
        with patch.object(fa_module.logger, "info_once") as mock_info:
            self._run(impl, self._metadata())

        mock_info.assert_called_once()
        logged = str(mock_info.call_args.args)
        self.assertIn("forward", logged)
        self.assertIn("v4", logged)

    def test_forward_is_non_causal_and_writes_the_output_buffer(self):
        impl = self._impl()
        metadata = self._metadata()
        module, query, output, result = self._run(impl, metadata)

        kwargs = module.flash_attn_varlen_func.call_args.kwargs
        self.assertFalse(kwargs["causal"])
        self.assertEqual(kwargs["window_size"], (-1, -1))
        self.assertEqual(kwargs["max_seqlen_q"], 4)
        # The paged cache goes in with its own layout: the wheel wants
        # (num_blocks, page_size, num_kv_heads, head_size), which is what Ascend
        # already allocates, so unlike the sink operator there is no view.
        self.assertIs(module.flash_attn_varlen_func.call_args.args[1], impl.key_cache)
        self.assertIs(result, output)
        self.assertTrue(torch.equal(output, torch.ones_like(query)))

    def test_block_table_shorter_than_the_batch_is_refused(self):
        impl = self._impl()
        metadata = self._metadata()
        metadata.block_tables = torch.zeros((1, 5), dtype=torch.int32)
        query = torch.zeros((8, impl.num_heads, impl.head_size), dtype=torch.float16)

        with (
            patch.object(fa_module, "_SELECTED", "v4"),
            patch.object(fa_module, "_load", return_value=MagicMock()),
        ):
            with self.assertRaisesRegex(RuntimeError, "fewer rows than requests"):
                impl._forward_flash_attn_npu(query, None, None, metadata, torch.zeros_like(query), None)
