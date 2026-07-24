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

from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention import adn_fused_infer_attention as adn_mod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

ADN_MOD = "vllm_ascend._310p.attention.adn_fused_infer_attention"

NUM_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
NUM_BLOCKS = 8
BATCH = 3
DFLASH_Q = 9  # K + 1
KV_LENS = [200, 133, 65]


def make_vllm_config(*, method="dflash", num_spec=8, arch="DFlashQwen3ForCausalLM", eager=True, tp=2):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            num_speculative_tokens=num_spec,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])),
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


def make_cache(*, dtype=torch.float16, block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS):
    shape = (NUM_BLOCKS, num_kv_heads * HEAD_DIM // 16, block_size, 16)
    return torch.zeros(shape, dtype=dtype)


def make_impl(*, vllm_config=None, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_DIM, cache=None):
    key_cache = cache if cache is not None else make_cache()
    return SimpleNamespace(
        vllm_config=vllm_config if vllm_config is not None else make_vllm_config(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        scale=HEAD_DIM**-0.5,
        key_cache=key_cache,
        value_cache=key_cache.clone(),
        _adn_scope_validated=False,
    )


def make_metadata(*, q_lens=None, kv_lens=None, block_cols=4, block_dtype=torch.int32):
    q_lens = q_lens if q_lens is not None else [DFLASH_Q] * BATCH
    kv_lens = kv_lens if kv_lens is not None else list(KV_LENS)
    md = SimpleNamespace(
        num_actual_tokens=sum(q_lens),
        seq_lens_list=kv_lens,
        block_tables=torch.zeros(len(kv_lens), block_cols, dtype=block_dtype),
        # Deliberately wrong: the base builder overwrites this field with
        # cumulative endpoints, and the adapter must not read it.
        actual_seq_lengths_q=list(torch.tensor(q_lens).cumsum(0).tolist()),
        causal=False,
        attn_mask=torch.ones(2048, 2048),  # must be ignored
    )
    md.query_lens_cpu = torch.tensor(q_lens, dtype=torch.int32)
    return md


class _FakeAdn:
    """Stands in for adn_custom_ops, which is not installed on CI hosts."""

    def __init__(self, out_builder=None):
        self.calls = []
        self._out_builder = out_builder

    def adn_fused_infer_attention(self, **kwargs):
        self.calls.append(kwargs)
        if self._out_builder is not None:
            return self._out_builder(kwargs)
        return torch.zeros_like(kwargs["query"])


def run_forward(impl=None, md=None, adn=None, num_tokens=None):
    impl = impl if impl is not None else make_impl()
    md = md if md is not None else make_metadata()
    adn = adn if adn is not None else _FakeAdn()
    n = num_tokens if num_tokens is not None else md.num_actual_tokens
    # Width follows the impl so head-layout cases reshape cleanly and fail on the
    # scope check rather than on the reshape itself.
    width = impl.num_heads * impl.head_size
    query = torch.zeros(n, width, dtype=torch.float16)
    output = torch.zeros(n, width, dtype=torch.float16)
    with (
        mock_patch(f"{ADN_MOD}.load_adn", return_value=adn),
        mock_patch(f"{ADN_MOD}.torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_NZ),
    ):
        result = adn_mod.forward_parallel_draft_adn(impl, query, md, output)
    return result, adn, output


class TestAdnCallContract(TestBase):
    def test_mask_is_none_regardless_of_metadata(self):
        _, adn, _ = run_forward()
        self.assertIsNone(adn.calls[0]["attn_mask"])

    def test_precision_layout_and_force_call_flags(self):
        _, adn, _ = run_forward()
        kwargs = adn.calls[0]
        self.assertEqual(kwargs["inner_precise"], 2)
        self.assertIs(kwargs["force_call"], False)
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertEqual(kwargs["block_size"], BLOCK_SIZE)

    def test_q_lens_are_raw_not_cumulative(self):
        md = make_metadata()
        _, adn, _ = run_forward(md=md)
        self.assertEqual(adn.calls[0]["actual_seq_lengths_q"], [DFLASH_Q] * BATCH)
        self.assertNotEqual(
            adn.calls[0]["actual_seq_lengths_q"],
            md.actual_seq_lengths_q,
            "adapter used the cumulative endpoints from the base metadata",
        )

    def test_kv_lens_are_passed_through(self):
        _, adn, _ = run_forward()
        self.assertEqual(adn.calls[0]["actual_seq_lengths_kv"], KV_LENS)

    def test_caches_are_passed_by_reference(self):
        impl = make_impl()
        _, adn, _ = run_forward(impl=impl)
        self.assertIs(adn.calls[0]["key"], impl.key_cache)
        self.assertIs(adn.calls[0]["value"], impl.value_cache)

    def test_head_counts_and_scale(self):
        _, adn, _ = run_forward()
        kwargs = adn.calls[0]
        self.assertEqual(kwargs["num_heads"], NUM_HEADS)
        self.assertEqual(kwargs["num_key_value_heads"], NUM_KV_HEADS)
        self.assertAlmostEqual(kwargs["scale_value"], HEAD_DIM**-0.5)

    def test_query_is_reshaped_to_tnd(self):
        _, adn, _ = run_forward()
        self.assertEqual(tuple(adn.calls[0]["query"].shape), (BATCH * DFLASH_Q, NUM_HEADS, HEAD_DIM))

    def test_result_is_copied_into_the_caller_buffer(self):
        marker = 0.5

        def build_out(kwargs):
            return torch.full_like(kwargs["query"], marker)

        result, _, output = run_forward(adn=_FakeAdn(build_out))
        self.assertTrue(bool((output[: BATCH * DFLASH_Q] == marker).all()))
        self.assertIs(result, output, "adapter must return the caller's output buffer")

    def test_dspark_expects_k_queries_not_k_plus_one(self):
        impl = make_impl(vllm_config=make_vllm_config(method="dspark", num_spec=7, arch="Qwen3DSparkForCausalLM"))
        md = make_metadata(q_lens=[7] * BATCH)
        _, adn, _ = run_forward(impl=impl, md=md)
        self.assertEqual(adn.calls[0]["actual_seq_lengths_q"], [7] * BATCH)


class TestAdnDynamicGuards(TestBase):
    def test_missing_raw_q_lens_is_refused(self):
        md = make_metadata()
        del md.query_lens_cpu
        with self.assertRaisesRegex(RuntimeError, "query_lens_cpu is missing"):
            run_forward(md=md)

    def test_wrong_q_len_for_method_is_refused(self):
        # DFlash must query K + 1 = 9; 8 would silently drop the anchor.
        md = make_metadata(q_lens=[8] * BATCH)
        with self.assertRaisesRegex(RuntimeError, "expects every request to query 9"):
            run_forward(md=md)

    def test_cumulative_q_lens_are_refused(self):
        md = make_metadata()
        md.query_lens_cpu = torch.tensor([9, 18, 27], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "expects every request to query 9"):
            run_forward(md=md)

    def test_token_count_disagreement_is_refused(self):
        md = make_metadata()
        md.num_actual_tokens = BATCH * DFLASH_Q - 1
        with self.assertRaisesRegex(RuntimeError, "must all agree"):
            run_forward(md=md)

    def test_batch_size_disagreement_is_refused(self):
        md = make_metadata(kv_lens=[200, 133])
        with self.assertRaisesRegex(RuntimeError, "batch size disagreement"):
            run_forward(md=md)

    def test_non_int32_block_table_is_refused(self):
        md = make_metadata(block_dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "rank-2 int32"):
            run_forward(md=md)

    def test_kv_len_exceeding_block_table_capacity_is_refused(self):
        # 2 columns x 128 = 256 addressable, but one request needs 300.
        md = make_metadata(kv_lens=[300, 133, 65], block_cols=2)
        with self.assertRaisesRegex(RuntimeError, "exceeds what its block table can address"):
            run_forward(md=md)

    def test_q_len_greater_than_kv_len_is_refused(self):
        md = make_metadata(kv_lens=[200, 133, 5])
        with self.assertRaisesRegex(RuntimeError, r"need 0 < q_len\(9\) <= kv_len\(5\)"):
            run_forward(md=md)


class TestAdnScopeValidation(TestBase):
    def _expect_refusal(self, pattern, **impl_kwargs):
        with self.assertRaisesRegex(RuntimeError, pattern):
            run_forward(impl=make_impl(**impl_kwargs))

    def test_unsupported_method_is_refused(self):
        # Refused by the adapter's own early check, before the scope validator: it
        # has to resolve queries-per-request from the method first. Reaching here
        # at all means routing let something through, hence the wording.
        self._expect_refusal("reached with unsupported method", vllm_config=make_vllm_config(method="eagle3"))

    def test_scope_validator_refuses_unsupported_method_on_its_own(self):
        """The validator is the documented scope lock, so its method branch is
        covered directly -- the adapter's early check shadows it in the forward
        path, which would otherwise leave it untested."""
        cache = make_cache()
        with self.assertRaisesRegex(RuntimeError, "only covers"):
            adn_mod.validate_adn_scope(
                vllm_config=make_vllm_config(method="eagle3"),
                query=torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16),
                key_cache=cache,
                value_cache=cache.clone(),
                num_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_DIM,
            )

    def test_unexpected_k_is_refused(self):
        self._expect_refusal("only validated at num_speculative_tokens=8", vllm_config=make_vllm_config(num_spec=5))

    def test_unsupported_architecture_is_refused(self):
        self._expect_refusal("outside this scope", vllm_config=make_vllm_config(arch="LlamaForCausalLM"))

    def test_graph_mode_is_refused(self):
        self._expect_refusal("eager-only", vllm_config=make_vllm_config(eager=False))

    def test_other_tp_is_refused(self):
        self._expect_refusal("TP=2 only", vllm_config=make_vllm_config(tp=4))

    def test_other_head_layout_is_refused(self):
        self._expect_refusal("only covers local Nq=16", num_heads=32)

    def test_bf16_is_refused(self):
        self._expect_refusal("only supports float16", cache=make_cache(dtype=torch.bfloat16))

    def test_wrong_cache_block_size_is_refused(self):
        self._expect_refusal("this scope only covers 128", cache=make_cache(block_size=64))

    def test_non_nz_cache_format_is_refused(self):
        impl = make_impl()
        md = make_metadata()
        query = torch.zeros(md.num_actual_tokens, NUM_HEADS * HEAD_DIM, dtype=torch.float16)
        output = torch.zeros_like(query)
        with (
            mock_patch(f"{ADN_MOD}.load_adn", return_value=_FakeAdn()),
            mock_patch(f"{ADN_MOD}.torch_npu.get_npu_format", return_value=2),
        ):
            with self.assertRaisesRegex(RuntimeError, "expected ACL_FORMAT_FRACTAL_NZ"):
                adn_mod.forward_parallel_draft_adn(impl, query, md, output)

    def test_scope_is_validated_once_then_cached(self):
        impl = make_impl()
        adn = _FakeAdn()
        with mock_patch(f"{ADN_MOD}.validate_adn_scope") as validator:
            with (
                mock_patch(f"{ADN_MOD}.load_adn", return_value=adn),
                mock_patch(f"{ADN_MOD}.torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_NZ),
            ):
                for _ in range(3):
                    md = make_metadata()
                    query = torch.zeros(md.num_actual_tokens, NUM_HEADS * HEAD_DIM, dtype=torch.float16)
                    adn_mod.forward_parallel_draft_adn(impl, query, md, torch.zeros_like(query))
        self.assertEqual(validator.call_count, 1, "startup invariants must not be re-checked per step")
        self.assertEqual(len(adn.calls), 3)


class TestAdnLoader(TestBase):
    def test_missing_package_fails_loud_without_fallback(self):
        with mock_patch.dict("sys.modules", {"adn_custom_ops": None}):
            with mock_patch(f"{ADN_MOD}._adn_module", None):
                with self.assertRaisesRegex(RuntimeError, "no fallback"):
                    adn_mod.load_adn()


class TestAdnReturnShape(TestBase):
    def test_mismatched_return_shape_is_refused(self):
        def bad_shape(kwargs):
            q = kwargs["query"]
            # Same element count, wrong layout -- numel alone would accept this.
            return torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=q.dtype)

        with self.assertRaisesRegex(RuntimeError, "expected the query shape"):
            run_forward(adn=_FakeAdn(bad_shape))

    def test_mismatched_return_dtype_is_refused(self):
        def bad_dtype(kwargs):
            return torch.zeros_like(kwargs["query"], dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "expected the query shape"):
            run_forward(adn=_FakeAdn(bad_dtype))
