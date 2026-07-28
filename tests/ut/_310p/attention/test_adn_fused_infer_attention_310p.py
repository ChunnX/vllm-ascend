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


def make_vllm_config(*, method="dflash", num_spec=8, arch="DFlashDraftModel", eager=True, tp=2):
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
    """Stands in for the NPU-only _C_ascend out operator on CPU CI hosts."""

    def __init__(self, out_builder=None):
        self.calls = []
        self._out_builder = out_builder

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self._out_builder is not None:
            return self._out_builder(kwargs)
        return kwargs["output"]


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
        mock_patch(f"{ADN_MOD}._call_adn_fused_infer_attention_out", side_effect=adn),
        mock_patch(f"{ADN_MOD}.torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_NZ),
    ):
        result = adn_mod.forward_parallel_draft_adn(impl, query, md, output)
    return result, adn, output


class TestAdnCallContract(TestBase):
    def test_mask_is_none_regardless_of_metadata(self):
        _, adn, _ = run_forward()
        self.assertIsNone(adn.calls[0]["attn_mask"])

    def test_precision_layout_and_out_contract(self):
        _, adn, _ = run_forward()
        kwargs = adn.calls[0]
        self.assertEqual(kwargs["inner_precise"], 2)
        self.assertNotIn("force_call", kwargs)
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertEqual(kwargs["block_size"], BLOCK_SIZE)
        self.assertEqual(tuple(kwargs["output"].shape), tuple(kwargs["query"].shape))

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

    def test_result_is_written_into_the_caller_buffer(self):
        marker = 0.5
        padded_rows = 5

        def build_out(kwargs):
            kwargs["output"].fill_(marker)
            return kwargs["output"]

        result, _, output = run_forward(
            adn=_FakeAdn(build_out),
            num_tokens=BATCH * DFLASH_Q + padded_rows,
        )
        self.assertTrue(bool((output[: BATCH * DFLASH_Q] == marker).all()))
        self.assertTrue(bool((output[BATCH * DFLASH_Q :] == 0).all()))
        self.assertIs(result, output, "adapter must return the caller's output buffer")

    def test_dspark_expects_k_queries_not_k_plus_one(self):
        impl = make_impl(vllm_config=make_vllm_config(method="dspark", num_spec=7, arch="Qwen3DSparkModel"))
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

    def test_engine_graph_mode_is_allowed(self):
        """Whole-engine graph mode must NOT be refused: the target may run in
        ACLGraph while the drafter stays eager. What matters is that ADN itself is
        not captured, which is checked per call (see TestAdnCaptureGuard)."""
        _, adn, _ = run_forward(impl=make_impl(vllm_config=make_vllm_config(eager=False)))
        self.assertEqual(len(adn.calls), 1)

    def test_tp4_layout_is_allowed(self):
        """TP only shards heads; the per-rank layout is what is checked. Qwen3-8B
        at TP=4 is 8 query / 2 KV heads, which satisfies the structural rules, so
        it must run rather than be refused."""
        impl = make_impl(
            vllm_config=make_vllm_config(tp=4),
            num_heads=8,
            num_kv_heads=2,
            cache=make_cache(num_kv_heads=2),
        )
        _, adn, _ = run_forward(impl=impl)
        self.assertEqual(len(adn.calls), 1)
        self.assertEqual(adn.calls[0]["num_heads"], 8)
        self.assertEqual(adn.calls[0]["num_key_value_heads"], 2)

    def test_illegal_gqa_is_refused(self):
        # 17 query heads do not divide evenly by 4 KV heads.
        self._expect_refusal("invalid GQA layout", num_heads=17)

    def test_non_16_aligned_kv_is_refused(self):
        # Any num_kv_heads * 128 is 16-aligned, so break it with a small head_dim:
        # 1 KV head * 8 = 8 is not a multiple of 16.
        self._expect_refusal(
            "must be a multiple of 16",
            num_heads=4,
            num_kv_heads=1,
            head_size=8,
            cache=make_cache(num_kv_heads=1),
        )

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
            mock_patch(f"{ADN_MOD}._call_adn_fused_infer_attention_out", side_effect=_FakeAdn()),
            mock_patch(f"{ADN_MOD}.torch_npu.get_npu_format", return_value=2),
        ):
            with self.assertRaisesRegex(RuntimeError, "expected ACL_FORMAT_FRACTAL_NZ"):
                adn_mod.forward_parallel_draft_adn(impl, query, md, output)

    def test_scope_is_validated_once_then_cached(self):
        impl = make_impl()
        adn = _FakeAdn()
        with mock_patch(f"{ADN_MOD}.validate_adn_scope") as validator:
            with (
                mock_patch(f"{ADN_MOD}._call_adn_fused_infer_attention_out", side_effect=adn),
                mock_patch(f"{ADN_MOD}.torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_NZ),
            ):
                for _ in range(3):
                    md = make_metadata()
                    query = torch.zeros(md.num_actual_tokens, NUM_HEADS * HEAD_DIM, dtype=torch.float16)
                    adn_mod.forward_parallel_draft_adn(impl, query, md, torch.zeros_like(query))
        self.assertEqual(validator.call_count, 1, "startup invariants must not be re-checked per step")
        self.assertEqual(len(adn.calls), 3)


class TestAdnBinding(TestBase):
    def test_success_path_wraps_caches_and_forwards_keywords(self):
        calls = []
        query = torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        key = make_cache()
        value = make_cache()
        output = torch.empty_like(query)
        block_table = torch.zeros(1, 1, dtype=torch.int32)

        def fake_out_op(query_arg, key_arg, value_arg, output_arg, **kwargs):
            calls.append((query_arg, key_arg, value_arg, output_arg, kwargs))
            return output_arg

        fake_ops = SimpleNamespace(
            _C_ascend=SimpleNamespace(
                npu_adn_fused_infer_attention_out=fake_out_op,
            )
        )
        with (
            mock_patch(f"{ADN_MOD}.enable_custom_op", return_value=True),
            mock_patch.object(adn_mod.torch, "ops", fake_ops),
        ):
            result = adn_mod._call_adn_fused_infer_attention_out(
                query=query,
                key=key,
                value=value,
                output=output,
                attn_mask=None,
                actual_seq_lengths_q=[1],
                actual_seq_lengths_kv=[1],
                block_table=block_table,
                num_heads=NUM_HEADS,
                scale_value=HEAD_DIM**-0.5,
                input_layout="TND",
                num_key_value_heads=NUM_KV_HEADS,
                block_size=BLOCK_SIZE,
                inner_precise=2,
            )

        self.assertIs(result, output)
        self.assertEqual(len(calls), 1)
        query_arg, key_arg, value_arg, output_arg, kwargs = calls[0]
        self.assertIs(query_arg, query)
        self.assertEqual(len(key_arg), 1)
        self.assertEqual(len(value_arg), 1)
        self.assertIs(key_arg[0], key)
        self.assertIs(value_arg[0], value)
        self.assertIs(output_arg, output)
        self.assertIsNone(kwargs["attn_mask"])
        self.assertEqual(kwargs["actual_seq_lengths_q"], [1])
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [1])
        self.assertIs(kwargs["block_table"], block_table)
        self.assertEqual(kwargs["inner_precise"], 2)

    def test_disabled_custom_ops_fail_loud_without_fallback(self):
        with mock_patch(f"{ADN_MOD}.enable_custom_op", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "There is no causal fallback"):
                adn_mod._call_adn_fused_infer_attention_out(
                    query=torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16),
                    key=make_cache(),
                    value=make_cache(),
                    output=torch.zeros(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16),
                    attn_mask=None,
                    actual_seq_lengths_q=[1],
                    actual_seq_lengths_kv=[1],
                    block_table=torch.zeros(1, 1, dtype=torch.int32),
                    num_heads=NUM_HEADS,
                    scale_value=HEAD_DIM**-0.5,
                    input_layout="TND",
                    num_key_value_heads=NUM_KV_HEADS,
                    block_size=BLOCK_SIZE,
                    inner_precise=2,
                )


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


class TestAdnCaptureGuard(TestBase):
    """ADN must never be captured into an ACLGraph.

    Its lengths are SymInt[] -- frozen as constants at capture time -- while
    actual_seq_lengths_kv grows every decode step, so a captured graph would
    attend over stale ranges and be silently wrong. The target being captured is
    fine and expected; only this path must stay eager.
    """

    def test_capture_is_refused(self):
        with mock_patch(f"{ADN_MOD}.is_forward_context_available", return_value=True):
            with mock_patch(f"{ADN_MOD}._EXTRA_CTX", SimpleNamespace(capturing=True)):
                with self.assertRaisesRegex(RuntimeError, "during ACLGraph capture"):
                    run_forward()

    def test_replay_and_eager_are_allowed(self):
        """capturing=False covers both eager and graph *replay*: the drafter runs
        eagerly even when the target replays a captured graph."""
        with mock_patch(f"{ADN_MOD}.is_forward_context_available", return_value=True):
            with mock_patch(f"{ADN_MOD}._EXTRA_CTX", SimpleNamespace(capturing=False)):
                _, adn, _ = run_forward()
        self.assertEqual(len(adn.calls), 1)

    def test_no_forward_context_is_not_treated_as_capturing(self):
        """Reading _EXTRA_CTX without a forward context raises, so the guard must
        short-circuit on availability -- otherwise it would break callers that
        never needed a context."""
        class _Exploding:
            # A real class, not SimpleNamespace: the latter is immutable, so
            # __getattr__ cannot be attached to it.
            def __getattr__(self, name):
                raise AssertionError(f"read _EXTRA_CTX.{name} without a forward context")

        with mock_patch(f"{ADN_MOD}.is_forward_context_available", return_value=False):
            with mock_patch(f"{ADN_MOD}._EXTRA_CTX", _Exploding()):
                _, adn, _ = run_forward()
        self.assertEqual(len(adn.calls), 1)
