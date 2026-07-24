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

"""Phase 0 hardware gate: ADN must read vLLM's NZ paged cache directly.

The whole 310P adapter is built on one assumption -- that ADN can consume the
cache vLLM already allocated and wrote, with no gather, no NZ->ND conversion and
no second copy. This script is what makes that assumption a fact. If it fails,
the adapter needs rewriting, so run it before anything downstream.

Deliberately not a pytest test: it needs real 310P hardware plus the ADN custom
OPP and PTA wheels, which CI hosts do not have. Run it by hand:

    python tests/e2e/_310p/adn/smoke_adn_nz_readback.py

Everything here mirrors production rather than convenience:

* two separate rank-4 caches from ``torch_npu.empty_with_format``, exactly as
  ``_310p/model_runner_310p.py:845-850`` allocates them. Allocating one 5D
  tensor and slicing it gives different storage descriptors, which is precisely
  what this gate is meant to check.
* ``DeviceOperator.reshape_and_cache`` for the writes, so the dispatch is the
  one production uses.
* ``scale = head_dim ** -0.5``. Note the ATK cases use ``1 / head_dim``
  (``atk_test/fia_common.py:508``); they are self-consistent but exercise a
  numeric range production never sees.
"""

import sys

import torch

FAILURES = []


def require_env():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        sys.exit("torch_npu is unavailable: this script must run on the 310P host.")
    try:
        import adn_custom_ops  # noqa: F401
    except Exception as exc:
        sys.exit(
            f"adn_custom_ops could not be imported ({exc}).\n"
            "Build and install the Ascend_Ops custom_opp and PTA wheels first; see "
            "Ascend_Ops/AGENTS.md. This gate cannot be skipped -- the adapter has no "
            "fallback path."
        )


NUM_HEADS = 16  # Qwen3-8B at TP=2
NUM_KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
SCALE = HEAD_DIM**-0.5
DEVICE = "npu"
DTYPE = torch.float16

# Tolerance placeholder. Replace with the value Phase 0.2 read out of the ATK
# result JSON or framework config; the YAML only says `standard: acc: default`.
ATOL = 5e-3
RTOL = 5e-3


def allocate_production_caches(num_blocks):
    """Allocate exactly as the 310P model runner does."""
    import torch_npu

    from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    full_shape = AscendAttentionBackend310.get_kv_cache_shape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    per_cache_shape = full_shape[1:]  # drop the leading 2; K and V are separate tensors
    key_cache = torch_npu.empty_with_format(
        size=per_cache_shape, dtype=DTYPE, device=DEVICE, acl_format=ACL_FORMAT_FRACTAL_NZ
    )
    value_cache = torch_npu.empty_with_format(
        size=per_cache_shape, dtype=DTYPE, device=DEVICE, acl_format=ACL_FORMAT_FRACTAL_NZ
    )
    key_cache.zero_()
    value_cache.zero_()
    return key_cache, value_cache


def shuffled_block_table(batch, pages_per_req, num_blocks):
    """Non-contiguous, out-of-order physical pages.

    If logical page i mapped to physical page i, a kernel that ignored the block
    table entirely would still produce the right answer and this gate would pass
    while proving nothing.
    """
    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(num_blocks, generator=generator)[: batch * pages_per_req]
    table = perm.reshape(batch, pages_per_req).to(torch.int32)
    identity = torch.arange(pages_per_req, dtype=torch.int32)
    assert bool((table != identity).all()), "block table has a fixed point; regenerate with another seed"
    return table


def write_cache(key_cache, value_cache, key_nd, value_nd, kv_lens, block_table_cpu):
    """Fill the caches through the production writer.

    Slots are computed from a CPU mirror of the block table: reading page ids
    back from the device one at a time would sync on every token.
    """
    from vllm_ascend.device.device_op import DeviceOperator

    for b, kv_len in enumerate(kv_lens):
        positions = torch.arange(kv_len, dtype=torch.int64)
        pages = block_table_cpu[b, positions // BLOCK_SIZE].to(torch.int64)
        slots = (pages * BLOCK_SIZE + positions % BLOCK_SIZE).to(torch.int32).to(DEVICE)
        DeviceOperator.reshape_and_cache(
            key_nd[b, :kv_len],
            value_nd[b, :kv_len],
            key_cache,
            value_cache,
            slots,
        )


def golden(query, key_nd, value_nd, q_lens, kv_lens, *, causal=False):
    """softmax(QK^T * scale) V in fp32.

    With causal=True the query block is treated as sitting at the end of the KV
    range, so query row i sees keys up to kv_len - q_len + i. That variant exists
    only to show the non-causal expectation is not vacuous.
    """
    outputs = []
    q_offset = 0
    repeats = NUM_HEADS // NUM_KV_HEADS
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        q = query[q_offset : q_offset + q_len].float()
        k = key_nd[b, :kv_len].float().repeat_interleave(repeats, dim=1)
        v = value_nd[b, :kv_len].float().repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * SCALE
        if causal:
            rows = torch.arange(q_len).unsqueeze(1)
            cols = torch.arange(kv_len).unsqueeze(0)
            blocked = cols > (kv_len - q_len + rows)
            scores = scores.masked_fill(blocked.unsqueeze(0), float("-inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v))
        q_offset += q_len
    return torch.cat(outputs, dim=0)


def run_case(name, q_lens, kv_lens, *, value_builder=None):
    import adn_custom_ops

    batch = len(q_lens)
    max_kv = max(kv_lens)
    pages_per_req = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max(8, batch * pages_per_req + 4)

    block_table_cpu = shuffled_block_table(batch, pages_per_req, num_blocks)
    block_table = block_table_cpu.to(DEVICE)

    generator = torch.Generator().manual_seed(1234)
    key_nd = torch.randn(batch, max_kv, NUM_KV_HEADS, HEAD_DIM, generator=generator).to(DTYPE)
    value_nd = (
        value_builder(batch, max_kv, kv_lens)
        if value_builder is not None
        else torch.randn(batch, max_kv, NUM_KV_HEADS, HEAD_DIM, generator=generator).to(DTYPE)
    )
    query = torch.randn(sum(q_lens), NUM_HEADS, HEAD_DIM, generator=generator).to(DTYPE)

    key_cache, value_cache = allocate_production_caches(num_blocks)
    write_cache(key_cache, value_cache, key_nd.to(DEVICE), value_nd.to(DEVICE), kv_lens, block_table_cpu)

    out = adn_custom_ops.adn_fused_infer_attention(
        query=query.to(DEVICE),
        key=key_cache,
        value=value_cache,
        attn_mask=None,
        actual_seq_lengths_q=list(q_lens),
        actual_seq_lengths_kv=list(kv_lens),
        block_table=block_table,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE,
        input_layout="TND",
        scale_value=SCALE,
        inner_precise=2,
        force_call=False,
    )

    reference = golden(query, key_nd, value_nd, q_lens, kv_lens)
    actual = out.cpu().float()
    diff = (actual - reference).abs()
    max_err, mean_err = diff.max().item(), diff.mean().item()
    ok = torch.allclose(actual, reference, atol=ATOL, rtol=RTOL)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_abs={max_err:.6f} mean_abs={mean_err:.6f}")
    if not ok:
        FAILURES.append(name)
    return dict(actual=actual, reference=reference, query=query, key_nd=key_nd, value_nd=value_nd)


def future_token_dominance():
    """Prove the result is genuinely non-causal, not merely plausible.

    Only the final KV position carries signal. Under full attention every query
    row picks it up; under causal attention the earliest row cannot see it at
    all. Rather than thresholding the magnitude -- which depends on how much
    softmax weight the position happens to attract -- compare against both
    goldens: the output must match the non-causal one and must not match the
    causal one. The second half is what makes the first half meaningful.
    """
    kv_len = 200
    q_lens, kv_lens = [9], [kv_len]

    def build_values(batch, max_kv, _kv_lens):
        values = torch.zeros(batch, max_kv, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
        values[0, kv_len - 1] = 100.0  # only the final position carries signal
        return values

    case = run_case("future-token dominance", q_lens, kv_lens, value_builder=build_values)

    causal_reference = golden(case["query"], case["key_nd"], case["value_nd"], q_lens, kv_lens, causal=True)
    first_row_gap = (case["reference"][0] - causal_reference[0]).abs().max().item()
    print(f"    causal vs non-causal differ on the first query row by {first_row_gap:.4f}")
    if first_row_gap <= ATOL:
        FAILURES.append(
            "future-token dominance: the two goldens agree, so this case cannot tell "
            "causal from non-causal -- fix the fixture before trusting it"
        )
        return

    if torch.allclose(case["actual"], causal_reference, atol=ATOL, rtol=RTOL):
        FAILURES.append("future-token dominance: ADN output matches the CAUSAL golden")


def main():
    require_env()
    print(f"scale = {SCALE:.8f}  (ATK uses 1/head_dim = {1.0 / HEAD_DIM:.8f}; production uses this one)")
    print(f"tolerance atol={ATOL} rtol={RTOL}  <-- replace with the value Phase 0.2 recorded\n")

    # DFlash queries K+1 = 9 per request, DSpark K = 7. KV lengths span one, two
    # and three pages, with the last two straddling a page boundary.
    run_case("DFlash q=9, single request, 2 pages", [9], [200])
    run_case("DFlash q=9, ragged batch, 1/2/3 pages", [9, 9, 9], [65, 133, 257])
    run_case("DSpark q=7, ragged batch, 1/2/3 pages", [7, 7, 7], [65, 133, 257])
    run_case("page-boundary KV lengths", [9, 9, 9], [127, 128, 129])
    future_token_dominance()

    print()
    if FAILURES:
        print("FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        print(
            "\nDiagnose in this order: a descriptor/shape error means the NZ allocation or "
            "writer layout disagrees with ADN; correct shapes but wrong numbers usually "
            "means num_key_value_heads, block_size or scale. Do not add a repack to the "
            "hot path to work around it."
        )
        sys.exit(1)
    print("PASS: ADN reads vLLM's NZ cache directly, and the result is non-causal.")


if __name__ == "__main__":
    main()
