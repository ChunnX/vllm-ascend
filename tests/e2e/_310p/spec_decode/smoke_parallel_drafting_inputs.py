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

"""Phase 0 hardware gate: the Triton-free input expansion must run on 310P NPU.

The CPU unit tests already prove the helper is equivalent to the Triton kernel.
What they cannot prove is that every tensor op it uses -- advanced indexing with
a computed index tensor, gather on a rank-2 block table, integer division and
modulo, broadcast, flatten, column assignment -- is actually supported on 310P.
This runs the same helper on device and against CPU and compares.

Deliberately not a pytest test: it needs real 310P hardware. Run it by hand:

    python tests/e2e/_310p/spec_decode/smoke_parallel_drafting_inputs.py

If an op turns out to be unsupported, stop and implement a small AscendC helper.
Falling back to CPU with a device sync per step is a debugging aid, never the
production path.
"""

import sys

import torch

from vllm_ascend._310p.spec_decode.parallel_drafting_inputs import expand_parallel_drafting_inputs

MASK_ID = 151666
BLOCK_SIZE = 128
DEVICE = "npu"

FAILURES = []


def require_env():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        sys.exit("torch_npu is unavailable: this script must run on the 310P host.")
    if not torch.npu.is_available():  # type: ignore[attr-defined]
        sys.exit("no NPU device visible; this gate must run on real hardware.")


def build_inputs(*, ctx_lens, seq_lens, rejected, num_query_per_req, num_spec):
    """Mirror the fixtures the CPU unit tests use.

    ctx_lens is this round's scheduled segment; seq_lens is the total KV length.
    Positions are absolute, so a request with 257 total KV that scheduled 4
    tokens carries positions [253..256].
    """
    batch = len(ctx_lens)
    total = sum(ctx_lens)

    qsl = torch.zeros(batch + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(ctx_lens, dtype=torch.int32).cumsum(0)
    positions = torch.cat(
        [torch.arange(n_seq - n_ctx, n_seq, dtype=torch.int32) for n_ctx, n_seq in zip(ctx_lens, seq_lens)]
    )
    max_blocks = (max(seq_lens) + num_query_per_req) // BLOCK_SIZE + 2
    # Shifted past the logical index range so no logical page maps to itself;
    # otherwise a helper ignoring the block table would still look correct.
    block_table = torch.arange(batch * max_blocks, dtype=torch.int32).flip(0).reshape(batch, max_blocks) + max_blocks

    return dict(
        next_token_ids=torch.arange(batch, dtype=torch.int32) + 1000,
        target_positions=positions,
        context_slot_mapping=torch.arange(total, dtype=torch.int32) * 3 + 7,
        block_table=block_table,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_rejected_tokens=(torch.tensor(rejected, dtype=torch.int32) if rejected is not None else None),
        parallel_drafting_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_spec,
        total_input_tokens=total,
        batch_size=batch,
    )


def make_outputs(batch, num_query_per_req, num_spec, total, device):
    n_query = batch * num_query_per_req
    return dict(
        out_input_ids=torch.zeros(n_query, dtype=torch.int32, device=device),
        out_context_positions=torch.zeros(total, dtype=torch.int32, device=device),
        out_query_positions=torch.zeros(n_query, dtype=torch.int32, device=device),
        out_context_slot_mapping=torch.zeros(total, dtype=torch.int32, device=device),
        out_query_slot_mapping=torch.zeros(n_query, dtype=torch.int32, device=device),
        out_token_indices=torch.zeros(batch * num_spec, dtype=torch.int32, device=device),
    )


def to_device(inputs, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def run_case(name, *, ctx_lens, seq_lens, rejected, num_query_per_req, num_spec, sample_from_anchor):
    inputs = build_inputs(
        ctx_lens=ctx_lens,
        seq_lens=seq_lens,
        rejected=rejected,
        num_query_per_req=num_query_per_req,
        num_spec=num_spec,
    )
    batch, total = len(ctx_lens), sum(ctx_lens)

    cpu_out = make_outputs(batch, num_query_per_req, num_spec, total, "cpu")
    npu_out = make_outputs(batch, num_query_per_req, num_spec, total, DEVICE)

    try:
        expand_parallel_drafting_inputs(**inputs, **cpu_out, sample_from_anchor=sample_from_anchor)
        expand_parallel_drafting_inputs(
            **to_device(inputs, DEVICE), **npu_out, sample_from_anchor=sample_from_anchor
        )
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        FAILURES.append(name)
        return

    mismatched = []
    for key in cpu_out:
        if not torch.equal(npu_out[key].cpu(), cpu_out[key]):
            mismatched.append(key)
    if mismatched:
        print(f"[FAIL] {name}: NPU and CPU disagree on {mismatched}")
        for key in mismatched:
            print(f"    npu[{key}] = {npu_out[key].cpu().tolist()}")
            print(f"    cpu[{key}] = {cpu_out[key].tolist()}")
        FAILURES.append(name)
    else:
        print(f"[PASS] {name}")


def main():
    require_env()
    print("Comparing the 310P helper on NPU against the same helper on CPU.\n")

    run_case(
        "DFlash q=9, ragged segment, distinct seq_lens",
        ctx_lens=[1, 4, 2],
        seq_lens=[257, 134, 66],
        rejected=None,
        num_query_per_req=9,
        num_spec=8,
        sample_from_anchor=False,
    )
    run_case(
        "DSpark q=7, ragged segment, distinct seq_lens",
        ctx_lens=[1, 4, 2],
        seq_lens=[257, 134, 66],
        rejected=None,
        num_query_per_req=7,
        num_spec=7,
        sample_from_anchor=True,
    )
    run_case(
        "rejected tail (advanced indexing with a computed index)",
        ctx_lens=[1, 4, 2],
        seq_lens=[257, 134, 66],
        rejected=[0, 3, 1],
        num_query_per_req=9,
        num_spec=8,
        sample_from_anchor=False,
    )
    for seq_len in (127, 128, 129):
        run_case(
            f"query slots crossing a page boundary (seq_len={seq_len})",
            ctx_lens=[2],
            seq_lens=[seq_len],
            rejected=None,
            num_query_per_req=9,
            num_spec=8,
            sample_from_anchor=False,
        )

    print()
    if FAILURES:
        print("FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        print(
            "\nAn exception means the op is unsupported on 310P: implement a small AscendC "
            "helper instead. A silent mismatch means the op behaves differently there, which "
            "is worse -- find which one before going further."
        )
        sys.exit(1)
    print("PASS: every op the helper needs works on 310P and agrees with CPU.")


if __name__ == "__main__":
    main()
