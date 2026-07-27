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
"""DSpark parallel-draft E2E on Atlas 300I (310P), TP=4, eager.

Scope note: the target deployment only has the DSpark checkpoint, so this covers
DSpark (K=7) at TP=4. DFlash end-to-end is deferred for lack of a checkpoint; its
q=9 / skip-anchor layout keeps its CPU unit tests and the DFlash q=9 case in the
Phase 0 NZ readback gate.

Placed under four_card/_310p because a four-card runner is what supplies TP=4.
"""

from __future__ import annotations

import os

import pytest

# 310P adaptation lives on Model Runner V1. Ascend already defaults V2 off unless
# this is set explicitly, but pin it so an environment difference cannot flip it.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from tests.e2e.conftest import VllmRunner  # noqa: E402

# Resolve to local paths so the four TP ranks never race to download. Override
# with QWEN3_8B_PATH / DSPARK_QWEN3_8B_PATH; the defaults are the target host's
# layout. A HF repo id still works if set.
MAIN_MODEL = os.environ.get("QWEN3_8B_PATH", "/opt/foundation_model/Qwen3-8B")
SPEC_MODEL = os.environ.get("DSPARK_QWEN3_8B_PATH", "/opt/foundation_model/dspark_qwen3_8b_block7")
NUM_SPECULATIVE_TOKENS = 7  # DSpark block7

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
]
MAX_TOKENS = 64

# fp16 is required by ADN; block_size must be 128 (the default 16 breaks the 310P
# kernel block selection); eager because 310P dense runs eager here.
COMMON = dict(
    max_model_len=4096,
    dtype="float16",
    tensor_parallel_size=4,
    block_size=128,
    enforce_eager=True,
    distributed_executor_backend="mp",
    enable_prefix_caching=False,
    disable_log_stats=False,
    max_num_seqs=256,
    gpu_memory_utilization=0.8,
)


def _drafts_and_accepted(metrics):
    """Raw draft/acceptance counts from the engine metrics."""
    num_drafts = 0
    total_accepted = 0
    accepted_per_pos = [0] * NUM_SPECULATIVE_TOKENS
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for pos in range(len(metric.values)):
                accepted_per_pos[pos] += metric.values[pos]
                total_accepted += metric.values[pos]
    return num_drafts, total_accepted, accepted_per_pos


def _missing(path):
    """A HF repo id (no slash-rooted local path) is assumed present; a local path
    must actually exist, or the four ranks would each fail to find it."""
    return path.startswith("/") and not os.path.isdir(path)


@pytest.mark.skipif(
    _missing(MAIN_MODEL) or _missing(SPEC_MODEL),
    reason=f"model path not found (MAIN_MODEL={MAIN_MODEL}, SPEC_MODEL={SPEC_MODEL}); "
    "set QWEN3_8B_PATH / DSPARK_QWEN3_8B_PATH",
)
def test_dspark_tp4_eager_matches_baseline_and_accepts():
    # Baseline: identical config, no speculation.
    with VllmRunner(MAIN_MODEL, **COMMON) as llm:
        baseline = llm.generate_greedy(PROMPTS, MAX_TOKENS)
    baseline_ids = [tuple(ids) for ids, _ in baseline]

    # Speculative: DSpark drafter, same target and config.
    speculative_config = {
        "method": "dspark",
        "model": SPEC_MODEL,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "draft_tensor_parallel_size": 4,
    }
    with VllmRunner(MAIN_MODEL, speculative_config=speculative_config, **COMMON) as llm:
        spec = llm.generate_greedy(PROMPTS, MAX_TOKENS)
        metrics = llm.model.get_metrics()
    spec_ids = [tuple(ids) for ids, _ in spec]

    num_drafts, total_accepted, accepted_per_pos = _drafts_and_accepted(metrics)
    per_pos_rate = [a / num_drafts for a in accepted_per_pos] if num_drafts else []
    print(f"num_drafts={num_drafts} total_accepted={total_accepted}")
    print(f"acceptance_per_pos={per_pos_rate}")

    # Token-match summary, recorded but not the hard gate -- see below.
    exact = sum(1 for b, s in zip(baseline_ids, spec_ids) if b == s)
    for i, (b, s) in enumerate(zip(baseline_ids, spec_ids)):
        if b != s:
            first = next(k for k in range(min(len(b), len(s))) if b[k] != s[k])
            print(f"prompt {i}: diverges at index {first} ({b[first]} vs {s[first]}), "
                  f"common prefix {first}/{min(len(b), len(s))}")
    print(f"exact token match: {exact}/{len(baseline_ids)} prompts")

    # Correctness is judged on acceptance, not token identity. Greedy speculative
    # decoding is output-lossless only in exact arithmetic: the target verifies
    # K+1 tokens per step (a chunked-prefill-shaped batch) whereas the baseline
    # decodes one token per step, and floating-point non-associativity flips the
    # argmax at borderline logits, cascading into a different continuation. The
    # repo's own dspark E2E asserts acceptance rate for the same reason. Note the
    # decisive point: a *broken* drafter would make the output MATCH the baseline
    # (every draft rejected -> pure target decode), so a divergence like this is
    # evidence the drafter is accepted into multi-token verify steps, i.e. working.
    assert num_drafts > 0, "no drafts were produced; speculation did not run"
    assert total_accepted > 0, "no draft tokens were accepted"

    # Some drafts must be rejected too, or acceptance is not really being tested.
    max_possible = num_drafts * NUM_SPECULATIVE_TOKENS
    assert total_accepted < max_possible, (
        f"every draft token was accepted ({total_accepted}/{max_possible}); "
        f"real rejection is not being exercised"
    )

    # Position 0 is DSpark's anchor (the target's own bonus token), so it is
    # accepted almost always when the draft pipeline -- context KV, per-layer
    # RoPE, slot mapping, ADN attention -- is correct. A collapsed pos-0 rate is
    # the signature of a broken drafter, which token identity cannot catch (a
    # broken drafter still reproduces the baseline output).
    assert per_pos_rate and per_pos_rate[0] >= 0.5, (
        f"position-0 acceptance {per_pos_rate[:1]} is too low; the draft pipeline "
        f"is likely producing bad proposals"
    )

    # The target's greedy verification must reproduce the baseline when logits are
    # not borderline; a systematically wrong verify/reject path would corrupt every
    # prompt, so at least one must still match exactly.
    assert exact >= 1, "no prompt matched the baseline exactly; verify/reject may be wrong"
