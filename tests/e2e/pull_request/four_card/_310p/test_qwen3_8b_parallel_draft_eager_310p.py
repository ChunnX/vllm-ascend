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

This mirrors the A2 test (one_card/spec_decode/test_dspark.py) deliberately:
same checkpoints, same prompt through the same chat template, same max_tokens,
same acceptance helper and the same per-position tolerance. Acceptance depends
heavily on how the prompt is formatted -- the drafter is tuned on instruct-style
input, and a raw continuation prompt produces a much steeper curve -- so a
comparison against BASELINES only means anything if these match. Only what 310P
forces is allowed to differ:

* fp16 rather than the A2 default: the custom FIA kernel is fp16-only here
* block_size 128: the 310P kernel block selection covers no other value
* eager: validate_fia_scope refuses graph mode, so no cudagraph_mode=PIECEWISE
* TP=4 rather than 1, which shards heads and does not change the numerics

Scope note: the target deployment only has the DSpark checkpoint, so this covers
DSpark (K=7). DFlash end-to-end is deferred for lack of a checkpoint; its
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

from transformers import AutoTokenizer  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.v1.metrics.reader import Counter, Vector  # noqa: E402

from tests.e2e.conftest import VllmRunner  # noqa: E402
from tests.e2e.pull_request.one_card.spec_decode.utils import (  # noqa: E402
    BASELINES,
    DSPARK,
    calculate_acceptance_per_pos,
)

METHOD = "dspark"
NUM_SPECULATIVE_TOKENS = 7  # DSpark block7

# Same checkpoints as the A2 test. Override to point the four TP ranks at
# pre-fetched local copies so they do not race to download.
MAIN_MODEL = os.environ.get("QWEN3_8B_PATH", DSPARK[METHOD]["main"])
SPEC_MODEL = os.environ.get("DSPARK_QWEN3_8B_PATH", DSPARK[METHOD]["spec"])

# The A2 test's tolerance, applied one-sided. Upstream uses abs(a - b) < 0.1,
# which also fails when acceptance improves; as a port gate only the downside
# matters, and a regression is what this is here to catch.
ACCEPTANCE_TOLERANCE = 0.1


def _missing(path):
    """A HF repo id (no slash-rooted local path) is assumed present; a local path
    must actually exist, or the four ranks would each fail to find it."""
    return path.startswith("/") and not os.path.isdir(path)


@pytest.mark.skipif(
    _missing(MAIN_MODEL) or _missing(SPEC_MODEL),
    reason=f"model path not found (MAIN_MODEL={MAIN_MODEL}, SPEC_MODEL={SPEC_MODEL}); "
    "set QWEN3_8B_PATH / DSPARK_QWEN3_8B_PATH",
)
def test_dspark_tp4_eager_acceptance():
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=256)

    # Chat-formatted, as on A2. A raw continuation prompt measures something else.
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello, your name is"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    ]

    speculative_config = {
        "method": METHOD,
        "model": SPEC_MODEL,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "draft_tensor_parallel_size": 4,
    }

    with VllmRunner(
        MAIN_MODEL,
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
        speculative_config=speculative_config,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

    num_drafts = sum(m.value for m in metrics if m.name == "vllm:spec_decode_num_drafts")
    acceptance_per_pos = calculate_acceptance_per_pos(metrics, NUM_SPECULATIVE_TOKENS, Counter, Vector)
    golden = BASELINES[METHOD]
    print(f"num_drafts={num_drafts}")
    print(f"acceptance_per_pos={acceptance_per_pos}")
    print(f"golden={golden}")

    # The sample size is the denominator of every number below, and this prompt
    # hits EOS long before max_tokens, so it is small -- BASELINES is itself
    # quantised to fifths, the fingerprint of a five-draft run. Assert it rather
    # than leave it invisible: if a change makes generation stop after one step,
    # the vector collapses to 0s or 1s and would pass or fail for the wrong
    # reason. Widen the workload, not this floor, to make the numbers mean more.
    assert num_drafts >= 5, (
        f"only {num_drafts} draft steps; the per-position rates below are too "
        f"coarse to compare against anything"
    )

    # Every position is gated, not just position 0. The 310P-specific machinery
    # (context KV precompute, per-layer drafting RoPE, query slot mapping,
    # non-causal FIA over the query block) shows up as a decaying tail while
    # position 0 -- the target's own bonus token -- still looks healthy.
    low = [
        i
        for i, (rate, ref) in enumerate(zip(acceptance_per_pos, golden))
        if rate < ref - ACCEPTANCE_TOLERANCE
    ]
    assert not low, (
        f"acceptance below the A2 baseline at positions {low}: "
        f"got {[round(r, 4) for r in acceptance_per_pos]}, golden {golden}, "
        f"tolerance {ACCEPTANCE_TOLERANCE}"
    )
