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
"""DSpark on 310P with the TARGET captured in ACLGraph, drafter still eager.

Why only the target: ADN declares its sequence lengths as SymInt[]
(Ascend_Ops custom_pta/csrc/registration.cpp), so they are frozen as constants at
capture time -- and actual_seq_lengths_kv grows every decode step, which would
make a replayed graph attend over stale ranges and be silently wrong. ADN also
allocates its output per call with no out= variant. A2/A3's FIA sidesteps both by
taking the lengths as a tensor updated in place; ADN has no such overload yet.

That split needs no code to arrange: the proposer's use_cuda_graph is independent
of the runner's, and DSpark already pins its own to False
(dspark_proposer.py: "DSpark runs eager only"). The ADN adapter additionally
refuses to run under capture, so if that ever changes this fails loudly instead of
producing wrong numbers.

Mode is FULL_DECODE_ONLY, not PIECEWISE: the 310P tutorial states that is what
Atlas 300I DUO supports. It also warns that with TP > 1 the number of capturable
graphs is limited by hardware event-id resources and scales with model depth
(Qwen3-32B captures 2), which is why the capture-size list here is deliberately
tiny. If capture fails outright on this 36-layer model at TP=4, that limit is the
first thing to check.
"""

from __future__ import annotations

import os

import pytest

# 310P adaptation lives on Model Runner V1.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm.config import CompilationConfig  # noqa: E402

from tests.e2e.conftest import VllmRunner  # noqa: E402

MAIN_MODEL = os.environ.get("QWEN3_8B_PATH", "/opt/foundation_model/Qwen3-8B")
SPEC_MODEL = os.environ.get("DSPARK_QWEN3_8B_PATH", "/opt/foundation_model/dspark_qwen3_8b_block7")
NUM_SPECULATIVE_TOKENS = 7  # DSpark block7

# Bisect knobs for the 07-25 aicore fault, which first appeared on the step where
# the running batch shrank (3 reqs -> 2, graph 24 -> 16). GRAPH_E2E_NUM_PROMPTS
# shrinks the batch *and* the capture list together -- shrinking only the capture
# list makes the batch fall back to eager and proves nothing.
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
][: int(os.environ.get("GRAPH_E2E_NUM_PROMPTS", "3"))]
MAX_TOKENS = 64
# Separates "graph-size switch" from "async spec-decode batch-change bookkeeping".
ASYNC_SCHEDULING = os.environ.get("GRAPH_E2E_ASYNC_SCHEDULING", "1") == "1"

# Same as the eager E2E except enforce_eager is off. fp16 is required by ADN and
# block_size must be 128 (the default 16 breaks 310P kernel block selection).
COMMON = dict(
    max_model_len=4096,
    dtype="float16",
    tensor_parallel_size=4,
    block_size=128,
    enforce_eager=False,
    distributed_executor_backend="mp",
    enable_prefix_caching=False,
    disable_log_stats=False,
    max_num_seqs=256,
    gpu_memory_utilization=0.8,
    async_scheduling=ASYNC_SCHEDULING,
)

SPECULATIVE_CONFIG = {
    "method": "dspark",
    "model": SPEC_MODEL,
    "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    "draft_tensor_parallel_size": 4,
}


# A batch counts as uniform decode only when num_tokens == uniform_decode_query_len
# * num_reqs (gpu_model_runner: the max_num_scheduled_tokens check), and
# uniform_decode_query_len is 1 + num_spec_tokens -- 8 here, not 7. Capture sizes
# are token counts, so they must be multiples of 8; a size of 7 would never match
# and the batch would fall back to eager.
#
# 7 is right for the A2/A3 PIECEWISE test, where the drafter is captured too and
# DSpark's drafter forward is exactly K tokens per request. Here the drafter stays
# eager, so only the target's K+1 shape matters.
UNIFORM_DECODE_QUERY_LEN = 1 + NUM_SPECULATIVE_TOKENS
CAPTURE_SIZES = [UNIFORM_DECODE_QUERY_LEN * n for n in range(1, len(PROMPTS) + 1)]


def _graph_compilation_config():
    # Cover batch 1..len(PROMPTS): with all prompts in flight the batch is
    # len(PROMPTS) * 8 tokens, and if that size is not captured the run silently
    # falls back to eager and this test would prove nothing.
    #
    # At TP > 1 the 310P event-id budget caps how many graphs a model of this
    # depth can capture. If capture fails outright, shrink this list (and the
    # prompt count with it) rather than assuming the path is broken.
    return CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=CAPTURE_SIZES,
    )


def _drafts_and_accepted(metrics):
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
    return path.startswith("/") and not os.path.isdir(path)


@pytest.mark.skipif(
    _missing(MAIN_MODEL) or _missing(SPEC_MODEL),
    reason=f"model path not found (MAIN_MODEL={MAIN_MODEL}, SPEC_MODEL={SPEC_MODEL}); "
    "set QWEN3_8B_PATH / DSPARK_QWEN3_8B_PATH",
)
def test_dspark_target_in_aclgraph_drafter_eager():
    # Eager baseline for reference. Its own correctness is covered by the eager
    # E2E; here it is the comparison point for the captured run.
    eager_common = dict(COMMON, enforce_eager=True)
    with VllmRunner(MAIN_MODEL, **eager_common) as llm:
        baseline = llm.generate_greedy(PROMPTS, MAX_TOKENS)
    baseline_ids = [tuple(ids) for ids, _ in baseline]

    with VllmRunner(
        MAIN_MODEL,
        speculative_config=SPECULATIVE_CONFIG,
        compilation_config=_graph_compilation_config(),
        **COMMON,
    ) as llm:
        graphed = llm.generate_greedy(PROMPTS, MAX_TOKENS)
        metrics = llm.model.get_metrics()
    graph_ids = [tuple(ids) for ids, _ in graphed]

    num_drafts, total_accepted, accepted_per_pos = _drafts_and_accepted(metrics)
    per_pos_rate = [a / num_drafts for a in accepted_per_pos] if num_drafts else []
    print(f"num_drafts={num_drafts} total_accepted={total_accepted}")
    print(f"acceptance_per_pos={per_pos_rate}")

    exact = sum(1 for b, g in zip(baseline_ids, graph_ids) if b == g)
    for i, (b, g) in enumerate(zip(baseline_ids, graph_ids)):
        if b != g:
            first = next(k for k in range(min(len(b), len(g))) if b[k] != g[k])
            print(f"prompt {i}: diverges at index {first} ({b[first]} vs {g[first]})")
    print(f"exact token match vs eager baseline: {exact}/{len(baseline_ids)} prompts")

    # As in the eager E2E, correctness is judged on acceptance rather than token
    # identity: the target verifies K+1 tokens per step against the baseline's
    # one, and floating-point non-associativity can flip a borderline argmax.
    assert num_drafts > 0, "no drafts were produced; speculation did not run"
    assert total_accepted > 0, "no draft tokens were accepted"

    max_possible = num_drafts * NUM_SPECULATIVE_TOKENS
    assert total_accepted < max_possible, (
        f"every draft token was accepted ({total_accepted}/{max_possible}); "
        f"real rejection is not being exercised"
    )

    # Position 0 is DSpark's anchor, accepted almost always when the draft
    # pipeline is intact. A collapse here is the signature of a broken drafter --
    # including one silently corrupted by having been captured.
    assert per_pos_rate and per_pos_rate[0] >= 0.5, (
        f"position-0 acceptance {per_pos_rate[:1]} is too low; the draft pipeline "
        f"may have been affected by graph mode"
    )

    assert exact >= 1, "no prompt matched the eager baseline exactly; capture/replay may be wrong"
