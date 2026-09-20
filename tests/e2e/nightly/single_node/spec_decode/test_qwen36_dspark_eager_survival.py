# SPDX-License-Identifier: Apache-2.0
"""Opt-in 910B4 TP=4 model gate; each mode owns a fresh engine process and state cache."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("threshold", [0.0, 0.4, 1.0])
def test_greedy_eager_survival_matches_fixed_k(threshold, tmp_path):
    model = os.getenv("VLLM_TEST_QWEN36_MODEL")
    draft = os.getenv("VLLM_TEST_DSPARK_MODEL")
    if not model or not draft:
        pytest.skip("Set VLLM_TEST_QWEN36_MODEL and VLLM_TEST_DSPARK_MODEL to local checkpoints")
    devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    assert len(devices) == 4 and all(d.strip() for d in devices) and len(set(devices)) == 4, (
        "Set ASCEND_RT_VISIBLE_DEVICES to the four allocated development cards before this TP=4 test"
    )
    program = r"""
import json,sys
from vllm import LLM, SamplingParams
model,draft,adaptive=json.loads(sys.argv[1])
llm=LLM(model=model, enforce_eager=True, dtype="bfloat16", max_model_len=2048,
        max_num_seqs=4, enable_prefix_caching=False, async_scheduling=False,
        tensor_parallel_size=4,
        speculative_config={"method":"dspark","model":draft,"num_speculative_tokens":7,
                            "enforce_eager":True,"enable_adaptive_verification":adaptive})
prompts=["Explain why the sky is blue.", "Calculate 13 times 17, showing the steps.",
         "Write a short story about a lost key.", "List three properties of prime numbers."]
outputs=llm.generate(prompts, SamplingParams(temperature=0, max_tokens=48, seed=17))
print("EAGER_AV_RESULT="+json.dumps([list(o.outputs[0].token_ids) for o in outputs]))
"""

    def run(adaptive):
        env = os.environ.copy()
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        key = "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD"
        env.pop(key, None)
        if adaptive:
            env[key] = str(threshold)
        log_path = tmp_path / ("adaptive.log" if adaptive else "fixed.log")
        with log_path.open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-c", program, json.dumps([model, draft, adaptive])],
                env=env,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        log = log_path.read_text(errors="replace")
        print(f"Model process log: {log_path}")
        assert result.returncode == 0, f"{log_path}\n{log[-12000:]}"
        if adaptive:
            assert "DSpark eager survival verification active" in log
        line = next(line for line in log.splitlines() if line.startswith("EAGER_AV_RESULT="))
        return json.loads(line.split("=", 1)[1])

    assert run(True) == run(False)
