# SPDX-License-Identifier: Apache-2.0
"""Whole-network check for the eager DSpark adaptive-verification lanes.

Runs the real engine end to end -- not a unit test -- and compares each eager
adaptive lane against the fixed-K baseline on the same prompts.

The invariant: with greedy sampling the target model decides every token, so
trimming the *verification* budget changes how many drafts are checked per step
but never which tokens come out. Any token-level difference is a real defect in
the trimmed layout (query boundaries, GDN state selection, logits placement),
not sampling noise. That makes exact token equality a usable gate without
needing a reference implementation.

Each lane gets a fresh engine process, run one at a time over the same cards,
because two TP=4 engines cannot share them. Every child's full log is kept, and
the aggregated ``[DSPARK-EAGER-AV/...]`` lines are echoed so the trimming
actually exercised is visible next to the verdict.

Usage on the validation server (four allocated cards, TP=4):

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export VLLM_TEST_QWEN36_MODEL=/path/to/Qwen3.6-27B
    export VLLM_TEST_DSPARK_MODEL=/path/to/DSpark
    python examples/dspark_eager_adaptive_verify.py

    # or a subset / different trimming strength
    python examples/dspark_eager_adaptive_verify.py --lanes threshold:0.4 upstream

Exit status is 0 only when every requested lane matched the baseline.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THRESHOLD_ENV = "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD"
UPSTREAM_ENV = "VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV"
LANE_ENVS = (THRESHOLD_ENV, UPSTREAM_ENV)

PROMPTS = [
    "Explain why the sky is blue.",
    "Calculate 13 times 17, showing the steps.",
    "Write a short story about a lost key.",
    "List three properties of prime numbers.",
]

RESULT_PREFIX = "EAGER_AV_RESULT="
LOG_TAG = "[DSPARK-EAGER-AV"
# Long enough for a 27B load plus generation on a cold page cache.
CHILD_TIMEOUT_S = 1800


def run_engine(args: argparse.Namespace) -> int:
    """Child mode: build one engine, generate greedily, print the token ids."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        async_scheduling=False,
        tensor_parallel_size=args.tensor_parallel_size,
        speculative_config={
            "method": "dspark",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "enforce_eager": True,
            # Off for the baseline. With it on and no lane selected the upstream
            # factory refuses to start at all: GDN is an SSM backend, so it opts
            # out of the device/CPU query-length mismatch adaptive verification
            # relies on. The baseline is therefore fixed-K without a manager,
            # which is exactly the behaviour these lanes have to preserve.
            "enable_adaptive_verification": args.adaptive,
        },
    )
    outputs = llm.generate(
        PROMPTS,
        SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=17),
    )
    print(RESULT_PREFIX + json.dumps([list(o.outputs[0].token_ids) for o in outputs]), flush=True)
    return 0


def child_env(lane: str) -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # A TP=4 engine spawns worker processes; the default fork start method
    # inherits the launcher's torch thread pool and aborts worker init with
    # "Invalid thread pool!". Spawn each worker fresh, and use the NPU allocator
    # and HCCL buffer settings the working four-card runs use.
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HCCL_BUFFSIZE", "2048")
    # Start from a clean slate so an exported lane variable cannot leak into the
    # baseline run and quietly turn this into a comparison of a lane with itself.
    for key in LANE_ENVS:
        env.pop(key, None)
    if lane == "baseline":
        pass
    elif lane.startswith("threshold:"):
        env[THRESHOLD_ENV] = lane.split(":", 1)[1]
    elif lane == "upstream":
        env[UPSTREAM_ENV] = "1"
    else:
        raise ValueError(f"unknown lane {lane!r}")
    return env


def run_lane(lane: str, args: argparse.Namespace, log_dir: Path) -> list[list[int]]:
    log_path = log_dir / f"{lane.replace(':', '-')}.log"
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--engine-child",
        "--model",
        args.model,
        "--draft",
        args.draft,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--num-speculative-tokens",
        str(args.num_speculative_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-tokens",
        str(args.max_tokens),
    ]
    if lane != "baseline":
        cmd.append("--adaptive")
    print(f"\n=== lane {lane}: starting, log -> {log_path}", flush=True)
    started = time.monotonic()
    with log_path.open("w") as stream:
        result = subprocess.run(
            cmd,
            env=child_env(lane),
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=CHILD_TIMEOUT_S,
        )
    log = log_path.read_text(errors="replace")
    elapsed = time.monotonic() - started

    observed = [line for line in log.splitlines() if LOG_TAG in line]
    for line in observed:
        print(f"    {line.strip()}", flush=True)
    # Report a crash before the lane-engaged check, so a startup failure is not
    # described as a lane that declined to engage.
    if result.returncode != 0:
        raise RuntimeError(f"lane {lane} exited {result.returncode}; tail of {log_path}:\n{log[-12000:]}")
    if lane != "baseline" and not observed:
        raise RuntimeError(
            f"lane {lane} produced no {LOG_TAG} line: the lane never engaged, so "
            f"a token match would only prove the baseline equals itself ({log_path})"
        )

    lines = [line for line in log.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise RuntimeError(f"lane {lane} printed no result line; tail of {log_path}:\n{log[-12000:]}")
    print(f"=== lane {lane}: done in {elapsed:.0f}s", flush=True)
    return json.loads(lines[-1].removeprefix(RESULT_PREFIX))


def check_devices(expected: int) -> None:
    raw = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in raw.split(",") if d.strip()]
    if len(devices) != expected or len(set(devices)) != expected:
        raise SystemExit(
            f"Set ASCEND_RT_VISIBLE_DEVICES to the {expected} allocated development cards "
            f"(got {raw!r}). The lanes run one at a time and share this same set."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--adaptive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=os.getenv("VLLM_TEST_QWEN36_MODEL"))
    parser.add_argument("--draft", default=os.getenv("VLLM_TEST_DSPARK_MODEL"))
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["threshold:0.0", "threshold:0.4", "threshold:1.0", "upstream"],
        help=(
            "Lanes to compare against the baseline. 'threshold:<x>' runs the "
            "survival-threshold lane at x; 'upstream' runs the upstream-manager "
            "lane. threshold:0.0 keeps every draft, so it is the equivalence "
            "check that isolates the new GDN path from any trimming."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--log-dir", default="")
    args = parser.parse_args()

    if not args.model or not args.draft:
        raise SystemExit("Set VLLM_TEST_QWEN36_MODEL and VLLM_TEST_DSPARK_MODEL, or pass --model/--draft")
    if args.engine_child:
        return run_engine(args)

    check_devices(args.tensor_parallel_size)
    log_dir = Path(args.log_dir or f"dspark_eager_av_{time.strftime('%Y%m%d-%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir.resolve()}")

    baseline = run_lane("baseline", args, log_dir)
    failures = []
    for lane in args.lanes:
        try:
            tokens = run_lane(lane, args, log_dir)
        except Exception as exc:  # noqa: BLE001 - report every lane, fail at the end
            failures.append(f"{lane}: {exc}")
            continue
        if tokens == baseline:
            print(f"=== lane {lane}: MATCH", flush=True)
            continue
        # Name the first divergence: the prompt and token index localize which
        # request's layout went wrong, which is where to look next.
        detail = "output count differs"
        for i, (got, want) in enumerate(zip(tokens, baseline)):
            if got == want:
                continue
            pos = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            detail = f"prompt {i} first differs at token {pos}: got {got[pos : pos + 4]} want {want[pos : pos + 4]}"
            break
        failures.append(f"{lane}: {detail}")
        print(f"=== lane {lane}: MISMATCH -- {detail}", flush=True)

    print("\n==== summary ====")
    if not failures:
        print(f"All {len(args.lanes)} lane(s) matched the fixed-K baseline.")
        return 0
    for line in failures:
        print(f"FAIL {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
