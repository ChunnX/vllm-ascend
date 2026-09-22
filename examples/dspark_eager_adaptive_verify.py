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
import re
import subprocess
import sys
import time
from pathlib import Path

THRESHOLD_ENV = "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD"
UPSTREAM_ENV = "VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV"
LANE_ENVS = (THRESHOLD_ENV, UPSTREAM_ENV)

_BASE_PROMPTS = (
    "Explain why the sky is blue.",
    "Calculate 13 times 17, showing the steps.",
    "Write a short story about a lost key.",
    "List three properties of prime numbers.",
)


def build_prompts(count: int) -> list[str]:
    """Distinct, deterministic prompts, enough to fill the requested concurrency.

    Concurrency is what puts Q where the cost table stops being flat, and with
    four prompts the batch never exceeds four requests however high
    max_num_seqs is set. The index keeps them distinct so they decode as
    separate sequences rather than sharing a prefix, and deterministic so the
    baseline and every lane compare on identical input.
    """
    return [f"{_BASE_PROMPTS[i % len(_BASE_PROMPTS)]} (variation {i})" for i in range(max(1, count))]


RESULT_PREFIX = "EAGER_AV_RESULT="
LOG_TAG = "[DSPARK-EAGER-AV"
# A lane logs a construction banner and then aggregated data lines. Only the
# data lines carry " steps | ", and only they prove the lane actually decided a
# budget: the banner just says the manager was built.
DATA_LINE_MARK = " steps | "
LOG_INTERVAL_ENV = "VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL"
CPU_UPPER_BOUND_ENV = "VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND"
AV_GRAPH_ENV = "VLLM_ASCEND_DSPARK_AV_GRAPH"
PROFILE_CONTEXT_ENV = "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN"
FIXED_AXIS_ENV = "VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS"
# Long enough for a 27B load plus generation on a cold page cache.
CHILD_TIMEOUT_S = 1800


def run_engine(args: argparse.Namespace) -> int:
    """Child mode: build one engine, generate greedily, print the token ids."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        enforce_eager=not args.graph,
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
            "enforce_eager": not args.graph,
            # Off for the baseline. With it on and no lane selected the upstream
            # factory refuses to start at all: GDN is an SSM backend, so it opts
            # out of the device/CPU query-length mismatch adaptive verification
            # relies on. The baseline is therefore fixed-K without a manager,
            # which is exactly the behaviour these lanes have to preserve.
            "enable_adaptive_verification": args.adaptive,
        },
    )
    outputs = llm.generate(
        build_prompts(args.num_prompts or args.max_num_seqs),
        SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=17),
    )
    print(RESULT_PREFIX + json.dumps([list(o.outputs[0].token_ids) for o in outputs]), flush=True)
    return 0


def child_env(lane: str, max_model_len: int) -> dict[str, str]:
    env = os.environ.copy()
    # Price the cost table at the context this run actually uses. The upstream
    # default profiles at 8192 tokens of context, and attention cost grows with
    # it, so profiling long while serving short inflates the fixed part of every
    # measurement and flattens whatever gradient Q has. A flat curve is exactly
    # the thing we are trying to tell apart from a real one.
    env.setdefault(PROFILE_CONTEXT_ENV, str(max_model_len))
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # A TP=4 engine spawns worker processes; the default fork start method
    # inherits the launcher's torch thread pool and aborts worker init with
    # "Invalid thread pool!". Spawn each worker fresh, and use the NPU allocator
    # and HCCL buffer settings the working four-card runs use.
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HCCL_BUFFSIZE", "2048")
    # These runs are a few dozen decode steps, well under the serve-oriented
    # default, so aggregate over a short window to get several data lines.
    env.setdefault(LOG_INTERVAL_ENV, "5")
    # Start from a clean slate so an exported lane variable cannot leak into the
    # baseline run and quietly turn this into a comparison of a lane with itself.
    for key in LANE_ENVS:
        env.pop(key, None)
    env.pop(CPU_UPPER_BOUND_ENV, None)
    env.pop(AV_GRAPH_ENV, None)
    env.pop(FIXED_AXIS_ENV, None)
    # "+ub" asks the lane to stop copying the trimmed boundaries back to host,
    # so only the device view is exact. That is the contract a captured graph
    # runs under, and matching the baseline under it is what lets GDN claim
    # supports_device_cpu_query_lens_mismatch.
    if lane.endswith("+ub"):
        lane = lane[: -len("+ub")]
        env[CPU_UPPER_BOUND_ENV] = "1"
    # "+graph" runs the lane under the uniform graph descriptor and "+ragged"
    # under the variable-length one. Either way a captured graph cannot read the
    # trimmed boundaries back each step, so both imply the inexact host view and
    # the engine must not be launched with enforce_eager. The difference is which
    # batches replay: under "+graph" only an untrimmed one does and a trimmed one
    # falls back to PIECEWISE, under "+ragged" a trimmed one replays too, which
    # is the whole point and also the part that has never run on device.
    # "+axis" pins the GDN request axis to max_num_seqs; "+ragged" pins it
    # anyway, so the suffix is only meaningful on the other modes. Suffix order
    # is outermost-last, so "threshold:0.4+graph+axis" reads as written.
    if lane.endswith("+axis"):
        lane = lane[: -len("+axis")]
        env[FIXED_AXIS_ENV] = "1"
    for suffix, mode in (("+graph", "uniform"), ("+ragged", "ragged")):
        if lane.endswith(suffix):
            lane = lane[: -len(suffix)]
            env[CPU_UPPER_BOUND_ENV] = "1"
            env[AV_GRAPH_ENV] = mode
            break
    else:
        # Pin every other lane to eager explicitly. Unset now means ragged --
        # that is the shipping default -- so leaving it out would silently give
        # a bare lane the graph, and the suffix would stop meaning anything.
        env[AV_GRAPH_ENV] = "none"
    if lane in ("baseline", "baseline2"):
        pass
    elif lane.startswith("threshold:"):
        env[THRESHOLD_ENV] = lane.split(":", 1)[1]
    elif lane == "upstream":
        env[UPSTREAM_ENV] = "1"
    else:
        raise ValueError(f"unknown lane {lane!r}")
    return env


def run_lane(lane: str, args: argparse.Namespace, log_dir: Path) -> tuple[list[list[int]], bool, str | None]:
    log_path = log_dir / f"{lane.replace(':', '-')}.log"
    env_for_lane = child_env(lane, args.max_model_len)
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
        "--num-prompts",
        str(args.num_prompts or args.max_num_seqs),
    ]
    if not lane.startswith("baseline"):
        cmd.append("--adaptive")
    if AV_GRAPH_ENV in env_for_lane:
        cmd.append("--graph")
    print(f"\n=== lane {lane}: starting, log -> {log_path}", flush=True)
    started = time.monotonic()
    with log_path.open("w") as stream:
        result = subprocess.run(
            cmd,
            env=env_for_lane,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=CHILD_TIMEOUT_S,
        )
    log = log_path.read_text(errors="replace")
    elapsed = time.monotonic() - started

    tagged = [line for line in log.splitlines() if LOG_TAG in line]
    data_lines = [line for line in tagged if DATA_LINE_MARK in line]
    # Everything the lane says that is not a per-window data line -- the banner,
    # the measured cost table -- plus the last few windows. Filtering to data
    # lines alone silently dropped the cost table, which was the whole point of
    # the run that produced it. One rank's view is enough: TP ranks choose
    # identical capacities by construction, so dedup on the message itself.
    seen: set[str] = set()
    notes = []
    for line in tagged:
        if DATA_LINE_MARK in line:
            continue
        message = line[line.index(LOG_TAG) :]
        if message not in seen:
            seen.add(message)
            notes.append(line)
    for line in notes + data_lines[-4:]:
        print(f"    {line.strip()}", flush=True)
    # Report a crash before the lane-engaged check, so a startup failure is not
    # described as a lane that declined to engage.
    if result.returncode != 0:
        raise RuntimeError(f"lane {lane} exited {result.returncode}; tail of {log_path}:\n{log[-12000:]}")
    if not lane.startswith("baseline") and not data_lines:
        # The construction banner also carries LOG_TAG, so requiring only the tag
        # would accept a lane that was built and then never asked for a budget.
        raise RuntimeError(
            f"lane {lane} logged no {DATA_LINE_MARK.strip()!r} line, so it never decided a "
            f"budget: a token match would only prove the baseline equals itself ({log_path})"
        )

    lines = [line for line in log.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise RuntimeError(f"lane {lane} printed no result line; tail of {log_path}:\n{log[-12000:]}")
    trimmed = any("kept=100.0%" not in line for line in data_lines)
    frozen = frozen_confidence(data_lines)
    print(f"=== lane {lane}: done in {elapsed:.0f}s{trimming_note(lane, data_lines)}", flush=True)
    if frozen:
        print(f"    FROZEN CONFIDENCE: {frozen}", flush=True)
    return json.loads(lines[-1].removeprefix(RESULT_PREFIX)), trimmed, frozen


def frozen_confidence(data_lines: list[str]) -> str | None:
    """Detect a confidence buffer that stopped moving.

    Under graph the confidence op is only recomputed per replay if it was traced
    into the captured draft graph; left out, the buffer freezes and every budget
    is decided on pre-capture numbers. Output stays byte-identical to the
    baseline, because trimming is only a policy -- so this is invisible to every
    other check in this script, and a lane can pass the whole gate while its
    adaptive verification is dead.

    ``conf_moved=n/N`` is per window: N steps ran and the confidence differed
    from the previous step on n of them. A live op moves on essentially every
    step; a frozen buffer moves on none after the first. Only windows with at
    least two steps can tell the two apart.
    """
    seen = [
        (int(m.group(1)), int(m.group(2))) for line in data_lines if (m := re.search(r"conf_moved=(\d+)/(\d+)", line))
    ]
    # Allow the one move a frozen buffer still shows on its very first step.
    usable = [(moved, steps) for moved, steps in seen if steps >= 2]
    if not usable or any(moved > 1 for moved, _ in usable):
        return None
    steps = sum(steps for _, steps in usable)
    return (
        f"{len(usable)} window(s) covering {steps} steps saw the confidence change at most "
        "once, so the signal is not being recomputed. Under graph this is the confidence op "
        "missing from the captured draft graph; the budget is then decided on pre-capture "
        "numbers and a token match proves nothing about adaptive verification."
    )


def trimming_note(lane: str, data_lines: list[str]) -> str:
    """Say whether this lane ever trimmed, because a match otherwise proves little.

    A lane that kept every draft ran the same verification width as the baseline,
    so equal output says nothing about the trimmed layout. threshold:0.0 is meant
    to keep everything -- that is its job as the equivalence check -- but for any
    other lane this is the difference between evidence and a tautology.
    """
    if lane.startswith("baseline") or not data_lines:
        return ""
    trimmed = sum(1 for line in data_lines if "kept=100.0%" not in line)
    if trimmed:
        return f", trimmed in {trimmed}/{len(data_lines)} reported windows"
    expected = (
        lane.removesuffix("+axis").removesuffix("+graph").removesuffix("+ragged").removesuffix("+ub") == "threshold:0.0"
    )
    return ", kept every draft" + ("" if expected else " -- THIS MATCH IS NOT EVIDENCE")


def divergence(tokens: list[list[int]], reference: list[list[int]]) -> tuple[int, int | None, str]:
    """How far two runs of the same prompts drift apart.

    Token equality is a correctness instrument only where the reference is
    deterministic. It is not at higher concurrency: requests finish at different
    steps, the batch shrinks and is re-sorted around them, and a kernel that
    picks its tiling or reduction order from the batch shape rounds differently,
    which flips a greedy argmax wherever two logits are near-tied. That is
    numerical noise, not a defect, so a lane has to be judged against how far the
    reference drifts from itself rather than against exact equality.
    """
    if len(tokens) != len(reference):
        return len(reference), 0, "output count differs"
    differing = [i for i, (got, want) in enumerate(zip(tokens, reference)) if got != want]
    if not differing:
        return 0, None, "identical"

    def first_diff(i: int) -> int:
        got, want = tokens[i], reference[i]
        return next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))

    head = differing[0]
    at = first_diff(head)
    earliest = min(first_diff(i) for i in differing)
    got, want = tokens[head], reference[head]
    detail = (
        f"{len(differing)}/{len(reference)} prompts differ, earliest at token {earliest}; "
        f"prompt {head} first differs at token {at}: got {got[at : at + 4]} want {want[at : at + 4]}"
    )
    return len(differing), earliest, detail


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
    parser.add_argument("--graph", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=os.getenv("VLLM_TEST_QWEN36_MODEL"))
    parser.add_argument("--draft", default=os.getenv("VLLM_TEST_DSPARK_MODEL"))
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["upstream", "upstream+ub", "upstream+graph", "upstream+ragged"],
        help=(
            "Lanes to compare against the baseline. 'upstream' runs the "
            "upstream manager -- the real cost-argmax budget, device survival "
            "top-k and measured cost table -- which is the path that would ship. "
            "'threshold:<x>' runs the survival-threshold lane, kept as a bisect "
            "tool: it isolated the GDN path in eager and its exact host "
            "boundaries are the opposite of what a graph needs. Suffixes "
            "compose, outermost last: '+ub' leaves the host view inexact (the "
            "contract a captured graph runs under), '+graph' adds the uniform "
            "graph descriptor and '+ragged' the variable-length one, so a "
            "trimmed batch replays instead of falling back, '+axis' pins the "
            "GDN request axis to max_num_seqs ('+ragged' pins it anyway) "
            "-- e.g. 'upstream+graph+axis'. The lane 'baseline2' runs the "
            "baseline a second time and compares it to the first, which checks "
            "that the comparison itself is reproducible before any mismatch is "
            "attributed to a lane."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help=(
            "How many concurrent requests to submit. 0 (default) follows "
            "--max-num-seqs, so raising the concurrency raises Q, which is what "
            "moves the cost table off flat."
        ),
    )
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

    baseline, _, _ = run_lane("baseline", args, log_dir)

    # The noise floor. Without it a mismatch at concurrency cannot be attributed
    # to a lane, and a match cannot be told apart from luck.
    second, _, _ = run_lane("baseline2", args, log_dir)
    floor_count, floor_earliest, floor_detail = divergence(second, baseline)
    print(f"\n=== noise floor (baseline vs baseline): {floor_detail}", flush=True)
    if floor_count:
        print(
            "    The reference disagrees with itself here, so only a lane that drifts "
            "further than this can be called a defect.",
            flush=True,
        )

    failures = []
    inconclusive = []
    for lane in args.lanes:
        try:
            tokens, trimmed, frozen = run_lane(lane, args, log_dir)
        except Exception as exc:  # noqa: BLE001 - report every lane, fail at the end
            failures.append(f"{lane}: {exc}")
            continue
        if frozen:
            failures.append(f"{lane}: {frozen}")
            continue
        count, earliest, detail = divergence(tokens, baseline)
        bare = lane.removesuffix("+axis").removesuffix("+graph").removesuffix("+ragged").removesuffix("+ub")
        no_worse = count <= floor_count and (earliest is None or floor_earliest is None or earliest >= floor_earliest)
        if no_worse:
            print(f"=== lane {lane}: {'MATCH' if count == 0 else f'WITHIN NOISE ({detail})'}", flush=True)
            # threshold:0.0 is supposed to keep everything; for any other lane a
            # match without trimming means the run never exercised the path it
            # was supposed to check.
            if not trimmed and bare not in ("threshold:0.0", "baseline2"):
                inconclusive.append(lane)
            continue
        detail = f"{detail} | noise floor: {floor_detail}"
        failures.append(f"{lane}: {detail}")
        print(f"=== lane {lane}: WORSE THAN NOISE -- {detail}", flush=True)

    print("\n==== summary ====")
    for line in failures:
        print(f"FAIL {line}")
    for lane in inconclusive:
        print(
            f"INCONCLUSIVE {lane}: matched the baseline but never trimmed a draft, so it "
            "verified the same width the baseline ran. Raise the threshold, lengthen the "
            "run, or check that confidence reaches the manager."
        )
    if failures or inconclusive:
        return 1
    qualifier = "matched" if not floor_count else "stayed within the baseline's own noise against"
    print(f"All {len(args.lanes)} lane(s) {qualifier} the fixed-K baseline, and each trimming lane did trim.")
    if floor_count:
        print(f"Noise floor was non-zero: {floor_detail}")
        print(
            "A non-zero floor means this configuration cannot prove exact equality; it can "
            "only show a lane adds no drift of its own. Use a concurrency where the floor "
            "is zero for the correctness claim."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
