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
import hashlib
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
        # Match the deployment and the throughput benchmark. Left unset, vLLM
        # resolves a mode with attention splitting ops, adaptive verification
        # then forces FULL_AND_PIECEWISE on top, and the run captures a whole
        # piecewise family the ragged mode has no use for -- so the gate would
        # be measuring an engine nobody deploys.
        **({"compilation_config": {"cudagraph_mode": args.cudagraph_mode}} if args.graph else {}),
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
    prompts = build_prompts(args.num_prompts or args.max_num_seqs)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=17)
    # One result line per repeat. Loading a 27B engine at TP=4 costs about three
    # minutes and generating costs seconds, so the noise floor is far cheaper as
    # a second generate call in this engine than as a second process.
    for _ in range(max(1, args.repeats)):
        outputs = llm.generate(prompts, params)
        print(RESULT_PREFIX + json.dumps([list(o.outputs[0].token_ids) for o in outputs]), flush=True)
    return 0


def child_env(lane: str, max_model_len: int) -> dict[str, str]:
    env = os.environ.copy()
    # Price the cost table at the context this run serves, the way upstream's own
    # documentation does. Below 8192 this is a no-op: set_dummy_context clamps
    # the requested context to max_model_len - query_len, so a 2048-token gate
    # already profiled there. It starts to matter above 8192, where the upstream
    # default profiles short while the run serves long.
    env.setdefault(PROFILE_CONTEXT_ENV, str(max_model_len))
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # A gate must compile what it checks. The cache key does not capture
    # everything that decides the compiled graph -- the adaptive path rewrites
    # cudagraph_mode after the configuration is settled -- so a lane can pick up
    # an artifact built for a different one, which the graph compiler reports as
    # an unpack-count mismatch rather than a cache miss. The throughput
    # benchmark and the deployed serve configuration both set this.
    # A TP=4 engine spawns worker processes; the default fork start method
    # inherits the launcher's torch thread pool and aborts worker init with
    # "Invalid thread pool!". Spawn each worker fresh, and use the NPU allocator
    # and HCCL buffer settings the working four-card runs use.
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
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


def run_lane(
    lane: str, args: argparse.Namespace, log_dir: Path, repeats: int = 1
) -> tuple[list[list[list[int]]], bool, str | None]:
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
        "--repeats",
        str(repeats),
        "--cudagraph-mode",
        args.cudagraph_mode,
    ]
    if not lane.startswith("baseline"):
        cmd.append("--adaptive")
    # By the value, not by the key. Pinning every other lane to "none" made the
    # key always present, so this once read as true for every lane -- including
    # the baseline, which then ran with graphs instead of as the eager fixed-K
    # reference the comparison is supposed to have.
    if env_for_lane.get(AV_GRAPH_ENV, "none") != "none":
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
    runs = [json.loads(line.removeprefix(RESULT_PREFIX)) for line in lines]
    if len(runs) != repeats:
        raise RuntimeError(f"lane {lane} printed {len(runs)} result lines, expected {repeats} ({log_path})")
    trimmed = any("kept=100.0%" not in line for line in data_lines)
    frozen = frozen_confidence(data_lines)
    print(f"=== lane {lane}: done in {elapsed:.0f}s{trimming_note(lane, data_lines)}", flush=True)
    if frozen:
        print(f"    FROZEN CONFIDENCE: {frozen}", flush=True)
    return runs, trimmed, frozen


BASELINE_CACHE_DIR = Path("dspark_eager_av_baseline_cache")


def _git_state() -> tuple[str, bool] | None:
    """HEAD and whether any tracked file differs, or None outside a repo.

    Untracked files are ignored on purpose: this script drops a log directory in
    the working directory on every run, so counting those would make the cache
    permanently unusable, while what actually changes the baseline is an edit to
    a tracked file.
    """
    root = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode or status.returncode:
        return None
    return head.stdout.strip(), bool(status.stdout.strip())


def _baseline_cache_path(args: argparse.Namespace) -> Path | None:
    """Where this baseline may be cached, or None when it must not be.

    The baseline is fixed-K with no manager, so it depends on the engine
    configuration and on the code -- and on nothing about the lane under test.
    Keying on the commit means a pull re-measures it and an unchanged tree
    reuses it, which is the behaviour that makes reuse safe rather than
    convenient. A dirty tree is not cached at all: there is no key for "the
    edits I have right now".
    """
    state = _git_state()
    if state is None:
        return None
    head, dirty = state
    if dirty:
        return None
    key = json.dumps(
        [
            head,
            args.model,
            args.draft,
            args.tensor_parallel_size,
            args.num_speculative_tokens,
            args.max_model_len,
            args.max_num_seqs,
            args.max_tokens,
            args.num_prompts or args.max_num_seqs,
            # The cached value is a reference *set*, so how many passes it holds
            # and whether they came from one engine or two are part of its
            # identity, not of the run that reads it.
            args.floor_repeats,
            args.floor,
        ],
        sort_keys=True,
    )
    return BASELINE_CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def baseline_runs(args: argparse.Namespace, log_dir: Path) -> list[list[list[int]]]:
    """Every pass of the fixed-K baseline, from cache when possible.

    Two savings, both aimed at the same thing -- a 27B engine at TP=4 takes
    about three minutes to load and seconds to generate, so the cost of this
    script is the number of processes it starts, not the work they do.

    The passes are repeated ``generate`` calls in one engine rather than one
    process each, which removes a launch per pass -- so the reference set can be
    wide for almost nothing. That floor is tighter than the cross-process one: it
    exercises scheduling and reduce-order nondeterminism, but not allocator
    layout or worker init order. Tighter is the conservative direction for
    judging a lane, but it can also turn real reference noise into a lane
    failure, so ``--floor cross-process`` restores the two-process measurement
    for confirming a marginal verdict.

    The result is then cached against the commit, so iterating on lanes without
    touching the code starts one process instead of three.
    """
    cache = None if args.refresh_baseline else _baseline_cache_path(args)
    if cache is not None and cache.exists():
        try:
            passes = json.loads(cache.read_text())["passes"]
            if not isinstance(passes, list) or len(passes) < 2:
                raise ValueError("a reference set needs at least two passes")
        except (OSError, TypeError, ValueError, KeyError) as exc:
            print(f"=== baseline cache at {cache} unusable ({exc}); re-measuring", flush=True)
        else:
            print(
                f"\n=== baseline: reused from {cache} ({len(passes)} passes, same commit and "
                "configuration), skipping an engine launch. Use --refresh-baseline to re-measure.",
                flush=True,
            )
            return passes

    if args.floor == "cross-process":
        # Two processes with one pass each. This mode exists to add allocator
        # layout and worker init order to the reference's own variation, and a
        # third launch buys less of that than a third in-process pass buys.
        first, _, _ = run_lane("baseline", args, log_dir)
        second, _, _ = run_lane("baseline2", args, log_dir)
        passes = [first[0], second[0]]
    else:
        passes, _, _ = run_lane("baseline", args, log_dir, repeats=args.floor_repeats)

    target = _baseline_cache_path(args)
    if target is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"passes": passes}))
            print(f"=== baseline: cached to {target}", flush=True)
        except OSError as exc:
            print(f"=== baseline: not cached ({exc})", flush=True)
    else:
        print("=== baseline: not cached (no commit to key on, or tracked files are modified)", flush=True)
    return passes


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


def _worse(count: int, earliest: int | None, ref_count: int, ref_earliest: int | None) -> bool:
    """Whether one divergence is worse than another: earlier, or over more prompts.

    One definition, used for the floor and for every lane verdict, so a lane can
    never be judged by a rule the floor was not measured with.
    """
    if earliest is None:
        return False  # identical is never worse than anything.
    if ref_earliest is None:
        return count > 0  # the reference agreed with itself; any drift is worse.
    return earliest < ref_earliest or count > ref_count


def baseline_spread(passes: list[list[list[int]]]) -> tuple[int, int | None, str]:
    """How far the reference drifts from itself, over every pair of its passes.

    Not just the first two: with more passes the earliest-diverging pair is the
    honest floor, and anything narrower would call a lane a defect for landing
    where the reference itself lands.
    """
    worst: tuple[int, int | None, str] = (0, None, "identical")
    for i in range(len(passes)):
        for j in range(i + 1, len(passes)):
            count, earliest, detail = divergence(passes[j], passes[i])
            if _worse(count, earliest, worst[0], worst[1]):
                worst = (count, earliest, f"passes {i + 1} and {j + 1} differ -- {detail}")
    return worst


def closest(tokens: list[list[int]], passes: list[list[list[int]]]) -> tuple[int, int, int | None, str]:
    """The baseline pass this output is nearest to, and how far off it is.

    A reference that disagrees with itself has no single right answer, so the
    question exact equality can still settle is whether the lane produced one of
    the continuations the baseline itself produces. That is a stronger result
    than drifting less than the floor from whichever pass ran first, and unlike
    exact equality against one pass it stays available when the floor is not zero.
    """
    best: tuple[int, int, int | None, str] | None = None
    for idx, reference in enumerate(passes):
        count, earliest, detail = divergence(tokens, reference)
        if count == 0:
            return idx, 0, None, "identical"
        if best is None or _worse(best[1], best[2], count, earliest):
            best = (idx, count, earliest, detail)
    assert best is not None, "a reference set is never empty"
    return best


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
    parser.add_argument(
        "--cudagraph-mode",
        default="FULL_DECODE_ONLY",
        help=(
            "Applied to the graph lanes, matching the deployed serve configuration. Leaving it to "
            "vLLM captures a piecewise family as well, which the ragged mode does not use."
        ),
    )
    parser.add_argument("--graph", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--floor-repeats",
        type=int,
        default=3,
        help=(
            "Baseline generate passes in one engine, and the size of the reference set: a lane "
            "that reproduces any pass exactly is a match. Cheap -- a pass costs seconds against "
            "minutes for the load -- so a wider reference costs almost nothing. Ignored by "
            "'--floor cross-process', which is two processes of one pass."
        ),
    )
    parser.add_argument(
        "--floor",
        choices=("in-process", "cross-process"),
        default="in-process",
        help=(
            "How to measure the baseline's own noise. 'in-process' (default) generates twice "
            "in one engine, which costs one launch instead of two; it exercises scheduling and "
            "reduce-order nondeterminism but not allocator layout or worker init order, so it "
            "reads no higher than the cross-process floor and can turn real reference noise "
            "into a lane failure. Use 'cross-process' to confirm a marginal verdict."
        ),
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help=(
            "Re-measure the baseline instead of reusing the cached one. The cache is keyed on "
            "the commit and the engine configuration and is skipped entirely when tracked files "
            "are modified, so this is only needed to re-measure the same code."
        ),
    )
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

    if args.floor_repeats < 2:
        raise SystemExit("--floor-repeats must be at least 2: one pass cannot disagree with itself")
    if not args.model or not args.draft:
        raise SystemExit("Set VLLM_TEST_QWEN36_MODEL and VLLM_TEST_DSPARK_MODEL, or pass --model/--draft")
    if args.engine_child:
        return run_engine(args)

    check_devices(args.tensor_parallel_size)
    log_dir = Path(args.log_dir or f"dspark_eager_av_{time.strftime('%Y%m%d-%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir.resolve()}")

    # The baseline carries its own noise floor: without one, a mismatch at
    # concurrency cannot be attributed to a lane and a match cannot be told
    # apart from luck.
    passes = baseline_runs(args, log_dir)
    floor_count, floor_earliest, floor_detail = baseline_spread(passes)
    print(f"\n=== noise floor ({len(passes)} baseline passes): {floor_detail}", flush=True)
    if floor_count:
        print(
            "    The reference disagrees with itself here, so only a lane that drifts "
            "further than this can be called a defect.",
            flush=True,
        )

    failures = []
    inconclusive = []
    exact = []
    for lane in args.lanes:
        try:
            runs, trimmed, frozen = run_lane(lane, args, log_dir)
            tokens = runs[0]
        except Exception as exc:  # noqa: BLE001 - report every lane, fail at the end
            failures.append(f"{lane}: {exc}")
            continue
        if frozen:
            failures.append(f"{lane}: {frozen}")
            continue
        which, count, earliest, detail = closest(tokens, passes)
        bare = lane.removesuffix("+axis").removesuffix("+graph").removesuffix("+ragged").removesuffix("+ub")
        nearest = f"pass {which + 1} of {len(passes)}"
        if count == 0:
            exact.append(lane)
            print(f"=== lane {lane}: MATCH (exact against baseline {nearest})", flush=True)
        elif not _worse(count, earliest, floor_count, floor_earliest):
            print(f"=== lane {lane}: WITHIN NOISE (nearest {nearest}: {detail})", flush=True)
        else:
            detail = f"nearest {nearest}: {detail} | noise floor: {floor_detail}"
            failures.append(f"{lane}: {detail}")
            print(f"=== lane {lane}: WORSE THAN NOISE -- {detail}", flush=True)
            continue
        # threshold:0.0 is supposed to keep everything; for any other lane a
        # match without trimming means the run never exercised the path it was
        # supposed to check.
        if not trimmed and bare not in ("threshold:0.0", "baseline2"):
            inconclusive.append(lane)

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
    all_exact = len(exact) == len(args.lanes)
    qualifier = "matched" if all_exact else "stayed within the baseline's own noise against"
    print(f"All {len(args.lanes)} lane(s) {qualifier} the fixed-K baseline, and each trimming lane did trim.")
    if floor_count:
        print(f"Noise floor was non-zero: {floor_detail}")
        if all_exact:
            print(
                f"Every lane still reproduced one of the {len(passes)} baseline passes exactly. "
                "A reference that disagrees with itself cannot make exact equality mean 'the "
                "only correct output', but producing an output the baseline itself produces is "
                "a stronger result than drifting less than the floor."
            )
        else:
            print(
                "No lane reproduced a baseline pass exactly, so this run shows only that the "
                "lanes add no drift of their own. Widen the reference with --floor-repeats, "
                "which is cheap: a pass costs seconds and the engine load costs minutes."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
