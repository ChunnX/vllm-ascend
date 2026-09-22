# SPDX-License-Identifier: Apache-2.0
"""Throughput A/B for DSpark adaptive verification, shaped like PR 15098's table.

That PR reported, per draft-token count, the target model's TPS and acceptance
with and without adaptive verification, both under a full target graph. Its
numbers were +12.1% at nine draft tokens, +6.9% at seven and +2.1% at five for
Qwen3-8B, and -7.0% at seven for DeepSeek-V4-Flash. Three things follow from the
shape of that table, and this script exists to reproduce all three:

- **The draft-token count is the axis.** Trimming can only pay where there is
  something to trim, so the benefit grows with the count. Measuring a single
  count says nothing about the others, and every measurement here so far has
  been at seven.
- **The figure of merit is throughput, not per-token latency.** Adaptive
  verification trades accepted tokens per step for cheaper steps, and the
  controller's own objective is accepted tokens per unit cost -- which is
  throughput. A per-request latency benchmark measures the trade from one side.
- **Acceptance is context, not the verdict.** It falls in that PR's own data
  (4.10 to 4.01) because trimming removes drafts. A drop is the feature working;
  what would be bad is a drop without a throughput gain.

Two things are added on top, both because of what the earlier runs here showed.

**Output length is fixed** with ``ignore_eos``. Comparing TPS across runs whose
outputs differ in length compares different work: the specbench comparison that
motivated this script had the no-feature run generating 6.5% more tokens and 7%
longer sequences, which is the same order as the effect being measured. With a
fixed length every configuration emits exactly ``concurrency x output_len``
tokens, so TPS is a pure time comparison.

**Repeats happen inside one engine.** A one percent effect cannot be read off a
single run when the same configuration varies by several percent between runs,
and a 27B engine at TP=4 costs about three minutes to load against seconds to
generate. So each process loads once and times the same workload several times,
and the table carries the spread next to the mean. A mean whose spread overlaps
the other lane's is not a result.

Usage (one process per configuration, repeats inside each):

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export VLLM_TEST_QWEN36_MODEL=/path/Qwen3.6-27B
    export VLLM_TEST_DSPARK_MODEL=/path/DSpark
    python examples/dspark_adaptive_verify_throughput.py

    # one draft count, more repeats
    python examples/dspark_adaptive_verify_throughput.py -k 9 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULT_PREFIX = "AV_TPUT_RESULT="
# vLLM's periodic logger prints this once the engine has spec-decode stats. It
# is the only source of acceptance length available from outside the engine, so
# the child enables stat logging and the parent reads the line back.
ACCEPT_RE = re.compile(r"Mean acceptance length: ([\d.]+)")
DRAFT_ACCEPT_RE = re.compile(r"Avg Draft acceptance rate: ([\d.]+)%")
AV_LOG_TAG = "[DSPARK-EAGER-AV"
GRAPH_FIELD_RE = re.compile(r"graph=([A-Z_=,0-9]+)")
KEPT_RE = re.compile(r"kept=([\d.]+)%")
# A 27B load at TP=4, plus a warmup and several timed repeats.
CHILD_TIMEOUT_S = 3600

_BASE_PROMPTS = (
    "Explain why the sky is blue, in detail.",
    "Calculate 13 times 17, showing every step.",
    "Write a short story about a lost key.",
    "List three properties of prime numbers and prove one.",
    "Summarise how a four-stroke engine works.",
    "Describe the difference between a list and a tuple.",
    "Explain what a hash table is and when it degrades.",
    "Walk through solving x^2 - 5x + 6 = 0.",
)


def build_prompts(count: int) -> list[str]:
    """Deterministic, distinct prompts -- real text, not random token ids.

    Random ids would give exact input lengths, but the draft model cannot
    predict them, so confidence would be uniformly low and trimming trivially
    aggressive. That measures the wrong thing: the benefit depends on confidence
    *varying* across requests and positions. Real prompts keep that structure,
    and since every configuration gets the identical list, the varying input
    lengths are a constant of the comparison rather than a confound.
    """
    return [f"{_BASE_PROMPTS[i % len(_BASE_PROMPTS)]} (variation {i})" for i in range(max(1, count))]


def run_engine(args: argparse.Namespace) -> int:
    """Child: one engine, one configuration, several timed repeats."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.prefix_caching,
        async_scheduling=args.async_scheduling,
        tensor_parallel_size=args.tensor_parallel_size,
        compilation_config={"cudagraph_mode": args.cudagraph_mode},
        # The acceptance column comes from the periodic stat logger, which the
        # offline entrypoint disables by default.
        disable_log_stats=False,
        speculative_config={
            "method": "dspark",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "enable_adaptive_verification": args.adaptive,
        },
    )

    prompts = build_prompts(args.concurrency)
    # ignore_eos is the point: every configuration then emits exactly
    # concurrency x output_len tokens, so TPS compares time and nothing else.
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True, seed=17)

    # Discard the first pass. It carries graph capture, the cost-table
    # profiling and every lazy allocation, none of which recur -- charging them
    # to the first timed repeat would make the mean depend on the repeat count.
    llm.generate(build_prompts(min(2, args.concurrency)), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))

    runs = []
    for _ in range(max(1, args.repeats)):
        started = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - started
        emitted = sum(len(o.outputs[0].token_ids) for o in outputs)
        runs.append({"elapsed": elapsed, "output_tokens": emitted})
        print(RESULT_PREFIX + json.dumps(runs[-1]), flush=True)
    return 0


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # A TP=4 engine spawns workers; the default fork start method inherits the
    # launcher's torch thread pool and aborts worker init.
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HCCL_BUFFSIZE", "2048")
    # Long windows: this script measures throughput, and a per-window warn line
    # every few steps is itself overhead. Long enough to still see a few.
    env.setdefault("VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL", "200")
    # Clear the diagnostic lanes. The shipping path is config-driven, and a lane
    # variable left in the environment would quietly replace it -- lane A in
    # particular pays a blocking copy per step, which would be measured as the
    # feature's cost.
    for key in (
        "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD",
        "VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV",
        "VLLM_ASCEND_DSPARK_AV_GRAPH",
        "VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND",
        "VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS",
        "VLLM_ASCEND_DSPARK_AV_ADAPT",
    ):
        env.pop(key, None)
    return env


def run_config(k: int, adaptive: bool, args: argparse.Namespace, log_dir: Path) -> dict:
    lane = "dynamics" if adaptive else "fixed"
    log_path = log_dir / f"k{k}-{lane}.log"
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--engine-child",
        "--model", args.model,
        "--draft", args.draft,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--num-speculative-tokens", str(k),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--concurrency", str(args.concurrency),
        "--output-len", str(args.output_len),
        "--repeats", str(args.repeats),
        "--cudagraph-mode", args.cudagraph_mode,
    ]  # fmt: skip
    if adaptive:
        cmd.append("--adaptive")
    if args.prefix_caching:
        cmd.append("--prefix-caching")
    if args.async_scheduling:
        cmd.append("--async-scheduling")

    print(f"\n=== K={k} {lane}: starting, log -> {log_path}", flush=True)
    started = time.monotonic()
    with log_path.open("w") as stream:
        result = subprocess.run(
            cmd, env=child_env(), text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=CHILD_TIMEOUT_S
        )
    log = log_path.read_text(errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"K={k} {lane} exited {result.returncode}; tail of {log_path}:\n{log[-12000:]}")

    runs = [json.loads(line.removeprefix(RESULT_PREFIX)) for line in log.splitlines() if line.startswith(RESULT_PREFIX)]
    if not runs:
        raise RuntimeError(f"K={k} {lane} printed no timing; tail of {log_path}:\n{log[-12000:]}")
    expected = args.concurrency * args.output_len
    for run in runs:
        if run["output_tokens"] != expected:
            # Without this the whole comparison silently degrades into the one
            # the specbench runs already showed to be unreadable: different
            # configurations emitting different numbers of tokens.
            raise RuntimeError(
                f"K={k} {lane} emitted {run['output_tokens']} tokens, expected {expected}. "
                "ignore_eos did not hold, so TPS is not comparable across lanes."
            )

    tps = [run["output_tokens"] / run["elapsed"] for run in runs]
    accept = [float(m) for m in ACCEPT_RE.findall(log)]
    draft_accept = [float(m) for m in DRAFT_ACCEPT_RE.findall(log)]
    av_lines = [line for line in log.splitlines() if AV_LOG_TAG in line and " steps | " in line]
    record = {
        "k": k,
        "lane": lane,
        "tps": tps,
        "accept": statistics.fmean(accept) if accept else None,
        "draft_accept": statistics.fmean(draft_accept) if draft_accept else None,
        "graph": sorted({m for line in av_lines for m in GRAPH_FIELD_RE.findall(line)}),
        "kept": [float(m) for line in av_lines for m in KEPT_RE.findall(line)],
        "log": str(log_path),
    }
    elapsed = time.monotonic() - started
    print(
        f"=== K={k} {lane}: done in {elapsed:.0f}s | "
        f"TPS {statistics.fmean(tps):.1f}" + (f" +/- {max(tps) - min(tps):.1f}" if len(tps) > 1 else ""),
        flush=True,
    )
    if adaptive:
        # The feature has to be shown to have engaged before its number means
        # anything. A run that never trimmed verified the same width as the
        # fixed lane, so equal throughput would be a tautology.
        if not record["kept"]:
            raise RuntimeError(f"K={k} dynamics logged no budget decision; see {log_path}")
        print(
            f"    engaged: kept {min(record['kept']):.1f}-{max(record['kept']):.1f}%, graph {record['graph']}",
            flush=True,
        )
        if min(record["kept"]) >= 100.0:
            raise RuntimeError(
                f"K={k} dynamics never trimmed a draft, so its throughput is the fixed lane's "
                f"with extra bookkeeping, not a measurement of trimming; see {log_path}"
            )
    return record


def report(results: list[dict], args: argparse.Namespace) -> int:
    by_key = {(r["k"], r["lane"]): r for r in results}
    print("\n==== throughput ====")
    print(
        f"model={os.path.basename(args.model or '?')} TP={args.tensor_parallel_size} "
        f"concurrency={args.concurrency} output_len={args.output_len} "
        f"cudagraph_mode={args.cudagraph_mode} repeats={args.repeats}"
    )
    header = (
        f"{'Draft':>5} | {'Fixed TPS':>18} | {'Fixed Acc':>9} | "
        f"{'Dynamics TPS':>18} | {'Dyn Acc':>9} | {'Acceleration':>13}"
    )
    print(header)
    print("-" * len(header))
    verdicts = []
    for k in sorted({r["k"] for r in results}):
        fixed, dyn = by_key.get((k, "fixed")), by_key.get((k, "dynamics"))
        if not (fixed and dyn):
            continue
        f_mean, d_mean = statistics.fmean(fixed["tps"]), statistics.fmean(dyn["tps"])
        f_spread, d_spread = _spread(fixed["tps"]), _spread(dyn["tps"])
        gain = (d_mean - f_mean) / f_mean * 100
        # A gain smaller than the runs' own spread is not a gain. Comparing
        # against the summed spread is the crude version of a significance test,
        # and crude is the right level for three repeats.
        decisive = abs(d_mean - f_mean) > (f_spread + d_spread)
        verdicts.append((k, gain, decisive))
        print(
            f"{k:>5} | {f_mean:>10.1f} +/-{f_spread:>5.1f} | {_fmt(fixed['accept']):>9} | "
            f"{d_mean:>10.1f} +/-{d_spread:>5.1f} | {_fmt(dyn['accept']):>9} | "
            f"{gain:>+11.1f}%{'' if decisive else ' ?'}"
        )
    print(
        "\n'?' marks a difference smaller than the two lanes' own run-to-run spread. "
        "Acceptance is context: trimming removes drafts, so it is expected to fall. "
        "A fall without a throughput gain is the case to worry about -- it means the cost "
        "table is overestimating what trimming saves."
    )
    if args.repeats < 2:
        print("\nWARNING: --repeats 1 gives no spread, so no difference here can be called decisive.")
    undecided = [k for k, _, decisive in verdicts if not decisive]
    if undecided:
        print(f"Draft counts with no decisive difference: {undecided}. More repeats, or a larger concurrency.")
    return 0


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if len(values) > 1 else 0.0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.getenv("VLLM_TEST_QWEN36_MODEL"))
    parser.add_argument("--draft", default=os.getenv("VLLM_TEST_DSPARK_MODEL"))
    parser.add_argument(
        "-k",
        "--num-speculative-tokens",
        type=int,
        nargs="+",
        default=[5, 7, 9],
        help=(
            "Draft-token counts to sweep. This is the axis PR 15098 reported along, and the one "
            "the benefit depends on: trimming can only pay where there is something to trim."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Simultaneous requests. Throughput needs the batch full; the cost table is flat at low concurrency.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Tokens generated per request, enforced with ignore_eos so every lane emits the same total.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed passes per engine. Repeats are in-process: the load dominates, and one pass cannot resolve 1%%.",
    )
    parser.add_argument(
        "--cudagraph-mode",
        default="FULL_DECODE_ONLY",
        help=(
            "Applied to both lanes, so the comparison is 'Target FULL' against 'Target FULL + Dynamics' "
            "as in PR 15098 rather than a comparison of two graph configurations."
        ),
    )
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--engine-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--adaptive", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.engine_child:
        args.num_speculative_tokens = args.num_speculative_tokens[0]
        return run_engine(args)
    if not args.model or not args.draft:
        parser.error("set VLLM_TEST_QWEN36_MODEL and VLLM_TEST_DSPARK_MODEL, or pass --model/--draft")

    log_dir = Path(f"dspark_av_tput_{time.strftime('%Y%m%d-%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir.resolve()}", flush=True)

    results, failures = [], []
    for k in args.num_speculative_tokens:
        # Fixed first: if it fails, the dynamics number has nothing to compare to.
        for adaptive in (False, True):
            try:
                results.append(run_config(k, adaptive, args, log_dir))
            except Exception as exc:  # noqa: BLE001 - report every cell, fail at the end
                failures.append(f"K={k} {'dynamics' if adaptive else 'fixed'}: {exc}")
                print(f"=== K={k} {'dynamics' if adaptive else 'fixed'}: FAILED -- {exc}", flush=True)

    rc = report(results, args) if results else 1
    for line in failures:
        print(f"FAIL {line}")
    return 1 if failures else rc


if __name__ == "__main__":
    raise SystemExit(main())
