# SPDX-License-Identifier: Apache-2.0
"""Throughput A/B for DSpark adaptive verification, shaped like PR 15098's table.

That PR reported, per draft-token count, the target model's TPS and acceptance
with and without adaptive verification, both under a full target graph. Its
numbers were +12.1% at nine draft tokens, +6.9% at seven and +2.1% at five for
Qwen3-8B, and -7.0% at seven for DeepSeek-V4-Flash. Three things follow from the
shape of that table, and this script exists to reproduce all three:

- **The benefit grows with how much there is to trim.** That PR varied the draft
  count to show it. We cannot: ``speculative.py`` requires
  ``num_speculative_tokens`` to *equal* the draft checkpoint's trained
  ``block_size``, so a block-7 model makes every other count fail config
  validation rather than run slower. That PR's three rows are three
  checkpoints. The axis we do have is **concurrency**, which moves the same
  quantity: the reachable spread of the cost table was 12.31ms at four requests
  and 29.97ms at sixteen, and trimming can only earn what that spread contains.
  But ``max_num_seqs`` is not a load knob -- it sets the captured bucket set,
  the graph limit, that spread and the pinned GDN axis together, so two values
  are two graph configurations rather than one under two loads. It therefore
  defaults to a single value, the deployment point, and a sweep is opt-in.
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

PR 15147, the D-Cut author's own Ascend implementation, states its method where
15098 does not: "D-Cut runs in PIECEWISE mode at concurrency 64, using 500
requests per dataset and a two-run average", reporting +8.5% throughput on
Math500 and +19.1% on Dolly. Three things taken from that:

- **Named datasets, not synthetic text.** The benefit depends on confidence
  varying across requests, and a handful of repeated prompts understates that
  spread. ``--dataset`` reads Math500 (``problem``) or Dolly
  (``instruction`` + ``context``, joined). Expect Dolly to show more: that PR
  measured +19.1% on it against +8.5% on Math500, and the ordering follows from
  acceptance -- maths drafts are accepted well, so there is less to trim.
- **Workload size is not concurrency.** 500 requests through a 64-wide engine is
  sustained throughput; sending as many requests as the batch is wide measures
  one wave. So ``--num-requests`` (500, as there) is separate from capacity.
- **Repeats.** Two runs averaged there, three here, with the spread printed.

Scale expectations accordingly: those gains are at concurrency 64, four times
what 4x910B4 allows, and the reachable cost-table spread grows with concurrency.

Usage (one process per configuration, repeats inside each):

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export VLLM_TEST_QWEN36_MODEL=/path/Qwen3.6-27B
    export VLLM_TEST_DSPARK_MODEL=/path/DSpark
    python examples/dspark_adaptive_verify_throughput.py --dataset /path/math500.jsonl

    # more repeats, or a sweep over engine capacity (each row its own config)
    python examples/dspark_adaptive_verify_throughput.py --repeats 5
    python examples/dspark_adaptive_verify_throughput.py --max-num-seqs 4 8 16 --waves 8
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
# Occupancy actually reached. The workload is submitted in full and the engine
# schedules up to max_num_seqs of it, so the batch should sit near capacity --
# but that is an inference about the scheduler, and the log states it.
REQS_RE = re.compile(r"mean reqs=([\d.]+)")
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


_PROMPT_KEYS = ("prompt", "question", "problem", "instruction", "text", "context")


def load_dataset(path: str) -> list[str]:
    """Prompts from a jsonl/json/txt file, in file order.

    PR 15147 measured D-Cut on Math500 and Dolly rather than on synthetic text,
    and that matters for more than realism: the benefit depends on confidence
    *varying* across requests and positions, so a handful of repeated prompts
    understates the spread the controller has to work with. File order, not a
    shuffle, so every configuration sees the identical workload.
    """
    file = Path(path)
    raw = file.read_text(errors="replace")
    # A Git LFS pointer is a ~130-byte stub that a clone without the LFS smudge
    # filter leaves in place of the data. Left to the JSON parser it surfaces as
    # "Expecting value: line 1 column 1", which reads like a bug in this script.
    if raw.lstrip().startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"{path} is a Git LFS pointer, not the dataset ({file.stat().st_size} bytes). "
            "Fetch the real file: `git lfs install && git lfs pull` in its directory, or download "
            "it directly, e.g. "
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
            "databricks-dolly-15k.jsonl"
        )

    records: list = []
    if file.suffix == ".jsonl":
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                # A .jsonl that is really a single JSON array is a common
                # mislabelling; flatten it rather than skipping every row.
                records.extend(parsed) if isinstance(parsed, list) else records.append(parsed)
            except ValueError as exc:
                # Name the line and show it: a jsonl file that is actually a
                # single JSON array, or has a header row, fails on line 1 and
                # the bare decoder message says nothing about which is which.
                raise ValueError(f"{path} line {number} is not JSON ({exc}); line starts: {line[:80]!r}") from exc
    elif file.suffix == ".json":
        try:
            loaded = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{path} is not JSON ({exc}); file starts: {raw[:80]!r}") from exc
        records = loaded if isinstance(loaded, list) else loaded.get("data") or loaded.get("rows") or []
    else:
        return [line.strip() for line in raw.splitlines() if line.strip()]

    prompts = []
    for record in records:
        if isinstance(record, str):
            prompts.append(record)
            continue
        if not isinstance(record, dict):
            continue
        # Dolly splits a request across two fields: `instruction` plus a
        # `context` that the closed-QA and summarisation categories are
        # meaningless without ("Summarise the following" with nothing to
        # summarise). Join them, so the prompt is the record.
        instruction = record.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            context = record.get("context")
            if isinstance(context, str) and context.strip():
                prompts.append(f"{instruction.strip()}\n\n{context.strip()}")
            else:
                prompts.append(instruction.strip())
            continue
        for key in _PROMPT_KEYS:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                prompts.append(value.strip())
                break
        else:
            # Chat-shaped rows: take the first user turn.
            messages = record.get("messages") or record.get("conversations")
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                content = messages[0].get("content") or messages[0].get("value")
                if isinstance(content, str) and content.strip():
                    prompts.append(content.strip())
    if not prompts:
        raise ValueError(f"no prompts found in {path}; expected one of {_PROMPT_KEYS}, messages, or plain lines")
    return prompts


def _stride(pool: list[str], count: int) -> list[str]:
    """`count` items spread evenly through `pool`, cycling if it is too small.

    Evenly spaced rather than the first N: a 500-request workload out of Dolly's
    15k records would otherwise be whatever sits at the top of the file, and in
    an instruction set that is often one category -- which would set the
    acceptance rate, and so the result, by accident.
    """
    if len(pool) >= count:
        step = len(pool) / count
        return [pool[int(i * step)] for i in range(count)]
    return [pool[i % len(pool)] for i in range(count)]


def build_prompts(count: int, dataset: str | None = None, fits=None) -> list[str]:
    """The workload: `count` prompts, deterministic and identical across lanes.

    Synthetic prompts are the fallback, not the intent. They are real text
    rather than random token ids on purpose -- the draft model cannot predict
    random ids, so confidence would be uniformly low and trimming trivially
    aggressive, which measures the wrong thing. But a handful of variations
    still has less confidence spread than a real dataset, so prefer --dataset.
    """
    count = max(1, count)
    pool = load_dataset(dataset) if dataset else [f"{p} (variation {i})" for i, p in enumerate(_BASE_PROMPTS)]
    if fits is not None:
        # ignore_eos stops a request ending early, but it cannot make room: a
        # prompt within max_model_len of the limit emits fewer than output_len
        # tokens and the totals no longer match. Dolly's long-context rows do
        # exactly that. Over-sample, drop what cannot fit, then take the
        # workload -- deterministic, so both lanes get the identical set.
        candidates = _stride(pool, min(len(pool), count * 4))
        kept = [prompt for prompt in candidates if fits(prompt)]
        if len(kept) < count:
            raise ValueError(
                f"only {len(kept)} of {len(candidates)} sampled prompts leave room for "
                f"{count} full-length outputs; raise --max-model-len or lower --output-len"
            )
        pool = kept
    return _stride(pool, count)


def draft_block_size(draft: str) -> int | None:
    """The draft checkpoint's trained block size, which fixes the draft count.

    ``vllm/config/speculative.py`` refuses to start unless
    ``num_speculative_tokens`` equals this exactly, so it is not a tuning knob:
    a different count needs a different checkpoint. Reading it here means the
    sweep defaults to the one value that can run, instead of failing most cells.
    """
    path = Path(draft) / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return None

    def find(node):
        if isinstance(node, dict):
            for key in ("block_size", "dspark_block_size"):
                value = node.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        return None

    return find(config)


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
        **({"max_num_batched_tokens": args.max_num_batched_tokens} if args.max_num_batched_tokens else {}),
        **({"block_size": args.block_size} if args.block_size else {}),
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

    # Filter against the real tokenizer, not a character heuristic: what
    # matters is whether prompt + output_len fits max_model_len.
    tokenizer = llm.get_tokenizer()
    room = args.max_model_len - args.output_len
    prompts = build_prompts(
        args.num_requests,
        args.dataset,
        fits=lambda text: len(tokenizer(text).input_ids) <= room,
    )
    # ignore_eos is the point: every configuration then emits exactly
    # concurrency x output_len tokens, so TPS compares time and nothing else.
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True, seed=17)

    # Discard the first pass. It carries graph capture, the cost-table
    # profiling and every lazy allocation, none of which recur -- charging them
    # to the first timed repeat would make the mean depend on the repeat count.
    llm.generate(prompts[: args.max_num_seqs], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))

    runs = []
    for _ in range(max(1, args.repeats)):
        started = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - started
        emitted = sum(len(o.outputs[0].token_ids) for o in outputs)
        runs.append({"elapsed": elapsed, "output_tokens": emitted, "decode": decode_spread(outputs)})
        print(RESULT_PREFIX + json.dumps(runs[-1]), flush=True)
    return 0


def decode_spread(outputs) -> dict | None:
    """Per-request decode time, summarised.

    Every request emits the same number of tokens, so a difference in decode
    time is a difference in how many steps that request needed -- which is
    exactly what per-request trimming changes: drafts that get cut mean fewer
    tokens accepted per step and more steps to reach the same output.

    The mean carries nothing: the workload is submitted at once and drained at
    a fixed width, so mean completion is about half the elapsed time and
    restates throughput. The spread does not, and it is the only place where
    "throughput improved while some requests got slower" can show up.

    first_token_ts and last_token_ts are both engine-core monotonic timestamps.
    arrival_time is a frontend wall-clock one and is deliberately not mixed in.
    """
    samples = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        first = getattr(metrics, "first_token_ts", 0.0) or 0.0
        last = getattr(metrics, "last_token_ts", 0.0) or 0.0
        if first and last > first:
            samples.append(last - first)
    if len(samples) < 2:
        return None
    samples.sort()
    return {
        "p50": samples[len(samples) // 2],
        "p99": samples[min(len(samples) - 1, int(len(samples) * 0.99))],
        "max": samples[-1],
        "n": len(samples),
    }


def child_env(max_model_len: int) -> dict[str, str]:
    env = os.environ.copy()
    # Price the cost table at the context this run actually uses. The upstream
    # default profiles at 8192 tokens and attention cost grows with context, so
    # profiling long while serving short inflates the fixed part of every
    # measurement and flattens the gradient the controller decides on -- the
    # table would then argue against trimming for a reason the run never sees.
    env.setdefault("VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", str(max_model_len))
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # A benchmark must compile what it measures. The cache key does not capture
    # everything that decides the compiled graph -- the adaptive path rewrites
    # cudagraph_mode after the configuration was settled -- so a cell can pick
    # up an artifact built for a different configuration, which surfaces from
    # the graph compiler as an unpack-count mismatch rather than as a cache
    # miss. The deployed serve configuration sets this for the same reason.
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
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


def requests_for(max_num_seqs: int, args: argparse.Namespace) -> int:
    """Workload size for a cell.

    ``--waves`` exists for a swept run: 500 requests through a 4-wide engine is
    125 waves against 31 through a 16-wide one, so a fixed count makes swept
    cells different shapes as well as different configurations.
    """
    if args.waves:
        return max(max_num_seqs, args.waves * max_num_seqs)
    return args.num_requests


def run_config(
    k: int, max_num_seqs: int, adaptive: bool, args: argparse.Namespace, log_dir: Path, label: str | None = None
) -> dict:
    lane = label or ("dynamics" if adaptive else "fixed")
    num_requests = requests_for(max_num_seqs, args)
    log_path = log_dir / f"k{k}-b{max_num_seqs}-{lane}.log"
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--engine-child",
        "--model", args.model,
        "--draft", args.draft,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--num-speculative-tokens", str(k),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--num-requests", str(num_requests),
        "--output-len", str(args.output_len),
        "--repeats", str(args.repeats),
        "--cudagraph-mode", args.cudagraph_mode,
    ]  # fmt: skip
    if adaptive:
        cmd.append("--adaptive")
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.block_size:
        cmd += ["--block-size", str(args.block_size)]
    if args.prefix_caching:
        cmd.append("--prefix-caching")
    if args.async_scheduling:
        cmd.append("--async-scheduling")

    print(f"\n=== K={k} max_num_seqs={max_num_seqs} {lane}: starting, log -> {log_path}", flush=True)
    started = time.monotonic()
    with log_path.open("w") as stream:
        try:
            result = subprocess.run(
                cmd,
                env=child_env(args.max_model_len),
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=CHILD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            # A worker that dies during startup can leave the parent process
            # alive, so the timeout is reached with the real error sitting in
            # the log an hour earlier. Report that instead of the timeout.
            tail = log_path.read_text(errors="replace")[-12000:]
            raise RuntimeError(
                f"K={k} b={max_num_seqs} {lane} did not finish within {CHILD_TIMEOUT_S:.0f}s; "
                f"tail of {log_path}:\n{tail}"
            ) from exc
    log = log_path.read_text(errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"K={k} b={max_num_seqs} {lane} exited {result.returncode}; tail of {log_path}:\n{log[-12000:]}"
        )

    runs = [json.loads(line.removeprefix(RESULT_PREFIX)) for line in log.splitlines() if line.startswith(RESULT_PREFIX)]
    if not runs:
        raise RuntimeError(f"K={k} b={max_num_seqs} {lane} printed no timing; tail of {log_path}:\n{log[-12000:]}")
    expected = num_requests * args.output_len
    for run in runs:
        if run["output_tokens"] != expected:
            # Without this the whole comparison silently degrades into the one
            # the specbench runs already showed to be unreadable: different
            # configurations emitting different numbers of tokens.
            raise RuntimeError(
                f"K={k} b={max_num_seqs} {lane} emitted {run['output_tokens']} tokens, expected {expected}. "
                "ignore_eos did not hold, so TPS is not comparable across lanes."
            )

    tps = [run["output_tokens"] / run["elapsed"] for run in runs]
    decode = [run["decode"] for run in runs if run.get("decode")]
    accept = [float(m) for m in ACCEPT_RE.findall(log)]
    draft_accept = [float(m) for m in DRAFT_ACCEPT_RE.findall(log)]
    av_lines = [line for line in log.splitlines() if AV_LOG_TAG in line and " steps | " in line]
    # Sum the per-window dispatch counts. The share of steps that got no graph
    # is the same for both lanes under a decode-only mode -- a mixed batch is
    # eager either way -- so it does not explain a difference between them, but
    # it does say how much of the run was never in a graph at all.
    dispatch: dict[str, int] = {}
    for line in av_lines:
        for field in GRAPH_FIELD_RE.findall(line):
            for entry in field.split(","):
                name, _, count = entry.partition("=")
                if count.isdigit():
                    dispatch[name] = dispatch.get(name, 0) + int(count)
    record = {
        "k": k,
        "max_num_seqs": max_num_seqs,
        "lane": lane,
        "tps": tps,
        "accept": statistics.fmean(accept) if accept else None,
        "draft_accept": statistics.fmean(draft_accept) if draft_accept else None,
        "dispatch": dispatch,
        "decode": {key: statistics.fmean([d[key] for d in decode]) for key in ("p50", "p99", "max")}
        if decode
        else None,
        "kept": [float(m) for line in av_lines for m in KEPT_RE.findall(line)],
        "occupancy": [float(m) for line in av_lines for m in REQS_RE.findall(line)],
        "log": str(log_path),
    }
    elapsed = time.monotonic() - started
    print(
        f"=== K={k} max_num_seqs={max_num_seqs} {lane}: done in {elapsed:.0f}s | "
        f"TPS {statistics.fmean(tps):.1f}" + (f" +/- {max(tps) - min(tps):.1f}" if len(tps) > 1 else ""),
        flush=True,
    )
    if adaptive:
        # The feature has to be shown to have engaged before its number means
        # anything. A run that never trimmed verified the same width as the
        # fixed lane, so equal throughput would be a tautology.
        if not record["kept"]:
            raise RuntimeError(f"K={k} b={max_num_seqs} dynamics logged no budget decision; see {log_path}")
        print(
            f"    engaged: kept {min(record['kept']):.1f}-{max(record['kept']):.1f}%, dispatch {dispatch}",
            flush=True,
        )
        if min(record["kept"]) >= 100.0:
            raise RuntimeError(
                f"K={k} b={max_num_seqs} dynamics never trimmed a draft, so its throughput is the fixed lane's "
                f"with extra bookkeeping, not a measurement of trimming; see {log_path}"
            )
    return record


def report(results: list[dict], args: argparse.Namespace) -> int:
    by_key = {(r["k"], r["max_num_seqs"], r["lane"]): r for r in results}
    print("\n==== throughput ====")
    print(
        f"model={os.path.basename(args.model or '?')} TP={args.tensor_parallel_size} "
        f"requests={f'{args.waves}x cap' if args.waves else args.num_requests} "
        f"output_len={args.output_len} "
        f"dataset={os.path.basename(args.dataset) if args.dataset else 'synthetic'} "
        f"cudagraph_mode={args.cudagraph_mode} repeats={args.repeats}"
    )
    header = (
        f"{'Draft':>5} | {'Cap':>4} | {'Reqs':>5} | {'Fixed TPS':>18} | {'Fixed Acc':>9} | "
        f"{'Dynamics TPS':>18} | {'Dyn Acc':>9} | {'Occ':>5} | {'Kept':>13} | {'NoGraph':>7} | "
        f"{'Acceleration':>13}"
    )
    print(header)
    print("-" * len(header))
    verdicts = []
    unpaired = []
    floors: list = []
    for k, cap in sorted({(r["k"], r["max_num_seqs"]) for r in results}):
        fixed, dyn = by_key.get((k, cap, "fixed")), by_key.get((k, cap, "dynamics"))
        if not (fixed and dyn):
            unpaired.extend(r for r in (fixed, dyn) if r)
            continue
        f_mean, d_mean = statistics.fmean(fixed["tps"]), statistics.fmean(dyn["tps"])
        f_spread, d_spread = _spread(fixed["tps"]), _spread(dyn["tps"])
        gain = (d_mean - f_mean) / f_mean * 100
        # A gain smaller than the runs' own spread is not a gain. Comparing
        # against the summed spread is the crude version of a significance test,
        # and crude is the right level for three repeats.
        #
        # Those repeats share a process, so their spread misses everything that
        # differs between engine loads. When a control run gives that figure,
        # the difference has to beat it too.
        control = by_key.get((k, cap, "control"))
        floor = abs(statistics.fmean(control["tps"]) - f_mean) if control else 0.0
        decisive = abs(d_mean - f_mean) > max(f_spread + d_spread, floor)
        if control:
            floors.append((k, cap, floor, floor / f_mean * 100))
        verdicts.append(((k, cap), gain, decisive))
        occupancy = dyn.get("occupancy") or []
        occ_text = f"{statistics.fmean(occupancy):.1f}" if occupancy else "n/a"
        kept = dyn["kept"]
        kept_text = f"{min(kept):.0f}-{max(kept):.0f}%" if kept else "n/a"
        total = sum(dyn["dispatch"].values())
        no_graph = f"{dyn['dispatch'].get('NONE', 0) / total * 100:.1f}%" if total else "n/a"
        print(
            f"{k:>5} | {cap:>4} | {requests_for(cap, args):>5} | "
            f"{f_mean:>10.1f} +/-{f_spread:>5.1f} | {_fmt(fixed['accept']):>9} | "
            f"{d_mean:>10.1f} +/-{d_spread:>5.1f} | {_fmt(dyn['accept']):>9} | "
            f"{occ_text:>5} | {kept_text:>13} | {no_graph:>7} | {gain:>+11.1f}%{'' if decisive else ' ?'}"
        )
    for k, cap, floor, pct in floors:
        print(
            f"\nNoise floor at draft {k}, capacity {cap}: two separate runs of the fixed lane "
            f"differ by {floor:.1f} TPS ({pct:.1f}%). A difference smaller than that is not a result."
        )
    for row in unpaired:
        if row["lane"] == "control":
            continue
        # Ran without its counterpart: a throughput number on its own is not an
        # acceleration, so report it as a measurement and say what is missing.
        print(
            f"{row['k']:>5} | {row['max_num_seqs']:>4} | {requests_for(row['max_num_seqs'], args):>5} | "
            f"{row['lane']} alone: TPS {statistics.fmean(row['tps']):.1f} +/-{_spread(row['tps']):.1f}, "
            f"acceptance {_fmt(row['accept'])} -- no counterpart, so no acceleration"
        )
    paired = [(k, cap) for k, cap in sorted({(r["k"], r["max_num_seqs"]) for r in results})]
    if any(by_key.get((k, cap, lane), {}).get("decode") for k, cap in paired for lane in ("fixed", "dynamics")):
        print("\nPer-request decode time (seconds), same output length for every request:")
        print(f"{'Draft':>5} | {'Cap':>4} | {'lane':>9} | {'p50':>7} | {'p99':>7} | {'max':>7}")
        for k, cap in paired:
            for lane in ("fixed", "dynamics"):
                row = by_key.get((k, cap, lane))
                spread = row and row.get("decode")
                if spread:
                    print(
                        f"{k:>5} | {cap:>4} | {lane:>9} | {spread['p50']:>7.2f} | "
                        f"{spread['p99']:>7.2f} | {spread['max']:>7.2f}"
                    )
        print(
            "Mean is omitted on purpose: the workload is submitted at once and drained at a fixed "
            "width, so mean completion is about half the elapsed time and restates throughput. The "
            "spread does not. Trimming is per request, and a request whose drafts are cut accepts "
            "fewer tokens per step, so it needs more steps for the same output -- which is how "
            "throughput can improve while some requests get slower. A p99 that grows much faster "
            "than p50 is that case."
        )
    print(
        "\nOcc is the mean number of requests actually in the batch. The whole workload is "
        "submitted at once, so the engine should hold it near capacity; a figure well below "
        "capacity means the measurement is not the saturated one it is meant to be. Note that a "
        "saturated batch is the most favourable regime for this feature -- the cost table's spread "
        "grows with the batch -- so a result here does not transfer to a lightly loaded server."
    )
    print(
        "\nKept near 100% means the controller decided not to trim, which is a correct decision "
        "about a flat cost table, not a fault -- but it also means the feature is paying its "
        "per-step cost for nothing, so a loss there is expected rather than surprising. NoGraph is "
        "the share of budgeted steps that ran eager because a prefill shared the batch; it is the "
        "same for both lanes under a decode-only mode, so it does not explain a difference."
    )
    print(
        "\nTPOT is not reported: with ignore_eos every request emits the same number of tokens, so "
        "mean per-token time is exactly capacity/TPS and would restate this table. PR 15147's "
        "separate TPS and TPOT columns carry independent information only because its output "
        "lengths varied (its +19.1% and -17.4% are the same measurement seen twice)."
    )
    print(
        "\n'?' marks a difference smaller than the two lanes' own run-to-run spread. "
        "Acceptance is context: trimming removes drafts, so it is expected to fall. "
        "A fall without a throughput gain is the case to worry about -- it means the cost "
        "table is overestimating what trimming saves."
    )
    if args.repeats < 2:
        print("\nWARNING: --repeats 1 gives no spread, so no difference here can be called decisive.")
    if not floors:
        print(
            "\nNOTE: the spread above is between passes of one engine, which misses the "
            "process-level variation that moved earlier measurements by several percent. For a "
            "difference of a few percent, re-run with --control to measure that floor."
        )
    undecided = [key for key, _, decisive in verdicts if not decisive]
    if undecided:
        print(
            f"(draft, capacity) cells with no decisive difference: {undecided}. "
            "More repeats, or a larger capacity -- the cost table's reachable spread grows with it, "
            "and trimming can only earn what that spread contains."
        )
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
        default=None,
        help=(
            "Draft-token counts. Defaults to the draft checkpoint's trained block_size, which is "
            "the only value that can run: speculative.py requires num_speculative_tokens to equal "
            "it exactly, so passing several here needs one checkpoint per value -- which is what "
            "PR 15098's three rows are."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        nargs="+",
        default=[16],
        help=(
            "Engine capacity. Defaults to a single value -- the deployment point -- because this is "
            "not a load knob: it sets the captured bucket set, the graph limit, the cost table's "
            "reachable range and the pinned GDN request axis, all at once. Measured graph_limit was "
            "32 at four and 128 at sixteen, so two such cells are different graph configurations, "
            "not one configuration under two loads. Passing several still works and is a useful "
            "sweep -- the reachable spread was 12.31ms at four and 29.97ms at sixteen, and trimming "
            "can only earn what it contains -- but read each row as its own configuration."
        ),
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=500,
        help=(
            "Workload size, independent of capacity -- PR 15147 used 500 requests. A dataset larger "
            "than this is sampled at an even stride, so there is no need to trim the file: 500 out "
            "of Dolly's 15k spans the whole dataset, while its first 500 rows would not."
        ),
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=None,
        help=(
            "Workload size as a multiple of capacity instead of a fixed count. Only useful when "
            "sweeping --max-num-seqs: 500 requests is 31 waves through a 16-wide engine but 125 "
            "through a 4-wide one, which took 29 minutes per cell."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "jsonl/json/txt file of prompts, used in file order. PR 15147 used Math500 and Dolly. "
            "Prefer this over the synthetic fallback: the benefit depends on confidence varying "
            "across requests, and a few repeated prompts understate that spread."
        ),
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
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Match the deployment. It bounds a scheduler step, so it shapes the prefill/decode mix.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Match the deployment; the KV block size is not a neutral choice for a hybrid model.",
    )
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument(
        "--control",
        action="store_true",
        help=(
            "Run the fixed lane a second time, in its own process, as a noise floor. The repeats "
            "inside one engine only see drift between passes, not the process-level variation that "
            "moved earlier measurements by several percent, so without this a small difference can "
            "be marked decisive on a spread that understates the real one. Costs one more process."
        ),
    )
    parser.add_argument(
        "--lanes",
        nargs="+",
        choices=("fixed", "dynamics"),
        default=["fixed", "dynamics"],
        help=(
            "Which lanes to run. Both by default, since one alone cannot produce an acceleration. "
            "Use one when bisecting a startup failure: it fails during engine construction, so the "
            "other lane's twenty minutes buy nothing."
        ),
    )
    parser.add_argument("--engine-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--adaptive", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.engine_child:
        args.num_speculative_tokens = args.num_speculative_tokens[0]
        args.max_num_seqs = args.max_num_seqs[0]
        return run_engine(args)
    if not args.model or not args.draft:
        parser.error("set VLLM_TEST_QWEN36_MODEL and VLLM_TEST_DSPARK_MODEL, or pass --model/--draft")
    if args.num_speculative_tokens is None:
        block = draft_block_size(args.draft)
        if block is None:
            parser.error(
                "could not read block_size from the draft config; pass -k with the checkpoint's "
                "trained block size (any other value fails config validation)"
            )
        args.num_speculative_tokens = [block]
        print(f"Draft count {block}, from the checkpoint's trained block_size.", flush=True)

    log_dir = Path(f"dspark_av_tput_{time.strftime('%Y%m%d-%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir.resolve()}", flush=True)
    # These change what engine is built. A benchmark whose engine differs from
    # the deployed one measures something nobody runs, and the difference is
    # invisible unless it is printed.
    shaping = {
        name: os.environ.get(name)
        for name in (
            "VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK",
            "VLLM_ASCEND_KV_GROUP_MIN_SIZE",
            "VLLM_ASCEND_DSPARK_AV_GRAPH",
            "VLLM_ASCEND_DSPARK_AV_KEEP_PIECEWISE",
            "VLLM_ASCEND_DSPARK_AV_TP_BROADCAST",
            "VLLM_ASCEND_DSPARK_AV_ADAPT",
        )
    }
    print("Engine-shaping env: " + ", ".join(f"{k}={v or 'unset'}" for k, v in shaping.items()), flush=True)

    results, failures = [], []
    for k in args.num_speculative_tokens:
        for max_num_seqs in args.max_num_seqs:
            # Fixed first: if it fails, the dynamics number has nothing to compare to.
            for lane in ("fixed", "dynamics"):
                if lane not in args.lanes:
                    continue
                adaptive = lane == "dynamics"
                try:
                    results.append(run_config(k, max_num_seqs, adaptive, args, log_dir))
                except Exception as exc:  # noqa: BLE001 - report every cell, fail at the end
                    failures.append(f"K={k} b={max_num_seqs} {lane}: {exc}")
                    print(f"=== K={k} max_num_seqs={max_num_seqs} {lane}: FAILED -- {exc}", flush=True)
                if lane == "fixed" and args.control:
                    # The same configuration in a second process. Whatever this
                    # differs from the first by is noise, and a lane has to beat
                    # it before a difference means anything.
                    try:
                        results.append(run_config(k, max_num_seqs, False, args, log_dir, label="control"))
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"K={k} b={max_num_seqs} control: {exc}")
                        print(f"=== K={k} max_num_seqs={max_num_seqs} control: FAILED -- {exc}", flush=True)

    rc = report(results, args) if results else 1
    for line in failures:
        print(f"FAIL {line}")
    return 1 if failures else rc


if __name__ == "__main__":
    raise SystemExit(main())
