#!/usr/bin/env python3
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
# This file is a part of the vllm-ascend project.
#
"""Compare the operators that can serve parallel-drafting draft attention.

One shape -- non-causal, TND varlen query, paged KV, GQA, which is what a
DSpark/DFlash draft asks for -- run through every operator that can answer it, so
accuracy and cost can be read side by side:

  torch_npu_fia  torch_npu.npu_fused_infer_attention_score. The ordinary path.
                 Wants its sequence lengths as host lists, so a draft has to pay
                 seq_lens.tolist() -- a device-to-host sync on a value the verify
                 kernel wrote moments ago. That sync is timed here as `prep`.
  fia_sink       omni_custom_ops npu_fused_infer_attention_sink. Takes the lengths
                 as device tensors and tiles on AICPU, so `prep` is a device op
                 rather than a sync. Serves head sizes 128/192/512 only.
  fa_v3 / fa_v4  flash-attention-npu. Same AICPU-tiling idea through
                 get_scheduler_metadata; head_dim is a runtime tiling field, so
                 anything up to 256 works. v3 goes through flash_attn_with_kvcache,
                 v4 through flash_attn_varlen_func.

`prep` and `attn` are timed separately on purpose. In a real step the draft builds
its lengths and tiling once and every layer reuses them, so per-layer cost is
`attn` while `prep` is paid once -- a backend can win on one and lose on the other.

Deliberately standalone: no vllm or vllm_ascend imports, so it runs on a bare
torch_npu install and its failures are the operator's, not the framework's.

Examples
--------
    # the case that motivated this: head_dim 256, which fia_sink cannot serve
    python compare_draft_attention.py --head-size 256

    # head_dim 128, where every backend can answer, so the numbers are comparable
    python compare_draft_attention.py --head-size 128

    # collect a profile per backend (op_statistic.csv lands under --profile-dir)
    python compare_draft_attention.py --head-size 128 --profile
"""

from __future__ import annotations

import argparse
import importlib
import math
import time
import traceback
from dataclasses import dataclass, field
from functools import partial

import torch
import torch_npu  # noqa: F401  # registers the NPU device and torch.ops.npu

# vllm_ascend/attention/attention_v1.py uses this for "no window"; inlined rather
# than imported so this script does not pull in vllm.
SWA_INT_MAX = 2147483647

# Head sizes npu_fused_infer_attention_sink serves. Outside this set it fails
# inside the operator, so the backend is skipped with a reason instead.
FIA_SINK_HEAD_SIZES = (128, 192, 512)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    batch: int
    q_per_req: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_blocks_per_seq: int
    dtype: torch.dtype
    seed: int

    @property
    def num_tokens(self) -> int:
        return self.batch * self.q_per_req

    @property
    def kv_capacity(self) -> int:
        """Page capacity per request.

        Both the FA metadata call and the sink operator address the block table
        with a stride derived from this, not from the actual maximum KV length, so
        it is what gets passed as the KV bound everywhere below.
        """
        return self.max_blocks_per_seq * self.block_size

    @property
    def scale(self) -> float:
        return self.head_size**-0.5


@dataclass
class Inputs:
    query: torch.Tensor  # (T, N, D)
    key_cache: torch.Tensor  # (num_blocks, block_size, Nkv, D)
    value_cache: torch.Tensor
    block_table: torch.Tensor  # (B, max_blocks_per_seq) int32
    seq_lens: torch.Tensor  # (B,) int32, device -- the value a draft cannot read
    kv_lens_host: list[int] = field(default_factory=list)


def build_inputs(cfg: Config) -> Inputs:
    generator = torch.Generator().manual_seed(cfg.seed)

    def rand(shape):
        return ((torch.rand(shape, generator=generator) - 0.5) * 2.0).to(cfg.dtype).npu()

    num_blocks = cfg.batch * cfg.max_blocks_per_seq
    # Spread the KV lengths so a backend that mixes up per-request bounds shows it,
    # and keep them inside the page capacity.
    lo = max(1, cfg.kv_capacity // 8)
    kv_lens = [lo + (cfg.kv_capacity - lo) * i // max(1, cfg.batch - 1) for i in range(cfg.batch)]
    kv_lens = [min(length, cfg.kv_capacity) for length in kv_lens]

    return Inputs(
        query=rand((cfg.num_tokens, cfg.num_heads, cfg.head_size)),
        key_cache=rand((num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_size)),
        value_cache=rand((num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_size)),
        block_table=torch.arange(num_blocks, dtype=torch.int32)
        .reshape(cfg.batch, cfg.max_blocks_per_seq)
        .npu(),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32).npu(),
        kv_lens_host=kv_lens,
    )


def golden(cfg: Config, inp: Inputs, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Per-request non-causal attention on CPU over the gathered paged KV.

    ``dtype=None`` computes in fp32 and is the reference. Passing the test dtype
    instead gives the noise floor: how far the same maths in the same precision
    lands from fp32, which is the scale a backend's error should be read against.
    """
    work = torch.float32 if dtype is None else dtype
    query = inp.query.detach().cpu().to(work)
    key_cache = inp.key_cache.detach().cpu().to(work)
    value_cache = inp.value_cache.detach().cpu().to(work)
    block_table = inp.block_table.cpu().to(torch.long)
    group = cfg.num_heads // cfg.num_kv_heads

    outs = []
    for b, kv_len in enumerate(inp.kv_lens_host):
        positions = torch.arange(kv_len)
        blocks = block_table[b][positions // cfg.block_size]
        offsets = positions % cfg.block_size
        key = key_cache[blocks, offsets]  # (kv_len, Nkv, D)
        value = value_cache[blocks, offsets]
        q = query[b * cfg.q_per_req : (b + 1) * cfg.q_per_req]  # (q, N, D)

        # (N, q, D) x (N, D, kv) -> (N, q, kv); no mask, every query sees all KV.
        scores = torch.einsum("qnd,knd->nqk", q, key.repeat_interleave(group, dim=1)) * cfg.scale
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(work)
        out = torch.einsum("nqk,knd->qnd", probs, value.repeat_interleave(group, dim=1))
        outs.append(out)
    return torch.cat(outs, dim=0).to(torch.float32)


# --------------------------------------------------------------------------- #
# execution-flow trace
# --------------------------------------------------------------------------- #
class Flow:
    """Records the operator calls a backend issues, so the flow is readable."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.lines: list[str] = []
        self._seen = False

    def op(self, phase: str, name: str, **details) -> None:
        if self._seen:
            return
        rendered = ", ".join(f"{k}={v}" for k, v in details.items())
        self.lines.append(f"    [{phase}] {name}({rendered})")

    def freeze(self) -> None:
        """Stop recording after the first pass, so warmup does not repeat it."""
        self._seen = True

    def show(self, title: str) -> None:
        if not self.enabled or not self.lines:
            return
        print(f"  flow of {title}:")
        for line in self.lines:
            print(line)


def shape(t) -> str:
    if t is None:
        return "None"
    if isinstance(t, torch.Tensor):
        return f"{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"
    if isinstance(t, list):
        return f"list[{len(t)}](host)"
    return str(t)


# --------------------------------------------------------------------------- #
# backends: each returns (prep_fn, attn_fn); prep_fn's result feeds attn_fn
# --------------------------------------------------------------------------- #
def backend_torch_npu_fia(cfg: Config, inp: Inputs, flow: Flow):
    num_blocks = inp.key_cache.shape[0]
    key = inp.key_cache.view(num_blocks, cfg.block_size, -1)
    value = inp.value_cache.view(num_blocks, cfg.block_size, -1)

    def prep():
        # The cost this whole exercise exists to remove: both length lists have to
        # be on the host, and seq_lens is written on device by the verify kernel.
        flow.op("prep", "seq_lens.tolist()", note="device-to-host sync")
        kv_lens = inp.seq_lens.tolist()
        q_lens = [(i + 1) * cfg.q_per_req for i in range(cfg.batch)]
        return q_lens, kv_lens

    def attn(prepared):
        q_lens, kv_lens = prepared
        flow.op(
            "attn",
            "torch_npu.npu_fused_infer_attention_score",
            query=shape(inp.query),
            key=shape(key),
            block_table=shape(inp.block_table),
            actual_seq_lengths=shape(q_lens),
            actual_seq_lengths_kv=shape(kv_lens),
            input_layout="TND",
            sparse_mode=0,
        )
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=inp.query,
            key=key,
            value=value,
            block_table=inp.block_table,
            input_layout="TND",
            block_size=cfg.block_size,
            actual_seq_lengths=q_lens,
            actual_seq_lengths_kv=kv_lens,
            num_key_value_heads=cfg.num_kv_heads,
            num_heads=cfg.num_heads,
            scale=cfg.scale,
            sparse_mode=0,
        )
        return out.view(cfg.num_tokens, cfg.num_heads, cfg.head_size)

    return prep, attn


def backend_fia_sink(cfg: Config, inp: Inputs, flow: Flow):
    importlib.import_module("omni_custom_ops")
    num_blocks = inp.key_cache.shape[0]
    # The sink operator reads KV as BnBsH, so the head dim is folded in.
    key = inp.key_cache.view(num_blocks, cfg.block_size, -1)
    value = inp.value_cache.view(num_blocks, cfg.block_size, -1)
    limit = torch.npu.get_stream_limit(torch.npu.current_stream())

    def prep():
        # Cumulative q lengths from the static shape, KV lengths straight off the
        # device tensor -- no host round trip anywhere in here.
        actual_seq_qlen = (
            torch.arange(1, cfg.batch + 1, dtype=torch.int64, device=inp.seq_lens.device) * cfg.q_per_req
        )
        actual_seq_kvlen = inp.seq_lens.to(torch.int64).clamp_min(1)
        flow.op(
            "prep",
            "torch.ops.custom._npu_fused_infer_attention_sink_metadata",
            actual_seq_lengths=shape(actual_seq_qlen),
            actual_seq_lengths_kv=shape(actual_seq_kvlen),
            input_layout="TND",
            input_layout_kv="BnBsH",
            sink_num=0,
            note="AICPU tiling",
        )
        meta = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_size,
            cfg.head_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            batch_size=cfg.batch,
            sparse_mode=0,
            pre_tokens=SWA_INT_MAX,
            next_tokens=SWA_INT_MAX,
            input_layout="TND",
            input_layout_kv="BnBsH",
            sink_num=0,
            block_size=cfg.block_size,
            aic_core_num=limit["cube_core_num"],
            aiv_core_num=limit["vector_core_num"],
        )
        return actual_seq_qlen, actual_seq_kvlen, meta

    def attn(prepared):
        actual_seq_qlen, actual_seq_kvlen, meta = prepared
        flow.op(
            "attn",
            "torch.ops.custom.npu_fused_infer_attention_sink",
            query=shape(inp.query),
            key=shape(key),
            block_table=shape(inp.block_table),
            meta_data=shape(meta),
            input_layout="TND",
            sparse_mode=0,
        )
        out, _ = torch.ops.custom.npu_fused_infer_attention_sink(
            inp.query,
            key,
            value,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=inp.block_table,
            num_query_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            softmax_scale=cfg.scale,
            input_layout="TND",
            sparse_mode=0,
            block_size=cfg.block_size,
            sink_number=0,
            meta_data=meta,
        )
        return out.view(cfg.num_tokens, cfg.num_heads, cfg.head_size)

    return prep, attn


def _backend_flash_attn_npu(cfg: Config, inp: Inputs, flow: Flow, generation: str):
    module_name = {"v3": "flash_attn_npu_3", "v4": "flash_attn_npu_4"}[generation]
    module = importlib.import_module(module_name)
    entry = "flash_attn_with_kvcache" if generation == "v3" else "flash_attn_varlen_func"
    for attr in ("get_scheduler_metadata", entry):
        if not hasattr(module, attr):
            raise RuntimeError(f"{module_name} has no {attr} (wrong device build?)")

    def prep():
        cu_seqlens_q = torch.arange(cfg.batch + 1, dtype=torch.int32, device=inp.seq_lens.device) * cfg.q_per_req
        seqused_k = inp.seq_lens.to(torch.int32).clamp_min(1)
        flow.op(
            "prep",
            f"{module_name}.get_scheduler_metadata",
            cu_seqlens_q=shape(cu_seqlens_q),
            cache_seqlens=shape(seqused_k),
            max_seqlen_k=cfg.kv_capacity,
            page_size=cfg.block_size,
            note="AICPU tiling; max_seqlen_k is the page capacity, not the actual max",
        )
        meta = module.get_scheduler_metadata(
            batch_size=cfg.batch,
            max_seqlen_q=cfg.q_per_req,
            max_seqlen_k=cfg.kv_capacity,
            num_heads_q=cfg.num_heads,
            num_heads_kv=cfg.num_kv_heads,
            headdim=cfg.head_size,
            cache_seqlens=seqused_k,
            qkv_dtype=cfg.dtype,
            headdim_v=cfg.head_size,
            cu_seqlens_q=cu_seqlens_q,
            page_size=cfg.block_size,
            causal=False,
            window_size=(-1, -1),
            softmax_scale=cfg.scale,
        )
        return cu_seqlens_q, seqused_k, meta

    def attn_v4(prepared):
        cu_seqlens_q, seqused_k, meta = prepared
        flow.op(
            "attn",
            "flash_attn_npu_4.flash_attn_varlen_func",
            q=shape(inp.query),
            k_cache=shape(inp.key_cache),
            page_table=shape(inp.block_table),
            seqused_k=shape(seqused_k),
            max_seqlen_k=cfg.kv_capacity,
            causal=False,
        )
        return module.flash_attn_varlen_func(
            inp.query,
            inp.key_cache,
            inp.value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=inp.block_table,
            max_seqlen_q=cfg.q_per_req,
            max_seqlen_k=cfg.kv_capacity,
            softmax_scale=cfg.scale,
            causal=False,
            window_size=(-1, -1),
            scheduler_metadata=meta,
            num_splits=0,
            return_lse=False,
        )

    def attn_v3(prepared):
        cu_seqlens_q, seqused_k, meta = prepared
        # No max_seqlen_k: v3 derives the KV bound as k_cache.shape[1] *
        # page_table.shape[1] and its _validate_scheduler_metadata rejects metadata
        # built against a different one -- it checks what v4 takes on trust.
        flow.op(
            "attn",
            "flash_attn_npu_3.flash_attn_with_kvcache",
            q=shape(inp.query),
            k_cache=shape(inp.key_cache),
            page_table=shape(inp.block_table),
            cache_seqlens=shape(seqused_k),
            causal=False,
            note="KV bound derived and validated by v3 itself",
        )
        return module.flash_attn_with_kvcache(
            inp.query,
            inp.key_cache,
            inp.value_cache,
            cache_seqlens=seqused_k,
            page_table=inp.block_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=cfg.q_per_req,
            softmax_scale=cfg.scale,
            causal=False,
            window_size=(-1, -1),
            rotary_interleaved=False,
            scheduler_metadata=meta,
            num_splits=0,
            return_softmax_lse=False,
        )

    attn = attn_v3 if generation == "v3" else attn_v4

    def attn_reshaped(prepared):
        return attn(prepared).view(cfg.num_tokens, cfg.num_heads, cfg.head_size)

    return prep, attn_reshaped


BACKENDS = {
    "torch_npu_fia": backend_torch_npu_fia,
    "fia_sink": backend_fia_sink,
    "fa_v3": lambda cfg, inp, flow: _backend_flash_attn_npu(cfg, inp, flow, "v3"),
    "fa_v4": lambda cfg, inp, flow: _backend_flash_attn_npu(cfg, inp, flow, "v4"),
}


def skip_reason(name: str, cfg: Config, force: bool) -> str | None:
    if name == "fia_sink" and not force and cfg.head_size not in FIA_SINK_HEAD_SIZES:
        return f"head_size {cfg.head_size} not in {FIA_SINK_HEAD_SIZES} (use --force to try anyway)"
    return None


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def bench(fn, warmup: int, iters: int) -> float:
    """Mean wall-clock ms per call, syncing around the whole measured region.

    Wall clock rather than NPU events on purpose: for the ordinary FIA path the
    cost being measured in `prep` *is* a host-side sync, which device timing would
    report as zero.
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def accuracy(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    out = out.detach().cpu().to(torch.float32)
    diff = (out - ref).abs()
    denom = ref.abs().max().item()
    return diff.max().item(), (diff.max().item() / denom if denom > 0 else math.inf)


def run_profile(name: str, prep, attn, directory: str, iters: int) -> None:
    path = f"{directory}/{name}"
    kwargs = {
        "activities": [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        "on_trace_ready": torch_npu.profiler.tensorboard_trace_handler(path),
    }
    try:
        # AI-core pipe utilisation is the interesting part when comparing kernels,
        # but these knobs move between torch_npu releases -- losing them should cost
        # detail, not the run.
        kwargs["experimental_config"] = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
    except (AttributeError, TypeError) as exc:
        print(f"  (profiling without experimental config: {exc})")

    with torch_npu.profiler.profile(**kwargs) as prof:
        for _ in range(iters):
            attn(prep())
            prof.step()
    torch.npu.synchronize()
    print(f"  profile written to {path} (see its ASCEND_PROFILER_OUTPUT/op_statistic.csv)")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=8, help="requests in the draft batch")
    parser.add_argument("--q-per-req", type=int, default=4, help="draft tokens per request (uniform)")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=256, help="128 to compare against fia_sink")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-blocks-per-seq", type=int, default=16, help="page capacity = this * block_size")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--backends", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="run backends outside their supported domain")
    parser.add_argument("--no-flow", action="store_true", help="do not print the operator flow")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="./prof_draft_attn")
    parser.add_argument("--profile-iters", type=int, default=5)
    args = parser.parse_args()

    if torch.npu.device_count() == 0:
        print("no NPU visible")
        return 1
    if args.num_heads % args.num_kv_heads != 0:
        print("num_heads must be divisible by num_kv_heads")
        return 1

    cfg = Config(
        batch=args.batch,
        q_per_req=args.q_per_req,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        max_blocks_per_seq=args.max_blocks_per_seq,
        dtype=DTYPES[args.dtype],
        seed=args.seed,
    )
    names = sorted(BACKENDS) if args.backends == "all" else [n.strip() for n in args.backends.split(",")]
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        print(f"unknown backend(s): {', '.join(unknown)}; known: {', '.join(sorted(BACKENDS))}")
        return 1

    print(f"device      : {torch_npu.npu.get_device_name()}")
    print(
        f"shape       : batch={cfg.batch} q/req={cfg.q_per_req} heads={cfg.num_heads}/{cfg.num_kv_heads} "
        f"head_size={cfg.head_size} dtype={args.dtype}"
    )
    print(f"paged KV    : block_size={cfg.block_size} capacity={cfg.kv_capacity} per request")
    inp = build_inputs(cfg)
    print(f"kv lengths  : {inp.kv_lens_host}")

    ref = golden(cfg, inp)
    noise, _ = accuracy(golden(cfg, inp, dtype=cfg.dtype), ref)
    print(f"noise floor : {noise:.3e} (same maths in {args.dtype} on CPU vs fp32)\n")

    rows = []
    for name in names:
        print(f"[{name}]")
        reason = skip_reason(name, cfg, args.force)
        if reason:
            print(f"  skipped: {reason}\n")
            rows.append((name, "skipped", "", "", reason))
            continue

        flow = Flow(enabled=not args.no_flow)
        try:
            prep, attn = BACKENDS[name](cfg, inp, flow)
            out = attn(prep())
            torch.npu.synchronize()
        except Exception as exc:  # an unavailable operator is a result, not a crash
            print(f"  unavailable: {type(exc).__name__}: {exc}")
            if args.force:
                traceback.print_exc()
            print()
            rows.append((name, "failed", "", "", f"{type(exc).__name__}: {exc}"))
            continue

        flow.show(name)
        flow.freeze()

        max_abs, rel = accuracy(out, ref)
        prep_ms = bench(prep, args.warmup, args.iters)
        prepared = prep()
        attn_ms = bench(partial(attn, prepared), args.warmup, args.iters)
        verdict = "ok" if max_abs <= max(noise * 4, 1e-3) else "SUSPECT"
        print(f"  accuracy: max|err|={max_abs:.3e} rel={rel:.3e} vs fp32 golden -> {verdict}")
        print(f"  timing  : prep={prep_ms:.3f} ms  attn={attn_ms:.3f} ms  total={prep_ms + attn_ms:.3f} ms")

        if args.profile:
            run_profile(name, prep, attn, args.profile_dir, args.profile_iters)
        print()
        rows.append((name, verdict, f"{max_abs:.3e}", f"{prep_ms:.3f}/{attn_ms:.3f}", ""))

    width = max(len(r[0]) for r in rows) if rows else 8
    print("summary (prep = per-step tiling or host sync; attn = per-layer)")
    print(f"  {'backend'.ljust(width)}  {'result':8}  {'max|err|':10}  {'prep/attn ms':14}  note")
    for name, verdict, err, timing, note in rows:
        print(f"  {name.ljust(width)}  {verdict:8}  {err:10}  {timing:14}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
