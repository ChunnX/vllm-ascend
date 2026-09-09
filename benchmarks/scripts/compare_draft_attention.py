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

fa_v3 comes with a caveat measured on hardware: once the tiling turns flash decode
on, its scheduler-metadata path returns zeros or garbage, because the split KV
workspace is initialised only in the host tiling branch. v4 is correct either way.
Flash decode is therefore reported for every run and recorded in the CSV -- a
result that does not say which mode it measured cannot be compared with one that
measured the other. `--kv-max` under 1024 turns it off at these shapes.

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

    # measure what production actually runs: one captured NPUGraph, replayed
    python compare_draft_attention.py --head-size 256 --graph --profile

    # a sweep is a shell loop plus --csv, which appends; --tag labels the build
    for h in 64 128 192 256; do
      for kv in 512 4096; do
        python compare_draft_attention.py --head-size $h --kv-max $kv \
            --csv runs.csv --tag "$(git -C ../flash-attention-npu rev-parse --short HEAD)"
      done
    done
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
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

# Tensors that depend only on the batch shape, so they are built once and reused.
# The E7 profile put the arange behind cu_seqlens_q at 11.9us of device time and
# 26us of host time per call, to produce nine int32 values that cannot change
# while the shape holds -- launch overhead on a tiny vector kernel, charged to
# every backend that takes its lengths on device. Leaving it in the timed path
# measures the harness rather than the operator.
_STATIC_TENSORS: dict[tuple, torch.Tensor] = {}


def static_tensor(key: tuple, build) -> torch.Tensor:
    cached = _STATIC_TENSORS.get(key)
    if cached is None:
        cached = build()
        _STATIC_TENSORS[key] = cached
    return cached

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
    kv_max: int
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

    def flash_decode(self, kv_lens: list[int]) -> bool | None:
        """Whether the tiling will turn flash decode on for this shape.

        Reproduced from the predicate the host and AICPU tiling share, rather than
        read back out of a metadata blob, which would mean hand-computing the C++
        layout of that struct. It is a label on the run, not something the run
        depends on.

        Worth labelling because it is not cosmetic: flash-attention-npu v3 returns
        zeros or garbage on this exact shape once flash decode is on, while v4 is
        correct either way. Any comparison that does not say which mode it measured
        is not reproducible.
        """
        try:
            cube = torch.npu.get_stream_limit(torch.npu.current_stream())["cube_core_num"]
        except Exception:
            return None
        group = self.num_heads // self.num_kv_heads
        num_tasks = self.batch * self.num_kv_heads
        max_kv = max(kv_lens)
        long_seq = num_tasks <= 0.8 * cube and max_kv >= cube * 512
        short_seq = num_tasks <= 0.4 * cube and max_kv >= 1024
        return (
            self.q_per_req * group <= 128
            and self.q_per_req <= 16
            and max_kv >= 1024
            and min(kv_lens) > 0
            and (long_seq or short_seq)
        )


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
    # and keep them inside the page capacity. The top of the spread is a knob
    # because it is one of the terms deciding flash decode, and flash decode is
    # where flash-attention-npu v3 and v4 were found to disagree.
    top = min(cfg.kv_max, cfg.kv_capacity)
    lo = max(1, top // 8)
    if cfg.batch == 1:
        # A one-request batch has nothing to spread, and taking the bottom of the
        # range would silently cap max KV at kv_max/8 -- which for --kv-max 4096
        # lands under the 1024 that flash decode needs, turning it off in exactly
        # the case where it is meant to be on. Batch 1 is also the only case that
        # reaches flash decode at Qwen3.6's 8 KV heads, so the whole mode would
        # have been unreachable.
        kv_lens = [top]
    else:
        kv_lens = [lo + (top - lo) * i // (cfg.batch - 1) for i in range(cfg.batch)]
    kv_lens = [max(1, min(length, cfg.kv_capacity)) for length in kv_lens]

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


def origin(module) -> str:
    """Where a backend's operator library was actually imported from.

    Worth a line of output because the failure it catches is silent and costly:
    running from the vllm-ascend tree imports flash_attn_npu_4 from site-packages,
    while the flash-attention-npu test suite imports the in-place build in its own
    source tree. The two can be different builds, and the only symptom is the
    operator behaving like an older version of itself -- a capture failing here
    while the same case passes under pytest, for instance.
    """
    return getattr(module, "__file__", None) or f"<{module.__name__}: no __file__>"


def axes(t, names: str) -> str:
    """Render a shape with its axis names.

    `key=(128, 128, 256)` is unreadable on its own: the first 128 is the block
    count, the second is the page size, and 256 is kv_heads * head_size folded
    together for BnBsH -- not a head dim of 256, which is what it looks like when
    head_size really is 128. Naming the axes costs one string and removes the
    ambiguity.
    """
    return f"{shape(t)} [{names}]"


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

    flow.op("load", "torch_npu", path=origin(torch_npu), version=torch_npu.__version__)

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
            query=axes(inp.query, "total_q, heads, head_size"),
            key=axes(key, "blocks, page_size, kv_heads*head_size"),
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
    ops_module = importlib.import_module("omni_custom_ops")
    flow.op("load", "omni_custom_ops", path=origin(ops_module))
    num_blocks = inp.key_cache.shape[0]
    # The sink operator reads KV as BnBsH, so the head dim is folded in.
    key = inp.key_cache.view(num_blocks, cfg.block_size, -1)
    value = inp.value_cache.view(num_blocks, cfg.block_size, -1)
    limit = torch.npu.get_stream_limit(torch.npu.current_stream())

    def prep():
        # Cumulative q lengths from the static shape, KV lengths straight off the
        # device tensor -- no host round trip anywhere in here.
        actual_seq_qlen = static_tensor(
            ("sink_qlen", cfg.batch, cfg.q_per_req, str(inp.seq_lens.device)),
            lambda: torch.arange(1, cfg.batch + 1, dtype=torch.int64, device=inp.seq_lens.device) * cfg.q_per_req,
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
            query=axes(inp.query, "total_q, heads, head_size"),
            key=axes(key, "blocks, page_size, kv_heads*head_size"),
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
    flow.op("load", module_name, path=origin(module))
    entry = "flash_attn_with_kvcache" if generation == "v3" else "flash_attn_varlen_func"
    for attr in ("get_scheduler_metadata", entry):
        if not hasattr(module, attr):
            raise RuntimeError(f"{module_name} has no {attr} (wrong device build?)")

    def prep():
        cu_seqlens_q = static_tensor(
            ("fa_cu_seqlens", cfg.batch, cfg.q_per_req, str(inp.seq_lens.device)),
            lambda: torch.arange(cfg.batch + 1, dtype=torch.int32, device=inp.seq_lens.device) * cfg.q_per_req,
        )
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
            q=axes(inp.query, "total_q, heads, head_size"),
            k_cache=axes(inp.key_cache, "blocks, page_size, kv_heads, head_size"),
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
            q=axes(inp.query, "total_q, heads, head_size"),
            k_cache=axes(inp.key_cache, "blocks, page_size, kv_heads, head_size"),
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


@dataclass
class Row:
    backend: str
    result: str
    max_err: str
    prep_ms: str
    attn_ms: str
    note: str


CSV_FIELDS = [
    "tag", "device", "backend", "result", "max_err", "noise_floor", "prep_ms", "attn_ms",
    "batch", "q_per_req", "num_heads", "num_kv_heads", "head_size", "block_size",
    "kv_capacity", "kv_max", "kv_lens", "flash_decode", "dtype", "note",
]


def write_csv(path: str, tag: str, cfg: Config, inp: Inputs, flash_decode, noise: float, rows) -> None:
    """Append the run, so a shell loop over shapes accumulates one comparable table.

    Every field the result depends on is written next to it, flash_decode included:
    a row that does not say which mode it measured cannot be compared with one that
    measured the other.
    """
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tag": tag,
                    "device": torch_npu.npu.get_device_name(),
                    "backend": row.backend,
                    "result": row.result,
                    "max_err": row.max_err,
                    "noise_floor": f"{noise:.3e}",
                    "prep_ms": row.prep_ms,
                    "attn_ms": row.attn_ms,
                    "batch": cfg.batch,
                    "q_per_req": cfg.q_per_req,
                    "num_heads": cfg.num_heads,
                    "num_kv_heads": cfg.num_kv_heads,
                    "head_size": cfg.head_size,
                    "block_size": cfg.block_size,
                    "kv_capacity": cfg.kv_capacity,
                    "kv_max": cfg.kv_max,
                    "kv_lens": " ".join(str(n) for n in inp.kv_lens_host),
                    "flash_decode": flash_decode,
                    "dtype": str(cfg.dtype).replace("torch.", ""),
                    "note": row.note,
                }
            )


# Which backends can have their prep captured. The AICPU-tiling ones can: their
# prep is device work reading device tensors, so replay re-runs it against
# whatever the sequence lengths hold now -- the whole reason they exist. The
# ordinary FIA path cannot: its prep is a device-to-host copy, which a capturing
# stream rejects outright. That asymmetry is not a detail of this harness; it is
# the difference between a draft whose lengths can live on device and one whose
# cannot.
PREP_ON_DEVICE = {"fia_sink", "fa_v3", "fa_v4"}


def capture(name: str, prep, attn, warmup: int):
    """Capture a backend into an NPUGraph the way production would replay it.

    Returns (step, out, note): `step` performs one replayed iteration and `out` is
    the tensor the graph writes into, valid after each step.

    For a backend whose prep is host-side, only the attention is captured and prep
    runs eagerly before each replay -- which is what vllm-ascend does today. It
    means the sequence lengths are baked into the captured kernel arguments, so a
    replay with different lengths needs a per-step graph parameter update
    (`update_graph_params` in attention_v1.py). The note says so, because the
    timing below does not include it.
    """
    for _ in range(warmup):
        attn(prep())
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    if name in PREP_ON_DEVICE:
        with torch.npu.graph(graph):
            out = attn(prep())
        return graph.replay, out, ""

    prepared = prep()
    with torch.npu.graph(graph):
        out = attn(prepared)

    def step():
        prep()  # the host sync cannot go inside; it stays per-step, as in production
        graph.replay()

    return step, out, "lengths baked into the graph; needs graph_task_update per step"


def run_profile(name: str, prep, attn, directory: str, iters: int, step=None) -> None:
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

    with torch_npu.profiler.profile(**kwargs):
        # No prof.step(): there is no schedule, so the whole with-block is one
        # recording. Calling step() against no schedule is what produced
        # "Stop profiler while current state is RECORD ... incomplete parsed data".
        for _ in range(iters):
            if step is not None:
                step()
            else:
                attn(prep())
        torch.npu.synchronize()
    torch.npu.synchronize()
    print(f"  profile written to {path} (see its ASCEND_PROFILER_OUTPUT/op_statistic.csv)")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=8, help="requests in the draft batch")
    parser.add_argument("--q-per-req", type=int, default=4, help="draft tokens per request (uniform)")
    # The Qwen3.6 draft's head topology. head_size defaults to 256 because that is
    # the shape this whole comparison exists for: the draft currently runs at 128
    # only because the sink operator cannot serve 256, while Qwen3.6-27B's own head
    # dim is 256. Pass --head-size 128 to reproduce what is deployed today.
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
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
    parser.add_argument(
        "--kv-max",
        type=int,
        default=0,
        help="top of the KV length spread; 0 = the page capacity. Below 1024 keeps "
        "flash decode off, which is the axis v3 and v4 disagree on",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="capture each backend into an NPUGraph and measure replay, which is how "
        "vllm-ascend runs it; eager numbers are dominated by host dispatch",
    )
    parser.add_argument("--csv", help="append one row per backend here, so a sweep accumulates")
    parser.add_argument("--tag", default="", help="free-form label recorded in the CSV, e.g. a build id")
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
        kv_max=args.kv_max or (args.max_blocks_per_seq * args.block_size),
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
    flash_decode = cfg.flash_decode(inp.kv_lens_host)
    print(f"kv lengths  : {inp.kv_lens_host}")
    print(f"flash decode: {flash_decode} (tiling predicate; --kv-max under 1024 turns it off)")

    ref = golden(cfg, inp)
    noise, _ = accuracy(golden(cfg, inp, dtype=cfg.dtype), ref)
    print(f"noise floor : {noise:.3e} (same maths in {args.dtype} on CPU vs fp32)\n")

    rows = []
    for name in names:
        print(f"[{name}]")
        reason = skip_reason(name, cfg, args.force)
        if reason:
            print(f"  skipped: {reason}\n")
            rows.append(Row(name, "skipped", "", "", "", reason))
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
            rows.append(Row(name, "failed", "", "", "", f"{type(exc).__name__}: {exc}"))
            continue

        flow.show(name)
        flow.freeze()

        step = None
        note = ""
        if args.graph:
            try:
                step, out, note = capture(name, prep, attn, args.warmup)
            except Exception as exc:
                # A failed capture leaves the stream in capture mode for the rest of
                # the process, so every later backend would fail for a reason that is
                # not its own. Stop rather than print a column of lies.
                print(f"  capture failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                rows.append(Row(name, "no-capture", "", "", "", f"{type(exc).__name__}: {exc}"))
                print("\n  aborting: a failed capture poisons the stream for the rest of the process")
                break
            step()
            torch.npu.synchronize()

        max_abs, rel = accuracy(out, ref)
        verdict = "ok" if max_abs <= max(noise * 4, 1e-3) else "SUSPECT"
        print(f"  accuracy: max|err|={max_abs:.3e} rel={rel:.3e} vs fp32 golden -> {verdict}")

        if step is not None:
            # One number, not two: replay is the whole captured step, and inside a
            # graph the prep/attn split is no longer something the host can see.
            step_ms = bench(step, args.warmup, args.iters)
            print(f"  timing  : replay={step_ms:.3f} ms/step" + (f"  ({note})" if note else ""))
            prep_ms, attn_ms = "", f"{step_ms:.3f}"
        else:
            prep_v = bench(prep, args.warmup, args.iters)
            prepared = prep()
            attn_v = bench(partial(attn, prepared), args.warmup, args.iters)
            print(f"  timing  : prep={prep_v:.3f} ms  attn={attn_v:.3f} ms  total={prep_v + attn_v:.3f} ms")
            prep_ms, attn_ms = f"{prep_v:.3f}", f"{attn_v:.3f}"

        if args.profile:
            run_profile(name, prep, attn, args.profile_dir, args.profile_iters, step=step)
        print()
        rows.append(Row(name, verdict, f"{max_abs:.3e}", prep_ms, attn_ms, note))

    width = max(len(r.backend) for r in rows) if rows else 8
    if args.graph:
        print("summary (graph replay; the attn column holds the whole captured step)")
    else:
        print("summary (prep = per-step tiling or host sync; attn = per-layer)")
    print(f"  {'backend'.ljust(width)}  {'result':8}  {'max|err|':10}  {'prep ms':8}  {'attn ms':8}  note")
    for row in rows:
        print(
            f"  {row.backend.ljust(width)}  {row.result:8}  {row.max_err:10}  "
            f"{row.prep_ms:8}  {row.attn_ms:8}  {row.note}"
        )

    if args.csv:
        write_csv(args.csv, args.tag, cfg, inp, flash_decode, noise, rows)
        print(f"\nappended {len(rows)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
