"""Debug probe: compare flash-attention-npu v4, the FIA sink operator and a torch
reference on one draft-attention call, inside a single process.

Call it from a debugger stopped in `_forward_flash_attn_npu` (or
`_forward_fia_sink`), passing the frame's locals:

    from vllm_ascend.attention.draft_attn_probe import probe; probe(locals())

Running all three on the same tensors is the point: it removes any question of
whether the two backends were fed matching inputs, because they are handed the
same objects, and goes straight to whether they agree on the answer. The torch
reference then says which one is wrong.

This module is a debugging aid. Nothing imports it.
"""

from __future__ import annotations

import torch

SWA_INT_MAX = 2147483647


def _fmt(t, n: int = 8) -> str:
    if not isinstance(t, torch.Tensor):
        return repr(t)
    head = t.flatten()[:n].tolist() if t.numel() else []
    return f"shape={tuple(t.shape)} dtype={t.dtype} stride={tuple(t.stride())} head={head}"


def _flash_decode_predicted(batch, num_heads, num_kv_heads, q_per_req, kv_lens) -> object:
    """The predicate the AICPU tiling uses to switch flash decode on.

    Reproduced rather than read back out of the metadata blob, which would mean
    hand-decoding the C++ struct layout. It is a label on the call.
    """
    try:
        cube = torch.npu.get_stream_limit(torch.npu.current_stream())["cube_core_num"]
    except Exception:
        return "unknown"
    group = num_heads // num_kv_heads
    num_tasks = batch * num_kv_heads
    max_kv = int(max(kv_lens))
    long_seq = num_tasks <= 0.8 * cube and max_kv >= cube * 512
    short_seq = num_tasks <= 0.4 * cube and max_kv >= 1024
    on = (
        q_per_req * group <= 128
        and q_per_req <= 16
        and max_kv >= 1024
        and int(min(kv_lens)) > 0
        and (long_seq or short_seq)
    )
    return f"{on} (cube={cube} num_tasks={num_tasks} max_kv={max_kv})"


def reference(query, key_cache, value_cache, block_table, kv_lens, num_reqs, scale):
    """Gather paged KV by hand and attend with no mask, in fp32.

    No mask because a parallel-drafting step gives every draft position the same
    prefix: the K draft tokens are independent hypotheses, not a sequence, which
    is why both backends run unmasked.
    """
    num_tokens, num_heads, head_size = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    group = num_heads // num_kv_heads
    q_per_req = num_tokens // num_reqs
    out = torch.empty_like(query, dtype=torch.float32)
    for b in range(num_reqs):
        length = int(kv_lens[b])
        pos = torch.arange(length, device=key_cache.device)
        blocks = block_table[b][pos // block_size].long()
        offs = pos % block_size
        k = key_cache[blocks, offs].float()  # (L, Nkv, D)
        v = value_cache[blocks, offs].float()
        k = k.repeat_interleave(group, dim=1)  # (L, H, D)
        v = v.repeat_interleave(group, dim=1)
        q = query[b * q_per_req : (b + 1) * q_per_req].float()  # (S, H, D)
        scores = torch.einsum("qhd,lhd->hql", q, k) * scale
        p = torch.softmax(scores, dim=-1)
        out[b * q_per_req : (b + 1) * q_per_req] = torch.einsum("hql,lhd->qhd", p, v)
    return out


def _diff(name, a, b):
    a32, b32 = a.float(), b.float()
    abs_d = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-6)
    print(
        f"  {name:<22} max_abs={abs_d.max().item():.6f} "
        f"mean_abs={abs_d.mean().item():.6f} "
        f"max_rel={(abs_d / denom).max().item():.6f} "
        f"cos={torch.nn.functional.cosine_similarity(a32.flatten(), b32.flatten(), dim=0).item():.6f}"
    )


def probe(loc: dict, run_sink: bool = True, run_ref: bool = True) -> dict:
    self = loc["self"]
    query = loc["query"]
    md = loc["attn_metadata"]
    key_cache = loc.get("key_cache", getattr(self, "key_cache", None))
    value_cache = loc.get("value_cache", getattr(self, "value_cache", None))
    block_table = loc["block_table"]
    num_tokens = loc["num_tokens"]
    num_reqs = loc["num_reqs"]
    num_block, block_size, num_kv_heads, head_size = key_cache.shape
    q_per_req = num_tokens // num_reqs

    print("=" * 78)
    print("METADATA AS RECEIVED")
    print(f"  num_tokens={num_tokens} num_reqs={num_reqs} q_per_req={q_per_req}")
    print(f"  heads={self.num_heads}/{self.num_kv_heads} head_size={self.head_size} scale={self.scale}")
    print(f"  query_start_loc     {_fmt(md.query_start_loc, 16)}")
    print(f"  seq_lens (full)     {_fmt(md.seq_lens, 16)}")
    print(f"  seq_lens[:num_reqs] {_fmt(md.seq_lens[:num_reqs], 16)}")
    if getattr(md, "slot_mapping", None) is not None:
        print(f"  slot_mapping[:nt]   {_fmt(md.slot_mapping[:num_tokens], 16)}")
    print(f"  block_table         {_fmt(block_table, 8)}")
    print(f"  query               {_fmt(query, 4)}")
    print(f"  key_cache           {_fmt(key_cache, 0)}")
    for k in ("cu_seqlens_q", "seqused_k", "actual_seq_qlen", "actual_seq_kvlen"):
        if k in loc:
            print(f"  {k:<19} {_fmt(loc[k], 16)}")

    kv_lens = md.seq_lens[:num_reqs].to(torch.int32).clamp_min(1)
    print(f"  flash_decode        {_flash_decode_predicted(num_reqs, self.num_heads, self.num_kv_heads, q_per_req, kv_lens.tolist())}")

    results: dict[str, torch.Tensor] = {}

    if "scheduler_metadata" in loc and "module" in loc:
        results["fa_v4"] = loc["module"].flash_attn_varlen_func(
            query, key_cache, value_cache,
            cu_seqlens_q=loc["cu_seqlens_q"], seqused_k=loc["seqused_k"],
            page_table=block_table, max_seqlen_q=loc["max_seqlen_q"],
            max_seqlen_k=loc["max_seqlen_k"], softmax_scale=self.scale,
            causal=False, window_size=(-1, -1),
            scheduler_metadata=loc["scheduler_metadata"], num_splits=0, return_lse=False,
        ).view(num_tokens, self.num_heads, self.head_size)

        # num_splits=1 takes fillCoreInfoNoSplit, i.e. no KV splitting, so the
        # flash-decode split workspace is never addressed. If this one is right and
        # the one above is not, the split path is the fault.
        results["fa_v4_nosplit"] = loc["module"].flash_attn_varlen_func(
            query, key_cache, value_cache,
            cu_seqlens_q=loc["cu_seqlens_q"], seqused_k=loc["seqused_k"],
            page_table=block_table, max_seqlen_q=loc["max_seqlen_q"],
            max_seqlen_k=loc["max_seqlen_k"], softmax_scale=self.scale,
            causal=False, window_size=(-1, -1),
            scheduler_metadata=loc["scheduler_metadata"], num_splits=1, return_lse=False,
        ).view(num_tokens, self.num_heads, self.head_size)

    if run_sink:
        try:
            from vllm_ascend.attention.fia_sink_v1 import (
                _build_fia_sink_seq_tensors,
                _ensure_fia_sink_ops_registered,
            )
            _ensure_fia_sink_ops_registered()
            qlen, kvlen = _build_fia_sink_seq_tensors(num_tokens, md.seq_lens[:num_reqs])
            print(f"  sink actual_seq_qlen  {_fmt(qlen, 16)}")
            print(f"  sink actual_seq_kvlen {_fmt(kvlen, 16)}")
            lim = torch.npu.get_stream_limit(torch.npu.current_stream())
            meta = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
                self.num_heads, self.num_kv_heads, self.head_size, self.head_size,
                actual_seq_lengths=qlen, actual_seq_lengths_kv=kvlen,
                batch_size=num_reqs, sparse_mode=0,
                pre_tokens=SWA_INT_MAX, next_tokens=SWA_INT_MAX,
                input_layout="TND", input_layout_kv="BnBsH", sink_num=0,
                block_size=block_size,
                aic_core_num=lim["cube_core_num"], aiv_core_num=lim["vector_core_num"],
            )
            sink_out, _ = torch.ops.custom.npu_fused_infer_attention_sink(
                query, key_cache.view(num_block, block_size, -1),
                value_cache.view(num_block, block_size, -1),
                actual_seq_qlen=qlen, actual_seq_kvlen=kvlen, block_table=block_table,
                num_query_heads=self.num_heads, num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale, input_layout="TND", sparse_mode=0,
                block_size=block_size, sink_number=0, meta_data=meta,
            )
            results["fia_sink"] = sink_out.view(num_tokens, self.num_heads, self.head_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  [sink unavailable] {type(exc).__name__}: {exc}")

    if run_ref:
        results["reference"] = reference(
            query, key_cache, value_cache, block_table, kv_lens.tolist(), num_reqs, self.scale
        )

    print("-" * 78)
    print("OUTPUT COMPARISON")
    ref = results.get("reference")
    if ref is not None:
        for name, out in results.items():
            if name != "reference":
                _diff(f"{name} vs reference", out, ref)
    if "fa_v4" in results and "fia_sink" in results:
        _diff("fa_v4 vs fia_sink", results["fa_v4"], results["fia_sink"])
    if "fa_v4" in results and "fa_v4_nosplit" in results:
        _diff("fa_v4 vs nosplit", results["fa_v4"], results["fa_v4_nosplit"])

    print("-" * 78)
    print("PER-REQUEST / PER-POSITION max_abs vs reference")
    if ref is not None:
        for name, out in results.items():
            if name == "reference":
                continue
            rows = []
            for b in range(num_reqs):
                for s in range(q_per_req):
                    i = b * q_per_req + s
                    rows.append(f"{(out[i].float() - ref[i].float()).abs().max().item():.4f}")
            print(f"  {name:<16} " + " ".join(rows))
    print("=" * 78)
    return results
