# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent NumPy references for D-Cut speculative conv and recurrence.

No device kernels, torch, vLLM, or tiling code are called. Inputs are already
normalized/activated where the operator API requires that. Arithmetic is FP64;
callers quantize outputs/state to the device storage dtype BETWEEN rounds.
These references implement the legal width-4 speculative contract, not kernel
fallbacks for undersized state buffers or malformed metadata.
"""

import numpy as np


def _segments(query_start_loc: np.ndarray, num_tokens: int) -> list[tuple[int, int]]:
    qsl = np.asarray(query_start_loc)
    if qsl.ndim != 1 or len(qsl) == 0 or not np.issubdtype(qsl.dtype, np.integer):
        raise ValueError("query_start_loc must be a nonempty integer cumsum vector")
    if qsl[0] != 0 or np.any(np.diff(qsl) < 0) or qsl[-1] > num_tokens:
        raise ValueError("query_start_loc must start at zero and stay within the token axis")
    return [(int(start), int(end)) for start, end in zip(qsl[:-1], qsl[1:])]


def speculative_conv_reference(
    x: np.ndarray,
    weight: np.ndarray,
    state: np.ndarray,
    query_start_loc: np.ndarray,
    cache_indices: np.ndarray,
    num_accepted: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    silu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Depthwise causal convolution and rollback history, with weight [4, C].

    Each active row reads three history samples at ``accepted - 1`` from its
    physical conv slot. It writes ``history[-2:] + this_round_x`` at offset 0;
    the remaining state is untouched. Thus next round can select ANY accepted
    prefix, independently of that next round's query length. A 2D cache table
    uses column 0; this is distinct from recurrent's per-token state table.
    Unused output tokens are zero in this reference; conv's device output tail
    is caller-owned and must be initialized if a full-tensor comparison is used.
    """
    x, weight, state = (np.asarray(a, dtype=np.float64) for a in (x, weight, state))
    segments = _segments(query_start_loc, len(x))
    indices = np.asarray(cache_indices)
    if indices.ndim == 2:
        indices = indices[:, 0]
    if indices.shape != (len(segments),) or np.shape(num_accepted) != (len(segments),):
        raise ValueError("request metadata must match the cumsum request axis")
    if x.ndim != 2 or weight.shape != (4, x.shape[1]) or state.ndim != 3 or state.shape[2] != x.shape[1]:
        raise ValueError("expected x [T,C], weight [4,C], state [slots,L,C]")
    output, updated = np.zeros_like(x), state.copy()
    active_slots = set()
    for row, (start, end) in enumerate(segments):
        if start == end:
            continue  # No accepted/index access for inactive rows (including -1).
        slot, offset = int(indices[row]), int(num_accepted[row]) - 1
        if slot < 0 or slot >= len(state) or slot in active_slots:
            raise ValueError("active conv requests need distinct valid cache slots")
        active_slots.add(slot)
        if offset < 0 or offset + 3 > state.shape[1] or end - start + 2 > state.shape[1]:
            raise ValueError("conv state cannot hold the selected history/candidate queries")
        history = state[slot, offset : offset + 3]
        sequence = np.concatenate((history, x[start:end]), axis=0)
        for local, token in enumerate(range(start, end)):
            y = (sequence[local : local + 4] * weight).sum(axis=0)
            if bias is not None:
                y = y + bias
            if silu:
                y = y * np.exp(-np.logaddexp(0.0, -y))
            output[token] = y
        candidates = sequence[1:]
        updated[slot, : len(candidates)] = candidates
    return output, updated


def recurrent_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    state: np.ndarray,
    beta: np.ndarray,
    g: np.ndarray,
    query_start_loc: np.ndarray,
    ssm_state_indices: np.ndarray,
    num_accepted: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential gated delta rule; H is [value_dim, key_dim].

    D = exp(g_t) * H; H = D + beta_t * outer(v_t - D @ k_t, k_t);
    o_t = H @ (scale * q_t). State starts at indices[row, accepted-1]
    and candidate t writes indices[row, t]. Neither the table stride nor the
    accepted count is shortened with this round's queries. GQA maps contiguous
    groups of value heads to a key head. g is scalar log-decay per value head.
    """
    query, key, value, state, beta, g = (np.asarray(a, dtype=np.float64) for a in (query, key, value, state, beta, g))
    segments = _segments(query_start_loc, len(query))
    indices = np.asarray(ssm_state_indices)
    if indices.ndim != 2 or indices.shape[0] != len(segments):
        raise ValueError("ssm_state_indices must retain shape [B,S]")
    if np.shape(num_accepted) != (len(segments),):
        raise ValueError("accepted counts must match B")
    tokens, key_heads, key_dim = query.shape
    value_heads, value_dim = value.shape[1:]
    if key.shape != query.shape or value.shape[0] != tokens or value_heads % key_heads:
        raise ValueError("incompatible Q/K/V token/head shapes")
    if (
        state.shape[1:] != (value_heads, value_dim, key_dim)
        or beta.shape != (tokens, value_heads)
        or g.shape != beta.shape
    ):
        raise ValueError("incompatible state/beta/g shapes")
    output, updated = np.zeros_like(value), state.copy()
    occupied = set()
    for row, (start, end) in enumerate(segments):
        if start == end:
            continue
        accepted = int(num_accepted[row])
        if not 1 <= accepted <= indices.shape[1] or end - start > indices.shape[1]:
            raise ValueError("accepted and query width must each fit S, independently")
        source = int(indices[row, accepted - 1])
        targets = indices[row, : end - start]
        touched = {source, *(int(index) for index in targets)}
        if min(touched) < 0 or max(touched) >= len(state) or occupied.intersection(touched):
            raise ValueError("active requests must address valid, disjoint state rows")
        if len(set(targets.tolist())) != len(targets):
            raise ValueError("candidate states must use distinct rows")
        occupied.update(touched)
        for head in range(value_heads):
            key_head = head // (value_heads // key_heads)
            h = state[source, head].copy()
            for local, token in enumerate(range(start, end)):
                k = key[token, key_head]
                decayed = np.exp(g[token, head]) * h
                residual = value[token, head] - decayed @ k
                h = decayed + beta[token, head] * np.outer(residual, k)
                output[token, head] = h @ (scale * query[token, key_head])
                updated[int(targets[local]), head] = h
    return output, updated
