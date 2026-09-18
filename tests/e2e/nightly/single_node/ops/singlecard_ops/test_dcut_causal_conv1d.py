import torch
import torch_npu


torch_npu.npu.set_compile_mode(jit_compile=False)


def test_dcut_causal_conv1d_matches_update_mode() -> None:
    """D-Cut keeps the stock update-mode math and state mutation."""
    torch.manual_seed(42)
    device = "npu"
    dtype = torch.bfloat16

    dim = 64
    width = 4
    state_len = 8
    num_cache_slots = 4
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    # These counts describe the previous verifier step and deliberately exceed
    # the current segment length for the second request.
    num_accepted_tokens = torch.tensor([3, 2], dtype=torch.int32, device=device)

    x = torch.randn(3, dim, dtype=dtype, device=device)
    weight = torch.randn(width, dim, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device)
    initial_conv_state = torch.randn(
        num_cache_slots,
        state_len,
        dim,
        dtype=dtype,
        device=device,
    )
    reference_conv_state = initial_conv_state.clone()
    dcut_conv_state = initial_conv_state.clone()
    reference_output = torch.empty_like(x)
    dcut_output = torch.empty_like(x)

    torch.ops._C_ascend.npu_causal_conv1d_custom(
        reference_output,
        x,
        weight,
        conv_state=reference_conv_state,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    torch.ops._C_ascend.npu_dcut_causal_conv1d(
        dcut_output,
        x,
        weight,
        conv_state=dcut_conv_state,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=-1,
    )

    torch.testing.assert_close(dcut_output, reference_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dcut_conv_state, reference_conv_state, rtol=1e-2, atol=1e-2)


def test_dcut_causal_conv1d_multi_round_state_carryover() -> None:
    """Continuous multi-round variable-length verification for the conv1d hook.

    Carries the D-Cut conv state forward across rounds; each round starts the
    stock reference from the same carried state, so a cross-round drift in the
    D-Cut conv-state update surfaces as a per-round mismatch. Covers length
    shorten/recover, previous accepted count > this query length, a request
    reorder (cache_indices permuted with the segments), and cap=0 (len 1)
    requests -- the same scenario as the recurrent multi-round test.
    """
    torch.manual_seed(42)
    device = "npu"
    dtype = torch.bfloat16

    dim = 64
    width = 4
    state_len = 8
    num_requests = 3  # one physical conv slot per permanent request
    generator = torch.Generator().manual_seed(1234)

    weight = torch.randn(width, dim, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device)
    carried_state = torch.randn(
        num_requests,
        state_len,
        dim,
        dtype=dtype,
        device=device,
    )

    # order = batch order (permanent request ids -> conv slot); seg = this round's
    # query length per position; acc = previous round's accepted count per position.
    rounds = [
        dict(order=[0, 1, 2], seg=[3, 2, 2], acc=[1, 1, 1]),
        dict(order=[0, 1, 2], seg=[1, 2, 1], acc=[3, 1, 2]),
        dict(order=[2, 0, 1], seg=[3, 2, 1], acc=[1, 1, 2]),
        dict(order=[0, 1, 2], seg=[2, 1, 1], acc=[2, 1, 3]),
    ]

    for round_index, spec in enumerate(rounds):
        total = sum(spec["seg"])
        query_start_loc = torch.zeros(len(spec["seg"]) + 1, dtype=torch.int32, device=device)
        for i, seg in enumerate(spec["seg"]):
            query_start_loc[i + 1] = query_start_loc[i] + seg
        cache_indices = torch.tensor(spec["order"], dtype=torch.int32, device=device)
        num_accepted_tokens = torch.tensor(spec["acc"], dtype=torch.int32, device=device)
        x = torch.randn(total, dim, generator=generator).to(dtype).to(device)

        # Both ops start this round from the exact same carried conv state.
        reference_conv_state = carried_state.clone()
        dcut_conv_state = carried_state
        reference_output = torch.empty_like(x)
        dcut_output = torch.empty_like(x)

        torch.ops._C_ascend.npu_causal_conv1d_custom(
            reference_output,
            x,
            weight,
            conv_state=reference_conv_state,
            bias_opt=bias,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )
        torch.ops._C_ascend.npu_dcut_causal_conv1d(
            dcut_output,
            x,
            weight,
            conv_state=dcut_conv_state,  # mutated in place, carried to the next round
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=-1,
        )

        torch.testing.assert_close(
            dcut_output,
            reference_output,
            rtol=1e-2,
            atol=1e-2,
            msg=f"round {round_index} output mismatch",
        )
        torch.testing.assert_close(
            dcut_conv_state,
            reference_conv_state,
            rtol=1e-2,
            atol=1e-2,
            msg=f"round {round_index} state mismatch",
        )
        carried_state = dcut_conv_state


def test_dcut_causal_conv1d_live_row_is_unaffected_by_empty_request_rows() -> None:
    """One wide live row beside empty rows must compute what it computes alone.

    This is the model's shape when a full-graph decode descriptor carries more
    request rows than there are requests: row zero holds the whole verification
    window, the rest are empty, and their state index points at the null block
    the allocator never hands out. A single request in that arrangement replays
    at roughly a third of the acceptance it reaches when the descriptor's
    request count matches the live one, with the same live-row inputs either
    way, so something about the arrangement itself has to differ.

    Asserted as an invariance rather than against a golden. Empty rows carry no
    tokens, so they cannot inform anything, and the live row's result must be
    identical with and without them -- a property that needs no commitment to
    how the operator lays out its window, which the other tests in this file
    pin by comparing against the stock operator. The recurrent hook has the
    same check and passes it; this is the other half of the state.

    The two-dimensional index table is the one the model passes: the spec path
    hands this operator spec_state_indices_tensor, a slot per candidate
    position, not one slot per request.
    """
    torch.manual_seed(11)
    device = "npu"
    dtype = torch.bfloat16

    dim = 64
    width = 4
    state_len = 8
    spec_len = 8
    num_tokens = 8
    num_cache_slots = 24

    x = torch.randn(num_tokens, dim, dtype=dtype, device=device)
    weight = torch.randn(width, dim, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device)
    initial_conv_state = torch.randn(
        num_cache_slots, state_len, dim, dtype=dtype, device=device
    )
    num_accepted_alone = torch.ones(1, dtype=torch.int32, device=device)
    num_accepted_padded = torch.ones(8, dtype=torch.int32, device=device)

    # Row zero owns slots [8, 16); the empty rows all address the null slot,
    # exactly as the builder fills them.
    live_slots = list(range(state_len, 2 * state_len))
    null_slots = [0] * spec_len
    indices_alone = torch.tensor([live_slots], dtype=torch.int32, device=device)
    indices_padded = torch.tensor(
        [live_slots] + [null_slots] * 7, dtype=torch.int32, device=device
    )
    # Cumulative, so rows one through seven are empty.
    qsl_alone = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    qsl_padded = torch.tensor([0] + [num_tokens] * 8, dtype=torch.int32, device=device)

    def run(query_start_loc, cache_indices, num_accepted_tokens):
        conv_state = initial_conv_state.clone()
        output = torch.empty_like(x)
        torch.ops._C_ascend.npu_dcut_causal_conv1d(
            output,
            x,
            weight,
            conv_state=conv_state,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=-1,
        )
        torch.npu.synchronize()
        return output, conv_state

    alone_output, alone_state = run(qsl_alone, indices_alone, num_accepted_alone)
    padded_output, padded_state = run(qsl_padded, indices_padded, num_accepted_padded)

    torch.testing.assert_close(padded_output, alone_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        padded_state[live_slots], alone_state[live_slots], rtol=1e-2, atol=1e-2
    )
    # The null slot must come back exactly as it went in: seven empty rows
    # addressed it and none of them may write.
    torch.testing.assert_close(
        padded_state[0], initial_conv_state[0], rtol=0, atol=0
    )
