import torch
import torch_npu


torch_npu.npu.set_compile_mode(jit_compile=False)


def _dcut_recurrent_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_float = query.float() * scale
    key_float = key.float()
    value_float = value.float()
    beta_float = beta.float()
    gate_float = g.float().exp()
    state_float = state.float().clone()
    output = torch.zeros_like(value_float)

    num_value_heads = value.shape[1]
    num_key_heads = key.shape[1]
    value_heads_per_key_head = num_value_heads // num_key_heads
    for request_index in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request_index])
        end = int(query_start_loc[request_index + 1])
        accepted_index = int(num_accepted_tokens[request_index]) - 1
        initial_state_index = int(ssm_state_indices[request_index, accepted_index])

        for value_head in range(num_value_heads):
            key_head = value_head // value_heads_per_key_head
            recurrent_state = state_float[initial_state_index, value_head].clone()
            for local_index, token_index in enumerate(range(start, end)):
                recurrent_state *= gate_float[token_index, value_head]
                projected_value = torch.mv(recurrent_state, key_float[token_index, key_head])
                delta = (value_float[token_index, value_head] - projected_value) * beta_float[
                    token_index, value_head
                ]
                recurrent_state += torch.outer(delta, key_float[token_index, key_head])
                output[token_index, value_head] = torch.mv(
                    recurrent_state,
                    query_float[token_index, key_head],
                )
                output_state_index = int(ssm_state_indices[request_index, local_index])
                state_float[output_state_index, value_head] = recurrent_state

    return output.to(value.dtype), state_float.to(state.dtype)


def test_dcut_recurrent_uses_fixed_state_rows_and_zero_padding() -> None:
    torch.manual_seed(42)
    dtype = torch.bfloat16
    num_tokens = 4
    num_key_heads = 2
    num_value_heads = 4
    head_dim = 64

    query = torch.nn.functional.normalize(
        torch.rand(num_tokens, num_key_heads, head_dim),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand(num_tokens, num_key_heads, head_dim),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand(num_tokens, num_value_heads, head_dim).to(dtype)
    beta = torch.rand(num_tokens, num_value_heads).to(dtype)
    g = torch.rand(num_tokens, num_value_heads, dtype=torch.float32)

    # Only the first three token rows are active. The fourth row exercises the
    # graph-safe zero-padded output path.
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    ssm_state_indices = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dtype=torch.int32,
    )
    # Select state from the previous verifier step, independently of this
    # step's [2, 1] segment lengths.
    num_accepted_tokens = torch.tensor([3, 2], dtype=torch.int32)
    state = torch.rand(
        8,
        num_value_heads,
        head_dim,
        head_dim,
        dtype=dtype,
    )
    scale = head_dim**-0.5

    expected_output, expected_state = _dcut_recurrent_golden(
        query,
        key,
        value,
        state,
        beta,
        scale,
        query_start_loc,
        ssm_state_indices,
        num_accepted_tokens,
        g,
    )

    state_npu = state.npu()
    output_npu = torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
        query.npu(),
        key.npu(),
        value.npu(),
        state_npu,
        beta=beta.npu(),
        scale=scale,
        query_start_loc=query_start_loc.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
        g=g.npu(),
        zero_padded_output=True,
    )

    torch.testing.assert_close(output_npu.cpu(), expected_output, rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(state_npu.cpu(), expected_state, rtol=3e-3, atol=1e-2)
    assert torch.count_nonzero(output_npu[-1]).item() == 0


def _make_round_inputs(
    seg_lens,
    block_ids,
    accepted,
    state_len,
    num_key_heads,
    num_value_heads,
    head_dim,
    dtype,
    generator,
):
    """Build one verifier round's inputs.

    ``block_ids`` is the batch order for this round: entry ``p`` is the permanent
    request id occupying batch position ``p``. Each request owns the physical
    state rows ``[block_id * state_len, ..., block_id * state_len + state_len - 1]``,
    so reordering requests just permutes the ``[., state_len]`` rows and segment
    lengths together (design doc 1.2). ``accepted[p]`` is the previous round's
    accepted count for that request; the kernel starts from row
    ``block_id * state_len + accepted - 1`` regardless of this round's segment
    length (design doc 1.3).
    """
    total = sum(seg_lens)
    query = torch.nn.functional.normalize(
        torch.rand(total, num_key_heads, head_dim, generator=generator),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand(total, num_key_heads, head_dim, generator=generator),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand(total, num_value_heads, head_dim, generator=generator).to(dtype)
    beta = torch.rand(total, num_value_heads, generator=generator).to(dtype)
    g = torch.rand(total, num_value_heads, generator=generator, dtype=torch.float32)

    query_start_loc = torch.zeros(len(seg_lens) + 1, dtype=torch.int32)
    for i, seg in enumerate(seg_lens):
        query_start_loc[i + 1] = query_start_loc[i] + seg
    ssm_state_indices = torch.tensor(
        [[block * state_len + col for col in range(state_len)] for block in block_ids],
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.tensor(accepted, dtype=torch.int32)
    return (
        query,
        key,
        value,
        beta,
        g,
        query_start_loc,
        ssm_state_indices,
        num_accepted_tokens,
    )


def test_dcut_recurrent_multi_round_state_carryover() -> None:
    """Continuous multi-round variable-length verification.

    Feeds the NPU state forward across rounds and, each round, recomputes the
    golden from the NPU's pre-round state so the two trajectories stay comparable
    (no compounding bf16 drift). This pins the cross-round contract the single
    round op test cannot: every round starts from the physical row the *previous*
    round actually wrote, so a wrong start-row / output-row index surfaces here.

    Scenario covers: length shorten then recover, previous accepted count greater
    than this round's query length (design doc 1.3), a request reorder (design
    doc 1.2), and cap=0 requests (segment length 1, design doc 1.5).
    """
    dtype = torch.bfloat16
    num_key_heads = 2
    num_value_heads = 4
    head_dim = 64
    state_len = 4  # candidate rows per request; >= max segment len and max accepted
    num_requests = 3
    num_rows = num_requests * state_len
    scale = head_dim**-0.5

    generator = torch.Generator().manual_seed(1234)
    state_npu = (
        torch.rand(
            num_rows,
            num_value_heads,
            head_dim,
            head_dim,
            generator=generator,
            dtype=torch.float32,
        )
        .to(dtype)
        .npu()
    )

    # order = batch order (permanent request ids); seg = this round's query length
    # per position; acc = previous round's accepted count per position. acc[p] must
    # be <= that request's previous seg (else it would read a stale, unwritten row).
    rounds = [
        # seed round: every request reads its row 0
        dict(order=[0, 1, 2], seg=[3, 2, 2], acc=[1, 1, 1]),
        # shorten; previous accepted (3 and 2) exceed this query length (1 and 1)
        dict(order=[0, 1, 2], seg=[1, 2, 1], acc=[3, 1, 2]),
        # reorder to [2, 0, 1]; recover length; request 1 is cap=0 (len 1) with acc 2
        dict(order=[2, 0, 1], seg=[3, 2, 1], acc=[1, 1, 2]),
        # back to [0, 1, 2]; request 2 has acc 3 > query length 1
        dict(order=[0, 1, 2], seg=[2, 1, 1], acc=[2, 1, 3]),
    ]

    for round_index, spec in enumerate(rounds):
        (
            query,
            key,
            value,
            beta,
            g,
            query_start_loc,
            ssm_state_indices,
            num_accepted_tokens,
        ) = _make_round_inputs(
            spec["seg"],
            spec["order"],
            spec["acc"],
            state_len,
            num_key_heads,
            num_value_heads,
            head_dim,
            dtype,
            generator,
        )

        # Golden starts from the exact state the NPU carries into this round.
        pre_round_state = state_npu.cpu()
        expected_output, expected_state = _dcut_recurrent_golden(
            query,
            key,
            value,
            pre_round_state,
            beta,
            scale,
            query_start_loc,
            ssm_state_indices,
            num_accepted_tokens,
            g,
        )

        output_npu = torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
            query.npu(),
            key.npu(),
            value.npu(),
            state_npu,  # mutated in place, carried to the next round
            beta=beta.npu(),
            scale=scale,
            query_start_loc=query_start_loc.npu(),
            ssm_state_indices=ssm_state_indices.npu(),
            num_accepted_tokens=num_accepted_tokens.npu(),
            g=g.npu(),
            zero_padded_output=False,
        )

        torch.testing.assert_close(
            output_npu.cpu(),
            expected_output,
            rtol=3e-3,
            atol=1e-2,
            msg=f"round {round_index} output mismatch",
        )
        torch.testing.assert_close(
            state_npu.cpu(),
            expected_state,
            rtol=3e-3,
            atol=1e-2,
            msg=f"round {round_index} state mismatch",
        )
