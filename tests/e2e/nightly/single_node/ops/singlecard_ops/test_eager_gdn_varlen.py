# SPDX-License-Identifier: Apache-2.0
"""Independent CPU-golden gates for the eager adaptive GDN operators."""

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.helpers.dcut_reference import recurrent_reference, speculative_conv_reference
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def initialize_custom_operators():
    assert enable_custom_op(), "This gate requires the rebuilt Ascend custom operators"
    torch.npu.set_compile_mode(jit_compile=False)


@pytest.mark.parametrize("lengths", [[8], [8, 0, 0, 0, 0, 0, 0, 0], [3, 1, 4], [1, 1, 1]])
def test_conv_varlen_matches_cpu_golden(lengths):
    torch.manual_seed(31)
    dim, width, history = 64, 4, 10
    rows = len(lengths)
    qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    x = torch.randn(int(qsl[-1]), dim, dtype=torch.bfloat16)
    weight = torch.randn(width, dim, dtype=torch.bfloat16) * 0.2
    bias = torch.randn(dim, dtype=torch.bfloat16) * 0.1
    state = torch.randn(rows + 1, history, dim, dtype=torch.bfloat16)
    indices = np.arange(1, rows + 1, dtype=np.int32)
    indices[np.array(lengths) == 0] = -1
    accepted = np.full(rows, 6, dtype=np.int32)  # History selector may exceed current length.
    expected, expected_state = speculative_conv_reference(
        x.float().numpy(),
        weight.float().numpy(),
        state.float().numpy(),
        qsl,
        indices,
        accepted,
        bias=bias.float().numpy(),
    )
    device_state = state.npu()
    output = torch.empty_like(x, device="npu")
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x.npu(),
        weight.npu(),
        conv_state=device_state,
        bias_opt=bias.npu(),
        query_start_loc_opt=torch.from_numpy(qsl).npu(),
        cache_indices_opt=torch.from_numpy(indices).npu(),
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=torch.from_numpy(accepted).npu(),
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), torch.from_numpy(expected).float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(device_state.cpu().float(), torch.from_numpy(expected_state).float(), rtol=0, atol=0)


def test_recurrent_multiround_ragged_state_matches_cpu_golden():
    torch.manual_seed(17)
    rows, stride, nk, nv, dim = 3, 8, 2, 4, 64
    indices = np.arange(1, rows * stride + 1, dtype=np.int32).reshape(rows, stride)
    state = torch.randn(rows * stride + 1, nv, dim, dim, dtype=torch.float32) * 0.1
    device_state = state.npu()
    # accepted is the previous round's output count, independent of this round's width.
    rounds = [([8, 8, 8], [1, 1, 1]), ([3, 1, 4], [6, 8, 3]), ([1, 4, 0], [3, 1, 4])]
    for lengths, accepted in rounds:
        qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
        tokens = int(qsl[-1])
        q = torch.nn.functional.normalize(torch.randn(tokens, nk, dim), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(torch.randn(tokens, nk, dim), dim=-1).to(torch.bfloat16)
        v = torch.randn(tokens, nv, dim, dtype=torch.bfloat16) * 0.2
        beta = torch.rand(tokens, nv, dtype=torch.bfloat16)
        g = -torch.rand(tokens, nv, dtype=torch.float32) * 0.1
        table = indices.copy()
        table[np.array(lengths) == 0] = -1
        expected, expected_state = recurrent_reference(
            q.float().numpy(),
            k.float().numpy(),
            v.float().numpy(),
            state.numpy(),
            beta.float().numpy(),
            g.numpy(),
            qsl,
            table,
            np.array(accepted),
            scale=dim**-0.5,
        )
        out = torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
            q.npu(),
            k.npu(),
            v.npu(),
            device_state,
            beta=beta.npu(),
            g=g.npu(),
            scale=dim**-0.5,
            query_start_loc=torch.from_numpy(qsl).npu(),
            ssm_state_indices=torch.from_numpy(table).npu(),
            num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device="npu"),
        )
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu().float(), torch.from_numpy(expected).float(), rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(device_state.cpu(), torch.from_numpy(expected_state).float(), rtol=2e-2, atol=2e-3)
        # Keep the independent golden state across rounds; never seed it from DUT.
        state = torch.from_numpy(expected_state).float()
