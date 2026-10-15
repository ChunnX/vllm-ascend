# SPDX-License-Identifier: Apache-2.0
"""DCUT parity between dense state and MRV2 padded physical pages."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def initialize_custom_operators():
    assert enable_custom_op(), "This test requires rebuilt DCUT custom operators"
    torch.npu.set_compile_mode(jit_compile=False)


def _execute(run, reset, graph_mode):
    if not graph_mode:
        return run()
    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    reset()
    graph.replay()
    torch.npu.synchronize()
    return output


def _state_pair(shape, dtype):
    initial = torch.randn(shape, dtype=dtype, device="npu") * 0.1
    row_elements = initial[0].numel()
    # An offset and a stride not divisible by the logical row size catch
    # accidental dense addressing and integer-multiple remapping shortcuts.
    backing = torch.full((shape[0], row_elements + 32), -123.0, dtype=dtype, device="npu")
    padded = backing[:, 16 : 16 + row_elements].view(shape)
    padded.copy_(initial)
    return initial, initial.clone(), padded, backing


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dcut_recurrent_padded_state(dtype, graph_mode):
    torch.manual_seed(35)
    initial, dense, padded, backing = _state_pair((16, 2, 16, 16), dtype)
    query, key, value = [torch.randn(5, 2, 16, dtype=torch.bfloat16, device="npu") * 0.1 for _ in range(3)]
    kwargs = dict(
        beta=torch.full((5, 2), 0.5, dtype=torch.bfloat16, device="npu"),
        g=torch.full((5, 2), -0.1, dtype=torch.float32, device="npu"),
        scale=0.25,
        query_start_loc=torch.tensor([0, 2, 2, 3], dtype=torch.int32, device="npu"),
        ssm_state_indices=torch.tensor(
            [[3, 4, 5, 6], [-1, -1, -1, -1], [11, 12, 13, 14]], dtype=torch.int32, device="npu"
        ),
        num_accepted_tokens=torch.tensor([3, 1, 2], dtype=torch.int32, device="npu"),
        zero_padded_output=True,
    )
    outputs = []
    for state in (dense, padded):

        def run(state=state):
            return torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(query, key, value, state, **kwargs)

        outputs.append(_execute(run, lambda state=state: state.copy_(initial), graph_mode))
    torch.testing.assert_close(outputs[1], outputs[0])
    torch.testing.assert_close(padded, dense)
    assert torch.isfinite(outputs[1]).all()
    assert torch.all(backing[:, :16] == -123)
    assert torch.all(backing[:, -16:] == -123)


@pytest.mark.parametrize("graph_mode", [False, True])
def test_dcut_conv_padded_state(graph_mode):
    torch.manual_seed(35)
    initial, dense, padded, backing = _state_pair((16, 7, 32), torch.bfloat16)
    x = torch.randn(5, 32, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(3, 32, dtype=torch.bfloat16, device="npu")
    query_start_loc = torch.tensor([0, 2, 2, 3], dtype=torch.int32, device="npu")
    indices = torch.tensor([[3, 4, 5, 6], [-1, -1, -1, -1], [11, 12, 13, 14]], dtype=torch.int32, device="npu")
    accepted = torch.tensor([3, 1, 2], dtype=torch.int32, device="npu")
    outputs = []
    for state in (dense, padded):

        def run(state=state):
            output = torch.zeros_like(x)
            torch.ops._C_ascend.npu_dcut_causal_conv1d(
                output,
                x,
                weight,
                state,
                query_start_loc=query_start_loc,
                cache_indices=indices,
                num_accepted_tokens=accepted,
                activation_mode=1,
                pad_slot_id=-1,
            )
            return output

        outputs.append(_execute(run, lambda state=state: state.copy_(initial), graph_mode))
    torch.testing.assert_close(outputs[1][:3], outputs[0][:3])
    torch.testing.assert_close(padded, dense)
    assert torch.isfinite(outputs[1][:3]).all()
    assert torch.all(backing[:, :16] == -123)
    assert torch.all(backing[:, -16:] == -123)
