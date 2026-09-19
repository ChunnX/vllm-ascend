# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests. Also runnable with --noconftest outside an NPU install.

Only vLLM device infrastructure is stubbed; policy, manager methods and their
CPU/device tensor copies execute the production source with real CPU torch.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PKG = "vllm_ascend.worker.v2.spec_decode.dspark"


def load(monkeypatch, name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy(monkeypatch):
    return load(monkeypatch, f"{PKG}.eager_policy", "vllm_ascend/worker/v2/spec_decode/dspark/eager_policy.py")


@pytest.mark.parametrize("threshold,expected", [(0, [4, 4]), (0.4, [2, 1]), (1, [0, 0])])
def test_survival_prefix(policy, threshold, expected):
    confidence = np.array([[0.9, 0.8, 0.4, 0.2], [0.8, 0.4, 0.9, 1]])
    np.testing.assert_array_equal(policy.select_capacities(confidence, [4, 4], threshold, 8), expected)


def test_scheduled_limit_and_stable_tied_global_budget(policy):
    np.testing.assert_array_equal(policy.select_capacities(np.ones((3, 4)), [1, 4, 0], 0, 3), [1, 2, 0])
    np.testing.assert_array_equal(policy.select_capacities(np.ones((3, 4)), [1, 4, 0], 0, 0), [0, 0, 0])


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold(policy, threshold):
    with pytest.raises(ValueError):
        policy.select_capacities(np.ones((1, 2)), [2], threshold, 2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_confidence(policy, bad):
    with pytest.raises(ValueError):
        policy.select_capacities(np.array([[0.5, bad]]), [2], 0.4, 2)


def test_layout_after_reorder_and_prefill(policy):
    caps, queries, qsl, logits = policy.batch_layout(
        ["prefill", "b", "a"], {"a": 2, "b": 0, "prefill": 0}, {"a": 1, "b": 1, "prefill": 5}, 1
    )
    np.testing.assert_array_equal(caps, [0, 0, 2])
    np.testing.assert_array_equal(queries, [5, 1, 3])
    np.testing.assert_array_equal(qsl, [0, 5, 6, 9])
    np.testing.assert_array_equal(logits, [0, 1, 2, 5])


@pytest.fixture
def manager_class(monkeypatch, policy):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("vllm.distributed", get_tp_group=lambda: SimpleNamespace(broadcast=lambda tensor, src: tensor))
    stub("vllm.logger", logger=SimpleNamespace(debug=lambda *args: None, info=lambda *args: None))
    stub("vllm.v1.worker.gpu.buffer_utils", async_copy_to_gpu=lambda value, out: out.copy_(torch.from_numpy(value)))
    stub("vllm.v1.worker.gpu.spec_decode.adaptive_verification", AdaptiveVerificationManager=object)
    return load(
        monkeypatch, f"{PKG}.eager_verification", "vllm_ascend/worker/v2/spec_decode/dspark/eager_verification.py"
    ).EagerSurvivalVerificationManager


def make_manager(cls, max_logits=100):
    states = SimpleNamespace(num_speculative_steps=4, max_num_reqs=4, req_id_to_index={"a": 2, "b": 0, "p": 1})
    return cls(states, torch.empty(6, dtype=torch.int32), 1, max_logits, threshold=0.4)


def publish(manager):
    manager.record_confidences(
        torch.tensor([[0.9, 0.8, 0.4, 0.2], [0.3, 1, 1, 1]]),
        SimpleNamespace(num_reqs=2, idx_mapping_np=np.array([2, 0])),
    )


def test_manager_live_confidence_reorder_exact_boundaries(manager_class):
    manager = make_manager(manager_class)
    publish(manager)
    assert manager.get_num_tokens({"a": 5, "b": 5, "p": 6}, {"a": [-1] * 4, "b": [-1] * 4}) == 10
    manager.prepare_request_order(["b", "a", "p"])
    queries, cpu_logits = manager.compact_batch(np.array([4, 4, 0]), np.array([5, 5, 6]), None)
    device_logits, qsl, budget = manager.reallocate_drafts(["b", "a", "p"], torch.tensor([0, 2, 1]))
    np.testing.assert_array_equal(queries, [1, 3, 6])
    np.testing.assert_array_equal(qsl.numpy(), [0, 1, 4, 10, 10, 10])
    np.testing.assert_array_equal(cpu_logits, [0, 1, 4, 5])
    torch.testing.assert_close(device_logits, torch.tensor(cpu_logits))
    assert budget == 2
    with pytest.raises(RuntimeError):
        manager.reallocate_drafts(["b", "a", "p"], torch.tensor([0, 2, 1]))


def test_new_slot_missing_and_consumed_confidence_fall_back(manager_class):
    manager = make_manager(manager_class)
    publish(manager)
    manager.add_request(0)  # slot b reused: no inherited 0.3 confidence
    assert manager.get_num_tokens({"b": 5}, {"b": [-1] * 4}) == 5
    assert manager.get_num_tokens({"a": 5}, {"a": [-1] * 4}) == 3
    assert manager.get_num_tokens({"a": 5}, {"a": [-1] * 4}) == 5


def test_manager_total_logits_bound(manager_class):
    manager = make_manager(manager_class, max_logits=3)
    assert manager.get_num_tokens({"a": 5, "b": 5}, {"a": [-1] * 4, "b": [-1] * 4}) == 3
    manager.prepare_request_order(["a", "b"])
    queries, logits = manager.compact_batch(None, None, None)
    assert logits[-1] == 3
    np.testing.assert_array_equal(queries, [2, 1])


def test_refuse_reorder_between_compact_and_reallocate(manager_class):
    manager = make_manager(manager_class)
    manager.get_num_tokens({"a": 5, "b": 5}, {"a": [-1] * 4, "b": [-1] * 4})
    manager.prepare_request_order(["a", "b"])
    with pytest.raises(RuntimeError):
        manager.reallocate_drafts(["b", "a"], torch.tensor([0, 2]))


def config_gate(policy, value):
    # Load the real guard without importing the Ascend platform bootstrap.
    import ast

    source = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/eager_config.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD=value),
        "validate_threshold": policy.validate_threshold,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["eager_survival_threshold"]


def eager_config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark", enable_adaptive_verification=True, enforce_eager=True, num_speculative_tokens=7
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        lora_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1, enable_dbo=False
        ),
    )


def test_config_opt_in_only(policy):
    assert config_gate(policy, None)(object()) is None
    assert config_gate(policy, 0.4)(eager_config()) == 0.4


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model_config", "enforce_eager", False),
        ("speculative_config", "enforce_eager", False),
        ("speculative_config", "enable_adaptive_verification", False),
        ("speculative_config", "method", "dflash"),
        ("speculative_config", "num_speculative_tokens", 16),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_dbo", True),
    ],
)
def test_config_rejects_unvalidated_modes(policy, section, key, value):
    config = eager_config()
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        config_gate(policy, 0.4)(config)


def load_method(relative, class_name, method_name, namespace):
    import ast

    path = ROOT / relative
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    # Postpone annotations so only execution dependencies need CPU stand-ins.
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[method_name]


def test_zero_draft_decode_preserves_previous_accepted_selector(monkeypatch, policy):
    config_module = types.ModuleType(f"{PKG}.eager_config")
    config_module.eager_survival_threshold = lambda _: 0.4
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    method = load_method(
        "vllm_ascend/worker/v2/model_states/mamba_hybrid.py",
        "AscendMambaHybridModelState",
        "prepare_attn",
        {
            "torch": torch,
            "np": np,
            "CUDAGraphMode": SimpleNamespace(FULL="full"),
            "MambaHybridAttnMetadata": lambda **kwargs: SimpleNamespace(**kwargs),
            "build_attn_metadata": lambda **kwargs: kwargs,
        },
    )
    state = SimpleNamespace(
        vllm_config=SimpleNamespace(num_speculative_tokens=7),
        num_accepted_tokens_gpu=torch.tensor([2, 6, 1]),
        max_model_len=100,
    )
    # Row a has no scheduled drafts but six accepted outputs last round. Row b
    # has one draft; row p is genuine prefill and must stay outside spec decode.
    batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=7,
        is_prefilling_np=np.array([False, False, True]),
        idx_mapping=torch.tensor([1, 0, 2]),
        num_scheduled_tokens=np.array([1, 2, 4]),
        num_draft_tokens_per_req=np.array([0, 1, 0]),
        query_start_loc=torch.tensor([0, 1, 3, 7]),
        query_start_loc_np=np.array([0, 1, 3, 7]),
        seq_lens=torch.tensor([30, 20, 4]),
        dcp_local_seq_lens=None,
        seq_lens_np=np.array([30, 20, 4]),
        positions=None,
        attn_state=None,
    )
    result = method(state, batch, "none", (), torch.empty(0), [], None)
    metadata = result["model_specific_attn_metadata"]
    torch.testing.assert_close(metadata.num_accepted_tokens, torch.tensor([6, 2, 1]))
    np.testing.assert_array_equal(metadata.num_decode_draft_tokens_cpu.numpy(), [0, 1, -1])


def test_target_graph_factory_cannot_upgrade_eager_lane():
    import ast
    from contextlib import contextmanager

    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "graph_manager_wrapper")
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    original = object()
    upstream = SimpleNamespace(ModelCudaGraphManager=original)
    namespace = {
        "contextmanager": contextmanager,
        "vllm_model_runner": upstream,
        "CUDAGraphMode": SimpleNamespace(NONE="none"),
        "ModelAclGraphManager": lambda *args, **kwargs: (args, kwargs),
    }
    exec(compile(code, str(path), "exec"), namespace)
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode="full_and_piecewise"))
    with namespace["graph_manager_wrapper"](SimpleNamespace(eager_survival_test=True)):
        args, _ = upstream.ModelCudaGraphManager(config, "cpu", "full_and_piecewise", 8, varlen_decode=True)
        assert args[2] == "none"
        assert config.compilation_config.cudagraph_mode == "none"
    assert upstream.ModelCudaGraphManager is original
