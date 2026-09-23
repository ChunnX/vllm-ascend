# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests. Also runnable with --noconftest outside an NPU install.

Only vLLM device infrastructure is stubbed; policy, manager methods and their
CPU/device tensor copies execute the production source with real CPU torch.
"""

import dataclasses
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PKG = "vllm_ascend.worker.v2.spec_decode.dspark"

# Every Ascend env the eager lanes read. Keep this complete: a missing name
# raises AttributeError from the stub rather than being defaulted, which is the
# point -- adding a lane variable without teaching these tests about it should
# fail here and not silently skip the guard it gates.
STUB_ENVS = {
    "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD": 0.4,
    "VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV": False,
    "VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL": 50,
    "VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND": False,
    "VLLM_ASCEND_DSPARK_AV_GRAPH": "none",
    "VLLM_ASCEND_DSPARK_AV_ADAPT": True,
    "VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS": False,
}


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
    stub("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    stub("vllm.v1.worker.gpu.buffer_utils", async_copy_to_gpu=lambda value, out: out.copy_(torch.from_numpy(value)))
    stub("vllm.v1.worker.gpu.spec_decode.adaptive_verification", AdaptiveVerificationManager=object)
    # ``import vllm_ascend.envs`` binds through the parent package, so the real
    # ``vllm_ascend/__init__.py`` -- and with it the platform bootstrap that
    # needs an installed vLLM -- would execute. Stub both halves.
    envs_module = stub("vllm_ascend.envs", **STUB_ENVS)
    stub("vllm_ascend", envs=envs_module)
    load(monkeypatch, f"{PKG}.eager_av_log", "vllm_ascend/worker/v2/spec_decode/dspark/eager_av_log.py")
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


@pytest.fixture
def av_logger_class(monkeypatch):
    module = types.ModuleType("vllm.logger")
    module.init_logger = lambda name: logging.getLogger(name)
    monkeypatch.setitem(sys.modules, "vllm.logger", module)
    return load(
        monkeypatch, f"{PKG}.eager_av_log", "vllm_ascend/worker/v2/spec_decode/dspark/eager_av_log.py"
    ).EagerAVLogger


def test_av_logger_emits_once_per_interval(av_logger_class, caplog):
    log = av_logger_class(lane="threshold", interval=3)
    with caplog.at_level(logging.WARNING):
        for _ in range(7):
            log.record(num_reqs=2, scheduled_drafts=8, admitted_drafts=4, verify_tokens=6, capacities=[3, 1])
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 3  # step 1, then steps 4 and 7 close each interval
    assert "[DSPARK-EAGER-AV/threshold]" in lines[0]
    assert "kept=50.0%" in lines[0]
    assert "last_caps=[3, 1]" in lines[0]
    assert "3 steps" in lines[1] and "trimmed_steps=3/3" in lines[1]


def test_av_logger_always_reports_a_run_shorter_than_one_interval(av_logger_class, caplog):
    """A short run must still produce a data line, not only the banner.

    The whole-network gate compares a lane against the fixed-K baseline over a
    few dozen decode steps. With the serve-oriented interval that is under one
    window, so without this the run would finish having logged only the
    construction banner -- which proves the manager exists, not that it ever
    trimmed. Equal output would then say nothing about the trimmed path.
    """
    log = av_logger_class(lane="upstream", interval=50)
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            log.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=3, verify_tokens=4)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "1 steps" in messages[0]
    assert "kept=42.9%" in messages[0]


def test_av_logger_reports_which_graph_each_step_earned(av_logger_class, caplog):
    """A match under a graph mode does not say a graph ran.

    A trimmed batch matches no uniform descriptor and falls back by design, and
    at a few dozen decode steps the wall clock cannot separate a replayed graph
    from one that was captured and never entered. So the line has to say.
    """
    log = av_logger_class(lane="threshold", interval=3)
    with caplog.at_level(logging.WARNING):
        log.note_graph_mode(SimpleNamespace(name="FULL"))
        log.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=7, verify_tokens=8)
        for _ in range(2):
            log.note_graph_mode(SimpleNamespace(name="NONE"))
            log.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=3, verify_tokens=4)
        log.note_graph_mode(SimpleNamespace(name="FULL"))
        log.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=7, verify_tokens=8)
    first, second = (r.getMessage() for r in caplog.records)
    assert "graph=FULL=1" in first
    # The window that closed next held two fallbacks and one replay.
    assert "graph=FULL=1,NONE=2" in second
    # Counts reset with the window, so the second line does not re-report the first.
    log2 = av_logger_class(lane="threshold", interval=1)
    with caplog.at_level(logging.WARNING):
        log2.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=7, verify_tokens=8)
    assert "graph=" not in caplog.records[-1].getMessage()


def test_av_logger_separates_a_live_confidence_from_a_frozen_one(av_logger_class, caplog):
    """The one failure mode no output comparison can see.

    If the confidence op is not traced into the captured draft graph, every
    replay skips it and the buffer stays at its pre-capture value. Trimming is
    only a policy, so the rejection sampler stays correct and the tokens are
    byte-identical to a fixed-K run -- while every budget is decided on numbers
    that stopped moving. So the line reports how often the signal moved, and as
    n/N rather than a flag: the caller counts on device, because a copy per step
    to measure this would cost more than the thing being measured.
    """
    log = av_logger_class(lane="upstream", interval=4)
    with caplog.at_level(logging.WARNING):
        log.note_confidence_steps(moved=50, steps=50)
        log.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=4, verify_tokens=5)
    assert "conf_moved=50/50" in caplog.records[-1].getMessage()

    frozen = av_logger_class(lane="upstream", interval=4)
    with caplog.at_level(logging.WARNING):
        frozen.note_confidence_steps(moved=0, steps=50)
        frozen.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=4, verify_tokens=5)
    assert "conf_moved=0/50" in caplog.records[-1].getMessage()

    # A lane that never reports one must not grow an empty field.
    silent = av_logger_class(lane="threshold", interval=1)
    with caplog.at_level(logging.WARNING):
        silent.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=7, verify_tokens=8)
    assert "conf_moved" not in caplog.records[-1].getMessage()


def test_av_logger_forgets_confidence_counts_between_windows(av_logger_class, caplog):
    # Carrying them over would report a stale window's totals against the next
    # window's steps, which is how a frozen buffer would look live again.
    log = av_logger_class(lane="upstream", interval=1)
    with caplog.at_level(logging.WARNING):
        log.note_confidence_steps(moved=3, steps=3)
        log.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=4, verify_tokens=5)
        log.record(num_reqs=1, scheduled_drafts=7, admitted_drafts=4, verify_tokens=5)
    first, second = (r.getMessage() for r in caplog.records)
    assert "conf_moved=3/3" in first
    assert "conf_moved" not in second


def test_av_logger_counts_the_steps_that_carried_prefill(av_logger_class, caplog):
    """Make the graph field explain itself.

    Under FULL_DECODE_ONLY a mixed batch dispatches to NONE by definition, so a
    window with NONE in it needs a reason, and the reason is available for free:
    a pure decode step verifies one bonus token per request plus the admitted
    drafts, so anything beyond that is prefill sharing the batch. Without this
    the count of NONE steps can only be attributed by doing the subtraction by
    hand against verify_tokens.
    """
    # The first step always emits on its own, so the window under test is the
    # three that follow it.
    log = av_logger_class(lane="upstream", interval=3)
    with caplog.at_level(logging.WARNING):
        # Pure decode: one bonus token per request on top of the drafts.
        log.record(num_reqs=2, scheduled_drafts=14, admitted_drafts=8, verify_tokens=10)
        log.record(num_reqs=2, scheduled_drafts=14, admitted_drafts=8, verify_tokens=10)
        # Prefill riding along: more tokens than the verify width accounts for.
        log.record(num_reqs=2, scheduled_drafts=14, admitted_drafts=8, verify_tokens=74)
        log.record(num_reqs=2, scheduled_drafts=14, admitted_drafts=8, verify_tokens=10)
    opening, window = (r.getMessage() for r in caplog.records)
    assert "prefill_steps" not in opening
    assert "prefill_steps=1" in window

    quiet = av_logger_class(lane="upstream", interval=1)
    with caplog.at_level(logging.WARNING):
        quiet.record(num_reqs=2, scheduled_drafts=14, admitted_drafts=8, verify_tokens=10)
    assert "prefill_steps" not in caplog.records[-1].getMessage()


def test_av_logger_sampling_predicts_the_emitting_step(av_logger_class):
    # Lane B only copies its device capacities when this says the next recorded
    # step will print, so a wrong answer either loses the field or adds a sync.
    log = av_logger_class(lane="upstream", interval=3)
    seen = []
    for _ in range(7):
        seen.append(log.sampling())
        log.record(num_reqs=1, scheduled_drafts=4, admitted_drafts=4, verify_tokens=5)
    # Step 1 always prints; after that every third step closes a window.
    assert seen == [True, False, False, True, False, False, True]


def test_av_logger_reports_untrimmed_steps_without_dividing_by_zero(av_logger_class, caplog):
    log = av_logger_class(lane="upstream", interval=1)
    with caplog.at_level(logging.WARNING):
        log.record(num_reqs=1, scheduled_drafts=0, admitted_drafts=0, verify_tokens=1)
    message = caplog.records[-1].getMessage()
    assert "kept=100.0%" in message
    assert "trimmed_steps=0/1" in message
    assert "last_caps=n/a" in message


def test_manager_distrusts_a_bad_confidence_row_instead_of_raising(manager_class):
    """A non-finite row keeps its drafts; it must not take the engine down.

    The confidence head emits non-finite rows during prefill bursts, which
    prefix caching and async scheduling make routine. Raising there killed the
    worker mid-serve. Declining to trim one request for one step is always
    available, so distrust the row and fall back to the same path a new slot
    takes -- and never let the value itself reach the survival product.
    """
    manager = make_manager(manager_class)
    manager.record_confidences(
        torch.tensor([[0.9, 0.8, 0.4, 0.2], [float("nan"), 1, 1, 1]]),
        SimpleNamespace(num_reqs=2, idx_mapping_np=np.array([2, 0])),
    )
    # Slot 2 ("a") keeps its real confidence: threshold 0.4 admits the 0.72
    # prefix and stops. Slot 0 ("b") is distrusted, so all four drafts survive.
    assert manager.get_num_tokens({"a": 5, "b": 5}, {"a": [-1] * 4, "b": [-1] * 4}) == 2 + 1 + 4 + 1
    assert np.isfinite(manager._confidence).all()


def test_manager_separates_out_of_range_from_the_prefill_artifact(manager_class):
    # Finite but not a probability is not the prefill artifact; it would be a new
    # defect, so it is counted apart while still being distrusted rather than fatal.
    manager = make_manager(manager_class)
    manager.record_confidences(
        torch.tensor([[0.9, 0.8, 0.4, 0.2], [1.5, 1, 1, 1]]),
        SimpleNamespace(num_reqs=2, idx_mapping_np=np.array([2, 0])),
    )
    assert manager._out_of_range_rows == 1
    assert manager._untrusted_rows == 1
    assert manager.get_num_tokens({"a": 5, "b": 5}, {"a": [-1] * 4, "b": [-1] * 4}) == 2 + 1 + 4 + 1


def test_manager_still_rejects_a_confidence_shape_mismatch(manager_class):
    # A shape mismatch is the speculator and this manager disagreeing about the
    # draft geometry, not a property of the data, so it stays fatal.
    manager = make_manager(manager_class)
    with pytest.raises(ValueError):
        manager.record_confidences(
            torch.tensor([[0.9, 0.8]]),
            SimpleNamespace(num_reqs=1, idx_mapping_np=np.array([2])),
        )


def test_av_logger_names_untrusted_rows_only_when_there_are_any(av_logger_class, caplog):
    log = av_logger_class(lane="threshold", interval=1)
    with caplog.at_level(logging.WARNING):
        log.record(num_reqs=1, scheduled_drafts=4, admitted_drafts=4, verify_tokens=5)
        log.record(num_reqs=1, scheduled_drafts=4, admitted_drafts=4, verify_tokens=5, untrusted_rows=2)
        log.record(
            num_reqs=1,
            scheduled_drafts=4,
            admitted_drafts=4,
            verify_tokens=5,
            untrusted_rows=3,
            out_of_range_rows=1,
        )
    first, second, third = (r.getMessage() for r in caplog.records)
    assert "untrusted_rows" not in first
    assert "untrusted_rows=2" in second and "out_of_range" not in second
    assert "untrusted_rows=3 (out_of_range=1)" in third


def config_gate(policy, value, upstream=False, graph_mode="none", gdn_fixed_axis=False, adapt=True):
    # Load the real guard without importing the Ascend platform bootstrap.
    import ast

    source = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/eager_config.py"
    module = ast.parse(source.read_text())
    # Keep the module-level constants alongside the functions: the guard reads
    # the graph-mode names, so dropping assignments leaves it with a NameError.
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef | ast.Assign)]
    envs = dict(STUB_ENVS)
    envs["VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD"] = value
    envs["VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV"] = upstream
    envs["VLLM_ASCEND_DSPARK_AV_GRAPH"] = graph_mode
    envs["VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS"] = gdn_fixed_axis
    envs["VLLM_ASCEND_DSPARK_AV_ADAPT"] = adapt
    namespace = {
        "envs_ascend": SimpleNamespace(**envs),
        "validate_threshold": policy.validate_threshold,
        # The module keeps the `logger = init_logger(__name__)` assignment, so
        # the stub namespace has to satisfy it without importing vllm.
        "init_logger": logging.getLogger,
        "__name__": "eager_config_stub",
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def test_graph_mode_none_still_demands_an_eager_target(policy):
    config = eager_config()
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError):
        config_gate(policy, 0.4, graph_mode="none")["eager_survival_threshold"](config)


def test_a_graph_mode_drops_the_eager_target_requirement(policy):
    # Capturing a graph is the whole point of the uniform mode, so the lane must
    # stop insisting on --enforce-eager once one is selected. Every other
    # precondition still applies.
    config = eager_config()
    config.model_config.enforce_eager = False
    gate = config_gate(policy, 0.4, graph_mode="uniform")
    assert gate["eager_survival_threshold"](config) == 0.4
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError):
        gate["eager_survival_threshold"](config)


def test_ragged_graph_mode_pins_the_gdn_axis_without_a_second_switch(policy):
    """The axis is part of the mode, not an orthogonal knob.

    Replaying a trimmed batch against a graph captured over a different
    per-request split is only safe if the request axis the state operators see
    is the same for every bucket. A ragged run with a per-bucket axis is the
    combination that is known to fail, so the mode must not be reachable
    without the pin -- while the pin stays selectable on its own so it can be
    bisected apart from the capture geometry.
    """
    gate = config_gate(policy, 0.4, graph_mode="ragged")
    assert gate["av_graph_mode"]() == "ragged"
    assert gate["av_graph_pins_gdn_axis"]() is True


@pytest.mark.parametrize("mode", ["none", "uniform"])
def test_the_other_graph_modes_leave_the_gdn_axis_to_the_variable(policy, mode):
    gate = config_gate(policy, 0.4, graph_mode=mode)
    assert gate["av_graph_pins_gdn_axis"]() is False
    gate = config_gate(policy, 0.4, graph_mode=mode, gdn_fixed_axis=True)
    assert gate["av_graph_pins_gdn_axis"]() is True


@pytest.mark.parametrize("mode", ["full", "piecewise", "None", "uniform "])
def test_an_unknown_graph_mode_is_rejected(policy, mode):
    with pytest.raises(ValueError, match="must be one of"):
        config_gate(policy, 0.4, graph_mode=mode)["av_graph_mode"]()


def test_an_unset_graph_mode_is_ragged_except_under_the_threshold_lane(policy):
    """Unset has to mean the path that ships, which is the ragged graph.

    Lane A is the one exception and it is not a preference: it computes exact
    host capacities every step, and a captured graph cannot pay that copy. So
    the default is per lane rather than global -- otherwise turning the feature
    on by config would silently put the bisect tool in a graph it cannot run in.
    """
    assert config_gate(policy, None, upstream=True, graph_mode="")["av_graph_mode"]() == "ragged"
    assert config_gate(policy, None, graph_mode="")["av_graph_mode"]() == "ragged"
    assert config_gate(policy, 0.4, graph_mode="")["av_graph_mode"]() == "none"
    # Named explicitly, lane A can still be put under a graph on purpose.
    assert config_gate(policy, 0.4, graph_mode="uniform")["av_graph_mode"]() == "uniform"


def test_config_alone_engages_the_upstream_lane(policy):
    """enable_adaptive_verification=true is the whole switch.

    Before this, the adaptation was gated on an environment variable, so the
    config flag on its own gave upstream's unmodified manager with none of the
    ragged plumbing -- which is the difference between a working experiment and
    a working feature.
    """
    config = eager_config()
    config.model_config.enforce_eager = False
    config.speculative_config.enforce_eager = False
    gate = config_gate(policy, None, graph_mode="")
    assert gate["eager_upstream_av_enabled"](config) is True
    assert gate["eager_adaptive_lane_active"](config) is True


def test_the_opt_out_hands_the_run_back_to_upstream(policy):
    config = eager_config()
    config.model_config.enforce_eager = False
    gate = config_gate(policy, None, graph_mode="", adapt=False)
    assert gate["eager_upstream_av_enabled"](config) is False
    assert gate["eager_adaptive_lane_active"](config) is False


def test_an_unsupported_config_falls_back_instead_of_failing_to_start(policy, caplog):
    """Default-on must not take down a config that used to work.

    Asked for by name, an unsupported config is a mistake and raises. Reached by
    default, it has to degrade to upstream's own manager -- which is exactly
    what the run would have got before this became the default -- and say so.
    """
    config = eager_config()
    config.model_config.enforce_eager = False
    config.parallel_config.pipeline_parallel_size = 2

    gate = config_gate(policy, None, graph_mode="")
    with caplog.at_level(logging.WARNING):
        assert gate["eager_upstream_av_enabled"](config) is False
    assert "falling back to the upstream manager" in caplog.text

    named = config_gate(policy, None, upstream=True, graph_mode="")
    with pytest.raises(ValueError, match="PP=PCP=DCP=1"):
        named["eager_upstream_av_enabled"](config)


def test_av_disabled_in_config_engages_nothing(policy):
    config = eager_config()
    config.speculative_config.enable_adaptive_verification = False
    gate = config_gate(policy, None, graph_mode="")
    assert gate["av_enabled_in_config"](config) is False
    assert gate["eager_upstream_av_enabled"](config) is False


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
    assert config_gate(policy, None)["eager_survival_threshold"](object()) is None
    assert config_gate(policy, 0.4)["eager_survival_threshold"](eager_config()) == 0.4


def test_upstream_lane_opt_in_and_mutual_exclusion(policy):
    off = config_gate(policy, None)
    assert off["eager_upstream_av_enabled"](object()) is False
    assert off["eager_adaptive_lane_active"](object()) is False

    on = config_gate(policy, None, upstream=True)
    assert on["eager_upstream_av_enabled"](eager_config()) is True
    assert on["eager_adaptive_lane_active"](eager_config()) is True

    # Both lanes trim the same budget, so running them together would silently
    # let one of the two managers win. Refuse instead of picking.
    both = config_gate(policy, 0.4, upstream=True)
    with pytest.raises(ValueError):
        both["eager_survival_threshold"](eager_config())


def test_upstream_lane_shares_the_threshold_lane_preconditions(policy):
    config = eager_config()
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError):
        config_gate(policy, None, upstream=True)["eager_upstream_av_enabled"](config)


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
        config_gate(policy, 0.4)["eager_survival_threshold"](config)


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
    config_module.eager_adaptive_lane_active = lambda _: True
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


def _engine_env_keys(relative: str) -> tuple[set[str], set[str]]:
    """Environment keys a script's child_env sets, and the ones it clears."""
    import ast

    tree = ast.parse((ROOT / relative).read_text())
    # One script writes these keys as literals and the other through module
    # constants, so resolve the constants or the comparison reports a
    # difference that is only spelling.
    constants: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            constants[target.id] = node.value.value
        elif isinstance(node.value, ast.Tuple):
            names = [
                constants[e.id] if isinstance(e, ast.Name) and e.id in constants else e.value
                for e in node.value.elts
                if (isinstance(e, ast.Constant) and isinstance(e.value, str))
                or (isinstance(e, ast.Name) and e.id in constants)
            ]
            if names:
                groups[target.id] = names

    def literals_in(node) -> list[str]:
        """String keys in a loop's iterable, whether written out or named."""
        if isinstance(node, ast.Name):
            return groups.get(node.id, [])
        if isinstance(node, (ast.Tuple, ast.List)):
            out = []
            for element in node.elts:
                key = None
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    key = element.value
                elif isinstance(element, ast.Name):
                    key = constants.get(element.id)
                if key is not None:
                    out.append(key)
            return out
        return []

    def key_of(node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        return None

    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "child_env")
    keys: set[str] = set()
    cleared: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and getattr(target.value, "id", None) == "env":
                    key = key_of(target.slice)
                    if key is not None:
                        keys.add(key)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and getattr(node.func.value, "id", None) == "env"
            and node.args
        ):
            key = key_of(node.args[0])
            if key is not None:
                keys.add(key)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and getattr(node.func.value, "id", None) == "env"
            and node.args
        ):
            # A key a script deliberately clears is a lane knob, not engine
            # setup -- the benchmark clears exactly the ones the gate sets per
            # lane, so this is what separates the two kinds.
            key = key_of(node.args[0])
            if key is not None:
                cleared.add(key)
        elif isinstance(node, ast.For) and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "pop"
            and getattr(inner.func.value, "id", None) == "env"
            for inner in ast.walk(node)
        ):
            # `for key in (...): env.pop(key, None)` -- the keys are on the
            # iterable, not on the call.
            cleared.update(literals_in(node.iter))
    return keys, cleared


def test_both_scripts_build_the_same_engine():
    """The gate and the benchmark must not drift apart in engine setup.

    They construct engines independently, so an alignment made in one silently
    does not apply to the other. That happened with the compile cache: the
    benchmark disabled it after a stale artifact crashed a run, the gate did
    not, and the same crash arrived there days later. Comparing the literal
    keys means the next addition has to be made in both or fail here.
    """
    gate_set, _ = _engine_env_keys("examples/dspark_eager_adaptive_verify.py")
    benchmark_set, lane_knobs = _engine_env_keys("examples/dspark_adaptive_verify_throughput.py")
    gate = gate_set - lane_knobs
    benchmark = benchmark_set - lane_knobs
    assert gate, "no engine env found in the gate script"
    assert lane_knobs, "no lane knobs found, so the two kinds cannot be told apart"
    assert gate == benchmark, (
        f"only in the gate: {sorted(gate - benchmark)}; only in the benchmark: {sorted(benchmark - gate)}"
    )
    # The one that caused it, named so a future removal is deliberate.
    assert "VLLM_DISABLE_COMPILE_CACHE" in gate


def test_every_per_request_buffer_has_the_padding_row():
    """The dummy row must exist in all of them, not just the one that crashed.

    query_start_loc was widened for token padding long ago and the rest were
    not, which is what let a saturated batch describe one more request than it
    had lengths for. Read the widths out of the source so widening one and
    forgetting another fails here rather than on a device.
    """
    import ast

    path = ROOT / "vllm_ascend/worker/v2/input_batch.py"
    tree = ast.parse(path.read_text())

    widths: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) and not isinstance(node, ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        name = next(
            (t.attr for t in targets if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)),
            None,
        )
        if name is None or node.value is None or not isinstance(node.value, ast.Call):
            continue
        for arg in node.value.args + [kw.value for kw in node.value.keywords]:
            if isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Name) and arg.left.id == "max_num_reqs":
                widths[name] = f"max_num_reqs + {ast.literal_eval(arg.right)}"
            elif isinstance(arg, ast.Name) and arg.id == "max_num_reqs":
                widths.setdefault(name, "max_num_reqs")

    assert widths.get("query_start_loc") == "max_num_reqs + 2", widths
    for name in ("seq_lens", "seq_lens_cpu", "dcp_local_seq_lens"):
        assert widths.get(name) == "max_num_reqs + 1", (name, widths)


def _impure_refusal():
    """Extract the refusal helper; it must not depend on anything from vllm."""
    import ast

    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_refuse_graph_for_impure_batch")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "dataclasses": dataclasses,
        "CUDAGraphMode": SimpleNamespace(NONE="none", FULL="full", PIECEWISE="piecewise"),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_refuse_graph_for_impure_batch"]


@dataclasses.dataclass(frozen=True)
class _Desc:
    """Mirrors v0.28.0's BatchExecutionDescriptor: frozen, so replace() is the way."""

    cg_mode: str
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None


def test_a_batch_with_a_prefill_does_not_get_the_ragged_graph():
    """Completes the guard v0.28.0 already has.

    That version documents max_query_len as what keeps a prefill batch out of a
    varlen decode graph, and it works while the prefill is longer than the
    decode width. Both device failures were a six or seven token prefill beside
    eight-token decodes, so the batch's longest query was still eight and it
    matched a captured decode graph whose geometry it did not share.
    """
    refuse = _impure_refusal()
    # A graph the manager matched, padded up to a captured size.
    matched = (_Desc(cg_mode="full", num_tokens=128, num_reqs=16, max_query_len=8), "dp")

    pure = SimpleNamespace(
        av_refuse_impure_batches=True,
        model_runner=SimpleNamespace(av_batch_is_pure_spec_decode=True),
    )
    assert refuse(pure, 16, 127, matched) is matched, "a pure batch keeps its graph"

    mixed = SimpleNamespace(
        av_refuse_impure_batches=True,
        model_runner=SimpleNamespace(av_batch_is_pure_spec_decode=False),
    )
    desc, rest = refuse(mixed, 16, 127, matched)
    assert desc.cg_mode == "none"
    # The padding existed to reach a captured size; there is no capture to reach.
    assert desc.num_tokens == 127 and desc.num_reqs == 16
    # Everything the manager set and this does not own survives untouched.
    assert desc.max_query_len == 8 and rest == "dp"

    # Piecewise is the native safe path, not something to refuse: v0.28.0
    # documents it as carrying no request padding and no replay-time request
    # limit, and the adaptive FIA padding only runs for FULL. Sending these to
    # eager instead is what cost about nine percent of throughput.
    piecewise = (_Desc(cg_mode="piecewise", num_tokens=128, num_reqs=None), "dp")
    assert refuse(mixed, 16, 127, piecewise) is piecewise

    # Off for any manager this repo did not mark, and for one already eager.
    unmarked = SimpleNamespace(model_runner=SimpleNamespace(av_batch_is_pure_spec_decode=False))
    assert refuse(unmarked, 16, 127, matched) is matched
    already = (_Desc(cg_mode="none", num_tokens=127, num_reqs=16), "dp")
    assert refuse(mixed, 16, 127, already) is already


def _fia_padding():
    """Extract the adaptive FIA padding method and bind it to a stub runner."""
    import ast

    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pad_adaptive_query_start_loc_for_fia"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_pad_adaptive_query_start_loc_for_fia"]


def test_fia_padding_row_exists_when_every_slot_is_live():
    """The dummy row that carries token padding is the (max_num_reqs + 1)th.

    With every request slot live there is no padding *request* to spread the
    padding tokens across, so they need a row of their own -- B_fia = B_live + 1
    with B_live at its maximum. The per-request buffers are allocated one wider
    for exactly this row; before that they were not, and a saturated trimmed
    batch went on with its query boundaries describing one more request than its
    lengths did.
    """
    pad = _fia_padding()
    runner = SimpleNamespace(max_num_reqs=16)
    boundaries = np.zeros(18, dtype=np.int32)
    boundaries[:17] = np.arange(0, 17) * 7  # 16 live requests, 112 tokens
    out, padded = pad(runner, 120, 16, 16, boundaries)
    assert padded == 17, "the padding tokens need their own row"
    assert out[16] == 112, "the live boundary is untouched"
    assert out[17] == 120, "and the dummy row absorbs the difference"

    # Exactly on a capture size: no padding tokens, so no extra row.
    out, padded = pad(runner, 112, 16, 16, boundaries.copy())
    assert padded == 16


def test_fia_padding_spreads_across_the_rows_the_descriptor_left():
    pad = _fia_padding()
    runner = SimpleNamespace(max_num_reqs=16)
    boundaries = np.zeros(18, dtype=np.int32)
    boundaries[:5] = np.arange(0, 5) * 10  # 4 live requests, 40 tokens
    out, padded = pad(runner, 48, 8, 4, boundaries)
    assert padded == 8, "pads out to the descriptor's request count"
    # Eight tokens spread over four padding rows, two each.
    assert list(out[5:9]) == [42, 44, 46, 48], list(out[5:9])


def test_fia_padding_refuses_a_row_the_buffers_do_not_have():
    # Not reachable from a descriptor -- B_fia cannot exceed B_live + 1 -- but
    # if it ever is, failing here names the cause instead of surfacing as a
    # tensor size mismatch several frames away.
    pad = _fia_padding()
    runner = SimpleNamespace(max_num_reqs=4)
    boundaries = np.zeros(8, dtype=np.int32)
    boundaries[:6] = np.arange(0, 6) * 7
    with pytest.raises(RuntimeError, match="past the max_num_reqs"):
        pad(runner, 60, 5, 5, boundaries)


def _graph_factory(monkeypatch, mode, keep_piecewise=False):
    """Extract graph_manager_wrapper and run it against a stubbed graph mode."""
    import ast
    from contextlib import contextmanager

    config_module = types.ModuleType(f"{PKG}.eager_config")
    config_module.GRAPH_MODE_NONE = "none"
    config_module.GRAPH_MODE_UNIFORM = "uniform"
    config_module.GRAPH_MODE_RAGGED = "ragged"
    config_module.av_graph_mode = lambda: mode
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)

    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text())
    wanted = ("graph_manager_wrapper",)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(functions) == len(wanted), f"expected {wanted}, found {[f.name for f in functions]}"
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)

    class _StubManager:
        """Mirrors ModelAclGraphManager's real signature.

        Positional up to model_runner, keyword after -- so a construction that
        omits model_runner or passes lora_capture_cases into its place fails
        here rather than as an AttributeError during engine start, which is what
        happened on 2026-09-23 and cost a device run.

        Unpacks as (args, kwargs) so the tests that only care about what the
        factory passed read the same as before.
        """

        def __init__(
            self,
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            model_runner,
            lora_capture_cases=None,
            varlen_decode=False,
        ):
            assert not isinstance(model_runner, list), (
                "model_runner is the fifth positional argument; passing "
                "lora_capture_cases there is the mistake this guards"
            )
            self.model_runner = model_runner
            self.args = (vllm_config, device, cudagraph_mode, decode_query_len, model_runner)
            self.kwargs = {"lora_capture_cases": lora_capture_cases, "varlen_decode": varlen_decode}

        def __iter__(self):
            return iter((self.args, self.kwargs))

        def _resolve_effective_loras(self, num_active_loras):
            return num_active_loras

        def dispatch(self, *args, **kwargs):
            return ("delegated", args, kwargs)

    upstream = SimpleNamespace(ModelCudaGraphManager=object())
    namespace = {
        "contextmanager": contextmanager,
        "vllm_model_runner": upstream,
        "BatchExecutionDescriptor": lambda **fields: SimpleNamespace(**fields),
        "CUDAGraphMode": SimpleNamespace(
            NONE="none", FULL_AND_PIECEWISE="full_and_piecewise", FULL_DECODE_ONLY="full_decode_only"
        ),
        "logger": logging.getLogger("model_runner_stub"),
        # The wrapper reads the keep-piecewise bisect flag; default it off so
        # these tests exercise the shipping path.
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_DSPARK_AV_KEEP_PIECEWISE=keep_piecewise),
        "ModelAclGraphManager": _StubManager,
    }
    exec(compile(code, str(path), "exec"), namespace)
    return namespace["graph_manager_wrapper"], upstream


def test_target_graph_factory_keeps_the_lane_eager_when_no_graph_mode_is_set(monkeypatch):
    wrapper, upstream = _graph_factory(monkeypatch, "none")
    original = upstream.ModelCudaGraphManager
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode="full_and_piecewise"))
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        args, _ = upstream.ModelCudaGraphManager(config, "cpu", "full_and_piecewise", 8, runner, varlen_decode=True)
        assert args[2] == "none"
        assert config.compilation_config.cudagraph_mode == "none"
    assert upstream.ModelCudaGraphManager is original


def test_uniform_graph_mode_keeps_the_graph_and_drops_the_varlen_descriptor(monkeypatch):
    """The captured geometry has to be the one a real batch replays.

    Adaptive verification asks for the variable-length decode descriptor, which
    spreads the dummy tokens evenly and so captures one token per request for
    every bucket at or below max_num_seqs -- while a speculative batch replays
    one request per verify width. Full attention re-issues its kernel with
    refreshed host lengths each replay and survives; the GDN layers get no such
    update, so whatever geometry was captured into the recurrent and conv tasks
    is the only one they ever run.
    """
    wrapper, upstream = _graph_factory(monkeypatch, "uniform")
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode="full_decode_only"))
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        args, kwargs = upstream.ModelCudaGraphManager(config, "cpu", "full_decode_only", 8, runner, varlen_decode=True)
        # The graph mode survives ...
        assert args[2] == "full_decode_only"
        assert config.compilation_config.cudagraph_mode == "full_decode_only"
        # ... and the descriptor is the uniform one.
        assert kwargs["varlen_decode"] is False


def test_ragged_drops_a_piecewise_family_that_cannot_be_piecewise(monkeypatch, caplog):
    """Upstream forces FULL_AND_PIECEWISE so trimmed batches have a fallback.

    Ragged mode removes that need -- a trimmed batch replays the decode graph --
    so the piecewise half becomes a second family to capture. Downgrade it only
    when it could not have been piecewise anyway: splitting_ops is decided at
    config time from the configured mode, before upstream's override runs, so a
    run that asked for FULL_DECODE_ONLY has none and its "piecewise" captures
    are unsplit whole-model graphs. Where splitting really was configured, the
    mode must be left alone.
    """
    wrapper, upstream = _graph_factory(monkeypatch, "ragged")
    unsplit = SimpleNamespace(cudagraph_mode="full_and_piecewise", splitting_ops_contain_attention=lambda: False)
    config = SimpleNamespace(compilation_config=unsplit)
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)), caplog.at_level(logging.WARNING):
        args, kwargs = upstream.ModelCudaGraphManager(
            config, "cpu", "full_and_piecewise", 8, runner, varlen_decode=True
        )
    assert args[2] == "full_decode_only"
    assert unsplit.cudagraph_mode == "full_decode_only"
    # The varlen descriptor has to survive the downgrade, or ragged loses the
    # only thing it needs: FULL_DECODE_ONLY is still a separate-routine mode.
    assert kwargs["varlen_decode"] is True
    assert "FULL_DECODE_ONLY" in caplog.text

    split = SimpleNamespace(cudagraph_mode="full_and_piecewise", splitting_ops_contain_attention=lambda: True)
    config = SimpleNamespace(compilation_config=split)
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        args, kwargs = upstream.ModelCudaGraphManager(
            config, "cpu", "full_and_piecewise", 8, runner, varlen_decode=True
        )
    assert args[2] == "full_and_piecewise"
    assert split.cudagraph_mode == "full_and_piecewise"


def test_keep_piecewise_flag_leaves_the_forced_mode_alone(monkeypatch):
    """The escape hatch has to reach the decision it guards.

    It exists to bisect a startup failure against the downgrade, which is
    exactly the moment when a flag that silently does nothing costs the most.
    So assert both states, not just that the default still downgrades.
    """
    unsplit = lambda: SimpleNamespace(  # noqa: E731 - a fresh config per call
        cudagraph_mode="full_and_piecewise", splitting_ops_contain_attention=lambda: False
    )

    wrapper, upstream = _graph_factory(monkeypatch, "ragged", keep_piecewise=True)
    config = SimpleNamespace(compilation_config=unsplit())
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        args, kwargs = upstream.ModelCudaGraphManager(
            config, "cpu", "full_and_piecewise", 8, runner, varlen_decode=True
        )
    assert args[2] == "full_and_piecewise", "the flag must leave the forced mode alone"
    assert config.compilation_config.cudagraph_mode == "full_and_piecewise"
    # Turning the downgrade off must not also turn off what ragged needs.
    assert kwargs["varlen_decode"] is True

    wrapper, upstream = _graph_factory(monkeypatch, "ragged", keep_piecewise=False)
    config = SimpleNamespace(compilation_config=unsplit())
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        args, _ = upstream.ModelCudaGraphManager(config, "cpu", "full_and_piecewise", 8, runner, varlen_decode=True)
    assert args[2] == "full_decode_only", "the default still downgrades"


def test_ragged_marks_the_manager_for_the_refusal(monkeypatch):
    """The factory only flags the manager; the refusal itself lives elsewhere.

    It used to wrap the manager's dispatch, which meant restating a vllm
    signature that differs between the tree here and the pinned deployment
    version -- a startup crash. The flag is an attribute this repo owns and the
    decision happens where the dispatch call is already intercepted.
    """
    wrapper, upstream = _graph_factory(monkeypatch, "ragged")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode="full_and_piecewise", splitting_ops_contain_attention=lambda: True
        )
    )
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        manager = upstream.ModelCudaGraphManager(config, "cpu", "full_and_piecewise", 8, runner, varlen_decode=True)
    assert manager.av_refuse_impure_batches is True

    # Not marked under the other modes.
    wrapper, upstream = _graph_factory(monkeypatch, "uniform")
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        manager = upstream.ModelCudaGraphManager(config, "cpu", "full_decode_only", 8, runner, varlen_decode=True)
    assert not hasattr(manager, "av_refuse_impure_batches")


def test_ragged_graph_mode_keeps_the_varlen_descriptor(monkeypatch):
    """The whole of item 1 is *not* dropping the variable-length descriptor.

    The manager then captures one decode graph per size at
    num_reqs=min(Q, max_num_seqs) and max_query_len=decode_query_len, and its
    compatibility rule admits any batch with no more requests, no more tokens
    and no longer a query -- which is exactly a trimmed batch. What made that
    unsafe is the geometry the state operators see, and that is the axis pin,
    not the descriptor.
    """
    wrapper, upstream = _graph_factory(monkeypatch, "ragged")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode="full_and_piecewise", splitting_ops_contain_attention=lambda: True
        )
    )
    with wrapper(runner := SimpleNamespace(eager_survival_test=True)):
        args, kwargs = upstream.ModelCudaGraphManager(
            config, "cpu", "full_and_piecewise", 8, runner, varlen_decode=True
        )
        assert args[2] == "full_and_piecewise"
        assert config.compilation_config.cudagraph_mode == "full_and_piecewise"
        assert kwargs["varlen_decode"] is True


def test_a_graph_mode_leaves_a_non_lane_runner_alone(monkeypatch):
    # The wrapper must not touch a run that is not using an adaptive lane.
    wrapper, upstream = _graph_factory(monkeypatch, "uniform")
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode="full_decode_only"))
    with wrapper(runner := SimpleNamespace(eager_survival_test=False)):
        args, kwargs = upstream.ModelCudaGraphManager(config, "cpu", "full_decode_only", 8, runner, varlen_decode=True)
        assert args[2] == "full_decode_only"
        assert kwargs["varlen_decode"] is True
