# SPDX-License-Identifier: Apache-2.0
"""Guard the diagnostic eager lanes without changing upstream AV defaults.

Two opt-in eager lanes share the same preconditions but differ in policy:

- Lane A (``VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD``): a synchronous
  survival-threshold policy that computes exact CPU capacities itself.
- Lane B (``VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV``): the real upstream
  ``AdaptiveVerificationManager`` (cost-argmax budget + device survival top-k +
  async D2H) with an injected synthetic cost curve.

They are mutually exclusive. ``eager_adaptive_lane_active`` is the single
predicate the runner, GDN builder, mamba state and speculator gate their eager
adaptations on, so both lanes share the same ragged-decode plumbing.
"""

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.eager_policy import validate_threshold

GRAPH_MODE_NONE = "none"
GRAPH_MODE_UNIFORM = "uniform"
GRAPH_MODE_RAGGED = "ragged"
_GRAPH_MODES = (GRAPH_MODE_NONE, GRAPH_MODE_UNIFORM, GRAPH_MODE_RAGGED)


def av_graph_mode() -> str:
    """Validated value of VLLM_ASCEND_DSPARK_AV_GRAPH.

    ``none`` keeps the lane eager, ``uniform`` captures at the full verify width
    so an untrimmed batch replays and a trimmed one falls back, and ``ragged``
    keeps the variable-length descriptor so a trimmed batch replays too. The
    last one only holds together with the request axes pinned, which is why
    selecting it pins the GDN axis rather than leaving that to a second switch.
    """
    mode = envs_ascend.VLLM_ASCEND_DSPARK_AV_GRAPH
    if mode not in _GRAPH_MODES:
        raise ValueError(f"VLLM_ASCEND_DSPARK_AV_GRAPH must be one of {_GRAPH_MODES}, got {mode!r}")
    return mode


def av_graph_pins_gdn_axis() -> bool:
    """Whether the GDN request axis must be pinned to max_num_seqs.

    The ragged mode replays a trimmed batch against a graph captured over a
    different per-request split, which only holds if the request axis the state
    operators see is the same for every bucket. Pinning it is therefore part of
    that mode rather than an independent switch, and the environment variable
    stays only so the pinning can be bisected on its own under the other modes.
    """
    if av_graph_mode() == GRAPH_MODE_RAGGED:
        return True
    return envs_ascend.VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS


def _validate_eager_lane(config) -> None:
    """Preconditions shared by both adaptive-verification lanes."""
    spec = config.speculative_config
    if spec is None or spec.method != "dspark" or not spec.enable_adaptive_verification:
        raise ValueError("Adaptive verification lane requires DSpark and enable_adaptive_verification=true")
    if av_graph_mode() == GRAPH_MODE_NONE:
        # The eager gate validated this lane with both target and draft eager.
        if not config.model_config.enforce_eager or spec.enforce_eager is False:
            raise ValueError(
                "The adaptive verification lane requires --enforce-eager and an eager drafter "
                "unless VLLM_ASCEND_DSPARK_AV_GRAPH selects a graph mode"
            )
    parallel = config.parallel_config
    if any(
        getattr(parallel, key, 1) != 1
        for key in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("Eager adaptive verification currently requires PP=PCP=DCP=1")
    if config.lora_config is not None or getattr(parallel, "enable_dbo", False):
        raise ValueError("Eager adaptive verification does not support LoRA or DBO")
    if not 1 <= spec.num_speculative_tokens <= 15:
        raise ValueError("Eager GDN verification requires 1 <= num_speculative_tokens <= 15")


def eager_survival_threshold(config):
    """Lane A: return the validated survival threshold, or None when off."""
    value = envs_ascend.VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD
    if value is None:
        return None
    threshold = validate_threshold(value)
    if envs_ascend.VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV:
        raise ValueError(
            "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD and "
            "VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV are mutually exclusive eager lanes"
        )
    _validate_eager_lane(config)
    return threshold


def eager_upstream_av_enabled(config) -> bool:
    """Lane B: True when the upstream-manager eager lane is engaged."""
    if not envs_ascend.VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV:
        return False
    _validate_eager_lane(config)
    return True


def eager_adaptive_lane_active(config) -> bool:
    """True when either eager AV lane (A threshold or B upstream) is engaged.

    This is the flag every eager ragged-decode adaptation keys on, so lane B
    reuses exactly the plumbing lane A already exercises.
    """
    return eager_survival_threshold(config) is not None or eager_upstream_av_enabled(config)
