# SPDX-License-Identifier: Apache-2.0
"""Decide which adaptive-verification manager runs, and under which graph mode.

The shipping path is lane B: the real upstream ``AdaptiveVerificationManager``
(cost-argmax budget, device survival top-k) with this repo's ragged plumbing,
under the ragged full graph. It is driven by the config alone, so
``enable_adaptive_verification: true`` is the whole switch.

Lane A (``VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD``) stays as a bisect tool:
a synchronous survival-threshold policy that computes exact CPU capacities
itself. Exact host boundaries are the opposite of what a captured graph can do,
so lane A defaults to staying eager. The two lanes are mutually exclusive.

Naming: everything here is still called ``eager_*`` from when both lanes were
eager rehearsals. The default path is now a graph path, so the names are stale
-- left alone deliberately, to keep this change about behaviour.

``eager_adaptive_lane_active`` is the single predicate the runner, GDN builder,
mamba state and speculator gate their adaptations on.
"""

from vllm.logger import init_logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.eager_policy import validate_threshold

logger = init_logger(__name__)

GRAPH_MODE_NONE = "none"
GRAPH_MODE_UNIFORM = "uniform"
GRAPH_MODE_RAGGED = "ragged"
_GRAPH_MODES = (GRAPH_MODE_NONE, GRAPH_MODE_UNIFORM, GRAPH_MODE_RAGGED)


def av_graph_mode() -> str:
    """Validated value of VLLM_ASCEND_DSPARK_AV_GRAPH.

    ``none`` stays eager, ``uniform`` captures at the full verify width so an
    untrimmed batch replays and a trimmed one falls back to piecewise, and
    ``ragged`` keeps the variable-length descriptor so a trimmed batch replays
    too. The last one only holds together with the request axes pinned, which is
    why selecting it pins the GDN axis rather than leaving that to a second
    switch.

    Unset means ``ragged``: that is the path that ships, and it is validated on
    device. The exception is lane A, whose exact host boundaries cannot survive
    a captured graph, so it stays eager unless a mode is named explicitly.
    """
    mode = envs_ascend.VLLM_ASCEND_DSPARK_AV_GRAPH
    if not mode:
        if envs_ascend.VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD is not None:
            return GRAPH_MODE_NONE
        return GRAPH_MODE_RAGGED
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


def av_enabled_in_config(config) -> bool:
    """Whether the config asks for DSpark adaptive verification at all."""
    spec = getattr(config, "speculative_config", None)
    return bool(spec is not None and spec.method == "dspark" and spec.enable_adaptive_verification)


def _validate_eager_lane(config) -> None:
    """Preconditions shared by both adaptive-verification lanes."""
    spec = config.speculative_config
    if not av_enabled_in_config(config):
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
        raise ValueError("Ascend adaptive verification currently requires PP=PCP=DCP=1")
    if config.lora_config is not None or getattr(parallel, "enable_dbo", False):
        raise ValueError("Ascend adaptive verification does not support LoRA or DBO")
    if not 1 <= spec.num_speculative_tokens <= 15:
        raise ValueError("Ascend GDN verification requires 1 <= num_speculative_tokens <= 15")


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
    """Lane B: the upstream manager with this repo's plumbing. The default.

    Config alone turns it on, so ``enable_adaptive_verification: true`` needs no
    environment variable. ``VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV`` stays as an
    explicit request, and changes exactly one thing -- see below.
    """
    if envs_ascend.VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD is not None:
        return False  # lane A owns the run; the two are mutually exclusive.
    if not envs_ascend.VLLM_ASCEND_DSPARK_AV_ADAPT:
        # Explicit opt-out: run upstream's manager exactly as upstream would.
        return False
    if not av_enabled_in_config(config):
        return False
    if envs_ascend.VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV:
        # Asked for by name: an unsupported config is a mistake worth refusing.
        _validate_eager_lane(config)
        return True
    # Reached by default. A config this adaptation does not cover must not take
    # the engine down -- upstream's own manager handles it, minus the ragged
    # plumbing, which is what would have happened before this became default.
    try:
        _validate_eager_lane(config)
    except ValueError as exc:
        logger.warning(
            "[DSPARK-AV] not adapting this run, falling back to the upstream manager: %s. "
            "Set VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV=1 to make this a hard error instead.",
            exc,
        )
        return False
    return True


def eager_adaptive_lane_active(config) -> bool:
    """True when either AV lane (A threshold or B upstream) is engaged.

    This is the flag every ragged-decode adaptation keys on, so lane B reuses
    exactly the plumbing lane A already exercises.
    """
    return eager_survival_threshold(config) is not None or eager_upstream_av_enabled(config)
