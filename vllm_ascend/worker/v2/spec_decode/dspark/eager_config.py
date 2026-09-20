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


def _validate_eager_lane(config) -> None:
    """Preconditions shared by both eager adaptive-verification lanes."""
    spec = config.speculative_config
    if spec is None or spec.method != "dspark" or not spec.enable_adaptive_verification:
        raise ValueError("Eager adaptive verification requires DSpark and enable_adaptive_verification=true")
    if not config.model_config.enforce_eager or spec.enforce_eager is False:
        raise ValueError("Eager adaptive verification requires --enforce-eager and an eager drafter")
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
