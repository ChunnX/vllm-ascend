# SPDX-License-Identifier: Apache-2.0
from functools import wraps

import vllm.v1.worker.gpu.model_runner as model_runner
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive
from vllm.config import get_current_vllm_config

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.eager_config import (
    eager_survival_threshold,
    eager_upstream_av_enabled,
)

adaptive._assign_draft_token_budget_compiled = adaptive._assign_draft_token_budget
_original_factory = getattr(
    adaptive.maybe_create_adaptive_verification_manager,
    "__wrapped__",
    adaptive.maybe_create_adaptive_verification_manager,
)


@wraps(_original_factory)
def _maybe_create_adaptive_verification_manager(**kwargs):
    # Only touch get_current_vllm_config when an eager lane env is set, so the
    # non-eager path is exactly upstream.
    if (
        envs_ascend.VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD is None
        and not envs_ascend.VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV
    ):
        return _original_factory(**kwargs)
    config = get_current_vllm_config()

    # Lane B: the real upstream manager, run eager with a synthetic cost curve.
    # It exercises the device survival top-k and async D2H the graph phase reuses.
    if eager_upstream_av_enabled(config):
        if not kwargs["enable_adaptive_verification"]:
            raise ValueError("Eager upstream AV requires a loaded DSpark confidence head")
        from vllm_ascend.worker.v2.spec_decode.dspark.eager_upstream_av import AscendEagerUpstreamAVManager

        return AscendEagerUpstreamAVManager(
            kwargs["req_states"],
            kwargs["query_start_loc"],
            kwargs["num_bonus_tokens"],
            kwargs["max_total_logits"],
        )

    # Lane A: synchronous survival-threshold policy.
    threshold = eager_survival_threshold(config)
    if threshold is None:
        return _original_factory(**kwargs)
    if not kwargs["enable_adaptive_verification"]:
        raise ValueError("Eager survival verification requires a loaded DSpark confidence head")
    from vllm_ascend.worker.v2.spec_decode.dspark.eager_verification import EagerSurvivalVerificationManager

    # Exact CPU boundaries remove the CPU/device mismatch; no graph is built,
    # so ALWAYS is neither required nor advertised by the GDN backend.
    return EagerSurvivalVerificationManager(
        kwargs["req_states"],
        kwargs["query_start_loc"],
        kwargs["num_bonus_tokens"],
        kwargs["max_total_logits"],
        threshold=threshold,
    )


adaptive.maybe_create_adaptive_verification_manager = _maybe_create_adaptive_verification_manager
model_runner.maybe_create_adaptive_verification_manager = _maybe_create_adaptive_verification_manager
