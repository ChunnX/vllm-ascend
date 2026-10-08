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
    # No confidence head means no lane can run, and it is also the common path,
    # so keep it free of any config lookup. A lane asked for by name is still
    # worth a clear error rather than a silent downgrade.
    if not kwargs["enable_adaptive_verification"]:
        if (
            envs_ascend.VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD is not None
            or envs_ascend.VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV
        ):
            raise ValueError("The DSpark adaptive-verification lanes require a loaded confidence head")
        return _original_factory(**kwargs)
    config = get_current_vllm_config()

    # Lane B: the upstream manager with this repo's ragged plumbing. Reached
    # whenever the config enables adaptive verification, which is what makes
    # enable_adaptive_verification=true the whole switch.
    if eager_upstream_av_enabled(config):
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
