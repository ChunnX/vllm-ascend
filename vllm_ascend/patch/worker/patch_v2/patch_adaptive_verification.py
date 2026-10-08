from functools import wraps

import torch
import vllm.v1.worker.gpu.model_runner as model_runner
from torch.overrides import TorchFunctionMode
from vllm.config import get_current_vllm_config
from vllm.v1.worker.gpu.spec_decode import adaptive_verification

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.eager_config import (
    eager_survival_threshold,
    eager_upstream_av_enabled,
)


def _index_fill(tensor, dim, index, value):
    # Import lazily to avoid circular imports during plugin startup.
    from vllm_ascend.device.device_op import DeviceOperator

    return DeviceOperator.index_fill(tensor, dim, index, value)


class _IndexFillMode(TorchFunctionMode):
    """Temporarily route index_fill_ through the Ascend device adaptor.

    A5's current native index_fill_ path synchronizes while converting the
    device index tensor to a host vector. Keep the upstream budget allocator
    unchanged and override only this operation within its dynamic scope.
    Remove this mode once the native A5 index_fill_ operator is ready.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.Tensor.index_fill_:
            return _index_fill(*args, **kwargs)
        return func(*args, **kwargs)


_original_assign_draft_token_budget = adaptive_verification._assign_draft_token_budget


def _assign_draft_token_budget_ascend(*args, **kwargs):
    with _IndexFillMode():
        return _original_assign_draft_token_budget(*args, **kwargs)


adaptive_verification._assign_draft_token_budget_compiled = _assign_draft_token_budget_ascend

_original_factory = getattr(
    adaptive_verification.maybe_create_adaptive_verification_manager,
    "__wrapped__",
    adaptive_verification.maybe_create_adaptive_verification_manager,
)


# The DSpark eager adaptive-verification lanes. The factory is wrapped rather
# than replaced, so a config that enables adaptive verification without naming
# a lane still gets the upstream manager.
#
# _assign_draft_token_budget_compiled is deliberately not touched here: the
# index-fill mode above already owns that assignment, and it bypasses the
# torch.compile'd allocator for a stronger reason than the lanes need.
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


adaptive_verification.maybe_create_adaptive_verification_manager = _maybe_create_adaptive_verification_manager
model_runner.maybe_create_adaptive_verification_manager = _maybe_create_adaptive_verification_manager
