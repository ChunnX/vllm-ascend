# SPDX-License-Identifier: Apache-2.0
from functools import wraps

import torch
import vllm.v1.worker.gpu.model_runner as model_runner
from torch.overrides import TorchFunctionMode
from vllm.config import get_current_vllm_config
from vllm.v1.worker.gpu.spec_decode import adaptive_verification

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.av_config import adaptive_verification_enabled


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

adaptive = adaptive_verification
_original_factory = getattr(
    adaptive.maybe_create_adaptive_verification_manager,
    "__wrapped__",
    adaptive.maybe_create_adaptive_verification_manager,
)


@wraps(_original_factory)
def _maybe_create_adaptive_verification_manager(**kwargs):
    # Preserve the ordinary path without a config lookup. An explicitly
    # requested AV manager still requires a loaded confidence head.
    if not kwargs["enable_adaptive_verification"]:
        if envs_ascend.VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV:
            raise ValueError("The DSpark adaptive-verification lanes require a loaded confidence head")
        return _original_factory(**kwargs)
    config = kwargs.get("vllm_config")
    if config is None:
        config = get_current_vllm_config()

    # The upstream cost policy with Ascend ragged-state plumbing. Reached
    # whenever the config enables adaptive verification, which is what makes
    # enable_adaptive_verification=true the whole switch.
    if adaptive_verification_enabled(config):
        from vllm_ascend.worker.v2.spec_decode.dspark.adaptive_verification import AscendAdaptiveVerificationManager

        return AscendAdaptiveVerificationManager(
            kwargs["req_states"],
            kwargs["query_start_loc"],
            kwargs["num_bonus_tokens"],
            kwargs["max_total_logits"],
        )

    return _original_factory(**kwargs)


adaptive.maybe_create_adaptive_verification_manager = _maybe_create_adaptive_verification_manager
model_runner.maybe_create_adaptive_verification_manager = _maybe_create_adaptive_verification_manager
