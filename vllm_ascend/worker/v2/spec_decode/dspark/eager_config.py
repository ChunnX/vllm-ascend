# SPDX-License-Identifier: Apache-2.0
"""Guard the diagnostic eager lane without changing upstream AV defaults."""

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.eager_policy import validate_threshold


def eager_survival_threshold(config):
    value = envs_ascend.VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD
    if value is None:
        return None
    threshold = validate_threshold(value)
    spec = config.speculative_config
    if spec is None or spec.method != "dspark" or not spec.enable_adaptive_verification:
        raise ValueError("Eager survival testing requires DSpark and enable_adaptive_verification=true")
    if not config.model_config.enforce_eager or spec.enforce_eager is False:
        raise ValueError("Eager survival testing requires --enforce-eager and an eager drafter")
    parallel = config.parallel_config
    if any(
        getattr(parallel, key, 1) != 1
        for key in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("Eager survival testing currently requires PP=PCP=DCP=1")
    if config.lora_config is not None or getattr(parallel, "enable_dbo", False):
        raise ValueError("Eager survival testing does not support LoRA or DBO")
    if not 1 <= spec.num_speculative_tokens <= 15:
        raise ValueError("Eager GDN verification requires 1 <= num_speculative_tokens <= 15")
    return threshold
