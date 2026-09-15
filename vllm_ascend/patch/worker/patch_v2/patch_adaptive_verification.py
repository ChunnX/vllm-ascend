import vllm.v1.worker.gpu.model_runner
import vllm.v1.worker.gpu.spec_decode.adaptive_verification
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    _assign_draft_token_budget,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    maybe_create_adaptive_verification_manager as _orig_maybe_create,
)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.dcut_manual_cap import (
    get_manual_cap_manager_cls,
)

logger = init_logger(__name__)

vllm.v1.worker.gpu.spec_decode.adaptive_verification._assign_draft_token_budget_compiled = _assign_draft_token_budget


def _maybe_create_adaptive_verification_manager(
    *,
    enable_adaptive_verification,
    attn_groups,
    attn_cg_support,
    req_states,
    query_start_loc,
    num_bonus_tokens,
    max_total_logits,
):
    """Create the D-Cut manual-cap manager for GDN, else fall back to upstream.

    The upstream factory rejects GDN (its varlen-mismatch backend check and the
    AttentionCGSupport.ALWAYS requirement), so DSpark GDN runs with no adaptive
    manager. When manual-cap D-Cut is requested, install the manual manager
    instead: the dcut GDN ops provide the variable-length path the upstream check
    guards against, and the manual budget needs neither confidence nor a cost
    table. Otherwise defer to upstream unchanged.
    """
    manual_cap = envs_ascend.VLLM_ASCEND_DSPARK_DCUT_MANUAL_CAP
    if envs_ascend.VLLM_ASCEND_DSPARK_ENABLE_DCUT and manual_cap >= 0:
        logger.warning(
            "[D-Cut] Manual-cap verification manager active (cap=%d): bypassing "
            "the varlen-backend rejection and the confidence cost model. Draft "
            "trimming is driven by a deterministic per-request cap, not "
            "confidence. For D-Cut GDN integration validation, not production.",
            manual_cap,
        )
        manager_cls = get_manual_cap_manager_cls()
        return manager_cls(
            req_states,
            query_start_loc,
            num_bonus_tokens,
            max_total_logits=max_total_logits,
            manual_cap=manual_cap,
        )
    return _orig_maybe_create(
        enable_adaptive_verification=enable_adaptive_verification,
        attn_groups=attn_groups,
        attn_cg_support=attn_cg_support,
        req_states=req_states,
        query_start_loc=query_start_loc,
        num_bonus_tokens=num_bonus_tokens,
        max_total_logits=max_total_logits,
    )


# model_runner.py imports the factory by name (base:132), so the live symbol the
# runner calls lives in its namespace; rebind there and at the definition site.
vllm.v1.worker.gpu.model_runner.maybe_create_adaptive_verification_manager = (
    _maybe_create_adaptive_verification_manager
)
vllm.v1.worker.gpu.spec_decode.adaptive_verification.maybe_create_adaptive_verification_manager = (
    _maybe_create_adaptive_verification_manager
)
