"""Manual per-request draft capacities for D-Cut GDN verification.

Step 3 of the D-Cut GDN integration (``docs/adaptive_verify/``): drive the
existing MRV2 trimming path (``AdaptiveVerificationManager.compact_batch`` ->
``reallocate_drafts``) with deterministic, operator-supplied per-request caps
instead of the confidence cost model. This validates the variable-length
execution layout -- per-request query boundaries, logit offsets, and draft
capacities -- without depending on the confidence head or a profiled cost table.

The caps are a pure function of the scheduled draft counts, so they are
reproducible and identical across TP ranks (every rank sees the same request
order and scheduled counts). Confidence never enters, by construction: the
policy functions below take no confidence tensor.

The manager subclass overrides only the two confidence entry points
(``get_num_tokens`` and ``reallocate_drafts``) and neuters cost profiling; every
other tensor it produces is the upstream layout unchanged, so the manual path
and the real adaptive path share one downstream (no second compaction).
"""

import functools

import numpy as np


def compute_manual_capacities(scheduled_drafts: np.ndarray, manual_cap: int) -> np.ndarray:
    """Per-request retained draft count under a manual cap.

    Args:
        scheduled_drafts: Per-request scheduled draft count (this round's drafts).
        manual_cap: ``< 0`` retains every scheduled draft (the no-trim step-1
            regression). ``>= 0`` retains at most ``manual_cap`` drafts per
            request.

    Returns:
        A fresh int array of retained draft counts, always ``<= scheduled_drafts``
        element-wise (a request can never retain more drafts than it scheduled),
        so the result is a legal capacity for ``reallocate_drafts``. The value
        depends only on ``scheduled_drafts`` and ``manual_cap`` -- confidence is
        structurally absent.
    """
    if manual_cap < 0:
        capped = scheduled_drafts
    else:
        capped = np.minimum(scheduled_drafts, manual_cap)
    return capped.astype(scheduled_drafts.dtype, copy=True)


def manual_batch_budget(
    num_drafts_per_req: dict[str, int],
    num_non_draft_tokens_per_req: dict[str, int],
    manual_cap: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    """Build the ``_batch_budget`` tuple ``get_num_tokens`` stashes, from a cap.

    Mirrors ``AdaptiveVerificationManager.get_num_tokens``'s stash shape
    ``(num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget)`` but sets
    ``draft_budget`` to the sum of the manual per-request capacities rather than
    the cost-model argmax. Order-independent: ``draft_budget`` is a sum.
    """
    scheduled_drafts = np.fromiter(
        num_drafts_per_req.values(), dtype=np.int32, count=len(num_drafts_per_req)
    )
    capacities = compute_manual_capacities(scheduled_drafts, manual_cap)
    draft_budget = int(capacities.sum())
    return num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget


@functools.lru_cache(maxsize=1)
def get_manual_cap_manager_cls():
    """Build the manual-cap manager subclass on first use.

    The vLLM/torch imports live here so importing this module (for the numpy-only
    policy functions above, e.g. in a CPU unit test) needs neither vLLM nor an
    accelerator. ``lru_cache`` keeps a single class object.
    """
    import torch
    from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )

    class DcutManualCapVerificationManager(AdaptiveVerificationManager):
        """Adaptive-verification manager whose budget is a manual cap, not
        confidence. Overrides the two confidence entry points and skips cost
        profiling; all downstream layout is the upstream path unchanged."""

        def __init__(self, *args, manual_cap: int, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.manual_cap = manual_cap

        def batches_to_profile(self, capture_sizes):
            # Manual budget needs no cost tables; profile nothing so a GDN
            # dummy-run cannot fail startup and no cost curve is required. Keep
            # the _cudagraph_limit side effect the base sets here.
            self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
            return iter(())

        def set_initial_cost_curves(self, samples) -> None:
            return

        def record_confidences(self, confidence_probs, input_batch) -> None:
            # Manual budgeting never reads confidence, so drop the per-step
            # publish/D2H copy entirely -- the manual path does not depend on a
            # confidence head at all.
            return

        def get_num_tokens(self, num_tokens_per_req, draft_tokens):
            req_ids = list(num_tokens_per_req)
            num_drafts_per_req = {
                req_id: len(draft_tokens.get(req_id, ())) for req_id in req_ids
            }
            num_non_draft_tokens_per_req = {
                req_id: int(num_tokens_per_req[req_id]) - num_drafts_per_req[req_id]
                for req_id in req_ids
            }
            self._batch_budget = manual_batch_budget(
                num_drafts_per_req, num_non_draft_tokens_per_req, self.manual_cap
            )
            _, _, draft_budget = self._batch_budget
            return sum(num_non_draft_tokens_per_req.values()) + draft_budget

        def reallocate_drafts(self, req_ids, idx_mapping):
            batch_budget, self._batch_budget = self._batch_budget, None
            assert batch_budget is not None
            num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = batch_budget
            num_reqs = idx_mapping.shape[0]
            scheduled_drafts = np.fromiter(
                (num_drafts_per_req[req_id] for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_non_draft_tokens = np.fromiter(
                (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_tokens = int(num_non_draft_tokens.sum()) + draft_budget

            # Manual capacities replace the confidence ranking. sum(manual
            # capacities) == draft_budget (both come from compute_manual_capacities
            # over the same scheduled counts), so the layout below is consistent.
            capacities = self._batch_draft_capacity[:num_reqs]
            manual_capacities = compute_manual_capacities(scheduled_drafts, self.manual_cap)
            async_copy_to_gpu(manual_capacities, out=capacities)

            num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
            async_copy_to_gpu(num_non_draft_tokens, out=num_non_draft_tokens_gpu)
            self._cu_num_logits[:1].zero_()
            torch.cumsum(
                capacities + self.num_bonus_tokens,
                dim=0,
                out=self._cu_num_logits[1 : num_reqs + 1],
            )
            self.query_start_loc[:1].zero_()
            torch.cumsum(
                capacities + num_non_draft_tokens_gpu,
                dim=0,
                out=self.query_start_loc[1 : num_reqs + 1],
            )
            self.query_start_loc[num_reqs + 1 :].fill_(num_tokens)
            return (
                self._cu_num_logits[: num_reqs + 1],
                self.query_start_loc,
                draft_budget,
            )

    return DcutManualCapVerificationManager
