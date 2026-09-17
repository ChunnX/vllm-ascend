"""Manual per-request draft capacities for D-Cut GDN verification.

Step 3 of the D-Cut GDN integration (``docs/adaptive_verify/``): drive the
existing MRV2 trimming path (``AdaptiveVerificationManager.compact_batch`` ->
``reallocate_drafts``) with deterministic, operator-supplied per-request caps
instead of the confidence cost model. This validates the variable-length
execution layout -- per-request query boundaries, logit offsets, and draft
capacities -- without depending on the confidence head or a profiled cost table.

The cap spec is a *pattern*, not a single number, and that distinction is the
point. A single global cap trims every request to the same width, so in steady
state (every request scheduling the full draft count) the batch comes out
uniform at ``cap + 1`` tokens per request -- indistinguishable, at the graph
layer, from running with ``num_speculative_tokens = cap``. It therefore cannot
exercise the ragged layout D-Cut exists to produce. A pattern such as ``7,0,3,1``
assigns a different cap per batch position, so the batch is genuinely ragged and
the variable-length path is actually under test.

The caps are a pure function of the scheduled draft counts and the request
order, so they are reproducible and identical across TP ranks (every rank sees
the same scheduler request order and the same scheduled counts). Confidence
never enters, by construction: the policy functions below take no confidence
tensor.

``get_num_tokens`` resolves the per-request capacity once and stashes it keyed
by request id; ``reallocate_drafts`` looks those values up rather than
recomputing them. That matters for a pattern: the two entry points see the same
requests in *different* orders (scheduler order vs. the ``sort_batch_req_ids``
batch order), so a position-indexed pattern recomputed on the second order can
pair caps with different requests and sum to a different budget than the one
already used to size the batch. Looking the values up makes
``sum(capacities) == draft_budget`` structural instead of an invariant that
happens to hold because a global minimum is order-independent.

The manager subclass overrides only the two confidence entry points
(``get_num_tokens`` and ``reallocate_drafts``) and neuters cost profiling; every
other tensor it produces is the upstream layout unchanged, so the manual path
and the real adaptive path share one downstream (no second compaction).
"""

import functools
from collections.abc import Sequence

import numpy as np

# A cap entry of -1 (or any negative value) leaves that request untrimmed.
NO_CAP = -1


def parse_manual_cap_spec(raw: str) -> tuple[int, ...]:
    """Parse the manual-cap env value into a per-batch-position cap pattern.

    Accepts a single integer (``"2"`` -> every request capped at 2, the original
    global-cap behaviour) or a comma-separated pattern (``"7,0,3,1"`` -> batch
    position i is capped at ``pattern[i % 4]``). Negative entries mean "do not
    trim this position".

    Raises:
        ValueError: if the spec is empty or holds a non-integer entry, so a typo
            fails at startup rather than silently disabling trimming.
    """
    entries = [item.strip() for item in raw.split(",") if item.strip()]
    if not entries:
        raise ValueError(f"empty D-Cut manual cap spec: {raw!r}")
    return tuple(int(item) for item in entries)


def manual_cap_enabled(manual_caps: Sequence[int]) -> bool:
    """Whether a cap spec asks for trimming at all.

    A spec of just ``-1`` is the default "disabled" value; anything that caps at
    least one position turns the manual manager on.
    """
    return any(cap >= 0 for cap in manual_caps)


def compute_manual_capacities(
    scheduled_drafts: np.ndarray, manual_caps: Sequence[int]
) -> np.ndarray:
    """Per-request retained draft count under a manual cap pattern.

    Args:
        scheduled_drafts: Per-request scheduled draft count (this round's
            drafts), in the order the caps should be assigned.
        manual_caps: The cap pattern. Batch position ``i`` is capped at
            ``manual_caps[i % len(manual_caps)]``; a negative entry retains every
            scheduled draft for that position (the no-trim step-1 regression).

    Returns:
        A fresh int array of retained draft counts, always ``<= scheduled_drafts``
        element-wise (a request can never retain more drafts than it scheduled),
        so the result is a legal capacity for ``reallocate_drafts``. The value
        depends only on ``scheduled_drafts`` and the pattern -- confidence is
        structurally absent.
    """
    num_reqs = scheduled_drafts.shape[0]
    pattern = np.asarray(manual_caps, dtype=np.int64)
    caps = pattern[np.arange(num_reqs) % pattern.shape[0]]
    capped = np.where(caps < 0, scheduled_drafts, np.minimum(scheduled_drafts, caps))
    return capped.astype(scheduled_drafts.dtype, copy=True)


def manual_batch_budget(
    num_drafts_per_req: dict[str, int],
    num_non_draft_tokens_per_req: dict[str, int],
    manual_caps: Sequence[int],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    """Build the ``_batch_budget`` tuple ``get_num_tokens`` stashes, from a cap
    pattern.

    Extends ``AdaptiveVerificationManager.get_num_tokens``'s stash with the
    resolved per-request capacities, so ``reallocate_drafts`` can look them up
    instead of re-deriving them from a different request order. ``draft_budget``
    is the sum of exactly those capacities, which is what makes the device-side
    layout consistent with the batch size already chosen from this budget.
    """
    scheduled_drafts = np.fromiter(
        num_drafts_per_req.values(), dtype=np.int32, count=len(num_drafts_per_req)
    )
    capacities = compute_manual_capacities(scheduled_drafts, manual_caps)
    capacity_per_req = {
        req_id: int(capacity)
        for req_id, capacity in zip(num_drafts_per_req, capacities)
    }
    draft_budget = int(capacities.sum())
    return (
        num_drafts_per_req,
        num_non_draft_tokens_per_req,
        capacity_per_req,
        draft_budget,
    )


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
        """Adaptive-verification manager whose budget is a manual cap pattern,
        not confidence. Overrides the two confidence entry points and skips cost
        profiling; all downstream layout is the upstream path unchanged."""

        def __init__(self, *args, manual_caps: Sequence[int], **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.manual_caps = tuple(manual_caps)

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
                num_drafts_per_req, num_non_draft_tokens_per_req, self.manual_caps
            )
            _, _, _, draft_budget = self._batch_budget
            return sum(num_non_draft_tokens_per_req.values()) + draft_budget

        def reallocate_drafts(self, req_ids, idx_mapping):
            batch_budget, self._batch_budget = self._batch_budget, None
            assert batch_budget is not None
            (
                _num_drafts_per_req,
                num_non_draft_tokens_per_req,
                capacity_per_req,
                draft_budget,
            ) = batch_budget
            num_reqs = idx_mapping.shape[0]
            num_non_draft_tokens = np.fromiter(
                (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_tokens = int(num_non_draft_tokens.sum()) + draft_budget

            # Manual capacities replace the confidence ranking. They are looked
            # up per request rather than recomputed, so their sum is exactly the
            # draft_budget the batch was already sized from even though req_ids
            # is in batch order while the budget was resolved in scheduler
            # order -- a position-indexed pattern is not order-independent.
            capacities = self._batch_draft_capacity[:num_reqs]
            manual_capacities = np.fromiter(
                (capacity_per_req[req_id] for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            assert int(manual_capacities.sum()) == draft_budget
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
