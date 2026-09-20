# SPDX-License-Identifier: Apache-2.0
"""Lane B: the upstream adaptive-verification budget, run eager on Ascend.

Reuses upstream ``get_num_tokens`` (cost-argmax budget over stale confidence) and
``reallocate_drafts`` (device survival top-k) verbatim -- the two paths the graph
phase will reuse -- while adapting the two things that do not hold under eager
Ascend:

1. No cudagraph capture phase runs, so the runner never profiles step costs and
   ``get_num_tokens`` would assert on a missing table. Install a fixed, synthetic
   cost curve at construction. The chosen budget is not a performance signal --
   the curve is invented -- but it makes the real argmax land at an interior,
   ragged budget.
2. The upstream manager builds ``torch.cuda`` copy streams/events for an async
   D2H double buffer. On the v2 Ascend path ``torch.cuda.Stream`` is not reliably
   aliased to the NPU stream at construction time, so this manager does not call
   ``super().__init__`` and records confidences synchronously instead (a blocking
   D2H, as the survival-threshold lane already does). Async D2H is a throughput
   optimisation for the graph phase, not a correctness requirement here.
"""

import numpy as np
import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    build_cost_tables_from_curves,
)

logger = init_logger(__name__)

# Synthetic verify curve shape. Convex in the token count so each extra
# verification token costs a little more than the last: the budget argmax then
# stops at an interior point and the batch is genuinely ragged, rather than
# collapsing to zero drafts or staying at the full K. Milliseconds in name only.
_VERIFY_BASE_MS = 0.2
_VERIFY_COEFF_MS = 0.02
_VERIFY_EXPONENT = 1.3
_DRAFT_FLAT_MS = 0.5


def _synthetic_verify_curve(max_batch_tokens: int) -> list[tuple[int, float]]:
    tokens = np.arange(1, max(2, max_batch_tokens) + 1, dtype=np.int64)
    costs = _VERIFY_BASE_MS + _VERIFY_COEFF_MS * tokens.astype(np.float64) ** _VERIFY_EXPONENT
    return list(zip(tokens.tolist(), costs.tolist()))


class AscendEagerUpstreamAVManager(AdaptiveVerificationManager):
    def __init__(self, req_states, query_start_loc, num_bonus_tokens, max_total_logits):
        # Deliberately not calling super().__init__: it allocates torch.cuda copy
        # streams/events that are not guaranteed to be NPU streams on the v2 path.
        # Replicate the stream-free attributes the inherited budget methods read.
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        self.num_bonus_tokens = num_bonus_tokens
        self._max_total_logits = max_total_logits
        self.query_start_loc = query_start_loc
        self._cudagraph_limit = 0
        self._batch_budget = None

        device = req_states.device
        max_num_reqs = req_states.max_num_reqs
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps), dtype=torch.float32, device=device
        )
        self._batch_draft_capacity = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)
        # Two slots to match the upstream stale-buffer indexing; the synchronous
        # recorder only ever writes slot 0, but add_request/get_num_tokens read
        # through self._stale_idx.
        self._stale_confidences = [
            CpuGpuBuffer(max_num_reqs, self.num_speculative_steps, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self._pending_resets: list[int] = []
        self._stale_idx = 0
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

        max_batch_tokens = req_states.max_num_batched_tokens
        # Draft cost is flat: constant across the budget within a batch, it only
        # shifts every ratio equally. Both curves are identical on every rank, so
        # no broadcast is needed to keep TP capacities consistent.
        draft_curve = [(1, _DRAFT_FLAT_MS), (max(1, max_num_reqs), _DRAFT_FLAT_MS)]
        verify_curve = _synthetic_verify_curve(max_batch_tokens)
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve, verify_curve, max_num_reqs, max_batch_tokens, self._cudagraph_limit
        )
        logger.info(
            "DSpark eager upstream AV active: synthetic cost curve, real cost-argmax "
            "budget and device survival top-k, synchronous confidence (no cudagraph, "
            "no async D2H)"
        )

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    def record_confidences(self, confidence_probs, input_batch) -> None:
        """Publish this step's confidences for the device top-k and CPU budget.

        Synchronous by design: a single blocking D2H makes the stale table exact
        (stale == live under eager), so the next step's budget reads landed
        values without any stream machinery. The TP broadcast keeps every rank's
        capacities identical, including survival ties.
        """
        num_reqs = input_batch.num_reqs
        current = confidence_probs[:num_reqs].detach().float().contiguous()
        get_tp_group().broadcast(current, src=0)
        if self._pending_resets:
            self._stale_confidences[self._stale_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        self._confidence_probs[input_batch.idx_mapping] = current
        # One blocking D2H; validate on the host copy, then publish it as stale.
        values = self._confidence_probs.cpu().numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("DSpark confidence contains non-finite or out-of-range values")
        self._stale_confidences[self._stale_idx].np[:] = values

    def batches_to_profile(self, capture_sizes):
        # Eager runs no capture; there is nothing to time.
        return iter(())

    def set_initial_cost_curves(self, samples):
        # Keep the synthetic table; eager has no valid cudagraph samples to price.
        return None

    def get_num_tokens(self, num_tokens_per_req, draft_tokens):
        total = super().get_num_tokens(num_tokens_per_req, draft_tokens)
        if self._batch_budget is not None:
            _, _, budget = self._batch_budget
            scheduled = sum(len(draft_tokens.get(r, ())) for r in num_tokens_per_req)
            # budget < scheduled means the batch is genuinely ragged; budget ==
            # scheduled means the synthetic curve left it at the full K.
            logger.debug(
                "DSpark eager upstream AV: reqs=%d scheduled_drafts=%d budget=%d total_tokens=%d",
                len(num_tokens_per_req),
                scheduled,
                budget,
                total,
            )
        return total
