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

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.eager_av_log import EagerAVLogger

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
        # Rows repaired since the last aggregation window closed.
        self._untrusted_rows = 0

        device = req_states.device
        max_num_reqs = req_states.max_num_reqs
        # Upstream leaves this uninitialised and relies on add_request to seed
        # each slot. Fill it with the neutral value instead: the ranking kernel
        # reads the whole buffer's rows for the batch's slots, and a neutral
        # start also keeps the repair counter below about real confidence rows
        # rather than slots nothing has written yet.
        self._confidence_probs = torch.full(
            (max_num_reqs, self.num_speculative_steps), 1.0, dtype=torch.float32, device=device
        )
        self._batch_draft_capacity = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)
        # Two slots to match the upstream stale-buffer indexing; the synchronous
        # recorder only ever writes slot 0, but add_request/get_num_tokens read
        # through self._stale_idx.
        self._stale_confidences = [
            CpuGpuBuffer(max_num_reqs, self.num_speculative_steps, dtype=torch.float32, device=device) for _ in range(2)
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
        self._log = EagerAVLogger(
            lane="upstream",
            interval=envs_ascend.VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL,
        )
        logger.warning(
            "[DSPARK-EAGER-AV/upstream] active: synthetic cost curve, real cost-argmax budget "
            "and device survival top-k, synchronous confidence (no cudagraph, no async D2H). "
            "The curve is invented, so the chosen budget is a layout signal, not a performance one."
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
        # clone(): for a float32 confidence head detach/float/contiguous are all
        # no-ops, so without it this is a view of the speculator's own buffer and
        # the broadcast below would overwrite that buffer on every rank but 0.
        current = confidence_probs[:num_reqs].detach().float().clone()
        get_tp_group().broadcast(current, src=0)
        if self._pending_resets:
            self._stale_confidences[self._stale_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # One blocking D2H of just this batch's rows; repair on the host copy.
        # Copying the whole buffer, as upstream's async path does, would also drag
        # in slots this batch never touched and count them as repaired.
        values = current.cpu().numpy()
        # The confidence head emits non-finite rows during prefill bursts, which
        # prefix caching and async scheduling make common. Here that is not just
        # a bad trimming decision: the inherited budget cumprods this table on
        # the host and _assign_draft_token_budget cumprods and top-ks the device
        # buffer, so one NaN makes the whole batch's ranking and argmax
        # meaningless. Substitute the neutral 1.0, which is what upstream's own
        # add_request uses for a slot it has no information about.
        #
        # 1.0 is the top of the ranking, so a repaired row can win budget from a
        # row with real confidence. That is the accepted trade: this lane's cost
        # curve is synthetic anyway, so its budget is a layout signal, and
        # retaining drafts can only cost throughput, never correctness.
        bad = ~np.isfinite(values) | (values < 0) | (values > 1)
        repaired_rows = int(bad.any(axis=1).sum())
        if repaired_rows:
            values = np.where(bad, 1.0, values)
            # The device buffer is the one the ranking kernel reads, so repairing
            # only the host stale table would leave the top-k running on NaN.
            current = torch.from_numpy(values).to(current.device)
        self._untrusted_rows += repaired_rows
        self._confidence_probs[input_batch.idx_mapping] = current
        # Per slot, so a request absent from this batch keeps its last value --
        # the same end state as upstream's whole-buffer copy, without the cost.
        self._stale_confidences[self._stale_idx].np[input_batch.idx_mapping_np] = values

    def batches_to_profile(self, capture_sizes):
        # Eager runs no capture; there is nothing to time.
        return iter(())

    def set_initial_cost_curves(self, samples):
        # Keep the synthetic table; eager has no valid cudagraph samples to price.
        return None

    def reallocate_drafts(self, req_ids, idx_mapping):
        """Upstream reallocation, then report the step it just decided.

        This is the only point where a whole step's decision is known: the
        budget comes from ``get_num_tokens`` earlier in the step, and the
        per-request split is produced by the device top-k inside the inherited
        call. Reporting from ``get_num_tokens`` instead would have to carry the
        previous step's split, so the line would mix two steps.

        The split is only readable by copying it back, so ask ``sampling()``
        first -- it is true exactly on the steps that print -- and skip the copy
        on every other step. ``_batch_budget`` is read before the inherited call
        consumes it.
        """
        sampling = self._log.sampling()
        num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = self._batch_budget
        scheduled_drafts = sum(num_drafts_per_req.values())
        verify_tokens = sum(num_non_draft_tokens_per_req.values()) + draft_budget

        result = super().reallocate_drafts(req_ids, idx_mapping)

        # budget < scheduled means the batch is genuinely ragged; budget ==
        # scheduled means the synthetic curve left it at the full K.
        self._log.record(
            num_reqs=len(req_ids),
            scheduled_drafts=scheduled_drafts,
            admitted_drafts=draft_budget,
            verify_tokens=verify_tokens,
            capacities=(self._batch_draft_capacity[: len(req_ids)].cpu().tolist() if sampling else None),
            untrusted_rows=self._untrusted_rows,
        )
        self._untrusted_rows = 0
        return result
