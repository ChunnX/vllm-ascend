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
   D2H double buffer. ``torch.cuda.Stream`` is not guaranteed to alias the NPU
   stream on the v2 path, so this manager does not call ``super().__init__`` and
   builds them defensively, falling back to a blocking copy if they cannot be
   created. The async path is not an optimisation to defer: a blocking copy
   waits on the confidence the drafter produced *this* step, so the host waits
   for the draft forward every step and async scheduling stops overlapping
   anything. Measured on specbench, that alone made turning the feature on cost
   about a tenth of TPOT while changing accepted length by two percent -- a cost
   of the instrumentation, not of adaptive verification.

Both diagnostics therefore accumulate on device and are read once per logging
window: a per-step copy to count them would reintroduce exactly the stall this
lane just removed.
"""

import numpy as np
import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import stream
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    build_cost_tables_from_curves,
)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.v2.spec_decode.dspark.eager_av_log import EagerAVLogger
from vllm_ascend.worker.v2.spec_decode.dspark.eager_config import GRAPH_MODE_NONE, av_graph_mode

logger = init_logger(__name__)

# Synthetic verify curve shape. Convex in the token count so each extra
# verification token costs a little more than the last: the budget argmax then
# stops at an interior point and the batch is genuinely ragged, rather than
# collapsing to zero drafts or staying at the full K. Milliseconds in name only.
_VERIFY_BASE_MS = 0.2
_VERIFY_COEFF_MS = 0.02
_VERIFY_EXPONENT = 1.3
_DRAFT_FLAT_MS = 0.5

# Below this, a difference between two buckets is not worth the control
# overhead trimming costs: probability handling, the CPU decision, the TP
# broadcast and ragged metadata all happen whether or not Q shrinks.
_FLAT_COST_TOLERANCE_MS = 1.0


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
        # With no graph mode there is nothing to time: every step runs eager, and
        # the measured curve would be flat in Q, which is a cost table that cannot
        # inform any trimming decision. Under a graph mode the inherited profiling
        # runs instead and replaces the synthetic curve installed below.
        self._graph_mode = av_graph_mode()
        self._profiles_cost = self._graph_mode != GRAPH_MODE_NONE
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
        self._async_confidence = self._setup_async_confidence(device)
        # [rows repaired, steps whose leading confidence row moved]. Both live on
        # device and are accumulated with device ops, so counting them costs no
        # synchronisation; the logging window reads the pair once.
        self._health = torch.zeros(2, dtype=torch.int64, device=device)
        self._health_steps = 0
        # NaN compares unequal to everything, so the first step counts as moved.
        self._last_row = torch.full((self.num_speculative_steps,), float("nan"), dtype=torch.float32, device=device)

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
        # Name the confidence path here, not just in the failure warning: this
        # banner is where someone checks what actually ran, and reading
        # "synchronous" while the async copy is live (or the reverse) is worse
        # than saying nothing.
        confidence = "async confidence copy" if self._async_confidence else "blocking confidence copy"
        if self._profiles_cost:
            logger.warning(
                "[DSPARK-EAGER-AV/upstream] active: graph mode %s, profiling a real cost table, "
                "real cost-argmax budget and device survival top-k, %s. "
                "The synthetic curve below is only a fallback if profiling yields nothing.",
                self._graph_mode,
                confidence,
            )
        else:
            logger.warning(
                "[DSPARK-EAGER-AV/upstream] active: synthetic cost curve, real cost-argmax budget "
                "and device survival top-k, %s (no cudagraph). The curve is invented, so the "
                "chosen budget is a layout signal, not a performance one.",
                confidence,
            )

    def _setup_async_confidence(self, device) -> bool:
        """Build the async D2H double buffer, or report why we cannot.

        Defensive because the reason this manager skips ``super().__init__`` is
        that ``torch.cuda.Stream`` may not be the NPU stream here. If it is not,
        staying synchronous costs throughput and nothing else, which is a much
        better outcome than failing to start.
        """
        try:
            self._copy_stream = torch.cuda.Stream(device)
            self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        except Exception as exc:  # noqa: BLE001 - any stream failure means stay sync
            logger.warning(
                "[DSPARK-EAGER-AV/upstream] async confidence copy unavailable (%s); falling back "
                "to a blocking copy each step. Correct, but the host then waits for the draft "
                "forward every step and throughput will show it.",
                exc,
            )
            return False
        return True

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    def record_confidences(self, confidence_probs, input_batch) -> None:
        """Publish this step's confidences for the device top-k and CPU budget."""
        num_reqs = input_batch.num_reqs
        # clone(): for a float32 confidence head detach/float/contiguous are all
        # no-ops, so without it this is a view of the speculator's own buffer and
        # the broadcast below would overwrite that buffer on every rank but 0.
        raw = confidence_probs[:num_reqs].detach().float().clone()
        get_tp_group().broadcast(raw, src=0)
        self._accumulate_health(raw)
        # The confidence head emits non-finite rows during prefill bursts, which
        # prefix caching and async scheduling make common. Here that is not just
        # a bad trimming decision: the inherited budget cumprods this table on
        # the host and _assign_draft_token_budget cumprods and top-ks the device
        # buffer, and NaN sorts to the front of the reversed ranking, so argmax
        # collapses every request's budget -- one bad row costs the whole batch.
        # Substitute the neutral 1.0, which is what add_request already uses for
        # a slot it has no information about.
        #
        # 1.0 is the top of the ranking, so a repaired row can win budget from a
        # row with real confidence. That is the accepted trade: retaining drafts
        # can only cost throughput, never correctness.
        #
        # On device on purpose. Repairing a host copy is what forced a blocking
        # copy every step, and the repair never needed the values on host.
        current = torch.nan_to_num(raw, nan=1.0, posinf=1.0, neginf=1.0)
        if self._async_confidence:
            self._record_async(current, input_batch)
        else:
            self._record_sync(current, input_batch)

    def _record_async(self, current, input_batch) -> None:
        """Upstream's double-buffered copy: only ever wait on an older step.

        The event synchronised here belongs to the copy started two steps ago,
        so by now it has landed and the wait is free. This step's copy is
        enqueued on a side stream and read by a later step's budget -- the host
        never waits for the drafter it just launched.
        """
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = current
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def _record_sync(self, current, input_batch) -> None:
        """Fallback when no side stream could be created. Blocks on this step."""
        if self._pending_resets:
            self._stale_confidences[self._stale_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        self._confidence_probs[input_batch.idx_mapping] = current
        # Per slot, so a request absent from this batch keeps its last value --
        # the same end state as the whole-buffer copy, without the cost.
        self._stale_confidences[self._stale_idx].np[input_batch.idx_mapping_np] = current.cpu().numpy()

    def _accumulate_health(self, raw) -> None:
        """Count repaired rows and whether the signal moved, without syncing.

        Two things need watching and neither may cost a copy per step:

        - Non-finite rows, because one of them collapses the batch's budget.
        - Whether the confidence is still being recomputed at all. Under graph
          the op is only recomputed per replay if it was traced into the
          captured draft graph; left out, every replay skips it and the buffer
          freezes at its pre-capture value. Trimming is only a policy, so the
          tokens stay byte-identical to a fixed-K run while every budget is
          decided on numbers that stopped moving -- invisible to any output
          comparison, and visible here as a row that never changes.

        Both accumulate into one device tensor with device ops, and the logging
        window reads the pair in a single copy.
        """
        if not raw.numel():
            return
        self._health_steps += 1
        self._health[0] += (~torch.isfinite(raw)).any(dim=1).sum()
        self._health[1] += (raw[0] != self._last_row).any()
        self._last_row.copy_(raw[0])
        if not self._log.sampling():
            return
        repaired, moved = (int(v) for v in self._health.cpu())
        self._untrusted_rows += repaired
        self._log.note_confidence_steps(moved=moved, steps=self._health_steps)
        self._health.zero_()
        self._health_steps = 0

    def note_graph_mode(self, cg_mode) -> None:
        """Report the cudagraph mode the runner dispatched this step under."""
        self._log.note_graph_mode(cg_mode)

    def batches_to_profile(self, capture_sizes):
        if not self._profiles_cost:
            # Eager runs no capture; there is nothing to time.
            return iter(())
        # The inherited generator also profiles past the capture limit on purpose:
        # real steps run there under piecewise, which is exactly where a trimmed
        # batch lands, and extrapolating from the captured sizes alone badly
        # underestimates them.
        return super().batches_to_profile(capture_sizes)

    def set_initial_cost_curves(self, samples):
        if not self._profiles_cost:
            # Keep the synthetic table; eager has no valid cudagraph samples to price.
            return None
        super().set_initial_cost_curves(samples)
        self._report_cost_table(samples)

    def _report_cost_table(self, samples) -> None:
        """Print the measured verify curve, because it decides what comes next.

        Whether trimming can pay is not a property of the trimming logic: it is
        whether a smaller Q lands in a cheaper graph. A curve that is flat in Q
        means no choice of budget changes step time, so the controller has
        nothing to optimise however good its confidence is. Printing it turns
        that from an assumption into a number.
        """
        _, verify_ms = self.cost_tables
        measured = sorted({int(s.num_target_tokens) for s in samples})
        measured = [q for q in measured if q < len(verify_ms)]
        if not measured:
            logger.warning("[DSPARK-EAGER-AV/upstream] cost table: profiling produced no samples")
            return
        points = ", ".join(f"Q={q}:{verify_ms[q]:.2f}ms" for q in measured)

        # Only the reachable range can inform a decision. This service can present
        # at most max_num_seqs * (num_spec + 1) verification tokens, and a Q above
        # the capture limit leaves the graph entirely -- a step change that says
        # "stay captured", not "trim to a cheaper bucket". Reporting the spread
        # over every profiled point, including Q values this configuration will
        # never see, makes a flat reachable range look like a steep curve.
        reachable_max = min(
            self.req_states.max_num_reqs * (self.num_speculative_steps + 1),
            self._cudagraph_limit or len(verify_ms) - 1,
        )
        reachable = [q for q in measured if q <= reachable_max]
        if len(reachable) < 2:
            logger.warning(
                "[DSPARK-EAGER-AV/upstream] cost table (%d samples, graph_limit=%d): %s | "
                "only %d profiled point(s) at or below the reachable Q=%d, so the curve says "
                "nothing about trimming here",
                len(samples),
                self._cudagraph_limit,
                points,
                len(reachable),
                reachable_max,
            )
            return
        spread = verify_ms[reachable[-1]] - verify_ms[reachable[0]]
        verdict = (
            "flat: trimming cannot pay at this scale, and declining to trim is the correct decision"
            if abs(spread) < _FLAT_COST_TOLERANCE_MS
            else "varies with Q, so a smaller budget can land in a cheaper bucket"
        )
        logger.warning(
            "[DSPARK-EAGER-AV/upstream] cost table (%d samples, graph_limit=%d): %s | "
            "reachable Q<=%d spread=%.2fms (%s)",
            len(samples),
            self._cudagraph_limit,
            points,
            reachable_max,
            spread,
            verdict,
        )

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
