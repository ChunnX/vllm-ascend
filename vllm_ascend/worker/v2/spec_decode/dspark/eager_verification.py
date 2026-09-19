# SPDX-License-Identifier: Apache-2.0
"""Exact-layout eager adapter using upstream manager lifecycle and runner calls."""

import numpy as np
import torch
from vllm.distributed import get_tp_group
from vllm.logger import logger
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager

from vllm_ascend.worker.v2.spec_decode.dspark.eager_policy import batch_layout, select_capacities, validate_threshold


class EagerSurvivalVerificationManager(AdaptiveVerificationManager):
    """Replace only policy/CPU layout; use the existing prepare/sample pipeline.

    Deliberately synchronous: the published confidence belongs to the drafts
    produced in the same propose call, keyed by persistent request slot. No
    stale double buffer, graph profiling or cost table is used in this test lane.
    """

    def __init__(self, req_states, query_start_loc, num_bonus_tokens, max_total_logits, *, threshold):
        # Do not allocate the base manager's unused asynchronous copy streams.
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        self.query_start_loc = query_start_loc
        self.num_bonus_tokens = num_bonus_tokens
        self._max_total_logits = max_total_logits
        self.threshold = validate_threshold(threshold)
        self._confidence = np.ones((req_states.max_num_reqs, self.num_speculative_steps), dtype=np.float32)
        self._valid = np.zeros(req_states.max_num_reqs, dtype=bool)
        self._batch_budget = None
        self._capacity_per_req = None
        self._cu_num_logits = torch.empty_like(query_start_loc)
        self._prepared_req_ids = None
        logger.info(
            "DSpark eager survival verification active: threshold=%s, synchronous confidence, no cost table",
            self.threshold,
        )

    def add_request(self, req_idx):
        self._valid[req_idx] = False
        self._confidence[req_idx].fill(1.0)

    def batches_to_profile(self, capture_sizes):
        return iter(())

    def set_initial_cost_curves(self, samples):
        pass

    def record_confidences(self, confidence_probs, input_batch):
        current = confidence_probs[: input_batch.num_reqs].detach().float().contiguous()
        # TP ranks must choose exactly the same capacities, including threshold
        # boundary cases. The drafter already samples identical tokens per rank.
        get_tp_group().broadcast(current, src=0)
        values = current.cpu().numpy()  # Intentional single blocking D2H for correctness testing.
        if values.shape != (input_batch.num_reqs, self.num_speculative_steps):
            raise ValueError("DSpark confidence shape does not match the proposed draft rows")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("DSpark confidence contains non-finite or out-of-range values")
        self._valid.fill(False)  # Never reuse a proposal from a batch older than the latest one.
        slots = input_batch.idx_mapping_np
        self._confidence[slots] = values
        self._valid[slots] = True

    def get_num_tokens(self, num_tokens_per_req, draft_tokens):
        req_ids = list(num_tokens_per_req)
        scheduled = np.array([len(draft_tokens.get(r, ())) for r in req_ids], dtype=np.int32)
        slots = np.array([self.req_states.req_id_to_index[r] for r in req_ids], dtype=np.intp)
        confidence = self._confidence[slots].copy()
        confidence[~self._valid[slots]] = 1.0  # Missing/new request: retain available drafts.
        max_drafts = max(0, self._max_total_logits - len(req_ids) * self.num_bonus_tokens)
        caps = select_capacities(confidence, scheduled, self.threshold, max_drafts)
        self._capacity_per_req = dict(zip(req_ids, caps.tolist()))
        non_drafts = {r: int(num_tokens_per_req[r]) - int(n) for r, n in zip(req_ids, scheduled)}
        if any(n < 0 for n in non_drafts.values()):
            raise ValueError("Scheduled tokens cannot be smaller than scheduled drafts")
        self._batch_budget = (dict(zip(req_ids, scheduled.tolist())), non_drafts, int(caps.sum()))
        self._valid[slots] = False  # A proposal's confidence is consumed at most once.
        logger.debug("DSpark eager AV: requests=%s capacity=%s threshold=%s", req_ids, caps.tolist(), self.threshold)
        return sum(non_drafts.values()) + int(caps.sum())

    def prepare_request_order(self, req_ids):
        self._prepared_req_ids = tuple(req_ids)

    def compact_batch(self, num_draft_tokens_per_req, num_scheduled_tokens, cu_num_logits_np):
        if self._prepared_req_ids is None or self._batch_budget is None:
            raise RuntimeError("Eager AV requires request order and a budget before compaction")
        _, non_drafts, _ = self._batch_budget
        _, queries, _, logits = batch_layout(
            self._prepared_req_ids, self._capacity_per_req, non_drafts, self.num_bonus_tokens
        )
        return queries, logits

    def reallocate_drafts(self, req_ids, idx_mapping):
        if tuple(req_ids) != self._prepared_req_ids or self._batch_budget is None:
            raise RuntimeError("Request order changed between compaction and reallocation")
        _, non_drafts, budget = self._batch_budget
        _, _, qsl, logits = batch_layout(req_ids, self._capacity_per_req, non_drafts, self.num_bonus_tokens)
        async_copy_to_gpu(qsl, out=self.query_start_loc[: len(qsl)])
        self.query_start_loc[len(qsl) :].fill_(int(qsl[-1]))
        async_copy_to_gpu(logits, out=self._cu_num_logits[: len(logits)])
        self._batch_budget = self._capacity_per_req = self._prepared_req_ids = None
        return self._cu_num_logits[: len(logits)], self.query_start_loc, budget
