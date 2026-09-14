#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Record-only D-Cut trim-decision shadow (stage 1 of docs/adaptive_verify/).

This does NOT trim and does NOT touch GDN/attention state. It answers one
question before the real trimming path is built: *at this concurrency and
context length, would adaptive verification choose to trim, and by how much?*

Each verify step it replays the upstream
``AdaptiveVerificationManager.get_num_tokens`` decision math -- survival =
cumulative product of the confidence head's per-position probabilities, then
``draft_budget = argmax(estimated_accepted / cost)`` against the profiled
Ascend cost table -- on the live confidence, and accumulates the budget it would
have chosen versus the full (untrimmed) budget. Every ``interval`` steps it logs
the trim ratio and the estimated accepted-token loss.

The decision math here MIRRORS ``get_num_tokens`` in
``vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`` (kept intentionally
close so the numbers are faithful) but reads only the confidence tensor and the
cost table, not the manager's request-state / stale-confidence plumbing. Stage 3
of the plan replaces this shadow with the real manager call once the trimming
path is wired; if this file and upstream ``get_num_tokens`` drift, this is the
side to re-sync.

Why a cost table at all: cost is graph-bucket-aware, and (per RFC #15149)
"removing a few logical tokens is useful only when it reaches a cheaper
execution shape". A flat cost slice makes ``argmax`` keep the full budget, which
is exactly the "no benefit at this operating point" signal we want to surface.
"""

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class DSparkAVShadow:
    """Replay the AV budget decision each step and log would-be trimming."""

    def __init__(
        self,
        *,
        cost_tables: tuple,
        num_speculative_steps: int,
        num_bonus_tokens: int,
        max_total_logits: int,
        interval: int,
    ) -> None:
        # draft_cost[num_reqs], verify_cost[num_total_tokens]; numpy on host.
        draft_cost, verify_cost = cost_tables
        self._draft_cost = np.asarray(draft_cost, dtype=np.float64)
        self._verify_cost = np.asarray(verify_cost, dtype=np.float64)
        self.n = int(num_speculative_steps)
        self.num_bonus = int(num_bonus_tokens)
        self.max_total_logits = int(max_total_logits)
        self.interval = max(1, int(interval))
        self._steps = 0
        self._sum_reqs = 0
        self._sum_full = 0.0  # full (untrimmed) draft tokens
        self._sum_budget = 0.0  # chosen draft budget
        self._sum_est_full = 0.0  # estimated accepted draft tokens at full budget
        self._sum_est_budget = 0.0  # ... at chosen budget
        self._sum_trim_ratio = 0.0

    def _decide(self, conf: np.ndarray) -> tuple[int, int, float, float]:
        """Return (chosen_budget, full_drafts, est_accept_full, est_accept_chosen).

        conf: [num_reqs, n] float per-position acceptance probability.
        """
        num_reqs, n = conf.shape
        full_drafts = num_reqs * n
        num_non_draft = num_reqs * self.num_bonus  # one anchor/bonus per request
        # survival[r, i] = P(draft positions 0..i all accepted) for request r.
        survival = np.cumprod(conf.astype(np.float64), axis=1)
        scores = np.sort(survival.reshape(-1))[::-1]  # best draft slots first
        max_budget = min(
            full_drafts,
            max(0, self.max_total_logits - num_reqs * self.num_bonus),
        )
        # Clamp to what the cost table can address.
        max_budget = min(max_budget, len(self._verify_cost) - 1 - num_non_draft)
        if max_budget <= 0:
            return 0, full_drafts, float(scores.sum()), 0.0
        scores = scores[:max_budget]
        # Estimated accepted DRAFT tokens after admitting the top-k slots.
        cumulative = np.concatenate(([0.0], np.cumsum(scores)))  # len max_budget+1
        # Total-tokens axis: non-draft anchors are always verified.
        num_sampling = num_reqs  # steady decode: every request is sampling
        estimated = num_sampling + cumulative
        draft_c = (
            self._draft_cost[num_reqs]
            if num_reqs < len(self._draft_cost)
            else self._draft_cost[-1]
        )
        costs = draft_c + self._verify_cost[num_non_draft : num_non_draft + max_budget + 1]
        m = min(len(estimated), len(costs))
        budget = int(np.argmax(estimated[:m] / costs[:m]))
        return budget, full_drafts, float(cumulative[-1]), float(cumulative[budget])

    def record(self, confidence: torch.Tensor) -> None:
        """Fold one verify step's confidence into the running trim estimate."""
        # One D2H copy per step; the decision math is host-side numpy (float64),
        # matching upstream get_num_tokens (Ascend has no double, but this runs
        # on the CPU, so float64 is fine here).
        conf = confidence.detach().to("cpu", dtype=torch.float32).numpy()
        if conf.ndim != 2 or conf.shape[0] == 0 or conf.shape[1] == 0:
            return
        budget, full, est_full, est_budget = self._decide(conf)
        self._steps += 1
        self._sum_reqs += conf.shape[0]
        self._sum_full += full
        self._sum_budget += budget
        self._sum_est_full += est_full
        self._sum_est_budget += est_budget
        self._sum_trim_ratio += (1.0 - budget / full) if full > 0 else 0.0
        if self._steps % self.interval == 0:
            self.flush()

    def flush(self) -> None:
        if self._steps == 0:
            return
        k = self._steps
        mean_reqs = self._sum_reqs / k
        mean_full = self._sum_full / k
        mean_budget = self._sum_budget / k
        mean_trim = 100.0 * self._sum_trim_ratio / k
        mean_est_full = self._sum_est_full / k
        mean_est_budget = self._sum_est_budget / k
        # Per-request verified draft length before vs after the would-be trim.
        full_len = mean_full / mean_reqs if mean_reqs else 0.0
        cut_len = mean_budget / mean_reqs if mean_reqs else 0.0
        # Emitted at warning level on purpose (like the observer): only appears
        # when VLLM_ASCEND_DSPARK_AV_SHADOW is set, and INFO is filtered by
        # default in many deployments.
        logger.warning(
            "[DSPARK-AV-SHADOW] steps=%d reqs/step~%.1f | draft/req full~%.2f "
            "chosen~%.2f trim=%.1f%% | est_accept_draft full~%.2f chosen~%.2f "
            "loss~%.3f",
            k,
            mean_reqs,
            full_len,
            cut_len,
            mean_trim,
            mean_est_full,
            mean_est_budget,
            mean_est_full - mean_est_budget,
        )
        self._steps = 0
        self._sum_reqs = 0
        self._sum_full = 0.0
        self._sum_budget = 0.0
        self._sum_est_full = 0.0
        self._sum_est_budget = 0.0
        self._sum_trim_ratio = 0.0
