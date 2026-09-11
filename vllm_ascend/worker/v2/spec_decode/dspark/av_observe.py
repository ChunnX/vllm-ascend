#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Record-only observation of DSpark adaptive-verification signals (MRV2).

This does NOT trim verification. It only watches, so the value of the DSpark
confidence head can be judged before the (upstream #47808) trimming path is
built on Ascend. Enable with ``VLLM_ASCEND_DSPARK_AV_OBSERVE=1``.

Per draft position it accumulates the mean predicted acceptance probability
(the confidence head's sigmoid output) alongside the *actual* per-position
acceptance rate observed from the rejection sampler, and logs the two side by
side every ``VLLM_ASCEND_DSPARK_AV_OBSERVE_INTERVAL`` verification steps.

Alignment note: the confidence recorded for a step's drafts was produced by
the *previous* step's ``propose`` in the speculator's own batch order, while
``num_sampled`` is this step's batch order. In steady-state decode the request
set and its order are stable, so the leading ``num_reqs`` rows line up; brief
request churn (prefill admission, preemption) can misalign a few rows, which
only adds statistical noise to an averaged calibration curve.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class DSparkAVObserver:
    """Accumulate confidence-vs-acceptance calibration and log it periodically."""

    def __init__(
        self,
        *,
        num_speculative_steps: int,
        num_bonus_tokens: int,
        device: torch.device,
        interval: int,
    ) -> None:
        self.n = num_speculative_steps
        self.num_bonus = num_bonus_tokens
        self.interval = max(1, interval)
        # float64 accumulators keep long runs from losing precision.
        self._sum_conf = torch.zeros(self.n, dtype=torch.float64, device=device)
        self._sum_acc = torch.zeros(self.n, dtype=torch.float64, device=device)
        self._count = 0
        self._steps = 0

    def record(self, confidence: torch.Tensor, num_sampled: torch.Tensor) -> None:
        """Fold one verification step in.

        confidence: [num_reqs, n] predicted per-position acceptance probability.
        num_sampled: [num_reqs] tokens accepted per request, including the bonus.
        """
        num_reqs = int(num_sampled.shape[0])
        if num_reqs == 0 or self.n == 0:
            return
        # Draft tokens accepted per request (drop the always-present bonus).
        accepted_draft = (num_sampled.to(torch.int64) - self.num_bonus).clamp_min(0)
        steps = torch.arange(self.n, device=num_sampled.device)
        # Position i is accepted iff at least i+1 draft tokens were accepted.
        accepted_mask = accepted_draft[:, None] > steps[None, :]
        self._sum_conf += confidence[:num_reqs].to(torch.float64).sum(dim=0)
        self._sum_acc += accepted_mask.to(torch.float64).sum(dim=0)
        self._count += num_reqs
        self._steps += 1
        if self._steps % self.interval == 0:
            self.flush()

    def flush(self) -> None:
        if self._count == 0:
            return
        # One D2H sync, only on the logging boundary.
        conf = (self._sum_conf / self._count).cpu().tolist()
        acc = (self._sum_acc / self._count).cpu().tolist()
        cols = " ".join(
            f"p{i}[conf={conf[i]:.3f} acc={acc[i]:.3f} gap={conf[i] - acc[i]:+.3f}]"
            for i in range(self.n)
        )
        # Mean accepted length ~= bonus + sum of per-position acceptance rates.
        mean_accept_len = self.num_bonus + sum(acc)
        logger.info(
            "[DSPARK-AV-OBSERVE] reqs=%d over %d steps | mean_accept_len~%.2f | %s",
            self._count,
            self.interval,
            mean_accept_len,
            cols,
        )
        self._sum_conf.zero_()
        self._sum_acc.zero_()
        self._count = 0
