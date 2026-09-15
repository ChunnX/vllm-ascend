#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Record-only observation of DSpark adaptive-verification signals (MRV2).

This does NOT trim verification. It only watches, so the value of the DSpark
confidence head can be judged before the (upstream #47808) trimming path is
built on Ascend. Enable with ``VLLM_ASCEND_DSPARK_AV_OBSERVE=1``.

Calibration semantics (important): the confidence head's output is the
*conditional* per-position acceptance probability -- upstream takes its
cumulative product to get a *survival* probability (see
``AdaptiveVerificationManager._assign_draft_token_budget``). Draft acceptance is
prefix-ordered, so the actual quantity the rejection sampler gives us per
position is the *survival* rate P(accepted >= i+1). The apples-to-apples
calibration is therefore ``cumprod(confidence)`` vs actual survival -- NOT raw
confidence vs survival, which compares conditional-against-cumulative and looks
like growing "over-confidence" even for a perfectly calibrated head. The
survival is accumulated *per sample* then averaged (mean(cumprod) != cumprod of
an averaged confidence). Each log line shows raw ``conf`` (reference), predicted
``surv`` = cumprod(conf), and actual ``acc``, with ``gap = surv - acc``.

Alignment caveat: the confidence for a step's drafts was produced by the
*previous* ``propose`` in the speculator's batch order, aligned to this step's
``num_sampled`` only by leading row. In steady-state decode the request set is
stable so the rows line up, but prefill admission / preemption / reorder can
misalign rows *systematically*, not just as zero-mean noise. This is a coarse
calibration estimate; the correct fix is a request/slot-identity-aligned
collection (recorded as a follow-up in the D-Cut plan). Non-finite confidence
rows (seen during prefill bursts) are dropped and counted, never summed.
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
        # float32 accumulators: Ascend NPU has poor/no float64 (double) support --
        # a float64 op here silently failed the whole record() before flush(),
        # which is why no observation line appeared. Confidence is in [0, 1] and
        # only summed over one interval, so float32 is plenty for an averaged
        # calibration curve.
        # Per-sample accumulators (summed over requests within one interval):
        #   _sum_conf: raw confidence[i]  -- the head's CONDITIONAL per-position
        #              acceptance probability (upstream cumprods it for survival).
        #   _sum_surv: cumprod(confidence)[i] -- PREDICTED prefix-survival, the
        #              apples-to-apples match for _sum_acc. Computed per sample
        #              then summed: mean(cumprod) != cumprod(mean), so we must NOT
        #              cumprod an averaged confidence.
        #   _sum_acc:  actual prefix-survival P(accepted >= i+1).
        # float32: Ascend NPU has poor/no float64 support; confidence is in [0,1]
        # summed over one interval, so float32 is plenty here.
        self._sum_conf = torch.zeros(self.n, dtype=torch.float32, device=device)
        self._sum_surv = torch.zeros(self.n, dtype=torch.float32, device=device)
        self._sum_acc = torch.zeros(self.n, dtype=torch.float32, device=device)
        self._count = 0  # valid (finite) request rows folded in
        self._nan_rows = 0  # request rows dropped for non-finite confidence
        self._steps = 0

    def record(self, confidence: torch.Tensor, num_sampled: torch.Tensor) -> None:
        """Fold one verification step in.

        confidence: [num_reqs, n] the head's CONDITIONAL per-position acceptance
            probability (NOT survival).
        num_sampled: [num_reqs] tokens accepted per request, including the bonus.

        Calibration caveat: the confidence for these drafts was produced by the
        previous propose() in the speculator's batch order, aligned to this step's
        num_sampled only by leading row (see module docstring). This is a coarse
        estimate; a request/slot-identity-aligned collection is the correct fix.
        """
        num_reqs = int(num_sampled.shape[0])
        if num_reqs == 0 or self.n == 0:
            return
        conf = confidence[:num_reqs].to(torch.float32)
        # Drop non-finite rows (e.g. NaN confidence seen during prefill bursts)
        # instead of poisoning the running sums; count them separately.
        finite = torch.isfinite(conf).all(dim=1)
        num_finite = int(finite.sum().item())
        self._nan_rows += num_reqs - num_finite
        if num_finite == 0:
            self._steps += 1
            if self._steps % self.interval == 0:
                self.flush()
            return
        conf = conf[finite]
        # Draft tokens accepted per request (drop the always-present bonus).
        accepted_draft = (num_sampled.to(torch.int64) - self.num_bonus).clamp_min(0)
        accepted_draft = accepted_draft[finite]
        steps = torch.arange(self.n, device=conf.device)
        # Position i is accepted iff at least i+1 draft tokens were accepted.
        accepted_mask = accepted_draft[:, None] > steps[None, :]
        self._sum_conf += conf.sum(dim=0)
        # Predicted prefix-survival: per-sample cumprod, then sum over samples.
        self._sum_surv += torch.cumprod(conf, dim=1).sum(dim=0)
        self._sum_acc += accepted_mask.to(torch.float32).sum(dim=0)
        self._count += num_finite
        self._steps += 1
        if self._steps % self.interval == 0:
            self.flush()

    def flush(self) -> None:
        if self._count == 0:
            if self._nan_rows:
                logger.warning(
                    "[DSPARK-AV-OBSERVE] no finite rows over %d steps "
                    "(dropped %d non-finite rows)",
                    self.interval,
                    self._nan_rows,
                )
                self._nan_rows = 0
            return
        # One D2H sync, only on the logging boundary.
        conf = (self._sum_conf / self._count).cpu().tolist()
        surv = (self._sum_surv / self._count).cpu().tolist()
        acc = (self._sum_acc / self._count).cpu().tolist()
        # Calibration = PREDICTED survival (cumprod of conditional conf) vs ACTUAL
        # prefix-survival. gap on raw conf would compare conditional-vs-survival
        # (always positive, growing with position) and is NOT a calibration error.
        cols = " ".join(
            f"p{i}[conf={conf[i]:.3f} surv={surv[i]:.3f} acc={acc[i]:.3f} "
            f"gap={surv[i] - acc[i]:+.3f}]"
            for i in range(self.n)
        )
        # Mean accepted length ~= bonus + sum of per-position survival.
        mean_accept_len = self.num_bonus + sum(acc)
        # Emitted at warning level on purpose: this line only appears when the
        # operator explicitly sets VLLM_ASCEND_DSPARK_AV_OBSERVE, and INFO is
        # filtered by default in many deployments, so warning guarantees the
        # observation is actually visible.
        logger.warning(
            "[DSPARK-AV-OBSERVE] reqs=%d dropped_nan=%d over %d steps | "
            "mean_accept_len~%.2f | %s",
            self._count,
            self._nan_rows,
            self.interval,
            mean_accept_len,
            cols,
        )
        self._sum_conf.zero_()
        self._sum_surv.zero_()
        self._sum_acc.zero_()
        self._count = 0
        self._nan_rows = 0
