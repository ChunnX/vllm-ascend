# SPDX-License-Identifier: Apache-2.0
"""Interval-aggregated warn logging for the eager adaptive-verification lanes.

Warn level on purpose: both lanes are diagnostic, so their trimming decisions
have to be visible in an ordinary ``vllm serve`` log without raising the whole
process to DEBUG -- at TP=4 a per-step debug line from every rank buries the
failure it was meant to locate.

Aggregated on purpose: one line per ``interval`` steps carries the means, so a
long run stays readable and a change of regime still shows up. Nothing here
reads device memory; a lane that wants to report per-request capacities asks
``sampling()`` first, so the copy happens only on the step that prints.
"""

from collections.abc import Sequence

from vllm.logger import init_logger

logger = init_logger(__name__)


class EagerAVLogger:
    """Accumulate one lane's trimming decisions and report them periodically."""

    def __init__(self, *, lane: str, interval: int) -> None:
        self.lane = lane
        self.interval = max(1, interval)
        self._steps = 0
        self._reqs = 0
        self._scheduled = 0
        self._admitted = 0
        self._verify_tokens = 0
        self._trimmed_steps = 0
        self._capacities: tuple[int, ...] | None = None

    def sampling(self) -> bool:
        """True when the step being recorded next will emit a line.

        Ask before any blocking copy. The upstream lane keeps its per-request
        capacities on device, and reading them every step would add a
        synchronization that lane does not otherwise need.
        """
        return (self._steps + 1) % self.interval == 0

    def record(
        self,
        *,
        num_reqs: int,
        scheduled_drafts: int,
        admitted_drafts: int,
        verify_tokens: int,
        capacities: Sequence[int] | None = None,
    ) -> None:
        self._steps += 1
        self._reqs += num_reqs
        self._scheduled += scheduled_drafts
        self._admitted += admitted_drafts
        self._verify_tokens += verify_tokens
        if admitted_drafts < scheduled_drafts:
            self._trimmed_steps += 1
        if capacities is not None:
            self._capacities = tuple(int(c) for c in capacities)
        if self._steps % self.interval:
            return

        steps = self._steps
        keep = 100.0 * self._admitted / self._scheduled if self._scheduled else 100.0
        logger.warning(
            "[DSPARK-EAGER-AV/%s] %d steps | mean reqs=%.2f scheduled_drafts=%.2f "
            "admitted=%.2f verify_tokens=%.2f | kept=%.1f%% | trimmed_steps=%d/%d | last_caps=%s",
            self.lane,
            steps,
            self._reqs / steps,
            self._scheduled / steps,
            self._admitted / steps,
            self._verify_tokens / steps,
            keep,
            self._trimmed_steps,
            steps,
            "n/a" if self._capacities is None else list(self._capacities),
        )
        self._steps = 0
        self._reqs = 0
        self._scheduled = 0
        self._admitted = 0
        self._verify_tokens = 0
        self._trimmed_steps = 0
