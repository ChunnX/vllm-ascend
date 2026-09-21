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
        self._untrusted = 0
        self._out_of_range = 0
        self._emitted = False

    def sampling(self) -> bool:
        """True when the step being recorded next will emit a line.

        Ask before any blocking copy. The upstream lane keeps its per-request
        capacities on device, and reading them every step would add a
        synchronization that lane does not otherwise need.
        """
        return not self._emitted or (self._steps + 1) % self.interval == 0

    def record(
        self,
        *,
        num_reqs: int,
        scheduled_drafts: int,
        admitted_drafts: int,
        verify_tokens: int,
        capacities: Sequence[int] | None = None,
        untrusted_rows: int = 0,
        out_of_range_rows: int = 0,
    ) -> None:
        self._steps += 1
        self._reqs += num_reqs
        self._scheduled += scheduled_drafts
        self._admitted += admitted_drafts
        self._verify_tokens += verify_tokens
        if admitted_drafts < scheduled_drafts:
            self._trimmed_steps += 1
        self._untrusted += untrusted_rows
        self._out_of_range += out_of_range_rows
        if capacities is not None:
            self._capacities = tuple(int(c) for c in capacities)
        # Always emit for the very first recorded step. A run shorter than one
        # interval would otherwise finish having printed only the construction
        # banner, and the banner proves the manager exists -- not that it ever
        # trimmed anything. Without a data line, equal output between a lane and
        # the baseline is not evidence about the trimmed path.
        if self._emitted and self._steps % self.interval:
            return
        self._emitted = True

        steps = self._steps
        keep = 100.0 * self._admitted / self._scheduled if self._scheduled else 100.0
        # Untrusted rows are expected during prefill bursts and only mean those
        # requests kept their drafts. An out_of_range count is different: finite
        # but not a probability is not the prefill artifact, so name it apart.
        suffix = ""
        if self._untrusted:
            suffix = f" | untrusted_rows={self._untrusted}"
            if self._out_of_range:
                suffix += f" (out_of_range={self._out_of_range})"
        logger.warning(
            "[DSPARK-EAGER-AV/%s] %d steps | mean reqs=%.2f scheduled_drafts=%.2f "
            "admitted=%.2f verify_tokens=%.2f | kept=%.1f%% | trimmed_steps=%d/%d | last_caps=%s%s",
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
            suffix,
        )
        self._steps = 0
        self._reqs = 0
        self._scheduled = 0
        self._admitted = 0
        self._verify_tokens = 0
        self._trimmed_steps = 0
        self._untrusted = 0
        self._out_of_range = 0
