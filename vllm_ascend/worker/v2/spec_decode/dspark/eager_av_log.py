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
        self._graph_modes: dict[str, int] = {}
        self._conf_moved: int | None = None
        self._conf_steps = 0
        self._emitted = False

    def sampling(self) -> bool:
        """True when the step being recorded next will emit a line.

        Ask before any blocking copy. The upstream lane keeps its per-request
        capacities on device, and reading them every step would add a
        synchronization that lane does not otherwise need.
        """
        return not self._emitted or (self._steps + 1) % self.interval == 0

    def note_confidence_steps(self, *, moved: int, steps: int) -> None:
        """Record how often the confidence signal actually changed.

        Under graph the confidence op is only recomputed per replay if it was
        traced into the captured draft graph. A Python ``if`` around it that was
        False at capture leaves it out, and then every replay skips it and the
        buffer stays frozen at its pre-capture value.

        No output comparison can catch that. Trimming is only a policy, so the
        rejection sampler stays correct and the tokens are byte-for-byte what a
        fixed-K run produces -- while every budget is decided on numbers that
        stopped moving. The only visible symptom would be "adaptive verification
        does not help", which hides for weeks.

        The caller counts on device so no step pays a copy for this, and hands
        over the totals when the window closes.
        """
        self._conf_moved = moved
        self._conf_steps = steps

    def note_graph_mode(self, cg_mode) -> None:
        """Record which cudagraph mode this step was dispatched under.

        Matching the baseline says the lane is correct; it does not say a graph
        ran. A trimmed batch matches no uniform descriptor and falls back, which
        is by design, so without this the aggregated line cannot tell "replayed
        a graph" from "captured one and never entered it" -- and at a few dozen
        decode steps neither can the wall clock.
        """
        name = getattr(cg_mode, "name", None) or str(cg_mode)
        self._graph_modes[name] = self._graph_modes.get(name, 0) + 1

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
        if self._conf_moved is not None:
            # n/N, not a boolean: zero over several steps is the frozen buffer,
            # and roughly N is an op being recomputed every step.
            suffix += f" | conf_moved={self._conf_moved}/{self._conf_steps}"
        if self._graph_modes:
            modes = ",".join(f"{name}={count}" for name, count in sorted(self._graph_modes.items()))
            suffix += f" | graph={modes}"
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
        self._graph_modes = {}
        self._conf_moved = None
        self._conf_steps = 0
