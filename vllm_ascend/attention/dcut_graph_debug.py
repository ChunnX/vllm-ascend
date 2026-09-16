"""One-line dumps of the D-Cut graph request axes, per capture and replay shape.

A full graph fixes every GDN input shape and address at capture, and the
linear-attention layers get no replay-time parameter update (unlike full
attention, which re-issues its kernel with refreshed host-side lengths), so a
capture and replay pair that disagree on any axis is silently wrong output
rather than an error.

Six different request counts are in play at once -- the live batch, the token
bucket, the graph descriptor's capacity, full attention's row count, GDN's
request axis, and the service maximum -- and deriving one from another is how
the D-Cut graph failure was misdiagnosed twice. So each component logs the axis
and the tensors it actually feeds its own operators, and the lines are compared
rather than reasoned about.

Off by default (``VLLM_ASCEND_DSPARK_DCUT_DEBUG_AXES``). Debug only: reads
device tensors back to the host, which is a synchronization point.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

import vllm_ascend.envs as envs_ascend

logger = init_logger(__name__)

# Printed values per tensor. Enough to read a request axis of 32 plus its
# padding tail without turning one step into pages of log.
_MAX_PRINTED_VALUES = 40

_emitted: dict[tuple, int] = {}
_disabled: set[str] = set()


def enabled(component: str = "") -> bool:
    return envs_ascend.VLLM_ASCEND_DSPARK_DCUT_DEBUG_AXES and component not in _disabled


@contextmanager
def guarded(component: str) -> Iterator[None]:
    """Keep the instrument from taking down the run it is meant to observe.

    The fields come from whatever the surrounding layer happens to expose, and
    reaching for the wrong one has twice aborted graph capture before its first
    shape -- a debug dump that is only switched on for an already-failing run
    has no business doing that. Report the failure with its traceback once and
    take this component out of service for the rest of the process; the other
    components keep logging.
    """
    try:
        yield
    except Exception:
        if component not in _disabled:
            _disabled.add(component)
            logger.warning(
                "[D-Cut AXES] %s dump failed and is now disabled for this process; the run continues without it",
                component,
                exc_info=True,
            )


def describe(tensor: torch.Tensor | None, *, values: bool = True) -> str:
    """Shape, address, and optionally contents of one graph input.

    The address is the point of most of these lines: a tensor that is freshly
    allocated per step cannot be an input to a captured graph, however correct
    its contents look.
    """
    if tensor is None:
        return "none"
    body = f"shape={tuple(tensor.shape)} ptr={tensor.data_ptr():#x}"
    if not values:
        return body
    flat = tensor.flatten()
    if flat.numel() > _MAX_PRINTED_VALUES:
        head = flat[:_MAX_PRINTED_VALUES].tolist()
        return f"{body} val={head}...(+{flat.numel() - _MAX_PRINTED_VALUES})"
    return f"{body} val={flat.tolist()}"


def log_axes(
    component: str,
    phase: str,
    shape_key: Any,
    *,
    repeats: int = 1,
    **fields: Any,
) -> None:
    """Emit one line for this component, phase, and graph shape.

    ``shape_key`` identifies the graph shape, so a run yields one comparable
    line per shape per phase instead of one per step.

    ``repeats`` raises that allowance for a component whose phase label cannot
    be trusted to separate capture from replay. Nothing below the model runner
    has a reliable flag: metadata is built before the forward context exists,
    and the capture path does not always announce itself (a piecewise
    ``prepare_inputs_to_capture`` arrives with the capture flag clear and no
    graph mode). With an allowance above one, each line carries its occurrence
    index, so warmup, capture and replay are told apart by arrival order.
    """
    key = (component, phase, shape_key)
    already = _emitted.get(key, 0)
    if already >= repeats:
        return
    _emitted[key] = already + 1

    parts = [] if repeats == 1 else [f"n={already}"]
    parts.extend(f"{name}={value}" for name, value in fields.items())
    logger.warning("[D-Cut AXES] %s phase=%s %s", component, phase, " | ".join(parts))


def reset() -> None:
    """Forget what has been logged and re-enable every component."""
    _emitted.clear()
    _disabled.clear()
