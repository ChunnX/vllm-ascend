# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.attention import dcut_graph_debug


class _RecordingLogger:
    """Collects formatted lines, so the test does not depend on log routing."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.lines.append(message % args)


@pytest.fixture
def lines(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorder = _RecordingLogger()
    monkeypatch.setattr(dcut_graph_debug, "logger", recorder)
    dcut_graph_debug.reset()
    return recorder.lines


def test_axes_lines_are_one_per_phase_and_shape(lines: list[str]) -> None:
    """One comparable line per graph shape, not one per decode step.

    The point of the dump is to put a capture line beside a replay line for the
    same shape; a per-step log would bury that in a live run.
    """
    for _ in range(3):
        dcut_graph_debug.log_axes("gdn", "capture", (8, 8), b_gdn=8)
        dcut_graph_debug.log_axes("gdn", "replay", (8, 8), b_gdn=8)
    # A different shape is a different line, and so is another component.
    dcut_graph_debug.log_axes("gdn", "replay", (32, 256), b_gdn=32)
    dcut_graph_debug.log_axes("fia", "replay", (8, 8), b_fia=8)

    assert len(lines) == 4
    assert sum("phase=capture" in line for line in lines) == 1
    assert sum(line.startswith("[D-Cut AXES] gdn ") for line in lines) == 3
    assert "b_gdn=32" in lines[2]


def test_repeats_keep_later_lines_and_number_them(lines: list[str]) -> None:
    """A component whose phase label cannot be trusted still gets both lines.

    Nothing below the model runner has a reliable capture flag, so arrival
    order is the discriminator and each line has to say where it sits in that
    order.
    """
    for index in range(5):
        dcut_graph_debug.log_axes("fia", "build", (8, 8), repeats=3, call=index)

    assert len(lines) == 3
    assert "n=0 | call=0" in lines[0]
    assert "n=1 | call=1" in lines[1]
    assert "n=2 | call=2" in lines[2]


def test_single_line_shapes_carry_no_occurrence_counter(lines: list[str]) -> None:
    dcut_graph_debug.log_axes("gdn", "capture", (8, 8), b_gdn=8)

    assert lines == ["[D-Cut AXES] gdn phase=capture b_gdn=8"]


def test_reset_lets_a_later_run_log_its_shapes_again(lines: list[str]) -> None:
    dcut_graph_debug.log_axes("gdn", "replay", (8, 8), b_gdn=8)
    dcut_graph_debug.log_axes("gdn", "replay", (8, 8), b_gdn=8)
    assert len(lines) == 1

    dcut_graph_debug.reset()
    dcut_graph_debug.log_axes("gdn", "replay", (8, 8), b_gdn=8)
    assert len(lines) == 2


def test_multiple_fields_stay_readable_on_one_line(lines: list[str]) -> None:
    dcut_graph_debug.log_axes("mamba-hybrid", "replay", (8, 8), b_live=1, b_graph=8, q=8)

    assert lines == ["[D-Cut AXES] mamba-hybrid phase=replay b_live=1 | b_graph=8 | q=8"]


def test_a_broken_dump_disables_itself_instead_of_failing_the_run(lines: list[str]) -> None:
    """The instrument must not take down the run it is switched on to observe.

    The fields come from whatever the surrounding layer exposes, and reaching
    for the wrong one has already aborted graph capture twice.
    """
    with dcut_graph_debug.guarded("gdn"):
        raise AttributeError("no such attribute")

    assert "gdn" in dcut_graph_debug._disabled
    assert len(lines) == 1
    assert "disabled" in lines[0]

    # A second failure of the same component stays quiet.
    with dcut_graph_debug.guarded("gdn"):
        raise AttributeError("again")
    assert len(lines) == 1

    # Other components keep logging.
    dcut_graph_debug.log_axes("fia", "build", (8, 8), b_fia=8)
    assert len(lines) == 2

    dcut_graph_debug.reset()
    assert "gdn" not in dcut_graph_debug._disabled


def test_describe_reports_address_and_truncates_long_values() -> None:
    """The address matters as much as the contents.

    A graph input that is freshly allocated per step cannot be an input to a
    captured graph however right its values look, so every description carries
    the pointer.
    """
    assert dcut_graph_debug.describe(None) == "none"

    tensor = torch.arange(4, dtype=torch.int32)
    described = dcut_graph_debug.describe(tensor)
    assert "shape=(4,)" in described
    assert f"ptr={tensor.data_ptr():#x}" in described
    assert "val=[0, 1, 2, 3]" in described

    assert "val=" not in dcut_graph_debug.describe(tensor, values=False)

    long_tensor = torch.arange(dcut_graph_debug._MAX_PRINTED_VALUES + 7, dtype=torch.int32)
    assert "...(+7)" in dcut_graph_debug.describe(long_tensor)
