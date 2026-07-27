#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the G0.5 staged fault-boundary probe.

The property that matters most is the default: with the environment variable
unset this must not synchronize anywhere, because every eager path already in
production runs through these same five call sites.
"""

from unittest import mock

import pytest

from vllm_ascend._310p import debug_sync
from vllm_ascend._310p.debug_sync import (
    SYNC_STAGE_ENV,
    SYNC_STAGES,
    sync_stage,
    validate_sync_stage_env,
)


class _RecordingStream:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


def _patched(env_value, capturing=False, context_available=True):
    """Patch the environment, the stream and the forward-context probes.

    `torch` is replaced wholesale rather than patching `torch.npu.current_stream`
    so the test does not require torch_npu on the host.
    """
    stream = _RecordingStream()
    env = {} if env_value is None else {SYNC_STAGE_ENV: env_value}
    ctx = mock.MagicMock()
    ctx.capturing = capturing
    fake_torch = mock.MagicMock()
    fake_torch.npu.current_stream.return_value = stream
    return stream, [
        mock.patch.dict(debug_sync.os.environ, env, clear=True),
        mock.patch.object(debug_sync, "torch", fake_torch),
        mock.patch.object(debug_sync, "is_forward_context_available", return_value=context_available),
        mock.patch.object(debug_sync, "_EXTRA_CTX", ctx),
    ]


def _run(stage, env_value, **kwargs):
    stream, patches = _patched(env_value, **kwargs)
    for patch in patches:
        patch.start()
    try:
        sync_stage(stage)
    finally:
        for patch in reversed(patches):
            patch.stop()
    return stream.synchronize_calls


@pytest.mark.parametrize("stage", SYNC_STAGES)
def test_unset_env_never_synchronizes(stage):
    """The production default. Every eager run goes through these call sites."""
    assert _run(stage, None) == 0


@pytest.mark.parametrize("stage", SYNC_STAGES)
def test_empty_env_never_synchronizes(stage):
    # An exported-but-empty variable is a common way to "turn it off".
    assert _run(stage, "") == 0


@pytest.mark.parametrize("armed", SYNC_STAGES)
def test_exactly_one_boundary_is_armed(armed):
    """Arming one stage must not synchronize at any of the other four."""
    for stage in SYNC_STAGES:
        expected = 1 if stage == armed else 0
        assert _run(stage, armed) == expected, f"armed={armed} fired at {stage}"


def test_capture_is_never_synchronized():
    """_model_forward also runs under ACLGraph capture, where sync is illegal."""
    assert _run("post_target_replay", "post_target_replay", capturing=True) == 0


def test_missing_forward_context_does_not_block_the_sync():
    # Reading _EXTRA_CTX without a context raises, so availability is checked
    # first -- but "no context" is not "capturing" and must still synchronize.
    assert _run("post_target_replay", "post_target_replay", context_available=False) == 1


def test_unknown_stage_is_rejected():
    """A typo must fail loudly; arming nothing would let a run prove nothing."""
    stream, patches = _patched("post_targt_replay")
    for patch in patches:
        patch.start()
    try:
        with pytest.raises(RuntimeError, match="is not a known stage"):
            validate_sync_stage_env()
        # And the boundary itself refuses too, in case validation is bypassed.
        with pytest.raises(RuntimeError, match="is not a known stage"):
            sync_stage("post_target_replay")
    finally:
        for patch in reversed(patches):
            patch.stop()
    assert stream.synchronize_calls == 0


def test_validate_accepts_unset_and_each_known_stage():
    for env_value in (None, *SYNC_STAGES):
        _, patches = _patched(env_value)
        for patch in patches:
            patch.start()
        try:
            validate_sync_stage_env()
        finally:
            for patch in reversed(patches):
                patch.stop()
