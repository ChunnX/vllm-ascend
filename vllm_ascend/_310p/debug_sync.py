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
# This file is a part of the vllm-ascend project.
#
"""Test-only staged fault-boundary probe for the 310P ACLGraph investigation.

An NPU fault surfaces at the next implicit synchronization point, not where it
was raised: the 07-25 aicore fault reports a Python stack stopped in
``aclnnNonzeroV2`` purely because that is the first sync after the speculative
step. Placing one explicit synchronize at a known boundary and moving it makes
the first failing boundary the answer.

Exactly one boundary is armed per run, by name. That is the point rather than a
limitation -- opening several at once changes the very timing the probe is
trying to localize, so the environment variable holds a single stage and any
other value arms nothing.

Off by default: with ``VLLM_ASCEND_310P_SYNC_STAGE`` unset this costs one
dict lookup per boundary and changes no behaviour.
"""

import os

import torch
from vllm.forward_context import is_forward_context_available
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

SYNC_STAGE_ENV = "VLLM_ASCEND_310P_SYNC_STAGE"

# Boundaries in execution order within one speculative step. Read as: "the
# fault is at or before the first stage that fails."
SYNC_STAGES = (
    "post_target_replay",
    "post_logits",
    "post_sampler",
    "post_expansion",
    "post_draft",
)


def _selected_stage() -> str | None:
    stage = os.environ.get(SYNC_STAGE_ENV)
    if stage is None or stage == "":
        return None
    if stage not in SYNC_STAGES:
        # A typo must not silently arm nothing: the run would complete and
        # "prove" a boundary that was never instrumented.
        raise RuntimeError(
            f"{SYNC_STAGE_ENV}={stage!r} is not a known stage. "
            f"Choose exactly one of {list(SYNC_STAGES)}, or unset it."
        )
    return stage


def sync_stage(stage: str) -> None:
    """Synchronize the current stream iff ``stage`` is the armed boundary."""
    if _selected_stage() != stage:
        return

    # Synchronizing inside a capture stream is illegal, and the boundaries below
    # are reached during ACLGraph capture too (the dummy run drives the same
    # _model_forward). Short-circuit on availability first: reading _EXTRA_CTX
    # without an active forward context raises.
    if is_forward_context_available() and _EXTRA_CTX.capturing:
        return

    logger.warning_once(
        "310P debug: %s=%s is armed; synchronizing at this boundary. "
        "This changes timing and must never be set in a normal run.",
        SYNC_STAGE_ENV,
        stage,
    )
    torch.npu.current_stream().synchronize()


def validate_sync_stage_env() -> None:
    """Fail at startup on a mistyped stage rather than at the first boundary."""
    _selected_stage()
