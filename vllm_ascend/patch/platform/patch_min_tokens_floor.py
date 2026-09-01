#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Give every request a floor on ``min_tokens``, so a turn cannot be empty.

Four turns in a 2812-run sweep ended on the very first step: no text, no tool call, just
the stop token at a probability of essentially 1. ``min_tokens`` already describes the
remedy exactly -- it masks every id in ``all_stop_token_ids`` until that many tokens have
been produced -- but it only ever arrives on the request, and unlike temperature or top_p
it is absent from ``_DEFAULT_SAMPLING_PARAMS``, so ``generation_config.json`` cannot set it
and each caller would have to remember to. This applies a floor server-side instead.

Two things make ``min_tokens`` the right mechanism rather than another rule inside
``EosArgmaxOnly``. It survives speculative decoding: ``MinTokensLogitsProcessor`` is the one
processor upstream keeps in the spec-decode branch, and it implements
``apply_with_spec_decode`` with the row-to-request mapping that path needs. And it survives
async scheduling, because it reads only ``len(output_tok_ids)`` -- the -1 placeholders held
there are wrong in value but right in length. Teaching EosArgmaxOnly to count tokens would
cost it the statelessness that lets it run under speculative decoding at all.

``update_from_generation_config`` is the single funnel every generate request passes
through (``vllm/v1/engine/input_processor.py:266``), whichever API it came in on. The floor
is applied after the original call, which is what populates ``_all_stop_token_ids`` -- the
set MinTokensLogitsProcessor masks.

Set ``VLLM_ASCEND_MIN_TOKENS_FLOOR`` to raise the floor, or to 0 to restore stock
behaviour. A request asking for more than the floor keeps its own value.
"""

import os

from vllm.logger import logger
from vllm.sampling_params import SamplingParams

MIN_TOKENS_FLOOR_ENV = "VLLM_ASCEND_MIN_TOKENS_FLOOR"
DEFAULT_MIN_TOKENS_FLOOR = 1


def _read_floor() -> int:
    raw = os.getenv(MIN_TOKENS_FLOOR_ENV)
    if raw is None:
        return DEFAULT_MIN_TOKENS_FLOOR
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "%s=%r is not an integer, falling back to %d", MIN_TOKENS_FLOOR_ENV, raw, DEFAULT_MIN_TOKENS_FLOOR
        )
        return DEFAULT_MIN_TOKENS_FLOOR


MIN_TOKENS_FLOOR = _read_floor()

_original_update_from_generation_config = SamplingParams.update_from_generation_config


def update_from_generation_config(self, generation_config, eos_token_id=None) -> None:
    _original_update_from_generation_config(self, generation_config, eos_token_id)
    if MIN_TOKENS_FLOOR and self.min_tokens < MIN_TOKENS_FLOOR:
        self.min_tokens = MIN_TOKENS_FLOOR


if MIN_TOKENS_FLOOR:
    SamplingParams.update_from_generation_config = update_from_generation_config
    logger.info(
        "patch_min_tokens_floor active: every request gets min_tokens >= %d, so a turn "
        "cannot end on its first token. Set %s=0 to disable.",
        MIN_TOKENS_FLOOR,
        MIN_TOKENS_FLOOR_ENV,
    )
else:
    logger.info("patch_min_tokens_floor disabled via %s=0.", MIN_TOKENS_FLOOR_ENV)
