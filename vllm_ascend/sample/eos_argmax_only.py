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
"""A logits processor that only lets the model stop when stopping is the best option.

Sampling picks the winner of an exponential race, ``argmax(probs / q)`` with
``q ~ Exp(1)`` (see ``vllm_ascend.sample.sampler.random_sample``). That draws exactly in
proportion to the probabilities, so a low-probability end-of-sequence token is still
picked at its own rate rather than never. Measured on a 1500-run sweep of one agent
trajectory, 50 of 1428 turns ended on an eos token that was not the argmax -- one of them
at 2.45% against a 43.48% alternative -- and every one of those turns was a truncated
answer or a half-emitted tool call.

This processor masks the eos tokens whenever some other token has a higher logit, which
restores greedy semantics for eos alone and leaves every other token sampled as before.
On the same sweep it would have blocked all 50 bad endings and none of the 1378 good ones,
because a turn that genuinely ends has eos as the argmax by a wide margin.

Prefer this over ``logit_bias`` on the eos ids. A fixed bias is capped by how much margin
a legitimate ending has (~6.4 nat on the measured trajectory); push past that and normal
endings break too. This rule is adaptive and has no such ceiling.

Enable it by passing the fully qualified class name to vLLM::

    --logits-processors vllm_ascend.sample.eos_argmax_only:EosArgmaxOnly

Note that vLLM rejects custom logits processors when speculative decoding is enabled.
"""

import os

import torch
from vllm.logger import logger
from vllm.v1.sample.logits_processor import LogitsProcessor

# Comma separated token ids, overriding whatever is discovered from the model config.
EOS_TOKEN_IDS_ENV = "VLLM_ASCEND_EOS_ARGMAX_ONLY_IDS"


def _as_id_list(value) -> list[int]:
    """generation_config stores eos_token_id as either a single id or a list of them."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(token_id) for token_id in value]


def _resolve_eos_token_ids(vllm_config) -> list[int]:
    """Collect every token id that ends a turn, most explicit source first."""
    override = os.getenv(EOS_TOKEN_IDS_ENV)
    if override:
        return sorted({int(part) for part in override.replace(" ", "").split(",") if part})

    model_config = vllm_config.model_config
    token_ids: set[int] = set()
    try:
        token_ids.update(_as_id_list(model_config.try_get_generation_config().get("eos_token_id")))
    except Exception:  # a missing or malformed generation_config.json must not stop the server
        logger.exception("EosArgmaxOnly: could not read generation_config, falling back to hf_config")
    token_ids.update(_as_id_list(getattr(model_config.hf_config, "eos_token_id", None)))
    return sorted(token_ids)


class EosArgmaxOnly(LogitsProcessor):
    """Suppress an end-of-sequence token unless it is the highest-scoring token.

    Stateless: it reads only the logits of the current step and never touches the
    generated token ids. That matters on this platform, because under async scheduling
    ``output_token_ids`` holds -1 placeholders rather than real ids, which silently
    disables everything that does depend on them (``bad_words``, the penalties).
    """

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        eos_token_ids = _resolve_eos_token_ids(vllm_config)
        self.eos_token_ids = torch.tensor(eos_token_ids, device=device, dtype=torch.long)
        if eos_token_ids:
            logger.info("EosArgmaxOnly enabled for eos token ids %s", eos_token_ids)
        else:
            logger.warning(
                "EosArgmaxOnly found no eos token ids and will do nothing. Set %s to enable it.",
                EOS_TOKEN_IDS_ENV,
            )

    def is_argmax_invariant(self) -> bool:
        """False: it changes which token random sampling lands on."""
        return False

    def update_state(self, batch_update) -> None:
        """No per-request state to track."""

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.eos_token_ids.numel():
            return logits
        eos_logits = logits[:, self.eos_token_ids]
        # Comparing against the global max rather than the max over non-eos tokens gives the
        # same answer -- an eos below the global max is beaten by something -- and saves
        # materialising a second [batch, vocab] tensor on every step.
        row_max = logits.max(dim=-1, keepdim=True).values
        logits[:, self.eos_token_ids] = torch.where(
            eos_logits < row_max,
            torch.full_like(eos_logits, float("-inf")),
            eos_logits,
        )
        return logits
