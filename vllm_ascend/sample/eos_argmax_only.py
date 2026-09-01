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
"""A logits processor that only lets the model stop when stopping is clearly the best option.

Sampling picks the winner of an exponential race, ``argmax(probs / q)`` with ``q ~ Exp(1)``
(see ``vllm_ascend.sample.sampler.random_sample``). That draws exactly in proportion to the
probabilities, so a low-probability end-of-sequence token is still picked at its own rate.
Two rules follow from that, both stateless -- they read only the current step's logits and
never touch the generated token ids, which matters here because under async scheduling
``output_token_ids`` holds -1 placeholders and silently disables everything that does
depend on them (``bad_words``, the penalties).

Rule 1, always on: mask an eos token whenever another token scores higher. This restores
greedy semantics for eos alone and leaves the rest of the distribution sampled as before.

Rule 2, on once ``SOFT_STOP_RIVAL_TOKEN_IDS`` is set: mask an eos that *is* the argmax when
it leads the runner-up by less than ``SOFT_STOP_MARGIN`` and that runner-up is one of the
rival ids -- in practice the "\\n\\n" token. A turn about to emit a tool call writes
``:\\n\\n<tool_call>``, so the fork between stopping and continuing shows up as eos against
"\\n\\n" specifically. Measured over 1500 runs of one agent trajectory:

    runner-up at the eos step   good endings          bad endings
    "\\n"                        1107 / 1283           0 / 19
    "<tool_call>"                  97 / 1283           0 / 19
    "\\n\\n"                        38 / 1283 (3.0%)    15 / 19 (79%)

Gating on the rival is what makes rule 2 cheap. At a margin of 4 it blocks 15 of the 19 bad
endings for 1 of 1283 good ones (0.08%); the same margin without the gate blocks 19 but
costs 148 (11.5%), because a legitimate ending is usually followed by "\\n", not "\\n\\n".
A false positive here is mild as well -- the turn emits one more blank line and then ends,
rather than being forced to keep writing past a finished tool call.

The four remaining bad endings had runner-ups like " try" or " check" and are out of reach
of any rule of this shape; they need a retry at the request level.

Enable it by passing the fully qualified class name to vLLM::

    --logits-processors vllm_ascend.sample.eos_argmax_only:EosArgmaxOnly

The three variants worth comparing, each needing its own process since the environment is
read once at startup::

    (no --logits-processors flag)          both rules off
    VLLM_ASCEND_EOS_SOFT_STOP=0            rule 1 only
    (default)                              rules 1 and 2

Upstream vLLM rejects custom logits processors when speculative decoding is enabled.
``vllm_ascend.patch.worker.patch_logits_processors`` narrows that to processors which do
not declare ``supports_spec_decode``, so this one runs with MTP on.
"""

import os

import torch
from vllm.logger import logger
from vllm.v1.sample.logits_processor import LogitsProcessor

# Comma separated token ids, overriding whatever is discovered from the model config.
EOS_TOKEN_IDS_ENV = "VLLM_ASCEND_EOS_ARGMAX_ONLY_IDS"

# Runner-up token ids that make an eos win suspicious rather than decisive. Empty leaves
# rule 2 off.
#
# 271 is "\n\n" for the qwen3.6-27b tokenizer this branch targets. It is tokenizer
# specific, so re-derive it for any other model and check the id printed at startup:
#
#     tokenizer.encode("\n\n", add_special_tokens=False)   # -> [271]
#
# It has to come back as a single id, otherwise the rule cannot be expressed on logits.
SOFT_STOP_RIVAL_TOKEN_IDS: tuple[int, ...] = (271,)
SOFT_STOP_RIVAL_IDS_ENV = "VLLM_ASCEND_EOS_SOFT_STOP_RIVAL_IDS"

# Set to 0/false/off to run rule 1 alone without editing the source, which is how the two
# variants get compared. Rule 1 has no switch: drop --logits-processors for that baseline.
SOFT_STOP_ENABLED_ENV = "VLLM_ASCEND_EOS_SOFT_STOP"

# How far ahead of the rival an eos has to be, in nat, to be taken at face value.
SOFT_STOP_MARGIN = 4.0
SOFT_STOP_MARGIN_ENV = "VLLM_ASCEND_EOS_SOFT_STOP_MARGIN"


def _as_id_list(value) -> list[int]:
    """generation_config stores eos_token_id as either a single id or a list of them."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(token_id) for token_id in value]


def _ids_from_env(name: str) -> list[int]:
    raw = os.getenv(name)
    if not raw:
        return []
    return sorted({int(part) for part in raw.replace(" ", "").split(",") if part})


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "off", "no")


def _resolve_rival_token_ids() -> list[int]:
    """Which runner-up ids arm rule 2. Empty means rule 2 is off.

    An unset ids variable keeps the built-in default, while an explicitly empty one turns
    rule 2 off -- the `or` shorthand cannot tell those apart, and being unable to disable
    rule 2 from the environment is what makes an A/B impossible.
    """
    if not _env_flag(SOFT_STOP_ENABLED_ENV, True):
        return []
    raw = os.getenv(SOFT_STOP_RIVAL_IDS_ENV)
    if raw is None:
        return list(SOFT_STOP_RIVAL_TOKEN_IDS)
    return _ids_from_env(SOFT_STOP_RIVAL_IDS_ENV)


def _resolve_eos_token_ids(vllm_config) -> list[int]:
    """Collect every token id that ends a turn, most explicit source first."""
    override = _ids_from_env(EOS_TOKEN_IDS_ENV)
    if override:
        return override

    model_config = vllm_config.model_config
    token_ids: set[int] = set()
    try:
        token_ids.update(_as_id_list(model_config.try_get_generation_config().get("eos_token_id")))
    except Exception:  # a missing or malformed generation_config.json must not stop the server
        logger.exception("EosArgmaxOnly: could not read generation_config, falling back to hf_config")
    token_ids.update(_as_id_list(getattr(model_config.hf_config, "eos_token_id", None)))
    return sorted(token_ids)


class EosArgmaxOnly(LogitsProcessor):
    """Suppress an end-of-sequence token unless it wins clearly. See the module docstring."""

    # Read by vllm_ascend.patch.worker.patch_logits_processors, which lets a processor
    # carrying this flag through the speculative decoding path that upstream closes off.
    # Only claim it for a rule that is row-local and stateless, as this one is: under
    # speculative decoding the logits tensor holds one row per draft position, several of
    # them belonging to the same request, so anything needing a row-to-request mapping --
    # min_p and logit_bias, which is why upstream refuses them -- would be wrong here.
    supports_spec_decode = True

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        eos_token_ids = _resolve_eos_token_ids(vllm_config)
        rival_token_ids = _resolve_rival_token_ids()
        self.margin = float(os.getenv(SOFT_STOP_MARGIN_ENV, SOFT_STOP_MARGIN))

        self.eos_token_ids = torch.tensor(eos_token_ids, device=device, dtype=torch.long)
        self.rival_token_ids = torch.tensor(rival_token_ids, device=device, dtype=torch.long)

        if not eos_token_ids:
            logger.warning(
                "EosArgmaxOnly found no eos token ids and will do nothing. Set %s to enable it.",
                EOS_TOKEN_IDS_ENV,
            )
        elif rival_token_ids:
            logger.info(
                "EosArgmaxOnly enabled for eos token ids %s, also rejecting an eos that leads "
                "rival ids %s by less than %.2f nat",
                eos_token_ids,
                rival_token_ids,
                self.margin,
            )
        else:
            logger.info(
                "EosArgmaxOnly enabled for eos token ids %s, argmax rule only (rule 2 off via %s).",
                eos_token_ids,
                SOFT_STOP_ENABLED_ENV,
            )

    def is_argmax_invariant(self) -> bool:
        """False: it changes which token random sampling lands on."""
        return False

    def update_state(self, batch_update) -> None:
        """No per-request state to track."""

    def apply_with_spec_decode(self, logits: torch.Tensor, num_draft_tokens: list[int]) -> torch.Tensor:
        """Speculative decoding hook. See ``supports_spec_decode``.

        The rule is row-local, so the draft positions need no different treatment and
        ``num_draft_tokens`` -- the row-to-request mapping -- goes unused.
        """
        return self.apply(logits)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.eos_token_ids.numel():
            return logits

        # topk(2) gives both what rule 1 needs (the row max) and what rule 2 needs (which
        # token is the runner-up, and by how much it loses).
        values, indices = torch.topk(logits, 2, dim=-1)
        row_max = values[:, :1]
        eos_logits = logits[:, self.eos_token_ids]

        # Rule 1: an eos beaten by anything is not a real ending.
        suppress = eos_logits < row_max

        if self.rival_token_ids.numel():
            top1_is_eos = (indices[:, :1] == self.eos_token_ids.view(1, -1)).any(dim=-1)
            top2_is_rival = (indices[:, 1:2] == self.rival_token_ids.view(1, -1)).any(dim=-1)
            narrow = (values[:, 0] - values[:, 1]) < self.margin
            # Rule 2 fires on the eos that *is* the argmax, so pair the row-level condition
            # with the per-eos "this is the one on top" test.
            fires = (top1_is_eos & top2_is_rival & narrow).unsqueeze(1)
            suppress = suppress | (fires & (eos_logits >= row_max))

        logits[:, self.eos_token_ids] = torch.where(
            suppress,
            torch.full_like(eos_logits, float("-inf")),
            eos_logits,
        )
        return logits
