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
"""Let a row-local custom logits processor run under speculative decoding.

Upstream closes this off in two places. ``build_logitsprocs`` raises as soon as any custom
processor is combined with a speculative config, and ``RejectionSampler`` reaches the draft
positions through an ``isinstance(processor, MinTokensLogitsProcessor)`` test that skips
everything else in silence. The reasoning is sound for the processors it was written for:
under speculative decoding the logits tensor holds one row per draft position and several
rows belong to the same request, so ``min_p`` and ``logit_bias``, which have to know which
request a row came from, really are wrong there.

A rule that is row-local and stateless has no such problem, and this narrows the two gates
to let exactly those through -- a processor class opting in with ``supports_spec_decode =
True`` and implementing ``apply_with_spec_decode``. Anything that does not opt in still
raises, so the failure for a processor that genuinely cannot work stays loud.

Note that only the draft positions need this. The bonus token goes through the ordinary
``Sampler``, whose ``apply_logits_processors`` calls every non-argmax-invariant processor
already, so that half works untouched.
"""

import sys

import vllm.v1.sample.logits_processor as lp
import vllm.v1.sample.rejection_sampler as rs
from vllm.logger import logger

# Where model_runner_v1 keeps its own reference to build_logitsprocs, see below.
_MODEL_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"

_original_build_logitsprocs = lp.build_logitsprocs
_original_apply_logits_processors = rs.RejectionSampler.apply_logits_processors


def _supports_spec_decode(processor_or_class) -> bool:
    cls = processor_or_class if isinstance(processor_or_class, type) else type(processor_or_class)
    return bool(getattr(cls, "supports_spec_decode", False))


def build_logitsprocs(vllm_config, device, is_pin_memory, is_pooling_model, custom_logitsprocs=()):
    """Keep upstream's behaviour except for opted-in processors under speculative decoding."""
    if is_pooling_model or not vllm_config.speculative_config or not custom_logitsprocs:
        return _original_build_logitsprocs(
            vllm_config, device, is_pin_memory, is_pooling_model, custom_logitsprocs
        )

    classes = lp._load_custom_logitsprocs(custom_logitsprocs)
    rejected = [cls.__name__ for cls in classes if not _supports_spec_decode(cls)]
    if rejected:
        raise ValueError(
            f"{lp.STR_SPEC_DEC_REJECTS_LOGITSPROCS} Rejected: {rejected}. A processor whose "
            "rule is row-local and stateless may opt in with supports_spec_decode = True "
            "and an apply_with_spec_decode method."
        )

    logger.warning("min_p and logit_bias parameters won't work with speculative decoding.")
    logger.info(
        "Keeping spec-decode-safe custom logits processors under speculative decoding: %s",
        [cls.__name__ for cls in classes],
    )
    # Same shape as upstream's speculative branch -- min_tokens plus, here, the opted-in
    # processors. The other builtins stay out on purpose; upstream drops them because they
    # are broken under speculative decoding, not merely untested.
    return lp.LogitsProcessors(
        [lp.MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
        + [cls(vllm_config, device, is_pin_memory) for cls in classes]
    )


def apply_logits_processors(self, logits, sampling_metadata, metadata):
    """Run upstream's pass first, then the opted-in processors it skipped.

    Wrapping rather than reimplementing keeps this working when the upstream body changes.
    Upstream's loop only ever calls MinTokensLogitsProcessor, so nothing is applied twice.
    """
    logits = _original_apply_logits_processors(self, logits, sampling_metadata, metadata)
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        if _supports_spec_decode(processor):
            logits = processor.apply_with_spec_decode(logits, metadata.num_draft_tokens)
    return logits


lp.build_logitsprocs = build_logitsprocs
rs.RejectionSampler.apply_logits_processors = apply_logits_processors

# model_runner_v1 does `from vllm.v1.sample.logits_processor import build_logitsprocs` at
# import time, and only pulls this patch package in further down its own import block, so
# by now it holds the original function and rebinding the module attribute above misses it.
# Reach into the half-initialised module rather than importing it, which would be a cycle.
_model_runner = sys.modules.get(_MODEL_RUNNER_MODULE)
if _model_runner is not None and hasattr(_model_runner, "build_logitsprocs"):
    _model_runner.build_logitsprocs = build_logitsprocs
else:
    logger.warning(
        "%s was not importing when patch_logits_processors ran, so its build_logitsprocs "
        "reference is unpatched and a custom logits processor will still be rejected under "
        "speculative decoding.",
        _MODEL_RUNNER_MODULE,
    )
