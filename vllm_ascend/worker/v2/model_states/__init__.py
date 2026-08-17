# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/__init__.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
import torch.nn as nn
from vllm.config import VllmConfig
# from vllm.logger import logger  # Temporary DIAG logging; keep for re-enable.
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_asecnd_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    # Keep model-provided state overrides ahead of the platform defaults.
    if hasattr(model, "get_model_state_cls"):
        cls = model.get_model_state_cls()
        # logger.info_once(
        #     "[DIAG-MODEL-STATE-SELECT] model=%s.%s custom_override=true "
        #     "is_hybrid=%s selected=%s.%s",
        #     type(model).__module__,
        #     type(model).__qualname__,
        #     vllm_config.model_config.is_hybrid,
        #     cls.__module__,
        #     cls.__qualname__,
        # )
        return cls(vllm_config, model, encoder_cache, device)

    if vllm_config.model_config.is_hybrid:
        from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
            AscendMambaHybridModelState,
        )

        # logger.info_once(
        #     "[DIAG-MODEL-STATE-SELECT] model=%s.%s custom_override=false "
        #     "is_hybrid=true selected=%s.%s",
        #     type(model).__module__,
        #     type(model).__qualname__,
        #     AscendMambaHybridModelState.__module__,
        #     AscendMambaHybridModelState.__qualname__,
        # )
        return AscendMambaHybridModelState(
            vllm_config,
            model,
            encoder_cache,
            device,
        )

    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    # logger.info_once(
    #     "[DIAG-MODEL-STATE-SELECT] model=%s.%s custom_override=false "
    #     "is_hybrid=false selected=%s.%s",
    #     type(model).__module__,
    #     type(model).__qualname__,
    #     AscendModelState.__module__,
    #     AscendModelState.__qualname__,
    # )
    return AscendModelState(vllm_config, model, encoder_cache, device)
