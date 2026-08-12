#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Name the mamba align kernel a triton-ascend compile crash dies in.

An MLIR assertion calls ``abort()``: no Python traceback, no exception, and
whatever sits in Python's stdout/stderr buffers is discarded. The worker log
therefore ends at the assert line with no indication of which kernel was being
compiled. The only record that survives is a write issued to fd 2 *before* the
launch, so wrap each align-mode kernel at its own binding site and announce it.

Reading the output: Triton compiles inside the launch call, so a crash during
compilation leaves a ``->`` line with no matching ``<-``. The last ``->`` names
the kernel. No ``->`` line at all before the abort means none of these three ran
-- the crash is elsewhere, and the align path is exonerated.

Off unless ``VLLM_ASCEND_TRACE_MAMBA_KERNELS=1``.
"""

import os

_ENABLED = os.getenv("VLLM_ASCEND_TRACE_MAMBA_KERNELS", "0") == "1"
_installed = False


def _emit(message: str) -> None:
    # os.write, not print or logger: abort() discards Python-level buffers, and
    # this has to survive precisely the crash it is meant to locate.
    os.write(2, f"[mamba-trace pid={os.getpid()}] {message}\n".encode())


class _TracedKernel:
    """Announces every launch of a Triton JITFunction, then delegates to it."""

    def __init__(self, name: str, kernel) -> None:
        self._name = name
        self._kernel = kernel

    def __getitem__(self, grid):
        launcher = self._kernel[grid]

        def run(*args, **kwargs):
            _emit(f"-> {self._name} grid={grid} {kwargs}")
            result = launcher(*args, **kwargs)
            _emit(f"<- {self._name}")
            return result

        return run

    def __getattr__(self, item):
        return getattr(self._kernel, item)


def install(model_state) -> None:
    """Wrap the three kernels the MRV2 align path launches. Idempotent."""
    global _installed
    if not _ENABLED or _installed:
        return
    _installed = True

    import vllm.v1.worker.gpu.model_states.mamba_hybrid as upstream_state
    from vllm.v1.worker import mamba_utils

    cache_config = model_state.cache_config
    # The geodesic question: every kernel below is behind `_align_mode`, which is
    # only true for mamba_cache_mode "align". If this prints something else, the
    # align path is not running at all and nothing below will ever fire.
    _emit(
        f"mamba_cache_mode={cache_config.mamba_cache_mode!r} "
        f"align_mode={getattr(model_state, '_align_mode', None)} "
        f"enable_prefix_caching={cache_config.enable_prefix_caching}"
    )

    # preprocess_mamba_align_fused_kernel is bound by name into the model-state
    # module at import time, so patching it on mamba_utils would not reach the
    # call site. The other two are looked up as mamba_utils globals at call time.
    targets = (
        (upstream_state, "preprocess_mamba_align_fused_kernel"),
        (mamba_utils, "precopy_mamba_align_fused_kernel"),
        (mamba_utils, "postprocess_mamba_fused_kernel"),
    )
    for module, attr in targets:
        kernel = getattr(module, attr, None)
        if kernel is None:
            _emit(f"!! {module.__name__}.{attr} not found -- binding site moved")
            continue
        setattr(module, attr, _TracedKernel(attr, kernel))
        _emit(f"traced {module.__name__}.{attr} ({type(kernel).__module__}.{type(kernel).__name__})")
