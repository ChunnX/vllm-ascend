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
"""Record every Triton kernel actually launched in this process.

Static analysis cannot answer which Triton operators a given model + feature
combination executes: import reachability over-approximates badly (almost every
kernel in the tree is transitively importable), and the launch sites of kernels
that vllm-ascend patches into upstream modules are invisible to a grep of this
repo. One run with this enabled answers it exactly.

Hooks ``JITFunction.run``, which every ``kernel[grid](...)`` launch funnels
through, and reports each distinct kernel the first time it fires. Reporting
goes to fd 2 unbuffered, so the record survives a hard abort() -- the failure
mode triton-ascend compile errors take.

Off unless ``VLLM_ASCEND_TRACE_TRITON_OPS=1``. Collect with:

    grep '\[triton-op\]' <log> | sed 's/.*\[triton-op\] //' | sort -u
"""

import os

_ENABLED = os.getenv("VLLM_ASCEND_TRACE_TRITON_OPS", "0") == "1"


def _install() -> None:
    from triton.runtime.jit import JITFunction

    seen: set[str] = set()
    original_run = JITFunction.run

    def traced_run(self, *args, **kwargs):
        # Never let tracing break a run: identification is best-effort.
        try:
            fn = getattr(self, "fn", None)
            name = getattr(self, "__name__", None) or getattr(fn, "__name__", "?")
            key = f"{getattr(fn, '__module__', '?')}.{name}"
            if key not in seen:
                seen.add(key)
                # os.write, not logger: a triton-ascend compile failure aborts
                # the process, and this line has to be on the wire by then.
                os.write(2, f"[triton-op] {key}\n".encode())
        except Exception:
            pass
        return original_run(self, *args, **kwargs)

    JITFunction.run = traced_run
    os.write(2, b"[triton-op] tracing installed\n")


if _ENABLED:
    try:
        _install()
    except Exception as exc:  # pragma: no cover - diagnostic only
        os.write(2, f"[triton-op] install failed: {exc!r}\n".encode())
