#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
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
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


def _strict_binary_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
    return value == "1"


env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Emit per-layer KVPool ranged transfer audit events. Default: 0 (disabled).
    # Valid values: 0 or 1. This configuration is not sensitive.
    "VLLM_ASCEND_KVPOOL_RANGE_DEBUG": lambda: _strict_binary_env("VLLM_ASCEND_KVPOOL_RANGE_DEBUG"),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
    # Enable npu_fused_infer_attention_sink for the parallel-drafting (DSpark /
    # DFlash) draft model's non-causal attention. The sink op takes device-side
    # seq_lens directly and does tiling on AICPU, removing the seq_lens.tolist()
    # host sync in the draft hot path. Disabled by default.
    "VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK", "0"))
    ),
    # Record-only observation of DSpark adaptive verification (MRV2). 1 enables
    # it, 0 (default) disables. It does NOT trim verification -- it logs the
    # confidence head's predicted per-position acceptance vs the actual rate, so
    # the head's value can be judged before the trimming path is built. Requires
    # a DSpark checkpoint with a confidence head (enable_confidence_head).
    "VLLM_ASCEND_DSPARK_AV_OBSERVE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_DSPARK_AV_OBSERVE", "0"))
    ),
    # How many verification steps between DSpark AV observation log lines.
    "VLLM_ASCEND_DSPARK_AV_OBSERVE_INTERVAL": lambda: int(
        os.getenv("VLLM_ASCEND_DSPARK_AV_OBSERVE_INTERVAL", "200")
    ),
    # [Smoke] Run the DSpark adaptive-verification cost-table profiling path once
    # at startup to check it works on Ascend. It does NOT enable trimming: a
    # throwaway manager profiles per-shape step cost under ACLGraph and logs the
    # cost curves, per-shape timing spread, and monotonicity/NaN checks, so we
    # can tell whether ACLGraph replay timing can feed a cost model at all.
    # 1 enables, 0 (default) disables.
    "VLLM_ASCEND_DSPARK_AV_PROFILE_SMOKE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_DSPARK_AV_PROFILE_SMOKE", "0"))
    ),
    # [Shadow] Record-only D-Cut trim-decision dry run (stage 1 of the D-Cut
    # plan, docs/adaptive_verify/). It does NOT trim: at startup it profiles the
    # AV cost table (same path as the smoke), then every N verify steps it runs
    # the upstream AdaptiveVerificationManager.get_num_tokens decision math on the
    # live confidence + that cost table and logs the draft budget it *would*
    # choose vs the full budget (trim %, estimated accepted-token loss). This
    # answers "would AV trim at this concurrency / context length" without
    # touching GDN state, before the real trimming path is wired. Needs a
    # confidence head (like AV_OBSERVE). 1 enables, 0 (default) disables.
    "VLLM_ASCEND_DSPARK_AV_SHADOW": lambda: bool(
        int(os.getenv("VLLM_ASCEND_DSPARK_AV_SHADOW", "0"))
    ),
    # Route the GDN spec (draft-verification) path through the D-Cut variable-
    # length operators (npu_dcut_causal_conv1d / npu_dcut_recurrent_gated_delta_rule)
    # instead of the fixed-length npu_causal_conv1d_custom / npu_recurrent_gated_
    # delta_rule. Step 1 of the D-Cut GDN integration (docs/adaptive_verify/): with
    # this on but no trimming, the dcut path must match the existing path at full
    # verify length (a regression check) before manual/confidence trimming is
    # enabled. Passes the cumulative query_start_loc and the 2D [B, S]
    # ssm_state_indices the dcut kernels require. 1 enables, 0 (default) disables.
    "VLLM_ASCEND_DSPARK_ENABLE_DCUT": lambda: bool(
        int(os.getenv("VLLM_ASCEND_DSPARK_ENABLE_DCUT", "0"))
    ),
    # Manual per-request draft cap for D-Cut GDN verification (step 3 of the
    # D-Cut GDN integration, docs/adaptive_verify/). -1 (default) disables manual
    # trimming. A value >= 0 drives the existing MRV2 trimming path
    # (compact_batch -> reallocate_drafts) with a deterministic per-request cap
    # -- each verification request retains at most this many draft tokens
    # (bounded by its scheduled draft count) -- instead of the confidence cost
    # model, so the variable-length layout can be validated without depending on
    # the confidence head. Only takes effect when VLLM_ASCEND_DSPARK_ENABLE_DCUT
    # is 1; it also bypasses the varlen-backend rejection that would otherwise
    # leave GDN without an adaptive-verification manager. cap 0 keeps only the
    # anchor query per request; a large cap matches the no-trim step-1 path.
    "VLLM_ASCEND_DSPARK_DCUT_MANUAL_CAP": lambda: int(
        os.getenv("VLLM_ASCEND_DSPARK_DCUT_MANUAL_CAP", "-1")
    ),
    # Minimum KV-cache group width (layers per group). 0 disables the override
    # and keeps upstream grouping exactly. A positive value raises the group
    # width to at least this many layers, so a small heterogeneous draft bucket
    # (DSpark/DFlash) can no longer drag the width down and split a large
    # Mamba/attention bucket into many groups. E.g. set 16 for a DSpark draft
    # with 5 draft + 16 base + 48 mamba layers to collapse 15 groups into 5.
    "VLLM_ASCEND_KV_GROUP_MIN_SIZE": lambda: int(
        os.getenv("VLLM_ASCEND_KV_GROUP_MIN_SIZE", "0")
    ),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
