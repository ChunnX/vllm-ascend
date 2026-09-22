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
    # Explicit diagnostic policy: None preserves upstream AV. [0,1] selects
    # synchronous survival-prefix trimming, requiring DSpark AV and eager target
    # and draft. Not sensitive. This path intentionally pays D2H synchronization.
    "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD": lambda: (
        float(os.environ["VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD"])
        if "VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD" in os.environ
        else None
    ),
    # Opt-in eager lane B: run the real upstream AdaptiveVerificationManager
    # (cost-argmax budget + device survival top-k + async D2H double buffer) with
    # an injected synthetic cost curve, since eager has no cudagraph profiling to
    # price a real one. Requires DSpark AV and an eager target/draft. Default 0
    # (off). Not sensitive. Mutually exclusive with the survival-threshold lane.
    "VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV": lambda: bool(int(os.getenv("VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV", "0"))),
    # Pin the GDN speculative request axis to max_num_seqs for every graph
    # bucket, instead of letting it follow the batch. The fixed axis is the
    # contract ragged full graphs need -- a per-bucket axis gives each capture
    # size its own stateful tiling, which fails while concurrency ramps -- but it
    # only holds together with the attention-side axis and the padding cleanup in
    # the persistent seq_lens mirror, both of which the upstream FIA padding path
    # already provides. VLLM_ASCEND_DSPARK_AV_GRAPH=ragged therefore pins the axis
    # on its own and this variable is only for pinning it under the other modes,
    # to bisect the axis apart from the capture geometry. Default 0 (the
    # per-bucket axis). Not sensitive. See docs/adaptive_verify/graph_contract.md.
    "VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS": lambda: bool(int(os.getenv("VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS", "0"))),
    # Which graph mode the DSpark adaptive-verification lane runs under. Unset
    # is the default and means "ragged", except under the lane A threshold
    # below: its exact host boundaries are the opposite of what a captured graph
    # can do, so that lane stays eager unless a mode is named here explicitly.
    #   none    -- force CUDAGraphMode.NONE and stay eager.
    #   uniform -- allow graphs, but capture the decode descriptor at the uniform
    #              verify width. Adaptive verification otherwise switches the
    #              descriptor to min(num_tokens, max_num_seqs) requests with the
    #              dummy tokens spread evenly, so every bucket at or below
    #              max_num_seqs captures one token per request while a real
    #              speculative batch replays one request per verify width. Full
    #              attention survives that by re-issuing its kernel with refreshed
    #              host lengths; the GDN layers get no replay-time update, so the
    #              captured recurrent and conv geometry is the only geometry they
    #              ever run. A trimmed batch is not uniform and matches no
    #              descriptor, so it falls back to piecewise instead of replaying
    #              the wrong shape -- correct, with no trimming benefit in graph.
    #   ragged  -- keep the variable-length descriptor, so a trimmed batch replays
    #              too. The manager then captures one decode graph per size with
    #              num_reqs=min(Q,B_max) and max_query_len=decode_query_len, and a
    #              batch with no more requests, no more tokens and no longer a
    #              query replays it -- which is what a trimmed batch is. Selecting
    #              this pins the GDN request axis, because the axis is what makes
    #              replaying a different per-request split safe for the state
    #              operators. Validated on device, including a request axis far
    #              wider than the live rows.
    # Not sensitive. See docs/adaptive_verify/graph_contract.md.
    "VLLM_ASCEND_DSPARK_AV_GRAPH": lambda: os.getenv("VLLM_ASCEND_DSPARK_AV_GRAPH", ""),
    # Opt out of this repo's adaptive-verification adaptation entirely and run
    # upstream's own manager, for comparing against stock behaviour. Default 1
    # (adapt). Setting 0 leaves enable_adaptive_verification working, just
    # without the ragged plumbing and the graph modes above. Not sensitive.
    "VLLM_ASCEND_DSPARK_AV_ADAPT": lambda: bool(int(os.getenv("VLLM_ASCEND_DSPARK_AV_ADAPT", "1"))),
    # Broadcast each step's confidence from rank 0 before deciding the budget.
    # Upstream does not: it broadcasts the cost curves once at setup and then
    # trusts every rank to compute identical confidences, which they do because
    # the confidence head's output is already reduced across the group. This was
    # added as insurance against ranks disagreeing on a survival tie and so
    # building different metadata, but insurance priced per step at TP=4 is a
    # synchronising collective on every decode step. Default 0 (upstream
    # behaviour); 1 restores it if ranks are ever seen to diverge -- the
    # whole-network gate at a zero noise floor is what would show that.
    # Not sensitive.
    "VLLM_ASCEND_DSPARK_AV_TP_BROADCAST": lambda: bool(int(os.getenv("VLLM_ASCEND_DSPARK_AV_TP_BROADCAST", "0"))),
    # Stop making the host query/seq-length view exact for the eager AV lanes.
    # Both lanes currently read the trimmed boundaries back from device each
    # step, which a captured graph cannot do. With this set the host keeps the
    # evenly-distributed upper bound upstream produces and only the device view
    # is exact -- upstream's own contract, and the precondition for GDN to claim
    # supports_device_cpu_query_lens_mismatch. Default 0 (keep the readback).
    # Not sensitive. Eager-only switch; the graph phase makes it unconditional.
    "VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND": lambda: bool(int(os.getenv("VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND", "0"))),
    # Steps between the eager adaptive-verification lanes' aggregated warn
    # lines. Warn level so the trimming decisions are visible in an ordinary
    # serve log; aggregated so a long run stays readable. Not sensitive.
    "VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL": lambda: int(
        os.getenv("VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL", "50")
    ),
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
    "VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK", "0"))),
    # Minimum KV-cache group width (layers per group). 0 disables the override
    # and keeps upstream grouping exactly. A positive value raises the group
    # width to at least this many layers, so a small heterogeneous draft bucket
    # (DSpark/DFlash) can no longer drag the width down and split a large
    # Mamba/attention bucket into many groups. E.g. set 16 for a DSpark draft
    # with 5 draft + 16 base + 48 mamba layers to collapse 15 groups into 5.
    "VLLM_ASCEND_KV_GROUP_MIN_SIZE": lambda: int(os.getenv("VLLM_ASCEND_KV_GROUP_MIN_SIZE", "0")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
