# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        custom_fused_infer_attention_def.cpp
    )
endif()

# This kernel owns its event synchronization explicitly. Enabling automatic
# synchronization changes the pipeline ordering and is not supported.
add_ops_compile_options(
    OP_NAME CustomFusedInferAttention
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

add_ops_compile_options(
    OP_NAME CustomFusedInferAttention
    COMPUTE_UNIT ascend310p1
    OPTIONS
        -mllvm
        -cce-aicore-jump-expand=true
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE custom_fused_infer_attention
        ACLNNTYPE aclnn
    )
    target_include_directories(
        ${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
