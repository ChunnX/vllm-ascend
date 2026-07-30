# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        custom_fused_infer_attention_v310_def.cpp
    )
endif()

# This kernel owns its event synchronization explicitly (the SET_FLAG/WAIT_FLAG
# pairs in unpad_paged_attention_decoder.h). Automatic synchronization inserts
# its own barriers and changes the pipeline ordering, which this kernel does not
# support -- the operator repo builds it with auto-sync off.
add_ops_compile_options(
    OP_NAME CustomFusedInferAttentionV310
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

# 310p1 codegen: the kernel is large enough to need expanded jumps.
add_ops_compile_options(
    OP_NAME CustomFusedInferAttentionV310
    COMPUTE_UNIT ascend310p1
    OPTIONS -mllvm
            -cce-aicore-jump-expand=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE custom_fused_infer_attention_v310 ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
