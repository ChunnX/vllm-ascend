/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file l0c_to_ub_iterator.h
 * \brief
 */
#ifndef L0C_TO_UB_ITERATOR_H
#define L0C_TO_UB_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l0c_to_ub
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag, typename ElementIn, typename ElementOut, bool MatrixMode = true> struct l0c_to_ub {
    __aicore__ l0c_to_ub(AscendC::LocalTensor<ElementOut> ubTensor, AscendC::LocalTensor<ElementIn> l0cTensor,
                         uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        constexpr auto mode =
            MatrixMode ? AscendC::BlockMode::BLOCK_MODE_MATRIX : AscendC::BlockMode::BLOCK_MODE_VECTOR;
        AscendC::DataCopy(ubTensor, l0cTensor,
                          AscendC::DataCopyParams(nBurst,                              // count
                                                  lenBurst,                            // len
                                                  srcStride,                           // srcStrideIn
                                                  dstStride),                          // dstStrideIn
                          AscendC::DataCopyEnhancedParams(mode,                        // blockModeIn
                                                          AscendC::DeqScale::DEQ_NONE, // deqScaleIn
                                                          0,                           // deqValueIn
                                                          0,                           // sidStoreModeIn
                                                          false,                       // isReluIn
                                                          pad_t::PAD_NONE,             // padModeIn
                                                          0)                           // padValueIn
        );
    };
};

template <ArchType ArchTag>
struct l0c_to_ub<ArchTag, int32_t, half> {
    __aicore__ l0c_to_ub(AscendC::LocalTensor<half> ubTensor,
                         AscendC::LocalTensor<int32_t> l0cTensor,
                         uint16_t nBurst,
                         uint16_t lenBurst,
                         uint16_t srcStride,
                         uint16_t dstStride)
    {
        AscendC::DataCopy(ubTensor, l0cTensor,
                          AscendC::DataCopyParams(nBurst,                                        // count
                                                  lenBurst,                                      // len
                                                  srcStride,                                     // srcStrideIn
                                                  dstStride),                                    // dstStrideIn
                          AscendC::DataCopyEnhancedParams(AscendC::BlockMode::BLOCK_MODE_MATRIX, // blockModeIn
                                                          AscendC::DeqScale::VDEQ16,             // deqScaleIn
                                                          0,                                     // deqValueIn
                                                          0,                                     // sidStoreModeIn
                                                          false,                                 // isReluIn
                                                          pad_t::PAD_NONE,                       // padModeIn
                                                          0)                                     // padValueIn
        );
    };
};

#endif // L0C_TO_UB_ITERATOR_H