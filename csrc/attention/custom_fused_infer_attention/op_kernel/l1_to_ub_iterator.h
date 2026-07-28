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
 * \file l1_to_ub_iterator.h
 * \brief
 */

#ifndef L1_TO_UB_ITERATOR_H
#define L1_TO_UB_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l1_to_ub
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType>
struct l1_to_ub {
    __aicore__ l1_to_ub(AscendC::LocalTensor<DataType> ubTensor,
                        AscendC::LocalTensor<DataType> l1Tensor,
                        uint16_t nBurst,
                        uint16_t lenBurst,
                        uint16_t srcStride,
                        uint16_t dstStride)
    {
        AscendC::DataCopy(ubTensor, l1Tensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};

/////////////////////////////////////////////////////
// ub_to_l1
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType>
struct ub_to_l1 {
    __aicore__ ub_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::LocalTensor<DataType> ubTensor,
                        uint16_t nBurst,
                        uint16_t lenBurst,
                        uint16_t srcStride,
                        uint16_t dstStride)
    {
        AscendC::DataCopy(l1Tensor, ubTensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};
#endif // L1_TO_UB_ITERATOR_H