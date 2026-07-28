/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unpad_paged_attention_decoder.h
 * \brief
 */

#ifndef UNPAD_PAGED_ATTENTION_DECODER_H
#define UNPAD_PAGED_ATTENTION_DECODER_H

#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "iterator.h"
#include "mma.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ifa_public_define.h"
#include "gm_to_l1_iterator.h"
#include "gm_to_ub_iterator.h"
#include "l0c_to_ub_iterator.h"
#include "l1_to_l0_iterator.h"
#include "l1_to_ub_iterator.h"

constexpr int32_t LOCAL_STORAGE_BUFFER_SIZE = 4096;

constexpr int32_t MASK_TYPE_NORMAL = 1;
constexpr int32_t MASK_TYPE_COMPRESSED = 2;

template <CalcMode DECODE_MODE = CalcMode::CALC_MODE_DEFAULT>
class PagedAttentionDecoder {
public:
    __aicore__ inline PagedAttentionDecoder(__gm__ uint8_t *__restrict__ gmSrcQ, __gm__ uint8_t *__restrict__ gmSrcK,
                                            __gm__ uint8_t *__restrict__ gmSrcV, __gm__ uint8_t *__restrict__ gmSrcM,
                                            __gm__ uint8_t *__restrict__ gmDstO, half tor,
                                            uint32_t blockSize)
        : tor(tor), blockSize(blockSize)
    {
        gmSrcQTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcQ));
        gmSrcKTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcK));
        gmSrcVTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcV));
        gmSrcMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcM));
        gmDstOTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmDstO));

        switch (DECODE_MODE) {
            case (CalcMode::CALC_MODE_PREFILL):{
                InitOffsetPrefill();
                break;
            }
            case (CalcMode::CALC_MODE_DEFAULT):{
            }
            default: {
                InitOffsetDefault();
            }
        }

        lsUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lsUbufOffset);
        lpUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lpUbufOffset);
        ls32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32UbufOffset);
        loUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbufOffset);
        lmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lmUbufOffset);
        hmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hmUbufOffset);
        gmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gmUbufOffset);
        dmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dmUbufOffset);
        llUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(llUbufOffset);
        glUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(glUbufOffset);
        tvUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tvUbufOffset);
        goUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbufOffset);
        maskUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(maskUbufOffset);
    }

    __aicore__ inline void SetMask(int32_t len)
    {
        if (len >= VECTOR_SIZE_I) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - static_cast<int32_t>(MAX_LEN_64_BYTES) > 0 ? len - MAX_LEN_64_BYTES : 0;
        int32_t lowMask = len -static_cast<int32_t>(MAX_LEN_64_BYTES) >= 0 ? MAX_LEN_64_BYTES : len;
        if (len < MAX_LEN_64_BYTES) {
            SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
            return;
        }
        SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
    }

    __aicore__ inline void SetVcgMask(int32_t len)
    {
        if (len > BLOCK_SIZE_I) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline void ExpandToBlockHalf(AscendC::LocalTensor<half> dstTensor,
                                            AscendC::LocalTensor<half> srcTensor, int32_t len)
    {
        // 给srcTensor做扩展用，给每个block复制16份
        const uint32_t BLOCK_TWO = 2;
        const uint32_t BLOCK_NUM = 8;
        // (len,) -> len / 16 个 (16, 16)
        for (int32_t vaddsIdx = 0; vaddsIdx < BLOCK_TWO; ++vaddsIdx) {
            adds_v<ArchType::ASCEND_V200, half>(
                dstTensor[vaddsIdx * BLOCK_NUM * BLOCK_SIZE_I],
                srcTensor,
                0.0, len / BLOCK_SIZE_I, 1, 0, BLOCK_TWO * BLOCK_NUM, 1);
        }
        PIPE_BARRIER(V);
        // (16, len) -> (len, 16)
        for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE_I); ++vtransIdx) {
            tranpose_v<ArchType::ASCEND_V200, half>(
                dstTensor[vtransIdx * CUBE_MATRIX_SIZE_I],
                dstTensor[vtransIdx * CUBE_MATRIX_SIZE_I]);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ inline void InitOffsetPrefill()
    {
        uint32_t ls32_pre_block_index = 2;
        uint32_t lm_pre_block_index = 4;
        uint32_t tv_pre_block_index = 5;
        uint32_t go_pre_block_index = 6;

        uint32_t hm_pre_line_index = 1;
        uint32_t gm_pre_line_index = 2;
        uint32_t dm_pre_line_index = 3;
        uint32_t ll_pre_line_index = 5;
        uint32_t gl_pre_line_index = 7;

        lsUbufOffset = 0; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        lpUbufOffset = 0; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        ls32UbufOffset = ls32_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        loUbufOffset = ls32_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        lmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        hmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + hm_pre_line_index * UB_UINT8_LINE_SIZE_I;
        // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        gmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + gm_pre_line_index * UB_UINT8_LINE_SIZE_I;
        // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        dmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + dm_pre_line_index * UB_UINT8_LINE_SIZE_I;
        // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        llUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + ll_pre_line_index * UB_UINT8_LINE_SIZE_I;
        // 初始化为二十五个UB_UINT8_LINE_SIZE的偏移量
        glUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + gl_pre_line_index * UB_UINT8_LINE_SIZE_I;
        tvUbufOffset = tv_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        goUbufOffset = go_pre_block_index * UB_UINT8_BLOCK_SIZE_I;
    }

    __aicore__ inline void InitOffsetDefault()
    {
        uint32_t lp_dec_block_index = 2;
        uint32_t ls_dec_block_index = 4;
        uint32_t mask_dec_block_index = 8;
        uint32_t lo_dec_block_index = 10;

        uint32_t lm_block_index = 4;
        uint32_t tv_block_index = 5;
        uint32_t go_block_index = 6;

        uint32_t hm_line_index = 1;
        uint32_t dm_line_index = 2;
        uint32_t ll_line_index = 4;
        uint32_t gm_line_index = 6;
        uint32_t gl_line_index = 16;

        lsUbufOffset = 0; // 初始化为两个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        lpUbufOffset = lp_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE; // 初始化为两个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        ls32UbufOffset = ls_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE; // 初始化为四个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        maskUbufOffset = mask_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE; // 初始化为两个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        loUbufOffset = lo_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE;
        lmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        hmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + hm_line_index * UB_UINT8_LINE_SIZE_I;
        // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        dmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + dm_line_index * UB_UINT8_LINE_SIZE_I;
        // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        llUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + ll_line_index * UB_UINT8_LINE_SIZE_I;
        // 初始化为二十六个UB_UINT8_LINE_SIZE的偏移量
        gmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + gm_line_index * UB_UINT8_LINE_SIZE_I;
        tvUbufOffset = tv_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为十六个UB_UINT8_LINE_SIZE的偏移量
        // 初始化为十六个UB_UINT8_LINE_SIZE的偏移量
        glUbufOffset = tv_block_index * UB_UINT8_BLOCK_SIZE_I + gl_line_index * UB_UINT8_LINE_SIZE_I;
        goUbufOffset = go_block_index * UB_UINT8_BLOCK_SIZE_I;
    }

    __aicore__ inline void Init(uint64_t srcqOffsetReal, uint64_t srckOffsetReal, uint64_t srcvOffsetReal,
                                uint64_t srckOffsetReal1, uint64_t srcvOffsetReal1, uint64_t srcmOffsetReal,
                                uint64_t dstoOffsetReal, uint32_t initGReal, uint32_t wrapOReal,
                                uint32_t constextLenAlign16, uint32_t queryHeadNum, int32_t cmRowPing, int32_t cmRowPong)
    {
        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srckOffset1 = srckOffsetReal1;
        srcvOffset1 = srcvOffsetReal1;
        srcmOffset = srcmOffsetReal;
        dstoOffset = dstoOffsetReal;
        initG = initGReal;
        wrapO = wrapOReal;
        kvSeqAlign16 = constextLenAlign16;
        qHeadNum = queryHeadNum;
        compressMaskRowPing = cmRowPing;
        compressMaskRowPong = cmRowPong;
    }

public:
    __aicore__ inline void ProcessKvBlockPair(const uint32_t fm, const uint32_t fn, const uint32_t fk, const uint32_t bn,
                                const uint32_t mActual, const uint32_t n0Actual, const uint32_t n1Actual,
                                const uint32_t maskType, const uint32_t initKVE, const uint32_t headOffset = 0,
                                const uint32_t initKV = 1, half localTor = 1, const uint32_t scaleType = 0);

private:
    const uint32_t PingFlag = 0;
    const uint32_t PongFlag = 1;
    uint32_t vmPingPongFlag = 1;

    __aicore__ inline void LoadQueryAndAttentionMask(
        const uint32_t fm, const uint32_t fn, const uint32_t fk, const uint32_t bn,
        const uint32_t mActual, const uint32_t n1Actual,
        const uint32_t maskType, const uint32_t initGgO, const uint32_t initKVG,
        const uint32_t qOffset);

    __aicore__ inline void MoveQKPingToL0(
        const uint32_t fm, const uint32_t fn, const uint32_t fk,
        const uint32_t mActual, const uint32_t initGgO, const uint32_t initKV,
        const uint32_t qOffset,
        AscendC::LocalTensor<half> &l1kPingBufTensor);

    __aicore__ inline void MoveQKPongToL0(
        const uint32_t fm, const uint32_t bn, const uint32_t fk,
        const uint32_t mActual, const uint32_t n1Actual,
        const uint32_t initGgO, const uint32_t initKV,
        const uint32_t qOffset,
        AscendC::LocalTensor<half> &l1kPongBufTensor);

    __aicore__ inline void CalculateQKPingAndPrepareV(
        const uint32_t fm, const uint32_t fn, const uint32_t fk,
        const uint32_t mActual, const uint32_t n0Actual,
        const uint32_t initKV,
        AscendC::LocalTensor<half> &l1vPingBufTensor);

    __aicore__ inline void CalculateQKPongAndPrepareV(
        const uint32_t fm, const uint32_t bn, const uint32_t fk,
        const uint32_t mActual, const uint32_t n1Actual,
        const uint32_t initKV,
        AscendC::LocalTensor<half> &l1vPongBufTensor);

    __aicore__ inline void CalculateSoftmaxPing(
        const uint32_t fm, const uint32_t fn,
        const uint32_t mActual, const uint32_t n0Actual,
        const uint32_t maskType, half localTor,
        const uint32_t n0AlignVector, const uint32_t n0AlignBlock,
        const uint32_t pSize, const uint32_t pSizeAlignFloat,
        const uint32_t gmUOffset, uint32_t &initGgDm,
        AscendC::LocalTensor<half> &l1pPingBufTensor);

    __aicore__ inline void CalculateSoftmaxPong(
        const uint32_t fm, const uint32_t bn,
        const uint32_t mActual, const uint32_t n1Actual,
        const uint32_t maskType, half localTor,
        const uint32_t n1AlignVector, const uint32_t n1AlignBlock,
        const uint32_t pSizeB, const uint32_t pSizeBAlignFloat,
        const uint32_t gmUOffset,
        AscendC::LocalTensor<half> &l1pPingBufTensor,
        AscendC::LocalTensor<half> &l1pPongBufTensor);

    __aicore__ inline void CalculatePVPing(
        const uint32_t fm, const uint32_t fn, const uint32_t fk,
        const uint32_t mActual, const uint32_t n0Actual,
        const uint32_t initKVE, const uint32_t n1Actual,
        AscendC::LocalTensor<half> &l1pPingBufTensor);

    __aicore__ inline void CalculatePVPong(
        const uint32_t fm, const uint32_t bn, const uint32_t fk,
        const uint32_t mActual, const uint32_t n1Actual,
        const uint32_t initKVE,
        AscendC::LocalTensor<half> &l1pPingBufTensor,
        AscendC::LocalTensor<half> &l1pPongBufTensor);

    __aicore__ inline void MovePVResultsToUb(
        const uint32_t fm, const uint32_t fk,
        const uint32_t mActual, const uint32_t oSize,
        const uint32_t n1Actual);

    __aicore__ inline void UpdateOnlineSoftmaxPing(
        const uint32_t fm, const uint32_t fk,
        const uint32_t mActual, const uint32_t kAlignVector,
        const uint32_t oSizeAlignFloat,
        const uint32_t glUOffset, const uint32_t goUOffset,
        uint32_t &initGgO);

    __aicore__ inline void UpdateOnlineSoftmaxPong(
        const uint32_t fm, const uint32_t fk,
        const uint32_t mActual, const uint32_t kAlignVector,
        const uint32_t oSizeAlignFloat,
        const uint32_t n1Actual,
        const uint32_t glUOffset, const uint32_t goUOffset);

    __aicore__ inline void NormalizeAndStoreOutput(
        const uint32_t fm, const uint32_t fk,
        const uint32_t mActual, const uint32_t oSizeAlignFloat,
        const uint32_t glUOffset, const uint32_t goUOffset,
        const uint32_t wrapO);

    AscendC::GlobalTensor<half> gmSrcQTensor;
    AscendC::GlobalTensor<half> gmSrcKTensor;
    AscendC::GlobalTensor<half> gmSrcVTensor;
    AscendC::GlobalTensor<half> gmSrcMTensor;
    AscendC::GlobalTensor<half> gmDstOTensor;

    AsdopsBuffer<ArchType::ASCEND_V200> buf;

    uint32_t l1qBufAddrOffset = 0;
    uint32_t l1kBufAddrOffset = 2 * UB_UINT8_BLOCK_SIZE_I;
    uint32_t l1pBufAddrOffset = 2 * L1_UINT8_BLOCK_SIZE;
    uint32_t l1vBufAddrOffset = 2 * L1_UINT8_BLOCK_SIZE + 2 * UB_UINT8_BLOCK_SIZE_I;
    uint32_t l1maskBufAddrOffset = 4 * L1_UINT8_BLOCK_SIZE;

    uint32_t l0aBufOffset = 0;
    uint32_t l0bBufOffset = 0;
    uint32_t l0cBufOffset = 0;

    uint32_t lsUbufOffset = 0;
    uint32_t lpUbufOffset = 0;
    uint32_t ls32UbufOffset = 0;
    uint32_t maskUbufOffset = 0;
    uint32_t loUbufOffset = 0;
    uint32_t lmUbufOffset = 0;
    uint32_t hmUbufOffset = 0;
    uint32_t gmUbufOffset = 0;
    uint32_t dmUbufOffset = 0;
    uint32_t llUbufOffset = 0;
    uint32_t glUbufOffset = 0;
    uint32_t tvUbufOffset = 0;
    uint32_t goUbufOffset = 0;

    AscendC::LocalTensor<half> l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1qBufAddrOffset);
    AscendC::LocalTensor<half> l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1kBufAddrOffset);
    AscendC::LocalTensor<half> l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1pBufAddrOffset);
    AscendC::LocalTensor<half> l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1vBufAddrOffset);
    AscendC::LocalTensor<half> l1maskBufAddr_tensor =
        buf.GetBuffer<BufferType::ASCEND_CB, half>(l1maskBufAddrOffset);

    AscendC::LocalTensor<half> l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, half>(l0aBufOffset);
    AscendC::LocalTensor<half> l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, half>(l0bBufOffset);
    AscendC::LocalTensor<float> l0cBufTensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(l0cBufOffset);

    AscendC::LocalTensor<half> lsUbufTensor;
    AscendC::LocalTensor<half> lpUbufTensor;
    AscendC::LocalTensor<half> lmUbufTensor;
    AscendC::LocalTensor<half> hmUbufTensor;
    AscendC::LocalTensor<half> gmUbufTensor;
    AscendC::LocalTensor<half> dmUbufTensor;
    AscendC::LocalTensor<half> tvUbufTensor;
    AscendC::LocalTensor<half> maskUbufTensor;
    AscendC::LocalTensor<float> ls32UbufTensor;
    AscendC::LocalTensor<float> loUbufTensor;
    AscendC::LocalTensor<float> llUbufTensor;
    AscendC::LocalTensor<float> glUbufTensor;
    AscendC::LocalTensor<float> goUbufTensor;

    half tor = 1.0;
    uint32_t blockSize = VECTOR_SIZE_I;

    uint64_t srcqOffset = 0;
    uint64_t srckOffset = 0;
    uint64_t srcvOffset = 0;
    uint64_t srckOffset1 = 0;
    uint64_t srcvOffset1 = 0;
    uint64_t dstoOffset = 0;
    uint64_t srcmOffset = 0;

    // For compress mask，用于计算当前qblock的第一个q的因果偏移量。
    int32_t compressMaskRowPing = 0;
    int32_t compressMaskRowPong = 0;

    uint32_t initG = 0;
    uint32_t wrapO = 0;
    uint32_t kvSeqAlign16 = 0;
    uint32_t qHeadNum = 0;
};


template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::LoadQueryAndAttentionMask(
    const uint32_t fm, const uint32_t fn, const uint32_t fk, const uint32_t bn,
    const uint32_t mActual, const uint32_t n1Actual,
    const uint32_t maskType, const uint32_t initGgO, const uint32_t initKVG,
    const uint32_t qOffset)
{
    if (initGgO != 0) {
        if (initKVG) {
            WAIT_FLAG(MTE1, MTE2, PingFlag);
            WAIT_FLAG(MTE1, MTE2, PongFlag);
        }
        if (mActual == 1) {
            AscendC::DataCopy(l1qBufAddrTensor[qOffset], gmSrcQTensor[srcqOffset], fk);
        } else {
            Nd2NzParams nd2NzParams;
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = mActual;
            nd2NzParams.dValue = fk;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = fk * qHeadNum;
            nd2NzParams.dstNzC0Stride = fm;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;

            AscendC::DataCopy(l1qBufAddrTensor[qOffset], gmSrcQTensor[srcqOffset], nd2NzParams);
        }

        SET_FLAG(MTE2, MTE1, PingFlag);
        if (n1Actual != 0) {
            SET_FLAG(MTE2, MTE1, PongFlag);
        }
    }

    if (maskType != 0) {
        WAIT_FLAG(MTE1, MTE2, PingFlag + 2);
        if (maskType == MASK_TYPE_NORMAL) {
            AscendC::DataCopy(l1maskBufAddr_tensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
                                gmSrcMTensor[srcmOffset],
                                AscendC::DataCopyParams(mActual,
                                                        fn / 16,
                                                        (kvSeqAlign16 - fn) / 16,
                                                        0));
        } else if (maskType == MASK_TYPE_COMPRESSED) {
            uint32_t l1BaseOffset = PingFlag * L0AB_HALF_BUF_SIZE_I;
            uint64_t srcStartOffset = compressMaskRowPing * blockSize;

            AscendC::DataCopy(
                l1maskBufAddr_tensor[l1BaseOffset],
                gmSrcMTensor[srcStartOffset],
                AscendC::DataCopyParams(
                    mActual,
                    fn / 16,
                    (blockSize - fn) / 16,
                    0
                )
            );
        }

        SET_FLAG(MTE2, MTE1, PingFlag + 2);
        WAIT_FLAG(MTE2, MTE1, PingFlag + 2);
        WAIT_FLAG(V, MTE1, PingFlag);
        l1_to_ub<ArchType::ASCEND_V200, half>(
            maskUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            l1maskBufAddr_tensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            1, mActual * fn / BLOCK_SIZE_I, 0, 0);
        SET_FLAG(MTE1, V, PingFlag);
        SET_FLAG(MTE1, MTE2, PingFlag + 2);
        if (n1Actual != 0) {
            WAIT_FLAG(MTE1, MTE2, PongFlag + 2);
            if (maskType == MASK_TYPE_NORMAL) {
                AscendC::DataCopy(l1maskBufAddr_tensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
                                    gmSrcMTensor[srcmOffset + blockSize],
                                    AscendC::DataCopyParams(mActual,
                                                            bn / 16,
                                                            (kvSeqAlign16 - bn) / 16,
                                                            0));
            } else if (maskType == MASK_TYPE_COMPRESSED) {
                uint32_t l1BaseOffset = PongFlag * L0AB_HALF_BUF_SIZE_I;
                uint64_t srcStartOffset = compressMaskRowPong * blockSize;

                AscendC::DataCopy(
                    l1maskBufAddr_tensor[l1BaseOffset],
                    gmSrcMTensor[srcStartOffset],
                    AscendC::DataCopyParams(
                        mActual,
                        bn / 16,
                        (blockSize - bn) / 16,
                        0
                    )
                );
            }

            SET_FLAG(MTE2, MTE1, PongFlag + 2);
            WAIT_FLAG(MTE2, MTE1, PongFlag + 2);
            WAIT_FLAG(V, MTE1, PongFlag);
            l1_to_ub<ArchType::ASCEND_V200, half>(
                maskUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                l1maskBufAddr_tensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
                1, mActual * bn / BLOCK_SIZE_I, 0, 0);
            SET_FLAG(MTE1, V, PongFlag);
            SET_FLAG(MTE1, MTE2, PongFlag + 2);
        }
    }
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::MoveQKPingToL0(
    const uint32_t fm, const uint32_t fn, const uint32_t fk,
    const uint32_t mActual, const uint32_t initGgO, const uint32_t initKV,
    const uint32_t qOffset,
    AscendC::LocalTensor<half> &l1kPingBufTensor)
{
    if (initGgO == 1) {
        WAIT_FLAG(MTE2, MTE1, PingFlag);
    }
    if (mActual == 1) {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            l1qBufAddrTensor[qOffset].ReinterpretCast<half>(),
            0, 1, 0, 1, 0, 0);
    } else {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::NZ, DataFormatT::ZZ>(
            l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            l1qBufAddrTensor[qOffset].ReinterpretCast<half>(),
            fm, fk, 0, 0, 0, 0);
    }

    SET_FLAG(MTE1, M, PingFlag);
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    WAIT_FLAG(MTE1, MTE2, PingFlag + 4);
#endif
    if (initKV) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1kPingBufTensor,
            gmSrcKTensor[srckOffset],
            fn, blockSize, fn, fk, fk, fk);
    }
    SET_FLAG(MTE2, MTE1, PingFlag);
    WAIT_FLAG(MTE2, MTE1, PingFlag);
    WAIT_FLAG(M, MTE1, PingFlag + 2);
    l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l1kPingBufTensor,
        0, fk * fn / CUBE_MATRIX_SIZE_I, 0, 1, 0, 0);
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    SET_FLAG(MTE1, MTE2, PingFlag + 4);
#endif
    SET_FLAG(MTE1, M, PingFlag + 2);
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::MoveQKPongToL0(
    const uint32_t fm, const uint32_t bn, const uint32_t fk,
    const uint32_t mActual, const uint32_t n1Actual,
    const uint32_t initGgO, const uint32_t initKV,
    const uint32_t qOffset,
    AscendC::LocalTensor<half> &l1kPongBufTensor)
{
    WAIT_FLAG(M, MTE1, PongFlag);
    if (initGgO == 1) {
        WAIT_FLAG(MTE2, MTE1, PongFlag);
    }
    if (mActual == 1) {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1qBufAddrTensor[qOffset],
            0, 1, 0, 1, 0, 0);
    } else {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::NZ, DataFormatT::ZZ>(
            l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1qBufAddrTensor[qOffset].ReinterpretCast<half>(),
            fm, fk, 0, 0, 0, 0);
    }

    SET_FLAG(MTE1, M, PongFlag);
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE1, MTE2, PongFlag + 2);
#else
    WAIT_FLAG(MTE1, MTE2, PongFlag + 4);
#endif
    if (initKV) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1kPongBufTensor,
            gmSrcKTensor[srckOffset1],
            bn, blockSize, bn, fk, fk, fk);
    }
    SET_FLAG(MTE2, MTE1, PongFlag);
    WAIT_FLAG(MTE2, MTE1, PongFlag);
    WAIT_FLAG(M, MTE1, PongFlag + 2);
    l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
        l1kPongBufTensor,
        0, fk * bn / CUBE_MATRIX_SIZE_I, 0, 1, 0, 0);
    SET_FLAG(MTE1, M, PongFlag + 2);
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE1, MTE2, PongFlag + 2);
#else
    SET_FLAG(MTE1, MTE2, PongFlag + 4);
#endif
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::CalculateQKPingAndPrepareV(
    const uint32_t fm, const uint32_t fn, const uint32_t fk,
    const uint32_t mActual, const uint32_t n0Actual,
    const uint32_t initKV,
    AscendC::LocalTensor<half> &l1vPingBufTensor)
{
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    WAIT_FLAG(MTE1, MTE2, PingFlag + 6);
#endif
    if (initKV) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1vPingBufTensor,
            gmSrcVTensor[srcvOffset],
            fn, blockSize, fn, fk, fk, fk);
    }
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE2, MTE1, PingFlag + 2);
#else
    SET_FLAG(MTE2, MTE1, PingFlag + 4);
#endif
    WAIT_FLAG(MTE1, M, PingFlag + 2);
    WAIT_FLAG(MTE1, M, PingFlag);
    WAIT_FLAG(V, M, PingFlag);

    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        mActual, n0Actual, fk, 1);

    SET_FLAG(M, V, PingFlag);
    SET_FLAG(M, MTE1, PingFlag);
    SET_FLAG(M, MTE1, PingFlag + 2);
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE2, MTE1, PingFlag + 2);
#else
    WAIT_FLAG(MTE2, MTE1, PingFlag + 4);
#endif
    WAIT_FLAG(M, MTE1, PingFlag + 2);
    if (fk == 16) {
        l1_to_l0_b<ArchType::ASCEND_V200, half, 1, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            l1vPingBufTensor,
            0, fn / BLOCK_SIZE_I, 0, 1, 0, 0);
    } else {
        for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (fn / BLOCK_SIZE_I); ++l0bLoadIdx) {
            l1_to_l0_b<ArchType::ASCEND_V200, half, 1, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I + l0bLoadIdx * fk * BLOCK_SIZE_I],
                l1vPingBufTensor[l0bLoadIdx * CUBE_MATRIX_SIZE_I],
                0, fk / BLOCK_SIZE_I, 0, fn / BLOCK_SIZE_I, 0, 0);
        }
    }
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    SET_FLAG(MTE1, MTE2, PingFlag + 6);
#endif
    SET_FLAG(MTE1, M, PingFlag + 2);
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::CalculateQKPongAndPrepareV(
    const uint32_t fm, const uint32_t bn, const uint32_t fk,
    const uint32_t mActual, const uint32_t n1Actual,
    const uint32_t initKV,
    AscendC::LocalTensor<half> &l1vPongBufTensor)
{
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE1, MTE2, PongFlag + 2);
#else
    WAIT_FLAG(MTE1, MTE2, PongFlag + 6);
#endif
    if (initKV) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1vPongBufTensor,
            gmSrcVTensor[srcvOffset1],
            bn, blockSize, bn, fk, fk, fk);
    }
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE2, MTE1, PongFlag + 2);
#else
    SET_FLAG(MTE2, MTE1, PongFlag + 4);
#endif
    WAIT_FLAG(MTE1, M, PongFlag + 2);
    WAIT_FLAG(MTE1, M, PongFlag);
    WAIT_FLAG(V, M, PongFlag);

    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
        l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
        l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
        mActual, n1Actual, fk, 1);

    SET_FLAG(M, V, PongFlag);
    SET_FLAG(M, MTE1, PongFlag);
    SET_FLAG(M, MTE1, PongFlag + 2);
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE2, MTE1, PongFlag + 2);
#else
    WAIT_FLAG(MTE2, MTE1, PongFlag + 4);
#endif
    WAIT_FLAG(M, MTE1, PongFlag + 2);
    if (fk == 16) {
        l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1vPongBufTensor,
            0, bn / BLOCK_SIZE_I, 0, 1, 0, 0);
    } else {
        for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (bn / BLOCK_SIZE_I); ++l0bLoadIdx) {
            l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I + l0bLoadIdx * fk * BLOCK_SIZE_I],
                l1vPongBufTensor[l0bLoadIdx * CUBE_MATRIX_SIZE_I],
                0, fk / BLOCK_SIZE_I, 0, bn / BLOCK_SIZE_I, 0, 0);
        }
    }
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE1, MTE2, PongFlag + 2);
#else
    SET_FLAG(MTE1, MTE2, PongFlag + 6);
#endif
    SET_FLAG(MTE1, M, PongFlag + 2);
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::CalculateSoftmaxPing(
    const uint32_t fm, const uint32_t fn,
    const uint32_t mActual, const uint32_t n0Actual,
    const uint32_t maskType, half localTor,
    const uint32_t n0AlignVector, const uint32_t n0AlignBlock,
    const uint32_t pSize, const uint32_t pSizeAlignFloat,
    const uint32_t gmUOffset, uint32_t &initGgDm,
    AscendC::LocalTensor<half> &l1pPingBufTensor)
{
    WAIT_FLAG(M, V, PingFlag);
    l0c_to_ub<ArchType::ASCEND_V200, float, half>(
        ls32UbufTensor.ReinterpretCast<half>()[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        1, pSize / CUBE_MATRIX_SIZE_I, 0, 0);

    PIPE_BARRIER(V);
    SET_FLAG(V, M, PingFlag);

    // NZ->ND
    for (uint32_t fractalBlkIdx = 0; fractalBlkIdx < n0AlignBlock; fractalBlkIdx++) {
        ub_to_ub<ArchType::ASCEND_V200, half>(
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * BLOCK_SIZE_I],
            ls32UbufTensor.ReinterpretCast<half>()[
                PingFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * fm * BLOCK_SIZE_I],
            0,
            mActual,
            1,
            0,
            fn / BLOCK_SIZE_I - 1
        );
    }
    PIPE_BARRIER(V);

    AscendC::Muls(lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                  lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                  localTor,
                  pSize);
    PIPE_BARRIER(V);

    if (maskType != 0) {
        WAIT_FLAG(MTE1, V, PingFlag);
        add_v<ArchType::ASCEND_V200, half>(lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           maskUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           pSize / VECTOR_SIZE_I,
                                           1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE1, PingFlag);
    }

    if (n0Actual <= VECTOR_SIZE_I) {
        if (n0Actual != VECTOR_SIZE_I) {
            SetMask(n0Actual % VECTOR_SIZE_I);
        }
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbufTensor,
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            mActual, 1, 1, n0AlignBlock);
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, half>(
            tvUbufTensor,
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            0, mActual, 8, (fn - VECTOR_SIZE_I) / BLOCK_SIZE_I, 0);
        PIPE_BARRIER(V);
        if (n0Actual % VECTOR_SIZE_I != 0) {
            SetMask(n0Actual % VECTOR_SIZE_I);
        }
        max_v<ArchType::ASCEND_V200, half>(
            tvUbufTensor,
            tvUbufTensor,
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + VECTOR_SIZE_I],
            mActual, 1, 1, 1, 8, 8, n0AlignBlock);
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbufTensor,
            tvUbufTensor,
            mActual, 1, 1, 8);
        PIPE_BARRIER(V);
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);

    if (initGgDm == 0) {
        max_v<ArchType::ASCEND_V200, half>(
            hmUbufTensor,
            lmUbufTensor,
            gmUbufTensor[gmUOffset],
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(
            dmUbufTensor[PingFlag * UB_HALF_LINE_SIZE_I],
            gmUbufTensor[gmUOffset],
            hmUbufTensor,
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        // fm*2是因为返回了value和index, 310P不支持配置ONLY_VALUE
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbufTensor[gmUOffset],
            hmUbufTensor,
            0, 1, 2 * fm / BLOCK_SIZE_I, 0, 0);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbufTensor, hmUbufTensor, fm * 2);
    } else {
        initGgDm = 0;
        // fm*2是因为返回了value和index, 310P不支持配置ONLY_VALUE
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbufTensor[gmUOffset],
            lmUbufTensor,
            0, 1, 2 * fm / BLOCK_SIZE_I, 0, 0);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbufTensor, gmUbufTensor[gmUOffset], fm * 2);
    }

    if (fn < VECTOR_SIZE_I) {
        SetMask(fn);
    }
    for (uint32_t vSubIdx = 0; vSubIdx < n0AlignVector; vSubIdx++) {
        sub_v<ArchType::ASCEND_V200, half>(
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + vSubIdx * VECTOR_SIZE_I],
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + vSubIdx * VECTOR_SIZE_I],
            tvUbufTensor,
            mActual, 1, 1, 0, fn / BLOCK_SIZE_I, fn / BLOCK_SIZE_I, 2);
    }
    if (fn < VECTOR_SIZE_I) {
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    }

    PIPE_BARRIER(V);
    conv_v<ArchType::ASCEND_V200, half, float>(
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        pSizeAlignFloat, 1, 1, 8, 4);
    PIPE_BARRIER(V);
    exp_v<ArchType::ASCEND_V200, float>(
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        pSizeAlignFloat, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    WAIT_FLAG(MTE3, V, PingFlag);
    conv_v<ArchType::ASCEND_V200, float, half>(
        lpUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        pSizeAlignFloat, 1, 1, 4, 8);
    PIPE_BARRIER(V);
    SET_FLAG(V, MTE3, PingFlag);
    SetMaskNorm();

    if (n0Actual < FLOAT_VECTOR_SIZE_I) {
        if (n0Actual != FLOAT_VECTOR_SIZE_I) {
            SetVectorMask<int8_t>(0x0, ((long)1 << n0Actual) - 1);
        }
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            mActual,
            2,
            1,
            fn / BLOCK_SIZE_FLOAT);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    } else {
        for (int64_t vcalcIdx = 1; vcalcIdx < n0Actual / FLOAT_VECTOR_SIZE_I; vcalcIdx++) {
            add_v<ArchType::ASCEND_V200, float>(
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + vcalcIdx * FLOAT_VECTOR_SIZE_I],
                mActual, 1, 1, 1, fn / BLOCK_SIZE_FLOAT, fn / BLOCK_SIZE_FLOAT, fn / BLOCK_SIZE_FLOAT);
            PIPE_BARRIER(V);
        }
        if (n0Actual % FLOAT_VECTOR_SIZE_I != 0) {
            SetMask(n0Actual % FLOAT_VECTOR_SIZE_I);
            add_v<ArchType::ASCEND_V200, float>(
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE +
                    (n0Actual / FLOAT_VECTOR_SIZE_I) * FLOAT_VECTOR_SIZE_I],
                mActual, 1, 1, 1, fn / BLOCK_SIZE_FLOAT, fn / BLOCK_SIZE_FLOAT, fn / BLOCK_SIZE_FLOAT);
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        }
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            mActual, 2, 1, fn / BLOCK_SIZE_FLOAT);
    }

    PIPE_BARRIER(V);
    SetMaskNorm();
    SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    WAIT_FLAG(MTE1, MTE3, PingFlag);
    WAIT_FLAG(V, MTE3, PingFlag);

    if (mActual == 1) {
        ub_to_l1<ArchType::ASCEND_V200, half>(
            l1pPingBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            lpUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            1, fn / BLOCK_SIZE_I, 0, 0);
    } else {
        for (uint32_t fractalBlkIdx = 0; fractalBlkIdx < n0AlignBlock; fractalBlkIdx++) {
            ub_to_l1<ArchType::ASCEND_V200, half>(
                l1pPingBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I + fractalBlkIdx * fm * BLOCK_SIZE_I],
                lpUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * BLOCK_SIZE_I],
                mActual,
                1,
                fn / BLOCK_SIZE_I - 1,
                0);
        }
    }

    SET_FLAG(MTE3, V, PingFlag);
    SET_FLAG(MTE3, MTE1, PingFlag);
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::CalculateSoftmaxPong(
    const uint32_t fm, const uint32_t bn,
    const uint32_t mActual, const uint32_t n1Actual,
    const uint32_t maskType, half localTor,
    const uint32_t n1AlignVector, const uint32_t n1AlignBlock,
    const uint32_t pSizeB, const uint32_t pSizeBAlignFloat,
    const uint32_t gmUOffset,
    AscendC::LocalTensor<half> &l1pPingBufTensor,
    AscendC::LocalTensor<half> &l1pPongBufTensor)
{
    WAIT_FLAG(M, V, PongFlag);
    l0c_to_ub<ArchType::ASCEND_V200, float, half>(
        ls32UbufTensor.ReinterpretCast<half>()[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
        1, pSizeB / CUBE_MATRIX_SIZE_I, 0, 0);

    PIPE_BARRIER(V);
    SET_FLAG(V, M, PongFlag);

    // NZ->ND
    for (uint32_t fractalBlkIdx = 0; fractalBlkIdx < n1AlignBlock; fractalBlkIdx++) {
        ub_to_ub<ArchType::ASCEND_V200, half>(
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * BLOCK_SIZE_I],
            ls32UbufTensor.ReinterpretCast<half>()[
                PongFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * fm * BLOCK_SIZE_I],
            0,
            mActual,
            1,
            0,
            bn / BLOCK_SIZE_I - 1
        );
    }
    PIPE_BARRIER(V);

    AscendC::Muls(lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                  lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                  localTor,
                  pSizeB);
    PIPE_BARRIER(V);
    if (maskType != 0) {
        WAIT_FLAG(MTE1, V, PongFlag);
        add_v<ArchType::ASCEND_V200, half>(lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           maskUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           pSizeB / VECTOR_SIZE_I,
                                           1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE1, PongFlag);
    }

    if (n1Actual <= VECTOR_SIZE_I) {
        if (n1Actual != VECTOR_SIZE_I) {
            SetMask(n1Actual % VECTOR_SIZE_I);
        }
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbufTensor,
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            mActual, 1, 1, n1AlignBlock);
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, half>(
            tvUbufTensor,
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            0, mActual, 8, (bn - VECTOR_SIZE_I) / BLOCK_SIZE_I, 0);
        PIPE_BARRIER(V);
        if (n1Actual % VECTOR_SIZE_I != 0) {
            SetMask(n1Actual % VECTOR_SIZE_I);
        }
        max_v<ArchType::ASCEND_V200, half>(
            tvUbufTensor,
            tvUbufTensor,
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + VECTOR_SIZE_I],
            mActual, 1, 1, 1, 8, 8, n1AlignBlock);
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbufTensor,
            tvUbufTensor,
            mActual, 1, 1, 8);
        PIPE_BARRIER(V);
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

    PIPE_BARRIER(V);
    max_v<ArchType::ASCEND_V200, half>(
        hmUbufTensor,
        lmUbufTensor,
        gmUbufTensor[gmUOffset],
        1, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
    sub_v<ArchType::ASCEND_V200, half>(
        dmUbufTensor[PongFlag * UB_HALF_LINE_SIZE_I],
        gmUbufTensor[gmUOffset],
        hmUbufTensor,
        1, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
    ExpandToBlockHalf(tvUbufTensor, hmUbufTensor, 2 * fm);
    ub_to_ub<ArchType::ASCEND_V200, half>(
        gmUbufTensor[gmUOffset],
        hmUbufTensor,
        0, 1, 2 * fm / BLOCK_SIZE_I, 0, 0);
    PIPE_BARRIER(V);

    if (bn < VECTOR_SIZE_I) {
        SetMask(bn);
    }
    for (uint32_t vSubIdx = 0; vSubIdx < n1AlignVector; vSubIdx++) {
        sub_v<ArchType::ASCEND_V200, half>(
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + vSubIdx * VECTOR_SIZE_I],
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + vSubIdx * VECTOR_SIZE_I],
            tvUbufTensor,
            mActual, 1, 1, 0, bn / BLOCK_SIZE_I, bn / BLOCK_SIZE_I, 2);
    }
    if (bn < VECTOR_SIZE_I) {
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    }

    PIPE_BARRIER(V);
    conv_v<ArchType::ASCEND_V200, half, float>(
        ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        pSizeBAlignFloat, 1, 1, 8, 4);
    PIPE_BARRIER(V);
    exp_v<ArchType::ASCEND_V200, float>(
        ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        pSizeBAlignFloat, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    WAIT_FLAG(MTE3, V, PongFlag);
    conv_v<ArchType::ASCEND_V200, float, half>(
        lpUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        pSizeBAlignFloat, 1, 1, 4, 8);
    PIPE_BARRIER(V);
    SET_FLAG(V, MTE3, PongFlag);
    SetMaskNorm();

    if (n1Actual < FLOAT_VECTOR_SIZE_I) {
        if (n1Actual != FLOAT_VECTOR_SIZE_I) {
            SetVectorMask<int8_t>(0x0, ((long)1 << n1Actual) - 1);
        }
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbufTensor[PongFlag * UB_FLOAT_LINE_SIZE_I],
            ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            mActual,
            2,
            1,
            bn / BLOCK_SIZE_FLOAT);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    } else {
        for (int64_t vcalcIdx = 1; vcalcIdx < n1Actual / FLOAT_VECTOR_SIZE_I; vcalcIdx++) {
            add_v<ArchType::ASCEND_V200, float>(
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + vcalcIdx * FLOAT_VECTOR_SIZE_I],
                mActual,
                1,
                1,
                1,
                bn / BLOCK_SIZE_FLOAT,
                bn / BLOCK_SIZE_FLOAT,
                bn / BLOCK_SIZE_FLOAT);
            PIPE_BARRIER(V);
        }
        if (n1Actual % FLOAT_VECTOR_SIZE_I != 0) {
            SetVectorMask<int8_t>(0x0, ((long)1 << (n1Actual % FLOAT_VECTOR_SIZE_I)) - 1);
            add_v<ArchType::ASCEND_V200, float>(
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE +
                    (n1Actual / FLOAT_VECTOR_SIZE_I) * FLOAT_VECTOR_SIZE_I],
                mActual,
                1,
                1,
                1,
                bn / BLOCK_SIZE_FLOAT,
                bn / BLOCK_SIZE_FLOAT,
                bn / BLOCK_SIZE_FLOAT);
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        }
        // dstRepStride填2是为了和cmax保持一致，max会多返回一个索引，这样更新rowsum时才可以对应相乘
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbufTensor[PongFlag * UB_FLOAT_LINE_SIZE_I],
            ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            mActual, 2, 1, bn / BLOCK_SIZE_FLOAT);
    }

    PIPE_BARRIER(V);
    SetMaskNorm();
    SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    WAIT_FLAG(MTE1, MTE3, PongFlag);
    WAIT_FLAG(V, MTE3, PongFlag);

    if (mActual == 1) {
        ub_to_l1<ArchType::ASCEND_V200, half>(
            l1pPongBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            lpUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            1, bn / BLOCK_SIZE_I, 0, 0);
    } else {
        for (uint32_t fractalBlkIdx = 0; fractalBlkIdx < n1AlignBlock; fractalBlkIdx++) {
            ub_to_l1<ArchType::ASCEND_V200, half>(
                l1pPingBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I + fractalBlkIdx * fm * BLOCK_SIZE_I],
                lpUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * BLOCK_SIZE_I],
                mActual,
                1,
                bn / BLOCK_SIZE_I - 1,
                0);
        }
    }
    SET_FLAG(MTE3, V, PongFlag);
    SET_FLAG(MTE3, MTE1, PongFlag);
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::CalculatePVPing(
    const uint32_t fm, const uint32_t fn, const uint32_t fk,
    const uint32_t mActual, const uint32_t n0Actual,
    const uint32_t initKVE, const uint32_t n1Actual,
    AscendC::LocalTensor<half> &l1pPingBufTensor)
{
    WAIT_FLAG(MTE3, MTE1, PingFlag);
    WAIT_FLAG(M, MTE1, PingFlag);
    if (mActual == 1) {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            l1pPingBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            0, 1, 0, 1, 0, 0);
    } else {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::NZ, DataFormatT::ZZ>(
            l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            l1pPingBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            fm, fn, 0, 0, 0, 0);
    }

    SET_FLAG(MTE1, MTE3, PingFlag);
    SET_FLAG(MTE1, M, PingFlag);
    WAIT_FLAG(MTE1, M, PingFlag);
    WAIT_FLAG(MTE1, M, PingFlag + 2);
    WAIT_FLAG(V, M, PingFlag);

    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0aBufTensor.ReinterpretCast<half>()[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0bBufTensor.ReinterpretCast<half>()[PingFlag * L0AB_HALF_BUF_SIZE_I],
        mActual, fk, n0Actual, 1);

    SET_FLAG(M, MTE1, PingFlag);
    SET_FLAG(M, MTE1, PingFlag + 2);
    SET_FLAG(M, V, PingFlag);
    if (initKVE) {
        SET_FLAG(MTE1, MTE2, PingFlag);
        if (n1Actual == 0) {
            SET_FLAG(MTE1, MTE2, PongFlag);
        }
    }
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::CalculatePVPong(
    const uint32_t fm, const uint32_t bn, const uint32_t fk,
    const uint32_t mActual, const uint32_t n1Actual,
    const uint32_t initKVE,
    AscendC::LocalTensor<half> &l1pPingBufTensor,
    AscendC::LocalTensor<half> &l1pPongBufTensor)
{
    WAIT_FLAG(MTE3, MTE1, PongFlag);
    WAIT_FLAG(M, MTE1, PongFlag);
    if (mActual == 1) {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1pPongBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            0, 1, 0, 1, 0, 0);
    } else {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::NZ, DataFormatT::ZZ>(
            l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1pPingBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            fm, bn, 0, 0, 0, 0);
    }
    SET_FLAG(MTE1, MTE3, PongFlag);
    SET_FLAG(MTE1, M, PongFlag);
    WAIT_FLAG(MTE1, M, PongFlag);
    WAIT_FLAG(MTE1, M, PongFlag + 2);
    WAIT_FLAG(V, M, PongFlag);

    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
        l0aBufTensor.ReinterpretCast<half>()[PongFlag * L0AB_HALF_BUF_SIZE_I],
        l0bBufTensor.ReinterpretCast<half>()[PongFlag * L0AB_HALF_BUF_SIZE_I],
        mActual, fk, n1Actual, 1);

    SET_FLAG(M, MTE1, PongFlag);
    SET_FLAG(M, MTE1, PongFlag + 2);
    SET_FLAG(M, V, PongFlag);
    if (initKVE) {
        SET_FLAG(MTE1, MTE2, PongFlag);
    }
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::MovePVResultsToUb(
    const uint32_t fm, const uint32_t fk,
    const uint32_t mActual, const uint32_t oSize,
    const uint32_t n1Actual)
{
    WAIT_FLAG(M, V, PingFlag);
    l0c_to_ub<ArchType::ASCEND_V200, float, float>(
        lsUbufTensor.ReinterpretCast<float>(),
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        1, oSize / CUBE_MATRIX_SIZE_I, 0, 0);
    PIPE_BARRIER(V);

    for (uint32_t fractalBlkIdx = 0; fractalBlkIdx < fk / BLOCK_SIZE_I; fractalBlkIdx++) {
        ub_to_ub<ArchType::ASCEND_V200, float>(
            loUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * BLOCK_SIZE_I],
            lsUbufTensor.ReinterpretCast<float>()[fractalBlkIdx * fm * BLOCK_SIZE_I],
            0,
            mActual,
            2,
            0,
            2 * fk / BLOCK_SIZE_I - 2
        );
    }
    PIPE_BARRIER(V);

    if (n1Actual != 0) {
        WAIT_FLAG(M, V, PongFlag);
        l0c_to_ub<ArchType::ASCEND_V200, float, float>(
            lsUbufTensor.ReinterpretCast<float>()[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            1, oSize / CUBE_MATRIX_SIZE_I, 0, 0);
        PIPE_BARRIER(V);

        for (uint32_t fractalBlkIdx = 0; fractalBlkIdx < fk / BLOCK_SIZE_I; fractalBlkIdx++) {
            ub_to_ub<ArchType::ASCEND_V200, float>(
                loUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * BLOCK_SIZE_I],
                lsUbufTensor.ReinterpretCast<float>()[
                    PongFlag * LOCAL_STORAGE_BUFFER_SIZE + fractalBlkIdx * fm * BLOCK_SIZE_I],
                0,
                mActual,
                2,
                0,
                2 * fk / BLOCK_SIZE_I - 2
            );
        }
        PIPE_BARRIER(V);
    }
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::UpdateOnlineSoftmaxPing(
    const uint32_t fm, const uint32_t fk,
    const uint32_t mActual, const uint32_t kAlignVector,
    const uint32_t oSizeAlignFloat,
    const uint32_t glUOffset, const uint32_t goUOffset,
    uint32_t &initGgO)
{
    if (initGgO == 0) {
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            dmUbufTensor[PingFlag * UB_HALF_LINE_SIZE_I],
            1, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            tvUbufTensor.ReinterpretCast<float>(),
            1, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        SetMask(2 * fm);
        mul_v<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            tvUbufTensor.ReinterpretCast<float>(),
            glUbufTensor[glUOffset],
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            glUbufTensor[glUOffset],
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        ExpandToBlockHalf(tvUbufTensor, dmUbufTensor[PingFlag * UB_HALF_LINE_SIZE_I], 2 * fm);

        // (fm, 16) -> (fm, fk)
        // 往后放，这样half->float可以原地转
        if (fk < VECTOR_SIZE_I) {
            SetMask(fk % VECTOR_SIZE_I);
        }
        for (uint32_t mIdx = 0; mIdx < mActual; mIdx++) {
            // *2是因为max结果有value和idx两部分
            adds_v<ArchType::ASCEND_V200, half>(
                tvUbufTensor[L0AB_HALF_BUF_SIZE_I - fm * fk + mIdx * fk],
                tvUbufTensor[mIdx * BLOCK_SIZE_I * 2],
                0.0, kAlignVector, 1, 0, 8, 0);
            PIPE_BARRIER(V);
        }
        if (fk < VECTOR_SIZE_I) {
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        }

        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            tvUbufTensor[L0AB_HALF_BUF_SIZE_I - fm * fk],
            oSizeAlignFloat, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            tvUbufTensor.ReinterpretCast<float>(),
            oSizeAlignFloat, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        if (vmPingPongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingPongFlag = 0;
        }
        mul_v<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            goUbufTensor[goUOffset],
            tvUbufTensor.ReinterpretCast<float>(),
            oSizeAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            goUbufTensor[goUOffset],
            loUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            oSizeAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            0, 1, 2 * fm / BLOCK_SIZE_FLOAT, 0, 0);
        PIPE_BARRIER(V);
        if (vmPingPongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingPongFlag = 0;
        }
        ub_to_ub<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            loUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            0, 1, fm * fk / BLOCK_SIZE_FLOAT, 0, 0);
        PIPE_BARRIER(V);
    }
    PIPE_BARRIER(V);
    initGgO = 0;
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::UpdateOnlineSoftmaxPong(
    const uint32_t fm, const uint32_t fk,
    const uint32_t mActual, const uint32_t kAlignVector,
    const uint32_t oSizeAlignFloat,
    const uint32_t n1Actual,
    const uint32_t glUOffset, const uint32_t goUOffset)
{
    conv_v<ArchType::ASCEND_V200, half, float>(
        tvUbufTensor.ReinterpretCast<float>(),
        dmUbufTensor[PongFlag * UB_HALF_LINE_SIZE_I],
        1, 1, 1, uint16_t(8), uint16_t(4));
    PIPE_BARRIER(V);
    exp_v<ArchType::ASCEND_V200, float>(
        tvUbufTensor.ReinterpretCast<float>(),
        tvUbufTensor.ReinterpretCast<float>(),
        1, 1, 1, uint16_t(8), uint16_t(8));
    PIPE_BARRIER(V);
    SetMask(2 * fm);
    mul_v<ArchType::ASCEND_V200, float>(
        glUbufTensor[glUOffset],
        tvUbufTensor.ReinterpretCast<float>(),
        glUbufTensor[glUOffset],
        1, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
    add_v<ArchType::ASCEND_V200, float>(
        glUbufTensor[glUOffset],
        glUbufTensor[glUOffset],
        llUbufTensor[PongFlag * UB_FLOAT_LINE_SIZE_I],
        1, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
    SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    ExpandToBlockHalf(tvUbufTensor, dmUbufTensor[PongFlag * UB_HALF_LINE_SIZE_I], 2 * fm);

    if (fk < VECTOR_SIZE_I) {
        SetMask(fk % VECTOR_SIZE_I);
    }
    for (uint32_t mIdx = 0; mIdx < mActual; mIdx++) {
        adds_v<ArchType::ASCEND_V200, half>(
            tvUbufTensor[L0AB_HALF_BUF_SIZE_I - fm * fk + mIdx * fk],
            tvUbufTensor[mIdx * BLOCK_SIZE_I * 2],
            0.0, kAlignVector, 1, 0, 8, 0);
        PIPE_BARRIER(V);
    }
    if (fk < VECTOR_SIZE_I) {
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    }

    conv_v<ArchType::ASCEND_V200, half, float>(
        tvUbufTensor.ReinterpretCast<float>(),
        tvUbufTensor[L0AB_HALF_BUF_SIZE_I - fm * fk],
        oSizeAlignFloat, 1, 1, uint16_t(8), uint16_t(4));
    PIPE_BARRIER(V);
    exp_v<ArchType::ASCEND_V200, float>(
        tvUbufTensor.ReinterpretCast<float>(),
        tvUbufTensor.ReinterpretCast<float>(),
        oSizeAlignFloat, 1, 1, uint16_t(8), uint16_t(8));
    PIPE_BARRIER(V);

    if (vmPingPongFlag == 1) {
        WAIT_FLAG(MTE3, V, EVENT_ID2);
        vmPingPongFlag = 0;
    }

    mul_v<ArchType::ASCEND_V200, float>(
        goUbufTensor[goUOffset],
        goUbufTensor[goUOffset],
        tvUbufTensor.ReinterpretCast<float>(),
        oSizeAlignFloat, 1, 1, 1, 8, 8, 8);

    PIPE_BARRIER(V);
    add_v<ArchType::ASCEND_V200, float>(
        goUbufTensor[goUOffset],
        goUbufTensor[goUOffset],
        loUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
        oSizeAlignFloat, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
    SET_FLAG(V, M, PongFlag);
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::NormalizeAndStoreOutput(
    const uint32_t fm, const uint32_t fk,
    const uint32_t mActual, const uint32_t oSizeAlignFloat,
    const uint32_t glUOffset, const uint32_t goUOffset,
    const uint32_t wrapO)
{
    if (wrapO == 1) {
        SetMask(fm * 2);
        conv_v<ArchType::ASCEND_V200, float, half>(
            glUbufTensor[glUOffset].ReinterpretCast<half>(),
            glUbufTensor[glUOffset],
            1, 1, 1, uint16_t(4), uint16_t(8));
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        conv_v<ArchType::ASCEND_V200, float, half>(
            goUbufTensor[goUOffset].ReinterpretCast<half>(),
            goUbufTensor[goUOffset],
            oSizeAlignFloat, 1, 1, uint16_t(4), uint16_t(8));
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbufTensor, glUbufTensor[glUOffset].ReinterpretCast<half>(), 2 * fm);

        SetVectorMask<int8_t>(0x0, ((long)1 << (16)) - 1);

        for (int32_t vdivIdx = 0; vdivIdx < (fk / BLOCK_SIZE_I); ++vdivIdx) { // Oi / li
            div_v<ArchType::ASCEND_V200, half>(
                goUbufTensor[goUOffset].ReinterpretCast<half>()[vdivIdx * BLOCK_SIZE_I],
                goUbufTensor[goUOffset].ReinterpretCast<half>()[vdivIdx * BLOCK_SIZE_I],
                tvUbufTensor,
                mActual,
                1,
                1,
                1,
                fk / BLOCK_SIZE_I,
                fk / BLOCK_SIZE_I,
                2);
            PIPE_BARRIER(V);
        }
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, EVENT_ID2);
        WAIT_FLAG(V, MTE3, EVENT_ID2);

        ub_to_gm<ArchType::ASCEND_V200, half>(
                gmDstOTensor[(int64_t)dstoOffset],
                goUbufTensor[goUOffset].ReinterpretCast<half>(),
                0,
                mActual,
                fk / BLOCK_SIZE_I,
                0,
                (qHeadNum - 1) * fk / BLOCK_SIZE_I);

        if (vmPingPongFlag == 0) {
            SET_FLAG(MTE3, V, EVENT_ID2);
            vmPingPongFlag = 1;
        }
    }
}

template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::ProcessKvBlockPair(
    // fm      : 对齐后的 Q 块长度，mActual 向上对齐到 16。用于分配缓冲区大小和循环边界。
    const uint32_t fm,
    // fn      : 对齐后的第一个 KV 块长度，n0Actual 向上对齐到 16。定义第一块 QK^T 矩阵的列数。
    const uint32_t fn,
    // fk      : 对齐后的头维度（head_dim），向上对齐到 16。决定输出矩阵的列数。
    const uint32_t fk,
    // bn      : 对齐后的第二个 KV 块长度，n1Actual 向上对齐到 16。当 n1Actual > 0 时有效。
    const uint32_t bn,
    // mActual : 当前 Q 块的真实 token 数量（无 padding），控制所有行方向向量操作的实际循环次数。
    const uint32_t mActual,
    // n0Actual: 第一个 KV 块的真实 token 数量。用于计算矩阵乘法的 K 范围以及 softmax 的有效列数。
    const uint32_t n0Actual,
    // n1Actual: 第二个 KV 块的真实 token 数量。如果当前迭代不存在第二块，则传入 0。
    const uint32_t n1Actual,
    // maskType: Attention Mask 开关。非 0 时启用 mask 加法，将预先加载的 mask 加到 score 上。
    const uint32_t maskType,
    // initKVE : 结束标志。当该 KV 块被共享组内最后一个 Q 头处理完毕时置 1，允许释放 KV 缓冲区。
    const uint32_t initKVE,
    // headOffset: 当前 Q 头在 GQA 组内的偏移（0 ~ groupNum-1）。用于计算 Q、mask、输出等 UB 偏移地址。
    const uint32_t headOffset,
    // initKV   : 是否需要从 GM 加载 K/V 到 L1。组内第一个 Q 头为 1（加载），后续 Q 头复用为 0。
    const uint32_t initKV,
    // localTor : 局部缩放因子（通常为 1/sqrt(d)），当 scaleType 非 0 时使用。
    half localTor,
    // scaleType: 缩放选择标志。0 使用全局 tor；非 0 使用传入的 localTor。
    const uint32_t scaleType)
{
    if (scaleType == 0) {
        localTor = tor;
    }

    AscendC::LocalTensor<half> l1kPingBufTensor =
        l1kBufAddrTensor.ReinterpretCast<uint8_t>()[PingFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1kPongBufTensor =
        l1kBufAddrTensor.ReinterpretCast<uint8_t>()[PongFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1vPingBufTensor =
        l1vBufAddrTensor.ReinterpretCast<uint8_t>()[PingFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1vPongBufTensor =
        l1vBufAddrTensor.ReinterpretCast<uint8_t>()[PongFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1pPingBufTensor =
        l1pBufAddrTensor.ReinterpretCast<uint8_t>()[PingFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1pPongBufTensor =
        l1pBufAddrTensor.ReinterpretCast<uint8_t>()[PongFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();

    uint32_t oSize = fm * fk;
    uint32_t kAlignFloat = (fk + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;
    uint32_t oSizeAlignFloat = (oSize + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;
    uint32_t kAlignVector = (fk + VECTOR_SIZE_I - 1) / VECTOR_SIZE_I;
    uint32_t n0AlignVector = (fn + VECTOR_SIZE_I - 1) / VECTOR_SIZE_I;
    uint32_t n1AlignVector = (bn + VECTOR_SIZE_I - 1) / VECTOR_SIZE_I;
    uint32_t n0AlignBlock = (fn + BLOCK_SIZE_I - 1) / BLOCK_SIZE_I;
    uint32_t n1AlignBlock = (bn + BLOCK_SIZE_I - 1) / BLOCK_SIZE_I;
    uint32_t initGgDm = (initG == 1) ? 1 : 0;
    uint32_t initGgO = (initG == 1) ? 1 : 0;
    uint32_t initKVG = (initG && initKV) ? 1 : 0; // synchronization for head loop start
    uint32_t qOffset = headOffset * fm * FLOAT_VECTOR_SIZE_I * kAlignFloat;
    uint32_t gmUOffset = headOffset * fm * BLOCK_SIZE_I;
    uint32_t glUOffset = gmUOffset;
    uint32_t goUOffset = qOffset;

    uint32_t pSize = fm * fn;
    uint32_t pSizeB = fm * bn;
    uint32_t pSizeAlignFloat = (pSize + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;
    uint32_t pSizeBAlignFloat = (pSizeB + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;

    // 1. ################ Bmm1 Ping Start #######################
    LoadQueryAndAttentionMask(fm, fn, fk, bn, mActual, n1Actual, maskType, initGgO, initKVG, qOffset);

    WAIT_FLAG(M, MTE1, PingFlag);
    MoveQKPingToL0(fm, fn, fk, mActual, initGgO, initKV, qOffset, l1kPingBufTensor);

    // 2. ################ Bmm1 Pong Starts #######################
    // 2.1 ################ QK Pong PRELOAD ################
    if (n1Actual != 0) {
        MoveQKPongToL0(fm, bn, fk, mActual, n1Actual, initGgO, initKV, qOffset, l1kPongBufTensor);
    }
    // 1.2 ################ Bmm1 Ping + V PRELOAD ################
    CalculateQKPingAndPrepareV(fm, fn, fk, mActual, n0Actual, initKV, l1vPingBufTensor);

    // 1. ################ Bmm1 Ping Ends #######################
    // 2.2 ################ Bmm1 Pong + V PRELOAD ################
    if (n1Actual != 0) {
        CalculateQKPongAndPrepareV(fm, bn, fk, mActual, n1Actual, initKV, l1vPongBufTensor);
    }
    // 2. ################ Bmm1 Pong Ends #######################
    SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    // 3. ################ Softmax Ping Starts #######################
    CalculateSoftmaxPing(fm, fn, mActual, n0Actual, maskType, localTor,
                        n0AlignVector, n0AlignBlock, pSize, pSizeAlignFloat,
                        gmUOffset, initGgDm, l1pPingBufTensor);
    // 3. ################ Softmax Ping Ends #######################
    // 4. ################ Softmax Pong Starts #######################
    if (n1Actual != 0) {
        CalculateSoftmaxPong(fm, bn, mActual, n1Actual, maskType, localTor,
                            n1AlignVector, n1AlignBlock, pSizeB, pSizeBAlignFloat,
                            gmUOffset, l1pPingBufTensor, l1pPongBufTensor);
    }
    // 4. ################ Softmax Pong Ends #######################
    // 5. ################ Bmm2 Ping Starts #######################
    CalculatePVPing(fm, fn, fk, mActual, n0Actual, initKVE, n1Actual, l1pPingBufTensor);
    // 5. ################ Bmm2 Ping Ends #######################
    // 6. ################ Bmm2 Pong Starts #######################
    if (n1Actual != 0) {
        CalculatePVPong(fm, bn, fk, mActual, n1Actual, initKVE, l1pPingBufTensor, l1pPongBufTensor);
    }
    // 6. ################ Bmm2 Pong Ends #######################
    // 7. ################ Move PV Results Starts #####################
    MovePVResultsToUb(fm, fk, mActual, oSize, n1Actual);
    // 7. ################ Move PV Results Ends #######################
    // 8. ################ Update Ping Starts #######################
    UpdateOnlineSoftmaxPing(fm, fk, mActual, kAlignVector, oSizeAlignFloat, glUOffset, goUOffset, initGgO);
    // 8. ################ Update Ping Ends #######################
    // 9. ################ Update Pong Starts #######################
    if (n1Actual != 0) {
        UpdateOnlineSoftmaxPong(fm, fk, mActual, kAlignVector, oSizeAlignFloat, n1Actual, glUOffset, goUOffset);
    }
    SET_FLAG(V, M, PingFlag);
    // 9. ################ Update Pong Ends #######################
    // 10. ################ Line Output Starts #####################
    NormalizeAndStoreOutput(fm, fk, mActual, oSizeAlignFloat, glUOffset, goUOffset, wrapO);
}

template <typename IFAT>
class PagedAttentionDecoderMask {
public:
    __aicore__ inline PagedAttentionDecoderMask(){};
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *blockTable,  __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *workspace, const typename IFAT::TilingType *__restrict tiling);
    __aicore__ inline void Process();

protected:
    const typename IFAT::TilingType *__restrict tilingData_ = nullptr;

    GlobalTensor<int64_t> promptLensGmTensor_;
    GlobalTensor<int64_t> contextLensGmTensor_;
    GlobalTensor<int32_t> blockTablesGmTensor_;
    
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxNumBlocksPerQuery_ = 0;
    half tor_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t scaleType_ = 0;
    uint32_t headSplit_ = 1;
    uint32_t maskType_ = 0;
    int64_t headMaskStride_ = 0;
    int64_t batchMaskStride_ = 0;
    uint32_t maxTokensQ_ = 1;
    uint32_t maskKvLen_ = 0;

    uint32_t seqStepQ_ = 0;
    uint32_t startBatchId_ = 0;
    uint32_t qBlkNumByStartBatch_ = 0;
    uint32_t endBatchId_ = 0;

    uint32_t startTaskId_ = 0;
    uint32_t endTaskId_ = 0;
    
    __gm__ uint8_t *query_;
    __gm__ uint8_t *key_;
    __gm__ uint8_t *value_;
    __gm__ uint8_t *attenMask_;
    __gm__ uint8_t *attentionOut_;

    template <typename NumT> __aicore__ inline NumT Align(NumT num, NumT rnd) const
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
    }

    // ==========================================
    // Offset Calculators
    // ==========================================
    __aicore__ inline uint32_t GetTndQueryPrefix(uint32_t batchId) const
    {
        uint32_t prefix = 0;
        for (uint32_t b = 0; b < batchId; ++b) {
            prefix += static_cast<uint32_t>(promptLensGmTensor_.GetValue(b));
        }
        return prefix;
    }

    __aicore__ inline uint64_t GetQOffsetTndVarlen(uint32_t batchId, uint32_t qSeqBlockIdx, uint32_t headId) const
    {
        uint32_t tokenStart = GetTndQueryPrefix(batchId) + qSeqBlockIdx * seqStepQ_;
        return static_cast<uint64_t>(tokenStart) * numHeads_ * embeddingSize_ +
               static_cast<uint64_t>(headId) * embeddingSize_;
    }

    __aicore__ inline uint64_t GetQOffset(uint32_t curBatch, uint32_t qSeqBlockIdx, uint32_t headId) const
    {
        return static_cast<uint64_t>(curBatch) * embeddingSize_ * numHeads_ * maxTokensQ_ +
               static_cast<uint64_t>(qSeqBlockIdx) * seqStepQ_ * embeddingSize_ * numHeads_ +
               static_cast<uint64_t>(headId) * embeddingSize_;
    }

    __aicore__ inline uint64_t GetMaskOffset(uint32_t curBatch, uint32_t qSeqBlockIdx, uint32_t nIdx, uint32_t headId) const
    {
        return static_cast<uint64_t>(curBatch) * batchMaskStride_ +
               static_cast<uint64_t>(qSeqBlockIdx) * seqStepQ_ * maskKvLen_ +
               static_cast<uint64_t>(nIdx) * blockSize_ +
               static_cast<uint64_t>(headId) * headMaskStride_;
    }

    __aicore__ inline uint64_t GetBlockTableIndex(uint32_t curBatch, uint32_t nIdx) const
    {
        return static_cast<uint64_t>(curBatch) * maxNumBlocksPerQuery_ + nIdx;
    }

    // ==========================================
    // Core Methods
    // ==========================================
    __aicore__ inline void LoadTilingParameters();
    __aicore__ inline void LocateTaskRangeStart();
    
    __aicore__ inline bool AdvanceToNextBatch(uint32_t& curBatch, uint32_t& qBlkNumByCurBatch, uint32_t& qBlkNumCurBatch,
                                        uint32_t& promptLenCurBatch, uint32_t& contextLenCurBatch, uint32_t& qSeqBlockNum) const 
    {
        curBatch += 1;
        if (curBatch >= numTokens_) {
            return false;
        }
        qBlkNumByCurBatch += qBlkNumCurBatch;
        promptLenCurBatch = static_cast<uint32_t>(promptLensGmTensor_.GetValue(curBatch));
        contextLenCurBatch = static_cast<uint32_t>(contextLensGmTensor_.GetValue(curBatch));
        qSeqBlockNum = (promptLenCurBatch + seqStepQ_ - 1) / seqStepQ_;
        qBlkNumCurBatch = qSeqBlockNum * numHeads_;
        return true;
    }

    __aicore__ inline void ProcessAssignedTasks(PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT> &pa);

    __aicore__ inline void InitializePipelineEvents()
    {
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(V, M, EVENT_ID0);
        SET_FLAG(V, M, EVENT_ID1);
        SET_FLAG(V, M, EVENT_ID2);
        SET_FLAG(V, M, EVENT_ID3);
        SET_FLAG(V, MTE1, EVENT_ID0);
        SET_FLAG(V, MTE1, EVENT_ID1);
        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(MTE3, V, EVENT_ID1);
        SET_FLAG(MTE3, V, EVENT_ID2);
        SET_FLAG(MTE3, V, EVENT_ID3);
        SET_FLAG(MTE1, MTE3, EVENT_ID0);
        SET_FLAG(MTE1, MTE3, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
    #if __CCE_AICORE__ == 100
    #else
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(MTE1, MTE2, EVENT_ID6);
        SET_FLAG(MTE1, MTE2, EVENT_ID7);
    #endif
    }

    __aicore__ inline void FinalizePipelineEvents()
    {
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    #if __CCE_AICORE__ == 100
    #else
        WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
    #endif
        WAIT_FLAG(V, MTE1, EVENT_ID0);
        WAIT_FLAG(V, MTE1, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID1);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID1);
        WAIT_FLAG(MTE3, V, EVENT_ID2);
        WAIT_FLAG(MTE3, V, EVENT_ID3);
        WAIT_FLAG(V, M, EVENT_ID0);
        WAIT_FLAG(V, M, EVENT_ID1);
        WAIT_FLAG(V, M, EVENT_ID2);
        WAIT_FLAG(V, M, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        PIPE_BARRIER(ALL);
    }
};

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const typename IFAT::TilingType *__restrict tiling)
{
    promptLensGmTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actualSeqLengthsQ));
    contextLensGmTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actualSeqLengths));
    blockTablesGmTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
    ListTensorDesc keyListTensorDesc((__gm__ void *)key);
    ListTensorDesc valueListTensorDesc((__gm__ void *)value);
    
    tilingData_ = tiling;
    query_ = query;
    key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    attenMask_ = attenMask;
    attentionOut_ = attentionOut;

    LoadTilingParameters();
}

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::Process()
{
    SetMaskNorm();
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    SetLoadDataPaddingValue<uint64_t>(uint16_t(0));
    SetAtomicNone();

    PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT> pa(
        query_, key_, value_, attenMask_, attentionOut_, tor_, blockSize_);

    InitializePipelineEvents();
    ProcessAssignedTasks(pa);
    FinalizePipelineEvents();
}

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::LoadTilingParameters()
{
    uint32_t tmp_block_idx = GetBlockIdx();
    startTaskId_ = tilingData_->tilingPerCore.startTaskId[tmp_block_idx];
    endTaskId_ = tilingData_->tilingPerCore.endTaskId[tmp_block_idx];
    numTokens_ = tilingData_->tilingBase.batchSize;
    numHeads_ = tilingData_->tilingBase.qHeadNum;
    embeddingSize_ = tilingData_->tilingBase.headSize;
    blockSize_ = tilingData_->tilingBase.blockSize;
    maxNumBlocksPerQuery_ = tilingData_->tilingBase.maxBlockNumPerBatch;
    tor_ = tilingData_->tilingBase.scaleValue;
    kvHeads_ = tilingData_->tilingBase.kvHeadNum;
    maskType_ = tilingData_->tilingBase.attenMaskFlag;
    headMaskStride_ = tilingData_->tilingPerCore.maskHeadStride;
    batchMaskStride_ = tilingData_->tilingPerCore.maskBatchStride;
    maxTokensQ_ = tilingData_->tilingPerCore.qTokens;
    maskKvLen_ = tilingData_->tilingPerCore.maskKvLen;
    seqStepQ_ = tilingData_->tilingBase.querySeqStep;
    groupNum_ = numHeads_ / kvHeads_;

    LocateTaskRangeStart();
}

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::LocateTaskRangeStart()
{
    bool startInit = false;
    uint32_t qBlkNumByCurBatch = 0;

    for (uint32_t bId = 0; bId < numTokens_; bId++) {
        uint32_t qSeqBlock = ((uint32_t)(promptLensGmTensor_.GetValue(bId)) + seqStepQ_ - 1) / seqStepQ_;
        uint32_t qBlkNumCurBatch = qSeqBlock * numHeads_;
        if (!startInit && (qBlkNumByCurBatch + qBlkNumCurBatch > startTaskId_)) {
            startBatchId_ = bId;
            startInit = true;
            qBlkNumByStartBatch_ = qBlkNumByCurBatch;
        }
        if (qBlkNumByCurBatch + qBlkNumCurBatch >= endTaskId_) {
            endBatchId_ = bId;
            break;
        }
        qBlkNumByCurBatch += qBlkNumCurBatch;
    }
}

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::ProcessAssignedTasks(
    PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT> &pa)
{
    uint32_t curBatch = startBatchId_;
    uint32_t qBlkNumByCurBatch = qBlkNumByStartBatch_;
    uint32_t promptLenCurBatch = static_cast<uint32_t>(promptLensGmTensor_.GetValue(curBatch));
    uint32_t contextLenCurBatch = static_cast<uint32_t>(contextLensGmTensor_.GetValue(curBatch));
    uint32_t qSeqBlockNum = (promptLenCurBatch + seqStepQ_ - 1) / seqStepQ_;
    uint32_t qBlkNumCurBatch = qSeqBlockNum * numHeads_;
    
    uint64_t strideKV = blockSize_ * embeddingSize_;

    // 跳过开头的空 Batch
    while (curBatch < numTokens_ && qBlkNumCurBatch == 0) {
        if (!AdvanceToNextBatch(curBatch, qBlkNumByCurBatch, qBlkNumCurBatch, promptLenCurBatch, contextLenCurBatch, qSeqBlockNum)) {
            return;
        }
    }

    for (uint32_t taskId = startTaskId_; taskId < endTaskId_; taskId++) {
        while (taskId >= qBlkNumByCurBatch + qBlkNumCurBatch) {
            if (!AdvanceToNextBatch(curBatch, qBlkNumByCurBatch, qBlkNumCurBatch, promptLenCurBatch, contextLenCurBatch, qSeqBlockNum)) {
                return;
            }
            // 处理连续的空 Batch
            while (curBatch < numTokens_ && qBlkNumCurBatch == 0) {
                if (!AdvanceToNextBatch(curBatch, qBlkNumByCurBatch, qBlkNumCurBatch, promptLenCurBatch, contextLenCurBatch, qSeqBlockNum)) {
                    return;
                }
            }
        }

        uint32_t headId = (taskId - qBlkNumByCurBatch) / qSeqBlockNum;
        uint32_t qSeqBlockIdx = (taskId - qBlkNumByCurBatch) % qSeqBlockNum;
        uint32_t repeatLen = min(headSplit_, min(endTaskId_ - taskId, groupNum_ - headId % groupNum_));
        uint32_t nLoop = (contextLenCurBatch + blockSize_ - 1) / blockSize_;
        uint32_t tail = contextLenCurBatch % blockSize_ == 0 ? blockSize_ : contextLenCurBatch % blockSize_;
        uint32_t mActual = (qSeqBlockIdx == qSeqBlockNum - 1) ? (promptLenCurBatch - qSeqBlockIdx * seqStepQ_) : seqStepQ_;
        uint32_t roundM = Align<uint32_t>(mActual, 16);
        uint32_t roundK = Align<uint32_t>(embeddingSize_, 16);
        uint64_t kvHeadOffset = (headId / groupNum_) * strideKV;
        half localTor = 0;

        for (uint32_t nIdx = 0; nIdx < nLoop; nIdx += 2) {
            uint64_t blockTableIdx0 = GetBlockTableIndex(curBatch, nIdx);
            uint64_t numBlocksId0 = static_cast<uint64_t>(blockTablesGmTensor_.GetValue(blockTableIdx0));
            uint64_t kvOffset0 = numBlocksId0 * blockSize_ * kvHeads_ * embeddingSize_ + kvHeadOffset;

            uint64_t numBlocksId1 = 0;
            uint64_t kvOffset1 = 0;
            if ((nIdx + 1) != nLoop) {
                uint64_t blockTableIdx1 = GetBlockTableIndex(curBatch, nIdx + 1);
                numBlocksId1 = static_cast<uint64_t>(blockTablesGmTensor_.GetValue(blockTableIdx1));
                kvOffset1 = numBlocksId1 * blockSize_ * kvHeads_ * embeddingSize_ + kvHeadOffset;
            }

            // 计算causal边界，用于计算compress mask场景下每一个q所对应的有效长度。
            int32_t historyKvLen = static_cast<int32_t>(contextLenCurBatch) - static_cast<int32_t>(promptLenCurBatch);
            int32_t qGlobalIdxBase = static_cast<int32_t>(qSeqBlockIdx * seqStepQ_);
            int32_t kvStart0 = static_cast<int32_t>(nIdx * blockSize_);
            
            int32_t baseCausalOffset0 = historyKvLen + qGlobalIdxBase - kvStart0 + 1;
            int32_t baseCausalOffset1 = baseCausalOffset0 - static_cast<int32_t>(blockSize_);

            int32_t cmRowPing = baseCausalOffset0 + seqStepQ_ - 1;
            cmRowPing = (cmRowPing < 0) ? 0 : ((cmRowPing > blockSize_ + seqStepQ_ - 1) ? blockSize_ + seqStepQ_ - 1 : cmRowPing);
            int32_t cmRowPong = baseCausalOffset1 + seqStepQ_ - 1;
            cmRowPong = (cmRowPong < 0) ? 0 : ((cmRowPong > blockSize_ + seqStepQ_ - 1) ? blockSize_ + seqStepQ_ - 1 : cmRowPong);

            uint32_t warpO = (nIdx == (nLoop - 1) || (nIdx + 1) == (nLoop - 1)) ? 1 : 0;
            uint32_t initG = (nIdx == 0) ? 1 : 0;
            uint32_t n0Actual = (nIdx == nLoop - 1) ? tail : blockSize_;
            uint32_t n1Actual = ((nIdx + 1) == nLoop - 1) ? tail : blockSize_;
            uint32_t roundN0 = Align<uint32_t>(n0Actual, 16);
            uint32_t roundN1 = Align<uint32_t>(n1Actual, 16);
            if ((nIdx + 1) == nLoop) {
                n1Actual = 0;
            }

            uint64_t qOffset = 0;
            if constexpr (IFAT::layout == LAYOUT::TND) {
                qOffset = GetQOffsetTndVarlen(curBatch, qSeqBlockIdx, headId);
            } else {
                qOffset = GetQOffset(curBatch, qSeqBlockIdx, headId);
            }
            
            uint64_t maskOffset = GetMaskOffset(curBatch, qSeqBlockIdx, nIdx, headId);

            for (uint32_t headOffset = 0; headOffset < repeatLen; ++headOffset) {
                uint32_t initKV = (headOffset == 0) ? 1 : 0;
                uint32_t initKVE = (warpO && headOffset == repeatLen - 1) ? 1 : 0;

                // NOTE(Shengyi): 这里保持传正常mask的offset进去，兼容非压缩mask逻辑。若使用压缩mask特性，maskOffset不会被使用。
                pa.Init(qOffset, kvOffset0, kvOffset0, kvOffset1, kvOffset1, maskOffset, qOffset, initG, warpO,
                        maskKvLen_, numHeads_, cmRowPing, cmRowPong);
                pa.ProcessKvBlockPair(roundM, roundN0, roundK, roundN1, mActual, n0Actual, n1Actual, maskType_, initKVE,
                          headOffset, initKV, localTor, scaleType_);

                qOffset += embeddingSize_;
                maskOffset += headMaskStride_;
            }
        }
        taskId += repeatLen - 1;
    }
}
#endif // UNPAD_PAGED_ATTENTION_DECODER_H