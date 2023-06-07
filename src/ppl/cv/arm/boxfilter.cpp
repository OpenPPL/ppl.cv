// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/cv/arm/boxfilter.h"
#include "ppl/cv/arm/copymakeborder.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <limits.h>
#include <algorithm>
#include <vector>

namespace ppl::cv::arm {

template <typename T, typename ST>
struct RowSum {
    RowSum(int32_t _ksize)
    {
        ksize = _ksize;
    }

    void operator()(const T* src, ST* dst, int32_t width, int32_t cn)
    {
        const T* S = (const T*)src;
        ST* D = (ST*)dst;
        int32_t i = 0, k, ksz_cn = ksize * cn;

        width = (width - 1) * cn;
        for (k = 0; k < cn; k++, S++, D++) {
            ST s = 0;
            for (i = 0; i < ksz_cn; i += cn)
                s += S[i];
            D[0] = s;
            for (i = 0; i < width; i += cn) {
                s += S[i + ksz_cn] - S[i];
                D[i + cn] = s;
            }
        }
    }
    int32_t ksize;
};

static uint8_t saturate_cast(float32_t v)
{
    if (v > 255)
        return 255;
    else if (v < 0)
        return 0;
    return (uint8_t)v;
}

template <typename ST, typename T>
struct ColumnSum {
    ColumnSum(int32_t _ksize, float _scale)
    {
        ksize = _ksize;
        scale = _scale;
        sumCount = 0;
    }

    void reset()
    {
        sumCount = 0;
    }

    void operator()(const ST** src, T* dst, int32_t dststep, int32_t count, int32_t width)
    {
        int32_t i;
        ST* SUM;
        bool haveScale = scale != 1;
        if (width != (int32_t)sum.size()) {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if (sumCount == 0) {
            memset((void*)SUM, 0, width * sizeof(ST));

            for (; sumCount < ksize - 1; sumCount++, src++) {
                const ST* Sp = (const ST*)src[0];
                for (i = 0; i <= width - 2; i += 2) {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
                    SUM[i] = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++)
                    SUM[i] += Sp[i];
            }
        } else {
            // assert( sumCount == ksize-1 );
            src += ksize - 1;
        }

        for (; count--; src++) {
            const ST* Sp = (const ST*)src[0];
            const ST* Sm = (const ST*)src[1 - ksize];
            T* D = (T*)dst;
            if (haveScale) {
                for (i = 0; i <= width - 2; i += 2) {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
                    ST val0 = s0 * scale;
                    D[i] = (std::is_same<float32_t, T>::value) ? val0 : saturate_cast(val0);
                    ST val1 = s1 * scale;
                    D[i + 1] = (std::is_same<float32_t, T>::value) ? val1 : saturate_cast(val1);
                    s0 -= Sm[i];
                    s1 -= Sm[i + 1];
                    SUM[i] = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++) {
                    ST s0 = SUM[i] + Sp[i];
                    ST val = s0 * scale;
                    D[i] = (std::is_same<float32_t, T>::value) ? val : saturate_cast(val);
                    SUM[i] = s0 - Sm[i];
                }
            } else {
                for (i = 0; i <= width - 2; i += 2) {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
                    ST val0 = s0;
                    D[i] = (std::is_same<float32_t, T>::value) ? val0 : saturate_cast(val0);
                    ST val1 = s1;
                    D[i + 1] = (std::is_same<float32_t, T>::value) ? val1 : saturate_cast(val1);
                    s0 -= Sm[i];
                    s1 -= Sm[i + 1];
                    SUM[i] = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++) {
                    ST s0 = SUM[i] + Sp[i];
                    ST val = s0;
                    D[i] = (std::is_same<float32_t, T>::value) ? val : saturate_cast(val);
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }
    int32_t ksize;
    int32_t sumCount;
    float scale;
    std::vector<ST> sum;
};

template <int32_t cn>
void boxFilter_f(int32_t height,
                 int32_t width,
                 int32_t inWidthStride,
                 const float* inData,
                 int32_t ksize_x,
                 int32_t ksize_y,
                 bool normalize,
                 int32_t outWidthStride,
                 float* outData,
                 BorderType borderType,
                 float border_value = 0)
{
    int32_t radius_x = ksize_x / 2;
    int32_t radius_y = ksize_y / 2;

    int32_t bsrcHeight = height + 2 * radius_y;
    int32_t bsrcWidth = width + 2 * radius_x;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    float* bsrc_t = (float*)malloc((bsrcHeight + 1) * bsrcWidth * cn * sizeof(float));
    float* bsrc = bsrc_t + bsrcWidthStep;
    CopyMakeBorder<float, cn>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, borderType, border_value);

    RowSum<float, float> rowVecOp = RowSum<float, float>(ksize_x);
    for (int32_t i = 0; i < bsrcHeight; i++) {
        float* src = bsrc + i * bsrcWidthStep;
        float* dst = bsrc_t + i * bsrcWidthStep;
        rowVecOp.operator()(src, dst, width, cn);
    }
    const float** pReRowFilter = (const float**)malloc(bsrcHeight * sizeof(bsrc_t));
    for (int32_t i = 0; i < bsrcHeight; i++) {
        pReRowFilter[i] = bsrc_t + (i)*bsrcWidthStep;
    }

    ColumnSum<float, float> colVecOp = ColumnSum<float, float>(ksize_y, normalize ? 1. / (ksize_x * ksize_y) : 1);
    colVecOp.operator()((const float**)pReRowFilter, (float*)outData, outWidthStride, height, width * cn);

    free(bsrc_t);
    free(pReRowFilter);
    bsrc = NULL;
    pReRowFilter = NULL;
}

template <int32_t cn>
void boxFilter_b(int32_t height,
                 int32_t width,
                 int32_t inWidthStride,
                 const uint8_t* inData,
                 int32_t ksize_x,
                 int32_t ksize_y,
                 bool normalize,
                 int32_t outWidthStride,
                 uint8_t* outData,
                 BorderType borderType,
                 uint8_t border_value = 0)
{
    int32_t radius_x = ksize_x / 2;
    int32_t radius_y = ksize_y / 2;

    int32_t bsrcHeight = height + 2 * radius_y;
    int32_t bsrcWidth = width + 2 * radius_x;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    int32_t rfstep = width * cn;
    uint8_t* bsrc =
        (uint8_t*)malloc(bsrcHeight * bsrcWidth * cn * sizeof(uint8_t) + bsrcHeight * width * cn * sizeof(int32_t));
    CopyMakeBorder<uint8_t, cn>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, borderType, border_value);

    int32_t* resultRowFilter = (int32_t*)(bsrc + bsrcHeight * bsrcWidth * cn * sizeof(uint8_t));

    RowSum<uint8_t, int32_t> rowVecOp = RowSum<uint8_t, int32_t>(ksize_x);
    for (int32_t i = 0; i < bsrcHeight; i++) {
        uint8_t* src = bsrc + i * bsrcWidthStep;
        int32_t* dst = resultRowFilter + i * rfstep;
        rowVecOp.operator()(src, dst, width, cn);
    }

    ColumnSum<int32_t, uint8_t> colVecOp =
        ColumnSum<int32_t, uint8_t>(ksize_y, normalize ? 1. / (ksize_x * ksize_y) : 1);
    const int32_t** pReRowFilter = (const int32_t**)malloc(bsrcHeight * sizeof(resultRowFilter));
    for (int32_t i = 0; i < bsrcHeight; i++) {
        pReRowFilter[i] = resultRowFilter + (i)*rfstep;
    }
    colVecOp.operator()(pReRowFilter, outData, outWidthStride, height, width * cn);

    free(bsrc);
    free(pReRowFilter);
    bsrc = NULL;
    pReRowFilter = NULL;
}

template <>
::ppl::common::RetCode BoxFilter<float, 1>(int32_t height,
                                           int32_t width,
                                           int32_t inWidthStride,
                                           const float* inData,
                                           int32_t ksize_x,
                                           int32_t ksize_y,
                                           bool normalize,
                                           int32_t outWidthStride,
                                           float* outData,
                                           BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    boxFilter_f<1>(
        height, width, inWidthStride, inData, ksize_x, ksize_y, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<float, 3>(int32_t height,
                                           int32_t width,
                                           int32_t inWidthStride,
                                           const float* inData,
                                           int32_t ksize_x,
                                           int32_t ksize_y,
                                           bool normalize,
                                           int32_t outWidthStride,
                                           float* outData,
                                           BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    boxFilter_f<3>(
        height, width, inWidthStride, inData, ksize_x, ksize_y, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<float, 4>(int32_t height,
                                           int32_t width,
                                           int32_t inWidthStride,
                                           const float* inData,
                                           int32_t ksize_x,
                                           int32_t ksize_y,
                                           bool normalize,
                                           int32_t outWidthStride,
                                           float* outData,
                                           BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    boxFilter_f<4>(
        height, width, inWidthStride, inData, ksize_x, ksize_y, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<uint8_t, 1>(int32_t height,
                                             int32_t width,
                                             int32_t inWidthStride,
                                             const uint8_t* inData,
                                             int32_t ksize_x,
                                             int32_t ksize_y,
                                             bool normalize,
                                             int32_t outWidthStride,
                                             uint8_t* outData,
                                             BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    boxFilter_b<1>(
        height, width, inWidthStride, inData, ksize_x, ksize_y, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<uint8_t, 3>(int32_t height,
                                             int32_t width,
                                             int32_t inWidthStride,
                                             const uint8_t* inData,
                                             int32_t ksize_x,
                                             int32_t ksize_y,
                                             bool normalize,
                                             int32_t outWidthStride,
                                             uint8_t* outData,
                                             BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    boxFilter_b<3>(
        height, width, inWidthStride, inData, ksize_x, ksize_y, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<uint8_t, 4>(int32_t height,
                                             int32_t width,
                                             int32_t inWidthStride,
                                             const uint8_t* inData,
                                             int32_t ksize_x,
                                             int32_t ksize_y,
                                             bool normalize,
                                             int32_t outWidthStride,
                                             uint8_t* outData,
                                             BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    boxFilter_b<4>(
        height, width, inWidthStride, inData, ksize_x, ksize_y, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}

} // namespace ppl::cv::arm