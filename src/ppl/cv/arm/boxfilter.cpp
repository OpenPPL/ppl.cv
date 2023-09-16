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

#include "filter_engine.hpp"

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

        if( cn == 1 )
        {
            ST s = 0;
            for( i = 0; i < ksz_cn; i++ ) {
                s += (ST)S[i];
            }                
            D[0] = s;
            for( i = 0; i < width; i++ )
            {
                s += (ST)S[i + ksz_cn] - (ST)S[i];
                D[i+1] = s;
            }
        }
        else if( cn == 3 )
        {
            ST s0 = 0, s1 = 0, s2 = 0;
            for( i = 0; i < ksz_cn; i += 3 )
            {
                s0 += (ST)S[i];
                s1 += (ST)S[i+1];
                s2 += (ST)S[i+2];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            for( i = 0; i < width; i += 3 )
            {
                s0 += (ST)S[i + ksz_cn] - (ST)S[i];
                s1 += (ST)S[i + ksz_cn + 1] - (ST)S[i + 1];
                s2 += (ST)S[i + ksz_cn + 2] - (ST)S[i + 2];
                D[i+3] = s0;
                D[i+4] = s1;
                D[i+5] = s2;
            }
        }
        else if( cn == 4 )
        {
            ST s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for( i = 0; i < ksz_cn; i += 4 )
            {
                s0 += (ST)S[i];
                s1 += (ST)S[i+1];
                s2 += (ST)S[i+2];
                s3 += (ST)S[i+3];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            D[3] = s3;
            for( i = 0; i < width; i += 4 )
            {
                s0 += (ST)S[i + ksz_cn] - (ST)S[i];
                s1 += (ST)S[i + ksz_cn + 1] - (ST)S[i + 1];
                s2 += (ST)S[i + ksz_cn + 2] - (ST)S[i + 2];
                s3 += (ST)S[i + ksz_cn + 3] - (ST)S[i + 3];
                D[i+4] = s0;
                D[i+5] = s1;
                D[i+6] = s2;
                D[i+7] = s3;
            }
        }
        else {
            for( k = 0; k < cn; k++, S++, D++ )
            {
                ST s = 0;
                for( i = 0; i < ksz_cn; i += cn )
                    s += (ST)S[i];
                D[0] = s;
                for( i = 0; i < width; i += cn )
                {
                    s += (ST)S[i + ksz_cn] - (ST)S[i];
                    D[i+cn] = s;
                }
            }
        }
    }
    
    int32_t ksize;
};

static uint8_t saturate_cast(float val)
{
    int32_t v = roundf(val);
    if (v > 255)
        return 255;
    else if (v < 0)
        return 0;
    return (uint8_t)v;
}

template <typename ST, typename T>
struct ColumnSum {
    ColumnSum(int32_t _ksize, double _scale)
    {
        ksize = _ksize;
        scale = _scale;
        sumCount = 0;
    }

    void reset()
    {
        sumCount = 0;
    }

    void operator()(const ST* const* src, T* dst, int32_t dststep, int32_t count, int32_t width)
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
                    D[i] = s0 * scale;
                    D[i + 1] = s1 * scale;
                    s0 -= Sm[i];
                    s1 -= Sm[i + 1];
                    SUM[i] = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++) {
                    ST s0 = SUM[i] + Sp[i];
                    D[i] = s0 * scale;
                    SUM[i] = s0 - Sm[i];
                }
            } else {
                for (i = 0; i <= width - 2; i += 2) {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
                    D[i] = s0;
                    D[i + 1] = s1;
                    s0 -= Sm[i];
                    s1 -= Sm[i + 1];
                    SUM[i] = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++) {
                    ST s0 = SUM[i] + Sp[i];
                    D[i] = s0;
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }
    int32_t ksize;
    int32_t sumCount;
    double scale;
    std::vector<ST> sum;
};

template <>
struct ColumnSum<int32_t, uint8_t> {
    ColumnSum(int32_t _ksize, double _scale)
    {
        ksize = _ksize;
        scale = _scale;
        sumCount = 0;
    }

    void reset()
    {
        sumCount = 0;
    }

    void operator()(const int32_t* const* src, uint8_t* dst, int32_t dststep, int32_t count, int32_t width)
    {
        int32_t i;
        int32_t* SUM;
        bool haveScale = scale != 1;
        float _scale = scale;

        if (width != (int32_t)sum.size()) {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if (sumCount == 0) {
            memset((void*)SUM, 0, width * sizeof(int32_t));
            for (; sumCount < ksize - 1; sumCount++, src++) {
                const int32_t* Sp = (const int32_t*)src[0];
                i = 0;
                for (; i < width; i++)
                    SUM[i] += Sp[i];
            }
        } else {
            // assert( sumCount == ksize-1 );
            src += ksize - 1;
        }

        for (; count--; src++) {
            const int32_t* Sp = (const int32_t*)src[0];
            const int32_t* Sm = (const int32_t*)src[1 - ksize];
            uint8_t* D = (uint8_t*)dst;
            if (haveScale) {
                i = 0;
                for (; i < width; i++) {
                    int32_t s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast(s0 * _scale);
                    SUM[i] = s0 - Sm[i];
                }
            } else {
                i = 0;
                for (; i < width; i++) {
                    int32_t s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast(s0);
                    SUM[i] = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }
    int32_t ksize;
    double scale;
    int32_t sumCount;
    std::vector<int32_t> sum;
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
    RowSum<float, double> rowSum(ksize_x);
    ColumnSum<double, float> columnSum(ksize_y, normalize ? 1. / (ksize_x * ksize_y) : 1);
    SeparableFilterEngine<float, double, float, RowSum<float, double>, ColumnSum<double, float>> engine(
        height,
        width,
        cn,
        ksize_y,
        ksize_x,
        borderType,
        border_value,
        rowSum,
        columnSum
    );
    
    engine.process(inData, inWidthStride, outData, outWidthStride);
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
    RowSum<uint8_t, int32_t> rowSum(ksize_x);
    ColumnSum<int32_t, uint8_t> columnSum(ksize_y, normalize ? 1. / (ksize_x * ksize_y) : 1);
    SeparableFilterEngine<uint8_t, int32_t, uint8_t, RowSum<uint8_t, int32_t>, ColumnSum<int32_t, uint8_t>> engine(
        height,
        width,
        cn,
        ksize_y,
        ksize_x,
        borderType,
        border_value,
        rowSum,
        columnSum
    );
    
    engine.process(inData, inWidthStride, outData, outWidthStride);
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