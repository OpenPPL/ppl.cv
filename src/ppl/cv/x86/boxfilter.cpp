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

#include "ppl/cv/x86/boxfilter.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/copymakeborder.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/cv/x86/util.hpp"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <immintrin.h>
#include <algorithm>
#include <vector>

namespace ppl {
namespace cv {
namespace x86 {

template <typename T, typename ST>
struct RowSum {
    RowSum(int32_t _ksize)
    {
        ksize = _ksize;
    }

    void operator()(const T* src, ST* dst, int32_t width, int32_t cn)
    {
        const T* S = (const T*)src;
        ST* D      = (ST*)dst;
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
template <typename ST, typename T>
struct ColumnSum {
    ColumnSum(int32_t _ksize, float _scale)
    {
        ksize    = _ksize;
        scale    = _scale;
        sumCount = 0;
    }

    void reset()
    {
        sumCount = 0;
    }

    void operator()(const uint8_t** src, uint8_t* dst, int32_t dststep, int32_t count, int32_t width)
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
                    SUM[i]     = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++)
                    SUM[i] += Sp[i];
            }
        } else {
            //assert( sumCount == ksize-1 );
            src += ksize - 1;
        }

        for (; count--; src++) {
            const ST* Sp = (const ST*)src[0];
            const ST* Sm = (const ST*)src[1 - ksize];
            T* D         = (T*)dst;
            if (haveScale) {
                for (i = 0; i <= width - 2; i += 2) {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
                    D[i]     = s0 * scale;
                    D[i + 1] = s1 * scale;
                    s0 -= Sm[i];
                    s1 -= Sm[i + 1];
                    SUM[i]     = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++) {
                    ST s0  = SUM[i] + Sp[i];
                    D[i]   = s0 * scale;
                    SUM[i] = s0 - Sm[i];
                }
            } else {
                for (i = 0; i <= width - 2; i += 2) {
                    ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
                    D[i]     = s0;
                    D[i + 1] = s1;
                    s0 -= Sm[i];
                    s1 -= Sm[i + 1];
                    SUM[i]     = s0;
                    SUM[i + 1] = s1;
                }

                for (; i < width; i++) {
                    ST s0  = SUM[i] + Sp[i];
                    D[i]   = s0;
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

static uint8_t saturate_cast(float value)
{
    __m128 t  = _mm_set_ss(value);
    int32_t v = _mm_cvtss_si32(t);
    if (v > 255)
        return 255;
    else if (v < 0)
        return 0;
    return (uint8_t)v;
}

template <>
struct ColumnSum<int32_t, uint8_t> {
    ColumnSum(int32_t _ksize, float _scale)
    {
        ksize    = _ksize;
        scale    = _scale;
        sumCount = 0;
    }

    void reset()
    {
        sumCount = 0;
    }

    void operator()(const int32_t** src, uint8_t* dst, int32_t dststep, int32_t count, int32_t width)
    {
        int32_t i;
        int32_t* SUM;
        bool haveScale = scale != 1;
        float _scale   = scale;

        if (width != (int32_t)sum.size()) {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if (sumCount == 0) {
            memset((void*)SUM, 0, width * sizeof(int32_t));
            for (; sumCount < ksize - 1; sumCount++, src++) {
                const int32_t* Sp = (const int32_t*)src[0];
                i                 = 0;
                for (; i <= width - 4; i += 4) {
                    __m128i _sum = _mm_loadu_si128((const __m128i*)(SUM + i));
                    __m128i _sp  = _mm_loadu_si128((const __m128i*)(Sp + i));
                    _mm_storeu_si128((__m128i*)(SUM + i), _mm_add_epi32(_sum, _sp));
                }
                for (; i < width; i++)
                    SUM[i] += Sp[i];
            }
        } else {
            //assert( sumCount == ksize-1 );
            src += ksize - 1;
        }

        for (; count--; src++) {
            const int32_t* Sp = (const int32_t*)src[0];
            const int32_t* Sm = (const int32_t*)src[1 - ksize];
            uint8_t* D        = (uint8_t*)dst;
            if (haveScale) {
                i                   = 0;
                const __m128 scale4 = _mm_set1_ps(scale);
                for (; i <= width - 8; i += 8) {
                    __m128i _sm  = _mm_loadu_si128((const __m128i*)(Sm + i));
                    __m128i _sm1 = _mm_loadu_si128((const __m128i*)(Sm + i + 4));

                    __m128i _s0  = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
                                                _mm_loadu_si128((const __m128i*)(Sp + i)));
                    __m128i _s01 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i + 4)),
                                                 _mm_loadu_si128((const __m128i*)(Sp + i + 4)));

                    __m128i _s0T  = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s0)));
                    __m128i _s0T1 = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s01)));

                    _s0T = _mm_packs_epi32(_s0T, _s0T1);

                    _mm_storel_epi64((__m128i*)(D + i), _mm_packus_epi16(_s0T, _s0T));

                    _mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
                    _mm_storeu_si128((__m128i*)(SUM + i + 4), _mm_sub_epi32(_s01, _sm1));
                }
                for (; i < width; i++) {
                    int32_t s0 = SUM[i] + Sp[i];
                    D[i]       = saturate_cast(s0 * _scale);
                    SUM[i]     = s0 - Sm[i];
                }
            } else {
                i = 0;
                for (; i <= width - 8; i += 8) {
                    __m128i _sm  = _mm_loadu_si128((const __m128i*)(Sm + i));
                    __m128i _sm1 = _mm_loadu_si128((const __m128i*)(Sm + i + 4));

                    __m128i _s0  = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
                                                _mm_loadu_si128((const __m128i*)(Sp + i)));
                    __m128i _s01 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i + 4)),
                                                 _mm_loadu_si128((const __m128i*)(Sp + i + 4)));

                    __m128i _s0T = _mm_packs_epi32(_s0, _s01);

                    _mm_storel_epi64((__m128i*)(D + i), _mm_packus_epi16(_s0T, _s0T));

                    _mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
                    _mm_storeu_si128((__m128i*)(SUM + i + 4), _mm_sub_epi32(_s01, _sm1));
                }

                for (; i < width; i++) {
                    int32_t s0 = SUM[i] + Sp[i];
                    D[i]       = saturate_cast(s0);
                    SUM[i]     = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
    }
    int32_t ksize;
    float scale;
    int32_t sumCount;
    std::vector<int32_t> sum;
};

template <>
struct ColumnSum<float, float> {
    ColumnSum(int32_t _ksize, float _scale)
    {
        ksize    = _ksize;
        scale    = _scale;
        sumCount = 0;
        core     = 1;
    }

    void reset()
    {
        sumCount = 0;
    }

    void operator()(const float** src, float* dst, int32_t dststep, int32_t count, int32_t width)
    {
        int32_t i;

        float* SUM     = (float*)_mm_malloc(width * sizeof(float), 64);
        bool haveScale = scale != 1;
        if (sumCount == 0) {
            memset((void*)SUM, 0, width * sizeof(float));

            for (; sumCount < ksize - 1; sumCount++, src++) {
                const float* Sp = (const float*)src[0];
                i               = 0;
                for (; i < width; i++)
                    SUM[i] += Sp[i];
            }
        } else {
            src += ksize - 1;
        }

        for (; count--; src++) {
            const float* Sp = (const float*)src[0];
            const float* Sm = (const float*)src[1 - ksize];
            float* D        = (float*)dst;
            if (haveScale) {
                i = 0;
                for (; i < width; i++) {
                    float s0 = SUM[i] + Sp[i];
                    D[i]     = s0 * scale;
                    SUM[i]   = s0 - Sm[i];
                }
            } else {
                i = 0;
                for (; i < width; i++) {
                    float s0 = SUM[i] + Sp[i];
                    D[i]     = s0;
                    SUM[i]   = s0 - Sm[i];
                }
            }
            dst += dststep;
        }
        _mm_free(SUM);
    }
    int32_t ksize;
    float scale;
    int32_t sumCount;
    int32_t core;
    bool bSupportAVX;
};

template <int32_t cn>
void x86boxFilter_f(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    float* outData,
    BorderType borderType,
    float border_value = 0)
{
    int32_t radius_x = kernelx_len / 2;
    int32_t radius_y = kernely_len / 2;

    int32_t bsrcHeight = height + 2 * radius_y;
    int32_t bsrcWidth  = width + 2 * radius_x;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    float* bsrc_t         = (float*)malloc((bsrcHeight + 1) * bsrcWidth * cn * sizeof(float));
    float* bsrc           = bsrc_t + bsrcWidthStep;
    CopyMakeBorder<float, cn>(height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, borderType, border_value);

    RowSum<float, float> rowVecOp = RowSum<float, float>(kernelx_len);
    for (int32_t i = 0; i < bsrcHeight; i++) {
        float* src = bsrc + i * bsrcWidthStep;
        float* dst = bsrc_t + i * bsrcWidthStep;
        rowVecOp.operator()(src, dst, width, cn);
    }
    const float** pReRowFilter = (const float**)malloc(bsrcHeight * sizeof(bsrc_t));
    for (int32_t i = 0; i < bsrcHeight; i++) {
        pReRowFilter[i] = bsrc_t + (i)*bsrcWidthStep;
    }

    ColumnSum<float, float> colVecOp = ColumnSum<float, float>(kernely_len, normalize ? 1. / (kernelx_len * kernely_len) : 1);
    colVecOp.operator()((const float**)pReRowFilter, (float*)outData, outWidthStride, height, width* cn);

    free(bsrc_t);
    free(pReRowFilter);
    bsrc         = NULL;
    pReRowFilter = NULL;
}

template <int32_t cn>
void x86boxFilter_b(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType borderType,
    uint8_t border_value = 0)
{
    int32_t radius_x = kernelx_len / 2;
    int32_t radius_y = kernely_len / 2;

    int32_t bsrcHeight = height + 2 * radius_y;
    int32_t bsrcWidth  = width + 2 * radius_x;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    int32_t rfstep        = width * cn;
    uint8_t* bsrc         = (uint8_t*)malloc(bsrcHeight * bsrcWidth * cn * sizeof(uint8_t) +
                                     bsrcHeight * width * cn * sizeof(int32_t));
    CopyMakeBorder<uint8_t, cn>(height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, borderType, border_value);

    int32_t* resultRowFilter = (int32_t*)(bsrc + bsrcHeight * bsrcWidth * cn * sizeof(uint8_t));

    RowSum<uint8_t, int32_t> rowVecOp = RowSum<uint8_t, int32_t>(kernelx_len);
    for (int32_t i = 0; i < bsrcHeight; i++) {
        uint8_t* src = bsrc + i * bsrcWidthStep;
        int32_t* dst = resultRowFilter + i * rfstep;
        rowVecOp.operator()(src, dst, width, cn);
    }

    ColumnSum<int32_t, uint8_t> colVecOp = ColumnSum<int32_t, uint8_t>(kernely_len, normalize ? 1. / (kernelx_len * kernely_len) : 1);
    const int32_t** pReRowFilter         = (const int32_t**)malloc(bsrcHeight * sizeof(resultRowFilter));
    for (int32_t i = 0; i < bsrcHeight; i++) {
        pReRowFilter[i] = resultRowFilter + (i)*rfstep;
    }
    colVecOp.operator()(pReRowFilter, outData, outWidthStride, height, width* cn);

    free(bsrc);
    free(pReRowFilter);
    bsrc         = NULL;
    pReRowFilter = NULL;
}

template <>
::ppl::common::RetCode BoxFilter<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    x86boxFilter_f<1>(height, width, inWidthStride, inData, kernelx_len, kernely_len, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    x86boxFilter_f<3>(height, width, inWidthStride, inData, kernelx_len, kernely_len, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    x86boxFilter_f<4>(height, width, inWidthStride, inData, kernelx_len, kernely_len, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    x86boxFilter_b<1>(height, width, inWidthStride, inData, kernelx_len, kernely_len, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    x86boxFilter_b<3>(height, width, inWidthStride, inData, kernelx_len, kernely_len, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BoxFilter<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    bool normalize,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    x86boxFilter_b<4>(height, width, inWidthStride, inData, kernelx_len, kernely_len, normalize, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86