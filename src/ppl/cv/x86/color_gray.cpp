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

#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/x86/intrinutils.hpp"
#include "ppl/cv/types.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <immintrin.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {
template <typename _Tp>
struct ColorChannel {
};
template <>
struct ColorChannel<float> {
    typedef float worktype_f;
    static float max()
    {
        return 1.f;
    }
    static float half()
    {
        return 0.5f;
    }
};
template <>
struct ColorChannel<uint8_t> {
    typedef uint8_t worktype_f;
    static uint8_t max()
    {
        return 255;
    }
    static uint8_t half()
    {
        return 128;
    }
};
template <typename _Tp>
struct Gray2RGB {
    typedef _Tp channel_type;

    Gray2RGB(int32_t _dstcn)
        : dstcn(_dstcn) {}
    void operator()(const _Tp *src, _Tp *dst, int32_t n) const
    {
        if (dstcn == 3)
            for (int32_t i = 0; i < n; i++, dst += 3) {
                dst[0] = dst[1] = dst[2] = src[i];
            }
        else {
            _Tp alpha = ColorChannel<_Tp>::max();
            for (int32_t i = 0; i < n; i++, dst += 4) {
                dst[0] = dst[1] = dst[2] = src[i];
                dst[3]                   = alpha;
            }
        }
    }

    int32_t dstcn;
};

template <typename _Tp>
struct RGB2Gray {
    typedef _Tp channel_type;

    RGB2Gray(int32_t _srccn, int32_t blueIdx, const float *_coeffs)
        : srccn(_srccn)
    {
        static const float coeffs0[] = {0.299f, 0.587f, 0.114f};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3 * sizeof(coeffs[0]));
        if (blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
        v_cb = _mm_set1_ps(coeffs[0]);
        v_cg = _mm_set1_ps(coeffs[1]);
        v_cr = _mm_set1_ps(coeffs[2]);
    }
    void inline process(__m128 v_b, __m128 v_g, __m128 v_r, __m128 &v_gray) const
    {
        v_gray = _mm_mul_ps(v_r, v_cr);
        v_gray = _mm_add_ps(v_gray, _mm_mul_ps(v_g, v_cg));
        v_gray = _mm_add_ps(v_gray, _mm_mul_ps(v_b, v_cb));
    }
    void operator()(const _Tp *src, _Tp *dst, int32_t n) const
    {
        int32_t scn = srccn, i = 0;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];

        const int32_t vsize = 4;
        for (; i <= n - vsize;
             i += vsize, src += vsize * scn) {
            __m128 v_r0, v_g0, v_b0, v_a0;
            if (scn == 3) {
                v_load_deinterleave(src, v_r0, v_g0, v_b0);
            } else {
                v_load_deinterleave(src, v_r0, v_g0, v_b0, v_a0);
            }

            __m128 v_gray0;
            process(v_r0, v_g0, v_b0, v_gray0);
            _mm_storeu_ps(dst + i, v_gray0);
        }
        for (; i < n; i++, src += scn)
            dst[i] = src[0] * cb + src[1] * cg + src[2] * cr;
    }
    int32_t srccn;
    float coeffs[3];
    __m128 v_cb, v_cg, v_cr;
};

template <>
struct RGB2Gray<uint8_t> {
    typedef uint8_t channel_type;

    RGB2Gray(int32_t _srccn, int32_t blueIdx, const int32_t *coeffs)
        : srccn(_srccn)
    {
        yuv_shift               = 14;
        const int32_t coeffs0[] = {4899, 9617, 1868};
        if (!coeffs) coeffs = coeffs0;

        int32_t b = 0, g = 0, r = (1 << (yuv_shift - 1));
        int32_t db = coeffs[blueIdx ^ 2], dg = coeffs[1], dr = coeffs[blueIdx];

        for (int32_t i = 0; i < 256; i++, b += db, g += dg, r += dr) {
            tab[i]       = b;
            tab[i + 256] = g;
            tab[i + 512] = r;
        }
    }
    void operator()(const uint8_t *src, uint8_t *dst, int32_t n) const
    {
        int32_t scn         = srccn;
        const int32_t *_tab = tab;
        for (int32_t i = 0; i < n; i++, src += scn)
            dst[i] = sat_cast_u8((_tab[src[0]] + _tab[src[1] + 256] + _tab[src[2] + 512]) >> yuv_shift);
    }
    int32_t srccn;
    int32_t tab[256 * 3];
    int32_t yuv_shift;
};
::ppl::common::RetCode bgr2gray_operator(
    const uint8_t *src,
    uint8_t *dst,
    int32_t width,
    int32_t height,
    int32_t stride,
    bool flag)
{
    const int32_t shift     = 15;
    const int32_t halfshift = 1 << (shift - 1);

    int32_t coeff_b = 0.114f * (1 << shift), coeff_g = 0.587f * (1 << shift) + 0.5, coeff_r = (1 << shift) - coeff_b - coeff_g, coeff_c = 1;
    if (!flag) {
        int32_t swap = coeff_b;
        coeff_b      = coeff_r;
        coeff_r      = swap;
    }
    __m128i coeff_bg = _mm_setr_epi16(coeff_b, coeff_g, coeff_b, coeff_g, coeff_b, coeff_g, coeff_b, coeff_g);
    __m128i coeff_rc = _mm_setr_epi16(coeff_r, coeff_c, coeff_r, coeff_c, coeff_r, coeff_c, coeff_r, coeff_c);
    __m128i v_half   = _mm_setr_epi16(0, halfshift, 0, halfshift, 0, halfshift, 0, halfshift);
    __m128i v_zero   = _mm_setzero_si128();

    int32_t vsize = 16;
    for (int32_t h = 0; h < height; h++) {
        const uint8_t *src_ptr = src + h * stride;
        uint8_t *dst_ptr       = dst + h * width;
        int32_t w              = 0;
        for (; w < width; w += vsize, src_ptr += vsize * 3) {
            __m128i data1 = _mm_loadu_si128((__m128i *)(src_ptr + 0));
            __m128i data2 = _mm_loadu_si128((__m128i *)(src_ptr + 16));
            __m128i data3 = _mm_loadu_si128((__m128i *)(src_ptr + 32));

            __m128i v_bgl  = _mm_shuffle_epi8(data1, _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, -1, -1, -1, -1, -1));
            v_bgl          = _mm_or_si128(v_bgl, _mm_shuffle_epi8(data2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 3, 5, 6)));
            __m128i v_bgh  = _mm_shuffle_epi8(data2, _mm_setr_epi8(8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
            v_bgh          = _mm_or_si128(v_bgh, _mm_shuffle_epi8(data3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14)));
            __m128i v_rcl  = _mm_shuffle_epi8(data1, _mm_setr_epi8(2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1));
            v_rcl          = _mm_or_si128(v_rcl, _mm_shuffle_epi8(data2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 4, -1, 7, -1)));
            __m128i v_rch  = _mm_shuffle_epi8(data2, _mm_setr_epi8(10, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
            v_rch          = _mm_or_si128(v_rch, _mm_shuffle_epi8(data3, _mm_setr_epi8(-1, -1, -1, -1, 0, -1, 3, -1, 6, -1, 9, -1, 12, -1, 15, -1)));
            __m128i v_gbll = _mm_unpacklo_epi8(v_bgl, v_zero);
            __m128i v_gblh = _mm_unpackhi_epi8(v_bgl, v_zero);
            __m128i v_rcll = _mm_or_si128(_mm_unpacklo_epi8(v_rcl, v_zero), v_half);
            __m128i v_rclh = _mm_or_si128(_mm_unpackhi_epi8(v_rcl, v_zero), v_half);
            __m128i v_bghl = _mm_unpacklo_epi8(v_bgh, v_zero);
            __m128i v_bghh = _mm_unpackhi_epi8(v_bgh, v_zero);
            __m128i v_rchl = _mm_or_si128(_mm_unpacklo_epi8(v_rch, v_zero), v_half);
            __m128i v_rchh = _mm_or_si128(_mm_unpackhi_epi8(v_rch, v_zero), v_half);

            __m128i grayll = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v_gbll, coeff_bg), _mm_madd_epi16(v_rcll, coeff_rc)), shift);
            __m128i graylh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v_gblh, coeff_bg), _mm_madd_epi16(v_rclh, coeff_rc)), shift);
            __m128i grayhl = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v_bghl, coeff_bg), _mm_madd_epi16(v_rchl, coeff_rc)), shift);
            __m128i grayhh = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v_bghh, coeff_bg), _mm_madd_epi16(v_rchh, coeff_rc)), shift);
            _mm_storeu_si128((__m128i *)(dst_ptr + w), _mm_packus_epi16(_mm_packus_epi32(grayll, graylh), _mm_packus_epi32(grayhl, grayhh)));
        }
        for (; w < width; w++, src_ptr += 3) {
            int32_t blue = src_ptr[0], green = src_ptr[1], red = src_ptr[2];
            dst_ptr[w] = (coeff_b * blue + coeff_g * green + coeff_r * red + halfshift) >> shift;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::BGR2GRAY(height, width, inWidthStride, inData, outWidthStride, outData, false);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return BGR2GRAYImage_avx<float, 3, float, 1>(height, width, inWidthStride, inData, outWidthStride, outData);
    }
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(3, 0, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return bgr2gray_operator(inData, outData, width, height, inWidthStride, true);
}

template <>
::ppl::common::RetCode BGRA2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return BGR2GRAYImage_avx<float, 4, float, 1>(height, width, inWidthStride, inData, outWidthStride, outData);
    }
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(4, 0, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BGRA2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *src  = inData;
    uint8_t *dst        = outData;
    RGB2Gray<uint8_t> s = RGB2Gray<uint8_t>(4, 0, NULL);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::BGR2GRAY(height, width, inWidthStride, inData, outWidthStride, outData, true);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return RGB2GRAYImage_avx<float, 3, float, 1>(height, width, inWidthStride, inData, outWidthStride, outData);
    }
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(3, 2, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode RGB2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return bgr2gray_operator(inData, outData, width, height, inWidthStride, false);
    const uint8_t *src  = inData;
    uint8_t *dst        = outData;
    RGB2Gray<uint8_t> s = RGB2Gray<uint8_t>(3, 2, NULL);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return RGB2GRAYImage_avx<float, 4, float, 1>(height, width, inWidthStride, inData, outWidthStride, outData);
    }
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(4, 2, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode RGBA2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *src  = inData;
    uint8_t *dst        = outData;
    RGB2Gray<uint8_t> s = RGB2Gray<uint8_t>(4, 2, NULL);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GRAY2BGR<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const float *src  = inData;
    float *dst        = outData;
    Gray2RGB<float> s = Gray2RGB<float>(3);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode GRAY2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *src  = inData;
    uint8_t *dst        = outData;
    Gray2RGB<uint8_t> s = Gray2RGB<uint8_t>(3);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GRAY2BGRA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const float *src  = inData;
    float *dst        = outData;
    Gray2RGB<float> s = Gray2RGB<float>(4);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode GRAY2BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *src  = inData;
    uint8_t *dst        = outData;
    Gray2RGB<uint8_t> s = Gray2RGB<uint8_t>(4);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GRAY2RGB<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const float *src  = inData;
    float *dst        = outData;
    Gray2RGB<float> s = Gray2RGB<float>(3);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode GRAY2RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *src  = inData;
    uint8_t *dst        = outData;
    Gray2RGB<uint8_t> s = Gray2RGB<uint8_t>(3);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GRAY2RGBA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const float *src  = inData;
    float *dst        = outData;
    Gray2RGB<float> s = Gray2RGB<float>(4);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode GRAY2RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *src  = inData;
    uint8_t *dst        = outData;
    Gray2RGB<uint8_t> s = Gray2RGB<uint8_t>(4);
    for (int32_t i = 0; i < height; i++) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
