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

#include "ppl/cv/x86/rotate.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "intrinutils.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>
#include <assert.h>
#include <immintrin.h>

namespace ppl {
namespace cv {
namespace x86 {

inline void transpose(__m128i& a, __m128i& b, __m128i& c, __m128i& d, __m128i& e, __m128i& f, __m128i& g, __m128i& h)
{
    __m128i tmp0_vec = _mm_unpacklo_epi8(a, b); // a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7
    __m128i tmp1_vec = _mm_unpacklo_epi8(c, d); // c0,d0,c1,d1,c2,d2,c3,d3,c4,d4,c5,d5,c6,d6,c7,d7
    __m128i tmp2_vec = _mm_unpacklo_epi8(e, f); // e0,f0,e1,f1,e2,f2,e3,f3,e4,f4,e5,f5,e6,f6,e7,f7
    __m128i tmp3_vec = _mm_unpacklo_epi8(g, h); // g0,h0,g1,h1,g2,h2,g3,h3,g4,h4,g5,h5,g6,h6,g7,h7

    __m128i tmp4_vec = _mm_unpackhi_epi8(a, b);
    __m128i tmp5_vec = _mm_unpackhi_epi8(c, d);
    __m128i tmp6_vec = _mm_unpackhi_epi8(e, f);
    __m128i tmp7_vec = _mm_unpackhi_epi8(g, h);

    __m128i tmp8_vec  = _mm_unpacklo_epi16(tmp0_vec, tmp1_vec); // a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3
    __m128i tmp9_vec  = _mm_unpacklo_epi16(tmp2_vec, tmp3_vec); // e0,f0,g0,h0,e1,f1,g1,h1,e2,f2,g2,h2,e3,f3,g3,h3
    __m128i tmp10_vec = _mm_unpackhi_epi16(tmp0_vec, tmp1_vec); // a4,b4,c4,d4,a5,b5,c5,d5,a6,b6,c6,d6,a7,b7,c7,d7
    __m128i tmp11_vec = _mm_unpackhi_epi16(tmp2_vec, tmp3_vec); // e4,f4,g4,h4,e5,f5,g5,h5,e6,f6,g6,h6,e7,f7,g7,h7

    __m128i tmp12_vec = _mm_unpacklo_epi16(tmp4_vec, tmp5_vec);
    __m128i tmp13_vec = _mm_unpacklo_epi16(tmp6_vec, tmp7_vec);
    __m128i tmp14_vec = _mm_unpackhi_epi16(tmp4_vec, tmp5_vec);
    __m128i tmp15_vec = _mm_unpackhi_epi16(tmp6_vec, tmp7_vec);

    a = _mm_unpacklo_epi32(tmp8_vec, tmp9_vec); // a0,b0,c0,d0,e0,f0,g0,h0,a1,b1,c1,d1,e1,f1,g1,h1
    b = _mm_unpackhi_epi32(tmp8_vec, tmp9_vec); // a2,b2,c2,d2,e2,f2,g2,h2,a3,b3,c3,d3,e3,f3,g3,g3
    c = _mm_unpacklo_epi32(tmp10_vec, tmp11_vec); // a4,b4,c4,d4,e4,f4,g4,h4,a5,b5,c5,d5,e5,f5,g5,h5
    d = _mm_unpackhi_epi32(tmp10_vec, tmp11_vec); // a6,b6,c6,d6,e6,f6,g6,h6,a7,b7,c7,d7,e7,f7,g7,h7

    e = _mm_unpacklo_epi32(tmp12_vec, tmp13_vec);
    f = _mm_unpackhi_epi32(tmp12_vec, tmp13_vec);
    g = _mm_unpacklo_epi32(tmp14_vec, tmp15_vec);
    h = _mm_unpackhi_epi32(tmp14_vec, tmp15_vec);
}

template <typename T, int32_t nc>
void imgRotate90degree(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData)
{
    const int32_t BC   = 64;
    const int32_t BR   = 64;
    int32_t numColsEnd = outWidth - outWidth % BC;
    int32_t numRowsEnd = outHeight - outHeight % BR;
    for (int32_t i = 0; i < numRowsEnd; i += BR) {
        for (int32_t j = 0; j < numColsEnd; j += BC) {
            for (int32_t ii = 0; ii < BR; ii++) {
                for (int32_t jj = 0; jj < BC; jj++) {
                    for (int32_t c = 0; c < nc; c++) {
                        outData[(i + ii) * outWidthStride + (j + jj) * nc + c] =
                            inData[(inHeight - (j + jj) - 1) * inWidthStride + (i + ii) * nc + c];
                    }
                }
            }
        }
    }
    for (int32_t i = numRowsEnd; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] =
                    inData[(inHeight - j - 1) * inWidthStride + i * nc + c];
            }
        }
    }
    for (int32_t i = 0; i < numRowsEnd; i++) {
        for (int32_t j = numColsEnd; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] =
                    inData[(inHeight - j - 1) * inWidthStride + i * nc + c];
            }
        }
    }
}

template <>
void imgRotate90degree<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    const int32_t BC   = 64;
    const int32_t BR   = 64;
    int32_t numColsEnd = outWidth - outWidth % BC;
    int32_t numRowsEnd = outHeight - outHeight % BR;
    for (int32_t i = 0; i < numRowsEnd; i += BR) {
        for (int32_t j = 0; j < numColsEnd; j += BC) {
            for (int32_t ii = 0; ii < BR; ii += 16) {
                for (int32_t jj = 0; jj < BC; jj += 8) {
                    const uint8_t* base_in = inData + (inHeight - (j + jj) - 1) * inWidthStride + (i + ii);
                    uint8_t* base_out      = outData + (i + ii) * outWidthStride + (j + jj);
                    __m128i vec0           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 7 * inWidthStride));
                    __m128i vec1           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 6 * inWidthStride));
                    __m128i vec2           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 5 * inWidthStride));
                    __m128i vec3           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 4 * inWidthStride));
                    __m128i vec4           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 3 * inWidthStride));
                    __m128i vec5           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 2 * inWidthStride));
                    __m128i vec6           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 1 * inWidthStride));
                    __m128i vec7           = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in - 0 * inWidthStride));
                    transpose(vec7, vec6, vec5, vec4, vec3, vec2, vec1, vec0);
                    _mm_storel_pd(reinterpret_cast<double*>(base_out), _mm_castsi128_pd(vec7));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + outWidthStride), _mm_castsi128_pd(vec7));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 2 * outWidthStride), _mm_castsi128_pd(vec6));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 3 * outWidthStride), _mm_castsi128_pd(vec6));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 4 * outWidthStride), _mm_castsi128_pd(vec5));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 5 * outWidthStride), _mm_castsi128_pd(vec5));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 6 * outWidthStride), _mm_castsi128_pd(vec4));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 7 * outWidthStride), _mm_castsi128_pd(vec4));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 8 * outWidthStride), _mm_castsi128_pd(vec3));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 9 * outWidthStride), _mm_castsi128_pd(vec3));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 10 * outWidthStride), _mm_castsi128_pd(vec2));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 11 * outWidthStride), _mm_castsi128_pd(vec2));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 12 * outWidthStride), _mm_castsi128_pd(vec1));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 13 * outWidthStride), _mm_castsi128_pd(vec1));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 14 * outWidthStride), _mm_castsi128_pd(vec0));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 15 * outWidthStride), _mm_castsi128_pd(vec0));
                }
            }
        }
    }
    for (int32_t i = numRowsEnd; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            outData[i * outWidthStride + j] =
                inData[(inHeight - j - 1) * inWidthStride + i];
        }
    }
    for (int32_t i = 0; i < numRowsEnd; i++) {
        for (int32_t j = numColsEnd; j < outWidth; j++) {
            outData[i * outWidthStride + j] =
                inData[(inHeight - j - 1) * inWidthStride + i];
        }
    }
}

template <typename T, int32_t nc>
void imgRotate180degree(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData)
{
    for (int32_t i = 0; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] =
                    inData[(inHeight - i - 1) * inWidthStride + (inWidth - j - 1) * nc + c];
            }
        }
    }
}

template <>
void imgRotate180degree<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    __m128i reverse_mask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (int32_t i = 0; i < outHeight; i++) {
        uint8_t* baseOut      = outData + i * outWidthStride;
        const uint8_t* baseIn = inData + (inHeight - i - 1) * inWidthStride;
        for (int32_t j = 0; j < outWidth / 16 * 16; j += 16) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(baseOut + j),
                             _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(baseIn + inWidth - 16 - j)), reverse_mask));
        }
        for (int32_t j = outWidth / 16 * 16; j < outWidth; j++) {
            baseOut[j] = baseIn[inWidth - j - 1];
        }
    }
}

template <typename T, int32_t nc>
void imgRotate270degree(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData)
{
    const int32_t BC   = 64;
    const int32_t BR   = 64;
    int32_t numColsEnd = outWidth - outWidth % BC;
    int32_t numRowsEnd = outHeight - outHeight % BR;
    for (int32_t i = 0; i < numRowsEnd; i += BR) {
        for (int32_t j = 0; j < numColsEnd; j += BC) {
            for (int32_t ii = 0; ii < BR; ii++) {
                for (int32_t jj = 0; jj < BC; jj++) {
                    for (int32_t c = 0; c < nc; c++) {
                        outData[(i + ii) * outWidthStride + (j + jj) * nc + c] =
                            inData[(j + jj) * inWidthStride + (inWidth - (i + ii) - 1) * nc + c];
                    }
                }
            }
        }
    }
    for (int32_t i = numRowsEnd; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] =
                    inData[j * inWidthStride + (inWidth - i - 1) * nc + c];
            }
        }
    }
    for (int32_t i = 0; i < numRowsEnd; i++) {
        for (int32_t j = numColsEnd; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] =
                    inData[j * inWidthStride + (inWidth - i - 1) * nc + c];
            }
        }
    }
}

template <>
void imgRotate270degree<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    const int32_t BC     = 64;
    const int32_t BR     = 64;
    __m128i reverse_mask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    int32_t numColsEnd   = outWidth - outWidth % BC;
    int32_t numRowsEnd   = outHeight - outHeight % BR;
    for (int32_t i = 0; i < numRowsEnd; i += BR) {
        for (int32_t j = 0; j < numColsEnd; j += BC) {
            for (int32_t ii = 0; ii < BR; ii += 16) {
                for (int32_t jj = 0; jj < BC; jj += 8) {
                    const uint8_t* base_in = inData + (j + jj) * inWidthStride + (inWidth - (i + ii) - 16);
                    uint8_t* base_out      = outData + (i + ii) * outWidthStride + (j + jj);
                    __m128i vec0           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 0 * inWidthStride)), reverse_mask);
                    __m128i vec1           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 1 * inWidthStride)), reverse_mask);
                    __m128i vec2           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 2 * inWidthStride)), reverse_mask);
                    __m128i vec3           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 3 * inWidthStride)), reverse_mask);
                    __m128i vec4           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 4 * inWidthStride)), reverse_mask);
                    __m128i vec5           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 5 * inWidthStride)), reverse_mask);
                    __m128i vec6           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 6 * inWidthStride)), reverse_mask);
                    __m128i vec7           = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base_in + 7 * inWidthStride)), reverse_mask);
                    transpose(vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7);
                    _mm_storel_pd(reinterpret_cast<double*>(base_out), _mm_castsi128_pd(vec0));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + outWidthStride), _mm_castsi128_pd(vec0));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 2 * outWidthStride), _mm_castsi128_pd(vec1));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 3 * outWidthStride), _mm_castsi128_pd(vec1));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 4 * outWidthStride), _mm_castsi128_pd(vec2));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 5 * outWidthStride), _mm_castsi128_pd(vec2));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 6 * outWidthStride), _mm_castsi128_pd(vec3));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 7 * outWidthStride), _mm_castsi128_pd(vec3));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 8 * outWidthStride), _mm_castsi128_pd(vec4));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 9 * outWidthStride), _mm_castsi128_pd(vec4));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 10 * outWidthStride), _mm_castsi128_pd(vec5));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 11 * outWidthStride), _mm_castsi128_pd(vec5));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 12 * outWidthStride), _mm_castsi128_pd(vec6));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 13 * outWidthStride), _mm_castsi128_pd(vec6));
                    _mm_storel_pd(reinterpret_cast<double*>(base_out + 14 * outWidthStride), _mm_castsi128_pd(vec7));
                    _mm_storeh_pd(reinterpret_cast<double*>(base_out + 15 * outWidthStride), _mm_castsi128_pd(vec7));
                }
            }
        }
    }
    for (int32_t i = numRowsEnd; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            outData[i * outWidthStride + j] =
                inData[j * inWidthStride + (inWidth - i - 1)];
        }
    }
    for (int32_t i = 0; i < numRowsEnd; i++) {
        for (int32_t j = numColsEnd; j < outWidth; j++) {
            outData[i * outWidthStride + j] =
                inData[j * inWidthStride + (inWidth - i - 1)];
        }
    }
}

template <typename T, int32_t nc>
::ppl::common::RetCode imgRotate(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData,
    int32_t degree)
{
    if (degree == 90) {
        imgRotate90degree<T, nc>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    } else if (degree == 180) {
        imgRotate180degree<T, nc>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    } else if (degree == 270) {
        imgRotate270degree<T, nc>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<float, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode RotateNx90degree<float, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<float, 2>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<float, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<float, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<uint8_t, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree<uint8_t, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<uint8_t, 2>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<uint8_t, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t degree)
{
    assert(inData != NULL);
    assert(outData != NULL);
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0 && outHeight != 0 && outWidth != 0);
    assert(degree == 90 || degree == 180 || degree == 270);
    imgRotate<uint8_t, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree_NV12<uint8_t>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inYStride,
    const uint8_t* inDataY,
    int32_t inUVStride,
    const uint8_t* inDataUV,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outYStride,
    uint8_t* outDataY,
    int32_t outUVStride,
    uint8_t* outDataUV,
    int32_t degree)
{
    imgRotate<uint8_t, 1>(inHeight, inWidth, inYStride, inDataY, outHeight, outWidth, outYStride, outDataY, degree);
    imgRotate<uint8_t, 2>(inHeight / 2, inWidth / 2, inUVStride, inDataUV, outHeight / 2, outWidth / 2, outUVStride, outDataUV, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree_NV21<uint8_t>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inYStride,
    const uint8_t* inDataY,
    int32_t inVUStride,
    const uint8_t* inDataVU,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outYStride,
    uint8_t* outDataY,
    int32_t outVUStride,
    uint8_t* outDataVU,
    int32_t degree)
{
    RotateNx90degree_NV12<uint8_t>(inHeight, inWidth, inYStride, inDataY, inVUStride, inDataVU, outHeight, outWidth, outYStride, outDataY, outVUStride, outDataVU, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RotateNx90degree_I420<uint8_t>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inYStride,
    const uint8_t* inDataY,
    int32_t inUStride,
    const uint8_t* inDataU,
    int32_t inVStride,
    const uint8_t* inDataV,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outYStride,
    uint8_t* outDataY,
    int32_t outUStride,
    uint8_t* outDataU,
    int32_t outVStride,
    uint8_t* outDataV,
    int32_t degree)
{
    imgRotate<uint8_t, 1>(inHeight, inWidth, inYStride, inDataY, outHeight, outWidth, outYStride, outDataY, degree);
    imgRotate<uint8_t, 1>(inHeight / 2, inWidth / 2, inUStride, inDataU, outHeight / 2, outWidth / 2, outUStride, outDataU, degree);
    imgRotate<uint8_t, 1>(inHeight / 2, inWidth / 2, inVStride, inDataV, outHeight / 2, outWidth / 2, outVStride, outDataV, degree);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
