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

#include "ppl/cv/x86/arithmetic.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <immintrin.h>
#ifdef _WIN32
#include <algorithm>
#endif
#define EPS (1e-8)

namespace ppl {
namespace cv {
namespace x86 {

template <typename T, int32_t channels>
::ppl::common::RetCode Add(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData)
{
    if (nullptr == inData0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == inData1) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride0 <= 0 ||
        inWidthStride1 <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (std::is_same<T, float>().value) {
        for (int32_t h = 0; h < height; ++h) {
            Map(outData + h * outWidthStride,
                inData0 + h * inWidthStride0,
                inData1 + h * inWidthStride1,
                channels * width,
                [](T a, T b) {
                    return a + b;
                });
        }
    } else if (std::is_same<T, uint8_t>().value) {
        if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
            return fma::Add_fma<T, channels>(height, width, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
        }
        int32_t i = 0;
        for (; i <= height * width * channels - 16; i += 16) {
            __m128i vdata0 = _mm_loadu_si128((__m128i *)(inData0 + i));
            __m128i vdata1 = _mm_loadu_si128((__m128i *)(inData1 + i));
            __m128i vdst   = _mm_adds_epu8(vdata0, vdata1);
            _mm_storeu_si128((__m128i *)(outData + i), vdst);
        }
        for (; i < height * width * channels; i++) {
            outData[i] = sat_cast_u8(inData0[i] + inData1[i]);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t channels>
::ppl::common::RetCode Mul(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData,
    float alpha)
{
    if (nullptr == inData0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == inData1) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride0 <= 0 ||
        inWidthStride1 <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (std::abs(alpha - 1.0) < EPS) {
        if (std::is_same<T, float>().value) {
            for (int32_t h = 0; h < height; ++h) {
                Map(outData + h * outWidthStride,
                    inData0 + h * inWidthStride0,
                    inData1 + h * inWidthStride1,
                    channels * width,
                    [](T a, T b) {
                        return a * b;
                    });
            }
        } else if (std::is_same<T, uint8_t>().value) {
            if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
                return fma::Mul_fma<T, channels>(height, width, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData, alpha);
            }
            int32_t i = 0;
            for (; i <= height * width * channels - 16; i += 16) {
                __m128i vdata00 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData0 + i + 0)));
                __m128i vdata01 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData0 + i + 8)));
                __m128i vdata10 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData1 + i + 0)));
                __m128i vdata11 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData1 + i + 8)));
                __m128i vdst0   = _mm_abs_epi16(_mm_mullo_epi16(vdata00, vdata10));
                __m128i vdst1   = _mm_abs_epi16(_mm_mullo_epi16(vdata01, vdata11));
                _mm_storeu_si128((__m128i *)(outData + i), _mm_packus_epi16(vdst0, vdst1));
            }
            for (; i < height * width * channels; i++) {
                outData[i] = sat_cast_u8(inData0[i] * inData1[i]);
            }
        }
    } else {
        for (int32_t h = 0; h < height; ++h) {
            Map(outData + h * outWidthStride,
                inData0 + h * inWidthStride0,
                inData1 + h * inWidthStride1,
                alpha,
                channels * width,
                [](T a, T b) {
                    return a * b;
                });
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t channels>
::ppl::common::RetCode Mla(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData)
{
    if (nullptr == inData0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == inData1) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride0 <= 0 ||
        inWidthStride1 <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    for (int32_t i = 0; i < height; ++i) {
        T *base_outData       = outData + i * outWidthStride;
        const T *base_inData0 = inData0 + i * inWidthStride0;
        const T *base_inData1 = inData1 + i * inWidthStride1;
        for (int32_t j = 0; j < width * channels; ++j) {
            base_outData[j] += base_inData0[j] * base_inData1[j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t channels>
::ppl::common::RetCode Mls(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData)
{
    if (nullptr == inData0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == inData1) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride0 <= 0 ||
        inWidthStride1 <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    for (int32_t i = 0; i < height; ++i) {
        T *base_outData       = outData + i * outWidthStride;
        const T *base_inData0 = inData0 + i * inWidthStride0;
        const T *base_inData1 = inData1 + i * inWidthStride1;
        for (int32_t j = 0; j < width * channels; ++j) {
            base_outData[j] -= base_inData0[j] * base_inData1[j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t channels>
::ppl::common::RetCode Div(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData,
    float alpha)
{
    if (nullptr == inData0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == inData1) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride0 <= 0 ||
        inWidthStride1 <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (std::abs(alpha - 1.0) < EPS) {
        for (int32_t h = 0; h < height; ++h) {
            Map(outData + h * outWidthStride,
                inData0 + h * inWidthStride0,
                inData1 + h * inWidthStride1,
                channels * width,
                [](T a, T b) {
                    return a / b;
                });
        }
    } else {
        for (int32_t h = 0; h < height; ++h) {
            Map(outData + h * outWidthStride,
                inData0 + h * inWidthStride0,
                inData1 + h * inWidthStride1,
                alpha,
                channels * width,
                [](T a, T b) {
                    return a / b;
                });
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t channels>
::ppl::common::RetCode Subtract(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == scalar) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::Subtract<channels>(height, width, inWidthStride, inData, scalar, outWidthStride, outData);
    }
    if (channels == 1) {
        int32_t i = 0;
        for (; i <= height * width - 32; i += 32) {
            __m128i vdata0  = _mm_loadu_si128((__m128i *)(inData + i));
            __m128i vdata1  = _mm_loadu_si128((__m128i *)(inData + i + 16));
            __m128i vscalar = _mm_set1_epi8(scalar[0]);
            __m128i vdst0   = _mm_subs_epu8(vdata0, vscalar);
            __m128i vdst1   = _mm_subs_epu8(vdata1, vscalar);
            _mm_storeu_si128((__m128i *)(outData + i), vdst0);
            _mm_storeu_si128((__m128i *)(outData + i + 16), vdst1);
        }
        for (; i <= height * width - 16; i += 16) {
            __m128i vdata   = _mm_loadu_si128((__m128i *)(inData + i));
            __m128i vscalar = _mm_set1_epi8(scalar[0]);
            __m128i vdst    = _mm_subs_epu8(vdata, vscalar);
            _mm_storeu_si128((__m128i *)(outData + i), vdst);
        }
        for (; i < height * width; i++) {
            outData[i] = sat_cast_u8(inData[i] - scalar[0]);
        }
    } else if (channels == 3) {
        uint8_t scalar_tmp[16] = {0};
        for (int32_t i = 0; i < 15; i += 3) {
            scalar_tmp[i + 0] = scalar[0];
            scalar_tmp[i + 1] = scalar[1];
            scalar_tmp[i + 2] = scalar[2];
        }
        int32_t j = 0;
        for (; j <= height * width * 3 - 15; j += 15) {
            __m128i vdata   = _mm_lddqu_si128((__m128i *)(inData + j));
            __m128i vscalar = _mm_lddqu_si128((__m128i *)scalar_tmp);
            __m128i vdst    = _mm_subs_epu8(vdata, vscalar);
            _mm_storeu_si128((__m128i *)(outData + j), vdst);
        }
        for (; j < height * width * 3; j += 3) {
            outData[j + 0] = sat_cast_u8(inData[j + 0] - scalar[0]);
            outData[j + 1] = sat_cast_u8(inData[j + 1] - scalar[1]);
            outData[j + 2] = sat_cast_u8(inData[j + 2] - scalar[2]);
        }
    } else if (channels == 4) {
        uint8_t scalar_tmp[16] = {0};
        for (int32_t i = 0; i < 16; i += 4) {
            scalar_tmp[i + 0] = scalar[0];
            scalar_tmp[i + 1] = scalar[1];
            scalar_tmp[i + 2] = scalar[2];
            scalar_tmp[i + 3] = scalar[3];
        }
        int32_t j = 0;
        for (; j <= height * width * 4 - 16; j += 16) {
            __m128i vdata   = _mm_lddqu_si128((__m128i *)(inData + j));
            __m128i vscalar = _mm_lddqu_si128((__m128i *)scalar_tmp);
            __m128i vdst    = _mm_subs_epu8(vdata, vscalar);
            _mm_storeu_si128((__m128i *)(outData + j), vdst);
        }
        for (; j < height * width * 4; j += 4) {
            outData[j + 0] = sat_cast_u8(inData[j + 0] - scalar[0]);
            outData[j + 1] = sat_cast_u8(inData[j + 1] - scalar[1]);
            outData[j + 2] = sat_cast_u8(inData[j + 2] - scalar[2]);
            outData[j + 3] = sat_cast_u8(inData[j + 3] - scalar[3]);
        }
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t channels>
::ppl::common::RetCode Subtract(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    const float *scalar,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == scalar) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    for (int32_t i = 0; i < height; i++) {
        const float *ptr_in = inData + i * inWidthStride;
        float *ptr_out      = outData + i * outWidthStride;
        for (int32_t j = 0; j < width; j++) {
            for (int32_t k = 0; k < channels; k++) {
                ptr_out[j * channels + k] = ptr_in[j * channels + k] - scalar[k];
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t channels>
::ppl::common::RetCode Subtract(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T *inData,
    const T *scalar,
    int32_t outWidthStride,
    T *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == scalar) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    Subtract<channels>(height, width, inWidthStride, inData, scalar, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Add<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Add<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Add<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Add<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Add<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Add<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Mul<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Mul<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Mul<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Mul<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData,
    float alpha);

template ::ppl::common::RetCode Mul<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData,
    float alpha);

template ::ppl::common::RetCode Mul<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData,
    float alpha);

template ::ppl::common::RetCode Mla<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Mla<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Mla<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Mls<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Mls<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Mls<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Div<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Div<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Div<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Subtract<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    const float *scalar,
    int32_t outWidthStride,
    float *outData);
template ::ppl::common::RetCode Subtract<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    const float *scalar,
    int32_t outWidthStride,
    float *outData);
template ::ppl::common::RetCode Subtract<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    const float *scalar,
    int32_t outWidthStride,
    float *outData);
template ::ppl::common::RetCode Subtract<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData);
template ::ppl::common::RetCode Subtract<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData);
template ::ppl::common::RetCode Subtract<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData);

}
}
} // namespace ppl::cv::x86
