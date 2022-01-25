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

#include "ppl/cv/x86/erode.h"
#include "ppl/cv/x86/morph.hpp"

#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"

#include <assert.h>
#include <float.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <limits>
#include <immintrin.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

bool isErodeBorderSupported(BorderType border_type)
{
    return border_type == BORDER_CONSTANT ||
           border_type == BORDER_REPLICATE ||
           border_type == BORDER_REFLECT_101 ||
           border_type == BORDER_REFLECT101 ||
           border_type == BORDER_REFLECT ||
           border_type == BORDER_DEFAULT;
}

::ppl::common::RetCode x86minFilter_b(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t cn,
    uint8_t border_value)
{
    uint8_t maximum  = 255;
    uint8_t* gRowMin = (uint8_t*)malloc(height * inWidthStride * sizeof(uint8_t));

    int32_t leftPad  = cn * (kernely_len >> 1);
    int32_t rightPad = cn * width - leftPad;
    if (!(kernely_len & 1)) rightPad += cn;

    for (int32_t i = 0; i < height; ++i) {
        int32_t inIndex = i * inWidthStride;

        for (int32_t j = 0; j < leftPad; ++j) {
            int32_t yEnd = j - leftPad + cn * kernely_len;
            uint8_t _min = border_value;
            for (int32_t jj = j % cn; jj < yEnd; jj += cn) {
                if (inData[inIndex + jj] < _min) _min = inData[inIndex + jj];
            }
            gRowMin[inIndex + j] = _min;
        }

        int32_t j;
        for (j = leftPad; j < rightPad - 16; j += 16) {
            __m128i mm_min = _mm_set1_epi8(0xff);
            for (int32_t jj = j - leftPad; jj < j - leftPad + cn * kernely_len; jj += cn) {
                __m128i mm_temp = _mm_loadu_si128((__m128i*)(inData + inIndex + jj));
                mm_min          = _mm_min_epu8(mm_min, mm_temp);
            }
            _mm_storeu_si128((__m128i*)(gRowMin + inIndex + j), mm_min);
        }
        for (; j < width * cn; ++j) {
            int32_t yStart = j - leftPad;
            uint8_t _min   = (j < rightPad) ? maximum : border_value;
            int32_t yEnd   = yStart + cn * kernely_len;
            yEnd           = std::min<int32_t>(yEnd, width * cn);
            for (int32_t jj = yStart; jj < yEnd; jj += cn)
                if (inData[inIndex + jj] < _min) _min = inData[inIndex + jj];
            gRowMin[inIndex + j] = _min;
        }
    }

    int32_t upPad   = kernelx_len >> 1;
    int32_t downPad = height - upPad;
    if (!(kernelx_len & 1)) ++downPad;

    for (int32_t i = 0; i < height; ++i) {
        int32_t xStart = i - upPad;
        int32_t xEnd   = xStart + kernelx_len;
        bool valid     = (xStart >= 0) && (xEnd <= height);
        xEnd           = std::min<int32_t>(xEnd, height);
        xStart         = std::max<int32_t>(xStart, 0);
        int32_t j      = 0;
        for (; j < width * cn - 16; j += 16) {
            __m128i mm_min = _mm_set1_epi8(valid ? maximum : border_value);
            for (int32_t ii = xStart; ii < xEnd; ++ii) {
                __m128i mm_temp = _mm_loadu_si128((__m128i*)(gRowMin + ii * inWidthStride + j));
                mm_min          = _mm_min_epu8(mm_temp, mm_min);
            }
            _mm_storeu_si128((__m128i*)(outData + i * outWidthStride + j), mm_min);
        }
        for (; j < width * cn; ++j) {
            uint8_t _min = valid ? maximum : border_value;
            for (int32_t ii = xStart; ii < xEnd; ++ii) {
                if (gRowMin[ii * inWidthStride + j] < _min) _min = gRowMin[ii * inWidthStride + j];
            }
            outData[i * outWidthStride + j] = _min;
        }
    }

    free(gRowMin);
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode x86minFilter_f(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    int32_t outWidthStride,
    float* outData,
    int32_t cn,
    float border_value)
{
    float maximum  = FLT_MAX;
    float* gRowMin = (float*)malloc(height * inWidthStride * sizeof(float));

    int32_t leftPad  = cn * (kernely_len >> 1);
    int32_t rightPad = cn * width - leftPad;
    if (!(kernely_len & 1)) rightPad += cn;

    for (int32_t i = 0; i < height; ++i) {
        int32_t inIndex = i * inWidthStride;

        for (int32_t j = 0; j < leftPad; ++j) {
            int32_t yEnd = j - leftPad + cn * kernely_len;
            float _min   = border_value;
            for (int32_t jj = j % cn; jj < yEnd; jj += cn)
                if (inData[inIndex + jj] < _min) _min = inData[inIndex + jj];
            gRowMin[inIndex + j] = _min;
        }

        int32_t j;
        for (j = leftPad; j < rightPad - 4; j += 4) {
            __m128 mm_min = _mm_set_ps1(FLT_MAX);
            for (int32_t jj = j - leftPad; jj < j - leftPad + cn * kernely_len; jj += cn) {
                __m128 mm_temp = _mm_loadu_ps(inData + inIndex + jj);
                mm_min         = _mm_min_ps(mm_min, mm_temp);
            }
            _mm_storeu_ps(gRowMin + inIndex + j, mm_min);
        }
        for (; j < width * cn; ++j) {
            int32_t yStart = j - leftPad;
            float _min     = (j < rightPad) ? maximum : border_value;
            int32_t yEnd   = yStart + cn * kernely_len;
            yEnd           = std::min<int32_t>(yEnd, width * cn);
            for (int32_t jj = yStart; jj < yEnd; jj += cn)
                if (inData[inIndex + jj] < _min) _min = inData[inIndex + jj];
            gRowMin[inIndex + j] = _min;
        }
    }

    int32_t upPad   = kernelx_len >> 1;
    int32_t downPad = height - upPad;
    if (!(kernelx_len & 1)) ++downPad;

    for (int32_t i = 0; i < height; ++i) {
        int32_t xStart = i - upPad;
        int32_t xEnd   = xStart + kernelx_len;
        bool valid     = (xStart >= 0) && (xEnd <= height);
        xEnd           = std::min<int32_t>(xEnd, height);
        xStart         = std::max<int32_t>(xStart, 0);
        int32_t j      = 0;
        for (; j < width * cn - 4; j += 4) {
            __m128 mm_min = _mm_set_ps1(valid ? maximum : border_value);
            for (int32_t ii = xStart; ii < xEnd; ++ii) {
                __m128 mm_temp = _mm_loadu_ps(gRowMin + ii * inWidthStride + j);
                mm_min         = _mm_min_ps(mm_temp, mm_min);
            }
            _mm_storeu_ps(outData + i * outWidthStride + j, mm_min);
        }
        for (; j < width * cn; ++j) {
            float _min = valid ? maximum : border_value;
            for (int32_t ii = xStart; ii < xEnd; ++ii) {
                if (gRowMin[ii * inWidthStride + j] < _min) _min = gRowMin[ii * inWidthStride + j];
            }
            outData[i * outWidthStride + j] = _min;
        }
    }

    free(gRowMin);
    return ppl::common::RC_SUCCESS;
}

template <typename T>
::ppl::common::RetCode x86minFilter_normal(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* element,
    int32_t outWidthStride,
    T* outData,
    int32_t cn,
    T border_value)
{
    T maximum = std::numeric_limits<T>::max();
    ;
    for (int32_t i = 0; i < height; ++i) {
        for (int32_t j = 0; j < width; ++j) {
            for (int32_t c = 0; c < cn; ++c) {
                T _min = maximum;
                for (int32_t ky = 0; ky < kernely_len; ++ky) {
                    int32_t src_y = i + ky - (kernely_len >> 1);
                    bool valid_y  = ((src_y >= 0) && (src_y < height));
                    for (int32_t kx = 0; kx < kernelx_len; ++kx) {
                        int32_t src_x = j + kx - (kernelx_len >> 1);
                        bool valid_x  = ((src_x >= 0) && (src_x < width));
                        if (element[ky * kernelx_len + kx]) {
                            T value = (valid_x && valid_y) ? inData[src_y * inWidthStride + src_x * cn + c] : border_value;
                            _min    = std::min(_min, value);
                        }
                    }
                }
                outData[i * outWidthStride + j * cn + c] = _min;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Erode<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* element,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type,
    uint8_t border_value)
{
    if (!inData || !outData || !element || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (!isErodeBorderSupported(border_type)) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (border_type != BORDER_CONSTANT) {
        border_value = 255;
    }
    bool flag = true;
    for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {
        if (element[i] != 1) {
            flag = false;
            break;
        }
    }
    if (flag) {
        if (3 == kernely_len && 3 == kernelx_len) {
            morph_u8<ErodeVecOp, 1, 3>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else if (5 == kernely_len && 5 == kernelx_len) {
            morph_u8<ErodeVecOp, 1, 5>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else {
            return x86minFilter_b(height, width, inWidthStride, inData, kernelx_len, kernely_len, outWidthStride, outData, 1, border_value);
        }
    } else
        return x86minFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len, element, outWidthStride, outData, 1, border_value);
}

template <>
::ppl::common::RetCode Erode<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* element,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type,
    uint8_t border_value)
{
    if (!inData || !outData || !element || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (!isErodeBorderSupported(border_type)) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (border_type != BORDER_CONSTANT) {
        border_value = 255;
    }
    bool flag = true;
    for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {
        if (element[i] != 1) {
            flag = false;
            break;
        }
    }
    if (flag) {
        if (3 == kernely_len && 3 == kernelx_len) {
            morph_u8<ErodeVecOp, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else if (5 == kernely_len && 5 == kernelx_len) {
            morph_u8<ErodeVecOp, 3, 5>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else {
            return x86minFilter_b(height, width, inWidthStride, inData, kernelx_len, kernely_len, outWidthStride, outData, 3, border_value);
        }
    } else {
        return x86minFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len, element, outWidthStride, outData, 3, border_value);
    }
}

template <>
::ppl::common::RetCode Erode<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* element,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type,
    uint8_t border_value)
{
    if (!inData || !outData || !element || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (!isErodeBorderSupported(border_type)) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (border_type != BORDER_CONSTANT) {
        border_value = 255;
    }
    bool flag = true;
    for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {
        if (element[i] != 1) {
            flag = false;
            break;
        }
    }
    if (flag) {
        if (3 == kernely_len && 3 == kernelx_len) {
            morph_u8<ErodeVecOp, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else if (5 == kernely_len && 5 == kernelx_len) {
            morph_u8<ErodeVecOp, 4, 5>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else {
            return x86minFilter_b(height, width, inWidthStride, inData, kernelx_len, kernely_len, outWidthStride, outData, 4, border_value);
        }
    } else {
        return x86minFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len, element, outWidthStride, outData, 4, border_value);
    }
}

template <>
::ppl::common::RetCode Erode<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* element,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type,
    float border_value)
{
    if (!inData || !outData || !element || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (!isErodeBorderSupported(border_type)) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (border_type != BORDER_CONSTANT) {
        border_value = FLT_MAX;
    }
    bool flag = true;
    for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {
        if (element[i] != 1) {
            flag = false;
            break;
        }
    }
    if (flag) {
        if (3 == kernely_len && 3 == kernelx_len) {
            morph_f32<ErodeVecOp, 1, 3>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else if (5 == kernely_len && 5 == kernelx_len) {
            morph_f32<ErodeVecOp, 1, 5>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else {
            return x86minFilter_f(height, width, inWidthStride, inData, kernelx_len, kernely_len, outWidthStride, outData, 1, border_value);
        }
    } else {
        return x86minFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len, element, outWidthStride, outData, 1, border_value);
    }
}

template <>
::ppl::common::RetCode Erode<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* element,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type,
    float border_value)
{
    if (!inData || !outData || !element || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (!isErodeBorderSupported(border_type)) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (border_type != BORDER_CONSTANT) {
        border_value = FLT_MAX;
    }
    bool flag = true;
    for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {
        if (element[i] != 1) {
            flag = false;
            break;
        }
    }
    if (flag) {
        if (3 == kernely_len && 3 == kernelx_len) {
            morph_f32<ErodeVecOp, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else if (5 == kernely_len && 5 == kernelx_len) {
            morph_f32<ErodeVecOp, 3, 5>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else {
            return x86minFilter_f(height, width, inWidthStride, inData, kernelx_len, kernely_len, outWidthStride, outData, 3, border_value);
        }
    } else {
        return x86minFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len, element, outWidthStride, outData, 3, border_value);
    }
}

template <>
::ppl::common::RetCode Erode<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* element,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type,
    float border_value)
{
    if (!inData || !outData || !element || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (!isErodeBorderSupported(border_type)) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (border_type != BORDER_CONSTANT) {
        border_value = FLT_MAX;
    }
    bool flag = true;
    for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {
        if (element[i] != 1) {
            flag = false;
            break;
        }
    }
    if (flag) {
        if (3 == kernely_len && 3 == kernelx_len) {
            morph_f32<ErodeVecOp, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else if (5 == kernely_len && 5 == kernelx_len) {
            morph_f32<ErodeVecOp, 4, 5>(height, width, inWidthStride, inData, outWidthStride, outData, border_type, border_value);

            return ppl::common::RC_SUCCESS;
        } else {
            return x86minFilter_f(height, width, inWidthStride, inData, kernelx_len, kernely_len, outWidthStride, outData, 4, border_value);
        }
    } else {
        return x86minFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len, element, outWidthStride, outData, 4, border_value);
    }
}
}
}
} // namespace ppl::cv::x86
