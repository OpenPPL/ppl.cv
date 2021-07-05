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

#include "ppl/cv/x86/flip.h"

#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"

#include <assert.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <immintrin.h>

namespace ppl {
namespace cv {
namespace x86 {

::ppl::common::RetCode flip_vertical_f32(
    const float *src,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    float *dst)
{
    if (nullptr == src) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == dst) {
        return ppl::common::RC_INVALID_VALUE;
    }

    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        const float *up_in_ptr = src + i * inWidthStride;
        float *down_out_ptr    = dst + (height - i - 1) * outWidthStride;
        int32_t j              = 0;
        for (; j <= width - 4; j += 4) {
            __m128 up_vec = _mm_loadu_ps(up_in_ptr + j);
            _mm_storeu_ps(down_out_ptr + j, up_vec);
        }
        for (; j < width; ++j) {
            float up        = up_in_ptr[j];
            down_out_ptr[j] = up;
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode flip_horizontal_f32(
    const float *src,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    float *dst)
{
    switch (channels) {
        case 4:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; j++) {
                    __m128 right0 = _mm_loadu_ps(src + i * inWidthStride + (width - j - 1) * channels);
                    _mm_storeu_ps(dst + i * outWidthStride + j * channels, right0);
                }
            }
            break;
        case 3:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; ++j) {
                    __m128 right0 = _mm_loadu_ps(src + i * inWidthStride + (width - j - 1) * channels);
                    _mm_storeu_ps(dst + i * outWidthStride + j * channels, right0);
                }
            }
            break;
        case 1: {
            const int32_t KReverseMask = (0x3 | (0x2 << 2) | (0x1 << 4) | (0x0 << 6));
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 4; j += 4) {
                    __m128 right = _mm_loadu_ps(src + i * inWidthStride + (width - j - 4));
                    right        = _mm_shuffle_ps(right, right, KReverseMask);
                    _mm_storeu_ps(dst + i * outWidthStride + j, right);
                }
                for (; j < width; ++j) {
                    float right                 = src[i * inWidthStride + (width - j - 1)];
                    dst[i * outWidthStride + j] = right;
                }
            }
            break;
        }
        default:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        float left                                               = src[i * inWidthStride + j * channels + c];
                        float right                                              = src[i * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + (width - j - 1) * channels + c] = left;
                        dst[i * outWidthStride + j * channels + c]               = right;
                    }
                }
            }
            break;
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode flip_all_f32(
    const float *src,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    float *dst)
{
    if (nullptr == src) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == dst) {
        return ppl::common::RC_INVALID_VALUE;
    }

    switch (channels) {
        case 4:
            for (int32_t i = 0; i < (height + 1) / 2; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    __m128 up_left    = _mm_loadu_ps(src + i * inWidthStride + j * channels);
                    __m128 up_right   = _mm_loadu_ps(src + i * inWidthStride + (width - j - 1) * channels);
                    __m128 down_left  = _mm_loadu_ps(src + (height - i - 1) * inWidthStride + j * channels);
                    __m128 down_right = _mm_loadu_ps(src + (height - i - 1) * inWidthStride + (width - j - 1) * channels);
                    _mm_storeu_ps(dst + i * outWidthStride + j * channels, down_right);
                    _mm_storeu_ps(dst + i * outWidthStride + (width - j - 1) * channels, down_left);
                    _mm_storeu_ps(dst + (height - i - 1) * outWidthStride + j * channels, up_right);
                    _mm_storeu_ps(dst + (height - i - 1) * outWidthStride + (width - j - 1) * channels, up_left);
                }
            }
            break;
        case 3:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; ++j) {
                    __m128 right0 = _mm_loadu_ps(src + (height - i - 1) * inWidthStride + (width - j - 1) * channels);
                    _mm_storeu_ps(dst + i * outWidthStride + j * channels, right0);
                }
            }
            break;
        case 1: {
            const int32_t KReverseMask = (0x3 | (0x2 << 2) | (0x1 << 4) | (0x0 << 6));
            for (int32_t i = 0; i < height; i++) {
                int32_t j = 0;
                for (; j <= width - 4; j += 4) {
                    __m128 up_left = _mm_loadu_ps(src + i * inWidthStride + j);
                    up_left        = _mm_shuffle_ps(up_left, up_left, KReverseMask);
                    _mm_storeu_ps(dst + (height - 1 - i) * outWidthStride + (width - j - 4), up_left);
                }
                for (; j < width; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        float up_left                                                           = src[i * inWidthStride + j * channels + c];
                        dst[(height - i - 1) * outWidthStride + (width - j - 1) * channels + c] = up_left;
                    }
                }
            }
            break;
        }
        default:
            for (int32_t i = 0; i < (height + 1) / 2; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        float up_left                                                           = src[i * inWidthStride + j * channels + c];
                        float up_right                                                          = src[i * inWidthStride + (width - j - 1) * channels + c];
                        float down_left                                                         = src[(height - i - 1) * inWidthStride + j * channels + c];
                        float down_right                                                        = src[(height - i - 1) * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c]                              = down_right;
                        dst[i * outWidthStride + (width - j - 1) * channels + c]                = down_left;
                        dst[(height - i - 1) * outWidthStride + j * channels + c]               = up_right;
                        dst[(height - i - 1) * outWidthStride + (width - j - 1) * channels + c] = up_left;
                    }
                }
            }
            break;
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode flip_vertical_u8(
    const uint8_t *src,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    uint8_t *dst)
{
    if (nullptr == src) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == dst) {
        return ppl::common::RC_INVALID_VALUE;
    }

    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        const uint8_t *up_in_ptr = src + i * inWidthStride;
        uint8_t *down_out_ptr    = dst + (height - i - 1) * outWidthStride;
        int32_t j                = 0;
        for (; j <= width - 16; j += 16) {
            __m128i up_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(up_in_ptr + j));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(down_out_ptr + j), up_vec);
        }
        for (; j < width; j++) {
            uint8_t up      = up_in_ptr[j];
            down_out_ptr[j] = up;
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode flip_horizontal_u8(
    const uint8_t *src,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    uint8_t *dst)
{
    if (nullptr == src) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == dst) {
        return ppl::common::RC_INVALID_VALUE;
    }

    switch (channels) {
        case 4: {
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; ++j) {
                    uint32_t left                                                              = ((uint32_t *)src)[i * inWidthStride / sizeof(uint32_t) + j];
                    ((uint32_t *)dst)[i * outWidthStride / sizeof(uint32_t) + (width - j - 1)] = left;
                }
            }
            break;
        }
        case 3: {
            __m128i v_index = _mm_setr_epi8(12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, -1);
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 5; j += 5) {
                    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i * inWidthStride + (width - j - 5) * channels));
                    right         = _mm_shuffle_epi8(right, v_index);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i * outWidthStride + j * channels), right);
                }
                for (; j < width; j++) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t right                              = src[i * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c] = right;
                    }
                }
            }
            break;
        }
        case 1: {
            __m128i v_index = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 16; j += 16) {
                    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i * inWidthStride + (width - j - 16)));
                    right         = _mm_shuffle_epi8(right, v_index);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i * outWidthStride + j), right);
                }
                for (; j < width; j++) {
                    uint8_t right               = src[i * inWidthStride + (width - j - 1)];
                    dst[i * outWidthStride + j] = right;
                }
            }
            break;
        }
        default:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t left                                             = src[i * inWidthStride + j * channels + c];
                        uint8_t right                                            = src[i * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + (width - j - 1) * channels + c] = left;
                        dst[i * outWidthStride + j * channels + c]               = right;
                    }
                }
            }
            break;
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode flip_all_u8(
    const uint8_t *src,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    uint8_t *dst)
{
    if (nullptr == src) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == dst) {
        return ppl::common::RC_INVALID_VALUE;
    }

    switch (channels) {
        case 4: {
            const int32_t KReverseMask = (0x3 | (0x2 << 2) | (0x1 << 4) | (0x0 << 6));
            for (int32_t i = 0; i < height; i++) {
                int32_t j = 0;
                for (; j <= width - 4; j += 4) {
                    __m128 up_left = _mm_loadu_ps((const float *)(src + i * inWidthStride + j * channels));
                    up_left        = _mm_shuffle_ps(up_left, up_left, KReverseMask);
                    _mm_storeu_ps((float *)(dst + (height - 1 - i) * outWidthStride + (width - j - 4) * channels), up_left);
                }
                for (; j < width; ++j) {
                    uint32_t up_left                                                                          = ((uint32_t *)src)[i * inWidthStride / sizeof(uint32_t) + j];
                    ((uint32_t *)dst)[(height - i - 1) * outWidthStride / sizeof(uint32_t) + (width - j - 1)] = up_left;
                }
            }
            break;
        }
        case 3: {
            __m128i v_index = _mm_setr_epi8(12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, -1);
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 5; j += 5) {
                    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + (height - i - 1) * inWidthStride + (width - j - 5) * channels));
                    right         = _mm_shuffle_epi8(right, v_index);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i * outWidthStride + j * channels), right);
                }
                for (; j < width; j++) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t right                              = src[(height - i - 1) * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c] = right;
                    }
                }
            }
            break;
        }
        case 1: {
            __m128i v_index = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 16; j += 16) {
                    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + (height - i - 1) * inWidthStride + (width - j - 16)));
                    right         = _mm_shuffle_epi8(right, v_index);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i * outWidthStride + j), right);
                }
                for (; j < width; j++) {
                    uint8_t right               = src[(height - i - 1) * inWidthStride + (width - j - 1)];
                    dst[i * outWidthStride + j] = right;
                }
            }
            break;
        }
        default:
            for (int32_t i = 0; i < (height + 1) / 2; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t up_left                                                         = src[i * inWidthStride + j * channels + c];
                        uint8_t up_right                                                        = src[i * inWidthStride + (width - j - 1) * channels + c];
                        uint8_t down_left                                                       = src[(height - i - 1) * inWidthStride + j * channels + c];
                        uint8_t down_right                                                      = src[(height - i - 1) * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c]                              = down_right;
                        dst[i * outWidthStride + (width - j - 1) * channels + c]                = down_left;
                        dst[(height - i - 1) * outWidthStride + j * channels + c]               = up_right;
                        dst[(height - i - 1) * outWidthStride + (width - j - 1) * channels + c] = up_left;
                    }
                }
            }
            break;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Flip<float, 1>(int32_t height, int32_t width, int32_t inWidthStride, const float *inData, int32_t outWidthStride, float *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_f32(inData, 1, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_f32(inData, 1, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_f32(inData, 1, height, width, inWidthStride, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<float, 2>(int32_t height, int32_t width, int32_t inWidthStride, const float *inData, int32_t outWidthStride, float *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_f32(inData, 2, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_f32(inData, 2, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_f32(inData, 2, height, width, inWidthStride, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<float, 3>(int32_t height, int32_t width, int32_t inWidthStride, const float *inData, int32_t outWidthStride, float *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_f32(inData, 3, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_f32(inData, 3, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_f32(inData, 3, height, width, inWidthStride, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<float, 4>(int32_t height, int32_t width, int32_t inWidthStride, const float *inData, int32_t outWidthStride, float *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_f32(inData, 4, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_f32(inData, 4, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_f32(inData, 4, height, width, inWidthStride, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 1>(int32_t height, int32_t width, int32_t inWidthStride, const uint8_t *inData, int32_t outWidthStride, uint8_t *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_u8(inData, 1, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_u8(inData, 1, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_u8(inData, 1, height, width, inWidthStride, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 2>(int32_t height, int32_t width, int32_t inWidthStride, const uint8_t *inData, int32_t outWidthStride, uint8_t *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_u8(inData, 2, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_u8(inData, 2, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_u8(inData, 2, height, width, inWidthStride, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 3>(int32_t height, int32_t width, int32_t inWidthStride, const uint8_t *inData, int32_t outWidthStride, uint8_t *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_u8(inData, 3, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_u8(inData, 3, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_u8(inData, 3, height, width, inWidthStride, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 4>(int32_t height, int32_t width, int32_t inWidthStride, const uint8_t *inData, int32_t outWidthStride, uint8_t *outData, int32_t flipCode)
{
    if (flipCode == 0) {
        return flip_vertical_u8(inData, 4, height, width, inWidthStride, outWidthStride, outData);
    } else if (flipCode > 0) {
        return flip_horizontal_u8(inData, 4, height, width, inWidthStride, outWidthStride, outData);
    } else { //! flipCode < 0
        return flip_all_u8(inData, 4, height, width, inWidthStride, outWidthStride, outData);
    }
}

}
}
} // namespace ppl::cv::x86
