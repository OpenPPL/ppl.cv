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

#include "ppl/cv/arm/flip.h"

#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"

#include <string.h>
#include <cmath>
#include <limits.h>
#include <memory>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

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
        float *down_out_ptr = dst + (height - i - 1) * outWidthStride;
        memcpy(down_out_ptr, up_in_ptr, width * sizeof(float));
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
    if (nullptr == src) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == dst) {
        return ppl::common::RC_INVALID_VALUE;
    }
    switch (channels) {
        case 4:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; j++) {
                    float32x4_t right0 = vld1q_f32(src + i * inWidthStride + (width - j - 1) * channels);
                    vst1q_f32(dst + i * outWidthStride + j * channels, right0);
                }
            }
            break;
        case 3:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; ++j) {
                    float32x4_t right0 = vld1q_f32(src + i * inWidthStride + (width - j - 1) * channels);
                    vst1q_f32(dst + i * outWidthStride + j * channels, right0);
                }
            }
            break;
        case 1: {
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; ++j) {
                    float right = src[i * inWidthStride + (width - j - 1)];
                    dst[i * outWidthStride + j] = right;
                }
            }
            break;
        }
        default:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        float left = src[i * inWidthStride + j * channels + c];
                        float right = src[i * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + (width - j - 1) * channels + c] = left;
                        dst[i * outWidthStride + j * channels + c] = right;
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
                    float32x4_t up_left = vld1q_f32(src + i * inWidthStride + j * channels);
                    float32x4_t up_right = vld1q_f32(src + i * inWidthStride + (width - j - 1) * channels);
                    float32x4_t down_left = vld1q_f32(src + (height - i - 1) * inWidthStride + j * channels);
                    float32x4_t down_right = vld1q_f32(src + (height - i - 1) * inWidthStride + (width - j - 1) * channels);
                    vst1q_f32(dst + i * outWidthStride + j * channels, down_right);
                    vst1q_f32(dst + i * outWidthStride + (width - j - 1) * channels, down_left);
                    vst1q_f32(dst + (height - i - 1) * outWidthStride + j * channels, up_right);
                    vst1q_f32(dst + (height - i - 1) * outWidthStride + (width - j - 1) * channels, up_left);
                }
            }
            break;
        case 3:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; ++j) {
                    float32x4_t right0 = vld1q_f32(src + (height - i - 1) * inWidthStride + (width - j - 1) * channels);
                    vst1q_f32(dst + i * outWidthStride + j * channels, right0);
                }
            }
            break;
        case 1: {
            for (int32_t i = 0; i < height; i++) {
                int32_t j = 0;
                for (; j <= width - 4; j += 4) {
                    float32x4_t up_left = vld1q_f32(src + i * inWidthStride + j);
                    float32x4_t v_dst = vrev64q_f32(up_left);
                    v_dst = vcombine_f32(vget_high_f32(v_dst), vget_low_f32(v_dst));
                    vst1q_f32(dst + (height - 1 - i) * outWidthStride + (width - j - 4), v_dst);
                }
                for (; j < width; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        float up_left = src[i * inWidthStride + j * channels + c];
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
                        float up_left = src[i * inWidthStride + j * channels + c];
                        float up_right = src[i * inWidthStride + (width - j - 1) * channels + c];
                        float down_left = src[(height - i - 1) * inWidthStride + j * channels + c];
                        float down_right = src[(height - i - 1) * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c] = down_right;
                        dst[i * outWidthStride + (width - j - 1) * channels + c] = down_left;
                        dst[(height - i - 1) * outWidthStride + j * channels + c] = up_right;
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
        uint8_t *down_out_ptr = dst + (height - i - 1) * outWidthStride;
        memcpy(down_out_ptr, up_in_ptr, width * sizeof(uint8_t));
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
                    uint32_t left = ((uint32_t *)src)[i * inWidthStride / sizeof(uint32_t) + j];
                    ((uint32_t *)dst)[i * outWidthStride / sizeof(uint32_t) + (width - j - 1)] = left;
                }
            }
            break;
        }
        case 3: {
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 16; j += 16) {
                    uint8x16x3_t v_src = vld3q_u8(src + i * inWidthStride + (width - j - 16) * channels);
                    v_src.val[0] = vrev64q_u8(v_src.val[0]);
                    v_src.val[1] = vrev64q_u8(v_src.val[1]);
                    v_src.val[2] = vrev64q_u8(v_src.val[2]);
                    uint8x16x3_t v_dst;
                    v_dst.val[0] = vcombine_u8(vget_high_u8(v_src.val[0]), vget_low_u8(v_src.val[0]));
                    v_dst.val[1] = vcombine_u8(vget_high_u8(v_src.val[1]), vget_low_u8(v_src.val[1]));
                    v_dst.val[2] = vcombine_u8(vget_high_u8(v_src.val[2]), vget_low_u8(v_src.val[2]));
                    vst3q_u8(dst + i * outWidthStride + j * channels, v_dst);
                }
                for (; j < width; j++) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t right = src[i * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c] = right;
                    }
                }
            }
            break;
        }
        case 1: {
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < width; j++) {
                    uint8_t right = src[i * inWidthStride + (width - j - 1)];
                    dst[i * outWidthStride + j] = right;
                }
            }
            break;
        }
        default:
            for (int32_t i = 0; i < height; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t left = src[i * inWidthStride + j * channels + c];
                        uint8_t right = src[i * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + (width - j - 1) * channels + c] = left;
                        dst[i * outWidthStride + j * channels + c] = right;
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
            for (int32_t i = 0; i < height; i++) {
                int32_t j = 0;
                for (; j <= width - 4; j += 4) {
                    uint32x4_t up_left = vld1q_u32((const uint32_t *)(src + i * inWidthStride + j * channels));
                    uint32x4_t v_dst = vrev64q_u32(up_left);
                    v_dst = vcombine_u32(vget_high_u32(v_dst), vget_low_u32(v_dst));
                    vst1q_u32((uint32_t *)(dst + (height - 1 - i) * outWidthStride + (width - j - 4) * channels), v_dst);
                }
                for (; j < width; ++j) {
                    uint32_t up_left = ((uint32_t *)src)[i * inWidthStride / sizeof(uint32_t) + j];
                    ((uint32_t *)dst)[(height - i - 1) * outWidthStride / sizeof(uint32_t) + (width - j - 1)] = up_left;
                }
            }
            break;
        }
        case 3: {
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 16; j += 16) {
                    uint8x16x3_t v_src = vld3q_u8(src + (height - i - 1) * inWidthStride + (width - j - 16) * channels);
                    v_src.val[0] = vrev64q_u8(v_src.val[0]);
                    v_src.val[1] = vrev64q_u8(v_src.val[1]);
                    v_src.val[2] = vrev64q_u8(v_src.val[2]);
                    uint8x16x3_t v_dst;
                    v_dst.val[0] = vcombine_u8(vget_high_u8(v_src.val[0]), vget_low_u8(v_src.val[0]));
                    v_dst.val[1] = vcombine_u8(vget_high_u8(v_src.val[1]), vget_low_u8(v_src.val[1]));
                    v_dst.val[2] = vcombine_u8(vget_high_u8(v_src.val[2]), vget_low_u8(v_src.val[2]));
                    vst3q_u8(dst + i * outWidthStride + j * channels, v_dst);
                }
                for (; j < width; j++) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t right = src[(height - i - 1) * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c] = right;
                    }
                }
            }
            break;
        }
        case 1: {
            for (int32_t i = 0; i < height; ++i) {
                int32_t j = 0;
                for (; j <= width - 16; j += 16) {
                    uint8x16_t right = vld1q_u8(src + (height - i - 1) * inWidthStride + (width - j - 16));
                    uint8x16_t v_dst = vrev64q_u8(right);
                    v_dst = vcombine_u8(vget_high_u8(v_dst), vget_low_u8(v_dst));
                    vst1q_u8(dst + i * outWidthStride + j, v_dst);
                }
                for (; j < width; j++) {
                    uint8_t right = src[(height - i - 1) * inWidthStride + (width - j - 1)];
                    dst[i * outWidthStride + j] = right;
                }
            }
            break;
        }
        default:
            for (int32_t i = 0; i < (height + 1) / 2; ++i) {
                for (int32_t j = 0; j < (width + 1) / 2; ++j) {
                    for (int32_t c = 0; c < channels; ++c) {
                        uint8_t up_left = src[i * inWidthStride + j * channels + c];
                        uint8_t up_right = src[i * inWidthStride + (width - j - 1) * channels + c];
                        uint8_t down_left = src[(height - i - 1) * inWidthStride + j * channels + c];
                        uint8_t down_right = src[(height - i - 1) * inWidthStride + (width - j - 1) * channels + c];
                        dst[i * outWidthStride + j * channels + c] = down_right;
                        dst[i * outWidthStride + (width - j - 1) * channels + c] = down_left;
                        dst[(height - i - 1) * outWidthStride + j * channels + c] = up_right;
                        dst[(height - i - 1) * outWidthStride + (width - j - 1) * channels + c] = up_left;
                    }
                }
            }
            break;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Flip<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
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
::ppl::common::RetCode Flip<float, 2>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
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
::ppl::common::RetCode Flip<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
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
::ppl::common::RetCode Flip<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
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
::ppl::common::RetCode Flip<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
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
::ppl::common::RetCode Flip<uint8_t, 2>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
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
::ppl::common::RetCode Flip<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
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
::ppl::common::RetCode Flip<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
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
} // namespace ppl::cv::arm
