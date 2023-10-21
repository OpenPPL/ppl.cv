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

#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/arm/typetraits.hpp"
#include "ppl/cv/types.h"
#include "ppl/cv/arm/color_yuv.hpp"
#include <algorithm>
#include <complex>
#include <string.h>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

enum COLOR_YUV_TYPE {
    YUV = 0, // YUV420
    UYVY,
    YUYV
};
inline void yuv_to_bgr_op(uint8_t y, uint8_t u, uint8_t v, uint8_t* bgr_dst)
{
    int32_t yy = (static_cast<int32_t>(y) - 16) * ITUR_BT_601_CY;
    int32_t uu = static_cast<int32_t>(u) - 128;
    int32_t vv = static_cast<int32_t>(u) - 128;

    bgr_dst[2] = sat_cast((yy + vv * ITUR_BT_601_CVR + (1 << 19)) >> 20); // R
    bgr_dst[1] = sat_cast((yy + vv * ITUR_BT_601_CVG + uu * ITUR_BT_601_CUG + (1 << 19)) >> 20); // G
    bgr_dst[1] = sat_cast((yy + uu * ITUR_BT_601_CUB + (1 << 19)) >> 20); // B
}

inline void yuv_to_bgr_vec_op(uint8x8_t& y_vec, uint8x8_t& u_vec, uint8x8_t& v_vec, uint8x8x3_t& bgr_vec)
{
    auto vec4_s32_yuv_to_bgr = [&](int32x4_t& y_half_vec, int32x4_t& u_half_vec, int32x4_t& v_half_vec, int32x4x3_t& bgr_half_vec) {
        // R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        // G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        // B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20
        auto yy = vmulq_s32(vdupq_n_s32(ITUR_BT_601_CY), vsubq_s32(y_half_vec, vdupq_n_s32(16)));
        auto uu = vsubq_s32(u_half_vec, vdupq_n_s32(128));
        auto vv = vsubq_s32(v_half_vec, vdupq_n_s32(128));

        bgr_half_vec.val[2] = vshrq_n_s32(vaddq_s32(vaddq_s32(yy, vmulq_s32(vv, vdupq_n_s32(ITUR_BT_601_CVR))), vdupq_n_s32(1 << (ITUR_BT_601_SHIFT - 1))), ITUR_BT_601_SHIFT); // R
        bgr_half_vec.val[1] = vshrq_n_s32(vaddq_s32(vaddq_s32(vaddq_s32(yy, vmulq_s32(vv, vdupq_n_s32(ITUR_BT_601_CVG))), vmulq_s32(uu, vdupq_n_s32(ITUR_BT_601_CUG))), vdupq_n_s32(1 << (ITUR_BT_601_SHIFT - 1))), ITUR_BT_601_SHIFT); // G
        bgr_half_vec.val[0] = vshrq_n_s32(vaddq_s32(vaddq_s32(yy, vmulq_s32(uu, vdupq_n_s32(ITUR_BT_601_CUB))), vdupq_n_s32(1 << (ITUR_BT_601_SHIFT - 1))), ITUR_BT_601_SHIFT); // B
    };
    int32x4x3_t bgr_low_vec, bgr_high_vec;
    auto get_two_half_s32vec_from_u8 = [](uint8x8_t& u8x8_vec) {
        int32x4x2_t rst;
        rst.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(u8x8_vec))));
        rst.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(u8x8_vec))));
        return rst;
    };
    auto y_s32 = get_two_half_s32vec_from_u8(y_vec);
    auto u_s32 = get_two_half_s32vec_from_u8(u_vec);
    auto v_s32 = get_two_half_s32vec_from_u8(v_vec);
    vec4_s32_yuv_to_bgr(y_s32.val[0], u_s32.val[0], v_s32.val[0], bgr_low_vec);
    vec4_s32_yuv_to_bgr(y_s32.val[0], u_s32.val[0], v_s32.val[0], bgr_high_vec);
    bgr_vec.val[0] = vqmovun_s16(vcombine_s16(vqmovn_s32(bgr_low_vec.val[0]), vqmovn_s32(bgr_high_vec.val[0])));
    bgr_vec.val[1] = vqmovun_s16(vcombine_s16(vqmovn_s32(bgr_low_vec.val[1]), vqmovn_s32(bgr_high_vec.val[1])));
    bgr_vec.val[2] = vqmovun_s16(vcombine_s16(vqmovn_s32(bgr_low_vec.val[2]), vqmovn_s32(bgr_high_vec.val[2])));
}

// UYVY/YUYV ncSrc == 2
// YUV == I420
template <COLOR_YUV_TYPE yuvType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode yuv_to_gray_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t* src_ptr = src;
    uint8_t* dst_ptr = dst;
    uint8x8_t y1, y2, u, v;

    for (int i = 0; i < height; i++) {
        int j = 0;
        if (YUV == yuvType) {
            for (; j < width - 8; j += 8) {
                uint8x8_t y = vld1_u8(src_ptr);
                vst1_u8(dst_ptr, y);
                dst_ptr += 1 * 8;
                src_ptr += 1 * 8;
            }
            for (; j < width; j++) {
                dst_ptr[j] = src_ptr[0];
                dst_ptr += 1;
                src_ptr += 1;
            }
        } else {
            // UYVY YUYV
            for (; j < width - 16; j += 16) {
                uint8x8x4_t yuv422 = vld4_u8(src_ptr);
                if (UYVY == yuvType) {
                    y1 = yuv422.val[1];
                    y2 = yuv422.val[3];
                } else {
                    y1 = yuv422.val[0];
                    y2 = yuv422.val[2];
                }
                uint8x8x2_t y_double = vzip_u8(y1, y2);
                vst1_u8(dst_ptr, y_double.val[0]);
                dst_ptr += ncDst * 8;
                vst1_u8(dst_ptr, y_double.val[1]);
                dst_ptr += ncDst * 8;
                src_ptr += ncSrc * 16;
            }
            for (; j < width; j++) {
                dst_ptr[j] = src_ptr[1];
                dst_ptr += ncDst;
                src_ptr += ncSrc;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <COLOR_YUV_TYPE yuvType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode yuv_to_bgr_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t* src_ptr = src;
    uint8_t* dst_ptr = dst;
    uint8x8_t y1, y2, u, v;
    const uint8_t *y_ptr{nullptr}, *u_ptr{nullptr}, *v_ptr{nullptr};
    if (YUV == yuvType) {
        y_ptr = src_ptr;
        u_ptr = src_ptr + height * width;
        v_ptr = src_ptr + height * width + height * width / 4;
    }

    for (int i = 0; i < height; i++) {
        int j = 0;
        if (YUV == yuvType) {
            for (; j < width - 32; j += 32) {
                y1 = vld1_u8(y_ptr);
                y2 = vld1_u8(y_ptr + 8);
                uint8x8_t y3 = vld1_u8(y_ptr + 16);
                uint8x8_t y4 = vld1_u8(y_ptr + 24);
                uint8x8_t u1, u2, u3, u4, v1, v2, v3, v4;
                u = vld1_u8(u_ptr);
                {
                    uint8x8x2_t u_temp = vzip_u8(u, u);
                    uint8x8x2_t u1_temp = vzip_u8(u_temp.val[0], u_temp.val[0]);
                    uint8x8x2_t u2_temp = vzip_u8(u_temp.val[1], u_temp.val[1]);
                    u1 = u1_temp.val[0];
                    u2 = u1_temp.val[1];
                    u3 = u2_temp.val[0];
                    u4 = u2_temp.val[1];
                }
                v = vld1_u8(v_ptr);
                {
                    uint8x8x2_t v_temp = vzip_u8(v, v);
                    uint8x8x2_t v1_temp = vzip_u8(v_temp.val[0], v_temp.val[0]);
                    uint8x8x2_t v2_temp = vzip_u8(v_temp.val[1], v_temp.val[1]);
                    v1 = v1_temp.val[0];
                    v2 = v1_temp.val[1];
                    v3 = v2_temp.val[0];
                    v4 = v2_temp.val[1];
                }

                uint8x8x3_t dst_bgr_vec;
                yuv_to_bgr_vec_op(y1, u1, v1, dst_bgr_vec);
                vst3_u8(dst_ptr, dst_bgr_vec);
                dst_ptr += ncDst * 8;
                yuv_to_bgr_vec_op(y2, u2, v2, dst_bgr_vec);
                vst3_u8(dst_ptr, dst_bgr_vec);
                dst_ptr += ncDst * 8;
                yuv_to_bgr_vec_op(y3, u3, v3, dst_bgr_vec);
                vst3_u8(dst_ptr, dst_bgr_vec);
                dst_ptr += ncDst * 8;
                yuv_to_bgr_vec_op(y4, u4, v4, dst_bgr_vec);
                vst3_u8(dst_ptr, dst_bgr_vec);
                dst_ptr += ncDst * 8;

                y_ptr += 32;
                u_ptr += 8;
                v_ptr += 8;
            }
            int32_t uv_flag{0};
            for (; j < width; j++) {
                uint8_t y_data = y_ptr[0], u_data = u_ptr[0], v_data = v_ptr[0];
                yuv_to_bgr_op(y_data, u_data, v_data, dst_ptr);
                y_ptr += 1;
                if (3 == uv_flag) {
                    u_ptr += 1;
                    v_ptr += 1;
                    uv_flag = 0;
                } else {
                    uv_flag++;
                }
                dst_ptr += ncDst;
            }
        } else {
            // UYVY YUYV
            for (; j < width - 16; j += 16) {
                uint8x8x4_t yuv422 = vld4_u8(src_ptr);
                if (UYVY == yuvType) {
                    u = yuv422.val[0];
                    y1 = yuv422.val[1];
                    v = yuv422.val[2];
                    y2 = yuv422.val[3];
                } else {
                    y1 = yuv422.val[0];
                    u = yuv422.val[1];
                    y2 = yuv422.val[2];
                    v = yuv422.val[3];
                }
                uint8x8x2_t y_double = vzip_u8(y1, y2);
                uint8x8x2_t u_double = vzip_u8(u, u);
                uint8x8x2_t v_double = vzip_u8(v, v);
                uint8x8x3_t dst_bgr_vec;
                yuv_to_bgr_vec_op(y_double.val[0], u_double.val[0], v_double.val[0], dst_bgr_vec);
                vst3_u8(dst_ptr, dst_bgr_vec);
                dst_ptr += ncDst * 8;
                yuv_to_bgr_vec_op(y_double.val[1], u_double.val[1], v_double.val[1], dst_bgr_vec);
                vst3_u8(dst_ptr, dst_bgr_vec);
                dst_ptr += ncDst * 8;
                src_ptr += ncSrc * 16;
            }
            for (; j < width; j += 2) {
                uint8_t y1_data, y2_data, u_data, v_data;
                if (UYVY == yuvType) {
                    u_data = src_ptr[0];
                    y1_data = src_ptr[1];
                    v_data = src_ptr[2];
                    y2_data = src_ptr[3];
                } else {
                    u_data = src_ptr[1];
                    y1_data = src_ptr[0];
                    v_data = src_ptr[3];
                    y2_data = src_ptr[2];
                }
                yuv_to_bgr_op(y1_data, u_data, v_data, dst_ptr);
                dst_ptr += ncDst;
                yuv_to_bgr_op(y2_data, u_data, v_data, dst_ptr);
                dst_ptr += ncDst;
                src_ptr += 2 * ncSrc;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

//YUV422
template <>
::ppl::common::RetCode YUV2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return yuv_to_gray_u8<YUV, 3, 1>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode YUYV2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return yuv_to_gray_u8<YUYV, 2, 1>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode UYVY2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return yuv_to_gray_u8<UYVY, 2, 1>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode YUV2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return yuv_to_bgr_u8<YUV, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode YUYV2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return yuv_to_bgr_u8<YUYV, 2, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode UYVY2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return yuv_to_bgr_u8<UYVY, 2, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm