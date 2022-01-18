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
#ifndef __ST_HPC_PPL_CV_AARCH64_COLOR_YUV_SIMD_HPP__
#define __ST_HPC_PPL_CV_AARCH64_COLOR_YUV_SIMD_HPP__
#include <arm_neon.h>
#include "ppl/cv/types.h"
#include "ppl/cv/arm/color_yuv.hpp"
#include <algorithm>
#include <complex>
#include <string.h>

namespace ppl {
namespace cv {
namespace arm {

enum YUV_TYPE {
    YUV_YU12 = 0, //yyyyyyyy uu vv
    YUV_I420 = YUV_YU12,
    YUV_YV12 = 1, //yyyyyyyy vv uu
    YUV_NV12 = 2, //yyyyyyyy uvuv
    YUV_NV21 = 3, //yyyyyyyy vuvu
};

#define USE_QUANTIZED

struct YUV4202RGB_u8_neon {
    YUV4202RGB_u8_neon(int32_t _bIdx)
        : bIdx(_bIdx)
    {
        v_c0 = vdupq_n_s32(ITUR_BT_601_CVR);
        v_c1 = vdupq_n_s32(ITUR_BT_601_CVG);
        v_c2 = vdupq_n_s32(ITUR_BT_601_CUG);
        v_c3 = vdupq_n_s32(ITUR_BT_601_CUB);
        v_c4 = vdupq_n_s32(ITUR_BT_601_CY);
        v_zero = vdupq_n_s32(0);
        v128 = vdupq_n_s16(128);
        v16 = vdupq_n_s32(16);
        v_shift = vdupq_n_s32(1 << (ITUR_BT_601_SHIFT - 1));
    }
    inline void process(int32x4x2_t ruv_zip, int32x4x2_t guv_zip, int32x4x2_t buv_zip, int16x8_t vec_y_s16, uint8x8x3_t& v_dst) const
    {
        int32x4_t vec_ylo_s32 = vmovl_s16(vget_low_s16(vec_y_s16));
        vec_ylo_s32 = vmulq_s32(vmaxq_s32(vsubq_s32(vec_ylo_s32, v16), v_zero), v_c4);
        int32x4_t v_b0 = vshrq_n_s32(vaddq_s32(vec_ylo_s32, ruv_zip.val[0]), ITUR_BT_601_SHIFT);
        int32x4_t v_g0 = vshrq_n_s32(vaddq_s32(vec_ylo_s32, guv_zip.val[0]), ITUR_BT_601_SHIFT);
        int32x4_t v_r0 = vshrq_n_s32(vaddq_s32(vec_ylo_s32, buv_zip.val[0]), ITUR_BT_601_SHIFT);

        int32x4_t vec_yhi_s32 = vmovl_s16(vget_high_s16(vec_y_s16));
        vec_yhi_s32 = vmulq_s32(vmaxq_s32(vsubq_s32(vec_yhi_s32, v16), v_zero), v_c4);
        int32x4_t v_b1 = vshrq_n_s32(vaddq_s32(vec_yhi_s32, ruv_zip.val[1]), ITUR_BT_601_SHIFT);
        int32x4_t v_g1 = vshrq_n_s32(vaddq_s32(vec_yhi_s32, guv_zip.val[1]), ITUR_BT_601_SHIFT);
        int32x4_t v_r1 = vshrq_n_s32(vaddq_s32(vec_yhi_s32, buv_zip.val[1]), ITUR_BT_601_SHIFT);
        if (bIdx == 0) { //bgr
            v_dst.val[2] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_b0), vqmovn_s32(v_b1)));
            v_dst.val[1] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_g0), vqmovn_s32(v_g1)));
            v_dst.val[0] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_r0), vqmovn_s32(v_r1)));
        } else { //rgb
            v_dst.val[0] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_b0), vqmovn_s32(v_b1)));
            v_dst.val[1] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_g0), vqmovn_s32(v_g1)));
            v_dst.val[2] = vqmovun_s16(vcombine_s16(vqmovn_s32(v_r0), vqmovn_s32(v_r1)));
        }
    }

    void convert_per_2rows(
        int32_t width,
        const uint8_t* y1,
        const uint8_t* u1,
        const uint8_t* v1,
        uint8_t* row1,
        uint8_t* row2,
        int32_t stride) const
    {
        const uint8_t* y2 = y1 + stride;
        int32_t i = 0;
        for (; i <= width / 2 - 8; i += 8, row1 += 6 * 8, row2 += 6 * 8) {
            uint8x8x3_t v_dst1;
            uint8x8x3_t v_dst2;

            uint8x8_t vec_u_u8 = vld1_u8(u1 + i);
            uint8x8_t vec_v_u8 = vld1_u8(v1 + i);

            int16x8_t vec_u_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_u_u8));
            int16x8_t vec_v_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_v_u8));
            vec_u_s16 = vsubq_s16(vec_u_s16, v128);
            vec_v_s16 = vsubq_s16(vec_v_s16, v128);

            int32x4_t vec_ulo_s32 = vmovl_s16(vget_low_s16(vec_u_s16));
            int32x4_t vec_vlo_s32 = vmovl_s16(vget_low_s16(vec_v_s16));

            int32x4_t ruv1 = vaddq_s32(vmulq_s32(vec_vlo_s32, v_c0), v_shift);
            int32x4_t guv1 = vaddq_s32(vmulq_s32(vec_vlo_s32, v_c1), v_shift);
            guv1 = vaddq_s32(vmulq_s32(vec_ulo_s32, v_c2), guv1);
            int32x4_t buv1 = vaddq_s32(vmulq_s32(vec_ulo_s32, v_c3), v_shift);

            int32x4x2_t ruv_zip = vzipq_s32(ruv1, ruv1);
            int32x4x2_t guv_zip = vzipq_s32(guv1, guv1);
            int32x4x2_t buv_zip = vzipq_s32(buv1, buv1);

            uint8x8_t vec_y1_u8 = vld1_u8(y1 + 2 * i);
            int16x8_t vec_y1_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y1_u8));
            process(ruv_zip, guv_zip, buv_zip, vec_y1_s16, v_dst1);

            vst3_u8(row1, v_dst1);

            uint8x8_t vec_y2_u8 = vld1_u8(y2 + 2 * i);

            int16x8_t vec_y2_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y2_u8));
            process(ruv_zip, guv_zip, buv_zip, vec_y2_s16, v_dst2);

            vst3_u8(row2, v_dst2);

            int32x4_t vec_uhi_s32 = vmovl_s16(vget_high_s16(vec_u_s16));
            int32x4_t vec_vhi_s32 = vmovl_s16(vget_high_s16(vec_v_s16));

            int32x4_t ruv2 = vaddq_s32(vmulq_s32(vec_vhi_s32, v_c0), v_shift);
            int32x4_t guv2 = vaddq_s32(vmulq_s32(vec_vhi_s32, v_c1), v_shift);
            guv2 = vaddq_s32(vmulq_s32(vec_uhi_s32, v_c2), guv2);
            int32x4_t buv2 = vaddq_s32(vmulq_s32(vec_uhi_s32, v_c3), v_shift);

            int32x4x2_t ruv2_zip = vzipq_s32(ruv2, ruv2);
            int32x4x2_t guv2_zip = vzipq_s32(guv2, guv2);
            int32x4x2_t buv2_zip = vzipq_s32(buv2, buv2);

            vec_y1_u8 = vld1_u8(y1 + 2 * i + 8);
            vec_y1_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y1_u8));
            process(ruv2_zip, guv2_zip, buv2_zip, vec_y1_s16, v_dst1);

            vst3_u8(row1 + 24, v_dst1);

            vec_y2_u8 = vld1_u8(y2 + 2 * i + 8);
            vec_y2_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y2_u8));
            process(ruv2_zip, guv2_zip, buv2_zip, vec_y2_s16, v_dst2);

            vst3_u8(row2 + 24, v_dst2);
        }
        for (; i < width / 2; i += 1, row1 += 6, row2 += 6) {
            int32_t u = int32_t(u1[i]) - 128;
            int32_t v = int32_t(v1[i]) - 128;

            int32_t ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
            int32_t guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
            int32_t buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

            int32_t y00 = MAX(0, int32_t(y1[2 * i]) - 16) * ITUR_BT_601_CY;
            row1[2 - bIdx] = sat_cast((y00 + ruv) >> ITUR_BT_601_SHIFT);
            row1[1] = sat_cast((y00 + guv) >> ITUR_BT_601_SHIFT);
            row1[bIdx] = sat_cast((y00 + buv) >> ITUR_BT_601_SHIFT);

            int32_t y01 = MAX(0, int32_t(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
            row1[5 - bIdx] = sat_cast((y01 + ruv) >> ITUR_BT_601_SHIFT);
            row1[4] = sat_cast((y01 + guv) >> ITUR_BT_601_SHIFT);
            row1[3 + bIdx] = sat_cast((y01 + buv) >> ITUR_BT_601_SHIFT);

            int32_t y10 = MAX(0, int32_t(y2[2 * i]) - 16) * ITUR_BT_601_CY;
            row2[2 - bIdx] = sat_cast((y10 + ruv) >> ITUR_BT_601_SHIFT);
            row2[1] = sat_cast((y10 + guv) >> ITUR_BT_601_SHIFT);
            row2[bIdx] = sat_cast((y10 + buv) >> ITUR_BT_601_SHIFT);

            int32_t y11 = MAX(0, int32_t(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
            row2[5 - bIdx] = sat_cast((y11 + ruv) >> ITUR_BT_601_SHIFT);
            row2[4] = sat_cast((y11 + guv) >> ITUR_BT_601_SHIFT);
            row2[3 + bIdx] = sat_cast((y11 + buv) >> ITUR_BT_601_SHIFT);
        }
    }

    //nv12 or nv21 to bgr or rgb
    void convert_from_yuv420sp_layout(
        int32_t height,
        int32_t width,
        int32_t yStride,
        const uint8_t* y,
        int32_t uvStride,
        const uint8_t* uv,
        int32_t outWidthStride,
        uint8_t* dst,
        bool isUV) const
    {
        const uint8_t* y1 = y;
        uint8_t* u1 = (uint8_t*)malloc(width / 2);
        uint8_t* v1 = (uint8_t*)malloc(width / 2);
        for (int32_t j = 0; j < height; j += 2, y1 += yStride * 2, uv += uvStride) {
            uint8_t* row1 = dst + j * outWidthStride;
            uint8_t* row2 = dst + (j + 1) * outWidthStride;
            if (isUV) {
                for (int32_t i = 0; i < width / 2; i++) {
                    u1[i] = uv[2 * i];
                    v1[i] = uv[2 * i + 1];
                }
            } else {
                for (int32_t i = 0; i < width / 2; i++) {
                    v1[i] = uv[2 * i];
                    u1[i] = uv[2 * i + 1];
                }
            }

            convert_per_2rows(width, y1, u1, v1, row1, row2, yStride);
        }
        free(u1);
        free(v1);
    }

    //i420 or yv12 to bgr or rgb
    void convert_from_yuv420_continuous_layout(
        int32_t height,
        int32_t width,
        const uint8_t* y1,
        const uint8_t* u1,
        const uint8_t* v1,
        uint8_t* dst,
        int32_t stride,
        int32_t ustepIdx,
        int32_t vstepIdx,
        int32_t outWidthStride) const
    {
        int32_t uvsteps[2] = {width / 2, stride - width / 2};
        int32_t usIdx = ustepIdx, vsIdx = vstepIdx;

        for (int32_t j = 0; j < height; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1]) {
            uint8_t* row1 = dst + j * outWidthStride;
            uint8_t* row2 = dst + (j + 1) * outWidthStride;
            convert_per_2rows(width, y1, u1, v1, row1, row2, stride);
        }
    }

    //i420 to bgr or rgb
    void convert_from_yuv420_seperate_layout(
        int32_t height,
        int32_t width,
        const uint8_t* y1,
        const uint8_t* u1,
        const uint8_t* v1,
        uint8_t* dst,
        int32_t yStride,
        int32_t ustride,
        int32_t vstride,
        int32_t outWidthStride) const
    {
        for (int32_t j = 0; j < height; j += 2, y1 += 2 * yStride, u1 += ustride, v1 += vstride) {
            uint8_t* row1 = dst + j * outWidthStride;
            uint8_t* row2 = dst + (j + 1) * outWidthStride;
            convert_per_2rows(width, y1, u1, v1, row1, row2, yStride);
        }
    }
    int32_t bIdx;
    int32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_zero, v_shift, v16;
    int16x8_t v128;
};

//convert bgr or rgb to yuv420
struct RGBtoYUV420p_u8_neon {
    RGBtoYUV420p_u8_neon(int32_t _bIdx)
        : bIdx(_bIdx)
    {
        v_c0 = vdupq_n_s32(ITUR_BT_601_CRY);
        v_c1 = vdupq_n_s32(ITUR_BT_601_CGY);
        v_c2 = vdupq_n_s32(ITUR_BT_601_CBY);
        v_zero = vdupq_n_s32(0);
        v16 = vdupq_n_s32(16);
        v_shift16 = vdupq_n_s32(16 << (ITUR_BT_601_SHIFT));
        v_shift128 = vdupq_n_s32(128 << (ITUR_BT_601_SHIFT));
        v_halfshift = vdupq_n_s32(1 << (ITUR_BT_601_SHIFT - 1));
    }

    inline void getUV(int32x4_t r, int32x4_t g, int32x4_t b, int32x4_t& u, int32x4_t& v) const
    {
        u = vaddq_s32(v_shift128, v_halfshift);
        u = vmlaq_s32(u, r, vdupq_n_s32(ITUR_BT_601_CRU));
        u = vmlaq_s32(u, g, vdupq_n_s32(ITUR_BT_601_CGU));
        u = vmlaq_s32(u, b, vdupq_n_s32(ITUR_BT_601_CBU));
        u = vshrq_n_s32(u, ITUR_BT_601_SHIFT);

        v = vaddq_s32(v_shift128, v_halfshift);
        v = vmlaq_s32(v, r, vdupq_n_s32(ITUR_BT_601_CBU));
        v = vmlaq_s32(v, g, vdupq_n_s32(ITUR_BT_601_CGV));
        v = vmlaq_s32(v, b, vdupq_n_s32(ITUR_BT_601_CBV));
        v = vshrq_n_s32(v, ITUR_BT_601_SHIFT);
    }
    inline void getY(int32x4_t r, int32x4_t g, int32x4_t b, int32x4_t& y) const
    {
        y = vaddq_s32(v_shift16, v_halfshift);
        y = vmlaq_s32(y, r, v_c0);
        y = vmlaq_s32(y, g, v_c1);
        y = vmlaq_s32(y, b, v_c2);
        y = vshrq_n_s32(y, ITUR_BT_601_SHIFT);
    }
    inline void process1(int16x8_t r00_s16, int16x8_t g00_s16, int16x8_t b00_s16, uint8x8_t& y00_u8) const
    {
        int32x4_t r00_lo_s32 = vmovl_s16(vget_low_s16(r00_s16));
        int32x4_t r00_hi_s32 = vmovl_s16(vget_high_s16(r00_s16));

        int32x4_t g00_lo_s32 = vmovl_s16(vget_low_s16(g00_s16));
        int32x4_t g00_hi_s32 = vmovl_s16(vget_high_s16(g00_s16));

        int32x4_t b00_lo_s32 = vmovl_s16(vget_low_s16(b00_s16));
        int32x4_t b00_hi_s32 = vmovl_s16(vget_high_s16(b00_s16));

        int32x4_t y0_s32, y1_s32;
        getY(r00_lo_s32, g00_lo_s32, b00_lo_s32, y0_s32);
        getY(r00_hi_s32, g00_hi_s32, b00_hi_s32, y1_s32);
        y00_u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(y0_s32), vqmovn_s32(y1_s32)));
    }
    inline void process(int16x8_t r00_s16, int16x8_t g00_s16, int16x8_t b00_s16, uint8x8_t& y00_u8, int32x4_t& u_s32, int32x4_t& v_s32) const
    {
        int32x4_t r00_lo_s32 = vmovl_s16(vget_low_s16(r00_s16));
        int32x4_t r00_hi_s32 = vmovl_s16(vget_high_s16(r00_s16));

        int32x4_t g00_lo_s32 = vmovl_s16(vget_low_s16(g00_s16));
        int32x4_t g00_hi_s32 = vmovl_s16(vget_high_s16(g00_s16));

        int32x4_t b00_lo_s32 = vmovl_s16(vget_low_s16(b00_s16));
        int32x4_t b00_hi_s32 = vmovl_s16(vget_high_s16(b00_s16));

        int32x4_t y0_s32, y1_s32;
        getY(r00_lo_s32, g00_lo_s32, b00_lo_s32, y0_s32);
        getY(r00_hi_s32, g00_hi_s32, b00_hi_s32, y1_s32);
        y00_u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(y0_s32), vqmovn_s32(y1_s32)));

        int32x4_t u0_s32, u1_s32, v0_s32, v1_s32;
        getUV(r00_lo_s32, g00_lo_s32, b00_lo_s32, u0_s32, v0_s32);
        getUV(r00_hi_s32, g00_hi_s32, b00_hi_s32, u1_s32, v1_s32);

        int32x4x2_t u_unpack = vuzpq_s32(u0_s32, u1_s32);
        u_s32 = u_unpack.val[0];

        int32x4x2_t v_unpack = vuzpq_s32(v0_s32, v1_s32);
        v_s32 = v_unpack.val[0];
    }

    void convert_per_2rows(
        int32_t width,
        int32_t cn,
        const uint8_t* row0,
        const uint8_t* row1,
        uint8_t* y,
        uint8_t* u,
        uint8_t* v,
        int32_t yStride) const
    {
        int32_t w = width;

        int32_t j = 0, k = 0;
        for (; j <= w * cn - 2 * cn * 8; j += 2 * cn * 8, k += 8) {
            uint8x8_t u_dst;
            uint8x8_t v_dst;
            uint8x16x3_t v_row0_u8;
            uint8x16x3_t v_row1_u8;

            if (cn == 3) {
                if (bIdx == 0) {
                    uint8x16x3_t v_src = vld3q_u8(row0 + j);
                    v_row0_u8.val[0] = v_src.val[0];
                    v_row0_u8.val[1] = v_src.val[1];
                    v_row0_u8.val[2] = v_src.val[2];
                    uint8x16x3_t v_src1 = vld3q_u8(row1 + j);
                    v_row1_u8.val[0] = v_src1.val[0];
                    v_row1_u8.val[1] = v_src1.val[1];
                    v_row1_u8.val[2] = v_src1.val[2];
                } else {
                    uint8x16x3_t v_src = vld3q_u8(row0 + j);
                    v_row0_u8.val[0] = v_src.val[2];
                    v_row0_u8.val[1] = v_src.val[1];
                    v_row0_u8.val[2] = v_src.val[0];
                    uint8x16x3_t v_src1 = vld3q_u8(row1 + j);
                    v_row1_u8.val[0] = v_src1.val[2];
                    v_row1_u8.val[1] = v_src1.val[1];
                    v_row1_u8.val[2] = v_src1.val[0];
                }
            } else {
                if (bIdx == 0) {
                    uint8x16x4_t v_src = vld4q_u8(row0 + j);
                    v_row0_u8.val[0] = v_src.val[0];
                    v_row0_u8.val[1] = v_src.val[1];
                    v_row0_u8.val[2] = v_src.val[2];
                    uint8x16x4_t v_src1 = vld4q_u8(row1 + j);
                    v_row1_u8.val[0] = v_src1.val[0];
                    v_row1_u8.val[1] = v_src1.val[1];
                    v_row1_u8.val[2] = v_src1.val[2];
                } else {
                    uint8x16x4_t v_src = vld4q_u8(row0 + j);
                    v_row0_u8.val[0] = v_src.val[2];
                    v_row0_u8.val[1] = v_src.val[1];
                    v_row0_u8.val[2] = v_src.val[0];
                    uint8x16x4_t v_src1 = vld4q_u8(row1 + j);
                    v_row1_u8.val[0] = v_src1.val[2];
                    v_row1_u8.val[1] = v_src1.val[1];
                    v_row1_u8.val[2] = v_src1.val[0];
                }
            }

            uint8x8_t y00_u8, y01_u8;
            int32x4_t u0_s32, v0_s32, u1_s32, v1_s32;

            int16x8_t r00_lo_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_row0_u8.val[2])));
            int16x8_t g00_lo_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_row0_u8.val[1])));
            int16x8_t b00_lo_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_row0_u8.val[0])));
            process(r00_lo_s16, g00_lo_s16, b00_lo_s16, y00_u8, u0_s32, v0_s32);

            int16x8_t r00_hi_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_row0_u8.val[2])));
            int16x8_t g00_hi_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_row0_u8.val[1])));
            int16x8_t b00_hi_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_row0_u8.val[0])));
            process(r00_hi_s16, g00_hi_s16, b00_hi_s16, y01_u8, u1_s32, v1_s32);

            vst1_u8(y + 2 * k, y00_u8);
            vst1_u8(y + 2 * k + 8, y01_u8);
            u_dst = vqmovun_s16(vcombine_s16(vqmovn_s32(u0_s32), vqmovn_s32(u1_s32)));
            v_dst = vqmovun_s16(vcombine_s16(vqmovn_s32(v0_s32), vqmovn_s32(v1_s32)));
            vst1_u8(u + k, u_dst);
            vst1_u8(v + k, v_dst);

            r00_lo_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_row1_u8.val[2])));
            g00_lo_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_row1_u8.val[1])));
            b00_lo_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_row1_u8.val[0])));
            process1(r00_lo_s16, g00_lo_s16, b00_lo_s16, y00_u8);

            r00_hi_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_row1_u8.val[2])));
            g00_hi_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_row1_u8.val[1])));
            b00_hi_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_row1_u8.val[0])));
            process1(r00_hi_s16, g00_hi_s16, b00_hi_s16, y01_u8);

            vst1_u8(y + yStride + 2 * k, y00_u8);
            vst1_u8(y + yStride + 2 * k + 8, y01_u8);
        }

        for (; j < w * cn; j += 2 * cn, k++) {
            int32_t r00 = row0[2 - bIdx + j];
            int32_t g00 = row0[1 + j];
            int32_t b00 = row0[bIdx + j];
            int32_t r01 = row0[2 - bIdx + cn + j];
            int32_t g01 = row0[1 + cn + j];
            int32_t b01 = row0[bIdx + cn + j];
            int32_t r10 = row1[2 - bIdx + j];
            int32_t g10 = row1[1 + j];
            int32_t b10 = row1[bIdx + j];
            int32_t r11 = row1[2 - bIdx + cn + j];
            int32_t g11 = row1[1 + cn + j];
            int32_t b11 = row1[bIdx + cn + j];

            const int32_t shifted16 = (16 << ITUR_BT_601_SHIFT);
            const int32_t halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
            int32_t y00 = ITUR_BT_601_CRY * r00 + ITUR_BT_601_CGY * g00 + ITUR_BT_601_CBY * b00 + halfShift + shifted16;
            int32_t y01 = ITUR_BT_601_CRY * r01 + ITUR_BT_601_CGY * g01 + ITUR_BT_601_CBY * b01 + halfShift + shifted16;
            int32_t y10 = ITUR_BT_601_CRY * r10 + ITUR_BT_601_CGY * g10 + ITUR_BT_601_CBY * b10 + halfShift + shifted16;
            int32_t y11 = ITUR_BT_601_CRY * r11 + ITUR_BT_601_CGY * g11 + ITUR_BT_601_CBY * b11 + halfShift + shifted16;

            y[2 * k + 0] = sat_cast(y00 >> ITUR_BT_601_SHIFT);
            y[2 * k + 1] = sat_cast(y01 >> ITUR_BT_601_SHIFT);
            y[2 * k + yStride + 0] = sat_cast(y10 >> ITUR_BT_601_SHIFT);
            y[2 * k + yStride + 1] = sat_cast(y11 >> ITUR_BT_601_SHIFT);

            const int32_t shifted128 = (128 << ITUR_BT_601_SHIFT);
            int32_t u00 = ITUR_BT_601_CRU * r00 + ITUR_BT_601_CGU * g00 + ITUR_BT_601_CBU * b00 + halfShift + shifted128;
            int32_t v00 = ITUR_BT_601_CBU * r00 + ITUR_BT_601_CGV * g00 + ITUR_BT_601_CBV * b00 + halfShift + shifted128;

            u[k] = sat_cast(u00 >> ITUR_BT_601_SHIFT);
            v[k] = sat_cast(v00 >> ITUR_BT_601_SHIFT);
        }
    }

    //to I420 split
    void operator()(
        int32_t height,
        int32_t width,
        int32_t scn,
        const uint8_t* src,
        uint8_t* y,
        uint8_t* u,
        uint8_t* v,
        int32_t inWidthStride,
        int32_t yStride,
        int32_t uStride,
        int32_t vStride) const
    {
        int32_t h = height;

        for (int32_t i = 0; i < h / 2; i++) {
            const uint8_t* row0 = src + i * 2 * inWidthStride;
            const uint8_t* row1 = src + (i * 2 + 1) * inWidthStride;
            convert_per_2rows(width, scn, row0, row1, y + i * 2 * yStride, u + i * uStride, v + i * vStride, yStride);
        }
    }

    //to I420  continuous
    void operator()(
        int32_t height,
        int32_t width,
        int32_t scn,
        const uint8_t* src,
        uint8_t* y,
        uint8_t* u,
        uint8_t* v,
        int32_t inWidthStride,
        int32_t ystride) const
    {
        int32_t w = width;
        int32_t h = height;

        for (int32_t i = 0; i < h / 2; i++) {
            const uint8_t* row0 = src + i * 2 * inWidthStride;
            const uint8_t* row1 = src + (i * 2 + 1) * inWidthStride;
            convert_per_2rows(width, scn, row0, row1, y + i * 2 * ystride, u + i / 2 * ystride + (i % 2) * (w / 2), v + i / 2 * ystride + (i % 2) * (w / 2), ystride);
        }
    }

    //to NV12 or NV21
    void operator()(
        int32_t height,
        int32_t width,
        int32_t scn,
        int32_t inWidthStride,
        const uint8_t* src,
        int32_t yStride,
        uint8_t* y,
        int32_t uvStride,
        uint8_t* uv,
        bool isUV) const
    {
        int32_t h = height;

        uint8_t* u = new uint8_t[width];
        uint8_t* v = new uint8_t[width];

        for (int32_t i = 0; i < h / 2; i++, y += yStride * 2, uv += uvStride) {
            if (isUV) {
                for (int32_t i = 0; i < width / 2; i++) {
                    u[i] = uv[2 * i];
                    v[i] = uv[2 * i + 1];
                }
            } else {
                for (int32_t i = 0; i < width / 2; i++) {
                    v[i] = uv[2 * i];
                    u[i] = uv[2 * i + 1];
                }
            }
            const uint8_t* row0 = src + i * 2 * inWidthStride;
            const uint8_t* row1 = src + (i * 2 + 1) * inWidthStride;
            convert_per_2rows(width, scn, row0, row1, y, u, v, yStride);

            if (isUV) {
                for (int32_t i = 0; i < width / 2; i++) {
                    uv[2 * i] = u[i];
                    uv[2 * i + 1] = v[i];
                }
            } else {
                for (int32_t i = 0; i < width / 2; i++) {
                    uv[2 * i] = v[i];
                    uv[2 * i + 1] = u[i];
                }
            }
        }
        delete[] u;
        delete[] v;
    }

    int32_t bIdx;
    int32x4_t v_c0, v_c1, v_c2, v_zero, v16, v_shift16, v_shift128, v_halfshift;
};

//video range
//yv12,yu12,nv12,nv12 to rgb,rgba,bgr,bgra
template <YUV_TYPE yuvType, int32_t dst_c, int32_t b_idx>
void yuv420_to_bgr_uchar_video_range(
    int32_t h,
    int32_t w,
    int32_t yStride,
    const uint8_t* y_ptr,
    int32_t uStride,
    const uint8_t* u_ptr,
    int32_t vStride,
    const uint8_t* v_ptr,
    int32_t rgbStride,
    uint8_t* rgb);

//bgr,rgb,bgra,rgba to i420,nv12,nv21
template <int32_t b_idx, int32_t src_c, YUV_TYPE yuvType>
void bgr_to_yuv420_uchar_video_range(
    int32_t h,
    int32_t w,
    int32_t rgbStride,
    const uint8_t* rgb,
    int32_t yStride,
    uint8_t* y_ptr,
    int32_t uStride,
    uint8_t* u_ptr,
    int32_t vStride,
    uint8_t* v_ptr);

}
}
} // namespace ppl::cv::arm

#endif //__ST_HPC_PPL_CV_AARCH64_COLOR_YUV_SIMD_HPP__