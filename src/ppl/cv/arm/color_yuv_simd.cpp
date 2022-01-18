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

#include "ppl/cv/arm/color_yuv_simd.hpp"
#include "ppl/cv/arm/typetraits.hpp"

namespace ppl {
namespace cv {
namespace arm {

//i420,nv12,nv21 to bgr,rgb,bgra,rgba
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
    uint8_t* rgb)
{
    const uint8_t* yptr = y_ptr;
    const uint8_t* uptr = u_ptr;
    const uint8_t* vptr = v_ptr;

    int16x8_t _vCUB_s16 = vdupq_n_s16(ITUR_BT_601_CUB_6);
    int16x8_t _vCUG_s16 = vdupq_n_s16(ITUR_BT_601_CUG_6);
    int16x8_t _vCVG_s16 = vdupq_n_s16(ITUR_BT_601_CVG_6);
    int16x8_t _vCVR_s16 = vdupq_n_s16(ITUR_BT_601_CVR_6);
    int16x8_t _vCY_s16 = vdupq_n_s16(ITUR_BT_601_CY_6);
    int16x8_t _vShift_s16 = vdupq_n_s16(1 << (ITUR_BT_601_SHIFT_6 - 1));
    int16x8_t _v128_s16 = vdupq_n_s16(128);
    int16x8_t _v16_s16 = vdupq_n_s16(16);
    int16x8_t _v0_s16 = vdupq_n_s16(0);
    uint8x8_t _v255_u8 = vdup_n_u8(255);
    const uint8_t alpha = 255;

    for (int32_t y = 0; y < h; y += 2) {
        const uint8_t* y0 = yptr;
        const uint8_t* y1 = yptr + yStride;
        uint8_t* rgb0 = rgb;
        uint8_t* rgb1 = rgb + rgbStride;
        const uint8_t* u0 = uptr; //for yu12 or yv12
        const uint8_t* v0 = vptr;
        const uint8_t* uv = uptr; //or nv12
        const uint8_t* vu = vptr; //or nv21
        int32_t remain = w;

        for (; remain > 16; remain -= 16) {
            uint8x8_t vec_u_u8;
            uint8x8_t vec_v_u8;
            if ((YUV_I420 == yuvType) || (YUV_YV12 == yuvType)) {
                vec_u_u8 = vld1_u8(u0);
                vec_v_u8 = vld1_u8(v0);
                u0 += 8;
                v0 += 8;
            } else if (YUV_NV12 == yuvType) {
                uint8x8x2_t vec_uv_u8 = vld2_u8(uv);
                vec_u_u8 = vec_uv_u8.val[0];
                vec_v_u8 = vec_uv_u8.val[1];
                uv += 16;
            } else if (YUV_NV21 == yuvType) {
                uint8x8x2_t vec_vu_u8 = vld2_u8(vu);
                vec_v_u8 = vec_vu_u8.val[0];
                vec_u_u8 = vec_vu_u8.val[1];
                vu += 16;
            }

            //u - 128, v - 128
            int16x8_t vec_u_s16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vec_u_u8)), _v128_s16);
            int16x8_t vec_v_s16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vec_v_u8)), _v128_s16);

            //(y-16)>0?y-16:0
            uint8x16_t vec_y0_u8 = vld1q_u8(y0);
            uint8x16_t vec_y1_u8 = vld1q_u8(y1);
            uint8x8_t vec_y0_l_u8 = vget_low_u8(vec_y0_u8);
            uint8x8_t vec_y0_h_u8 = vget_high_u8(vec_y0_u8);
            uint8x8_t vec_y1_l_u8 = vget_low_u8(vec_y1_u8);
            uint8x8_t vec_y1_h_u8 = vget_high_u8(vec_y1_u8);
            int16x8_t vec_y0_l_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y0_l_u8));
            int16x8_t vec_y0_h_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y0_h_u8));
            int16x8_t vec_y1_l_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y1_l_u8));
            int16x8_t vec_y1_h_s16 = vreinterpretq_s16_u16(vmovl_u8(vec_y1_h_u8));
            vec_y0_l_s16 = vmaxq_s16(vsubq_s16(vec_y0_l_s16, _v16_s16), _v0_s16);
            vec_y0_h_s16 = vmaxq_s16(vsubq_s16(vec_y0_h_s16, _v16_s16), _v0_s16);
            vec_y1_l_s16 = vmaxq_s16(vsubq_s16(vec_y1_l_s16, _v16_s16), _v0_s16);
            vec_y1_h_s16 = vmaxq_s16(vsubq_s16(vec_y1_h_s16, _v16_s16), _v0_s16);

            //y * ITUR_BT_601_CY_6
            vec_y0_l_s16 = vmulq_s16(vec_y0_l_s16, _vCY_s16);
            vec_y0_h_s16 = vmulq_s16(vec_y0_h_s16, _vCY_s16);
            vec_y1_l_s16 = vmulq_s16(vec_y1_l_s16, _vCY_s16);
            vec_y1_h_s16 = vmulq_s16(vec_y1_h_s16, _vCY_s16);

            //u and v
            int16x8_t vec_ruv_s16 = vmlaq_s16(_vShift_s16, vec_v_s16, _vCVR_s16);
            int16x8_t vec_buv_s16 = vmlaq_s16(_vShift_s16, vec_u_s16, _vCUB_s16);
            int16x8_t vec_guv_s16 = vmlaq_s16(_vShift_s16, vec_v_s16, _vCVG_s16);
            vec_guv_s16 = vmlaq_s16(vec_guv_s16, vec_u_s16, _vCUG_s16);
            int16x8x2_t vec_ruv_s16_2 = vzipq_s16(vec_ruv_s16, vec_ruv_s16);
            int16x8x2_t vec_guv_s16_2 = vzipq_s16(vec_guv_s16, vec_guv_s16);
            int16x8x2_t vec_buv_s16_2 = vzipq_s16(vec_buv_s16, vec_buv_s16);

            //b
            uint8x8_t vec_b00_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y0_l_s16, vec_buv_s16_2.val[0]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_b01_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y0_h_s16, vec_buv_s16_2.val[1]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_b10_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y1_l_s16, vec_buv_s16_2.val[0]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_b11_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y1_h_s16, vec_buv_s16_2.val[1]), ITUR_BT_601_SHIFT_6));

            //g
            uint8x8_t vec_g00_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y0_l_s16, vec_guv_s16_2.val[0]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_g01_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y0_h_s16, vec_guv_s16_2.val[1]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_g10_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y1_l_s16, vec_guv_s16_2.val[0]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_g11_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y1_h_s16, vec_guv_s16_2.val[1]), ITUR_BT_601_SHIFT_6));

            //r
            uint8x8_t vec_r00_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y0_l_s16, vec_ruv_s16_2.val[0]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_r01_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y0_h_s16, vec_ruv_s16_2.val[1]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_r10_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y1_l_s16, vec_ruv_s16_2.val[0]), ITUR_BT_601_SHIFT_6));
            uint8x8_t vec_r11_u8 = vqmovun_s16(vshrq_n_s16(vqaddq_s16(vec_y1_h_s16, vec_ruv_s16_2.val[1]), ITUR_BT_601_SHIFT_6));

            //bgr
            if ((0 == b_idx) && (3 == dst_c)) //bgr
            {
                uint8x8x3_t vec_bgr_u8_0;
                uint8x8x3_t vec_bgr_u8_1;
                vec_bgr_u8_0.val[0] = vec_b00_u8;
                vec_bgr_u8_0.val[1] = vec_g00_u8;
                vec_bgr_u8_0.val[2] = vec_r00_u8;
                vst3_u8(rgb0, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_b01_u8;
                vec_bgr_u8_1.val[1] = vec_g01_u8;
                vec_bgr_u8_1.val[2] = vec_r01_u8;
                vst3_u8(rgb0 + 24, vec_bgr_u8_1);

                vec_bgr_u8_0.val[0] = vec_b10_u8;
                vec_bgr_u8_0.val[1] = vec_g10_u8;
                vec_bgr_u8_0.val[2] = vec_r10_u8;
                vst3_u8(rgb1, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_b11_u8;
                vec_bgr_u8_1.val[1] = vec_g11_u8;
                vec_bgr_u8_1.val[2] = vec_r11_u8;
                vst3_u8(rgb1 + 24, vec_bgr_u8_1);

                rgb0 += 16 * 3;
                rgb1 += 16 * 3;
            } else if ((0 == b_idx) && (4 == dst_c)) //bgra
            {
                uint8x8x4_t vec_bgr_u8_0;
                uint8x8x4_t vec_bgr_u8_1;
                vec_bgr_u8_0.val[0] = vec_b00_u8;
                vec_bgr_u8_0.val[1] = vec_g00_u8;
                vec_bgr_u8_0.val[2] = vec_r00_u8;
                vec_bgr_u8_0.val[3] = _v255_u8;
                vst4_u8(rgb0, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_b01_u8;
                vec_bgr_u8_1.val[1] = vec_g01_u8;
                vec_bgr_u8_1.val[2] = vec_r01_u8;
                vec_bgr_u8_1.val[3] = _v255_u8;
                vst4_u8(rgb0 + 32, vec_bgr_u8_1);

                vec_bgr_u8_0.val[0] = vec_b10_u8;
                vec_bgr_u8_0.val[1] = vec_g10_u8;
                vec_bgr_u8_0.val[2] = vec_r10_u8;
                vec_bgr_u8_0.val[3] = _v255_u8;
                vst4_u8(rgb1, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_b11_u8;
                vec_bgr_u8_1.val[1] = vec_g11_u8;
                vec_bgr_u8_1.val[2] = vec_r11_u8;
                vec_bgr_u8_1.val[3] = _v255_u8;
                vst4_u8(rgb1 + 32, vec_bgr_u8_1);

                rgb0 += 16 * 4;
                rgb1 += 16 * 4;
            } else if ((2 == b_idx) && (3 == dst_c)) //rgb or rgba
            {
                uint8x8x3_t vec_bgr_u8_0;
                uint8x8x3_t vec_bgr_u8_1;
                vec_bgr_u8_0.val[0] = vec_r00_u8;
                vec_bgr_u8_0.val[1] = vec_g00_u8;
                vec_bgr_u8_0.val[2] = vec_b00_u8;
                vst3_u8(rgb0, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_r01_u8;
                vec_bgr_u8_1.val[1] = vec_g01_u8;
                vec_bgr_u8_1.val[2] = vec_b01_u8;
                vst3_u8(rgb0 + 24, vec_bgr_u8_1);

                vec_bgr_u8_0.val[0] = vec_r10_u8;
                vec_bgr_u8_0.val[1] = vec_g10_u8;
                vec_bgr_u8_0.val[2] = vec_b10_u8;
                vst3_u8(rgb1, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_r11_u8;
                vec_bgr_u8_1.val[1] = vec_g11_u8;
                vec_bgr_u8_1.val[2] = vec_b11_u8;
                vst3_u8(rgb1 + 24, vec_bgr_u8_1);

                rgb0 += 16 * 3;
                rgb1 += 16 * 3;
            } else if ((2 == b_idx) && (4 == dst_c)) //rgb or rgba
            {
                uint8x8x4_t vec_bgr_u8_0;
                uint8x8x4_t vec_bgr_u8_1;
                vec_bgr_u8_0.val[0] = vec_r00_u8;
                vec_bgr_u8_0.val[1] = vec_g00_u8;
                vec_bgr_u8_0.val[2] = vec_b00_u8;
                vec_bgr_u8_0.val[3] = _v255_u8;
                vst4_u8(rgb0, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_r01_u8;
                vec_bgr_u8_1.val[1] = vec_g01_u8;
                vec_bgr_u8_1.val[2] = vec_b01_u8;
                vec_bgr_u8_1.val[3] = _v255_u8;
                vst4_u8(rgb0 + 32, vec_bgr_u8_1);

                vec_bgr_u8_0.val[0] = vec_r10_u8;
                vec_bgr_u8_0.val[1] = vec_g10_u8;
                vec_bgr_u8_0.val[2] = vec_b10_u8;
                vec_bgr_u8_0.val[3] = _v255_u8;
                vst4_u8(rgb1, vec_bgr_u8_0);

                vec_bgr_u8_1.val[0] = vec_r11_u8;
                vec_bgr_u8_1.val[1] = vec_g11_u8;
                vec_bgr_u8_1.val[2] = vec_b11_u8;
                vec_bgr_u8_1.val[3] = _v255_u8;
                vst4_u8(rgb1 + 32, vec_bgr_u8_1);

                rgb0 += 16 * 4;
                rgb1 += 16 * 4;
            }
            y0 += 16;
            y1 += 16;
        }
        for (; remain > 0; remain -= 2) {
            int32_t u, v;
            if ((YUV_I420 == yuvType) || (YUV_YV12 == yuvType)) {
                u = int32_t(u0[0]) - 128;
                v = int32_t(v0[0]) - 128;
                u0 += 1;
                v0 += 1;
            } else if (YUV_NV12 == yuvType) {
                u = int32_t(uv[0]) - 128;
                v = int32_t(uv[1]) - 128;
                uv += 2;
            } else if (YUV_NV21 == yuvType) {
                v = int32_t(vu[0]) - 128;
                u = int32_t(vu[1]) - 128;
                vu += 2;
            }

            int32_t ruv = (1 << (ITUR_BT_601_SHIFT_6 - 1)) + ITUR_BT_601_CVR_6 * v;
            int32_t guv = (1 << (ITUR_BT_601_SHIFT_6 - 1)) + ITUR_BT_601_CVG_6 * v + ITUR_BT_601_CUG_6 * u;
            int32_t buv = (1 << (ITUR_BT_601_SHIFT_6 - 1)) + ITUR_BT_601_CUB_6 * u;

            int32_t y00 = MAX(0, int32_t(y0[0]) - 16) * ITUR_BT_601_CY_6;

            int32_t r00 = sat_cast((y00 + ruv) >> ITUR_BT_601_SHIFT_6);
            int32_t g00 = sat_cast((y00 + guv) >> ITUR_BT_601_SHIFT_6);
            int32_t b00 = sat_cast((y00 + buv) >> ITUR_BT_601_SHIFT_6);

            int32_t y01 = MAX(0, int32_t(y0[1]) - 16) * ITUR_BT_601_CY_6;
            int32_t r01 = sat_cast((y01 + ruv) >> ITUR_BT_601_SHIFT_6);
            int32_t g01 = sat_cast((y01 + guv) >> ITUR_BT_601_SHIFT_6);
            int32_t b01 = sat_cast((y01 + buv) >> ITUR_BT_601_SHIFT_6);

            int32_t y10 = MAX(0, int32_t(y1[0]) - 16) * ITUR_BT_601_CY_6;
            int32_t r10 = sat_cast((y10 + ruv) >> ITUR_BT_601_SHIFT_6);
            int32_t g10 = sat_cast((y10 + guv) >> ITUR_BT_601_SHIFT_6);
            int32_t b10 = sat_cast((y10 + buv) >> ITUR_BT_601_SHIFT_6);

            int32_t y11 = MAX(0, int32_t(y1[1]) - 16) * ITUR_BT_601_CY_6;
            int32_t r11 = sat_cast((y11 + ruv) >> ITUR_BT_601_SHIFT_6);
            int32_t g11 = sat_cast((y11 + guv) >> ITUR_BT_601_SHIFT_6);
            int32_t b11 = sat_cast((y11 + buv) >> ITUR_BT_601_SHIFT_6);

            if ((0 == b_idx) && (3 == dst_c)) //bgr
            {
                rgb0[0] = b00;
                rgb0[1] = g00;
                rgb0[2] = r00;
                rgb0[3] = b01;
                rgb0[4] = g01;
                rgb0[5] = r01;
                rgb1[0] = b10;
                rgb1[1] = g10;
                rgb1[2] = r10;
                rgb1[3] = b11;
                rgb1[4] = g11;
                rgb1[5] = r11;
                rgb0 += 6;
                rgb1 += 6;
            } else if ((2 == b_idx) && (3 == dst_c)) //rgb
            {
                rgb0[0] = r00;
                rgb0[1] = g00;
                rgb0[2] = b00;
                rgb0[3] = r01;
                rgb0[4] = g01;
                rgb0[5] = b01;
                rgb1[0] = r10;
                rgb1[1] = g10;
                rgb1[2] = b10;
                rgb1[3] = r11;
                rgb1[4] = g11;
                rgb1[5] = b11;
                rgb0 += 6;
                rgb1 += 6;
            } else if ((0 == b_idx) && (4 == dst_c)) //bgra
            {
                rgb0[0] = b00;
                rgb0[1] = g00;
                rgb0[2] = r00;
                rgb0[3] = alpha;
                rgb0[4] = b01;
                rgb0[5] = g01;
                rgb0[6] = r01;
                rgb0[7] = alpha;
                rgb1[0] = b10;
                rgb1[1] = g10;
                rgb1[2] = r10;
                rgb1[3] = alpha;
                rgb1[4] = b11;
                rgb1[5] = g11;
                rgb1[6] = r11;
                rgb1[7] = alpha;
                rgb0 += 8;
                rgb1 += 8;
            } else if ((2 == b_idx) && (4 == dst_c)) //rgba
            {
                rgb0[0] = r00;
                rgb0[1] = g00;
                rgb0[2] = b00;
                rgb0[3] = alpha;
                rgb0[4] = r01;
                rgb0[5] = g01;
                rgb0[6] = b01;
                rgb0[7] = alpha;
                rgb1[0] = r10;
                rgb1[1] = g10;
                rgb1[2] = b10;
                rgb1[3] = alpha;
                rgb1[4] = r11;
                rgb1[5] = g11;
                rgb1[6] = b11;
                rgb1[7] = alpha;
                rgb0 += 8;
                rgb1 += 8;
            }

            y0 += 2;
            y1 += 2;
        }

        if ((YUV_I420 == yuvType) || (YUV_YV12 == yuvType)) {
            uptr += uStride;
            vptr += vStride;
        } else if (YUV_NV12 == yuvType) {
            uptr += uStride;
        } else if (YUV_NV21 == yuvType) {
            vptr += vStride;
        }
        yptr += 2 * yStride;
        rgb += 2 * rgbStride;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//to bgr
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_I420, 3, 0>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_YV12, 3, 0>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV12, 3, 0>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV21, 3, 0>(
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

//to bgra
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_I420, 4, 0>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_YV12, 4, 0>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV12, 4, 0>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV21, 4, 0>(
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

//to rgb
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_I420, 3, 2>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_YV12, 3, 2>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV12, 3, 2>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV21, 3, 2>(
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

//to rgba
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_I420, 4, 2>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_YV12, 4, 2>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV12, 4, 2>(
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
template void yuv420_to_bgr_uchar_video_range<YUV_TYPE::YUV_NV21, 4, 2>(
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    uint8_t* v_ptr)
{
    uint8_t* yptr = y_ptr;
    uint8_t* uptr = u_ptr;
    uint8_t* vptr = v_ptr;

    int16x8_t _vCRY_s16 = vdupq_n_s16(ITUR_BT_601_CRY_7);
    int16x8_t _vCGY_s16 = vdupq_n_s16(ITUR_BT_601_CGY_7);
    int16x8_t _vCBY_s16 = vdupq_n_s16(ITUR_BT_601_CBY_7);
    int16x8_t _vCRU_s16 = vdupq_n_s16(ITUR_BT_601_CRU_7);
    int16x8_t _vCGU_s16 = vdupq_n_s16(ITUR_BT_601_CGU_7);
    int16x8_t _vCBU_s16 = vdupq_n_s16(ITUR_BT_601_CBU_7);
    int16x8_t _vCGV_s16 = vdupq_n_s16(ITUR_BT_601_CGV_7);
    int16x8_t _vCBV_s16 = vdupq_n_s16(ITUR_BT_601_CBV_7);

    const int32_t shifted16 = (16 << ITUR_BT_601_SHIFT_7);
    const int32_t halfShift = (1 << (ITUR_BT_601_SHIFT_7 - 1));
    const int32_t shifted128 = (128 << ITUR_BT_601_SHIFT_7);
    const int32_t tail16 = halfShift + shifted16;
    const int32_t tail128 = halfShift + shifted128;

    int16x8_t _vtail16_s16 = vdupq_n_s16(tail16); //halfShift + shifted16;
    int16x8_t _vtail128_s16 = vdupq_n_s16(tail128); //halfShift + shifted128;

    for (int32_t i = 0; i < h; i += 2) {
        uint8_t* y0 = yptr;
        uint8_t* y1 = yptr + yStride;
        const uint8_t* rgb0 = rgb;
        const uint8_t* rgb1 = rgb + rgbStride;
        uint8_t* u0 = uptr; //for yu12 or yv12
        uint8_t* v0 = vptr;
        uint8_t* uv = uptr; //or nv12
        // uint8_t* vu = vptr; //or nv21
        int32_t remain = w;

        for (; remain > 16; remain -= 16) {
            uint8x8_t b00_u8;
            uint8x8_t b01_u8;
            uint8x8_t b10_u8;
            uint8x8_t b11_u8;
            uint8x8_t g00_u8;
            uint8x8_t g01_u8;
            uint8x8_t g10_u8;
            uint8x8_t g11_u8;
            uint8x8_t r00_u8;
            uint8x8_t r01_u8;
            uint8x8_t r10_u8;
            uint8x8_t r11_u8;

            //bgr
            if ((0 == b_idx) && (3 == src_c)) //bgr
            {
                uint8x8x3_t vec_bgr00_u8 = vld3_u8(rgb0);
                uint8x8x3_t vec_bgr01_u8 = vld3_u8(rgb0 + 8 * 3);
                uint8x8x3_t vec_bgr10_u8 = vld3_u8(rgb1);
                uint8x8x3_t vec_bgr11_u8 = vld3_u8(rgb1 + 8 * 3);
                b00_u8 = vec_bgr00_u8.val[0];
                b01_u8 = vec_bgr01_u8.val[0];
                b10_u8 = vec_bgr10_u8.val[0];
                b11_u8 = vec_bgr11_u8.val[0];
                g00_u8 = vec_bgr00_u8.val[1];
                g01_u8 = vec_bgr01_u8.val[1];
                g10_u8 = vec_bgr10_u8.val[1];
                g11_u8 = vec_bgr11_u8.val[1];
                r00_u8 = vec_bgr00_u8.val[2];
                r01_u8 = vec_bgr01_u8.val[2];
                r10_u8 = vec_bgr10_u8.val[2];
                r11_u8 = vec_bgr11_u8.val[2];

                rgb0 += 16 * 3;
                rgb1 += 16 * 3;
            } else if ((0 == b_idx) && (4 == src_c)) //bgra
            {
                uint8x8x4_t vec_bgr00_u8 = vld4_u8(rgb0);
                uint8x8x4_t vec_bgr01_u8 = vld4_u8(rgb0 + 8 * 4);
                uint8x8x4_t vec_bgr10_u8 = vld4_u8(rgb1);
                uint8x8x4_t vec_bgr11_u8 = vld4_u8(rgb1 + 8 * 4);
                b00_u8 = vec_bgr00_u8.val[0];
                b01_u8 = vec_bgr01_u8.val[0];
                b10_u8 = vec_bgr10_u8.val[0];
                b11_u8 = vec_bgr11_u8.val[0];
                g00_u8 = vec_bgr00_u8.val[1];
                g01_u8 = vec_bgr01_u8.val[1];
                g10_u8 = vec_bgr10_u8.val[1];
                g11_u8 = vec_bgr11_u8.val[1];
                r00_u8 = vec_bgr00_u8.val[2];
                r01_u8 = vec_bgr01_u8.val[2];
                r10_u8 = vec_bgr10_u8.val[2];
                r11_u8 = vec_bgr11_u8.val[2];

                rgb0 += 16 * 4;
                rgb1 += 16 * 4;
            } else if ((2 == b_idx) && (3 == src_c)) //rgb
            {
                uint8x8x3_t vec_bgr00_u8 = vld3_u8(rgb0);
                uint8x8x3_t vec_bgr01_u8 = vld3_u8(rgb0 + 8 * 3);
                uint8x8x3_t vec_bgr10_u8 = vld3_u8(rgb1);
                uint8x8x3_t vec_bgr11_u8 = vld3_u8(rgb1 + 8 * 3);
                b00_u8 = vec_bgr00_u8.val[2];
                b01_u8 = vec_bgr01_u8.val[2];
                b10_u8 = vec_bgr10_u8.val[2];
                b11_u8 = vec_bgr11_u8.val[2];
                g00_u8 = vec_bgr00_u8.val[1];
                g01_u8 = vec_bgr01_u8.val[1];
                g10_u8 = vec_bgr10_u8.val[1];
                g11_u8 = vec_bgr11_u8.val[1];
                r00_u8 = vec_bgr00_u8.val[0];
                r01_u8 = vec_bgr01_u8.val[0];
                r10_u8 = vec_bgr10_u8.val[0];
                r11_u8 = vec_bgr11_u8.val[0];

                rgb0 += 16 * 3;
                rgb1 += 16 * 3;
            } else if ((2 == b_idx) && (4 == src_c)) //rgba
            {
                uint8x8x4_t vec_bgr00_u8 = vld4_u8(rgb0);
                uint8x8x4_t vec_bgr01_u8 = vld4_u8(rgb0 + 8 * 4);
                uint8x8x4_t vec_bgr10_u8 = vld4_u8(rgb1);
                uint8x8x4_t vec_bgr11_u8 = vld4_u8(rgb1 + 8 * 4);
                b00_u8 = vec_bgr00_u8.val[2];
                b01_u8 = vec_bgr01_u8.val[2];
                b10_u8 = vec_bgr10_u8.val[2];
                b11_u8 = vec_bgr11_u8.val[2];
                g00_u8 = vec_bgr00_u8.val[1];
                g01_u8 = vec_bgr01_u8.val[1];
                g10_u8 = vec_bgr10_u8.val[1];
                g11_u8 = vec_bgr11_u8.val[1];
                r00_u8 = vec_bgr00_u8.val[0];
                r01_u8 = vec_bgr01_u8.val[0];
                r10_u8 = vec_bgr10_u8.val[0];
                r11_u8 = vec_bgr11_u8.val[0];

                rgb0 += 16 * 4;
                rgb1 += 16 * 4;
            }

            int16x8_t vec_b00, vec_g00, vec_r00, vec_b01, vec_g01, vec_r01;
            int16x8_t vec_b10, vec_g10, vec_r10, vec_b11, vec_g11, vec_r11;
            vec_b00 = vmulq_s16(_vCBY_s16, vreinterpretq_s16_u16(vmovl_u8(b00_u8)));
            vec_b01 = vmulq_s16(_vCBY_s16, vreinterpretq_s16_u16(vmovl_u8(b01_u8)));
            vec_b10 = vmulq_s16(_vCBY_s16, vreinterpretq_s16_u16(vmovl_u8(b10_u8)));
            vec_b11 = vmulq_s16(_vCBY_s16, vreinterpretq_s16_u16(vmovl_u8(b11_u8)));

            vec_g00 = vmlaq_s16(vec_b00, _vCGY_s16, vreinterpretq_s16_u16(vmovl_u8(g00_u8)));
            vec_g01 = vmlaq_s16(vec_b01, _vCGY_s16, vreinterpretq_s16_u16(vmovl_u8(g01_u8)));
            vec_g10 = vmlaq_s16(vec_b10, _vCGY_s16, vreinterpretq_s16_u16(vmovl_u8(g10_u8)));
            vec_g11 = vmlaq_s16(vec_b11, _vCGY_s16, vreinterpretq_s16_u16(vmovl_u8(g11_u8)));

            vec_r00 = vmlaq_s16(_vtail16_s16, _vCRY_s16, vreinterpretq_s16_u16(vmovl_u8(r00_u8)));
            vec_r01 = vmlaq_s16(_vtail16_s16, _vCRY_s16, vreinterpretq_s16_u16(vmovl_u8(r01_u8)));
            vec_r10 = vmlaq_s16(_vtail16_s16, _vCRY_s16, vreinterpretq_s16_u16(vmovl_u8(r10_u8)));
            vec_r11 = vmlaq_s16(_vtail16_s16, _vCRY_s16, vreinterpretq_s16_u16(vmovl_u8(r11_u8)));

            uint8x8_t y00 = vqmovun_s16(vshrq_n_s16(vaddq_s16(vec_g00, vec_r00), ITUR_BT_601_SHIFT_7));
            uint8x8_t y01 = vqmovun_s16(vshrq_n_s16(vaddq_s16(vec_g01, vec_r01), ITUR_BT_601_SHIFT_7));
            uint8x8_t y10 = vqmovun_s16(vshrq_n_s16(vaddq_s16(vec_g10, vec_r10), ITUR_BT_601_SHIFT_7));
            uint8x8_t y11 = vqmovun_s16(vshrq_n_s16(vaddq_s16(vec_g11, vec_r11), ITUR_BT_601_SHIFT_7));

            vst1_u8(y0, y00);
            vst1_u8(y0 + 8, y01);
            vst1_u8(y1, y10);
            vst1_u8(y1 + 8, y11);
            y0 += 16;
            y1 += 16;

            uint8x8x2_t b00 = vuzp_u8(b00_u8, b01_u8);
            uint8x8x2_t g00 = vuzp_u8(g00_u8, g01_u8);
            uint8x8x2_t r00 = vuzp_u8(r00_u8, r01_u8);

            vec_b00 = vreinterpretq_s16_u16(vmovl_u8(b00.val[0]));
            vec_g00 = vreinterpretq_s16_u16(vmovl_u8(g00.val[0]));
            vec_r00 = vreinterpretq_s16_u16(vmovl_u8(r00.val[0]));

            vec_b01 = vmulq_s16(vec_b00, _vCBU_s16);
            vec_g01 = vmulq_s16(vec_g00, _vCGU_s16);
            vec_r01 = vmlaq_s16(_vtail128_s16, vec_r00, _vCRU_s16);

            vec_b10 = vmulq_s16(vec_b00, _vCBV_s16);
            vec_g10 = vmulq_s16(vec_g00, _vCGV_s16);
            vec_r10 = vmlaq_s16(_vtail128_s16, vec_r00, _vCBU_s16);

            uint8x8_t vec_u = vqmovun_s16(vshrq_n_s16(vaddq_s16(vaddq_s16(vec_b01, vec_g01), vec_r01), ITUR_BT_601_SHIFT_7));
            uint8x8_t vec_v = vqmovun_s16(vshrq_n_s16(vaddq_s16(vaddq_s16(vec_b10, vec_g10), vec_r10), ITUR_BT_601_SHIFT_7));

            if (yuvType == YUV_NV12) {
                uint8x8x2_t vec_uv_u8;
                vec_uv_u8.val[0] = vec_u;
                vec_uv_u8.val[1] = vec_v;
                vst2_u8(uv, vec_uv_u8);
                uv += 16;
            } else if (yuvType == YUV_NV21) {
                uint8x8x2_t vec_uv_u8;
                vec_uv_u8.val[0] = vec_v;
                vec_uv_u8.val[1] = vec_u;
                vst2_u8(uv, vec_uv_u8);
                uv += 16;
            } else if ((yuvType == YUV_I420) || (yuvType == YUV_YV12)) {
                vst1_u8(u0, vec_u);
                vst1_u8(v0, vec_v);
                u0 += 8;
                v0 += 8;
            }
        }

        for (; remain > 0; remain -= 2) {
            //bgr or rgb or bgra or rgba
            int32_t r00 = rgb0[2 - b_idx];
            int32_t g00 = rgb0[1];
            int32_t b00 = rgb0[b_idx];
            int32_t r01 = rgb0[2 - b_idx + src_c];
            int32_t g01 = rgb0[1 + src_c];
            int32_t b01 = rgb0[b_idx + src_c];
            int32_t r10 = rgb1[2 - b_idx];
            int32_t g10 = rgb1[1];
            int32_t b10 = rgb1[b_idx];
            int32_t r11 = rgb1[2 - b_idx + src_c];
            int32_t g11 = rgb1[1 + src_c];
            int32_t b11 = rgb1[b_idx + src_c];
            rgb0 += src_c * 2;
            rgb1 += src_c * 2;

            int32_t y00 = ITUR_BT_601_CRY_7 * r00 + ITUR_BT_601_CGY_7 * g00 + ITUR_BT_601_CBY_7 * b00 + halfShift + shifted16;
            int32_t y01 = ITUR_BT_601_CRY_7 * r01 + ITUR_BT_601_CGY_7 * g01 + ITUR_BT_601_CBY_7 * b01 + halfShift + shifted16;
            int32_t y10 = ITUR_BT_601_CRY_7 * r10 + ITUR_BT_601_CGY_7 * g10 + ITUR_BT_601_CBY_7 * b10 + halfShift + shifted16;
            int32_t y11 = ITUR_BT_601_CRY_7 * r11 + ITUR_BT_601_CGY_7 * g11 + ITUR_BT_601_CBY_7 * b11 + halfShift + shifted16;

            y0[0] = sat_cast(y00 >> ITUR_BT_601_SHIFT_7);
            y0[1] = sat_cast(y01 >> ITUR_BT_601_SHIFT_7);
            y1[0] = sat_cast(y10 >> ITUR_BT_601_SHIFT_7);
            y1[1] = sat_cast(y11 >> ITUR_BT_601_SHIFT_7);
            y0 += 2;
            y1 += 2;

            int32_t u00 = ITUR_BT_601_CRU_7 * r00 + ITUR_BT_601_CGU_7 * g00 + ITUR_BT_601_CBU_7 * b00 + halfShift + shifted128;
            int32_t v00 = ITUR_BT_601_CBU_7 * r00 + ITUR_BT_601_CGV_7 * g00 + ITUR_BT_601_CBV_7 * b00 + halfShift + shifted128;

            if (yuvType == YUV_NV12) {
                uv[0] = sat_cast(u00 >> ITUR_BT_601_SHIFT_7);
                uv[1] = sat_cast(v00 >> ITUR_BT_601_SHIFT_7);
                uv += 2;
            } else if (yuvType == YUV_NV21) {
                uv[0] = sat_cast(v00 >> ITUR_BT_601_SHIFT_7);
                uv[1] = sat_cast(u00 >> ITUR_BT_601_SHIFT_7);
                uv += 2;
            } else if ((yuvType == YUV_I420) || (yuvType == YUV_YV12)) {
                u0[0] = sat_cast(u00 >> ITUR_BT_601_SHIFT_7);
                v0[0] = sat_cast(v00 >> ITUR_BT_601_SHIFT_7);
                u0++;
                v0++;
            }
        }
        uptr += uStride;
        vptr += vStride;
        yptr += 2 * yStride;
        rgb += 2 * rgbStride;
    }
}

template void bgr_to_yuv420_uchar_video_range<0, 3, YUV_NV12>(
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

template void bgr_to_yuv420_uchar_video_range<0, 4, YUV_NV12>(
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

template void bgr_to_yuv420_uchar_video_range<2, 3, YUV_NV12>(
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
template void bgr_to_yuv420_uchar_video_range<2, 4, YUV_NV12>(
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

template void bgr_to_yuv420_uchar_video_range<0, 3, YUV_NV21>(
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

template void bgr_to_yuv420_uchar_video_range<0, 4, YUV_NV21>(
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

template void bgr_to_yuv420_uchar_video_range<2, 3, YUV_NV21>(
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
template void bgr_to_yuv420_uchar_video_range<2, 4, YUV_NV21>(
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

template void bgr_to_yuv420_uchar_video_range<0, 3, YUV_I420>(
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

template void bgr_to_yuv420_uchar_video_range<0, 4, YUV_I420>(
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

template void bgr_to_yuv420_uchar_video_range<2, 3, YUV_I420>(
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
template void bgr_to_yuv420_uchar_video_range<2, 4, YUV_I420>(
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

template void bgr_to_yuv420_uchar_video_range<0, 3, YUV_YV12>(
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

template void bgr_to_yuv420_uchar_video_range<0, 4, YUV_YV12>(
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

template void bgr_to_yuv420_uchar_video_range<2, 3, YUV_YV12>(
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
template void bgr_to_yuv420_uchar_video_range<2, 4, YUV_YV12>(
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