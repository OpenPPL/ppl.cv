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
#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/types.h"
#include "typetraits.hpp"
#include "color_yuv_simd.hpp"
#include <algorithm>
namespace ppl {
namespace cv {
namespace arm {

#define CY_coeff  1220542
#define CUB_coeff 2116026
#define CUG_coeff -409993
#define CVG_coeff -852492
#define CVR_coeff 1673527
#define SHIFT     20

// Coefficients for RGB to YUV420p conversion
#define CRY_coeff 269484
#define CGY_coeff 528482
#define CBY_coeff 102760
#define CRU_coeff -155188
#define CGU_coeff -305135
#define CBU_coeff 460324
#define CGV_coeff -385875
#define CBV_coeff -74448

#define COEFF_Y  (149)
#define COEFF_BU (129)
#define COEFF_RV (102)
#define COEFF_GU (25)
#define COEFF_GV (52)
#define COEFF_R  (-14248)
#define COEFF_G  (8663)
#define COEFF_B  (-17705)

inline void prefetch(const void* ptr, size_t offset = 32 * 10)
{
#if defined __GNUC__
    __builtin_prefetch(reinterpret_cast<const char*>(ptr) + offset);
#else
    (void)ptr;
    (void)offset;
#endif
}

inline const uint8_t sat_cast_u8(int32_t data)
{
    return data > 255 ? 255 : (data < 0 ? 0 : data);
}

#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

template <YUV_TYPE yuvType, int32_t dst_c, int32_t b_idx>
void nv_to_bgr_uchar_video_range(
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

    uint16x8_t v14216 = vdupq_n_u16(-COEFF_R);
    uint16x8_t v17672 = vdupq_n_u16(-COEFF_B);
    uint16x8_t v8696 = vdupq_n_u16(COEFF_G);
    uint8x8_t v102 = vdup_n_u8(COEFF_RV);
    uint8x8_t v25 = vdup_n_u8(COEFF_GU);
    uint8x8_t v129 = vdup_n_u8(COEFF_BU);
    uint8x8_t v52 = vdup_n_u8(COEFF_GV);
    uint16x8_t v_1 = vdupq_n_u16((uint16_t)-1);
    uint8x8_t v149 = vdup_n_u8(COEFF_Y);
    uint8x8_t v16 = vdup_n_u8(16);
    const uint8_t alpha = 255;
    int32_t remain = w >= 15 ? w - 15 : 0;

    for (int32_t i = 0; i < h; i += 2) {
        const uint8_t* y0 = yptr + i * yStride;
        const uint8_t* y1 = yptr + (i + 1) * yStride;
        uint8_t* dst1 = rgb + i * rgbStride;
        uint8_t* dst2 = rgb + (i + 1) * rgbStride;
        const uint8_t* uv = uptr; //or nv12
        const uint8_t* vu = vptr; //or nv21

        int32_t j = 0, dj = 0;

        for (; j < remain; j += 16, dj += 48) {
            prefetch(uv + j);
            prefetch(y0 + j);
            prefetch(y1 + j);
            uint8x8_t vec_u_u8;
            uint8x8_t vec_v_u8;
            if (YUV_NV12 == yuvType) {
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

            uint16x8_t gu = vmlsl_u8(v8696, vec_u_u8, v25);
            int16x8_t ruv = (int16x8_t)vmlsl_u8(v14216, vec_v_u8, v102);
            int16x8_t buv = (int16x8_t)vmlsl_u8(v17672, vec_u_u8, v129);
            int16x8_t guv = (int16x8_t)vmlsl_u8(gu, vec_v_u8, v52);
            uint8x16x3_t rgbl;
            {
                uint8x8x2_t yl = vld2_u8(y0);
                yl.val[0] = vmax_u8(yl.val[0], v16);
                yl.val[1] = vmax_u8(yl.val[1], v16);

                uint16x8_t yodd1 = vmlal_u8(v_1, yl.val[0], v149);
                uint16x8_t yevn1 = vmlal_u8(v_1, yl.val[1], v149);
                int16x8_t yodd1h = (int16x8_t)vshrq_n_u16(yodd1, 1);
                int16x8_t yevn1h = (int16x8_t)vshrq_n_u16(yevn1, 1);

                int16x8_t rodd1w = vhsubq_s16(yodd1h, ruv);
                int16x8_t gevn1w = vhaddq_s16(yevn1h, guv);
                int16x8_t bodd1w = vhsubq_s16(yodd1h, buv);
                int16x8_t revn1w = vhsubq_s16(yevn1h, ruv);
                int16x8_t godd1w = vhaddq_s16(yodd1h, guv);
                int16x8_t bevn1w = vhsubq_s16(yevn1h, buv);

                uint8x8_t rodd1n = vqshrun_n_s16(rodd1w, 5);
                uint8x8_t revn1n = vqshrun_n_s16(revn1w, 5);
                uint8x8_t godd1n = vqshrun_n_s16(godd1w, 5);
                uint8x8x2_t r1 = vzip_u8(rodd1n, revn1n);
                uint8x8_t gevn1n = vqshrun_n_s16(gevn1w, 5);
                uint8x8_t bodd1n = vqshrun_n_s16(bodd1w, 5);
                uint8x8x2_t g1 = vzip_u8(godd1n, gevn1n);
                uint8x8_t bevn1n = vqshrun_n_s16(bevn1w, 5);
                uint8x8x2_t b1 = vzip_u8(bodd1n, bevn1n);
                rgbl.val[2 - b_idx] = vcombine_u8(r1.val[0], r1.val[1]);
                rgbl.val[1] = vcombine_u8(g1.val[0], g1.val[1]);
                rgbl.val[0 + b_idx] = vcombine_u8(b1.val[0], b1.val[1]);
            }
            vst3q_u8(dst1, rgbl);
            {
                uint8x8x2_t yl = vld2_u8(y1);
                yl.val[0] = vmax_u8(yl.val[0], v16);
                yl.val[1] = vmax_u8(yl.val[1], v16);

                uint16x8_t yodd1 = vmlal_u8(v_1, yl.val[0], v149);
                uint16x8_t yevn1 = vmlal_u8(v_1, yl.val[1], v149);
                int16x8_t yodd1h = (int16x8_t)vshrq_n_u16(yodd1, 1);
                int16x8_t yevn1h = (int16x8_t)vshrq_n_u16(yevn1, 1);

                int16x8_t rodd1w = vhsubq_s16(yodd1h, ruv);
                int16x8_t gevn1w = vhaddq_s16(yevn1h, guv);
                int16x8_t bodd1w = vhsubq_s16(yodd1h, buv);
                int16x8_t revn1w = vhsubq_s16(yevn1h, ruv);
                int16x8_t godd1w = vhaddq_s16(yodd1h, guv);
                int16x8_t bevn1w = vhsubq_s16(yevn1h, buv);

                uint8x8_t rodd1n = vqshrun_n_s16(rodd1w, 5);
                uint8x8_t revn1n = vqshrun_n_s16(revn1w, 5);
                uint8x8_t godd1n = vqshrun_n_s16(godd1w, 5);
                uint8x8x2_t r1 = vzip_u8(rodd1n, revn1n);
                uint8x8_t gevn1n = vqshrun_n_s16(gevn1w, 5);
                uint8x8_t bodd1n = vqshrun_n_s16(bodd1w, 5);
                uint8x8x2_t g1 = vzip_u8(godd1n, gevn1n);
                uint8x8_t bevn1n = vqshrun_n_s16(bevn1w, 5);
                uint8x8x2_t b1 = vzip_u8(bodd1n, bevn1n);
                rgbl.val[2 - b_idx] = vcombine_u8(r1.val[0], r1.val[1]);
                rgbl.val[1] = vcombine_u8(g1.val[0], g1.val[1]);
                rgbl.val[0 + b_idx] = vcombine_u8(b1.val[0], b1.val[1]);
            }
            vst3q_u8(dst2, rgbl);
            y0 += 16;
            y1 += 16;
            dst1 += 48;
            dst2 += 48;
        }
        for (; j + 2 <= w; j += 2, dj += 6) {
            int32_t u, v;
            if (YUV_NV12 == yuvType) {
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
                dst1[0] = b00;
                dst1[1] = g00;
                dst1[2] = r00;
                dst1[3] = b01;
                dst1[4] = g01;
                dst1[5] = r01;
                dst2[0] = b10;
                dst2[1] = g10;
                dst2[2] = r10;
                dst2[3] = b11;
                dst2[4] = g11;
                dst2[5] = r11;
                dst1 += 6;
                dst2 += 6;
            } else if ((2 == b_idx) && (3 == dst_c)) //rgb
            {
                dst1[0] = r00;
                dst1[1] = g00;
                dst1[2] = b00;
                dst1[3] = r01;
                dst1[4] = g01;
                dst1[5] = b01;
                dst2[0] = r10;
                dst2[1] = g10;
                dst2[2] = b10;
                dst2[3] = r11;
                dst2[4] = g11;
                dst2[5] = b11;
                dst1 += 6;
                dst2 += 6;
            } else if ((0 == b_idx) && (4 == dst_c)) //bgra
            {
                dst1[0] = b00;
                dst1[1] = g00;
                dst1[2] = r00;
                dst1[3] = alpha;
                dst1[4] = b01;
                dst1[5] = g01;
                dst1[6] = r01;
                dst1[7] = alpha;
                dst2[0] = b10;
                dst2[1] = g10;
                dst2[2] = r10;
                dst2[3] = alpha;
                dst2[4] = b11;
                dst2[5] = g11;
                dst2[6] = r11;
                dst2[7] = alpha;
                dst1 += 8;
                dst2 += 8;
            } else if ((2 == b_idx) && (4 == dst_c)) //rgba
            {
                dst1[0] = r00;
                dst1[1] = g00;
                dst1[2] = b00;
                dst1[3] = alpha;
                dst1[4] = r01;
                dst1[5] = g01;
                dst1[6] = b01;
                dst1[7] = alpha;
                dst2[0] = r10;
                dst2[1] = g10;
                dst2[2] = b10;
                dst2[3] = alpha;
                dst2[4] = r11;
                dst2[5] = g11;
                dst2[6] = b11;
                dst2[7] = alpha;
                dst1 += 8;
                dst2 += 8;
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
    }
}

template <int32_t srccn, int32_t bIdx, bool isUV>
void rgb_2_nv(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    for (int32_t i = 0; i < height; i += 2) {
        const uint8_t* src0 = inData + i * inWidthStride;
        const uint8_t* src1 = inData + (i + 1) * inWidthStride;
        uint8_t* dst0 = outY + i * outYStride;
        uint8_t* dst1 = outY + (i + 1) * outYStride;
        uint8_t* dst2 = outUV + (i / 2) * outUVStride;
        for (int32_t j = 0; j < width / 2; ++j, src0 += 2 * srccn, src1 += 2 * srccn) {
            int32_t r00 = src0[2 - bIdx];
            int32_t g00 = src0[1];
            int32_t b00 = src0[bIdx];
            int32_t r01 = src0[2 - bIdx + srccn];
            int32_t g01 = src0[1 + srccn];
            int32_t b01 = src0[bIdx + srccn];
            int32_t r10 = src1[2 - bIdx];
            int32_t g10 = src1[1];
            int32_t b10 = src1[bIdx];
            int32_t r11 = src1[2 - bIdx + srccn];
            int32_t g11 = src1[1 + srccn];
            int32_t b11 = src1[bIdx + srccn];

            const int32_t shifted16 = (16 << SHIFT);
            const int32_t halfShift = (1 << (SHIFT - 1));

            int32_t y00 = CRY_coeff * r00 + CGY_coeff * g00 + CBY_coeff * b00 + halfShift + shifted16;
            int32_t y01 = CRY_coeff * r01 + CGY_coeff * g01 + CBY_coeff * b01 + halfShift + shifted16;
            int32_t y10 = CRY_coeff * r10 + CGY_coeff * g10 + CBY_coeff * b10 + halfShift + shifted16;
            int32_t y11 = CRY_coeff * r11 + CGY_coeff * g11 + CBY_coeff * b11 + halfShift + shifted16;

            dst0[2 * j + 0] = sat_cast_u8(y00 >> SHIFT);
            dst0[2 * j + 1] = sat_cast_u8(y01 >> SHIFT);
            dst1[2 * j + 0] = sat_cast_u8(y10 >> SHIFT);
            dst1[2 * j + 1] = sat_cast_u8(y11 >> SHIFT);

            const int32_t shifted128 = (128 << SHIFT);
            int32_t u00 = CRU_coeff * r00 + CGU_coeff * g00 + CBU_coeff * b00 + halfShift + shifted128;
            int32_t v00 = CBU_coeff * r00 + CGV_coeff * g00 + CBV_coeff * b00 + halfShift + shifted128;

            if (isUV) {
                dst2[2 * j] = sat_cast_u8(u00 >> SHIFT);
                dst2[2 * j + 1] = sat_cast_u8(v00 >> SHIFT);
            } else {
                dst2[2 * j] = sat_cast_u8(v00 >> SHIFT);
                dst2[2 * j + 1] = sat_cast_u8(u00 >> SHIFT);
            }
        }
    }
}

template <>
::ppl::common::RetCode BGR2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t uStride = inWidthStride;
    const uint8_t* u_ptr = inData + inWidthStride * height;
    int32_t vStride = 0;
    const uint8_t* v_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV12, 3, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    int32_t inYStride = inWidthStride;
    int32_t inUVStride = inWidthStride;
    const uint8_t* inY = inData;
    const uint8_t* inUV = inY + inWidthStride * height;
    YUV4202RGB_u8_neon s(0);
    s.convert_from_yuv420sp_layout(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData, true);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inUV,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride = inYStride;
    const uint8_t* y_ptr = inY;
    int32_t uStride = inUVStride;
    const uint8_t* u_ptr = inUV;
    int32_t vStride = 0;
    const uint8_t* v_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV12, 3, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    YUV4202RGB_u8_neon s(0);
    s.convert_from_yuv420sp_layout(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData, true);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t uStride = inWidthStride;
    const uint8_t* u_ptr = inData + inWidthStride * height;
    int32_t vStride = 0;
    const uint8_t* v_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV12, 4, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    int32_t yStride = inWidthStride;
    int32_t uvStride = inWidthStride;
    const uint8_t* y = inData;
    const uint8_t* uv = inData + inWidthStride * height;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(0);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, true);

#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inUV,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
// #ifdef USE_QUANTIZED
//     int32_t yStride = inYStride;
//     const uint8_t* y_ptr = inY;
//     int32_t uStride = inUVStride;
//     const uint8_t* u_ptr = inUV;
//     int32_t vStride = 0;
//     const uint8_t* v_ptr = nullptr;
//     int32_t rgbStride = outWidthStride;
//     uint8_t* rgb = outData;
//     nv_to_bgr_uchar_video_range<YUV_NV12, 4, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
// #else
    int32_t yStride = inYStride;
    int32_t uvStride = inUVStride;
    const uint8_t* y = inY;
    const uint8_t* uv = inUV;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(0);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, true);

// #endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t vStride = inWidthStride;
    const uint8_t* v_ptr = inData + inWidthStride * height;
    int32_t uStride = 0;
    const uint8_t* u_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV21, 3, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    int32_t inYStride = inWidthStride;
    int32_t inUVStride = inWidthStride;
    const uint8_t* inY = inData;
    const uint8_t* inVU = inY + inWidthStride * height;
    YUV4202RGB_u8_neon s(0);
    s.convert_from_yuv420sp_layout(height, width, inYStride, inY, inUVStride, inVU, outWidthStride, outData, false);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode NV212BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inVU,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inVU) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride = inYStride;
    const uint8_t* y_ptr = inY;
    int32_t vStride = inUVStride;
    const uint8_t* v_ptr = inVU;
    int32_t uStride = 0;
    const uint8_t* u_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV21, 3, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    YUV4202RGB_u8_neon s(0);
    s.convert_from_yuv420sp_layout(height, width, inYStride, inY, inUVStride, inVU, outWidthStride, outData, false);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t vStride = inWidthStride;
    const uint8_t* v_ptr = inData + inWidthStride * height;
    int32_t uStride = 0;
    const uint8_t* u_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV21, 4, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    int32_t yStride = inWidthStride;
    int32_t uvStride = inWidthStride;
    const uint8_t* y = inData;
    const uint8_t* uv = inData + inWidthStride * height;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(0);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, false);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inVU,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inVU) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
// #ifdef USE_QUANTIZED
//     int32_t yStride = inYStride;
//     const uint8_t* y_ptr = inY;
//     int32_t vStride = inUVStride;
//     const uint8_t* v_ptr = inVU;
//     int32_t uStride = 0;
//     const uint8_t* u_ptr = nullptr;
//     int32_t rgbStride = outWidthStride;
//     uint8_t* rgb = outData;
//     nv_to_bgr_uchar_video_range<YUV_NV21, 4, 0>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
// #else
    int32_t yStride = inYStride;
    int32_t uvStride = inUVStride;
    const uint8_t* y = inY;
    const uint8_t* uv = inVU;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(0);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, false);

// #endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode RGBA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t uStride = inWidthStride;
    const uint8_t* u_ptr = inData + inWidthStride * height;
    int32_t vStride = 0;
    const uint8_t* v_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV12, 3, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    int32_t inYStride = inWidthStride;
    int32_t inUVStride = inWidthStride;
    const uint8_t* inY = inData;
    const uint8_t* inUV = inY + inWidthStride * height;
    YUV4202RGB_u8_neon s(2);
    s.convert_from_yuv420sp_layout(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData, true);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode NV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t uStride = inWidthStride;
    const uint8_t* u_ptr = inData + inWidthStride * height;
    int32_t vStride = 0;
    const uint8_t* v_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV12, 4, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    int32_t yStride = inWidthStride;
    int32_t uvStride = inWidthStride;
    const uint8_t* y = inData;
    const uint8_t* uv = inData + inWidthStride * height;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(2);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, true);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inUV,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inYStride;
    const uint8_t* y_ptr = inY;
    int32_t uStride = inUVStride;
    const uint8_t* u_ptr = inUV;
    int32_t vStride = 0;
    const uint8_t* v_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV12, 3, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    YUV4202RGB_u8_neon s(2);
    s.convert_from_yuv420sp_layout(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData, true);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode NV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inUV,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
// #ifdef USE_QUANTIZED
//     int32_t yStride = inYStride;
//     const uint8_t* y_ptr = inY;
//     int32_t uStride = inUVStride;
//     const uint8_t* u_ptr = inUV;
//     int32_t vStride = 0;
//     const uint8_t* v_ptr = nullptr;
//     int32_t rgbStride = outWidthStride;
//     uint8_t* rgb = outData;
//     nv_to_bgr_uchar_video_range<YUV_NV12, 4, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
// #else
    int32_t yStride = inYStride;
    int32_t uvStride = inUVStride;
    const uint8_t* y = inY;
    const uint8_t* uv = inUV;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(2);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, true);
// #endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode RGBA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outYStride,
    uint8_t* outY,
    int32_t outUVStride,
    uint8_t* outUV)
{
    if (nullptr == inData || nullptr == outY || nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t vStride = inWidthStride;
    const uint8_t* v_ptr = inData + inWidthStride * height;
    int32_t uStride = 0;
    const uint8_t* u_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV21, 3, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    YUV4202RGB_u8_neon s(2);
    s.convert_from_yuv420sp_layout(height, width, inWidthStride, inData, inWidthStride, inData + inWidthStride * height, outWidthStride, outData, false);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode NV212RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t vStride = inWidthStride;
    const uint8_t* v_ptr = inData + inWidthStride * height;
    int32_t uStride = 0;
    const uint8_t* u_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV21, 4, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    int32_t yStride = inWidthStride;
    int32_t uvStride = inWidthStride;
    const uint8_t* y = inData;
    const uint8_t* uv = inData + inWidthStride * height;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(2);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, false);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inUV,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride = inYStride;
    const uint8_t* y_ptr = inY;
    int32_t vStride = inUVStride;
    const uint8_t* v_ptr = inUV;
    int32_t uStride = 0;
    const uint8_t* u_ptr = nullptr;
    int32_t rgbStride = outWidthStride;
    uint8_t* rgb = outData;
    nv_to_bgr_uchar_video_range<YUV_NV21, 3, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
#else
    YUV4202RGB_u8_neon s(2);
    s.convert_from_yuv420sp_layout(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData, false);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode NV212RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t* inY,
    int32_t inUVStride,
    const uint8_t* inUV,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inY || nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
// #ifdef USE_QUANTIZED
//     int32_t yStride = inYStride;
//     const uint8_t* y_ptr = inY;
//     int32_t vStride = inUVStride;
//     const uint8_t* v_ptr = inUV;
//     int32_t uStride = 0;
//     const uint8_t* u_ptr = nullptr;
//     int32_t rgbStride = outWidthStride;
//     uint8_t* rgb = outData;
//     nv_to_bgr_uchar_video_range<YUV_NV21, 4, 2>(height, width, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr, rgbStride, rgb);
// #else
    int32_t yStride = inYStride;
    int32_t uvStride = inUVStride;
    const uint8_t* y = inY;
    const uint8_t* uv = inUV;
    YUV4202RGBA_u8 s = YUV4202RGBA_u8(2);
    s.convert_from_yuv420sp_layout(height, width, yStride, y, uvStride, uv, outWidthStride, outData, false);

// #endif
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
