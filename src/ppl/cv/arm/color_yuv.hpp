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
#ifndef __ST_HPC_PPL_CV_AARCH64_COLOR_YUV_HPP__
#define __ST_HPC_PPL_CV_AARCH64_COLOR_YUV_HPP__

#include <arm_neon.h>
#include "ppl/cv/types.h"
#include "ppl/cv/arm/typetraits.hpp"
#include <algorithm>
namespace ppl {
namespace cv {
namespace arm {
//constants for conversion from/to RGB and YUV, YCrCb according to BT.601

///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////
//R = 1.164(Y - 16) + 1.596(V - 128)
//G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
//B = 1.164(Y - 16)                  + 2.018(U - 128)

//R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
//G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
//B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

///////////////////////////////////// RGB -> YUV420 /////////////////////////////////////
// Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
// U = -0.148 * R00 - 0.291 * G00 + 0.439 * B00 + 128
// V = 0.439 * R00 - 0.368 * G00 - 0.071 * B00 + 128

// Y = ((269484 * R + 528482 * G + 102760 * B + (1 << 19) + (16 << 20)) >> 20
// U = (-155188 * R00 - 305135 * G00 + 460324 * B00 + (1 << 19) + (128 << 20)) >> 20
// V = (460324 * R00 - 385875 * G00 - 74448 * B00 + (1 << 19) + (128 << 20)) >> 20

// Coefficients for YUV420 to RGB conversion
const int32_t ITUR_BT_601_CY    = 1220542; // 1.164 * (1 << 20)
const int32_t ITUR_BT_601_CUB   = 2116026; // 2.018 * (1 << 20)
const int32_t ITUR_BT_601_CUG   = -409993; // -0.391 * (1 << 20)
const int32_t ITUR_BT_601_CVG   = -852492; // -0.813 * (1 << 20)
const int32_t ITUR_BT_601_CVR   = 1673527; // 1.596 * (1 << 20)
const int32_t ITUR_BT_601_SHIFT = 20;

// Coefficients for RGB to YUV420 conversion
const int32_t ITUR_BT_601_CRY = 269484; // 0.257
const int32_t ITUR_BT_601_CGY = 528482; // 0.504
const int32_t ITUR_BT_601_CBY = 102760; // 0.098
const int32_t ITUR_BT_601_CRU = -155188; // -0.148
const int32_t ITUR_BT_601_CGU = -305135; // -0.291
const int32_t ITUR_BT_601_CBU = 460324; // 0.439
const int32_t ITUR_BT_601_CGV = -385875; // -0.368
const int32_t ITUR_BT_601_CBV = -74448; // -0.071

///////////////////////////////////////////////////////
// quantized by 6
const int32_t ITUR_BT_601_SHIFT_6 = 6;
const int32_t ITUR_BT_601_CY_6    = 74; // 1.164
const int32_t ITUR_BT_601_CUB_6   = 129; // 2.018
const int32_t ITUR_BT_601_CUG_6   = -25; // -0.391
const int32_t ITUR_BT_601_CVG_6   = -52; // -0.813
const int32_t ITUR_BT_601_CVR_6   = 102; // 1.596

// Coefficients for RGB to YUV420 conversion
const int32_t ITUR_BT_601_SHIFT_7 = 7;
const int32_t ITUR_BT_601_CRY_7   = 33; // 0.257
const int32_t ITUR_BT_601_CGY_7   = 65; // 0.504
const int32_t ITUR_BT_601_CBY_7   = 13; // 0.098
const int32_t ITUR_BT_601_CRU_7   = -19; // -0.148
const int32_t ITUR_BT_601_CGU_7   = -37; // -0.291
const int32_t ITUR_BT_601_CBU_7   = 56; // 0.439
const int32_t ITUR_BT_601_CGV_7   = -47; // -0.368
const int32_t ITUR_BT_601_CBV_7   = -9; // -0.071
///////////////////////////////////////////////////////

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)
#define SWAP(a, b)  \
    {               \
        auto c = a; \
        a      = b; \
        b      = c; \
    }

static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = data < 0 ? 0 : val;
    return val;
}

struct YUV4202RGBA_u8 {
    YUV4202RGBA_u8(int32_t _bIdx)
        : bIdx(_bIdx) {}

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
        for (int32_t i = 0; i < width / 2; i += 1, row1 += 8, row2 += 8) {
            int32_t u = int32_t(u1[i]) - 128;
            int32_t v = int32_t(v1[i]) - 128;

            int32_t ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
            int32_t guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
            int32_t buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

            int32_t y00    = std::max(0, int32_t(y1[2 * i]) - 16) * ITUR_BT_601_CY;
            row1[2 - bIdx] = sat_cast((y00 + ruv) >> ITUR_BT_601_SHIFT);
            row1[1]        = sat_cast((y00 + guv) >> ITUR_BT_601_SHIFT);
            row1[bIdx]     = sat_cast((y00 + buv) >> ITUR_BT_601_SHIFT);
            row1[3]        = uint8_t(0xff);

            int32_t y01    = std::max(0, int32_t(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
            row1[6 - bIdx] = sat_cast((y01 + ruv) >> ITUR_BT_601_SHIFT);
            row1[5]        = sat_cast((y01 + guv) >> ITUR_BT_601_SHIFT);
            row1[4 + bIdx] = sat_cast((y01 + buv) >> ITUR_BT_601_SHIFT);
            row1[7]        = uint8_t(0xff);

            int32_t y10    = std::max(0, int32_t(y2[2 * i]) - 16) * ITUR_BT_601_CY;
            row2[2 - bIdx] = sat_cast((y10 + ruv) >> ITUR_BT_601_SHIFT);
            row2[1]        = sat_cast((y10 + guv) >> ITUR_BT_601_SHIFT);
            row2[bIdx]     = sat_cast((y10 + buv) >> ITUR_BT_601_SHIFT);
            row2[3]        = uint8_t(0xff);

            int32_t y11    = std::max(0, int32_t(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
            row2[6 - bIdx] = sat_cast((y11 + ruv) >> ITUR_BT_601_SHIFT);
            row2[5]        = sat_cast((y11 + guv) >> ITUR_BT_601_SHIFT);
            row2[4 + bIdx] = sat_cast((y11 + buv) >> ITUR_BT_601_SHIFT);
            row2[7]        = uint8_t(0xff);
        }
    }

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

    void convert_from_yuv420_seperate_layout(
        int32_t height,
        int32_t width,
        const uint8_t* y1,
        const uint8_t* u1,
        const uint8_t* v1,
        uint8_t* dst,
        int32_t ystride,
        int32_t ustride,
        int32_t vstride,
        int32_t outWidthStride) const
    {
        for (int32_t j = 0; j < height; j += 2, y1 += ystride * 2, u1 += ustride, v1 += vstride) {
            uint8_t* row1 = dst + j * outWidthStride;
            uint8_t* row2 = dst + (j + 1) * outWidthStride;
            convert_per_2rows(width, y1, u1, v1, row1, row2, ystride);
        }
    }

    //nv12 or nv21
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
        uint8_t* u1       = (uint8_t*)malloc(width / 2);
        uint8_t* v1       = (uint8_t*)malloc(width / 2);
        for (int32_t j = 0; j < height; j += 2, y1 += yStride * 2, uv += yStride) {
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

    int32_t bIdx;
};

}
}
} // namespace ppl::cv::arm
#endif //__ST_HPC_PPL_CV_AARCH64_COLOR_YUV_HPP__