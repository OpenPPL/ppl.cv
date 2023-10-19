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

#include "ppl/cv/arm/integral.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include "intrinutils_neon.hpp"
#include <string.h>
#include <arm_neon.h>
#include <cstdio>

namespace ppl {
namespace cv {
namespace arm {

template <typename TSrc, typename TDst, int32_t channels>
void IntegralImage(int32_t height,
                   int32_t width,
                   int32_t inWidthStride,
                   const TSrc *in,
                   int32_t outWidthStride,
                   TDst *out)
{
    int32_t outWidth = width + 1;
    memset(out, 0, sizeof(TDst) * outWidth * channels);
    for (int32_t h = 0; h < height; h++) {
        TDst sum[channels] = {0};
        memset(out + (h + 1) * outWidthStride, 0, sizeof(TDst) * channels);
        for (int32_t w = 0; w < width; w++) {
            for (int32_t c = 0; c < channels; c++) {
                TSrc in_v = in[h * inWidthStride + w * channels + c];
                sum[c] += in_v;
                out[(h + 1) * outWidthStride + (w + 1) * channels + c] =
                    sum[c] + out[h * outWidthStride + (w + 1) * channels + c];
            }
        }
    }
}

template <>
void IntegralImage<uint8_t, uint32_t, 1>(int32_t height,
                                         int32_t width,
                                         int32_t inWidthStride,
                                         const uint8_t *in,
                                         int32_t outWidthStride,
                                         uint32_t *out)
{
    constexpr int channels = 1;

    int32_t outWidth = width + 1;
    memset(out, 0, sizeof(uint32_t) * outWidth * channels);

    for (int32_t h = 0; h < height; h++) {
        memset(out + (h + 1) * outWidthStride, 0, sizeof(uint32_t) * channels);

        int32_t w = 0;
        int32_t sum[channels] = {0};
        for (; w <= width - 16; w += 16) {
            prefetch(in + h * inWidthStride + w * channels);
            uint8x16_t vInData0 = vld1q_u8(in + h * inWidthStride + w * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 0) * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 8) * channels);
            uint32x4_t vInData1_0 = vld1q_u32(out + h * outWidthStride + (w + 1 + 0) * channels);
            uint32x4_t vInData1_1 = vld1q_u32(out + h * outWidthStride + (w + 1 + 4) * channels);
            uint32x4_t vInData1_2 = vld1q_u32(out + h * outWidthStride + (w + 1 + 8) * channels);
            uint32x4_t vInData1_3 = vld1q_u32(out + h * outWidthStride + (w + 1 + 12) * channels);

            uint16x8_t vUhInData0 = vmovl_u8(vget_low_u8(vInData0));
            uint16x8_t vUhInData1 = vmovl_high_u8(vInData0);

            uint16x8_t vUhScanRes0 = neon_scan_across_vector_u16x8(vUhInData0);
            uint16x8_t vUhScanRes1 = neon_scan_across_vector_u16x8(vUhInData1);

            uint32x4_t vScanRes0 = vaddw_u16(vdupq_n_u32(sum[0]), vget_low_u16(vUhScanRes0));
            uint32x4_t vScanRes1 = vaddw_u16(vdupq_n_u32(sum[0]), vget_high_u16(vUhScanRes0));
            uint32x4_t vScanRes2 = vaddw_u16(vdupq_laneq_u32(vScanRes1, 3), vget_low_u16(vUhScanRes1));
            uint32x4_t vScanRes3 = vaddw_u16(vdupq_laneq_u32(vScanRes1, 3), vget_high_u16(vUhScanRes1));

            sum[0] = vgetq_lane_u32(vScanRes3, 3);

            uint32x4_t vRes0 = vaddq_u32(vInData1_0, vScanRes0);
            uint32x4_t vRes1 = vaddq_u32(vInData1_1, vScanRes1);
            uint32x4_t vRes2 = vaddq_u32(vInData1_2, vScanRes2);
            uint32x4_t vRes3 = vaddq_u32(vInData1_3, vScanRes3);

            vst1q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 0) * channels), vRes0);
            vst1q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 4) * channels), vRes1);
            vst1q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 8) * channels), vRes2);
            vst1q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 12) * channels), vRes3);
        }

        for (; w < width; w++) {
            for (int32_t c = 0; c < channels; c++) {
                uint8_t in_v = in[h * inWidthStride + w * channels + c];
                sum[c] += in_v;
                out[(h + 1) * outWidthStride + (w + 1) * channels + c] =
                    sum[c] + out[h * outWidthStride + (w + 1) * channels + c];
            }
        }
    }
}

template <>
void IntegralImage<uint8_t, uint32_t, 3>(int32_t height,
                                         int32_t width,
                                         int32_t inWidthStride,
                                         const uint8_t *in,
                                         int32_t outWidthStride,
                                         uint32_t *out)
{
    constexpr int channels = 3;

    int32_t outWidth = width + 1;
    memset(out, 0, sizeof(uint32_t) * outWidth * channels);

    for (int32_t h = 0; h < height; h++) {
        memset(out + (h + 1) * outWidthStride, 0, sizeof(uint32_t) * channels);

        int32_t w = 0;
        int32_t sum[channels] = {0};
        for (; w <= width - 16; w += 16) {
            prefetch(in + h * inWidthStride + w * channels);
            uint8x16x3_t vInData0 = vld3q_u8(in + h * inWidthStride + w * channels);

            uint16x8x3_t vUhInData0, vUhInData1;
            vUhInData0.val[0] = vmovl_u8(vget_low_u8(vInData0.val[0]));
            vUhInData1.val[0] = vmovl_high_u8(vInData0.val[0]);
            vUhInData0.val[1] = vmovl_u8(vget_low_u8(vInData0.val[1]));
            vUhInData1.val[1] = vmovl_high_u8(vInData0.val[1]);
            vUhInData0.val[2] = vmovl_u8(vget_low_u8(vInData0.val[2]));
            vUhInData1.val[2] = vmovl_high_u8(vInData0.val[2]);

            uint16x8x3_t vUhScanRes0, vUhScanRes1;
            vUhScanRes0.val[0] = neon_scan_across_vector_u16x8(vUhInData0.val[0]);
            vUhScanRes1.val[0] = neon_scan_across_vector_u16x8(vUhInData1.val[0]);
            vUhScanRes0.val[1] = neon_scan_across_vector_u16x8(vUhInData0.val[1]);
            vUhScanRes1.val[1] = neon_scan_across_vector_u16x8(vUhInData1.val[1]);
            vUhScanRes0.val[2] = neon_scan_across_vector_u16x8(vUhInData0.val[2]);
            vUhScanRes1.val[2] = neon_scan_across_vector_u16x8(vUhInData1.val[2]);

            uint32x4x3_t vScanRes0, vScanRes1, vScanRes2, vScanRes3;
            vScanRes0.val[0] = vaddw_u16(vdupq_n_u32(sum[0]), vget_low_u16(vUhScanRes0.val[0]));
            vScanRes1.val[0] = vaddw_u16(vdupq_n_u32(sum[0]), vget_high_u16(vUhScanRes0.val[0]));
            vScanRes2.val[0] = vaddw_u16(vdupq_laneq_u32(vScanRes1.val[0], 3), vget_low_u16(vUhScanRes1.val[0]));
            vScanRes3.val[0] = vaddw_u16(vdupq_laneq_u32(vScanRes1.val[0], 3), vget_high_u16(vUhScanRes1.val[0]));
            vScanRes0.val[1] = vaddw_u16(vdupq_n_u32(sum[1]), vget_low_u16(vUhScanRes0.val[1]));
            vScanRes1.val[1] = vaddw_u16(vdupq_n_u32(sum[1]), vget_high_u16(vUhScanRes0.val[1]));
            vScanRes2.val[1] = vaddw_u16(vdupq_laneq_u32(vScanRes1.val[1], 3), vget_low_u16(vUhScanRes1.val[1]));
            vScanRes3.val[1] = vaddw_u16(vdupq_laneq_u32(vScanRes1.val[1], 3), vget_high_u16(vUhScanRes1.val[1]));
            vScanRes0.val[2] = vaddw_u16(vdupq_n_u32(sum[2]), vget_low_u16(vUhScanRes0.val[2]));
            vScanRes1.val[2] = vaddw_u16(vdupq_n_u32(sum[2]), vget_high_u16(vUhScanRes0.val[2]));
            vScanRes2.val[2] = vaddw_u16(vdupq_laneq_u32(vScanRes1.val[2], 3), vget_low_u16(vUhScanRes1.val[2]));
            vScanRes3.val[2] = vaddw_u16(vdupq_laneq_u32(vScanRes1.val[2], 3), vget_high_u16(vUhScanRes1.val[2]));

            sum[0] = vgetq_lane_u32(vScanRes3.val[0], 3);
            sum[1] = vgetq_lane_u32(vScanRes3.val[1], 3);
            sum[2] = vgetq_lane_u32(vScanRes3.val[2], 3);

            prefetch(out + h * outWidthStride + (w + 1 + 0) * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 4) * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 8) * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 12) * channels);
            uint32x4x3_t vInData1_0 = vld3q_u32(out + h * outWidthStride + (w + 1 + 0) * channels);
            uint32x4x3_t vInData1_1 = vld3q_u32(out + h * outWidthStride + (w + 1 + 4) * channels);
            uint32x4x3_t vInData1_2 = vld3q_u32(out + h * outWidthStride + (w + 1 + 8) * channels);
            uint32x4x3_t vInData1_3 = vld3q_u32(out + h * outWidthStride + (w + 1 + 12) * channels);

            uint32x4x3_t vRes0, vRes1, vRes2, vRes3;
            vRes0.val[0] = vaddq_u32(vInData1_0.val[0], vScanRes0.val[0]);
            vRes1.val[0] = vaddq_u32(vInData1_1.val[0], vScanRes1.val[0]);
            vRes2.val[0] = vaddq_u32(vInData1_2.val[0], vScanRes2.val[0]);
            vRes3.val[0] = vaddq_u32(vInData1_3.val[0], vScanRes3.val[0]);
            vRes0.val[1] = vaddq_u32(vInData1_0.val[1], vScanRes0.val[1]);
            vRes1.val[1] = vaddq_u32(vInData1_1.val[1], vScanRes1.val[1]);
            vRes2.val[1] = vaddq_u32(vInData1_2.val[1], vScanRes2.val[1]);
            vRes3.val[1] = vaddq_u32(vInData1_3.val[1], vScanRes3.val[1]);
            vRes0.val[2] = vaddq_u32(vInData1_0.val[2], vScanRes0.val[2]);
            vRes1.val[2] = vaddq_u32(vInData1_1.val[2], vScanRes1.val[2]);
            vRes2.val[2] = vaddq_u32(vInData1_2.val[2], vScanRes2.val[2]);
            vRes3.val[2] = vaddq_u32(vInData1_3.val[2], vScanRes3.val[2]);

            vst3q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 0) * channels), vRes0);
            vst3q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 4) * channels), vRes1);
            vst3q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 8) * channels), vRes2);
            vst3q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 12) * channels), vRes3);
        }

        for (; w < width; w++) {
            for (int32_t c = 0; c < channels; c++) {
                uint8_t in_v = in[h * inWidthStride + w * channels + c];
                sum[c] += in_v;
                out[(h + 1) * outWidthStride + (w + 1) * channels + c] =
                    sum[c] + out[h * outWidthStride + (w + 1) * channels + c];
            }
        }
    }
}

template <>
void IntegralImage<uint8_t, uint32_t, 4>(int32_t height,
                                         int32_t width,
                                         int32_t inWidthStride,
                                         const uint8_t *in,
                                         int32_t outWidthStride,
                                         uint32_t *out)
{
    constexpr int channels = 4;

    int32_t outWidth = width + 1;
    memset(out, 0, sizeof(uint32_t) * outWidth * channels);

    for (int32_t h = 0; h < height; h++) {
        memset(out + (h + 1) * outWidthStride, 0, sizeof(uint32_t) * channels);

        int32_t w = 0;
        int32_t sum[channels] = {0};
        for (; w <= width - 8; w += 8) {
            prefetch(in + h * inWidthStride + w * channels);
            uint8x8x4_t vInData0 = vld4_u8(in + h * inWidthStride + w * channels);

            uint16x8x4_t vUhInData0;
            vUhInData0.val[0] = vmovl_u8(vInData0.val[0]);
            vUhInData0.val[1] = vmovl_u8(vInData0.val[1]);
            vUhInData0.val[2] = vmovl_u8(vInData0.val[2]);
            vUhInData0.val[3] = vmovl_u8(vInData0.val[3]);

            uint16x8x4_t vUhScanRes0;
            vUhScanRes0.val[0] = neon_scan_across_vector_u16x8(vUhInData0.val[0]);
            vUhScanRes0.val[1] = neon_scan_across_vector_u16x8(vUhInData0.val[1]);
            vUhScanRes0.val[2] = neon_scan_across_vector_u16x8(vUhInData0.val[2]);
            vUhScanRes0.val[3] = neon_scan_across_vector_u16x8(vUhInData0.val[3]);

            uint32x4x4_t vScanRes0, vScanRes1;
            vScanRes0.val[0] = vaddw_u16(vdupq_n_u32(sum[0]), vget_low_u16(vUhScanRes0.val[0]));
            vScanRes1.val[0] = vaddw_u16(vdupq_n_u32(sum[0]), vget_high_u16(vUhScanRes0.val[0]));
            vScanRes0.val[1] = vaddw_u16(vdupq_n_u32(sum[1]), vget_low_u16(vUhScanRes0.val[1]));
            vScanRes1.val[1] = vaddw_u16(vdupq_n_u32(sum[1]), vget_high_u16(vUhScanRes0.val[1]));
            vScanRes0.val[2] = vaddw_u16(vdupq_n_u32(sum[2]), vget_low_u16(vUhScanRes0.val[2]));
            vScanRes1.val[2] = vaddw_u16(vdupq_n_u32(sum[2]), vget_high_u16(vUhScanRes0.val[2]));
            vScanRes0.val[3] = vaddw_u16(vdupq_n_u32(sum[3]), vget_low_u16(vUhScanRes0.val[3]));
            vScanRes1.val[3] = vaddw_u16(vdupq_n_u32(sum[3]), vget_high_u16(vUhScanRes0.val[3]));

            sum[0] = vgetq_lane_u32(vScanRes1.val[0], 3);
            sum[1] = vgetq_lane_u32(vScanRes1.val[1], 3);
            sum[2] = vgetq_lane_u32(vScanRes1.val[2], 3);
            sum[3] = vgetq_lane_u32(vScanRes1.val[3], 3);

            prefetch(out + h * outWidthStride + (w + 1 + 0) * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 2) * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 4) * channels);
            prefetch(out + h * outWidthStride + (w + 1 + 6) * channels);
            uint32x4x4_t vInData1_0 = vld4q_u32(out + h * outWidthStride + (w + 1 + 0) * channels);
            uint32x4x4_t vInData1_1 = vld4q_u32(out + h * outWidthStride + (w + 1 + 4) * channels);

            uint32x4x4_t vRes0, vRes1;
            vRes0.val[0] = vaddq_u32(vInData1_0.val[0], vScanRes0.val[0]);
            vRes0.val[1] = vaddq_u32(vInData1_0.val[1], vScanRes0.val[1]);
            vRes0.val[2] = vaddq_u32(vInData1_0.val[2], vScanRes0.val[2]);
            vRes0.val[3] = vaddq_u32(vInData1_0.val[3], vScanRes0.val[3]);
            vRes1.val[0] = vaddq_u32(vInData1_1.val[0], vScanRes1.val[0]);
            vRes1.val[1] = vaddq_u32(vInData1_1.val[1], vScanRes1.val[1]);
            vRes1.val[2] = vaddq_u32(vInData1_1.val[2], vScanRes1.val[2]);
            vRes1.val[3] = vaddq_u32(vInData1_1.val[3], vScanRes1.val[3]);

            vst4q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 0) * channels), vRes0);
            vst4q_u32((uint32_t *)(out + (h + 1) * outWidthStride + (w + 1 + 4) * channels), vRes1);
        }

        for (; w < width; w++) {
            for (int32_t c = 0; c < channels; c++) {
                uint8_t in_v = in[h * inWidthStride + w * channels + c];
                sum[c] += in_v;
                out[(h + 1) * outWidthStride + (w + 1) * channels + c] =
                    sum[c] + out[h * outWidthStride + (w + 1) * channels + c];
            }
        }
    }
}

#ifdef PPLCV_ARM_FASTMODE

// floating point error introduced by application of associative law and commutative law during vectorization
// 1. During handling serval chunks of data to exploit ILP
// 2. During parallel reduction inside data chunk itself
template <>
void IntegralImage<float, float, 1>(int32_t height,
                                    int32_t width,
                                    int32_t inWidthStride,
                                    const float *in,
                                    int32_t outWidthStride,
                                    float *out)
{
    constexpr int channels = 1;

    int32_t outWidth = width + 1;
    memset(out, 0, sizeof(float) * outWidth * channels);

    for (int32_t h = 0; h < height; h++) {
        memset(out + (h + 1) * outWidthStride, 0, sizeof(float) * channels);

        int32_t w = 0;
        float sum[channels] = {0};
        for (; w <= width - 8; w += 8) {
            float32x4_t vInData00 = vld1q_f32(in + h * inWidthStride + (w + 0) * channels);
            float32x4_t vInData01 = vld1q_f32(in + h * inWidthStride + (w + 4) * channels);
            float32x4_t vInData10 = vld1q_f32(out + h * outWidthStride + (w + 1 + 0) * channels);
            float32x4_t vInData11 = vld1q_f32(out + h * outWidthStride + (w + 1 + 4) * channels);

            float32x4_t vScanRes0 = neon_scan_across_vector_f32x4(vInData00);
            float32x4_t vScanRes1 = neon_scan_across_vector_f32x4(vInData01);

            vScanRes0 = vaddq_f32(vScanRes0, vdupq_n_f32(sum[0]));
            vScanRes1 = vaddq_f32(vScanRes1, vdupq_laneq_f32(vScanRes0, 3));

            sum[0] = vgetq_lane_f32(vScanRes1, 3);

            float32x4_t vRes0 = vaddq_f32(vInData10, vScanRes0);
            float32x4_t vRes1 = vaddq_f32(vInData11, vScanRes1);

            vst1q_f32((out + (h + 1) * outWidthStride + (w + 1 + 0) * channels), vRes0);
            vst1q_f32((out + (h + 1) * outWidthStride + (w + 1 + 4) * channels), vRes1);
        }

        for (; w < width; w++) {
            for (int32_t c = 0; c < channels; c++) {
                float in_v = in[h * inWidthStride + w * channels + c];
                sum[c] += in_v;
                out[(h + 1) * outWidthStride + (w + 1) * channels + c] =
                    sum[c] + out[h * outWidthStride + (w + 1) * channels + c];
            }
        }
    }
}

#endif

template <typename TSrc, typename TDst, int32_t channels>
void IntegralImageDeprecate(int32_t height,
                            int32_t width,
                            int32_t inWidthStride,
                            const TSrc *in,
                            int32_t outWidthStride,
                            TDst *out)
{
    // integral by row
    for (int32_t h = 0; h < height; h++) {
        TDst sum[channels] = {0};
        for (int32_t w = 0; w < width; w++) {
            for (int32_t c = 0; c < channels; c++) {
                TSrc in_v = in[h * inWidthStride + w * channels + c];
                sum[c] += in_v;
                TDst prev_line_data = (h == 0) ? 0 : out[(h - 1) * outWidthStride + w * channels + c];
                out[h * outWidthStride + w * channels + c] = sum[c] + prev_line_data;
            }
        }
    }
}

template <>
::ppl::common::RetCode Integral<float, float, 1>(int32_t inHeight,
                                                 int32_t inWidth,
                                                 int32_t inWidthStride,
                                                 const float *inData,
                                                 int32_t outHeight,
                                                 int32_t outWidth,
                                                 int32_t outWidthStride,
                                                 float *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<float, float, 1>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<float, float, 1>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<float, float, 3>(int32_t inHeight,
                                                 int32_t inWidth,
                                                 int32_t inWidthStride,
                                                 const float *inData,
                                                 int32_t outHeight,
                                                 int32_t outWidth,
                                                 int32_t outWidthStride,
                                                 float *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<float, float, 3>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<float, float, 3>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<float, float, 4>(int32_t inHeight,
                                                 int32_t inWidth,
                                                 int32_t inWidthStride,
                                                 const float *inData,
                                                 int32_t outHeight,
                                                 int32_t outWidth,
                                                 int32_t outWidthStride,
                                                 float *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<float, float, 4>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<float, float, 4>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<uint8_t, int32_t, 1>(int32_t inHeight,
                                                     int32_t inWidth,
                                                     int32_t inWidthStride,
                                                     const uint8_t *inData,
                                                     int32_t outHeight,
                                                     int32_t outWidth,
                                                     int32_t outWidthStride,
                                                     int32_t *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<uint8_t, uint32_t, 1>(
            inHeight, inWidth, inWidthStride, inData, outWidthStride, (uint32_t *)outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<uint8_t, uint32_t, 1>(
            inHeight, inWidth, inWidthStride, inData, outWidthStride, (uint32_t *)outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<uint8_t, int32_t, 3>(int32_t inHeight,
                                                     int32_t inWidth,
                                                     int32_t inWidthStride,
                                                     const uint8_t *inData,
                                                     int32_t outHeight,
                                                     int32_t outWidth,
                                                     int32_t outWidthStride,
                                                     int32_t *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<uint8_t, uint32_t, 3>(
            inHeight, inWidth, inWidthStride, inData, outWidthStride, (uint32_t *)outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<uint8_t, uint32_t, 3>(
            inHeight, inWidth, inWidthStride, inData, outWidthStride, (uint32_t *)outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<uint8_t, int32_t, 4>(int32_t inHeight,
                                                     int32_t inWidth,
                                                     int32_t inWidthStride,
                                                     const uint8_t *inData,
                                                     int32_t outHeight,
                                                     int32_t outWidth,
                                                     int32_t outWidthStride,
                                                     int32_t *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<uint8_t, uint32_t, 4>(
            inHeight, inWidth, inWidthStride, inData, outWidthStride, (uint32_t *)outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<uint8_t, uint32_t, 4>(
            inHeight, inWidth, inWidthStride, inData, outWidthStride, (uint32_t *)outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
