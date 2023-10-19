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

#include "ppl/cv/arm/crop.h"
#include "ppl/cv/types.h"
#include "common.hpp"
#include <string.h>
#include <arm_neon.h>

#include <cmath>

namespace ppl {
namespace cv {
namespace arm {

static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = data < 0 ? 0 : val;
    return val;
}

template <typename T>
static void crop_line_common(const T* src, T* dst, int32_t outWidth, float scale)
{
    int32_t i = 0;
    for (; i < outWidth; i++, src++) {
        dst[i] = scale * src[0];
    }
}

template <>
void crop_line_common(const uint8_t* src, uint8_t* dst, int32_t outWidth, float scale)
{
    int32_t i = 0;

    float32x4_t vScale = vdupq_n_f32(scale);
    for (; i <= outWidth - 16; i += 16) {
        prefetch(src + i);
        uint8x16_t vInData = vld1q_u8(src + i);

        uint16x8_t vUHInData0 = vmovl_u8(vget_low_u8(vInData));
        uint16x8_t vUHInData1 = vmovl_high_u8(vInData);

        uint32x4_t vUiInData0 = vmovl_u16(vget_low_u16(vUHInData0));
        uint32x4_t vUiInData1 = vmovl_high_u16(vUHInData0);
        uint32x4_t vUiInData2 = vmovl_u16(vget_low_u16(vUHInData1));
        uint32x4_t vUiInData3 = vmovl_high_u16(vUHInData1);

        float32x4_t vFData0 = vcvtq_f32_u32(vUiInData0);
        float32x4_t vFData1 = vcvtq_f32_u32(vUiInData1);
        float32x4_t vFData2 = vcvtq_f32_u32(vUiInData2);
        float32x4_t vFData3 = vcvtq_f32_u32(vUiInData3);

        float32x4_t vFRes0 = vmulq_f32(vFData0, vScale);
        float32x4_t vFRes1 = vmulq_f32(vFData1, vScale);
        float32x4_t vFRes2 = vmulq_f32(vFData2, vScale);
        float32x4_t vFRes3 = vmulq_f32(vFData3, vScale);

        int32x4_t vSiData0 = vcvtnq_s32_f32(vFRes0);
        int32x4_t vSiData1 = vcvtnq_s32_f32(vFRes1);
        int32x4_t vSiData2 = vcvtnq_s32_f32(vFRes2);
        int32x4_t vSiData3 = vcvtnq_s32_f32(vFRes3);

        uint16x4_t vUhData00 = vqmovun_s32(vSiData0);
        uint16x8_t vUhData0 = vqmovun_high_s32(vUhData00, vSiData1);
        uint16x4_t vUhData10 = vqmovun_s32(vSiData2);
        uint16x8_t vUhData1 = vqmovun_high_s32(vUhData10, vSiData3);

        uint8x8_t vOutData0 = vqmovn_u16(vUhData0);
        uint8x8_t vOutData1 = vqmovn_u16(vUhData1);
        vst1_u8(dst + i, vOutData0);
        vst1_u8(dst + i + 8, vOutData1);
    }

    for (; i < outWidth; i++) {
        int32_t val = lrintf(scale * src[i]);
        dst[i] = sat_cast(val);
    }
}

template <typename T>
void crop_line_noscale(const T* src, T* dst, int32_t outWidth)
{
    memcpy(dst, src, outWidth * sizeof(T));
}

template <typename T, int32_t channels>
ppl::common::RetCode Crop(const int32_t inHeight,
                          const int32_t inWidth,
                          int32_t inWidthStride,
                          const T* inData,
                          int32_t outHeight,
                          int32_t outWidth,
                          int32_t outWidthStride,
                          T* outData,
                          int32_t left,
                          int32_t top,
                          float scale)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) { return ppl::common::RC_INVALID_VALUE; }

    const T* src = inData + top * inWidthStride + left * channels;
    T* dst = outData;

    int32_t out_row_width = outWidth * channels;
    bool noscale = (scale == 1.0f);
    for (int32_t i = 0; i < outHeight; i++) {
        if (noscale) {
            crop_line_noscale(src, dst, out_row_width);
        } else {
            crop_line_common(src, dst, out_row_width, scale);
        }
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template ppl::common::RetCode Crop<float, 1>(const int32_t inHeight,
                                             const int32_t inWidth,
                                             int32_t inWidthStride,
                                             const float* inData,
                                             int32_t outHeight,
                                             int32_t outWidth,
                                             int32_t outWidthStride,
                                             float* outData,
                                             int32_t left,
                                             int32_t top,
                                             float scale);

template ppl::common::RetCode Crop<float, 3>(const int32_t inHeight,
                                             const int32_t inWidth,
                                             int32_t inWidthStride,
                                             const float* inData,
                                             int32_t outHeight,
                                             int32_t outWidth,
                                             int32_t outWidthStride,
                                             float* outData,
                                             int32_t left,
                                             int32_t top,
                                             float scale);

template ppl::common::RetCode Crop<float, 4>(const int32_t inHeight,
                                             const int32_t inWidth,
                                             int32_t inWidthStride,
                                             const float* inData,
                                             int32_t outHeight,
                                             int32_t outWidth,
                                             int32_t outWidthStride,
                                             float* outData,
                                             int32_t left,
                                             int32_t top,
                                             float scale);

template ppl::common::RetCode Crop<uint8_t, 1>(const int32_t inHeight,
                                               const int32_t inWidth,
                                               int32_t inWidthStride,
                                               const uint8_t* inData,
                                               int32_t outHeight,
                                               int32_t outWidth,
                                               int32_t outWidthStride,
                                               uint8_t* outData,
                                               int32_t left,
                                               int32_t top,
                                               float scale);

template ppl::common::RetCode Crop<uint8_t, 3>(const int32_t inHeight,
                                               const int32_t inWidth,
                                               int32_t inWidthStride,
                                               const uint8_t* inData,
                                               int32_t outHeight,
                                               int32_t outWidth,
                                               int32_t outWidthStride,
                                               uint8_t* outData,
                                               int32_t left,
                                               int32_t top,
                                               float scale);

template ppl::common::RetCode Crop<uint8_t, 4>(const int32_t inHeight,
                                               const int32_t inWidth,
                                               int32_t inWidthStride,
                                               const uint8_t* inData,
                                               int32_t outHeight,
                                               int32_t outWidth,
                                               int32_t outWidthStride,
                                               uint8_t* outData,
                                               int32_t left,
                                               int32_t top,
                                               float scale);

}
}
} // namespace ppl::cv::arm
