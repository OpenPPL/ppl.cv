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

#include "ppl/cv/arm/convertto.h"
#include "ppl/cv/types.h"
#include "common.hpp"
#include <string.h>
#include <arm_neon.h>

#include <cmath>
#include <algorithm>

namespace ppl {
namespace cv {
namespace arm {

template <typename TSrc, typename TDst>
::ppl::common::RetCode ChangeDataType(int32_t height,
                                                              int32_t width,
                                                              int32_t nc,
                                                              int32_t inWidthStride,
                                                              const TSrc* inData,
                                                              int32_t outWidthStride,
                                                              TDst* outData)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t row_width = width * nc;
    for (int32_t h = 0; h < height; ++h) {
        const TSrc* base_in = inData + h * inWidthStride;
        TDst* base_out = outData + h * outWidthStride;
        int32_t w = 0;
        for (; w < row_width; ++w) {
            base_out[w] = static_cast<TDst>(base_in[w]);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template<>
::ppl::common::RetCode ChangeDataType<float, uint8_t>(int32_t height,
                                                              int32_t width,
                                                              int32_t nc,
                                                              int32_t inWidthStride,
                                                              const float* inData,
                                                              int32_t outWidthStride,
                                                              uint8_t* outData)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t row_width = width * nc;
    for (int32_t h = 0; h < height; ++h) {
        const float* base_in = inData + h * inWidthStride;
        uint8_t* base_out = outData + h * outWidthStride;
        int32_t w = 0;
        for (; w <= row_width - 16; w += 16) {
            prefetch(base_in + w);
            float32x4_t vFData0 = vld1q_f32(base_in + w + 0);
            float32x4_t vFData1 = vld1q_f32(base_in + w + 4);
            float32x4_t vFData2 = vld1q_f32(base_in + w + 8);
            float32x4_t vFData3 = vld1q_f32(base_in + w + 12);

            float32x4_t vFRes0 = vFData0;
            float32x4_t vFRes1 = vFData1;
            float32x4_t vFRes2 = vFData2;
            float32x4_t vFRes3 = vFData3;

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
            vst1_u8(base_out + w, vOutData0);
            vst1_u8(base_out + w + 8, vOutData1);
        }
        for (; w < row_width; ++w) {
            float value = base_in[w];
            base_out[w] = static_cast<uint8_t>(lrintf(std::min(std::max(value, 0.0f), 255.0f)));
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename TSrc, typename TDst>
::ppl::common::RetCode ChangeDataTypeAndScale(int32_t height,
                                              int32_t width,
                                              int32_t nc,
                                              int32_t inWidthStride,
                                              const TSrc* inData,
                                              int32_t outWidthStride,
                                              TDst* outData,
                                              float scale,
                                              float delta);

template <>
::ppl::common::RetCode ChangeDataTypeAndScale<uint8_t, float>(int32_t height,
                                                              int32_t width,
                                                              int32_t nc,
                                                              int32_t inWidthStride,
                                                              const uint8_t* inData,
                                                              int32_t outWidthStride,
                                                              float* outData,
                                                              float scale,
                                                              float delta)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t row_width = width * nc;
    for (int32_t h = 0; h < height; ++h) {
        const uint8_t* base_in = inData + h * inWidthStride;
        float* base_out = outData + h * outWidthStride;
        int32_t w = 0;
        for (; w < row_width; ++w) {
            base_out[w] = std::fma(scale, static_cast<float>(base_in[w]), delta);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ChangeDataTypeAndScale<float, float>(int32_t height,
                                                            int32_t width,
                                                            int32_t nc,
                                                            int32_t inWidthStride,
                                                            const float* inData,
                                                            int32_t outWidthStride,
                                                            float* outData,
                                                            float scale,
                                                            float delta)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t row_width = width * nc;
    for (int32_t h = 0; h < height; ++h) {
        const float* base_in = inData + h * inWidthStride;
        float* base_out = outData + h * outWidthStride;
        int32_t w = 0;
        for (; w < row_width; ++w) {
            base_out[w] = std::fma(scale, base_in[w], delta);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ChangeDataTypeAndScale<float, uint8_t>(int32_t height,
                                                              int32_t width,
                                                              int32_t nc,
                                                              int32_t inWidthStride,
                                                              const float* inData,
                                                              int32_t outWidthStride,
                                                              uint8_t* outData,
                                                              float scale,
                                                              float delta)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t row_width = width * nc;
    for (int32_t h = 0; h < height; ++h) {
        const float* base_in = inData + h * inWidthStride;
        uint8_t* base_out = outData + h * outWidthStride;
        int32_t w = 0;
        for (; w <= row_width - 16; w += 16) {
            prefetch(base_in + w);
            float32x4_t vFData0 = vld1q_f32(base_in + w + 0);
            float32x4_t vFData1 = vld1q_f32(base_in + w + 4);
            float32x4_t vFData2 = vld1q_f32(base_in + w + 8);
            float32x4_t vFData3 = vld1q_f32(base_in + w + 12);

            float32x4_t vScale = vdupq_n_f32(scale);
            float32x4_t vFRes0 = vdupq_n_f32(delta);
            float32x4_t vFRes1 = vdupq_n_f32(delta);
            float32x4_t vFRes2 = vdupq_n_f32(delta);
            float32x4_t vFRes3 = vdupq_n_f32(delta);
            vFRes0 = vfmaq_f32(vFRes0, vFData0, vScale);
            vFRes1 = vfmaq_f32(vFRes1, vFData1, vScale);
            vFRes2 = vfmaq_f32(vFRes2, vFData2, vScale);
            vFRes3 = vfmaq_f32(vFRes3, vFData3, vScale);

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
            vst1_u8(base_out + w, vOutData0);
            vst1_u8(base_out + w + 8, vOutData1);
        }
        for (; w < row_width; ++w) {
            float value = scale * base_in[w] + delta;
            base_out[w] = static_cast<uint8_t>(lrintf(std::min(std::max(value, 0.0f), 255.0f)));
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ChangeDataTypeAndScale<uint8_t, uint8_t>(int32_t height,
                                                                int32_t width,
                                                                int32_t nc,
                                                                int32_t inWidthStride,
                                                                const uint8_t* inData,
                                                                int32_t outWidthStride,
                                                                uint8_t* outData,
                                                                float scale,
                                                                float delta)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t row_width = width * nc;
    for (int32_t h = 0; h < height; ++h) {
        const uint8_t* base_in = inData + h * inWidthStride;
        uint8_t* base_out = outData + h * outWidthStride;
        int32_t w = 0;
        for (; w <= row_width - 16; w += 16) {
            prefetch(base_in + w);
            uint8x16_t vInData = vld1q_u8(base_in + w);

            float32x4_t vScale = vdupq_n_f32(scale);
            float32x4_t vFRes0 = vdupq_n_f32(delta);
            float32x4_t vFRes1 = vdupq_n_f32(delta);
            float32x4_t vFRes2 = vdupq_n_f32(delta);
            float32x4_t vFRes3 = vdupq_n_f32(delta);

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

            vFRes0 = vfmaq_f32(vFRes0, vFData0, vScale);
            vFRes1 = vfmaq_f32(vFRes1, vFData1, vScale);
            vFRes2 = vfmaq_f32(vFRes2, vFData2, vScale);
            vFRes3 = vfmaq_f32(vFRes3, vFData3, vScale);

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
            vst1_u8(base_out + w, vOutData0);
            vst1_u8(base_out + w + 8, vOutData1);
        }
        for (; w < row_width; ++w) {
            float value = scale * base_in[w] + delta;
            base_out[w] = static_cast<uint8_t>(lrintf(std::min(std::max(value, 0.0f), 255.0f)));
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ConvertTo<float, uint8_t, 1>(int32_t height,
                                                    int32_t width,
                                                    int32_t inWidthStride,
                                                    const float* inData,
                                                    int32_t outWidthStride,
                                                    uint8_t* outData,
                                                    float scale,
                                                    float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<float, uint8_t>(
            height, width, 1, inWidthStride, inData, outWidthStride, outData);
    }
    
    return ChangeDataTypeAndScale<float, uint8_t>(
        height, width, 1, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<float, uint8_t, 3>(int32_t height,
                                                    int32_t width,
                                                    int32_t inWidthStride,
                                                    const float* inData,
                                                    int32_t outWidthStride,
                                                    uint8_t* outData,
                                                    float scale,
                                                    float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<float, uint8_t>(
            height, width, 3, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<float, uint8_t>(
        height, width, 3, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<float, uint8_t, 4>(int32_t height,
                                                    int32_t width,
                                                    int32_t inWidthStride,
                                                    const float* inData,
                                                    int32_t outWidthStride,
                                                    uint8_t* outData,
                                                    float scale,
                                                    float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<float, uint8_t>(
            height, width, 4, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<float, uint8_t>(
        height, width, 4, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, uint8_t, 1>(int32_t height,
                                                      int32_t width,
                                                      int32_t inWidthStride,
                                                      const uint8_t* inData,
                                                      int32_t outWidthStride,
                                                      uint8_t* outData,
                                                      float scale,
                                                      float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<uint8_t, uint8_t>(
            height, width, 1, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<uint8_t, uint8_t>(
        height, width, 1, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, uint8_t, 3>(int32_t height,
                                                      int32_t width,
                                                      int32_t inWidthStride,
                                                      const uint8_t* inData,
                                                      int32_t outWidthStride,
                                                      uint8_t* outData,
                                                      float scale,
                                                      float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<uint8_t, uint8_t>(
            height, width, 3, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<uint8_t, uint8_t>(
        height, width, 3, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, uint8_t, 4>(int32_t height,
                                                      int32_t width,
                                                      int32_t inWidthStride,
                                                      const uint8_t* inData,
                                                      int32_t outWidthStride,
                                                      uint8_t* outData,
                                                      float scale,
                                                      float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<uint8_t, uint8_t>(
            height, width, 4, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<uint8_t, uint8_t>(
        height, width, 4, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, float, 1>(int32_t height,
                                                    int32_t width,
                                                    int32_t inWidthStride,
                                                    const uint8_t* inData,
                                                    int32_t outWidthStride,
                                                    float* outData,
                                                    float scale,
                                                    float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<uint8_t, float>(
            height, width, 1, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<uint8_t, float>(
        height, width, 1, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, float, 3>(int32_t height,
                                                    int32_t width,
                                                    int32_t inWidthStride,
                                                    const uint8_t* inData,
                                                    int32_t outWidthStride,
                                                    float* outData,
                                                    float scale,
                                                    float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<uint8_t, float>(
            height, width, 3, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<uint8_t, float>(
        height, width, 3, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, float, 4>(int32_t height,
                                                    int32_t width,
                                                    int32_t inWidthStride,
                                                    const uint8_t* inData,
                                                    int32_t outWidthStride,
                                                    float* outData,
                                                    float scale,
                                                    float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<uint8_t, float>(
            height, width, 4, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<uint8_t, float>(
        height, width, 4, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<float, float, 1>(int32_t height,
                                                  int32_t width,
                                                  int32_t inWidthStride,
                                                  const float* inData,
                                                  int32_t outWidthStride,
                                                  float* outData,
                                                  float scale,
                                                  float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<float, float>(
            height, width, 1, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<float, float>(
        height, width, 1, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<float, float, 3>(int32_t height,
                                                  int32_t width,
                                                  int32_t inWidthStride,
                                                  const float* inData,
                                                  int32_t outWidthStride,
                                                  float* outData,
                                                  float scale,
                                                  float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<float, float>(
            height, width, 3, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<float, float>(
        height, width, 3, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

template <>
::ppl::common::RetCode ConvertTo<float, float, 4>(int32_t height,
                                                  int32_t width,
                                                  int32_t inWidthStride,
                                                  const float* inData,
                                                  int32_t outWidthStride,
                                                  float* outData,
                                                  float scale,
                                                  float delta)
{
    if (scale == 1 && delta == 0) {
        return ChangeDataType<float, float>(
            height, width, 4, inWidthStride, inData, outWidthStride, outData);
    }

    return ChangeDataTypeAndScale<float, float>(
        height, width, 4, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

}
}
} // namespace ppl::cv::arm
