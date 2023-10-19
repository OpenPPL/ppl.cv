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

#include "ppl/cv/arm/merge.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <cstdio>
namespace ppl {
namespace cv {
namespace arm {

template <typename T>
void mergeSOA2AOS_3Channels(int32_t height,
                            int32_t width,
                            int32_t inWidthStride,
                            const T** in,
                            int32_t outWidthStride,
                            T* out);

template <>
void mergeSOA2AOS_3Channels(int32_t height,
                            int32_t width,
                            int32_t inWidthStride,
                            const uint8_t** in,
                            int32_t outWidthStride,
                            uint8_t* out)
{
    const uint8_t* src0 = in[0];
    const uint8_t* src1 = in[1];
    const uint8_t* src2 = in[2];
    uint8_t* out0 = out;
    for (int32_t h = 0; h < height; h++) {
        const uint8_t* src0_row = src0 + h * inWidthStride;
        const uint8_t* src1_row = src1 + h * inWidthStride;
        const uint8_t* src2_row = src2 + h * inWidthStride;
        uint8_t* out0_row = out0 + h * outWidthStride;
        int32_t w = 0;
        for (; w <= width - 16; w += 16) {
            prefetch(src0_row + w);
            prefetch(src1_row + w);
            prefetch(src2_row + w);
            uint8x16x3_t vData;
            vData.val[0] = vld1q_u8(src0_row + w);
            vData.val[1] = vld1q_u8(src1_row + w);
            vData.val[2] = vld1q_u8(src2_row + w);
            vst3q_u8(out0_row + w * 3, vData);
        }
        for (; w < width; w++) {
            out0_row[w * 3 + 0] = src0_row[w];
            out0_row[w * 3 + 1] = src1_row[w];
            out0_row[w * 3 + 2] = src2_row[w];
        }
    }
}

template <>
void mergeSOA2AOS_3Channels(int32_t height,
                            int32_t width,
                            int32_t inWidthStride,
                            const float** in,
                            int32_t outWidthStride,
                            float* out)
{
    const float* src0 = in[0];
    const float* src1 = in[1];
    const float* src2 = in[2];
    float* out0 = out;
    for (int32_t h = 0; h < height; h++) {
        const float* src0_row = src0 + h * inWidthStride;
        const float* src1_row = src1 + h * inWidthStride;
        const float* src2_row = src2 + h * inWidthStride;
        float* out0_row = out0 + h * outWidthStride;
        int32_t w = 0;
        for (; w <= width - 4; w += 4) {
            prefetch(src0_row + w);
            prefetch(src1_row + w);
            prefetch(src2_row + w);
            float32x4x3_t vData;
            vData.val[0] = vld1q_f32(src0_row + w);
            vData.val[1] = vld1q_f32(src1_row + w);
            vData.val[2] = vld1q_f32(src2_row + w);
            vst3q_f32(out0_row + w * 3, vData);
        }

        for (; w < width; w++) {
            out0_row[w * 3 + 0] = src0_row[w];
            out0_row[w * 3 + 1] = src1_row[w];
            out0_row[w * 3 + 2] = src2_row[w];
        }
    }
}

template <typename T>
void mergeSOA2AOS_4Channels(int32_t height,
                            int32_t width,
                            int32_t inWidthStride,
                            const T** in,
                            int32_t outWidthStride,
                            T* out);

template <>
void mergeSOA2AOS_4Channels(int32_t height,
                            int32_t width,
                            int32_t inWidthStride,
                            const uint8_t** in,
                            int32_t outWidthStride,
                            uint8_t* out)
{
    const uint8_t* src0 = in[0];
    const uint8_t* src1 = in[1];
    const uint8_t* src2 = in[2];
    const uint8_t* src3 = in[3];
    uint8_t* out0 = out;
    for (int32_t h = 0; h < height; h++) {
        const uint8_t* src0_row = src0 + h * inWidthStride;
        const uint8_t* src1_row = src1 + h * inWidthStride;
        const uint8_t* src2_row = src2 + h * inWidthStride;
        const uint8_t* src3_row = src3 + h * inWidthStride;
        uint8_t* out0_row = out0 + h * outWidthStride;
        int32_t w = 0;
        for (; w <= width - 16; w += 16) {
            prefetch(src0_row + w);
            prefetch(src1_row + w);
            prefetch(src2_row + w);
            prefetch(src3_row + w);
            uint8x16x4_t vData;
            vData.val[0] = vld1q_u8(src0_row + w);
            vData.val[1] = vld1q_u8(src1_row + w);
            vData.val[2] = vld1q_u8(src2_row + w);
            vData.val[3] = vld1q_u8(src3_row + w);
            vst4q_u8(out0_row + w * 4, vData);
        }
        for (; w < width; w++) {
            out0_row[w * 4 + 0] = src0_row[w];
            out0_row[w * 4 + 1] = src1_row[w];
            out0_row[w * 4 + 2] = src2_row[w];
            out0_row[w * 4 + 3] = src3_row[w];
        }
    }
}

template <>
void mergeSOA2AOS_4Channels(int32_t height,
                            int32_t width,
                            int32_t inWidthStride,
                            const float** in,
                            int32_t outWidthStride,
                            float* out)
{
    const float* src0 = in[0];
    const float* src1 = in[1];
    const float* src2 = in[2];
    const float* src3 = in[3];
    float* out0 = out;
    for (int32_t h = 0; h < height; h++) {
        const float* src0_row = src0 + h * inWidthStride;
        const float* src1_row = src1 + h * inWidthStride;
        const float* src2_row = src2 + h * inWidthStride;
        const float* src3_row = src3 + h * inWidthStride;
        float* out0_row = out0 + h * outWidthStride;
        int32_t w = 0;
        for (; w <= width - 4; w += 4) {
            prefetch(src0_row + w);
            prefetch(src1_row + w);
            prefetch(src2_row + w);
            prefetch(src3_row + w);
            float32x4x4_t vData;
            vData.val[0] = vld1q_f32(src0_row + w);
            vData.val[1] = vld1q_f32(src1_row + w);
            vData.val[2] = vld1q_f32(src2_row + w);
            vData.val[3] = vld1q_f32(src3_row + w);
            vst4q_f32(out0_row + w * 4, vData);
        }
        for (; w < width; w++) {
            out0_row[w * 4 + 0] = src0_row[w];
            out0_row[w * 4 + 1] = src1_row[w];
            out0_row[w * 4 + 2] = src2_row[w];
            out0_row[w * 4 + 3] = src3_row[w];
        }
    }
}

template <typename T>
::ppl::common::RetCode Merge3Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const T* inDataC0,
                                      const T* inDataC1,
                                      const T* inDataC2,
                                      int32_t outWidthStride,
                                      T* outData)
{
    if (nullptr == inDataC0 || nullptr == inDataC1 || nullptr == inDataC2 || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (inWidthStride == width && outWidthStride == width * 3) {
        width *= height;
        inWidthStride *= height;
        outWidthStride *= height;
        height = 1;
    }

    const T* inData[3] = {inDataC0, inDataC1, inDataC2};
    mergeSOA2AOS_3Channels<T>(height, width, inWidthStride, inData, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <typename T>
::ppl::common::RetCode Merge4Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const T* inDataC0,
                                      const T* inDataC1,
                                      const T* inDataC2,
                                      const T* inDataC3,
                                      int32_t outWidthStride,
                                      T* outData)
{
    if (nullptr == inDataC0 || nullptr == inDataC1 || nullptr == inDataC2 || nullptr == inDataC3 ||
        nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (inWidthStride == width && outWidthStride == width * 4) {
        width *= height;
        inWidthStride *= height;
        outWidthStride *= height;
        height = 1;
    }

    const T* inData[4] = {inDataC0, inDataC1, inDataC2, inDataC3};
    mergeSOA2AOS_4Channels<T>(height, width, inWidthStride, inData, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Merge3Channels<uint8_t>(int32_t height,
                                                        int32_t width,
                                                        int32_t inWidthStride,
                                                        const uint8_t* inDataC0,
                                                        const uint8_t* inDataC1,
                                                        const uint8_t* inDataC2,
                                                        int32_t outWidthStride,
                                                        uint8_t* outData);
template ::ppl::common::RetCode Merge3Channels<float>(int32_t height,
                                                      int32_t width,
                                                      int32_t inWidthStride,
                                                      const float* inDataC0,
                                                      const float* inDataC1,
                                                      const float* inDataC2,
                                                      int32_t outWidthStride,
                                                      float* outData);
template ::ppl::common::RetCode Merge4Channels<uint8_t>(int32_t height,
                                                        int32_t width,
                                                        int32_t inWidthStride,
                                                        const uint8_t* inDataC0,
                                                        const uint8_t* inDataC1,
                                                        const uint8_t* inDataC2,
                                                        const uint8_t* inDataC3,
                                                        int32_t outWidthStride,
                                                        uint8_t* outData);
template ::ppl::common::RetCode Merge4Channels<float>(int32_t height,
                                                      int32_t width,
                                                      int32_t inWidthStride,
                                                      const float* inDataC0,
                                                      const float* inDataC1,
                                                      const float* inDataC2,
                                                      const float* inDataC3,
                                                      int32_t outWidthStride,
                                                      float* outData);

}
}
} // namespace ppl::cv::arm
