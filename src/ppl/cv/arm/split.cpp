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

#include "ppl/cv/arm/split.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

template <>
::ppl::common::RetCode Split3Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const float* inData,
                                      int32_t outWidthStride,
                                      float* outDataChannel0,
                                      float* outDataChannel1,
                                      float* outDataChannel2)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (inWidthStride == width * 3 && outWidthStride == width) {
        width *= height;
        inWidthStride *= height;
        outWidthStride *= height;
        height = 1;
    }

    for (int32_t h = 0; h < height; h++) {
        const float* src_ptr = inData + h * inWidthStride;
        float* dst0_ptr = outDataChannel0 + h * outWidthStride;
        float* dst1_ptr = outDataChannel1 + h * outWidthStride;
        float* dst2_ptr = outDataChannel2 + h * outWidthStride;
        int32_t i = 0;
        for (; i <= width - 4; i += 4) {
            prefetch(src_ptr + i * 3);
            float32x4x3_t vData = vld3q_f32(src_ptr + 3 * i);
            vst1q_f32(dst0_ptr + i, vData.val[0]);
            vst1q_f32(dst1_ptr + i, vData.val[1]);
            vst1q_f32(dst2_ptr + i, vData.val[2]);
        }
        for (; i < width; i++) {
            dst0_ptr[i] = src_ptr[i * 3 + 0];
            dst1_ptr[i] = src_ptr[i * 3 + 1];
            dst2_ptr[i] = src_ptr[i * 3 + 2];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Split3Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const uint8_t* inData,
                                      int32_t outWidthStride,
                                      uint8_t* outDataChannel0,
                                      uint8_t* outDataChannel1,
                                      uint8_t* outDataChannel2)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (inWidthStride == width * 3 && outWidthStride == width) {
        width *= height;
        inWidthStride *= height;
        outWidthStride *= height;
        height = 1;
    }

    for (int32_t h = 0; h < height; h++) {
        const uint8_t* src_ptr = inData + h * inWidthStride;
        uint8_t* dst0_ptr = outDataChannel0 + h * outWidthStride;
        uint8_t* dst1_ptr = outDataChannel1 + h * outWidthStride;
        uint8_t* dst2_ptr = outDataChannel2 + h * outWidthStride;
        int32_t i = 0;
        for (; i <= width - 16; i += 16) {
            prefetch(src_ptr + i * 3);
            uint8x16x3_t vData = vld3q_u8(src_ptr + 3 * i);
            vst1q_u8(dst0_ptr + i, vData.val[0]);
            vst1q_u8(dst1_ptr + i, vData.val[1]);
            vst1q_u8(dst2_ptr + i, vData.val[2]);
        }
        for (; i < width; i++) {
            dst0_ptr[i] = src_ptr[i * 3 + 0];
            dst1_ptr[i] = src_ptr[i * 3 + 1];
            dst2_ptr[i] = src_ptr[i * 3 + 2];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Split4Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const float* inData,
                                      int32_t outWidthStride,
                                      float* outDataChannel0,
                                      float* outDataChannel1,
                                      float* outDataChannel2,
                                      float* outDataChannel3)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (inWidthStride == width * 4 && outWidthStride == width) {
        width *= height;
        inWidthStride *= height;
        outWidthStride *= height;
        height = 1;
    }

    for (int32_t h = 0; h < height; h++) {
        const float* src_ptr = inData + h * inWidthStride;
        float* dst0_ptr = outDataChannel0 + h * outWidthStride;
        float* dst1_ptr = outDataChannel1 + h * outWidthStride;
        float* dst2_ptr = outDataChannel2 + h * outWidthStride;
        float* dst3_ptr = outDataChannel3 + h * outWidthStride;
        int32_t i = 0;
        for (; i <= width - 4; i += 4) {
            prefetch(src_ptr + i * 4);
            float32x4x4_t vData = vld4q_f32(src_ptr + 4 * i);
            vst1q_f32(dst0_ptr + i, vData.val[0]);
            vst1q_f32(dst1_ptr + i, vData.val[1]);
            vst1q_f32(dst2_ptr + i, vData.val[2]);
            vst1q_f32(dst3_ptr + i, vData.val[3]);
        }
        for (; i < width; i++) {
            dst0_ptr[i] = src_ptr[i * 4 + 0];
            dst1_ptr[i] = src_ptr[i * 4 + 1];
            dst2_ptr[i] = src_ptr[i * 4 + 2];
            dst3_ptr[i] = src_ptr[i * 4 + 3];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Split4Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const uint8_t* inData,
                                      int32_t outWidthStride,
                                      uint8_t* outDataChannel0,
                                      uint8_t* outDataChannel1,
                                      uint8_t* outDataChannel2,
                                      uint8_t* outDataChannel3)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (inWidthStride == width * 4 && outWidthStride == width) {
        width *= height;
        inWidthStride *= height;
        outWidthStride *= height;
        height = 1;
    }

    for (int32_t h = 0; h < height; h++) {
        const uint8_t* src_ptr = inData + h * inWidthStride;
        uint8_t* dst0_ptr = outDataChannel0 + h * outWidthStride;
        uint8_t* dst1_ptr = outDataChannel1 + h * outWidthStride;
        uint8_t* dst2_ptr = outDataChannel2 + h * outWidthStride;
        uint8_t* dst3_ptr = outDataChannel3 + h * outWidthStride;
        int32_t i = 0;
        for (; i <= width - 16; i += 16) {
            prefetch(src_ptr + i * 4);
            uint8x16x4_t vData = vld4q_u8(src_ptr + 4 * i);
            vst1q_u8(dst0_ptr + i, vData.val[0]);
            vst1q_u8(dst1_ptr + i, vData.val[1]);
            vst1q_u8(dst2_ptr + i, vData.val[2]);
            vst1q_u8(dst3_ptr + i, vData.val[3]);
        }
        for (; i < width; i++) {
            dst0_ptr[i] = src_ptr[i * 4 + 0];
            dst1_ptr[i] = src_ptr[i * 4 + 1];
            dst2_ptr[i] = src_ptr[i * 4 + 2];
            dst3_ptr[i] = src_ptr[i * 4 + 3];
        }
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
