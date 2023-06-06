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

namespace ppl::cv::arm {

template <typename T>
::ppl::common::RetCode Split3Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const T* inData,
                                      int32_t outWidthStride,
                                      T* outDataChannel0,
                                      T* outDataChannel1,
                                      T* outDataChannel2)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }

    for (int32_t h = 0; h < height; h++) {
        const T* src_ptr = inData + h * inWidthStride;
        T* dst0_ptr = outDataChannel0 + h * outWidthStride;
        T* dst1_ptr = outDataChannel1 + h * outWidthStride;
        T* dst2_ptr = outDataChannel2 + h * outWidthStride;
        for (int32_t i = 0; i < width; i++) {
            dst0_ptr[i] = src_ptr[i * 3 + 0];
            dst1_ptr[i] = src_ptr[i * 3 + 1];
            dst2_ptr[i] = src_ptr[i * 3 + 2];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Split3Channels<float>(int32_t height,
                                                      int32_t width,
                                                      int32_t inWidthStride,
                                                      const float* inData,
                                                      int32_t outWidthStride,
                                                      float* outDataChannel0,
                                                      float* outDataChannel1,
                                                      float* outDataChannel2);

template ::ppl::common::RetCode Split3Channels<uint8_t>(int32_t height,
                                                        int32_t width,
                                                        int32_t inWidthStride,
                                                        const uint8_t* inData,
                                                        int32_t outWidthStride,
                                                        uint8_t* outDataChannel0,
                                                        uint8_t* outDataChannel1,
                                                        uint8_t* outDataChannel2);

template <typename T>
::ppl::common::RetCode Split4Channels(int32_t height,
                                      int32_t width,
                                      int32_t inWidthStride,
                                      const T* inData,
                                      int32_t outWidthStride,
                                      T* outDataChannel0,
                                      T* outDataChannel1,
                                      T* outDataChannel2,
                                      T* outDataChannel3)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }

    for (int32_t h = 0; h < height; h++) {
        const T* src_ptr = inData + h * inWidthStride;
        T* dst0_ptr = outDataChannel0 + h * outWidthStride;
        T* dst1_ptr = outDataChannel1 + h * outWidthStride;
        T* dst2_ptr = outDataChannel2 + h * outWidthStride;
        T* dst3_ptr = outDataChannel3 + h * outWidthStride;
        for (int32_t i = 0; i < width; i++) {
            dst0_ptr[i] = src_ptr[i * 4 + 0];
            dst1_ptr[i] = src_ptr[i * 4 + 1];
            dst2_ptr[i] = src_ptr[i * 4 + 2];
            dst3_ptr[i] = src_ptr[i * 4 + 3];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Split4Channels<float>(int32_t height,
                                                      int32_t width,
                                                      int32_t inWidthStride,
                                                      const float* inData,
                                                      int32_t outWidthStride,
                                                      float* outDataChannel0,
                                                      float* outDataChannel1,
                                                      float* outDataChannel2,
                                                      float* outDataChannel3);

template ::ppl::common::RetCode Split4Channels<uint8_t>(int32_t height,
                                                        int32_t width,
                                                        int32_t inWidthStride,
                                                        const uint8_t* inData,
                                                        int32_t outWidthStride,
                                                        uint8_t* outDataChannel0,
                                                        uint8_t* outDataChannel1,
                                                        uint8_t* outDataChannel2,
                                                        uint8_t* outDataChannel3);

} // namespace ppl::cv::arm
