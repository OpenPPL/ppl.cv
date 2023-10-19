// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for mulitional information
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

#include "ppl/cv/arm/arithmetic.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>

namespace ppl {
namespace cv {
namespace arm {

::ppl::common::RetCode mls_f32(int32_t height,
                               int32_t width,
                               int32_t channels,
                               int32_t inWidthStride0,
                               const float32_t *inData0,
                               int32_t inWidthStride1,
                               const float32_t *inData1,
                               int32_t outWidthStride,
                               float32_t *outData)
{
    if (nullptr == inData0 || nullptr == inData1 || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride0 <= 0 || inWidthStride1 <= 0 || outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const float32_t *in0 = inData0 + i * inWidthStride0;
        const float32_t *in1 = inData1 + i * inWidthStride1;
        float32_t *out = outData + i * outWidthStride;
        for (; j <= width - 8; j += 8) {
            prefetch(in0 + j);
            prefetch(in1 + j);
            prefetch(out + j);
            float32x4_t vdata0 = vld1q_f32(in0 + j);
            float32x4_t vdata2 = vld1q_f32(in0 + j + 4);
            float32x4_t vdata1 = vld1q_f32(in1 + j);
            float32x4_t vdata3 = vld1q_f32(in1 + j + 4);
            float32x4_t voutData0 = vld1q_f32(out + j);
            float32x4_t voutData1 = vld1q_f32(out + j + 4);
            voutData0 = vmlsq_f32(voutData0, vdata0, vdata1);
            voutData1 = vmlsq_f32(voutData1, vdata2, vdata3);
            vst1q_f32(out + j, voutData0);
            vst1q_f32(out + j + 4, voutData1);
        }
        for (; j <= width - 4; j += 4) {
            float32x4_t vdata0 = vld1q_f32(in0 + j);
            float32x4_t vdata1 = vld1q_f32(in1 + j);
            float32x4_t voutData = vld1q_f32(out + j);
            voutData = vmlsq_f32(voutData, vdata0, vdata1);
            vst1q_f32(out + j, voutData);
        }
        for (; j < width; ++j) {
            out[j] -= in0[j] * in1[j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Mls<float, 1>(int32_t height,
                                     int32_t width,
                                     int32_t inWidthStride0,
                                     const float32_t *inData0,
                                     int32_t inWidthStride1,
                                     const float32_t *inData1,
                                     int32_t outWidthStride,
                                     float32_t *outData)
{
    return mls_f32(height, width, 1, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Mls<float, 3>(int32_t height,
                                     int32_t width,
                                     int32_t inWidthStride0,
                                     const float32_t *inData0,
                                     int32_t inWidthStride1,
                                     const float32_t *inData1,
                                     int32_t outWidthStride,
                                     float32_t *outData)
{
    return mls_f32(height, width, 3, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Mls<float, 4>(int32_t height,
                                     int32_t width,
                                     int32_t inWidthStride0,
                                     const float32_t *inData0,
                                     int32_t inWidthStride1,
                                     const float32_t *inData1,
                                     int32_t outWidthStride,
                                     float32_t *outData)
{
    return mls_f32(height, width, 4, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm
