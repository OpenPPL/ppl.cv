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

#include "ppl/cv/x86/mean.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

template <typename T, int32_t nc>
::ppl::common::RetCode Mean(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    float* outMeanData,
    int32_t inMaskStride,
    uint8_t* inMask)
{
    if (inData == nullptr || outMeanData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    } 

    bool useMask = false;
    if (inMask != NULL)
        useMask = true;

    double sum[4]    = {0, 0, 0, 0};
    int32_t count[4] = {0, 0, 0, 0};
    if (useMask) {
        for (int32_t i = 0; i < height; i++) {
            const T* ptr_in = inData + i * inWidthStride;
            for (int32_t j = 0; j < width; j++) {
                if (inMask[i * inMaskStride + j]) {
                    for (int32_t k = 0; k < nc; k++) {
                        sum[k] += ptr_in[j * nc + k];
                        count[k]++;
                    }
                }
            }
        }
    } else {
        for (int32_t k = 0; k < nc; ++k) {
            count[k] = height * width;
        }
        for (int32_t i = 0; i < height; i++) {
            const T* ptr_in = inData + i * inWidthStride;
            for (int32_t j = 0; j < width; j++) {
                for (int32_t k = 0; k < nc; k++) {
                    sum[k] += ptr_in[j * nc + k];
                }
            }
        }
    }

    for (int32_t i = 0; i < nc; i++) {
        outMeanData[i] = (float)(sum[i] / count[i]);
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Mean<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    float* outMeanData,
    int32_t inMaskStride,
    uint8_t* inMask);
template ::ppl::common::RetCode Mean<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    float* outMeanData,
    int32_t inMaskStride,
    uint8_t* inMask);
template ::ppl::common::RetCode Mean<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    float* outMeanData,
    int32_t inMaskStride,
    uint8_t* inMask);

template ::ppl::common::RetCode Mean<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    float* outMeanData,
    int32_t inMaskStride,
    uint8_t* inMask);
template ::ppl::common::RetCode Mean<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    float* outMeanData,
    int32_t inMaskStride,
    uint8_t* inMask);
template ::ppl::common::RetCode Mean<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    float* outMeanData,
    int32_t inMaskStride,
    uint8_t* inMask);

}
}
} // namespace ppl::cv::x86
