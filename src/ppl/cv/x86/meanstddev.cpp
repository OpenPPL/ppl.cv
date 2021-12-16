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

#include "ppl/cv/x86/meanstddev.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

template <typename T, int32_t channels>
::ppl::common::RetCode MeanStdDev(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const T* inData,
    float* mean,
    float* stddev,
    int32_t maskStride,
    const uint8_t* mask)
{
    if (inData == nullptr || mean == nullptr || stddev == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || srcStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    } 

    float sum[channels]   = {0};
    float sqsum[channels] = {0};

    int32_t count = 0;
    if (nullptr == mask) {
        count = height * width;
        for (int32_t y = 0; y < height; ++y) {
            const T* base_in = inData + y * srcStride;
            for (int32_t x = 0; x < width; ++x) {
                for (int32_t c = 0; c < channels; c++) {
                    float v = (float)(base_in[channels * x + c]);
                    sum[c] += v;
                    sqsum[c] += v * v;
                }
            }
        }
    } else {
        for (int32_t y = 0; y < height; ++y) {
            const T* base_in         = inData + y * srcStride;
            const uint8_t* base_mask = mask + y * maskStride;
            for (int32_t x = 0; x < width; ++x) {
                if (base_mask[x]) {
                    for (int32_t c = 0; c < channels; c++) {
                        float v = (float)(base_in[channels * x + c]);
                        sum[c] += v;
                        sqsum[c] += v * v;
                    }
                    count++;
                }
            }
        }
    }

    float scale = 1. / count;
    for (int32_t i = 0; i < channels; i++) {
        mean[i]        = (float)sum[i] * scale;
        float variance = std::max(sqsum[i] * scale - (mean[i]) * (mean[i]), 0.0f);
        stddev[i]      = std::sqrt(variance);
    }
    return 0;
}

template ::ppl::common::RetCode MeanStdDev<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const uint8_t* inData,
    float* mean,
    float* stddev,
    int32_t maskStride,
    const uint8_t* mask);
template ::ppl::common::RetCode MeanStdDev<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const uint8_t* inData,
    float* mean,
    float* stddev,
    int32_t maskStride,
    const uint8_t* mask);
template ::ppl::common::RetCode MeanStdDev<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const uint8_t* inData,
    float* mean,
    float* stddev,
    int32_t maskStride,
    const uint8_t* mask);

template ::ppl::common::RetCode MeanStdDev<float, 1>(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const float* inData,
    float* mean,
    float* stddev,
    int32_t maskStride,
    const uint8_t* mask);
template ::ppl::common::RetCode MeanStdDev<float, 3>(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const float* inData,
    float* mean,
    float* stddev,
    int32_t maskStride,
    const uint8_t* mask);
template ::ppl::common::RetCode MeanStdDev<float, 4>(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const float* inData,
    float* mean,
    float* stddev,
    int32_t maskStride,
    const uint8_t* mask);

}
}
} // namespace ppl::cv::x86
