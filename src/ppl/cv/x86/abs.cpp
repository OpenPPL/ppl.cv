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

#include "ppl/cv/types.h"
#include "ppl/cv/x86/abs.h"
#include <immintrin.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

template <typename T>
void abs(T *dst, const T *src, int32_t n)
{
    for (int32_t i = 0; i < n; ++i) {
        if (src[i] == -128) {
            dst[i] = 127;
        } else {
            dst[i] = src[i] < 0 ? -(src[i]) : src[i];
        }
    }
}

template <>
void abs(float *dst, const float *src, int32_t n)
{
    for (int32_t i = 0; i < n; ++i) {
        dst[i] = src[i] < 0 ? -(src[i]) : src[i];
    }
}

template <typename T, int32_t nc>
::ppl::common::RetCode Abs(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T *inData,
    int32_t outWidthStride,
    T *outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    for (int32_t h = 0; h < height; h++) {
        abs<T>(outData + h * outWidthStride, inData + h * inWidthStride, nc * width);
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Abs<float, 1>(int32_t height, int32_t width, int32_t inWidthStride, const float *inData, int32_t outWidthStride, float *outData);
template ::ppl::common::RetCode Abs<float, 3>(int32_t height, int32_t width, int32_t inWidthStride, const float *inData, int32_t outWidthStride, float *outData);
template ::ppl::common::RetCode Abs<float, 4>(int32_t height, int32_t width, int32_t inWidthStride, const float *inData, int32_t outWidthStride, float *outData);

template ::ppl::common::RetCode Abs<int8_t, 1>(int32_t height, int32_t width, int32_t inWidthStride, const int8_t *inData, int32_t outWidthStride, int8_t *outData);
template ::ppl::common::RetCode Abs<int8_t, 3>(int32_t height, int32_t width, int32_t inWidthStride, const int8_t *inData, int32_t outWidthStride, int8_t *outData);
template ::ppl::common::RetCode Abs<int8_t, 4>(int32_t height, int32_t width, int32_t inWidthStride, const int8_t *inData, int32_t outWidthStride, int8_t *outData);

}
}
} // namespace ppl::cv::x86
