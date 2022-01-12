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

#include "ppl/cv/x86/perspectivetransform.h"
#include "ppl/cv/types.h"
#include <cmath>
#include <algorithm>

#define EPS (1e-8)

namespace ppl {
namespace cv {
namespace x86 {

template <typename T, int32_t scn, int32_t dcn>
::ppl::common::RetCode PerspectiveTransform(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData,
    const float* affineMatrix)
{
    if (inData == nullptr || outData == nullptr || affineMatrix == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int32_t affineMatrixWidth = scn + 1;
    const double eps          = EPS;
    auto reduce_function      = [](const T* affineMatrix, const T* data) {
        T sum = 0;
        for (int32_t k = 0; k < scn; ++k) {
            sum += affineMatrix[k] * data[k];
        }
        sum += affineMatrix[scn];
        return sum;
    };
    for (int32_t i = 0; i < height; ++i) {
        T* base_out      = outData + i * outWidthStride;
        const T* base_in = inData + i * inWidthStride;
        for (int32_t j = 0; j < width; ++j) {
            const T* cur_in = base_in + j * dcn;
            T* cur_out      = base_out + j * dcn;
            T w             = reduce_function(affineMatrix + dcn * affineMatrixWidth, cur_in);
            w               = (std::abs(w) > eps) ? 1.0 / w : 0;
            for (int32_t m = 0; m < dcn; ++m) {
                T sum      = reduce_function(affineMatrix + m * affineMatrixWidth, cur_in);
                cur_out[m] = sum * w;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode PerspectiveTransform<float, 3, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    const float* affineMatrix);
template ::ppl::common::RetCode PerspectiveTransform<float, 2, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    const float* affineMatrix);
template ::ppl::common::RetCode PerspectiveTransform<float, 3, 2>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    const float* affineMatrix);
template ::ppl::common::RetCode PerspectiveTransform<float, 2, 2>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    const float* affineMatrix);

}
}
} // namespace ppl::cv::x86
