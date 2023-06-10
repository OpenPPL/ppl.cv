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
#include "ppl/common/sys.h"
#include <string.h>
#include <arm_neon.h>

#include <cmath>
#include <algorithm>

namespace ppl::cv::arm {

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
            base_out[w] = scale * static_cast<float>(base_in[w]) + delta;
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
            base_out[w] = scale * base_in[w] + delta;
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
        for (; w < row_width; ++w) {
            float value = scale * base_in[w] + delta;
            base_out[w] = static_cast<uint8_t>(std::round(std::min(std::max(value, 0.0f), 255.0f)));
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
        for (; w < row_width; ++w) {
            float value = scale * base_in[w] + delta;
            base_out[w] = static_cast<uint8_t>(std::round(std::min(std::max(value, 0.0f), 255.0f)));
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
    return ChangeDataTypeAndScale<float, float>(
        height, width, 4, inWidthStride, inData, outWidthStride, outData, scale, delta);
}

} // namespace ppl::cv::arm
