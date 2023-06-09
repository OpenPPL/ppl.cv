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

#include "ppl/cv/arm/filter2d.h"
#include "ppl/cv/arm/copymakeborder.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <limits.h>
#include <algorithm>
#include <vector>

namespace ppl::cv::arm {
static int32_t senseRound_f(float value)
{
    return static_cast<int32_t>(roundf(value));
}

static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = val < 0 ? 0 : val;
    return (uint8_t)val;
}

void convolution_b(int32_t imageInSizeX,
                   int32_t imageInSizeY,
                   int32_t inWidthStride,
                   const uint8_t *imageIn,
                   int32_t filterSize,
                   const float *filter,
                   float delta,
                   int32_t outWidthStride,
                   uint8_t *imageOut,
                   int32_t cn)
{
    int32_t imageOutSizeX = imageInSizeX - filterSize + 1;
    int32_t imageOutSizeY = imageInSizeY - filterSize + 1;

    int32_t y;
    for (y = 0; y < imageOutSizeY; y++) {
        int32_t x = 0;
        for (; x < imageOutSizeX * cn; x++) {
            float sum = 0;
            for (int32_t fx = 0; fx < filterSize; fx++) {
                for (int32_t fy = 0; fy < filterSize; fy++) {
                    float f = filter[fx + fy * filterSize];
                    sum += f * imageIn[(fy + y) * inWidthStride + x + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = sat_cast(senseRound_f(sum + delta));
        }
    }
}

template <int32_t BX, int32_t BY>
void convolutionSerialBlocking_f(int32_t imageInSizeX,
                                 int32_t imageInSizeY,
                                 int32_t inWidthStride,
                                 const float *imageIn,
                                 int32_t filterSize,
                                 const float *filter,
                                 float delta,
                                 int32_t outWidthStride,
                                 float *imageOut,
                                 int32_t cn)
{
    int32_t imageOutSizeX = imageInSizeX - filterSize + 1;
    int32_t imageOutSizeY = imageInSizeY - filterSize + 1;

    int32_t y = 0;
    for (; y < imageOutSizeY; y++) {
        int32_t x = 0;
        for (; x <= imageOutSizeX * cn - BX; x += BX) {
            float sum[BX] = {(float)0};
            for (int32_t fy = 0; fy < filterSize; fy++) {
                for (int32_t fx = 0; fx < filterSize; fx++) {
                    float filterItem = filter[fx + fy * filterSize];
                    for (int32_t j = 0; j < BX; j++) {
                        float imageItem = imageIn[x + j + fx * cn + (fy + y) * inWidthStride];
                        sum[j] += filterItem * imageItem;
                    }
                }
            }
            for (int32_t j = 0; j < BX; j++) {
                imageOut[x + j + (y)*outWidthStride] = sum[j] + delta;
            }
        }
        for (; x < imageOutSizeX * cn; x++) {
            float sum = 0;
            for (int32_t fy = 0; fy < filterSize; fy++) {
                for (int32_t fx = 0; fx < filterSize; fx++) {
                    float filterItem = filter[fx + fy * filterSize];
                    {
                        float imageItem = imageIn[x + fx * cn + (fy + y) * inWidthStride];
                        sum += filterItem * imageItem;
                    }
                }
            }
            imageOut[x + (y)*outWidthStride] = sum + delta;
        }
    }
}

template <>
::ppl::common::RetCode Filter2D<float, 1>(int32_t height,
                                          int32_t width,
                                          int32_t inWidthStride,
                                          const float *inData,
                                          int32_t kernel_len,
                                          const float *filter,
                                          int32_t outWidthStride,
                                          float *outData,
                                          float delta,
                                          BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_CONSTANT && border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;
    int32_t cn = 1;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    float *bsrc = (float *)malloc(bsrcHeight * bsrcWidth * cn * sizeof(float));

    CopyMakeBorder<float, 1>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    convolutionSerialBlocking_f<3, 3>(
        bsrcWidth, bsrcHeight, bsrcWidthStep, bsrc, kernel_len, filter, delta, outWidthStride, outData, cn);

    free(bsrc);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode Filter2D<float, 3>(int32_t height,
                                          int32_t width,
                                          int32_t inWidthStride,
                                          const float *inData,
                                          int32_t kernel_len,
                                          const float *filter,
                                          int32_t outWidthStride,
                                          float *outData,
                                          float delta,
                                          BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_CONSTANT && border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;
    int32_t cn = 3;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    float *bsrc = (float *)malloc(bsrcHeight * bsrcWidth * cn * sizeof(float));

    CopyMakeBorder<float, 3>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    convolutionSerialBlocking_f<3, 3>(
        bsrcWidth, bsrcHeight, bsrcWidthStep, bsrc, kernel_len, filter, delta, outWidthStride, outData, cn);

    free(bsrc);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<float, 4>(int32_t height,
                                          int32_t width,
                                          int32_t inWidthStride,
                                          const float *inData,
                                          int32_t kernel_len,
                                          const float *filter,
                                          int32_t outWidthStride,
                                          float *outData,
                                          float delta,
                                          BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_CONSTANT && border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;
    int32_t cn = 4;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    float *bsrc = (float *)malloc(bsrcHeight * bsrcWidth * cn * sizeof(float));

    CopyMakeBorder<float, 4>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    convolutionSerialBlocking_f<3, 3>(
        bsrcWidth, bsrcHeight, bsrcWidthStep, bsrc, kernel_len, filter, delta, outWidthStride, outData, cn);

    free(bsrc);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<uint8_t, 1>(int32_t height,
                                            int32_t width,
                                            int32_t inWidthStride,
                                            const uint8_t *inData,
                                            int32_t kernel_len,
                                            const float *filter,
                                            int32_t outWidthStride,
                                            uint8_t *outData,
                                            float delta,
                                            BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_CONSTANT && border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;
    int32_t cn = 1;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    uint8_t *bsrc = (uint8_t *)malloc(bsrcHeight * bsrcWidth * cn * sizeof(uint8_t));

    CopyMakeBorder<uint8_t, 1>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    convolution_b(bsrcWidth, bsrcHeight, bsrcWidthStep, bsrc, kernel_len, filter, delta, outWidthStride, outData, cn);

    free(bsrc);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<uint8_t, 3>(int32_t height,
                                            int32_t width,
                                            int32_t inWidthStride,
                                            const uint8_t *inData,
                                            int32_t kernel_len,
                                            const float *filter,
                                            int32_t outWidthStride,
                                            uint8_t *outData,
                                            float delta,
                                            BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_CONSTANT && border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;
    int32_t cn = 3;
    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    uint8_t *bsrc = (uint8_t *)malloc(bsrcHeight * bsrcWidth * cn * sizeof(uint8_t));

    CopyMakeBorder<uint8_t, 3>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    convolution_b(bsrcWidth, bsrcHeight, bsrcWidthStep, bsrc, kernel_len, filter, delta, outWidthStride, outData, cn);

    free(bsrc);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<uint8_t, 4>(int32_t height,
                                            int32_t width,
                                            int32_t inWidthStride,
                                            const uint8_t *inData,
                                            int32_t kernel_len,
                                            const float *filter,
                                            int32_t outWidthStride,
                                            uint8_t *outData,
                                            float delta,
                                            BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_CONSTANT && border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;
    int32_t cn = 4;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    uint8_t *bsrc = (uint8_t *)malloc(bsrcHeight * bsrcWidth * cn * sizeof(uint8_t));

    CopyMakeBorder<uint8_t, 4>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    convolution_b(bsrcWidth, bsrcHeight, bsrcWidthStep, bsrc, kernel_len, filter, delta, outWidthStride, outData, cn);

    free(bsrc);
    return ppl::common::RC_SUCCESS;
}

} // namespace ppl::cv::arm
