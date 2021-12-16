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

#include "ppl/cv/x86/merge.h"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"

namespace ppl {
namespace cv {
namespace x86 {

template <typename T>
void mergeSOA2AOS(
    int32_t numChannels,
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T** in,
    int32_t outWidthStride,
    T* out)
{
    for (int32_t h = 0; h < height; h++) {
        for (int32_t w = 0; w < width; w++) {
            int32_t inIndex  = h * inWidthStride + w;
            int32_t outIndex = h * outWidthStride + w * numChannels;
            for (int32_t c = 0; c < numChannels; c++) {
                out[outIndex + c] = in[c][inIndex];
            }
        }
    }
}

template <typename T, int32_t nc>
void Merge(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T** inData,
    int32_t outWidthStride,
    T* outData)
{
    mergeSOA2AOS<float>(nc, height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
void Merge<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float** inData,
    int32_t outWidthStride,
    float* outData)
{
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        fma::mergeSOA2AOS<float, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        mergeSOA2AOS<float>(4, height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
void Merge<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float** inData,
    int32_t outWidthStride,
    float* outData)
{
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        fma::mergeSOA2AOS<float, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        mergeSOA2AOS<float>(3, height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
void Merge<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t** inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        fma::mergeSOA2AOS<uint8_t, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        mergeSOA2AOS<uint8_t>(4, height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
void Merge<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t** inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        fma::mergeSOA2AOS<uint8_t, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        mergeSOA2AOS<uint8_t>(3, height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <typename T>
::ppl::common::RetCode Merge3Channels(
    int32_t height,
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
    const T* inData[3] = {inDataC0, inDataC1, inDataC2};
    Merge<T, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <typename T>
::ppl::common::RetCode Merge4Channels(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inDataC0,
    const T* inDataC1,
    const T* inDataC2,
    const T* inDataC3,
    int32_t outWidthStride,
    T* outData)
{
    if (nullptr == inDataC0 || nullptr == inDataC1 || nullptr == inDataC2 || nullptr == inDataC3 || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const T* inData[4] = {inDataC0, inDataC1, inDataC2, inDataC3};
    Merge<T, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Merge3Channels<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inDataC0,
    const uint8_t* inDataC1,
    const uint8_t* inDataC2,
    int32_t outWidthStride,
    uint8_t* outData);
template ::ppl::common::RetCode Merge3Channels<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inDataC0,
    const float* inDataC1,
    const float* inDataC2,
    int32_t outWidthStride,
    float* outData);
template ::ppl::common::RetCode Merge4Channels<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inDataC0,
    const uint8_t* inDataC1,
    const uint8_t* inDataC2,
    const uint8_t* inDataC3,
    int32_t outWidthStride,
    uint8_t* outData);
template ::ppl::common::RetCode Merge4Channels<float>(
    int32_t height,
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
} // namespace ppl::cv::x86
