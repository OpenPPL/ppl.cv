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

#include "ppl/cv/x86/crop.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>

namespace ppl {
namespace cv {
namespace x86 {

using namespace ppl::common;

static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = data < 0 ? 0 : val;
    return val;
}
template <typename _TpSrc, typename _TpDst>
struct imageCrop {
    imageCrop(float _scale)
        : scale(_scale) {}
    void operator()(const _TpSrc* src, _TpDst* dst, int32_t n) const
    {
        for (int32_t i = 0; i < n; i++) {
            dst[i] = (_TpDst)(scale * src[i]);
        }
    }
    float scale;
};

template <>
struct imageCrop<uint8_t, uint8_t> {
    imageCrop(float _scale)
        : scale(_scale)
    {
    }

    void operator()(const uint8_t* src, uint8_t* dst, int32_t n) const
    {
        if (scale < 1.000001f && scale > 0.9999999f) {
            memcpy(dst, src, n * sizeof(uint8_t));
        } else {
            int32_t i = 0;
            for (; i < n; i++, src++) {
                int32_t val = scale * src[0];
                dst[i]      = sat_cast(val);
            }
        }
    }

    float scale;
};

template <>
ppl::common::RetCode Crop<float, 1>(
    const int32_t inHeight,
    const int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    int32_t left,
    int32_t top,
    float ratio)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        x86ImageCrop_avx<float, 1, float, 1, 1>(top, left, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, ratio);
        return ppl::common::RC_SUCCESS;
    }
    const float* src = inData + top * inWidthStride + left;
    float* dst       = outData;

    imageCrop<float, float> s = imageCrop<float, float>(ratio);
    for (int32_t i = 0; i < outHeight; i++) {
        s.operator()(src, dst, outWidth);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
ppl::common::RetCode Crop<float, 3>(
    const int32_t inHeight,
    const int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    int32_t left,
    int32_t top,
    float ratio)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        x86ImageCrop_avx<float, 3, float, 3, 3>(top, left, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, ratio);
        return ppl::common::RC_SUCCESS;
    }
    const float* src = inData + top * inWidthStride + left * 3;
    float* dst       = outData;

    imageCrop<float, float> s = imageCrop<float, float>(ratio);
    for (int32_t i = 0; i < outHeight; ++i) {
        s.operator()(src, dst, outWidth * 3);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
ppl::common::RetCode Crop<float, 4>(
    const int32_t inHeight,
    const int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    int32_t left,
    int32_t top,
    float ratio)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        x86ImageCrop_avx<float, 4, float, 4, 4>(top, left, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, ratio);
        return ppl::common::RC_SUCCESS;
    }
    const float* src = inData + top * inWidthStride + left * 4;
    float* dst       = outData;

    imageCrop<float, float> s = imageCrop<float, float>(ratio);
    for (int32_t i = 0; i < outHeight; i++) {
        s.operator()(src, dst, outWidth * 4);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
ppl::common::RetCode Crop<uint8_t, 1>(
    const int32_t inHeight,
    const int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t left,
    int32_t top,
    float ratio)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const uint8_t* src = inData + top * inWidthStride + left;
    uint8_t* dst       = outData;

    imageCrop<uint8_t, uint8_t> s = imageCrop<uint8_t, uint8_t>(ratio);
    for (int32_t i = 0; i < outHeight; i++) {
        s.operator()(src, dst, outWidth);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
ppl::common::RetCode Crop<uint8_t, 2>(
    const int32_t inHeight,
    const int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t left,
    int32_t top,
    float ratio)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const uint8_t* src = inData + top * inWidthStride + left * 2;
    uint8_t* dst       = outData;

    imageCrop<uint8_t, uint8_t> s = imageCrop<uint8_t, uint8_t>(ratio);
    for (int32_t i = 0; i < outHeight; i++) {
        s.operator()(src, dst, outWidth * 2);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
ppl::common::RetCode Crop<uint8_t, 3>(
    const int32_t inHeight,
    const int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t left,
    int32_t top,
    float ratio)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const uint8_t* src = inData + top * inWidthStride + left * 3;
    uint8_t* dst       = outData;

    imageCrop<uint8_t, uint8_t> s = imageCrop<uint8_t, uint8_t>(ratio);
    for (int32_t i = 0; i < outHeight; i++) {
        s.operator()(src, dst, outWidth * 3);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
ppl::common::RetCode Crop<uint8_t, 4>(
    const int32_t inHeight,
    const int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t left,
    int32_t top,
    float ratio)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const uint8_t* src = inData + top * inWidthStride + left * 4;
    uint8_t* dst       = outData;

    imageCrop<uint8_t, uint8_t> s = imageCrop<uint8_t, uint8_t>(ratio);
    for (int32_t i = 0; i < outHeight; i++) {
        s.operator()(src, dst, outWidth * 4);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
