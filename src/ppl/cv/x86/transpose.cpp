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

#include "ppl/cv/x86/transpose.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "intrinutils.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>
#include <immintrin.h>
#include <iostream>

namespace ppl {
namespace cv {
namespace x86 {

template <typename T>
::ppl::common::RetCode transpose(
    const T *src,
    int channels,
    int height,
    int width,
    int inWidthStride,
    int outWidthStride,
    T *dst)
{
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                dst[j * outWidthStride + i * channels + c] =
                    src[i * inWidthStride + j * channels + c];
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Transpose<float, 1>(int height, int width, int inWidthStride, const float *inData, int outWidthStride, float *outData)
{
    return transpose<float>(inData, 1, height, width, inWidthStride, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Transpose<float, 3>(int height, int width, int inWidthStride, const float *inData, int outWidthStride, float *outData)
{
    return transpose<float>(inData, 3, height, width, inWidthStride, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Transpose<float, 4>(int height, int width, int inWidthStride, const float *inData, int outWidthStride, float *outData)
{
    return transpose<float>(inData, 4, height, width, inWidthStride, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Transpose<uint8_t, 1>(int height, int width, int inWidthStride, const uint8_t *inData, int outWidthStride, uint8_t *outData)
{
    return transpose<uint8_t>(inData, 1, height, width, inWidthStride, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Transpose<uint8_t, 3>(int height, int width, int inWidthStride, const uint8_t *inData, int outWidthStride, uint8_t *outData)
{
    return transpose<uint8_t>(inData, 3, height, width, inWidthStride, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Transpose<uint8_t, 4>(int height, int width, int inWidthStride, const uint8_t *inData, int outWidthStride, uint8_t *outData)
{
    return transpose<uint8_t>(inData, 4, height, width, inWidthStride, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::x86
