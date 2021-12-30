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
#include <string.h>
#include <limits>
#include <float.h>
#include "ppl/cv/x86/bitwise.h"

namespace ppl {
namespace cv {
namespace x86 {

template <typename T, int32_t c>
::ppl::common::RetCode BitwiseAnd(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T* inData0,
    int32_t inWidthStride1,
    const T* inData1,
    int32_t outWidthStride,
    T* outData,
    int32_t inMaskWidthStride,
    const uint8_t* inMask)
{
    if (nullptr == inData0 && nullptr == inData1 && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride0 == 0 || inWidthStride1 == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inMask == NULL) {
        for (int32_t i = 0; i < height; i++) {
            const T* ptr_s1 = inData0 + i * inWidthStride0;
            const T* ptr_s2 = inData1 + i * inWidthStride1;
            T* ptr_d        = outData + i * outWidthStride;
            for (int32_t j = 0; j < width * c; j++) {
                ptr_d[j] = ptr_s1[j] & ptr_s2[j];
            }
        }
    } else {
        for (int32_t i = 0; i < height; i++) {
            const T* ptr_s1      = inData0 + i * inWidthStride0;
            const T* ptr_s2      = inData1 + i * inWidthStride1;
            T* ptr_d             = outData + i * outWidthStride;
            const uint8_t* ptr_m = inMask + i * inMaskWidthStride;
            for (int32_t j = 0; j < width; j++) {
                if (ptr_m[j]) {
                    for (int32_t k = 0; k < c; k++) {
                        ptr_d[j * c + k] = ptr_s1[j * c + k] & ptr_s2[j * c + k];
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode BitwiseAnd<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t* inData0,
    int32_t inWidthStride1,
    const uint8_t* inData1,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t inMaskWidthStride,
    const uint8_t* inMask);

template ::ppl::common::RetCode BitwiseAnd<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t* inData0,
    int32_t inWidthStride1,
    const uint8_t* inData1,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t inMaskWidthStride,
    const uint8_t* inMask);

template ::ppl::common::RetCode BitwiseAnd<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t* inData0,
    int32_t inWidthStride1,
    const uint8_t* inData1,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t inMaskWidthStride,
    const uint8_t* inMask);

}
}
} // namespace ppl::cv::x86
