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

#include "ppl/cv/arm/crop.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <string.h>
#include <arm_neon.h>

namespace ppl::cv::arm {

static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = data < 0 ? 0 : val;
    return val;
}

template <typename T>
static void crop_line_common(const T* src, T* dst, int32_t outWidth, float scale)
{
    int32_t i = 0;
    for (; i < outWidth; i++, src++) {
        dst[i] = scale * src[0];
    }
}

template <>
void crop_line_common(const uint8_t* src, uint8_t* dst, int32_t outWidth, float scale)
{
    if (scale < 1.000001f && scale > 0.9999999f) {
        memcpy(dst, src, outWidth * sizeof(uint8_t));
    } else {
        int32_t i = 0;
        for (; i < outWidth; i++, src++) {
            int32_t val = scale * src[0];
            dst[i] = sat_cast(val);
        }
    }
}

template <typename T, int32_t channels>
ppl::common::RetCode Crop(const int32_t inHeight,
                          const int32_t inWidth,
                          int32_t inWidthStride,
                          const T* inData,
                          int32_t outHeight,
                          int32_t outWidth,
                          int32_t outWidthStride,
                          T* outData,
                          int32_t left,
                          int32_t top,
                          float scale)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if ((left + outWidth) > inWidth || (top + outHeight) > inHeight) { return ppl::common::RC_INVALID_VALUE; }

    const T* src = inData + top * inWidthStride + left * channels;
    T* dst = outData;

    int32_t out_row_width = outWidth * channels;
    for (int32_t i = 0; i < outHeight; i++) {
        crop_line_common(src, dst, out_row_width, scale);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template ppl::common::RetCode Crop<float, 1>(const int32_t inHeight,
                                             const int32_t inWidth,
                                             int32_t inWidthStride,
                                             const float* inData,
                                             int32_t outHeight,
                                             int32_t outWidth,
                                             int32_t outWidthStride,
                                             float* outData,
                                             int32_t left,
                                             int32_t top,
                                             float scale);

template ppl::common::RetCode Crop<float, 3>(const int32_t inHeight,
                                             const int32_t inWidth,
                                             int32_t inWidthStride,
                                             const float* inData,
                                             int32_t outHeight,
                                             int32_t outWidth,
                                             int32_t outWidthStride,
                                             float* outData,
                                             int32_t left,
                                             int32_t top,
                                             float scale);

template ppl::common::RetCode Crop<float, 4>(const int32_t inHeight,
                                             const int32_t inWidth,
                                             int32_t inWidthStride,
                                             const float* inData,
                                             int32_t outHeight,
                                             int32_t outWidth,
                                             int32_t outWidthStride,
                                             float* outData,
                                             int32_t left,
                                             int32_t top,
                                             float scale);

template ppl::common::RetCode Crop<uint8_t, 1>(const int32_t inHeight,
                                               const int32_t inWidth,
                                               int32_t inWidthStride,
                                               const uint8_t* inData,
                                               int32_t outHeight,
                                               int32_t outWidth,
                                               int32_t outWidthStride,
                                               uint8_t* outData,
                                               int32_t left,
                                               int32_t top,
                                               float scale);

template ppl::common::RetCode Crop<uint8_t, 3>(const int32_t inHeight,
                                               const int32_t inWidth,
                                               int32_t inWidthStride,
                                               const uint8_t* inData,
                                               int32_t outHeight,
                                               int32_t outWidth,
                                               int32_t outWidthStride,
                                               uint8_t* outData,
                                               int32_t left,
                                               int32_t top,
                                               float scale);

template ppl::common::RetCode Crop<uint8_t, 4>(const int32_t inHeight,
                                               const int32_t inWidth,
                                               int32_t inWidthStride,
                                               const uint8_t* inData,
                                               int32_t outHeight,
                                               int32_t outWidth,
                                               int32_t outWidthStride,
                                               uint8_t* outData,
                                               int32_t left,
                                               int32_t top,
                                               float scale);

} // namespace ppl::cv::arm
