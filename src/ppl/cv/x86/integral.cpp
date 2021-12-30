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

#include "ppl/cv/x86/integral.h"
#include "ppl/cv/types.h"
#include <string.h>

namespace ppl {
namespace cv {
namespace x86 {

template <typename TSrc, typename TDst, int32_t cn>
void IntegralImage(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const TSrc *in,
    int32_t outWidthStride,
    TDst *out)
{
    int32_t outWidth = width + 1;
    memset(out, 0, sizeof(TDst) * outWidth * cn);
    for (int32_t h = 0; h < height; h++) {
        TDst sum[cn] = {0};
        memset(out + (h + 1) * outWidthStride, 0, sizeof(TDst) * cn);
        for (int32_t w = 0; w < width; w++) {
            for (int32_t c = 0; c < cn; c++) {
                TSrc in_v = in[h * inWidthStride + w * cn + c];
                sum[c] += in_v;
                out[(h + 1) * outWidthStride + (w + 1) * cn + c] = sum[c];
            }
        }
    }

    for (int32_t w = 0; w < width * cn; w++) {
        TDst sum = (TDst)0;
        for (int32_t h = 0; h < height; h++) {
            TDst in_v = out[(h + 1) * outWidthStride + cn + w];
            sum += in_v;
            out[(h + 1) * outWidthStride + cn + w] = sum;
        }
    }
}

template <typename TSrc, typename TDst, int32_t cn>
void IntegralImageDeprecate(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const TSrc *in,
    int32_t outWidthStride,
    TDst *out)
{
    //integral by row
    for (int32_t h = 0; h < height; h++) {
        TDst sum[cn] = {0};
        for (int32_t w = 0; w < width; w++) {
            for (int32_t c = 0; c < cn; c++) {
                TSrc in_v = in[h * inWidthStride + w * cn + c];
                sum[c] += in_v;
                out[h * outWidthStride + w * cn + c] = sum[c];
            }
        }
    }

    for (int32_t w = 0; w < width * cn; w++) {
        TDst sum = (TDst)0;
        for (int32_t h = 0; h < height; h++) {
            TDst in_v = out[h * outWidthStride + w];
            sum += in_v;
            out[h * outWidthStride + w] = sum;
        }
    }
}

template <>
::ppl::common::RetCode Integral<float, float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<float, float, 1>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<float, float, 1>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<float, float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<float, float, 3>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<float, float, 3>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<float, float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<float, float, 4>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<float, float, 4>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<uint8_t, int32_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    int32_t *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<uint8_t, int32_t, 1>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<uint8_t, int32_t, 1>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<uint8_t, int32_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    int32_t *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<uint8_t, int32_t, 3>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<uint8_t, int32_t, 3>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Integral<uint8_t, int32_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    int32_t *outData)
{
    if ((outHeight == inHeight) && (outWidth == inWidth)) {
        IntegralImageDeprecate<uint8_t, int32_t, 4>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else if ((outHeight == (inHeight + 1)) && (outWidth == (inWidth + 1))) {
        IntegralImage<uint8_t, int32_t, 4>(inHeight, inWidth, inWidthStride, inData, outWidthStride, outData);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
