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

#include "ppl/cv/x86/copymakeborder.h"
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"
#include <vector>
#include <cstring>

namespace ppl {
namespace cv {
namespace x86 {

template <BorderType borderType>
inline int32_t BorderInterpolate(int32_t p, int32_t len)
{
    if (borderType == ppl::cv::BORDER_REFLECT_101) {
        p = p < 0 ? (-p) : 2 * len - p - 2;
    } else if (borderType == ppl::cv::BORDER_REFLECT) {
        p = p < 0 ? (-p - 1) : 2 * len - p - 1;
    } else if (borderType == ppl::cv::BORDER_REPLICATE) {
        p = (p < 0) ? 0 : len - 1;
    } else if (borderType == ppl::cv::BORDER_CONSTANT) {
        p = -1;
    }
    return p;
}

template <typename T, int32_t cn, BorderType border_type>
::ppl::common::RetCode CopyMakeNonConstBorder(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const T *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    T *dst)
{
    int32_t i, j, k, elemSize = sizeof(T);
    int32_t left   = (dstWidth - srcWidth) / 2;
    int32_t right  = (dstWidth - srcWidth) / 2;
    int32_t top    = (dstHeight - srcHeight) / 2;
    int32_t bottom = (dstHeight - srcHeight) / 2;
    std::vector<int32_t> tab((dstWidth - srcWidth) * cn);
    for (i = 0; i < left; i++) {
        j = BorderInterpolate<border_type>(i - left, srcWidth) * cn;
        for (k = 0; k < cn; k++)
            tab[i * cn + k] = j + k;
    }

    for (i = 0; i < right; i++) {
        j = BorderInterpolate<border_type>(srcWidth + i, srcWidth) * cn;
        for (k = 0; k < cn; k++)
            tab[(i + left) * cn + k] = j + k;
    }

    srcWidth *= cn;
    dstWidth *= cn;
    left *= cn;
    right *= cn;

    T *dstInner = dst + dstWidthStride * top + left;

    for (i = 0; i < srcHeight; i++, dstInner += dstWidthStride, src += srcWidthStride) {
        if (dstInner != src)
            memcpy(dstInner, src, srcWidth * elemSize);
        for (j = 0; j < left; j++)
            dstInner[j - left] = src[tab[j]];
        for (j = 0; j < right; j++)
            dstInner[j + srcWidth] = src[tab[j + left]];
    }

    dstWidth *= elemSize;
    dst += dstWidthStride * top;

    for (i = 0; i < top; i++) {
        j = BorderInterpolate<border_type>(i - top, srcHeight);
        memcpy(dst + (i - top) * dstWidthStride, dst + j * dstWidthStride, dstWidth);
    }

    for (i = 0; i < bottom; i++) {
        j = BorderInterpolate<border_type>(i + srcHeight, srcHeight);
        memcpy(dst + (i + srcHeight) * dstWidthStride, dst + j * dstWidthStride, dstWidth);
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t cn>
::ppl::common::RetCode CopyMakeConstBorder(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const T *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    T *dst,
    T border_value)
{
    int32_t left = (dstWidth - srcWidth) / 2;
    // int32_t right = (dstWidth - srcWidth) / 2;
    int32_t top  = (dstHeight - srcHeight) / 2;
    // int32_t bottom = (dstHeight - srcHeight) / 2;
    for (int32_t i = 0; i < dstHeight; ++i) {
        T *cur_dst = dst + i * dstWidthStride;
        if (i < top || i >= (top + srcHeight)) {
            for (int32_t j = 0; j < dstWidth * cn; ++j) {
                cur_dst[j] = border_value;
            }
        } else {
            // left padding
            for (int32_t j = 0; j < left * cn; ++j) {
                cur_dst[j] = border_value;
            }

            // memcpy
            const T *cur_src = src + (i - top) * srcWidthStride;
            memcpy(cur_dst + left * cn, cur_src, sizeof(T) * srcWidth * cn);

            // right padding
            for (int32_t j = (left + srcWidth) * cn; j < dstWidth * cn; ++j) {
                cur_dst[j] = border_value;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t cn>
::ppl::common::RetCode CopyMakeBorder(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const T *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    T *dst,
    BorderType border_type,
    T border_value)
{
    if (border_type != ppl::cv::BORDER_REFLECT_101 &&
        border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_CONSTANT &&
        border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        CopyMakeConstBorder<T, cn>(srcHeight, srcWidth, srcWidthStride, src, dstHeight, dstWidth, dstWidthStride, dst, border_value);
    } else {
        if (border_type == ppl::cv::BORDER_REFLECT) {
            CopyMakeNonConstBorder<T, cn, ppl::cv::BORDER_REFLECT>(srcHeight, srcWidth, srcWidthStride, src, dstHeight, dstWidth, dstWidthStride, dst);
        } else if (border_type == ppl::cv::BORDER_REFLECT_101) {
            CopyMakeNonConstBorder<T, cn, ppl::cv::BORDER_REFLECT101>(srcHeight, srcWidth, srcWidthStride, src, dstHeight, dstWidth, dstWidthStride, dst);
        } else if (border_type == ppl::cv::BORDER_REPLICATE) {
            CopyMakeNonConstBorder<T, cn, ppl::cv::BORDER_REPLICATE>(srcHeight, srcWidth, srcWidthStride, src, dstHeight, dstWidth, dstWidthStride, dst);
        }
    }
    return ppl::common::RC_SUCCESS;
}
template ::ppl::common::RetCode CopyMakeBorder<uint8_t, 1>(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const uint8_t *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    uint8_t *dst,
    BorderType border_type,
    uint8_t border_value);
template ::ppl::common::RetCode CopyMakeBorder<uint8_t, 3>(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const uint8_t *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    uint8_t *dst,
    BorderType border_type,
    uint8_t border_value);
template ::ppl::common::RetCode CopyMakeBorder<uint8_t, 4>(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const uint8_t *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    uint8_t *dst,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode CopyMakeBorder<float, 1>(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const float *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    float *dst,
    BorderType border_type,
    float border_value);
template ::ppl::common::RetCode CopyMakeBorder<float, 3>(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const float *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    float *dst,
    BorderType border_type,
    float border_value);
template ::ppl::common::RetCode CopyMakeBorder<float, 4>(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    const float *src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstWidthStride,
    float *dst,
    BorderType border_type,
    float border_value);

}
}
} // namespace ppl::cv::x86
