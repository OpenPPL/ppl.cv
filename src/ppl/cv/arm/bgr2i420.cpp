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
#include <arm_neon.h>
#include "ppl/cv/types.h"
#include "ppl/cv/arm/cvtcolor.h"
#include "color_yuv_simd.hpp"
#include <limits.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace arm {

template <int32_t dcn, int32_t bIdx, int32_t uIdx>
::ppl::common::RetCode YUV420ptoRGB(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *y = inData;
    const uint8_t *u = y + inWidthStride * height;
    const uint8_t *v = y + inWidthStride * (height + height / 4) + (width / 2) * ((height % 4) / 2);
    int32_t ustepIdx = 0;
    int32_t vstepIdx = height % 4 == 2 ? 1 : 0;

    if (uIdx == 1) { std::swap(u, v); }
    if (dcn == 3) {
        YUV4202RGB_u8_neon s = YUV4202RGB_u8_neon(bIdx);
        s.convert_from_yuv420_continuous_layout(height, width, y, u, v, outData, inWidthStride, ustepIdx, vstepIdx, outWidthStride);
    } else if (dcn == 4) {
        YUV4202RGBA_u8 s = YUV4202RGBA_u8(bIdx);
        s.convert_from_yuv420_continuous_layout(height, width, y, u, v, outData, inWidthStride, ustepIdx, vstepIdx, outWidthStride);
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t dcn, int32_t bIdx>
::ppl::common::RetCode YUV420ptoRGB(
    int32_t height,
    int32_t width,
    int32_t ystride,
    int32_t ustride,
    int32_t vstride,
    const uint8_t *y,
    const uint8_t *u,
    const uint8_t *v,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (dcn == 3) {
        YUV4202RGB_u8_neon s = YUV4202RGB_u8_neon(bIdx);
        s.convert_from_yuv420_seperate_layout(height, width, y, u, v, outData, ystride, ustride, vstride, outWidthStride);
    } else if (dcn == 4) {
        YUV4202RGBA_u8 s = YUV4202RGBA_u8(bIdx);
        s.convert_from_yuv420_seperate_layout(height, width, y, u, v, outData, ystride, ustride, vstride, outWidthStride);
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t scn, int32_t bIdx>
::ppl::common::RetCode RGBtoYUV420p(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    RGBtoYUV420p_u8_neon s = RGBtoYUV420p_u8_neon(bIdx);
    s.operator()(height, width, scn, inData, outData, outData + outWidthStride * height, outData + (height + (height / 2) / 2) * outWidthStride + ((height / 2) % 2) * (width / 2), inWidthStride, outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <int32_t scn, int32_t bIdx>
::ppl::common::RetCode RGBtoYUV420p(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t ystride,
    int32_t ustride,
    int32_t vstride,
    uint8_t *y,
    uint8_t *u,
    uint8_t *v)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    RGBtoYUV420p_u8_neon s = RGBtoYUV420p_u8_neon(bIdx);
    s.operator()(height, width, scn, inData, y, u, v, inWidthStride, ystride, ustride, vstride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode I4202BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *uptr = inData + inWidthStride * height;
    const uint8_t *vptr = inData + inWidthStride * height + inWidthStride * height / 4;

    yuv420_to_bgr_uchar_video_range<YUV_I420, 3, 0>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<3, 0, 0>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode I4202BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t ystride,
    const uint8_t *iny,
    int32_t ustride,
    const uint8_t *inu,
    int32_t vstride,
    const uint8_t *inv,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!iny || !inu || !inv || !outData || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    yuv420_to_bgr_uchar_video_range<YUV_I420, 3, 0>(
        height,
        width,
        ystride,
        iny,
        ustride,
        inu,
        vstride,
        inv,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<3, 0>(height, width, ystride, ustride, vstride, iny, inu, inv, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode I4202BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *uptr = inData + inWidthStride * height;
    const uint8_t *vptr = inData + inWidthStride * height + inWidthStride * height / 4;

    yuv420_to_bgr_uchar_video_range<YUV_I420, 4, 0>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<4, 0, 0>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode I4202BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t ystride,
    const uint8_t *iny,
    int32_t ustride,
    const uint8_t *inu,
    int32_t vstride,
    const uint8_t *inv,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!iny || !inu || !inv || !outData || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    yuv420_to_bgr_uchar_video_range<YUV_I420, 4, 0>(
        height,
        width,
        ystride,
        iny,
        ustride,
        inu,
        vstride,
        inv,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<4, 0>(height, width, ystride, ustride, vstride, iny, inu, inv, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *u_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *v_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<0, 3, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BGR2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t ystride,
    uint8_t *outy,
    int32_t ustride,
    uint8_t *outu,
    int32_t vstride,
    uint8_t *outv)
{
    if (!inData || !outy || !outu || !outv || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || inWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = ystride;
    uint8_t *y_ptr    = outy;
    int32_t uStride   = ustride;
    uint8_t *u_ptr    = outu;
    int32_t vStride   = vstride;
    uint8_t *v_ptr    = outv;
    bgr_to_yuv420_uchar_video_range<0, 3, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, ystride, ustride, vstride, outy, outu, outv);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *u_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *v_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<0, 4, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode BGRA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t ystride,
    uint8_t *outy,
    int32_t ustride,
    uint8_t *outu,
    int32_t vstride,
    uint8_t *outv)
{
    if (!inData || !outy || !outu || !outv || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || inWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = ystride;
    uint8_t *y_ptr    = outy;
    int32_t uStride   = ustride;
    uint8_t *u_ptr    = outu;
    int32_t vStride   = vstride;
    uint8_t *v_ptr    = outv;
    bgr_to_yuv420_uchar_video_range<0, 4, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, ystride, ustride, vstride, outy, outu, outv);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *v_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *u_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<0, 3, YUV_YV12>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outWidthStride, outWidthStride / 2, outWidthStride / 2, outDataY, outDataU, outDataV);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *v_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *u_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<0, 4, YUV_YV12>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    RGBtoYUV420p<4, 0>(height,
                       width,
                       inWidthStride,
                       inData,
                       outWidthStride,
                       outWidthStride / 2,
                       outWidthStride / 2,
                       outDataY,
                       outDataU,
                       outDataV);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode YV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *vptr = inData + inWidthStride * height;
    const uint8_t *uptr = inData + inWidthStride * height + inWidthStride * height / 4;

    yuv420_to_bgr_uchar_video_range<YUV_YV12, 3, 0>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    YUV420ptoRGB<3, 0>(height, width, inWidthStride, inWidthStride / 2, inWidthStride / 2, inDataY, inDataU, inDataV, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode YV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *vptr = inData + inWidthStride * height;
    const uint8_t *uptr = inData + inWidthStride * height + inWidthStride * height / 4;

    yuv420_to_bgr_uchar_video_range<YUV_YU12, 4, 0>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    YUV420ptoRGB<4, 0>(height,
                       width,
                       inWidthStride,
                       inWidthStride / 2,
                       inWidthStride / 2,
                       inDataY,
                       inDataU,
                       inDataV,
                       outWidthStride,
                       outData);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode I4202RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *uptr = inData + inWidthStride * height;
    const uint8_t *vptr = inData + inWidthStride * height + inWidthStride * height / 4;

    yuv420_to_bgr_uchar_video_range<YUV_I420, 3, 2>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<3, 2, 0>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode I4202RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t ystride,
    const uint8_t *iny,
    int32_t ustride,
    const uint8_t *inu,
    int32_t vstride,
    const uint8_t *inv,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!iny || !inu || !inv || !outData || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    yuv420_to_bgr_uchar_video_range<YUV_I420, 3, 2>(
        height,
        width,
        ystride,
        iny,
        ustride,
        inu,
        vstride,
        inv,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<3, 2>(height, width, ystride, ustride, vstride, iny, inu, inv, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode I4202RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *uptr = inData + inWidthStride * height;
    const uint8_t *vptr = inData + inWidthStride * height + inWidthStride * height / 4;

    yuv420_to_bgr_uchar_video_range<YUV_I420, 4, 2>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<4, 2, 0>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode I4202RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t ystride,
    const uint8_t *iny,
    int32_t ustride,
    const uint8_t *inu,
    int32_t vstride,
    const uint8_t *inv,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!iny || !inu || !inv || !outData || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    yuv420_to_bgr_uchar_video_range<YUV_I420, 4, 2>(
        height,
        width,
        ystride,
        iny,
        ustride,
        inu,
        vstride,
        inv,
        outWidthStride,
        outData);
#else
    YUV420ptoRGB<4, 2>(height, width, ystride, ustride, vstride, iny, inu, inv, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode YV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *vptr = inData + inWidthStride * height;
    const uint8_t *uptr = inData + inWidthStride * height + inWidthStride * height / 4;

    yuv420_to_bgr_uchar_video_range<YUV_I420, 3, 2>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    YUV420ptoRGB<3, 2>(height,
                       width,
                       inWidthStride,
                       inWidthStride / 2,
                       inWidthStride / 2,
                       inDataY,
                       inDataU,
                       inDataV,
                       outWidthStride,
                       outData);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode YV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t yStride     = inWidthStride;
    int32_t uStride     = inWidthStride >> 1;
    int32_t vStride     = inWidthStride >> 1;
    const uint8_t *yptr = inData;
    const uint8_t *vptr = inData + inWidthStride * height;
    const uint8_t *uptr = inData + inWidthStride * height + inWidthStride * height / 4;
    yuv420_to_bgr_uchar_video_range<YUV_I420, 4, 2>(
        height,
        width,
        yStride,
        yptr,
        uStride,
        uptr,
        vStride,
        vptr,
        outWidthStride,
        outData);
#else
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    YUV420ptoRGB<4, 2>(height,
                       width,
                       inWidthStride,
                       inWidthStride / 2,
                       inWidthStride / 2,
                       inDataY,
                       inDataU,
                       inDataV,
                       outWidthStride,
                       outData);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *u_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *v_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<2, 3, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode RGB2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t ystride,
    uint8_t *outy,
    int32_t ustride,
    uint8_t *outu,
    int32_t vstride,
    uint8_t *outv)
{
    if (!inData || !outy || !outu || !outv || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || inWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = ystride;
    uint8_t *y_ptr    = outy;
    int32_t uStride   = ustride;
    uint8_t *u_ptr    = outu;
    int32_t vStride   = vstride;
    uint8_t *v_ptr    = outv;
    bgr_to_yuv420_uchar_video_range<2, 3, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, ystride, ustride, vstride, outy, outu, outv);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *u_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *v_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<2, 4, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, outWidthStride, outData);
#endif
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode RGBA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t ystride,
    uint8_t *outy,
    int32_t ustride,
    uint8_t *outu,
    int32_t vstride,
    uint8_t *outv)
{
    if (!inData || !outy || !outu || !outv || height == 0 || width == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ystride == 0 || ustride == 0 || vstride == 0 || inWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = ystride;
    uint8_t *y_ptr    = outy;
    int32_t uStride   = ustride;
    uint8_t *u_ptr    = outu;
    int32_t vStride   = vstride;
    uint8_t *v_ptr    = outv;
    bgr_to_yuv420_uchar_video_range<2, 4, YUV_I420>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, ystride, ustride, vstride, outy, outu, outv);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *v_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *u_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<2, 3, YUV_YV12>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outWidthStride, outWidthStride / 2, outWidthStride / 2, outDataY, outDataU, outDataV);
#endif
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

#ifdef USE_QUANTIZED
    int32_t rgbStride = inWidthStride;
    int32_t yStride   = outWidthStride;
    uint8_t *y_ptr    = outData;
    int32_t uStride   = outWidthStride >> 1;
    uint8_t *v_ptr    = outData + outWidthStride * height;
    int32_t vStride   = outWidthStride >> 1;
    ;
    uint8_t *u_ptr = outData + outWidthStride * height + outWidthStride * height / 4;
    bgr_to_yuv420_uchar_video_range<2, 4, YUV_YV12>(height, width, rgbStride, inData, yStride, y_ptr, uStride, u_ptr, vStride, v_ptr);
#else
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    RGBtoYUV420p<4, 2>(height,
                       width,
                       inWidthStride,
                       inData,
                       outWidthStride,
                       outWidthStride / 2,
                       outWidthStride / 2,
                       outDataY,
                       outDataU,
                       outDataV);
#endif
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
