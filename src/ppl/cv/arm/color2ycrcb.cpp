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


//参考https://github1s.com/opencv/opencv/blob/4.x/3rdparty/carotene/src/colorconvert.cpp
//https://github1s.com/openppl-public/ppl.cv/blob/master/src/ppl/cv/arm/erode.cpp#L163
#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/arm/typetraits.hpp"
#include "ppl/cv/types.h"
#include <algorithm>
#include <complex>
#include <string.h>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

//fou u8
const int16_t kR2YIntCoeff = 4899;
const int16_t kG2YIntCoeff = 9617;
const int16_t kB2YIntCoeff = 1868;
const int16_t kCRIntCoeff = 11682;
const int16_t kCBIntCoeff = 9241;
const int16_t kCr2RIntCoeff = 22987;
const int16_t kCb2BIntCoeff = 29049;
const int16_t kY2GCrIntCoeff = -11698;
const int16_t kY2GCbIntCoeff = -5636;
const int16_t kShift14IntDelta = 2097152;
//for float32
const float32_t kR2YFloatCoeff = 0.299f;
const float32_t kG2YFloatCoeff = 0.587f;
const float32_t kB2YFloatCoeff = 0.114f;
const float32_t kCRFloatCoeff = 0.713f;
const float32_t kCBFloatCoeff = 0.564f;
const float32_t kCr2RFloatCoeff = 1.403f;
const float32_t kCb2BFloatCoeff = 1.773f;
const float32_t kY2GCrFloatCoeff = -0.714f;
const float32_t kY2GCbFloatCoeff = -0.344f;
const float32_t kYCrCbUcharDelta = 128;
const float32_t kYCrCbFloatDelta = 0.5f;


// rgb/rgba convert YCrCb
template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode rgb_to_ycrcb_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    const uint8_t* srcPtr = src;
    uint8_t* dstPtr       = dst;
    typedef typename DT<ncSrc, uint8_t>::vec_DT srcType;
    typedef typename DT<ncDst, uint8_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst;
        uint32_t i;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src = vldx_u8_f32<ncSrc, uint8_t, srcType>(srcPtr + ncSrc * i);
            int16x8_t r = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            int16x8_t g = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            int16x8_t b = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x8_t y = vrshrq_n_s16(vaddq_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(r, kR2YIntCoeff), vmulq_n_s16(g, kG2YIntCoeff)), vmulq_n_s16(b, kB2YIntCoeff)), vdupq_n_s16(1 << 13)), 14);
            
            
            int16x8_t cr = vshrq_n_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(vsubq_s16(r,y),kCRIntCoeff),vdupq_n_s16(kShift14IntDelta)),vdupq_n_s16(1<<14 - 1)),14);
        
            int16x8_t cb = vshrq_n_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(vsubq_s16(b,y),kCBIntCoeff),vdupq_n_s16(kShift14IntDelta)),vdupq_n_s16(1<<14 - 1)),14);

            v_dst.val[0] = vqmovun_s16(y);
            v_dst.val[1] = vqmovun_s16(cr);
            v_dst.val[2] = vqmovun_s16(cb);

            vstx_u8_f32<ncDst, uint8_t, dstType>(dstPtr + ncDst * i, v_dst);
        }

        for (; i < width; i++) {
            uint8_t r = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], b = srcPtr[ncSrc * i + 2];
            
            uint8_t y = (r*kR2YIntCoeff + g*kG2YIntCoeff + b*kB2YIntCoeff + (1 << 13)) >> 14;
            uint8_t Cr = ((r-y)*kCRIntCoeff + kShift14IntDelta + ((1 << 14) - 1) ) >> 14;
            uint8_t Cb = ((b-y)*kCBIntCoeff + kShift14IntDelta + ((1 << 14) - 1) ) >> 14;

            dstPtr[ncDst * i]     = y;
            dstPtr[ncDst * i + 1] = Cr;
            dstPtr[ncDst * i + 2] = Cb;
        }

    }
    return ppl::common::RC_SUCCESS;
}
template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode rgb_to_ycrcb_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float32_t* src,
    const int32_t dstStride,
    float32_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    const float32_t* srcPtr = src;
    float32_t* dstPtr       = dst;
    typedef typename DT<ncSrc, float32_t>::vec_DT srcType;
    typedef typename DT<ncDst, float32_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst;
        uint32_t i;
        for (i = 0; i <= width - 4; i += 4) {
            srcType v_src = vldx_u8_f32<ncSrc, float32_t, srcType>(srcPtr + ncSrc * i);
            
            float32x4_t r = v_src.val[0], g = v_src.val[1], b = v_src.val[2]; 
            float32x4_t y = vaddq_f32(vaddq_f32(vmulq_n_f32(r,kR2YFloatCoeff),vmulq_n_f32(g,kG2YFloatCoeff)),vmulq_n_f32(b,kB2YFloatCoeff));
            float32x4_t cr = vaddq_f32(vmulq_n_f32(vsubq_f32(r , y),kCRFloatCoeff),vdupq_n_f32(kYCrCbFloatDelta));
            float32x4_t cb = vaddq_f32(vmulq_n_f32(vsubq_f32(b , y),kCBFloatCoeff),vdupq_n_f32(kYCrCbFloatDelta));

            v_dst.val[0] = y;
            v_dst.val[1] = cr;
            v_dst.val[2] = cb;

            vstx_u8_f32<ncDst, float32_t, dstType>(dstPtr + ncDst * i, v_dst);
        }

        for (; i < width; i++) {
            float32_t r = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], b = srcPtr[ncSrc * i + 2];
            
            float32_t y = r*kR2YFloatCoeff + g*kG2YFloatCoeff + b*kB2YFloatCoeff;
            float32_t cr = (r-y)*kCRFloatCoeff + kYCrCbFloatDelta;
            float32_t cb = (b-y)*kCBFloatCoeff + kYCrCbFloatDelta;

            dstPtr[ncDst * i]     = y;
            dstPtr[ncDst * i + 1] = cr;
            dstPtr[ncDst * i + 2] = cb;
        }

    }
    return ppl::common::RC_SUCCESS;
}

// bgr/bgra convert YCrCb
template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode bgr_to_ycrcb_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    const uint8_t* srcPtr = src;
    uint8_t* dstPtr       = dst;
    typedef typename DT<ncSrc, uint8_t>::vec_DT srcType;
    typedef typename DT<ncDst, uint8_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst;
        uint32_t i;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src = vldx_u8_f32<ncSrc, uint8_t, srcType>(srcPtr + ncSrc * i);
            int16x8_t b = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            int16x8_t g = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            int16x8_t r = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x8_t y = vrshrq_n_s16(vaddq_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(r, kR2YIntCoeff), vmulq_n_s16(g, kG2YIntCoeff)), vmulq_n_s16(b, kB2YIntCoeff)), vdupq_n_s16(1 << 13)), 14);
            
            
            int16x8_t cr = vshrq_n_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(vsubq_s16(r,y),kCRIntCoeff),vdupq_n_s16(kShift14IntDelta)),vdupq_n_s16(1<<14 - 1)),14);
        
            int16x8_t cb = vshrq_n_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(vsubq_s16(b,y),kCBIntCoeff),vdupq_n_s16(kShift14IntDelta)),vdupq_n_s16(1<<14 - 1)),14);

            v_dst.val[0] = vqmovun_s16(y);
            v_dst.val[1] = vqmovun_s16(cr);
            v_dst.val[2] = vqmovun_s16(cb);

            vstx_u8_f32<ncDst, uint8_t, dstType>(dstPtr + ncDst * i, v_dst);
        }

        for (; i < width; i++) {
            uint8_t b = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], r = srcPtr[ncSrc * i + 2];
            
            uint8_t y = (r*kR2YIntCoeff + g*kG2YIntCoeff + b*kB2YIntCoeff + (1 << 13)) >> 14;
            uint8_t Cr = ((r-y)*kCRIntCoeff + kShift14IntDelta + ((1 << 14) - 1) ) >> 14;
            uint8_t Cb = ((b-y)*kCBIntCoeff + kShift14IntDelta + ((1 << 14) - 1) ) >> 14;

            dstPtr[ncDst * i]     = y;
            dstPtr[ncDst * i + 1] = Cr;
            dstPtr[ncDst * i + 2] = Cb;
        }

    }
    return ppl::common::RC_SUCCESS;
}
template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode bgr_to_ycrcb_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float32_t* src,
    const int32_t dstStride,
    float32_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    const float32_t* srcPtr = src;
    float32_t* dstPtr       = dst;
    typedef typename DT<ncSrc, float32_t>::vec_DT srcType;
    typedef typename DT<ncDst, float32_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst;
        uint32_t i;
        for (i = 0; i <= width - 4; i += 4) {
            srcType v_src = vldx_u8_f32<ncSrc, float32_t, srcType>(srcPtr + ncSrc * i);
            
            float32x4_t r = v_src.val[0], g = v_src.val[1], b = v_src.val[2]; 
            float32x4_t y = vaddq_f32(vaddq_f32(vmulq_n_f32(r,kR2YFloatCoeff),vmulq_n_f32(g,kG2YFloatCoeff)),vmulq_n_f32(b,kB2YFloatCoeff));
            float32x4_t cr = vaddq_f32(vmulq_n_f32(vsubq_f32(r , y),kCRFloatCoeff),vdupq_n_f32(kYCrCbFloatDelta));
            float32x4_t cb = vaddq_f32(vmulq_n_f32(vsubq_f32(b , y),kCBFloatCoeff),vdupq_n_f32(kYCrCbFloatDelta));

            v_dst.val[0] = y;
            v_dst.val[1] = cr;
            v_dst.val[2] = cb;

            vstx_u8_f32<ncDst, float32_t, dstType>(dstPtr + ncDst * i, v_dst);
        }

        for (; i < width; i++) {
            float32_t r = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], b = srcPtr[ncSrc * i + 2];
            
            float32_t y = r*kR2YFloatCoeff + g*kG2YFloatCoeff + b*kB2YFloatCoeff;
            float32_t cr = (r-y)*kCRFloatCoeff + kYCrCbFloatDelta;
            float32_t cb = (b-y)*kCBFloatCoeff + kYCrCbFloatDelta;

            dstPtr[ncDst * i]     = y;
            dstPtr[ncDst * i + 1] = cr;
            dstPtr[ncDst * i + 2] = cb;
        }

    }
    return ppl::common::RC_SUCCESS;
}


template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode ycrcb_to_rgb_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst){
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    const uint8_t* srcPtr = src;
    uint8_t* dstPtr       = dst;
    typedef typename DT<ncSrc, uint8_t>::vec_DT srcType;
    typedef typename DT<ncDst, uint8_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst;
        if(ncDst == 4){
            v_dst.val[3] = vdup_n_u8(255);
        }
        uint32_t i;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src = vldx_u8_f32<ncSrc, uint8_t, srcType>(srcPtr + ncSrc * i);
            int16x8_t y = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            int16x8_t cr = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            int16x8_t cb = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x8_t r = vaddq_s16(y,vshrq_n_s16(vaddq_s16(vmulq_n_s16(cr,kCr2RIntCoeff),vdupq_n_s16(1<<14 - 1)),14));
            int16x8_t g = vaddq_s16(y,vshrq_n_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(cr,kY2GCrIntCoeff),vmulq_n_s16(cb,kY2GCbIntCoeff)),vdupq_n_s16(1<<14 - 1)),14));
            int16x8_t b = vaddq_s16(y,vshrq_n_s16(vaddq_s16(vmulq_n_s16(cb,kCb2BIntCoeff),vdupq_n_s16(1<<14 - 1)),14));

            v_dst.val[0] = vqmovun_s16(r);
            v_dst.val[1] = vqmovun_s16(g);
            v_dst.val[2] = vqmovun_s16(b);

            vstx_u8_f32<ncDst, uint8_t, dstType>(dstPtr + ncDst * i, v_dst);
        }

        for (; i < width; i++) {
            int16_t y = srcPtr[ncSrc * i], cr = srcPtr[ncSrc * i + 1], cb = srcPtr[ncSrc * i + 2];
            
            int16_t r = y + ((cr*kCr2RIntCoeff + (1 << 14) - 1)>>14);
            int16_t g = y + ((cr*kY2GCrIntCoeff + cb*kY2GCbIntCoeff + (1 << 14) - 1)>>14);
            int16_t b = y + ((cb*kCb2BIntCoeff + (1 << 14) - 1)>>14);

            dstPtr[ncDst * i] = r > 255 ? 255 : (r < 0 ? 0 : r);
            dstPtr[ncDst * i + 1] = g > 255 ? 255 : (g < 0 ? 0 : g);
            dstPtr[ncDst * i + 2] = b > 255 ? 255 : (b < 0 ? 0 : b);
            if(ncDst == 4){
                dstPtr[ncDst * i + 3] = 255;
            }
        }

    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode ycrcb_to_rgb_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float32_t* src,
    const int32_t dstStride,
    float32_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    
    const float32_t* srcPtr = src;
    float32_t* dstPtr       = dst;
    typedef typename DT<ncSrc, float32_t>::vec_DT srcType;
    typedef typename DT<ncDst, float32_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst;
        if(ncDst == 4){
            v_dst.val[3] = vdupq_n_f32(1.0);
        }
        uint32_t i;
        for (i = 0; i <= width - 4; i += 4) {
            srcType v_src = vldx_u8_f32<ncSrc, float32_t, srcType>(srcPtr + ncSrc * i);
            
            float32x4_t y = v_src[0], cr = v_src[1], cb = v_src[2]; 
            float32x4_t r = vaddq_f32(y,vmulq_n_f32(cr,kCr2RFloatCoeff));
            float32x4_t g = vaddq_f32(vaddq_f32(y,vmulq_n_f32(cr,kY2GCrFloatCoeff)),vmulq_n_f32(cb,kY2GCbFloatCoeff));
            float32x4_t b = vaddq_f32(y,vmulq_n_f32(cb,kCb2BFloatCoeff));

            v_dst.val[0] = r;
            v_dst.val[1] = g;
            v_dst.val[2] = b;

            vstx_u8_f32<ncDst, float32_t, dstType>(dstPtr + ncDst * i, v_dst);
        }

        for (; i < width; i++) {
            float32_t y = srcPtr[ncSrc * i], cr = srcPtr[ncSrc * i + 1], cb = srcPtr[ncSrc * i + 2];
            
            float32_t r = y + cr*kCr2RFloatCoeff;
            float32_t g = y + cr*kY2GCrFloatCoeff + cb*kY2GCbFloatCoeff;
            float32_t b = y + cb*kCb2BFloatCoeff;

            dstPtr[ncDst * i]     = r;
            dstPtr[ncDst * i + 1] = g;
            dstPtr[ncDst * i + 2] = b;
            if(ncDst == 4){
                dstPtr[ncDst * i + 3] = 1.0f;
            }
        }

    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2YCrCb<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_ycrcb_u8<3,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode RGBA2YCrCb<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_ycrcb_u8<4,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode RGB2YCrCb<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return rgb_to_ycrcb_f32<3,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode RGBA2YCrCb<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return rgb_to_ycrcb_f32<4,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode BGR2YCrCb<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return bgr_to_ycrcb_u8<3,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode BGRA2YCrCb<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return bgr_to_ycrcb_u8<4,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGR2YCrCb<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return bgr_to_ycrcb_f32<3,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGRA2YCrCb<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return bgr_to_ycrcb_f32<4,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode YCrCb2RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return ycrcb_to_rgb_u8<3,3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm