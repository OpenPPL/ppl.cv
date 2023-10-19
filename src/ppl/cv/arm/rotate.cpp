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

#include "ppl/cv/arm/rotate.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include "intrinutils_neon.hpp"
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

// Applies to perfect square blocks only
template <typename T, int CHANNELS, int OUT_BLOCKDIM_R, int OUT_BLOCKDIM_C>
inline void rotate90_inner_block(const T* src,
                                 int obi,
                                 int obj,
                                 int outHeight,
                                 int outWidth,
                                 int inWidthStride,
                                 int outWidthStride,
                                 T* dst)
{
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i++) {
        prefetch(src + i * inWidthStride + obj * OUT_BLOCKDIM_C * CHANNELS, 0);
    }
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i++) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j++) {
            for (int c = 0; c < CHANNELS; ++c) {
                dst[j * outWidthStride + (outWidth - 1 - i) * CHANNELS + c] = src[i * inWidthStride + j * CHANNELS + c];
            }
        }
    }
}

template <>
inline void rotate90_inner_block<float, 1, 32, 32>(const float* src,
                                                   int obi,
                                                   int obj,
                                                   int outHeight,
                                                   int outWidth,
                                                   int inWidthStride,
                                                   int outWidthStride,
                                                   float* dst)
{
    constexpr int CHANNELS = 1;
    constexpr int OUT_BLOCKDIM_R = 32;
    constexpr int OUT_BLOCKDIM_C = 32;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            float32x4_t va = vld1q_f32(src + (i + 0) * inWidthStride + j * CHANNELS);
            float32x4_t vb = vld1q_f32(src + (i + 1) * inWidthStride + j * CHANNELS);
            float32x4_t vc = vld1q_f32(src + (i + 2) * inWidthStride + j * CHANNELS);
            float32x4_t vd = vld1q_f32(src + (i + 3) * inWidthStride + j * CHANNELS);
            neon_rotate90_f32_4x4(va, vb, vc, vd);
            vst1q_f32(dst + (j + 0) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, va);
            vst1q_f32(dst + (j + 1) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vb);
            vst1q_f32(dst + (j + 2) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vc);
            vst1q_f32(dst + (j + 3) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vd);
        }
    }
}

template <>
inline void rotate90_inner_block<float, 3, 32, 32>(const float* src,
                                                   int obi,
                                                   int obj,
                                                   int outHeight,
                                                   int outWidth,
                                                   int inWidthStride,
                                                   int outWidthStride,
                                                   float* dst)
{
    constexpr int CHANNELS = 3;
    constexpr int OUT_BLOCKDIM_R = 32;
    constexpr int OUT_BLOCKDIM_C = 32;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 0) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            float32x4x3_t va = vld3q_f32(src + (i + 0) * inWidthStride + j * CHANNELS);
            float32x4x3_t vb = vld3q_f32(src + (i + 1) * inWidthStride + j * CHANNELS);
            float32x4x3_t vc = vld3q_f32(src + (i + 2) * inWidthStride + j * CHANNELS);
            float32x4x3_t vd = vld3q_f32(src + (i + 3) * inWidthStride + j * CHANNELS);
            neon_rotate90_f32_4x4(va.val[0], vb.val[0], vc.val[0], vd.val[0]);
            neon_rotate90_f32_4x4(va.val[1], vb.val[1], vc.val[1], vd.val[1]);
            neon_rotate90_f32_4x4(va.val[2], vb.val[2], vc.val[2], vd.val[2]);
            vst3q_f32(dst + (j + 0) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, va);
            vst3q_f32(dst + (j + 1) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vb);
            vst3q_f32(dst + (j + 2) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vc);
            vst3q_f32(dst + (j + 3) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vd);
        }
    }
}

template <>
inline void rotate90_inner_block<float, 4, 32, 32>(const float* src,
                                                   int obi,
                                                   int obj,
                                                   int outHeight,
                                                   int outWidth,
                                                   int inWidthStride,
                                                   int outWidthStride,
                                                   float* dst)
{
    constexpr int CHANNELS = 4;
    constexpr int OUT_BLOCKDIM_R = 32;
    constexpr int OUT_BLOCKDIM_C = 32;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 0) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            float32x4x4_t va = vld4q_f32(src + (i + 0) * inWidthStride + j * CHANNELS);
            float32x4x4_t vb = vld4q_f32(src + (i + 1) * inWidthStride + j * CHANNELS);
            float32x4x4_t vc = vld4q_f32(src + (i + 2) * inWidthStride + j * CHANNELS);
            float32x4x4_t vd = vld4q_f32(src + (i + 3) * inWidthStride + j * CHANNELS);
            neon_rotate90_f32_4x4(va.val[0], vb.val[0], vc.val[0], vd.val[0]);
            neon_rotate90_f32_4x4(va.val[1], vb.val[1], vc.val[1], vd.val[1]);
            neon_rotate90_f32_4x4(va.val[2], vb.val[2], vc.val[2], vd.val[2]);
            neon_rotate90_f32_4x4(va.val[3], vb.val[3], vc.val[3], vd.val[3]);
            vst4q_f32(dst + (j + 0) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, va);
            vst4q_f32(dst + (j + 1) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vb);
            vst4q_f32(dst + (j + 2) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vc);
            vst4q_f32(dst + (j + 3) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS, vd);
        }
    }
}

template <>
inline void rotate90_inner_block<uint8_t, 1, 64, 64>(const uint8_t* src,
                                                     int obi,
                                                     int obj,
                                                     int outHeight,
                                                     int outWidth,
                                                     int inWidthStride,
                                                     int outWidthStride,
                                                     uint8_t* dst)
{
    constexpr int CHANNELS = 1;
    constexpr int OUT_BLOCKDIM_R = 64;
    constexpr int OUT_BLOCKDIM_C = 64;
    constexpr int INNER_BLOCKDIM = 8;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 4) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 5) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 6) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 7) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            uint8x8_t va = vld1_u8(src + (i + 0) * inWidthStride + j * CHANNELS);
            uint8x8_t vb = vld1_u8(src + (i + 1) * inWidthStride + j * CHANNELS);
            uint8x8_t vc = vld1_u8(src + (i + 2) * inWidthStride + j * CHANNELS);
            uint8x8_t vd = vld1_u8(src + (i + 3) * inWidthStride + j * CHANNELS);
            uint8x8_t ve = vld1_u8(src + (i + 4) * inWidthStride + j * CHANNELS);
            uint8x8_t vf = vld1_u8(src + (i + 5) * inWidthStride + j * CHANNELS);
            uint8x8_t vg = vld1_u8(src + (i + 6) * inWidthStride + j * CHANNELS);
            uint8x8_t vh = vld1_u8(src + (i + 7) * inWidthStride + j * CHANNELS);
            neon_rotate90_u8_8x8(va, vb, vc, vd, ve, vf, vg, vh);
            vst1_u8(dst + (j + 0) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, va);
            vst1_u8(dst + (j + 1) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vb);
            vst1_u8(dst + (j + 2) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vc);
            vst1_u8(dst + (j + 3) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vd);
            vst1_u8(dst + (j + 4) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, ve);
            vst1_u8(dst + (j + 5) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vf);
            vst1_u8(dst + (j + 6) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vg);
            vst1_u8(dst + (j + 7) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vh);
        }
    }
}

template <>
inline void rotate90_inner_block<uint8_t, 3, 64, 64>(const uint8_t* src,
                                                     int obi,
                                                     int obj,
                                                     int outHeight,
                                                     int outWidth,
                                                     int inWidthStride,
                                                     int outWidthStride,
                                                     uint8_t* dst)
{
    constexpr int CHANNELS = 3;
    constexpr int OUT_BLOCKDIM_R = 64;
    constexpr int OUT_BLOCKDIM_C = 64;
    constexpr int INNER_BLOCKDIM = 8;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 4) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 5) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 6) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 7) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            uint8x8x3_t va = vld3_u8(src + (i + 0) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vb = vld3_u8(src + (i + 1) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vc = vld3_u8(src + (i + 2) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vd = vld3_u8(src + (i + 3) * inWidthStride + j * CHANNELS);
            uint8x8x3_t ve = vld3_u8(src + (i + 4) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vf = vld3_u8(src + (i + 5) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vg = vld3_u8(src + (i + 6) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vh = vld3_u8(src + (i + 7) * inWidthStride + j * CHANNELS);
            neon_rotate90_u8_8x8(
                va.val[0], vb.val[0], vc.val[0], vd.val[0], ve.val[0], vf.val[0], vg.val[0], vh.val[0]);
            neon_rotate90_u8_8x8(
                va.val[1], vb.val[1], vc.val[1], vd.val[1], ve.val[1], vf.val[1], vg.val[1], vh.val[1]);
            neon_rotate90_u8_8x8(
                va.val[2], vb.val[2], vc.val[2], vd.val[2], ve.val[2], vf.val[2], vg.val[2], vh.val[2]);
            vst3_u8(dst + (j + 0) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, va);
            vst3_u8(dst + (j + 1) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vb);
            vst3_u8(dst + (j + 2) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vc);
            vst3_u8(dst + (j + 3) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vd);
            vst3_u8(dst + (j + 4) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, ve);
            vst3_u8(dst + (j + 5) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vf);
            vst3_u8(dst + (j + 6) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vg);
            vst3_u8(dst + (j + 7) * outWidthStride + (outWidth - 1 - i - 7) * CHANNELS, vh);
        }
    }
}

template <>
inline void rotate90_inner_block<uint8_t, 4, 64, 64>(const uint8_t* src,
                                                     int obi,
                                                     int obj,
                                                     int outHeight,
                                                     int outWidth,
                                                     int inWidthStride,
                                                     int outWidthStride,
                                                     uint8_t* dst)
{
    // little trick here: use uint32_t for uint8_t * 4 channel to avoid register overflow
    constexpr int CHANNELS = 4;
    constexpr int OUT_BLOCKDIM_R = 64;
    constexpr int OUT_BLOCKDIM_C = 64;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 0) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            // load 4 * 4channel pixel
            uint32x4_t va = vld1q_u32((uint32_t*)(src + (i + 0) * inWidthStride + j * CHANNELS));
            uint32x4_t vb = vld1q_u32((uint32_t*)(src + (i + 1) * inWidthStride + j * CHANNELS));
            uint32x4_t vc = vld1q_u32((uint32_t*)(src + (i + 2) * inWidthStride + j * CHANNELS));
            uint32x4_t vd = vld1q_u32((uint32_t*)(src + (i + 3) * inWidthStride + j * CHANNELS));
            // transpose 4channel pixel as an uint32_t
            neon_rotate90_u32_4x4(va, vb, vc, vd);
            // store 4 * 4channel pixel
            vst1q_u32((uint32_t*)(dst + (j + 0) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS), va);
            vst1q_u32((uint32_t*)(dst + (j + 1) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS), vb);
            vst1q_u32((uint32_t*)(dst + (j + 2) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS), vc);
            vst1q_u32((uint32_t*)(dst + (j + 3) * outWidthStride + (outWidth - 1 - i - 3) * CHANNELS), vd);
        }
    }
}

template <typename T, int32_t CHANNELS, int32_t BLOCKDIM_R, int32_t BLOCKDIM_C>
void imgRotate90degree(int32_t inHeight,
                       int32_t inWidth,
                       int32_t inWidthStride,
                       const T* src,
                       int32_t outHeight,
                       int32_t outWidth,
                       int32_t outWidthStride,
                       T* dst)
{
    // After outer block the image is splitted into 4 regions
    // Region A are perfect blocks, B and C are partially perfect blocks, and D is what remains
    // |           |   |
    // |     A     | B |
    // |___________|___|
    // |     C     | D |

    // Part A
    int row_blks = inHeight / BLOCKDIM_R;
    int col_blks = inWidth / BLOCKDIM_C;
    for (int obi = 0; obi < row_blks; ++obi) {
        for (int obj = 0; obj < col_blks; ++obj) {
            rotate90_inner_block<T, CHANNELS, BLOCKDIM_R, BLOCKDIM_C>(
                src, obi, obj, outHeight, outWidth, inWidthStride, outWidthStride, dst);
        }
    }
    // Part B
    for (int obi = 0; obi < row_blks; ++obi) {
        for (int i = obi * BLOCKDIM_R; i < (obi + 1) * BLOCKDIM_R; i++)
            prefetch(src + (i + 1) * inWidthStride + (col_blks - 1) * BLOCKDIM_C * CHANNELS, 0);
        for (int i = obi * BLOCKDIM_R; i < (obi + 1) * BLOCKDIM_R; i++) {
            for (int j = (col_blks - 1) * BLOCKDIM_C; j < inWidth; j++) {
                for (int c = 0; c < CHANNELS; ++c) {
                    dst[j * outWidthStride + (outWidth - 1 - i) * CHANNELS + c] =
                        src[i * inWidthStride + j * CHANNELS + c];
                }
            }
        }
    }
    // Part C
    for (int obj = 0; obj < col_blks; ++obj) {
        for (int i = (row_blks - 1) * BLOCKDIM_R; i < inHeight; i++) {
            prefetch(src + i * inWidthStride + (obj + 1) * BLOCKDIM_C * CHANNELS, 0);
            for (int j = obj * BLOCKDIM_C; j < (obj + 1) * BLOCKDIM_C; j++) {
                for (int c = 0; c < CHANNELS; ++c) {
                    dst[j * outWidthStride + (outWidth - 1 - i) * CHANNELS + c] =
                        src[i * inWidthStride + j * CHANNELS + c];
                }
            }
        }
    }
    // Part D
    for (int i = (row_blks - 1) * BLOCKDIM_R; i < inHeight; i++) {
        for (int j = (col_blks - 1) * BLOCKDIM_C; j < inWidth; j++) {
            for (int c = 0; c < CHANNELS; ++c) {
                dst[j * outWidthStride + (outWidth - 1 - i) * CHANNELS + c] = src[i * inWidthStride + j * CHANNELS + c];
            }
        }
    }
    return;
}

template <typename T, int32_t nc>
void imgRotate180degree(int32_t inHeight,
                        int32_t inWidth,
                        int32_t inWidthStride,
                        const T* inData,
                        int32_t outHeight,
                        int32_t outWidth,
                        int32_t outWidthStride,
                        T* outData)
{
    for (int32_t i = 0; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] =
                    inData[(inHeight - i - 1) * inWidthStride + (inWidth - j - 1) * nc + c];
            }
        }
    }
}

template <>
void imgRotate180degree<float, 1>(int32_t inHeight,
                                  int32_t inWidth,
                                  int32_t inWidthStride,
                                  const float* inData,
                                  int32_t outHeight,
                                  int32_t outWidth,
                                  int32_t outWidthStride,
                                  float* outData)
{
    constexpr int nc = 1;
    for (int32_t i = 0; i < inHeight; i++) {
        const float* in = inData + i * inWidthStride;
        float* out = outData + (outHeight - i - 1) * outWidthStride;
        int32_t j = 0;
        for (; j <= inWidth - 8; j += 8) {
            prefetch(in + j * nc);
            float32x2_t vdata0 = vld1_f32(in + (j + 0) * nc);
            float32x2_t vdata1 = vld1_f32(in + (j + 2) * nc);
            float32x2_t vdata2 = vld1_f32(in + (j + 4) * nc);
            float32x2_t vdata3 = vld1_f32(in + (j + 6) * nc);

            float32x2_t vout3 = neon_reverse_f32x2(vdata0);
            float32x2_t vout2 = neon_reverse_f32x2(vdata1);
            float32x2_t vout1 = neon_reverse_f32x2(vdata2);
            float32x2_t vout0 = neon_reverse_f32x2(vdata3);

            vst1_f32(out + (outWidth - j - 1 - 7) * nc, vout0);
            vst1_f32(out + (outWidth - j - 1 - 5) * nc, vout1);
            vst1_f32(out + (outWidth - j - 1 - 3) * nc, vout2);
            vst1_f32(out + (outWidth - j - 1 - 1) * nc, vout3);
        }
        for (; j <= inWidth - 2; j += 2) {
            float32x2_t vdata0 = vld1_f32(in + (j + 0) * nc);
            float32x2_t vout0 = neon_reverse_f32x2(vdata0);
            vst1_f32(out + (outWidth - j - 1 - 1) * nc, vout0);
        }
        for (; j < inWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[(outHeight - i - 1) * outWidthStride + (outWidth - j - 1) * nc + c] =
                    inData[i * inWidthStride + j * nc + c];
            }
        }
    }
}

template <>
void imgRotate180degree<uint8_t, 1>(int32_t inHeight,
                                    int32_t inWidth,
                                    int32_t inWidthStride,
                                    const uint8_t* inData,
                                    int32_t outHeight,
                                    int32_t outWidth,
                                    int32_t outWidthStride,
                                    uint8_t* outData)
{
    constexpr int nc = 1;
    for (int32_t i = 0; i < inHeight; i++) {
        const uint8_t* in = inData + i * inWidthStride;
        uint8_t* out = outData + (outHeight - i - 1) * outWidthStride;
        int32_t j = 0;
        for (; j <= inWidth - 32; j += 32) {
            prefetch(in + j * nc);
            prefetch(in + (j + 16) * nc);
            uint8x8_t vdata0 = vld1_u8(in + (j + 0) * nc);
            uint8x8_t vdata1 = vld1_u8(in + (j + 8) * nc);
            uint8x8_t vdata2 = vld1_u8(in + (j + 16) * nc);
            uint8x8_t vdata3 = vld1_u8(in + (j + 24) * nc);

            uint8x8_t vout3 = neon_reverse_u8x8(vdata0);
            uint8x8_t vout2 = neon_reverse_u8x8(vdata1);
            uint8x8_t vout1 = neon_reverse_u8x8(vdata2);
            uint8x8_t vout0 = neon_reverse_u8x8(vdata3);

            vst1_u8(out + (outWidth - j - 1 - 31) * nc, vout0);
            vst1_u8(out + (outWidth - j - 1 - 23) * nc, vout1);
            vst1_u8(out + (outWidth - j - 1 - 15) * nc, vout2);
            vst1_u8(out + (outWidth - j - 1 - 7) * nc, vout3);
        }
        for (; j <= inWidth - 8; j += 8) {
            uint8x8_t vdata0 = vld1_u8(in + (j + 0) * nc);
            uint8x8_t vout0 = neon_reverse_u8x8(vdata0);
            vst1_u8(out + (outWidth - j - 1 - 7) * nc, vout0);
        }
        for (; j < inWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[(outHeight - i - 1) * outWidthStride + (outWidth - j - 1) * nc + c] =
                    inData[i * inWidthStride + j * nc + c];
            }
        }
    }
}

template <>
void imgRotate180degree<uint8_t, 3>(int32_t inHeight,
                                    int32_t inWidth,
                                    int32_t inWidthStride,
                                    const uint8_t* inData,
                                    int32_t outHeight,
                                    int32_t outWidth,
                                    int32_t outWidthStride,
                                    uint8_t* outData)
{
    constexpr int nc = 3;
    for (int32_t i = 0; i < inHeight; i++) {
        const uint8_t* in = inData + i * inWidthStride;
        uint8_t* out = outData + (outHeight - i - 1) * outWidthStride;
        int32_t j = 0;
        for (; j <= inWidth - 32; j += 32) {
            prefetch(in + j * nc);
            prefetch(in + (j + 8) * nc);
            prefetch(in + (j + 16) * nc);
            prefetch(in + (j + 24) * nc);
            uint8x8x3_t vdata0 = vld3_u8(in + (j + 0) * nc);
            uint8x8x3_t vdata1 = vld3_u8(in + (j + 8) * nc);
            uint8x8x3_t vdata2 = vld3_u8(in + (j + 16) * nc);
            uint8x8x3_t vdata3 = vld3_u8(in + (j + 24) * nc);

            uint8x8x3_t vout3, vout2, vout1, vout0;
            vout3.val[0] = neon_reverse_u8x8(vdata0.val[0]);
            vout3.val[1] = neon_reverse_u8x8(vdata0.val[1]);
            vout3.val[2] = neon_reverse_u8x8(vdata0.val[2]);
            vout2.val[0] = neon_reverse_u8x8(vdata1.val[0]);
            vout2.val[1] = neon_reverse_u8x8(vdata1.val[1]);
            vout2.val[2] = neon_reverse_u8x8(vdata1.val[2]);
            vout1.val[0] = neon_reverse_u8x8(vdata2.val[0]);
            vout1.val[1] = neon_reverse_u8x8(vdata2.val[1]);
            vout1.val[2] = neon_reverse_u8x8(vdata2.val[2]);
            vout0.val[0] = neon_reverse_u8x8(vdata3.val[0]);
            vout0.val[1] = neon_reverse_u8x8(vdata3.val[1]);
            vout0.val[2] = neon_reverse_u8x8(vdata3.val[2]);

            vst3_u8(out + (outWidth - j - 1 - 31) * nc, vout0);
            vst3_u8(out + (outWidth - j - 1 - 23) * nc, vout1);
            vst3_u8(out + (outWidth - j - 1 - 15) * nc, vout2);
            vst3_u8(out + (outWidth - j - 1 - 7) * nc, vout3);
        }
        for (; j <= inWidth - 8; j += 8) {
            uint8x8x3_t vdata0 = vld3_u8(in + (j + 0) * nc);
            uint8x8x3_t vout0;
            vout0.val[0] = neon_reverse_u8x8(vdata0.val[0]);
            vout0.val[1] = neon_reverse_u8x8(vdata0.val[1]);
            vout0.val[2] = neon_reverse_u8x8(vdata0.val[2]);
            vst3_u8(out + (outWidth - j - 1 - 7) * nc, vout0);
        }
        for (; j < inWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[(outHeight - i - 1) * outWidthStride + (outWidth - j - 1) * nc + c] =
                    inData[i * inWidthStride + j * nc + c];
            }
        }
    }
}

// Applies to perfect square blocks only
template <typename T, int CHANNELS, int OUT_BLOCKDIM_R, int OUT_BLOCKDIM_C>
inline void rotate270_inner_block(const T* src,
                                  int obi,
                                  int obj,
                                  int outHeight,
                                  int outWidth,
                                  int inWidthStride,
                                  int outWidthStride,
                                  T* dst)
{
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i++) {
        prefetch(src + i * inWidthStride + obj * OUT_BLOCKDIM_C * CHANNELS, 0);
    }
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i++) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j++) {
            for (int c = 0; c < CHANNELS; ++c) {
                dst[(outHeight - 1 - j) * outWidthStride + i * CHANNELS + c] =
                    src[i * inWidthStride + j * CHANNELS + c];
            }
        }
    }
}

template <>
inline void rotate270_inner_block<float, 1, 32, 32>(const float* src,
                                                    int obi,
                                                    int obj,
                                                    int outHeight,
                                                    int outWidth,
                                                    int inWidthStride,
                                                    int outWidthStride,
                                                    float* dst)
{
    constexpr int CHANNELS = 1;
    constexpr int OUT_BLOCKDIM_R = 32;
    constexpr int OUT_BLOCKDIM_C = 32;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            float32x4_t va = vld1q_f32(src + (i + 0) * inWidthStride + j * CHANNELS);
            float32x4_t vb = vld1q_f32(src + (i + 1) * inWidthStride + j * CHANNELS);
            float32x4_t vc = vld1q_f32(src + (i + 2) * inWidthStride + j * CHANNELS);
            float32x4_t vd = vld1q_f32(src + (i + 3) * inWidthStride + j * CHANNELS);
            neon_rotate270_f32_4x4(va, vb, vc, vd);
            vst1q_f32(dst + (outHeight - 1 - j - 3) * outWidthStride + i * CHANNELS, va);
            vst1q_f32(dst + (outHeight - 1 - j - 2) * outWidthStride + i * CHANNELS, vb);
            vst1q_f32(dst + (outHeight - 1 - j - 1) * outWidthStride + i * CHANNELS, vc);
            vst1q_f32(dst + (outHeight - 1 - j - 0) * outWidthStride + i * CHANNELS, vd);
        }
    }
}

template <>
inline void rotate270_inner_block<float, 3, 32, 32>(const float* src,
                                                    int obi,
                                                    int obj,
                                                    int outHeight,
                                                    int outWidth,
                                                    int inWidthStride,
                                                    int outWidthStride,
                                                    float* dst)
{
    constexpr int CHANNELS = 3;
    constexpr int OUT_BLOCKDIM_R = 32;
    constexpr int OUT_BLOCKDIM_C = 32;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 0) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            float32x4x3_t va = vld3q_f32(src + (i + 0) * inWidthStride + j * CHANNELS);
            float32x4x3_t vb = vld3q_f32(src + (i + 1) * inWidthStride + j * CHANNELS);
            float32x4x3_t vc = vld3q_f32(src + (i + 2) * inWidthStride + j * CHANNELS);
            float32x4x3_t vd = vld3q_f32(src + (i + 3) * inWidthStride + j * CHANNELS);
            neon_rotate270_f32_4x4(va.val[0], vb.val[0], vc.val[0], vd.val[0]);
            neon_rotate270_f32_4x4(va.val[1], vb.val[1], vc.val[1], vd.val[1]);
            neon_rotate270_f32_4x4(va.val[2], vb.val[2], vc.val[2], vd.val[2]);
            vst3q_f32(dst + (outHeight - 1 - j - 3) * outWidthStride + i * CHANNELS, va);
            vst3q_f32(dst + (outHeight - 1 - j - 2) * outWidthStride + i * CHANNELS, vb);
            vst3q_f32(dst + (outHeight - 1 - j - 1) * outWidthStride + i * CHANNELS, vc);
            vst3q_f32(dst + (outHeight - 1 - j - 0) * outWidthStride + i * CHANNELS, vd);
        }
    }
}

template <>
inline void rotate270_inner_block<float, 4, 32, 32>(const float* src,
                                                    int obi,
                                                    int obj,
                                                    int outHeight,
                                                    int outWidth,
                                                    int inWidthStride,
                                                    int outWidthStride,
                                                    float* dst)
{
    constexpr int CHANNELS = 4;
    constexpr int OUT_BLOCKDIM_R = 32;
    constexpr int OUT_BLOCKDIM_C = 32;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 0) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 1) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 2) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            prefetch(src + (i + 3) * inWidthStride + (j + 2) * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(float));
            float32x4x4_t va = vld4q_f32(src + (i + 0) * inWidthStride + j * CHANNELS);
            float32x4x4_t vb = vld4q_f32(src + (i + 1) * inWidthStride + j * CHANNELS);
            float32x4x4_t vc = vld4q_f32(src + (i + 2) * inWidthStride + j * CHANNELS);
            float32x4x4_t vd = vld4q_f32(src + (i + 3) * inWidthStride + j * CHANNELS);
            neon_rotate270_f32_4x4(va.val[0], vb.val[0], vc.val[0], vd.val[0]);
            neon_rotate270_f32_4x4(va.val[1], vb.val[1], vc.val[1], vd.val[1]);
            neon_rotate270_f32_4x4(va.val[2], vb.val[2], vc.val[2], vd.val[2]);
            neon_rotate270_f32_4x4(va.val[3], vb.val[3], vc.val[3], vd.val[3]);
            vst4q_f32(dst + (outHeight - 1 - j - 3) * outWidthStride + i * CHANNELS, va);
            vst4q_f32(dst + (outHeight - 1 - j - 2) * outWidthStride + i * CHANNELS, vb);
            vst4q_f32(dst + (outHeight - 1 - j - 1) * outWidthStride + i * CHANNELS, vc);
            vst4q_f32(dst + (outHeight - 1 - j - 0) * outWidthStride + i * CHANNELS, vd);
        }
    }
}

template <>
inline void rotate270_inner_block<uint8_t, 1, 64, 64>(const uint8_t* src,
                                                      int obi,
                                                      int obj,
                                                      int outHeight,
                                                      int outWidth,
                                                      int inWidthStride,
                                                      int outWidthStride,
                                                      uint8_t* dst)
{
    constexpr int CHANNELS = 1;
    constexpr int OUT_BLOCKDIM_R = 64;
    constexpr int OUT_BLOCKDIM_C = 64;
    constexpr int INNER_BLOCKDIM = 8;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 4) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 5) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 6) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 7) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            uint8x8_t va = vld1_u8(src + (i + 0) * inWidthStride + j * CHANNELS);
            uint8x8_t vb = vld1_u8(src + (i + 1) * inWidthStride + j * CHANNELS);
            uint8x8_t vc = vld1_u8(src + (i + 2) * inWidthStride + j * CHANNELS);
            uint8x8_t vd = vld1_u8(src + (i + 3) * inWidthStride + j * CHANNELS);
            uint8x8_t ve = vld1_u8(src + (i + 4) * inWidthStride + j * CHANNELS);
            uint8x8_t vf = vld1_u8(src + (i + 5) * inWidthStride + j * CHANNELS);
            uint8x8_t vg = vld1_u8(src + (i + 6) * inWidthStride + j * CHANNELS);
            uint8x8_t vh = vld1_u8(src + (i + 7) * inWidthStride + j * CHANNELS);
            neon_rotate270_u8_8x8(va, vb, vc, vd, ve, vf, vg, vh);
            vst1_u8(dst + (outHeight - 1 - j - 7) * outWidthStride + i * CHANNELS, va);
            vst1_u8(dst + (outHeight - 1 - j - 6) * outWidthStride + i * CHANNELS, vb);
            vst1_u8(dst + (outHeight - 1 - j - 5) * outWidthStride + i * CHANNELS, vc);
            vst1_u8(dst + (outHeight - 1 - j - 4) * outWidthStride + i * CHANNELS, vd);
            vst1_u8(dst + (outHeight - 1 - j - 3) * outWidthStride + i * CHANNELS, ve);
            vst1_u8(dst + (outHeight - 1 - j - 2) * outWidthStride + i * CHANNELS, vf);
            vst1_u8(dst + (outHeight - 1 - j - 1) * outWidthStride + i * CHANNELS, vg);
            vst1_u8(dst + (outHeight - 1 - j - 0) * outWidthStride + i * CHANNELS, vh);
        }
    }
}

template <>
inline void rotate270_inner_block<uint8_t, 3, 64, 64>(const uint8_t* src,
                                                      int obi,
                                                      int obj,
                                                      int outHeight,
                                                      int outWidth,
                                                      int inWidthStride,
                                                      int outWidthStride,
                                                      uint8_t* dst)
{
    constexpr int CHANNELS = 3;
    constexpr int OUT_BLOCKDIM_R = 64;
    constexpr int OUT_BLOCKDIM_C = 64;
    constexpr int INNER_BLOCKDIM = 8;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 4) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 5) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 6) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 7) * inWidthStride + j * CHANNELS, 4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            uint8x8x3_t va = vld3_u8(src + (i + 0) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vb = vld3_u8(src + (i + 1) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vc = vld3_u8(src + (i + 2) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vd = vld3_u8(src + (i + 3) * inWidthStride + j * CHANNELS);
            uint8x8x3_t ve = vld3_u8(src + (i + 4) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vf = vld3_u8(src + (i + 5) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vg = vld3_u8(src + (i + 6) * inWidthStride + j * CHANNELS);
            uint8x8x3_t vh = vld3_u8(src + (i + 7) * inWidthStride + j * CHANNELS);
            neon_rotate270_u8_8x8(
                va.val[0], vb.val[0], vc.val[0], vd.val[0], ve.val[0], vf.val[0], vg.val[0], vh.val[0]);
            neon_rotate270_u8_8x8(
                va.val[1], vb.val[1], vc.val[1], vd.val[1], ve.val[1], vf.val[1], vg.val[1], vh.val[1]);
            neon_rotate270_u8_8x8(
                va.val[2], vb.val[2], vc.val[2], vd.val[2], ve.val[2], vf.val[2], vg.val[2], vh.val[2]);
            vst3_u8(dst + (outHeight - 1 - j - 7) * outWidthStride + i * CHANNELS, va);
            vst3_u8(dst + (outHeight - 1 - j - 6) * outWidthStride + i * CHANNELS, vb);
            vst3_u8(dst + (outHeight - 1 - j - 5) * outWidthStride + i * CHANNELS, vc);
            vst3_u8(dst + (outHeight - 1 - j - 4) * outWidthStride + i * CHANNELS, vd);
            vst3_u8(dst + (outHeight - 1 - j - 3) * outWidthStride + i * CHANNELS, ve);
            vst3_u8(dst + (outHeight - 1 - j - 2) * outWidthStride + i * CHANNELS, vf);
            vst3_u8(dst + (outHeight - 1 - j - 1) * outWidthStride + i * CHANNELS, vg);
            vst3_u8(dst + (outHeight - 1 - j - 0) * outWidthStride + i * CHANNELS, vh);
        }
    }
}

template <>
inline void rotate270_inner_block<uint8_t, 4, 64, 64>(const uint8_t* src,
                                                      int obi,
                                                      int obj,
                                                      int outHeight,
                                                      int outWidth,
                                                      int inWidthStride,
                                                      int outWidthStride,
                                                      uint8_t* dst)
{
    // little trick here: use uint32_t for uint8_t * 4 channel to avoid register overflow
    constexpr int CHANNELS = 4;
    constexpr int OUT_BLOCKDIM_R = 64;
    constexpr int OUT_BLOCKDIM_C = 64;
    constexpr int INNER_BLOCKDIM = 4;
    // assert((OUT_BLOCKDIM_R % INNER_BLOCKDIM == 0) && (OUT_BLOCKDIM_C % INNER_BLOCKDIM == 0))
    for (int i = obi * OUT_BLOCKDIM_R; i < (obi + 1) * OUT_BLOCKDIM_R; i += INNER_BLOCKDIM) {
        for (int j = obj * OUT_BLOCKDIM_C; j < (obj + 1) * OUT_BLOCKDIM_C; j += INNER_BLOCKDIM) {
            prefetch(src + (i + 0) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 0) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 1) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 2) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + (j + 0) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            prefetch(src + (i + 3) * inWidthStride + (j + 2) * CHANNELS,
                     4 * INNER_BLOCKDIM * CHANNELS * sizeof(uint8_t));
            // load 4 * 4channel pixel
            uint32x4_t va = vld1q_u32((uint32_t*)(src + (i + 0) * inWidthStride + j * CHANNELS));
            uint32x4_t vb = vld1q_u32((uint32_t*)(src + (i + 1) * inWidthStride + j * CHANNELS));
            uint32x4_t vc = vld1q_u32((uint32_t*)(src + (i + 2) * inWidthStride + j * CHANNELS));
            uint32x4_t vd = vld1q_u32((uint32_t*)(src + (i + 3) * inWidthStride + j * CHANNELS));
            // transpose 4channel pixel as an uint32_t
            neon_rotate270_u32_4x4(va, vb, vc, vd);
            // store 4 * 4channel pixel
            vst1q_u32((uint32_t*)(dst + (outHeight - 1 - j - 3) * outWidthStride + i * CHANNELS), va);
            vst1q_u32((uint32_t*)(dst + (outHeight - 1 - j - 2) * outWidthStride + i * CHANNELS), vb);
            vst1q_u32((uint32_t*)(dst + (outHeight - 1 - j - 1) * outWidthStride + i * CHANNELS), vc);
            vst1q_u32((uint32_t*)(dst + (outHeight - 1 - j - 0) * outWidthStride + i * CHANNELS), vd);
        }
    }
}

template <typename T, int32_t CHANNELS, int32_t BLOCKDIM_R, int32_t BLOCKDIM_C>
void imgRotate270degree(int32_t inHeight,
                        int32_t inWidth,
                        int32_t inWidthStride,
                        const T* src,
                        int32_t outHeight,
                        int32_t outWidth,
                        int32_t outWidthStride,
                        T* dst)
{
    // After outer block the image is splitted into 4 regions
    // Region A are perfect blocks, B and C are partially perfect blocks, and D is what remains
    // |           |   |
    // |     A     | B |
    // |___________|___|
    // |     C     | D |

    // Part A
    int row_blks = inHeight / BLOCKDIM_R;
    int col_blks = inWidth / BLOCKDIM_C;
    for (int obi = 0; obi < row_blks; ++obi) {
        for (int obj = 0; obj < col_blks; ++obj) {
            rotate270_inner_block<T, CHANNELS, BLOCKDIM_R, BLOCKDIM_C>(
                src, obi, obj, outHeight, outWidth, inWidthStride, outWidthStride, dst);
        }
    }
    // Part B
    for (int obi = 0; obi < row_blks; ++obi) {
        for (int i = obi * BLOCKDIM_R; i < (obi + 1) * BLOCKDIM_R; i++)
            prefetch(src + (i + 1) * inWidthStride + (col_blks - 1) * BLOCKDIM_C * CHANNELS, 0);
        for (int i = obi * BLOCKDIM_R; i < (obi + 1) * BLOCKDIM_R; i++) {
            for (int j = (col_blks - 1) * BLOCKDIM_C; j < inWidth; j++) {
                for (int c = 0; c < CHANNELS; ++c) {
                    dst[(outHeight - 1 - j) * outWidthStride + i * CHANNELS + c] =
                        src[i * inWidthStride + j * CHANNELS + c];
                }
            }
        }
    }
    // Part C
    for (int obj = 0; obj < col_blks; ++obj) {
        for (int i = (row_blks - 1) * BLOCKDIM_R; i < inHeight; i++) {
            prefetch(src + i * inWidthStride + (obj + 1) * BLOCKDIM_C * CHANNELS, 0);
            for (int j = obj * BLOCKDIM_C; j < (obj + 1) * BLOCKDIM_C; j++) {
                for (int c = 0; c < CHANNELS; ++c) {
                    dst[(outHeight - 1 - j) * outWidthStride + i * CHANNELS + c] =
                        src[i * inWidthStride + j * CHANNELS + c];
                }
            }
        }
    }
    // Part D
    for (int i = (row_blks - 1) * BLOCKDIM_R; i < inHeight; i++) {
        for (int j = (col_blks - 1) * BLOCKDIM_C; j < inWidth; j++) {
            for (int c = 0; c < CHANNELS; ++c) {
                dst[(outHeight - 1 - j) * outWidthStride + i * CHANNELS + c] =
                    src[i * inWidthStride + j * CHANNELS + c];
            }
        }
    }
    return;
}

template <typename T, int32_t CHANNELS>
::ppl::common::RetCode imgRotate(int32_t inHeight,
                                 int32_t inWidth,
                                 int32_t inWidthStride,
                                 const T* inData,
                                 int32_t outHeight,
                                 int32_t outWidth,
                                 int32_t outWidthStride,
                                 T* outData,
                                 int32_t degree)
{
    constexpr int BLK_SIZE = std::is_same<T, float>::value ? 32 : 64;
    if (degree == 90) {
        imgRotate90degree<T, CHANNELS, BLK_SIZE, BLK_SIZE>(
            inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    } else if (degree == 180) {
        imgRotate180degree<T, CHANNELS>(
            inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    } else if (degree == 270) {
        imgRotate270degree<T, CHANNELS, BLK_SIZE, BLK_SIZE>(
            inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Rotate<float, 1>(int32_t inHeight,
                                        int32_t inWidth,
                                        int32_t inWidthStride,
                                        const float* inData,
                                        int32_t outHeight,
                                        int32_t outWidth,
                                        int32_t outWidthStride,
                                        float* outData,
                                        int32_t degree)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (inHeight == 0 || inWidth == 0 || inWidthStride == 0 || outHeight == 0 || outWidth == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (degree != 90 && degree != 180 && degree != 270) { return ppl::common::RC_INVALID_VALUE; }

    imgRotate<float, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Rotate<float, 3>(int32_t inHeight,
                                        int32_t inWidth,
                                        int32_t inWidthStride,
                                        const float* inData,
                                        int32_t outHeight,
                                        int32_t outWidth,
                                        int32_t outWidthStride,
                                        float* outData,
                                        int32_t degree)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (inHeight == 0 || inWidth == 0 || inWidthStride == 0 || outHeight == 0 || outWidth == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (degree != 90 && degree != 180 && degree != 270) { return ppl::common::RC_INVALID_VALUE; }

    imgRotate<float, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Rotate<float, 4>(int32_t inHeight,
                                        int32_t inWidth,
                                        int32_t inWidthStride,
                                        const float* inData,
                                        int32_t outHeight,
                                        int32_t outWidth,
                                        int32_t outWidthStride,
                                        float* outData,
                                        int32_t degree)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (inHeight == 0 || inWidth == 0 || inWidthStride == 0 || outHeight == 0 || outWidth == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (degree != 90 && degree != 180 && degree != 270) { return ppl::common::RC_INVALID_VALUE; }

    imgRotate<float, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Rotate<uint8_t, 1>(int32_t inHeight,
                                          int32_t inWidth,
                                          int32_t inWidthStride,
                                          const uint8_t* inData,
                                          int32_t outHeight,
                                          int32_t outWidth,
                                          int32_t outWidthStride,
                                          uint8_t* outData,
                                          int32_t degree)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (inHeight == 0 || inWidth == 0 || inWidthStride == 0 || outHeight == 0 || outWidth == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (degree != 90 && degree != 180 && degree != 270) { return ppl::common::RC_INVALID_VALUE; }

    imgRotate<uint8_t, 1>(
        inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Rotate<uint8_t, 3>(int32_t inHeight,
                                          int32_t inWidth,
                                          int32_t inWidthStride,
                                          const uint8_t* inData,
                                          int32_t outHeight,
                                          int32_t outWidth,
                                          int32_t outWidthStride,
                                          uint8_t* outData,
                                          int32_t degree)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (inHeight == 0 || inWidth == 0 || inWidthStride == 0 || outHeight == 0 || outWidth == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (degree != 90 && degree != 180 && degree != 270) { return ppl::common::RC_INVALID_VALUE; }

    imgRotate<uint8_t, 3>(
        inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Rotate<uint8_t, 4>(int32_t inHeight,
                                          int32_t inWidth,
                                          int32_t inWidthStride,
                                          const uint8_t* inData,
                                          int32_t outHeight,
                                          int32_t outWidth,
                                          int32_t outWidthStride,
                                          uint8_t* outData,
                                          int32_t degree)
{
    if (nullptr == inData) { return ppl::common::RC_INVALID_VALUE; }
    if (nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (inHeight == 0 || inWidth == 0 || inWidthStride == 0 || outHeight == 0 || outWidth == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (degree != 90 && degree != 180 && degree != 270) { return ppl::common::RC_INVALID_VALUE; }

    imgRotate<uint8_t, 4>(
        inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, degree);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
