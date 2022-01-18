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
#include "ppl/common/retcode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <float.h>
#include <limits.h>
#include "ppl/cv/arm/resize.h"
#include "ppl/cv/types.h"
#include "operation_utils.hpp"
#include <vector>

namespace ppl {
namespace cv {
namespace arm {

template <typename Tsrc, typename Tdst, int32_t nc>
void resizeNearestPoint(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const Tsrc* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    Tdst* outData)
{
    int32_t x, y;
    int32_t* x_ofs = (int32_t*)malloc(outWidth * sizeof(int32_t));
    double fx = (double)outWidth / inWidth;
    double fy = (double)outHeight / inHeight;
    double ifx = 1. / fx;
    double ify = 1. / fy;
    int32_t pix_size = nc;
    for (x = 0; x < outWidth; x++) {
        int32_t sx = img_floor(x * ifx);
        x_ofs[x] = std::min(sx, inWidth - 1) * pix_size;
    }
    for (y = 0; y < outHeight; y++) {
        Tdst* D = outData + y * outWidthStride;
        int32_t sy = std::min(int32_t(y * ify), inHeight - 1);
        const Tsrc* S = inData + sy * inWidthStride;
        for (x = 0; x < outWidth; x++) {
            for (int32_t i = 0; i < nc; i++) {
                Tsrc t0 = S[x_ofs[x] + i];
                D[x * nc + i] = (Tdst)t0;
            }
        }
    }
    free(x_ofs);
}

template <>
void resizeNearestPoint<uint8_t, uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData) // resize_nearest_u8c1
{
    int32_t x, y;
    int32_t* x_ofs = (int32_t*)malloc(outWidth * sizeof(int32_t));
    double fx = (double)outWidth / inWidth;
    double fy = (double)outHeight / inHeight;
    double ifx = 1.0f / fx;
    double ify = 1.0f / fy;
    for (x = 0; x < outWidth; x++) {
        int32_t sx = img_floor(x * ifx);
        x_ofs[x] = std::min(sx, inWidth - 1);
    }
    for (y = 0; y + 4 <= outHeight; y += 4) {
        uint8_t* D0 = outData + y * outWidthStride;
        uint8_t* D1 = outData + (y + 1) * outWidthStride;
        uint8_t* D2 = outData + (y + 2) * outWidthStride;
        uint8_t* D3 = outData + (y + 3) * outWidthStride;
        int32_t sy0 = std::min(int32_t(y * ify), inHeight - 1);
        int32_t sy1 = std::min(int32_t((y + 1) * ify), inHeight - 1);
        int32_t sy2 = std::min(int32_t((y + 2) * ify), inHeight - 1);
        int32_t sy3 = std::min(int32_t((y + 3) * ify), inHeight - 1);
        const uint8_t* S0 = inData + sy0 * inWidthStride;
        const uint8_t* S1 = inData + sy1 * inWidthStride;
        const uint8_t* S2 = inData + sy2 * inWidthStride;
        const uint8_t* S3 = inData + sy3 * inWidthStride;
        for (x = 0; x + 4 <= outWidth; x += 4) {
            int32_t x0 = x_ofs[x];
            int32_t x1 = x_ofs[x + 1];
            int32_t x2 = x_ofs[x + 2];
            int32_t x3 = x_ofs[x + 3];
            D0[x] = S0[x0], D0[x + 1] = S0[x1], D0[x + 2] = S0[x2], D0[x + 3] = S0[x3];
            D1[x] = S1[x0], D1[x + 1] = S1[x1], D1[x + 2] = S1[x2], D1[x + 3] = S1[x3];
            D2[x] = S2[x0], D2[x + 1] = S2[x1], D2[x + 2] = S2[x2], D2[x + 3] = S2[x3];
            D3[x] = S3[x0], D3[x + 1] = S3[x1], D3[x + 2] = S3[x2], D3[x + 3] = S3[x3];
        }
        for (; x < outWidth; x++) {
            int32_t x0 = x_ofs[x];
            D0[x] = S0[x0], D1[x] = S1[x0], D2[x] = S2[x0], D3[x] = S3[x0];
        }
    }
    for (; y < outHeight; y++) {
        uint8_t* D = outData + y * outWidthStride;
        int32_t sy = std::min(int32_t(y * ify), inHeight - 1);
        const uint8_t* S = inData + sy * inWidthStride;
        for (x = 0; x + 4 <= outWidth; x += 4) {
            int32_t x0 = x_ofs[x];
            int32_t x1 = x_ofs[x + 1];
            int32_t x2 = x_ofs[x + 2];
            int32_t x3 = x_ofs[x + 3];
            D[x] = S[x0], D[x + 1] = S[x1], D[x + 2] = S[x2], D[x + 3] = S[x3];
        }
        for (; x < outWidth; x++) {
            int32_t x0 = x_ofs[x];
            D[x] = S[x0];
        }
    }
    free(x_ofs);
}

template <>
void resizeNearestPoint<uint8_t, uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData) // resize_nearest_u8c3
{
    int32_t x, y;
    int32_t* x_ofs = (int32_t*)malloc(outWidth * sizeof(int32_t));
    double fx = (double)outWidth / inWidth;
    double fy = (double)outHeight / inHeight;
    double ifx = 1.0f / fx;
    double ify = 1.0f / fy;
    const int32_t nc = 3;
    for (x = 0; x < outWidth; x++) {
        int32_t sx = img_floor(x * ifx);
        x_ofs[x] = std::min(sx, inWidth - 1) * nc;
    }
    for (y = 0; y < outHeight; y++) {
        uint8_t* D = outData + y * outWidthStride;
        int32_t sy = std::min(int32_t(y * ify), inHeight - 1);
        const uint8_t* S = inData + sy * inWidthStride;
        for (x = 0; x + 4 <= outWidth; x += 4) {
            int32_t x0 = x_ofs[x];
            int32_t x1 = x_ofs[x + 1];
            int32_t x2 = x_ofs[x + 2];
            int32_t x3 = x_ofs[x + 3];
            int32_t xd = x * nc;
            if (x3 + 4 >= inWidth || xd + 12 >= outWidth) { // to avoid segment fault
                break;
            }
            *((int32_t*)(D + xd)) = *((int32_t*)(S + x0));
            *((int32_t*)(D + xd + 3)) = *((int32_t*)(S + x1));
            *((int32_t*)(D + xd + 6)) = *((int32_t*)(S + x2));
            *((int32_t*)(D + xd + 9)) = *((int32_t*)(S + x3));
        }
        for (; x < outWidth; x++) {
            int32_t x0 = x_ofs[x];
            int32_t xd = x * nc;
            D[xd + 0] = S[x0], D[xd + 1] = S[x0 + 1], D[xd + 2] = S[x0 + 2];
        }
    }
    free(x_ofs);
}

template <>
void resizeNearestPoint<uint8_t, uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData) // resize_nearest_u8c4
{
    int32_t x, y;
    int32_t* x_ofs = (int32_t*)malloc(outWidth * sizeof(int32_t));
    double fx = (double)outWidth / inWidth;
    double fy = (double)outHeight / inHeight;
    double ifx = 1.0f / fx;
    double ify = 1.0f / fy;
    const int32_t nc = 4;
    for (x = 0; x < outWidth; x++) {
        int32_t sx = img_floor(x * ifx);
        x_ofs[x] = std::min(sx, inWidth - 1) * nc;
    }
    for (y = 0; y < outHeight; y++) {
        uint8_t* D = outData + y * outWidthStride;
        int32_t sy = std::min(int32_t(y * ify), inHeight - 1);
        const uint8_t* S = inData + sy * inWidthStride;
        for (x = 0; x + 4 <= outWidth; x += 4) {
            int32_t x0 = x_ofs[x];
            int32_t x1 = x_ofs[x + 1];
            int32_t x2 = x_ofs[x + 2];
            int32_t x3 = x_ofs[x + 3];
            int32_t xd = x * nc;
            *((int32_t*)(D + xd)) = *((int32_t*)(S + x0));
            *((int32_t*)(D + xd + 4)) = *((int32_t*)(S + x1));
            *((int32_t*)(D + xd + 8)) = *((int32_t*)(S + x2));
            *((int32_t*)(D + xd + 12)) = *((int32_t*)(S + x3));
        }
        for (; x < outWidth; x++) {
            int32_t x0 = x_ofs[x];
            int32_t xd = x * nc;
            *((int32_t*)(D + xd)) = *((int32_t*)(S + x0));
        }
    }
    free(x_ofs);
}

bool img_resize_bilinear_neon_shrink2_u8(
    uint8_t* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const uint8_t* src,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t src_stride,
    uint32_t channels)
{
    if (src_width % 2 != 0 || dst_width != src_width / 2 ||
        src_height % 2 != 0 || dst_height != src_height / 2) {
        return false;
    }

    int32_t dstw = dst_width;
    int32_t dsth = dst_height;
    int32_t cn = channels;
    if (channels == 1) {
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (2 * i) * src_stride;
            const uint8_t* row2 = src + (2 * i + 1) * src_stride;
            int32_t j = 0;
            for (; j <= dstw - 8; j += 8) {
                uint8x8x2_t q0 = vld2_u8(row1 + 2 * j);
                prefetch_l1(row1, j * 2 + 256);
                uint8x8x2_t q1 = vld2_u8(row2 + 2 * j);
                prefetch_l1(row2, j * 2 + 256);
                uint8x8_t q00 = q0.val[0];
                uint8x8_t q01 = q0.val[1];
                uint8x8_t q10 = q1.val[0];
                uint8x8_t q11 = q1.val[1];
                uint16x8_t res_u16 = vaddq_u16(vaddq_u16(vaddl_u8(q00, q01), vaddl_u8(q10, q11)), vdupq_n_u16(2));
                uint8x8_t res_u8 = vqmovn_u16(vshrq_n_u16(res_u16, 2));
                vst1_u8(dst + i * dst_stride + j, res_u8);
            }
            for (; j < dstw; j++) {
                dst[i * dst_stride + j] = (row1[j * 2 + 0] + row1[j * 2 + 1] +
                                           row2[j * 2 + 0] + row2[j * 2 + 1] + 2) >>
                                          2;
            }
        }
    } else if (cn == 3) {
        uint8x8_t tbl = {0, 1, 2, 4, 5, 6, 0, 0};
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (2 * i) * src_stride;
            const uint8_t* row2 = src + (2 * i + 1) * src_stride;
            int32_t j = 0;
            int32_t dstw_not_cross_boundary = dstw;
            if (i == dsth - 1) dstw_not_cross_boundary = dstw - 1;
            for (; j <= dstw_not_cross_boundary - 2; j += 2) {
                uint8x8_t q0 = vld1_u8(row1 + j * 6);
                prefetch_l1(row1, j * 6 + 256);
                uint8x8_t q1 = vld1_u8(row2 + j * 6);
                prefetch_l1(row2, j * 6 + 256);
                uint8x8_t q2 = vld1_u8(row1 + (j + 1) * 6);
                prefetch_l1(row1, (j + 1) * 6 + 256);
                uint8x8_t q3 = vld1_u8(row2 + (j + 1) * 6);
                prefetch_l1(row2, (j + 1) * 6 + 256);
                uint16x8_t q0_u16 = vmovl_u8(q0);
                uint16x8_t q1_u16 = vmovl_u8(q1);
                uint16x8_t q2_u16 = vmovl_u8(q2);
                uint16x8_t q3_u16 = vmovl_u8(q3);
                uint16x4_t q00 = vget_low_u16(q0_u16);
                uint16x4_t q01 = vget_low_u16(vextq_u16(q0_u16, q0_u16, 3));
                uint16x4_t q10 = vget_low_u16(q1_u16);
                uint16x4_t q11 = vget_low_u16(vextq_u16(q1_u16, q1_u16, 3));
                uint16x4_t q20 = vget_low_u16(q2_u16);
                uint16x4_t q21 = vget_low_u16(vextq_u16(q2_u16, q2_u16, 3));
                uint16x4_t q30 = vget_low_u16(q3_u16);
                uint16x4_t q31 = vget_low_u16(vextq_u16(q3_u16, q3_u16, 3));
                uint16x4_t res0_u16 = vadd_u16(vadd_u16(vadd_u16(q00, q01), vadd_u16(q10, q11)), vdup_n_u16(2));
                uint16x4_t res1_u16 = vadd_u16(vadd_u16(vadd_u16(q20, q21), vadd_u16(q30, q31)), vdup_n_u16(2));
                uint8x8_t res_u8 = vqmovn_u16(vcombine_u16(vshr_n_u16(res0_u16, 2), vshr_n_u16(res1_u16, 2)));
                uint8x8_t ans = vtbl1_u8(res_u8, tbl);
                asm volatile(
                    "st1 {%2.s}[0], [%0]\n\t"
                    "st1 {%2.h}[2], [%1]\n\t"
                    :
                    : "r"(dst + i * dst_stride + j * 3), "r"(dst + i * dst_stride + j * 3 + 4), "w"(ans)
                    : "cc", "memory");
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 2 * cn + c] + row1[j * 2 * cn + cn + c] +
                                                        row2[j * 2 * cn + c] + row2[j * 2 * cn + cn + c] + 2) >>
                                                       2;
                }
            }
        }
    } else if (cn == 4) {
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (2 * i) * src_stride;
            const uint8_t* row2 = src + (2 * i + 1) * src_stride;
            int32_t j = 0;
            for (; j <= dstw - 2; j += 2) {
                uint8x8_t q0 = vld1_u8(row1 + j * 8);
                prefetch_l1(row1, j * 8 + 256);
                uint8x8_t q1 = vld1_u8(row2 + j * 8);
                prefetch_l1(row2, j * 8 + 256);
                uint8x8_t q2 = vld1_u8(row1 + (j + 1) * 8);
                prefetch_l1(row1, (j + 1) * 8 + 256);
                uint8x8_t q3 = vld1_u8(row2 + (j + 1) * 8);
                prefetch_l1(row2, (j + 1) * 8 + 256);
                uint16x8_t q0_u16 = vmovl_u8(q0);
                uint16x8_t q1_u16 = vmovl_u8(q1);
                uint16x8_t q2_u16 = vmovl_u8(q2);
                uint16x8_t q3_u16 = vmovl_u8(q3);
                uint16x4_t q00 = vget_low_u16(q0_u16);
                uint16x4_t q01 = vget_high_u16(q0_u16);
                uint16x4_t q10 = vget_low_u16(q1_u16);
                uint16x4_t q11 = vget_high_u16(q1_u16);
                uint16x4_t q20 = vget_low_u16(q2_u16);
                uint16x4_t q21 = vget_high_u16(q2_u16);
                uint16x4_t q30 = vget_low_u16(q3_u16);
                uint16x4_t q31 = vget_high_u16(q3_u16);
                uint16x4_t res0_u16 = vadd_u16(vadd_u16(vadd_u16(q00, q01), vadd_u16(q10, q11)), vdup_n_u16(2));
                uint16x4_t res1_u16 = vadd_u16(vadd_u16(vadd_u16(q20, q21), vadd_u16(q30, q31)), vdup_n_u16(2));
                uint8x8_t res_u8 = vqmovn_u16(vcombine_u16(vshr_n_u16(res0_u16, 2), vshr_n_u16(res1_u16, 2)));
                vst1_u8(dst + i * dst_stride + j * 4, res_u8);
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 2 * cn + c] + row1[j * 2 * cn + cn + c] +
                                                        row2[j * 2 * cn + c] + row2[j * 2 * cn + cn + c] + 2) >>
                                                       2;
                }
            }
        }
    } else {
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (2 * i) * src_stride;
            const uint8_t* row2 = src + (2 * i + 1) * src_stride;
            int32_t j = 0;
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 2 * cn + c] + row1[j * 2 * cn + cn + c] +
                                                        row2[j * 2 * cn + c] + row2[j * 2 * cn + cn + c] + 2) >>
                                                       2;
                }
            }
        }
    }
    return true;
}
bool img_resize_bilinear_neon_shrink4_u8(
    uint8_t* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const uint8_t* src,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t src_stride,
    uint32_t channels)
{
    if (src_width % 4 != 0 || dst_width != src_width / 4 ||
        src_height % 4 != 0 || dst_height != src_height / 4) {
        return false;
    }

    int32_t dstw = dst_width;
    int32_t dsth = dst_height;
    int32_t cn = channels;

    if (cn == 1) {
        uint8_t* temp_buffer = (uint8_t*)malloc(dst_width * 2 * 2 * sizeof(uint8_t));

        uint16_t* row1_buffer = (uint16_t*)temp_buffer;
        uint16_t* row2_buffer = (uint16_t*)(temp_buffer + dst_width * 2);

        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (4 * i + 1) * src_stride;
            const uint8_t* row2 = src + (4 * i + 2) * src_stride;
            int32_t j = 0;
            // gather
            for (; j < dstw; j++) {
                row1_buffer[j] = *((uint16_t*)(row1 + j * 4 + 1));
                prefetch_l1(row1, j * 4 + 1 + 256);
                row2_buffer[j] = *((uint16_t*)(row2 + j * 4 + 1));
                prefetch_l1(row2, j * 4 + 1 + 256);
            }
            j = 0;
            uint8_t* row1_buffer_ptr = (uint8_t*)row1_buffer;
            uint8_t* row2_buffer_ptr = (uint8_t*)row2_buffer;
            for (; j <= dstw - 8; j += 8) {
                uint8x8x2_t q0 = vld2_u8(row1_buffer_ptr + 2 * j);
                uint8x8x2_t q1 = vld2_u8(row2_buffer_ptr + 2 * j);
                uint8x8_t q00 = q0.val[0];
                uint8x8_t q01 = q0.val[1];
                uint8x8_t q10 = q1.val[0];
                uint8x8_t q11 = q1.val[1];
                uint16x8_t res_u16 = vaddq_u16(vaddq_u16(vaddl_u8(q00, q01), vaddl_u8(q10, q11)), vdupq_n_u16(2));
                uint8x8_t res_u8 = vshrn_n_u16(res_u16, 2);
                vst1_u8(dst + i * dst_stride + j, res_u8);
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j] = (row1_buffer_ptr[j * 2 + 0] + row1_buffer_ptr[j * 2 + 1] +
                                               row2_buffer_ptr[j * 2 + 0] + row2_buffer_ptr[j * 2 + 1] + 2) >>
                                              2;
                }
            }
        }
        free(temp_buffer);
    } else if (cn == 3) {
        uint8x8_t tbl = {0, 1, 2, 4, 5, 6, 0, 0};
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (4 * i + 1) * src_stride;
            const uint8_t* row2 = src + (4 * i + 2) * src_stride;
            int32_t j = 0;
            int32_t dstw_not_cross_boundary = dstw;
            if (i == dsth - 1) dstw_not_cross_boundary = dstw - 1;
            for (; j <= dstw_not_cross_boundary - 2; j += 2) {
                uint8x8_t q0 = vld1_u8(row1 + j * 12 + 3);
                prefetch_l1(row1, j * 12 + 3 + 256);
                uint8x8_t q1 = vld1_u8(row2 + j * 12 + 3);
                prefetch_l1(row2, j * 12 + 3 + 256);
                uint8x8_t q2 = vld1_u8(row1 + (j + 1) * 12 + 3);
                prefetch_l1(row1, (j + 1) * 12 + 3 + 256);
                uint8x8_t q3 = vld1_u8(row2 + (j + 1) * 12 + 3);
                prefetch_l1(row2, (j + 1) * 12 + 3 + 256);
                uint16x8_t q0_u16 = vmovl_u8(q0);
                uint16x8_t q1_u16 = vmovl_u8(q1);
                uint16x8_t q2_u16 = vmovl_u8(q2);
                uint16x8_t q3_u16 = vmovl_u8(q3);
                uint16x4_t q00 = vget_low_u16(q0_u16);
                uint16x4_t q01 = vget_low_u16(vextq_u16(q0_u16, q0_u16, 3));
                uint16x4_t q10 = vget_low_u16(q1_u16);
                uint16x4_t q11 = vget_low_u16(vextq_u16(q1_u16, q1_u16, 3));
                uint16x4_t q20 = vget_low_u16(q2_u16);
                uint16x4_t q21 = vget_low_u16(vextq_u16(q2_u16, q2_u16, 3));
                uint16x4_t q30 = vget_low_u16(q3_u16);
                uint16x4_t q31 = vget_low_u16(vextq_u16(q3_u16, q3_u16, 3));
                uint16x4_t res0_u16 = vadd_u16(vadd_u16(vadd_u16(q00, q01), vadd_u16(q10, q11)), vdup_n_u16(2));
                uint16x4_t res1_u16 = vadd_u16(vadd_u16(vadd_u16(q20, q21), vadd_u16(q30, q31)), vdup_n_u16(2));
                uint8x8_t res_u8 = vqmovn_u16(vcombine_u16(vshr_n_u16(res0_u16, 2), vshr_n_u16(res1_u16, 2)));
                uint8x8_t ans = vtbl1_u8(res_u8, tbl);
                asm volatile(
                    "st1 {%2.s}[0], [%0]\n\t"
                    "st1 {%2.h}[2], [%1]\n\t"
                    :
                    : "r"(dst + i * dst_stride + j * 3), "r"(dst + i * dst_stride + j * 3 + 4), "w"(ans)
                    : "cc", "memory");
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 4 * cn + 1 * cn + c] + row1[j * 4 * cn + 2 * cn + c] +
                                                        row2[j * 4 * cn + 1 * cn + c] + row2[j * 4 * cn + 2 * cn + c] + 2) >>
                                                       2;
                }
            }
        }
    } else if (cn == 4) {
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (4 * i + 1) * src_stride;
            const uint8_t* row2 = src + (4 * i + 2) * src_stride;
            int32_t j = 0;
            for (; j <= dstw - 2; j += 2) {
                uint8x8_t q0 = vld1_u8(row1 + j * 16 + 4);
                prefetch_l1(row1, j * 16 + 4 + 256);
                uint8x8_t q2 = vld1_u8(row1 + (j + 1) * 16 + 4);
                prefetch_l1(row1, (j + 1) * 16 + 4 + 256);
                uint8x8_t q1 = vld1_u8(row2 + j * 16 + 4);
                prefetch_l1(row2, j * 16 + 4 + 256);
                uint8x8_t q3 = vld1_u8(row2 + (j + 1) * 16 + 4);
                prefetch_l1(row2, (j + 1) * 16 + 4 + 256);
                uint16x8_t q0_u16 = vmovl_u8(q0);
                uint16x8_t q1_u16 = vmovl_u8(q1);
                uint16x8_t q2_u16 = vmovl_u8(q2);
                uint16x8_t q3_u16 = vmovl_u8(q3);
                uint16x4_t q00 = vget_low_u16(q0_u16);
                uint16x4_t q01 = vget_high_u16(q0_u16);
                uint16x4_t q10 = vget_low_u16(q1_u16);
                uint16x4_t q11 = vget_high_u16(q1_u16);
                uint16x4_t q20 = vget_low_u16(q2_u16);
                uint16x4_t q21 = vget_high_u16(q2_u16);
                uint16x4_t q30 = vget_low_u16(q3_u16);
                uint16x4_t q31 = vget_high_u16(q3_u16);
                uint16x4_t res0_u16 = vadd_u16(vadd_u16(vadd_u16(q00, q01), vadd_u16(q10, q11)), vdup_n_u16(2));
                uint16x4_t res1_u16 = vadd_u16(vadd_u16(vadd_u16(q20, q21), vadd_u16(q30, q31)), vdup_n_u16(2));
                uint8x8_t res_u8 = vshrn_n_u16(vcombine_u16(res0_u16, res1_u16), 2);
                vst1_u8(dst + i * dst_stride + j * 4, res_u8);
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 4 * cn + 1 * cn + c] + row1[j * 4 * cn + 2 * cn + c] +
                                                        row2[j * 4 * cn + 1 * cn + c] + row2[j * 4 * cn + 2 * cn + c] + 2) >>
                                                       2;
                }
            }
        }
    } else {
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (4 * i + 1) * src_stride;
            const uint8_t* row2 = src + (4 * i + 2) * src_stride;
            int32_t j = 0;
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 4 * cn + 1 * cn + c] + row1[j * 4 * cn + 2 * cn + c] +
                                                        row2[j * 4 * cn + 1 * cn + c] + row2[j * 4 * cn + 2 * cn + c] + 2) >>
                                                       2;
                }
            }
        }
    }

    return true;
}

bool img_resize_bilinear_neon_shrink6_u8(
    uint8_t* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const uint8_t* src,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t src_stride,
    uint32_t channels)
{
    if (src_width % 6 != 0 || dst_width != src_width / 6 ||
        src_height % 6 != 0 || dst_height != src_height / 6) {
        return false;
    }

    int32_t dstw = dst_width;
    int32_t dsth = dst_height;
    int32_t cn = channels;

    if (cn == 1) {
        uint8_t* temp_buffer = (uint8_t*)malloc(dst_width * 2 * 2 * sizeof(uint8_t));

        uint16_t* row1_buffer = (uint16_t*)temp_buffer;
        uint16_t* row2_buffer = (uint16_t*)(temp_buffer + dst_width * 2);

        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (6 * i + 2) * src_stride;
            const uint8_t* row2 = src + (6 * i + 3) * src_stride;
            int32_t j = 0;
            // gather
            for (; j < dstw; j++) {
                row1_buffer[j] = *((uint16_t*)(row1 + j * 6 + 2));
                prefetch_l1(row1, j * 6 + 2 + 256);
                row2_buffer[j] = *((uint16_t*)(row2 + j * 6 + 2));
                prefetch_l1(row2, j * 6 + 2 + 256);
            }
            j = 0;
            uint8_t* row1_buffer_ptr = (uint8_t*)row1_buffer;
            uint8_t* row2_buffer_ptr = (uint8_t*)row2_buffer;
            for (; j <= dstw - 8; j += 8) {
                uint8x8x2_t q0 = vld2_u8(row1_buffer_ptr + 2 * j);
                uint8x8x2_t q1 = vld2_u8(row2_buffer_ptr + 2 * j);
                uint8x8_t q00 = q0.val[0];
                uint8x8_t q01 = q0.val[1];
                uint8x8_t q10 = q1.val[0];
                uint8x8_t q11 = q1.val[1];
                uint16x8_t res_u16 = vaddq_u16(vaddq_u16(vaddl_u8(q00, q01), vaddl_u8(q10, q11)), vdupq_n_u16(2));
                uint8x8_t res_u8 = vshrn_n_u16(res_u16, 2);
                vst1_u8(dst + i * dst_stride + j, res_u8);
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j] = (row1_buffer_ptr[j * 2 + 0] + row1_buffer_ptr[j * 2 + 1] +
                                               row2_buffer_ptr[j * 2 + 0] + row2_buffer_ptr[j * 2 + 1] + 2) >>
                                              2;
                }
            }
        }
        free(temp_buffer);
    } else if (cn == 3) {
        uint8x8_t tbl = {0, 1, 2, 4, 5, 6, 0, 0};
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (6 * i + 2) * src_stride;
            const uint8_t* row2 = src + (6 * i + 3) * src_stride;
            int32_t j = 0;
            int32_t dstw_not_cross_boundary = dstw;
            if (i == dsth - 1) dstw_not_cross_boundary = dstw - 1;
            for (; j <= dstw_not_cross_boundary - 2; j += 2) {
                uint8x8_t q0 = vld1_u8(row1 + j * 18 + 6);
                prefetch_l1(row1, j * 18 + 6 + 256);
                uint8x8_t q1 = vld1_u8(row2 + j * 18 + 6);
                prefetch_l1(row2, j * 18 + 6 + 256);
                uint8x8_t q2 = vld1_u8(row1 + (j + 1) * 18 + 6);
                prefetch_l1(row1, (j + 1) * 18 + 6 + 256);
                uint8x8_t q3 = vld1_u8(row2 + (j + 1) * 18 + 6);
                prefetch_l1(row2, (j + 1) * 18 + 6 + 256);
                uint16x8_t q0_u16 = vmovl_u8(q0);
                uint16x8_t q1_u16 = vmovl_u8(q1);
                uint16x8_t q2_u16 = vmovl_u8(q2);
                uint16x8_t q3_u16 = vmovl_u8(q3);
                uint16x4_t q00 = vget_low_u16(q0_u16);
                uint16x4_t q01 = vget_low_u16(vextq_u16(q0_u16, q0_u16, 3));
                uint16x4_t q10 = vget_low_u16(q1_u16);
                uint16x4_t q11 = vget_low_u16(vextq_u16(q1_u16, q1_u16, 3));
                uint16x4_t q20 = vget_low_u16(q2_u16);
                uint16x4_t q21 = vget_low_u16(vextq_u16(q2_u16, q2_u16, 3));
                uint16x4_t q30 = vget_low_u16(q3_u16);
                uint16x4_t q31 = vget_low_u16(vextq_u16(q3_u16, q3_u16, 3));
                uint16x4_t res0_u16 = vadd_u16(vadd_u16(vadd_u16(q00, q01), vadd_u16(q10, q11)), vdup_n_u16(2));
                uint16x4_t res1_u16 = vadd_u16(vadd_u16(vadd_u16(q20, q21), vadd_u16(q30, q31)), vdup_n_u16(2));
                uint8x8_t res_u8 = vqmovn_u16(vcombine_u16(vshr_n_u16(res0_u16, 2), vshr_n_u16(res1_u16, 2)));
                uint8x8_t ans = vtbl1_u8(res_u8, tbl);
                asm volatile(
                    "st1 {%2.s}[0], [%0]\n\t"
                    "st1 {%2.h}[2], [%1]\n\t"
                    :
                    : "r"(dst + i * dst_stride + j * 3), "r"(dst + i * dst_stride + j * 3 + 4), "w"(ans)
                    : "cc", "memory");
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 6 * cn + 2 * cn + c] + row1[j * 6 * cn + 3 * cn + c] +
                                                        row2[j * 6 * cn + 2 * cn + c] + row2[j * 6 * cn + 3 * cn + c] + 2) >>
                                                       2;
                }
            }
        }
    } else if (cn == 4) {
        // std::cerr << "running here" << std::endl;
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (6 * i + 2) * src_stride;
            const uint8_t* row2 = src + (6 * i + 3) * src_stride;
            int32_t j = 0;
            for (; j <= dstw - 2; j += 2) {
                uint8x8_t q0 = vld1_u8(row1 + j * 24 + 8);
                prefetch_l1(row1, j * 24 + 8 + 256);
                uint8x8_t q2 = vld1_u8(row1 + (j + 1) * 24 + 8);
                prefetch_l1(row1, (j + 1) * 24 + 8 + 256);
                uint8x8_t q1 = vld1_u8(row2 + j * 24 + 8);
                prefetch_l1(row2, j * 24 + 8 + 256);
                uint8x8_t q3 = vld1_u8(row2 + (j + 1) * 24 + 8);
                prefetch_l1(row2, (j + 1) * 24 + 8 + 256);
                uint16x8_t q0_u16 = vmovl_u8(q0);
                uint16x8_t q1_u16 = vmovl_u8(q1);
                uint16x8_t q2_u16 = vmovl_u8(q2);
                uint16x8_t q3_u16 = vmovl_u8(q3);
                uint16x4_t q00 = vget_low_u16(q0_u16);
                uint16x4_t q01 = vget_high_u16(q0_u16);
                uint16x4_t q10 = vget_low_u16(q1_u16);
                uint16x4_t q11 = vget_high_u16(q1_u16);
                uint16x4_t q20 = vget_low_u16(q2_u16);
                uint16x4_t q21 = vget_high_u16(q2_u16);
                uint16x4_t q30 = vget_low_u16(q3_u16);
                uint16x4_t q31 = vget_high_u16(q3_u16);
                uint16x4_t res0_u16 = vadd_u16(vadd_u16(vadd_u16(q00, q01), vadd_u16(q10, q11)), vdup_n_u16(2));
                uint16x4_t res1_u16 = vadd_u16(vadd_u16(vadd_u16(q20, q21), vadd_u16(q30, q31)), vdup_n_u16(2));
                uint8x8_t res_u8 = vshrn_n_u16(vcombine_u16(res0_u16, res1_u16), 2);
                vst1_u8(dst + i * dst_stride + j * 4, res_u8);
            }
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 6 * cn + 2 * cn + c] + row1[j * 6 * cn + 3 * cn + c] +
                                                        row2[j * 6 * cn + 2 * cn + c] + row2[j * 6 * cn + 3 * cn + c] + 2) >>
                                                       2;
                }
            }
        }
    } else {
        for (int32_t i = 0; i < dsth; i++) {
            const uint8_t* row1 = src + (6 * i + 2) * src_stride;
            const uint8_t* row2 = src + (6 * i + 3) * src_stride;
            int32_t j = 0;
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 6 * cn + 2 * cn + c] + row1[j * 6 * cn + 3 * cn + c] +
                                                        row2[j * 6 * cn + 2 * cn + c] + row2[j * 6 * cn + 3 * cn + c] + 2) >>
                                                       2;
                }
            }
        }
    }

    return true;
}

#define offset_precompute 1

void resize_generic_8UC3(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcStride,
    const uint8_t* src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstStride,
    uint8_t* dst,
    int32_t* xofs,
    int32_t* yofs,
    int16_t* ialpha,
    int16_t* ibeta,
    int32_t num_threads)
{
    int32_t w = dstWidth;
    int32_t h = dstHeight;

    int32x4_t _v2 = vdupq_n_s32(2);
    int16x4_t INTER_RESIZE_COEF_SCALE_vec = vdup_n_s16(INTER_RESIZE_COEF_SCALE);
    int8x8_t _tb = {0, 1, 2, 4, 5, 6, 0, 0};

    int32_t w_not_cross_boundary = w;
    for (int32_t dw = w - 1; dw >= 0; dw--) {
        if (xofs[dw] < (srcWidth - 2) * 3) {
            break;
        }
        w_not_cross_boundary--;
    }

    for (int32_t dy = 0; dy < h; dy++) {
        int32_t sy = yofs[dy];
        const uint8_t* S0 = src + sy * srcStride; //src.ptr(sy);
        const uint8_t* S1 = src + (sy + 1) * srcStride; //src.ptr(sy+1);

        uint8_t* Dp = dst + dy * dstStride; //dst.ptr(dy);
        const int16_t* ialphap = ialpha;

        int16x4_t _b1 = vdup_n_s16(ibeta[dy]);
        int16x4_t _b0 = vdup_n_s16(INTER_RESIZE_COEF_SCALE - ibeta[dy]);
        int32_t* tmp_xofs = xofs;

        int32_t remain = (w >> 2) << 2;
        if (sy >= srcHeight - 2) {
            remain = (w_not_cross_boundary >> 2) << 2;
        }
        int32_t nn = w - remain;
        if (remain > 0) {
            asm volatile(
                "ldpsw x10, x11, [%1], #8\n\t"
                "ldpsw x12, x13, [%1], #8\n\t"
                "ldpsw x19, x20, [%1], #8\n\t"
                "ldpsw x21, x22, [%1], #8\n\t"
                "ldr d2, [%8, x10]\n\t"
                "prfm pldl1keep, [%8, x19]\n\t"
                "ldr d3, [%8, x11]\n\t"
                "prfm pldl1keep, [%8, x20]\n\t"
                "ldr d4, [%8, x12]\n\t"
                "prfm pldl1keep, [%8, x21]\n\t"
                "ldr d5, [%8, x13]\n\t"
                "prfm pldl1keep, [%8, x22]\n\t"
                "ld1 {v1.4h}, [%2], #8\n\t"

                "0:\n\t"
                "#start dx*v0+(1-dx)*v1\n\t"
                "#shift 8bit to 16bit and construct vectors\n\t"
                "ushll v10.8h, v2.8b, #0\n\t"
                "ldr d6, [%9, x10]\n\t"
                "prfm pldl1keep, [%9, x19]\n\t"
                "ushll v11.8h, v3.8b, #0\n\t"
                "ldr d7, [%9, x11]\n\t"
                "prfm pldl1keep, [%9, x20]\n\t"
                "ushll v12.8h, v4.8b, #0\n\t"
                "ldr d8, [%9, x12]\n\t"
                "prfm pldl1keep, [%9, x21]\n\t"
                "ushll v13.8h, v5.8b, #0\n\t"
                "ldr d9, [%9, x13]\n\t"
                "prfm pldl1keep, [%9, x22]\n\t"
                "sub v0.4h, %14.4h, v1.4h\n\t"
                "mov v14.d[0], v10.d[1]\n\t"
                "mov x10, x19\n\t"
                "mov x11, x20\n\t"
                "mov x12, x21\n\t"
                "mov x13, x22\n\t"
                "ldpsw x19, x20, [%1], #8\n\t"
                "mov v15.d[0], v11.d[1]\n\t"
                "ldpsw x21, x22, [%1], #8\n\t"
                "mov v16.d[0], v12.d[1]\n\t"
                "mov v17.d[0], v13.d[1]\n\t"
                "ext v14.8b, v10.8b, v14.8b, #6\n\t"
                "ext v15.8b, v11.8b, v15.8b, #6\n\t"
                "ext v16.8b, v12.8b, v16.8b, #6\n\t"
                "ext v17.8b, v13.8b, v17.8b, #6\n\t"
                "#calculate\n\t"
                "smull v10.4s, v10.4h, v0.h[0]\n\t"
                "ldr d2, [%8, x10]\n\t"
                "prfm pldl1keep, [%8, x19]\n\t"
                "smull v11.4s, v11.4h, v0.h[1]\n\t"
                "ldr d3, [%8, x11]\n\t"
                "prfm pldl1keep, [%8, x20]\n\t"
                "smull v12.4s, v12.4h, v0.h[2]\n\t"
                "ldr d4, [%8, x12]\n\t"
                "prfm pldl1keep, [%8, x21]\n\t"
                "smull v13.4s, v13.4h, v0.h[3]\n\t"
                "ldr d5, [%8, x13]\n\t"
                "prfm pldl1keep, [%8, x22]\n\t"
                "smlal v10.4s, v14.4h, v1.h[0]\n\t"
                "ushll v18.8h, v6.8b, #0\n\t"
                "smlal v11.4s, v15.4h, v1.h[1]\n\t"
                "ushll v19.8h, v7.8b, #0\n\t"
                "smlal v12.4s, v16.4h, v1.h[2]\n\t"
                "ushll v20.8h, v8.8b, #0\n\t"
                "smlal v13.4s, v17.4h, v1.h[3]\n\t"
                "ushll v21.8h, v9.8b, #0\n\t"
                "shrn v10.4h, v10.4s, #4\n\t"
                "mov v22.d[0], v18.d[1]\n\t"
                "shrn v11.4h, v11.4s, #4\n\t"
                "mov v23.d[0], v19.d[1]\n\t"
                "shrn v12.4h, v12.4s, #4\n\t"
                "mov v24.d[0], v20.d[1]\n\t"
                "shrn v13.4h, v13.4s, #4\n\t"
                "mov v25.d[0], v21.d[1]\n\t"
                "#start dy*v0+(1-dy)*v1\n\t"
                "ext v22.8b, v18.8b, v22.8b, #6\n\t"
                "ext v23.8b, v19.8b, v23.8b, #6\n\t"
                "ext v24.8b, v20.8b, v24.8b, #6\n\t"
                "ext v25.8b, v21.8b, v25.8b, #6\n\t"
                "smull v18.4s, v18.4h, v0.h[0]\n\t"
                "smull v19.4s, v19.4h, v0.h[1]\n\t"
                "smull v20.4s, v20.4h, v0.h[2]\n\t"
                "smull v21.4s, v21.4h, v0.h[3]\n\t"
                "smlal v18.4s, v22.4h, v1.h[0]\n\t"
                "mov v14.16b, %12.16b\n\t"
                "smlal v19.4s, v23.4h, v1.h[1]\n\t"
                "mov v15.16b, %12.16b\n\t"
                "smlal v20.4s, v24.4h, v1.h[2]\n\t"
                "mov v16.16b, %12.16b\n\t"
                "smlal v21.4s, v25.4h, v1.h[3]\n\t"
                "mov v17.16b, %12.16b\n\t"
                "ld1 {v1.4h}, [%2], #8\n\t"
                "shrn v18.4h, v18.4s, #4\n\t"
                "smull v10.4s, v10.4h, %10.4h\n\t"
                "shrn v19.4h, v19.4s, #4\n\t"
                "smull v11.4s, v11.4h, %10.4h\n\t"
                "shrn v20.4h, v20.4s, #4\n\t"
                "smull v12.4s, v12.4h, %10.4h\n\t"
                "shrn v21.4h, v21.4s, #4\n\t"
                "smull v13.4s, v13.4h, %10.4h\n\t"
                "ssra v14.4s, v10.4s, #16\n\t"
                "smull v18.4s, v18.4h, %11.4h\n\t"
                "ssra v15.4s, v11.4s, #16\n\t"
                "smull v19.4s, v19.4h, %11.4h\n\t"
                "ssra v16.4s, v12.4s, #16\n\t"
                "smull v20.4s, v20.4h, %11.4h\n\t"
                "ssra v17.4s, v13.4s, #16\n\t"
                "smull v21.4s, v21.4h, %11.4h\n\t"
                "subs %0, %0, #4\n\t"
                "ssra v14.4s, v18.4s, #16\n\t"
                "ssra v15.4s, v19.4s, #16\n\t"
                "ssra v16.4s, v20.4s, #16\n\t"
                "ssra v17.4s, v21.4s, #16\n\t"
                "shrn v14.4h, v14.4s, #2\n\t"
                "shrn v15.4h, v15.4s, #2\n\t"
                "shrn v16.4h, v16.4s, #2\n\t"
                "shrn v17.4h, v17.4s, #2\n\t"

                "#start merge and TBL\n\t"
                "ins v14.d[1], v15.d[0]\n\t"
                "ins v16.d[1], v17.d[0]\n\t"
                "sqxtun v14.8b, v14.8h\n\t"
                "sqxtun v15.8b, v16.8h\n\t"
                "tbl v14.8b, {v14.16b}, %13.8b\n\t"
                "tbl v15.8b, {v15.16b}, %13.8b\n\t"

                "st1 {v14.s}[0], [%3], #4\n\t"
                "st1 {v14.h}[2], [%3], #2\n\t"
                "st1 {v15.s}[0], [%3], #4\n\t"
                "st1 {v15.h}[2], [%3], #2\n\t"

                "bne 0b\n\t"
                "sub %2, %2, #8\n\t"

                : "=r"(remain), "=r"(tmp_xofs), "=r"(ialphap), "=r"(Dp)
                : "0"(remain), "1"(tmp_xofs), "2"(ialphap), "3"(Dp), "r"(S0), "r"(S1), "w"(_b0), "w"(_b1), "w"(_v2), "w"(_tb), "w"(INTER_RESIZE_COEF_SCALE_vec)
                : "cc", "memory", "x10", "x11", "x12", "x13", "x19", "x20", "x21", "x22", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
        }
        //Corner Case
        for (; nn > 0; nn--) {
            int32_t ofs = xofs[w - nn];
            const uint8_t* pS0 = S0 + ofs;
            const uint8_t* pS1 = S1 + ofs;
            int16_t a1 = ialphap[0];
            int16_t a0 = INTER_RESIZE_COEF_SCALE - a1;
            int16_t b1 = ibeta[dy];
            int16_t b0 = INTER_RESIZE_COEF_SCALE - b1;

            for (int32_t i = 0; i < 3; i++) {
                int32_t kS0 = pS0[i] * a0 + pS0[i + 3] * a1;
                int32_t kS1 = pS1[i] * a0 + pS1[i + 3] * a1;
                Dp[i] = uint8_t(((int16_t)((b0 * (int16_t)(kS0 >> 4)) >> 16) + (int16_t)((b1 * (int16_t)(kS1 >> 4)) >> 16) + 2) >> 2);
            }

            ialphap++;
            Dp += 3;
        }
    }
}

void resize_generic_8UC4(
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcStride,
    const uint8_t* src,
    int32_t dstHeight,
    int32_t dstWidth,
    int32_t dstStride,
    uint8_t* dst,
    int32_t* xofs,
    int32_t* yofs,
    int16_t* ialpha,
    int16_t* ibeta,
    int32_t num_threads)
{
    int32_t w = dstWidth;
    int32_t h = dstHeight;

    int32x4_t _v2 = vdupq_n_s32(2);
    int16x4_t INTER_RESIZE_COEF_SCALE_vec = vdup_n_s16(INTER_RESIZE_COEF_SCALE);

    for (int32_t dy = 0; dy < h; dy++) {
        int32_t sy = yofs[dy];
        const uint8_t* S0 = src + sy * srcStride; //src.ptr(sy);
        const uint8_t* S1 = src + (sy + 1) * srcStride; //src.ptr(sy+1);
        uint8_t* Dp = dst + dy * dstStride; //dst.ptr(dy);
        const int16_t* ialphap = ialpha;

        int16x4_t _b1 = vdup_n_s16(ibeta[dy]);
        int16x4_t _b0 = vdup_n_s16(INTER_RESIZE_COEF_SCALE - ibeta[dy]);
        int32_t* tmp_xofs = xofs;

        int32_t remain = (w >> 2) << 2;
        int32_t nn = w - remain;
        if (remain > 0) {
            asm volatile(
                "ldpsw x10, x11, [%1], #8\n\t"
                "ldpsw x12, x13, [%1], #8\n\t"
                "ldpsw x19, x20, [%1], #8\n\t"
                "ldpsw x21, x22, [%1], #8\n\t"
                "ldr d2, [%8, x10]\n\t"
                "prfm pldl1keep, [%8, x19]\n\t"
                "ldr d3, [%8, x11]\n\t"
                "prfm pldl1keep, [%8, x20]\n\t"
                "ldr d4, [%8, x12]\n\t"
                "prfm pldl1keep, [%8, x21]\n\t"
                "ldr d5, [%8, x13]\n\t"
                "prfm pldl1keep, [%8, x22]\n\t"
                "ld1 {v1.4h}, [%2], #8\n\t"

                "0:\n\t"
                "#start dx*v0+(1-dx)*v1\n\t"
                "#shift 8bit to 16bit and construct vectors\n\t"
                "ushll v10.8h, v2.8b, #0\n\t"
                "ldr d6, [%9, x10]\n\t"
                "prfm pldl1keep, [%9, x19]\n\t"
                "ushll v11.8h, v3.8b, #0\n\t"
                "ldr d7, [%9, x11]\n\t"
                "prfm pldl1keep, [%9, x20]\n\t"
                "ushll v12.8h, v4.8b, #0\n\t"
                "ldr d8, [%9, x12]\n\t"
                "prfm pldl1keep, [%9, x21]\n\t"
                "ushll v13.8h, v5.8b, #0\n\t"
                "ldr d9, [%9, x13]\n\t"
                "prfm pldl1keep, [%9, x22]\n\t"
                "sub v0.4h, %13.4h, v1.4h\n\t"
                "mov v14.d[0], v10.d[1]\n\t"
                "mov x10, x19\n\t"
                "mov x11, x20\n\t"
                "mov x12, x21\n\t"
                "mov x13, x22\n\t"
                "ldpsw x19, x20, [%1], #8\n\t"
                "mov v15.d[0], v11.d[1]\n\t"
                "ldpsw x21, x22, [%1], #8\n\t"
                "mov v16.d[0], v12.d[1]\n\t"
                "mov v17.d[0], v13.d[1]\n\t"
                "#calculate\n\t"
                "smull v10.4s, v10.4h, v0.h[0]\n\t"
                "ldr d2, [%8, x10]\n\t"
                "prfm pldl1keep, [%8, x19]\n\t"
                "smull v11.4s, v11.4h, v0.h[1]\n\t"
                "ldr d3, [%8, x11]\n\t"
                "prfm pldl1keep, [%8, x20]\n\t"
                "smull v12.4s, v12.4h, v0.h[2]\n\t"
                "ldr d4, [%8, x12]\n\t"
                "prfm pldl1keep, [%8, x21]\n\t"
                "smull v13.4s, v13.4h, v0.h[3]\n\t"
                "ldr d5, [%8, x13]\n\t"
                "prfm pldl1keep, [%8, x22]\n\t"
                "smlal v10.4s, v14.4h, v1.h[0]\n\t"
                "ushll v18.8h, v6.8b, #0\n\t"
                "smlal v11.4s, v15.4h, v1.h[1]\n\t"
                "ushll v19.8h, v7.8b, #0\n\t"
                "smlal v12.4s, v16.4h, v1.h[2]\n\t"
                "ushll v20.8h, v8.8b, #0\n\t"
                "smlal v13.4s, v17.4h, v1.h[3]\n\t"
                "ushll v21.8h, v9.8b, #0\n\t"
                "shrn v10.4h, v10.4s, #4\n\t"
                "mov v22.d[0], v18.d[1]\n\t"
                "shrn v11.4h, v11.4s, #4\n\t"
                "mov v23.d[0], v19.d[1]\n\t"
                "shrn v12.4h, v12.4s, #4\n\t"
                "mov v24.d[0], v20.d[1]\n\t"
                "shrn v13.4h, v13.4s, #4\n\t"
                "mov v25.d[0], v21.d[1]\n\t"
                "#start dy*v0+(1-dy)*v1\n\t"
                "smull v18.4s, v18.4h, v0.h[0]\n\t"
                "smull v19.4s, v19.4h, v0.h[1]\n\t"
                "smull v20.4s, v20.4h, v0.h[2]\n\t"
                "smull v21.4s, v21.4h, v0.h[3]\n\t"
                "smlal v18.4s, v22.4h, v1.h[0]\n\t"
                "mov v14.16b, %12.16b\n\t"
                "smlal v19.4s, v23.4h, v1.h[1]\n\t"
                "mov v15.16b, %12.16b\n\t"
                "smlal v20.4s, v24.4h, v1.h[2]\n\t"
                "mov v16.16b, %12.16b\n\t"
                "smlal v21.4s, v25.4h, v1.h[3]\n\t"
                "mov v17.16b, %12.16b\n\t"
                "ld1 {v1.4h}, [%2], #8\n\t"
                "shrn v18.4h, v18.4s, #4\n\t"
                "smull v10.4s, v10.4h, %10.4h\n\t"
                "shrn v19.4h, v19.4s, #4\n\t"
                "smull v11.4s, v11.4h, %10.4h\n\t"
                "shrn v20.4h, v20.4s, #4\n\t"
                "smull v12.4s, v12.4h, %10.4h\n\t"
                "shrn v21.4h, v21.4s, #4\n\t"
                "smull v13.4s, v13.4h, %10.4h\n\t"
                "ssra v14.4s, v10.4s, #16\n\t"
                "smull v18.4s, v18.4h, %11.4h\n\t"
                "ssra v15.4s, v11.4s, #16\n\t"
                "smull v19.4s, v19.4h, %11.4h\n\t"
                "ssra v16.4s, v12.4s, #16\n\t"
                "smull v20.4s, v20.4h, %11.4h\n\t"
                "ssra v17.4s, v13.4s, #16\n\t"
                "smull v21.4s, v21.4h, %11.4h\n\t"
                "subs %0, %0, #4\n\t"
                "ssra v14.4s, v18.4s, #16\n\t"
                "ssra v15.4s, v19.4s, #16\n\t"
                "ssra v16.4s, v20.4s, #16\n\t"
                "ssra v17.4s, v21.4s, #16\n\t"
                "shrn v14.4h, v14.4s, #2\n\t"
                "shrn v15.4h, v15.4s, #2\n\t"
                "shrn v16.4h, v16.4s, #2\n\t"
                "shrn v17.4h, v17.4s, #2\n\t"

                "#start merge and TBL\n\t"
                "ins v14.d[1], v15.d[0]\n\t"
                "ins v16.d[1], v17.d[0]\n\t"
                "sqxtun v14.8b, v14.8h\n\t"
                "sqxtun v15.8b, v16.8h\n\t"

                "st1 {v14.8b}, [%3], #8\n\t"
                "st1 {v15.8b}, [%3], #8\n\t"

                "bne 0b\n\t"
                "sub %2, %2, #8\n\t"

                : "=r"(remain), "=r"(tmp_xofs), "=r"(ialphap), "=r"(Dp)
                : "0"(remain), "1"(tmp_xofs), "2"(ialphap), "3"(Dp), "r"(S0), "r"(S1), "w"(_b0), "w"(_b1), "w"(_v2), "w"(INTER_RESIZE_COEF_SCALE_vec)
                : "cc", "memory", "x10", "x11", "x12", "x13", "x19", "x20", "x21", "x22", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
        }
        //Corner Case
        for (; nn > 0; nn--) {
            int32_t ofs = xofs[w - nn];
            const uint8_t* pS0 = S0 + ofs;
            const uint8_t* pS1 = S1 + ofs;
            int16_t a1 = ialphap[0];
            int16_t a0 = INTER_RESIZE_COEF_SCALE - a1;
            int16_t b1 = ibeta[dy];
            int16_t b0 = INTER_RESIZE_COEF_SCALE - b1;

            for (int32_t i = 0; i < 4; i++) {
                int32_t kS0 = pS0[i] * a0 + pS0[i + 4] * a1;
                int32_t kS1 = pS1[i] * a0 + pS1[i + 4] * a1;
                Dp[i] = uint8_t(((int16_t)((b0 * (int16_t)(kS0 >> 4)) >> 16) + (int16_t)((b1 * (int16_t)(kS1 >> 4)) >> 16) + 2) >> 2);
            }

            ialphap++;
            Dp += 4;
        }
    }
}

void resize_linear_generic(
    int32_t channels,
    int32_t inHeight,
    int32_t inWidth,
    int32_t inStride,
    const uint8_t* in,
    int32_t outHeight,
    int32_t outWdith,
    int32_t outStride,
    uint8_t* out)
{
    int32_t w = outWdith;
    int32_t h = outHeight;

    double scale_x = (double)inWidth / outWdith;
    double scale_y = (double)inHeight / outHeight;

    int32_t* buf = new int32_t[w + 8 + h + w + h];

    int32_t* xofs = buf;
    // preload offset
    for (int32_t i = 0; i < 8; i++) {
        xofs[w + i] = 0;
    }
    int32_t* yofs = buf + w + 8;

    int16_t* ialpha = (int16_t*)(buf + w + 8 + h);
    int16_t* ibeta = (int16_t*)(buf + w + 8 + h + w);

    float fx;
    float fy;
    int32_t sx;
    int32_t sy;
    int32_t srcw = inWidth;
    int32_t srch = inHeight;

    int16_t* ialphap = ialpha;
    for (int32_t dx = 0; dx < w; dx++) {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = Floor(fx);
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0;
        }

        if (sx >= srcw - 1) {
            sx = srcw - 2;
            fx = 1.f;
        }

#if offset_precompute
        xofs[dx] = sx * channels;
#else
        xofs[dx] = sx;
#endif

        ialphap[0] = HPC::utils::saturate_cast<int16_t>(fx * INTER_RESIZE_COEF_SCALE);
        ialphap++;
    }

    int16_t* ibetap = ibeta;
    for (int32_t dy = 0; dy < h; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = Floor(fy);
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0;
        }

        if (sy >= srch - 1) {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        ibetap[0] = HPC::utils::saturate_cast<int16_t>(fy * INTER_RESIZE_COEF_SCALE);
        ibetap++;
    }

    if (channels == 3) {
        resize_generic_8UC3(inHeight, inWidth, inStride, in, outHeight, outWdith, outStride, out, xofs, yofs, ialpha, ibeta, 1);
    } else if (channels == 4) {
        resize_generic_8UC4(inHeight, inWidth, inStride, in, outHeight, outWdith, outStride, out, xofs, yofs, ialpha, ibeta, 1);
    }

    delete[] buf;
}

void resize_bilinear_rows(
    const Size& ssize,
    const Size& dsize,
    const uint8_t* srcBase,
    ptrdiff_t srcStride,
    uint8_t* dstBase,
    ptrdiff_t dstStride,
    float hr,
    const uint8_t** gcols,
    uint8_t* gcweight,
    uint8_t* buf)
{
    float scale_y_offset = 0.5f * hr - 0.5f;

    int32_t dst_h8 = dsize.height & ~7;
    int32_t dst_w8 = dsize.width & ~7;
    int32_t src_w8 = ssize.width & ~7;

    int32_t r = 0;
    for (; r < dst_h8; r += 8) {
    resize8u_xystretch:
        const uint8_t* rows[16];
        uint8_t rweight[8];

        for (uint32_t i = 0; i < 8; ++i) {
            float w = (i + r) * hr + scale_y_offset;
            ptrdiff_t src_row = floorf(w);
            ptrdiff_t src_row2 = src_row + 1;

            rweight[i] = (uint8_t)((src_row2 - w) * 128);

            if (src_row < 0)
                src_row = 0;
            if (src_row2 >= (ptrdiff_t)ssize.height)
                src_row2 = ssize.height - 1;

            rows[2 * i] = srcBase + src_row * srcStride;
            rows[2 * i + 1] = srcBase + src_row2 * srcStride;
        }

        uint8x8_t vr0w = vdup_n_u8(rweight[0]);
        uint8x8_t vr1w = vdup_n_u8(rweight[1]);
        uint8x8_t vr2w = vdup_n_u8(rweight[2]);
        uint8x8_t vr3w = vdup_n_u8(rweight[3]);
        uint8x8_t vr4w = vdup_n_u8(rweight[4]);
        uint8x8_t vr5w = vdup_n_u8(rweight[5]);
        uint8x8_t vr6w = vdup_n_u8(rweight[6]);
        uint8x8_t vr7w = vdup_n_u8(rweight[7]);

        uint8x8_t vr0w2 = vdup_n_u8(128 - rweight[0]);
        uint8x8_t vr1w2 = vdup_n_u8(128 - rweight[1]);
        uint8x8_t vr2w2 = vdup_n_u8(128 - rweight[2]);
        uint8x8_t vr3w2 = vdup_n_u8(128 - rweight[3]);
        uint8x8_t vr4w2 = vdup_n_u8(128 - rweight[4]);
        uint8x8_t vr5w2 = vdup_n_u8(128 - rweight[5]);
        uint8x8_t vr6w2 = vdup_n_u8(128 - rweight[6]);
        uint8x8_t vr7w2 = vdup_n_u8(128 - rweight[7]);

        int32_t col = 0;
        for (; col < src_w8; col += 8) {
            prefetch(rows[3] + col);
            prefetch(rows[7] + col);
            prefetch(rows[11] + col);
            prefetch(rows[15] + col);
        resize8u_ystretch:
            uint8x8_t vsrc0l1 = vld1_u8(rows[0] + col);
            uint8x8_t vsrc0l2 = vld1_u8(rows[1] + col);
            uint8x8_t vsrc1l1 = vld1_u8(rows[2] + col);
            uint8x8_t vsrc1l2 = vld1_u8(rows[3] + col);

            // (l1 * w + l2 * (128 - w) + 64) / 128
            uint16x8_t vdst0l = vmull_u8(vsrc0l1, vr0w);
            uint16x8_t vdst1l = vmull_u8(vsrc1l1, vr1w);

            uint8x8_t vsrc2l1 = vld1_u8(rows[4] + col);
            uint8x8_t vsrc2l2 = vld1_u8(rows[5] + col);
            uint8x8_t vsrc3l1 = vld1_u8(rows[6] + col);
            uint8x8_t vsrc3l2 = vld1_u8(rows[7] + col);

            vdst0l = vmlal_u8(vdst0l, vsrc0l2, vr0w2);
            vdst1l = vmlal_u8(vdst1l, vsrc1l2, vr1w2);
            uint16x8_t vdst2l = vmull_u8(vsrc2l1, vr2w);
            uint16x8_t vdst3l = vmull_u8(vsrc3l1, vr3w);

            uint8x8_t vsrc4l1 = vld1_u8(rows[8] + col);
            uint8x8_t vsrc4l2 = vld1_u8(rows[9] + col);
            uint8x8_t vsrc5l1 = vld1_u8(rows[10] + col);
            uint8x8_t vsrc5l2 = vld1_u8(rows[11] + col);

            vdst2l = vmlal_u8(vdst2l, vsrc2l2, vr2w2);
            vdst3l = vmlal_u8(vdst3l, vsrc3l2, vr3w2);
            uint16x8_t vdst4l = vmull_u8(vsrc4l1, vr4w);
            uint16x8_t vdst5l = vmull_u8(vsrc5l1, vr5w);

            uint8x8_t vsrc6l1 = vld1_u8(rows[12] + col);
            uint8x8_t vsrc6l2 = vld1_u8(rows[13] + col);
            uint8x8_t vsrc7l1 = vld1_u8(rows[14] + col);
            uint8x8_t vsrc7l2 = vld1_u8(rows[15] + col);

            uint8x8_t vdst0 = vshrn_n_u16(vdst0l, 7);
            uint8x8_t vdst1 = vshrn_n_u16(vdst1l, 7);
            vdst4l = vmlal_u8(vdst4l, vsrc4l2, vr4w2);
            vdst5l = vmlal_u8(vdst5l, vsrc5l2, vr5w2);
            uint16x8_t vdst6l = vmull_u8(vsrc6l1, vr6w);
            uint16x8_t vdst7l = vmull_u8(vsrc7l1, vr7w);

            uint8x8_t vdst2 = vshrn_n_u16(vdst2l, 7);
            uint8x8_t vdst3 = vshrn_n_u16(vdst3l, 7);
            vdst6l = vmlal_u8(vdst6l, vsrc6l2, vr6w2);
            vdst7l = vmlal_u8(vdst7l, vsrc7l2, vr7w2);

            uint8x8_t vdst4 = vshrn_n_u16(vdst4l, 7);
            uint8x8_t vdst5 = vshrn_n_u16(vdst5l, 7);
            uint8x8_t vdst6 = vshrn_n_u16(vdst6l, 7);
            uint8x8_t vdst7 = vshrn_n_u16(vdst7l, 7);

            // == 8x8 matrix transpose ==

            //00 01 02 03 04 05 06 07   d0
            //10 11 12 13 14 15 16 17   d1
            //20 21 22 23 24 25 26 27   d2
            //30 31 32 33 34 35 36 37   d3
            //40 41 42 43 44 45 46 47   d4
            //50 51 52 53 54 55 56 57   d5
            //60 61 62 63 64 65 66 67   d6
            //70 71 72 73 74 75 76 77   d7

            uint8x8x2_t vdst10t = vtrn_u8(vdst0, vdst1);
            uint8x8x2_t vdst32t = vtrn_u8(vdst2, vdst3);
            uint8x8x2_t vdst54t = vtrn_u8(vdst4, vdst5);
            uint8x8x2_t vdst76t = vtrn_u8(vdst6, vdst7);

            uint8x16_t vd1d0 = vcombine_u8(vdst10t.val[0], vdst10t.val[1]);
            uint8x16_t vd3d2 = vcombine_u8(vdst32t.val[0], vdst32t.val[1]);
            uint8x16_t vd5d4 = vcombine_u8(vdst54t.val[0], vdst54t.val[1]);
            uint8x16_t vd7d6 = vcombine_u8(vdst76t.val[0], vdst76t.val[1]);

            //00 10 02 12 04 14 06 16   d0
            //01 11 03 13 05 15 07 17   d1
            //20 30 22 32 24 34 26 36   d2
            //21 31 23 33 25 35 27 37   d3
            //40 50 42 52 44 54 46 56   d4
            //41 51 43 53 45 55 47 57   d5
            //60 70 62 72 64 74 66 76   d6
            //61 71 63 73 65 75 67 77   d7

            uint16x8x2_t vq1q0t = vtrnq_u16((uint16x8_t)vd1d0, (uint16x8_t)vd3d2);
            uint16x8x2_t vq3q2t = vtrnq_u16((uint16x8_t)vd5d4, (uint16x8_t)vd7d6);

            //00 10 20 30 04 14 24 34   d0
            //01 11 21 31 05 15 25 35   d1
            //02 12 22 32 06 16 26 36   d2
            //03 13 23 33 07 17 27 37   d3
            //40 50 60 70 44 54 64 74   d4
            //41 51 61 71 45 55 65 75   d5
            //42 52 62 72 46 56 66 76   d6
            //43 53 63 73 47 57 67 77   d7

            uint32x4x2_t vq2q0t = vtrnq_u32((uint32x4_t)vq1q0t.val[0], (uint32x4_t)vq3q2t.val[0]);
            uint32x4x2_t vq3q1t = vtrnq_u32((uint32x4_t)vq1q0t.val[1], (uint32x4_t)vq3q2t.val[1]);

            //00 10 20 30 40 50 60 70   d0
            //01 11 21 31 41 51 61 71   d1
            //02 12 22 32 42 52 62 72   d2
            //03 13 23 33 43 53 63 73   d3
            //04 14 24 34 44 54 64 74   d4
            //05 15 25 35 45 55 65 75   d5
            //06 16 26 36 46 56 66 76   d6
            //07 17 27 37 47 57 67 77   d7

            vst1q_u8(buf + col * 8 + 0, (uint8x16_t)vq2q0t.val[0]);
            vst1q_u8(buf + col * 8 + 16, (uint8x16_t)vq3q1t.val[0]);
            vst1q_u8(buf + col * 8 + 32, (uint8x16_t)vq2q0t.val[1]);
            vst1q_u8(buf + col * 8 + 48, (uint8x16_t)vq3q1t.val[1]);
        }

        if (col < ssize.width) {
            col = ssize.width - 8;
            goto resize8u_ystretch;
        }

        uint8_t* dst_data = dstBase + r * dstStride;
        const uint8_t** cols = gcols;
        uint8_t* cweight = gcweight;

        int32_t dcol = 0;
        for (; dcol < dst_w8; dcol += 8, cols += 16, cweight += 8) {
            prefetch(cols[0], 64 * 4);
        resize8u_xstretch:
            uint8x8_t vc0w = vdup_n_u8(cweight[0]);
            uint8x8_t vc1w = vdup_n_u8(cweight[1]);
            uint8x8_t vc2w = vdup_n_u8(cweight[2]);
            uint8x8_t vc3w = vdup_n_u8(cweight[3]);
            uint8x8_t vc4w = vdup_n_u8(cweight[4]);
            uint8x8_t vc5w = vdup_n_u8(cweight[5]);
            uint8x8_t vc6w = vdup_n_u8(cweight[6]);
            uint8x8_t vc7w = vdup_n_u8(cweight[7]);

            uint8x8_t vc0w2 = vdup_n_u8(128 - cweight[0]);
            uint8x8_t vc1w2 = vdup_n_u8(128 - cweight[1]);
            uint8x8_t vc2w2 = vdup_n_u8(128 - cweight[2]);
            uint8x8_t vc3w2 = vdup_n_u8(128 - cweight[3]);
            uint8x8_t vc4w2 = vdup_n_u8(128 - cweight[4]);
            uint8x8_t vc5w2 = vdup_n_u8(128 - cweight[5]);
            uint8x8_t vc6w2 = vdup_n_u8(128 - cweight[6]);
            uint8x8_t vc7w2 = vdup_n_u8(128 - cweight[7]);

            uint8x8_t vsrc0l1 = vld1_u8(cols[0]);
            uint8x8_t vsrc0l2 = vld1_u8(cols[1]);
            uint8x8_t vsrc1l1 = vld1_u8(cols[2]);
            uint8x8_t vsrc1l2 = vld1_u8(cols[3]);
            uint8x8_t vsrc2l1 = vld1_u8(cols[4]);
            uint8x8_t vsrc2l2 = vld1_u8(cols[5]);
            uint8x8_t vsrc3l1 = vld1_u8(cols[6]);
            uint8x8_t vsrc3l2 = vld1_u8(cols[7]);
            uint8x8_t vsrc4l1 = vld1_u8(cols[8]);
            uint8x8_t vsrc4l2 = vld1_u8(cols[9]);
            uint8x8_t vsrc5l1 = vld1_u8(cols[10]);
            uint8x8_t vsrc5l2 = vld1_u8(cols[11]);
            uint8x8_t vsrc6l1 = vld1_u8(cols[12]);
            uint8x8_t vsrc6l2 = vld1_u8(cols[13]);
            uint8x8_t vsrc7l1 = vld1_u8(cols[14]);
            uint8x8_t vsrc7l2 = vld1_u8(cols[15]);

            // (l1 * w + l2 * (128 - w) + 64) / 128
            uint16x8_t vdst0l = vmull_u8(vsrc0l1, vc0w);
            uint16x8_t vdst1l = vmull_u8(vsrc1l1, vc1w);
            uint16x8_t vdst2l = vmull_u8(vsrc2l1, vc2w);
            uint16x8_t vdst3l = vmull_u8(vsrc3l1, vc3w);
            uint16x8_t vdst4l = vmull_u8(vsrc4l1, vc4w);
            uint16x8_t vdst5l = vmull_u8(vsrc5l1, vc5w);
            uint16x8_t vdst6l = vmull_u8(vsrc6l1, vc6w);
            uint16x8_t vdst7l = vmull_u8(vsrc7l1, vc7w);

            vdst0l = vmlal_u8(vdst0l, vsrc0l2, vc0w2);
            vdst1l = vmlal_u8(vdst1l, vsrc1l2, vc1w2);
            vdst2l = vmlal_u8(vdst2l, vsrc2l2, vc2w2);
            vdst3l = vmlal_u8(vdst3l, vsrc3l2, vc3w2);
            vdst4l = vmlal_u8(vdst4l, vsrc4l2, vc4w2);
            vdst5l = vmlal_u8(vdst5l, vsrc5l2, vc5w2);
            vdst6l = vmlal_u8(vdst6l, vsrc6l2, vc6w2);
            vdst7l = vmlal_u8(vdst7l, vsrc7l2, vc7w2);

            uint8x8_t vdst0 = vshrn_n_u16(vdst0l, 7);
            uint8x8_t vdst1 = vshrn_n_u16(vdst1l, 7);
            uint8x8_t vdst2 = vshrn_n_u16(vdst2l, 7);
            uint8x8_t vdst3 = vshrn_n_u16(vdst3l, 7);
            uint8x8_t vdst4 = vshrn_n_u16(vdst4l, 7);
            uint8x8_t vdst5 = vshrn_n_u16(vdst5l, 7);
            uint8x8_t vdst6 = vshrn_n_u16(vdst6l, 7);
            uint8x8_t vdst7 = vshrn_n_u16(vdst7l, 7);

            // == 8x8 matrix transpose ==
            uint8x8x2_t vdst10t = vtrn_u8(vdst0, vdst1);
            uint8x8x2_t vdst32t = vtrn_u8(vdst2, vdst3);
            uint8x8x2_t vdst54t = vtrn_u8(vdst4, vdst5);
            uint8x8x2_t vdst76t = vtrn_u8(vdst6, vdst7);
            uint8x16_t vd1d0 = vcombine_u8(vdst10t.val[0], vdst10t.val[1]);
            uint8x16_t vd3d2 = vcombine_u8(vdst32t.val[0], vdst32t.val[1]);
            uint8x16_t vd5d4 = vcombine_u8(vdst54t.val[0], vdst54t.val[1]);
            uint8x16_t vd7d6 = vcombine_u8(vdst76t.val[0], vdst76t.val[1]);
            uint16x8x2_t vq1q0t = vtrnq_u16((uint16x8_t)vd1d0, (uint16x8_t)vd3d2);
            uint16x8x2_t vq3q2t = vtrnq_u16((uint16x8_t)vd5d4, (uint16x8_t)vd7d6);
            uint32x4x2_t vq2q0t = vtrnq_u32((uint32x4_t)vq1q0t.val[0], (uint32x4_t)vq3q2t.val[0]);
            uint32x4x2_t vq3q1t = vtrnq_u32((uint32x4_t)vq1q0t.val[1], (uint32x4_t)vq3q2t.val[1]);

            //save results
            vst1_u8(dst_data + 0 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq2q0t.val[0]));
            vst1_u8(dst_data + 1 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq2q0t.val[0]));
            vst1_u8(dst_data + 2 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq3q1t.val[0]));
            vst1_u8(dst_data + 3 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq3q1t.val[0]));
            vst1_u8(dst_data + 4 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq2q0t.val[1]));
            vst1_u8(dst_data + 5 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq2q0t.val[1]));
            vst1_u8(dst_data + 6 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq3q1t.val[1]));
            vst1_u8(dst_data + 7 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq3q1t.val[1]));
        }

        if (dcol < dsize.width) {
            dcol = dsize.width - 8;
            cols = gcols + dcol * 2;
            cweight = gcweight + dcol;
            goto resize8u_xstretch;
        }
    }

    if (r < dsize.height) {
        r = dsize.height - 8;
        goto resize8u_xystretch;
    }
}

template <int32_t channels>
struct resizeLinearInternals;
template <>
struct resizeLinearInternals<1> {
    int32x4_t vc_upd;
    int32x4_t vc0;
    int32x4_t vcmax;

    inline resizeLinearInternals(int32x4_t& vi, uint32_t srccols)
    {
        vc_upd = vdupq_n_s32(4);
        vc0 = vdupq_n_s32(0);
        vcmax = vdupq_n_s32(srccols - 1);

        int32_t tmp0123[] = {0, 1, 2, 3};
        vi = vld1q_s32(tmp0123);
    }
    inline void updateIndexes(int32x4_t& vi, int32x4_t& vsrch, int32x4_t& vsrcl)
    {
        vsrch = vminq_s32(vsrch, vcmax);
        vsrcl = vmaxq_s32(vsrcl, vc0);
        vsrcl = vminq_s32(vsrcl, vcmax); //for safe tail
        vsrch = vshlq_n_s32(vsrch, 3);
        vsrcl = vshlq_n_s32(vsrcl, 3);
        vi = vaddq_s32(vi, vc_upd);
    }
};
template <>
struct resizeLinearInternals<4> {
    int32x4_t vc_upd;
    int32x4_t vc0;
    int32x4_t vcmax;
    int32x4_t v0123x8;

    inline resizeLinearInternals(int32x4_t& vi, uint32_t srccols)
    {
        vc_upd = vdupq_n_s32(1);
        vc0 = vdupq_n_s32(0);
        vcmax = vdupq_n_s32(srccols - 1);
        int32_t tmp0123x8[] = {0, 8, 16, 24};
        v0123x8 = vld1q_s32(tmp0123x8);

        vi = vc0;
    }
    inline void updateIndexes(int32x4_t& vi, int32x4_t& vsrch, int32x4_t& vsrcl)
    {
        vsrch = vminq_s32(vsrch, vcmax);
        vsrcl = vmaxq_s32(vsrcl, vc0);
        vsrch = vshlq_n_s32(vsrch, 5);
        vsrcl = vshlq_n_s32(vsrcl, 5);
        vsrch = vaddq_s32(vsrch, v0123x8);
        vsrcl = vaddq_s32(vsrcl, v0123x8);
        vi = vaddq_s32(vi, vc_upd);
    }
};

template <int32_t channels>
void resize_linear_u8c1orc4(
    uint32_t srcHeight,
    uint32_t srcWidth,
    uint32_t srcStride,
    const uint8_t* srcBase,
    uint32_t dstHeight,
    uint32_t dstWidth,
    uint32_t dstStride,
    uint8_t* dstBase)
{
    float wr = (float)srcWidth / dstWidth;
    float hr = (float)srcHeight / dstHeight;
    float scale_x_offset = 0.5f * wr - 0.5f;

    Size ssize, dsize;
    ssize.width = srcWidth * channels;
    ssize.height = srcHeight;
    dsize.width = dstWidth * channels;
    dsize.height = dstHeight;

    std::vector<uint8_t> gcweight((dsize.width + 7) & ~7);
    std::vector<const uint8_t*> gcols(((dsize.width + 7) & ~7) * 2);
    std::vector<uint8_t> buf(((ssize.width + 7) & ~7) * 8); // (8 rows) x (width of src)

    float32x4_t vscale_x = vdupq_n_f32(wr);
    float32x4_t vscale_x_offset = vdupq_n_f32(scale_x_offset);
    int32x4_t vc1 = vdupq_n_s32(1);
    float32x4_t vc128f = vdupq_n_f32(128.0f);

    int32x4_t vi;
    resizeLinearInternals<channels> indexes(vi, srcWidth); //uint32_t is used to store indexes
        //so we could get issues on src image dimensions greater than (2^32-1)

    for (int32_t dcol = 0; dcol < dsize.width; dcol += 8) {
        int32_t idx[16];

        float32x4_t vif = vcvtq_f32_s32(vi);
        float32x4_t vw = vmlaq_f32(vscale_x_offset, vscale_x, vif);
        int32x4_t vwi = vcvtq_s32_f32(vw);
        float32x4_t vwif = vcvtq_f32_s32(vwi);
        int32x4_t vmask = (int32x4_t)vcltq_f32(vwif, vw);
        int32x4_t vsrch = vsubq_s32(vwi, vmask);
        int32x4_t vsrcl = vsubq_s32(vsrch, vc1);
        float32x4_t vsrchf = vcvtq_f32_s32(vsrch);
        float32x4_t vw2 = vsubq_f32(vsrchf, vw);

        vw2 = vmulq_f32(vw2, vc128f);
        uint32x4_t vw32u = vcvtq_u32_f32(vw2);
        uint16x4_t vw16ul = vmovn_u32(vw32u);
        indexes.updateIndexes(vi, vsrch, vsrcl);

        vst1q_s32(idx + 0, vsrcl);
        vst1q_s32(idx + 8, vsrch);

        vif = vcvtq_f32_s32(vi);
        vw = vmlaq_f32(vscale_x_offset, vscale_x, vif);
        vwi = vcvtq_s32_f32(vw);
        vwif = vcvtq_f32_s32(vwi);
        vmask = (int32x4_t)vcltq_f32(vwif, vw);
        vsrch = vsubq_s32(vwi, vmask);
        vsrcl = vsubq_s32(vsrch, vc1);
        vsrchf = vcvtq_f32_s32(vsrch);
        vw2 = vsubq_f32(vsrchf, vw);

        vw2 = vmulq_f32(vw2, vc128f);
        vw32u = vcvtq_u32_f32(vw2);
        indexes.updateIndexes(vi, vsrch, vsrcl);

        uint16x4_t vw16uh = vmovn_u32(vw32u);

        vst1q_s32(idx + 4, vsrcl);
        vst1q_s32(idx + 12, vsrch);

        uint8x8_t vw8u = vmovn_u16(vcombine_u16(vw16ul, vw16uh));

        for (uint32_t i = 0; i < 8; ++i) {
            gcols[dcol * 2 + i * 2] = &buf[idx[i]];
            gcols[dcol * 2 + i * 2 + 1] = &buf[idx[i + 8]];
        }

        vst1_u8(&gcweight[dcol], vw8u);
    }

    resize_bilinear_rows(ssize, dsize, srcBase, srcStride, dstBase, dstStride, hr, &gcols[0], &gcweight[0], &buf[0]);
}

static void img_resize_cal_offset_linear_uchar(
    int32_t* xofs,
    int16_t* ialpha,
    int32_t* yofs,
    int16_t* ibeta,
    int32_t* xmin,
    int32_t* xmax,
    int32_t ksize,
    int32_t ksize2,
    int32_t srcw,
    int32_t srch,
    int32_t dstw,
    int32_t dsth,
    int32_t channels)
{
    float inv_scale_x = (float)dstw / srcw;
    float inv_scale_y = (float)dsth / srch;

    int32_t cn = channels;
    float scale_x = 1. / inv_scale_x;
    float scale_y = 1. / inv_scale_y;
    int32_t k, sx, sy, dx, dy;

    float fx, fy;

    float cbuf[MAX_ESIZE];

    for (dx = 0; dx < dstw; dx++) {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = img_floor(fx);
        fx -= sx;

        if (sx < ksize2 - 1) {
            *xmin = dx + 1;
            if (sx < 0)
                fx = 0, sx = 0;
        }

        if (sx + ksize2 >= srcw) {
            *xmax = FUNC_MIN(*xmax, dx);
            if (sx >= srcw - 1)
                fx = 0, sx = srcw - 1;
        }

        for (k = 0, sx *= cn; k < cn; k++)
            xofs[dx * cn + k] = sx + k;

        cbuf[0] = 1.f - fx;
        cbuf[1] = fx;

        for (k = 0; k < ksize; k++)
            ialpha[dx * cn * ksize + k] = img_saturate_cast_short(cbuf[k] * INTER_RESIZE_COEF_SCALE);
        for (; k < cn * ksize; k++)
            ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];
    }

    for (dy = 0; dy < dsth; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = img_floor(fy);
        fy -= sy;

        yofs[dy] = sy;

        cbuf[0] = 1.f - fy;
        cbuf[1] = fy;

        for (k = 0; k < ksize; k++)
            ibeta[dy * ksize + k] = img_saturate_cast_short(cbuf[k] * INTER_RESIZE_COEF_SCALE);
    }
}

void img_hresize_4channels_linear_neon_uchar(
    const uint8_t** src,
    int32_t** dst,
    int32_t count,
    const int32_t* xofs,
    const int16_t* alpha,
    int32_t swidth,
    int32_t dwidth,
    int32_t cn,
    int32_t xmin,
    int32_t xmax)
{
    int32_t dx, k;
    int32_t dx0 = 0;

    int16x4x2_t alpha_vec;

    uint8x8_t dS0_vec, dS1_vec;
    int16x8_t qS0_vec, qS1_vec;
    int16x4_t dS0_0123, dS0_4567, dS1_0123, dS1_4567;

    int32x4_t qT0_vec, qT1_vec;

    int16x4_t dCoeff;
    dCoeff = vdup_n_s16(INTER_RESIZE_COEF_SCALE);

    //for (k = 0; k <= count - 2; k++)
    if (count == 2) {
        k = 0;
        const uint8_t *S0 = src[k], *S1 = src[k + 1];
        int32_t *D0 = dst[k], *D1 = dst[k + 1];

        for (dx = dx0; dx < xmax; dx += 4) {
            int32_t sx = xofs[dx];

            alpha_vec = vld2_s16(&alpha[dx * 2]);

            dS0_vec = vld1_u8(&S0[sx]);
            dS1_vec = vld1_u8(&S1[sx]);

            qS0_vec = vreinterpretq_s16_u16(vmovl_u8(dS0_vec));
            qS1_vec = vreinterpretq_s16_u16(vmovl_u8(dS1_vec));

            dS0_0123 = vget_low_s16(qS0_vec);
            dS0_4567 = vget_high_s16(qS0_vec);
            dS1_0123 = vget_low_s16(qS1_vec);
            dS1_4567 = vget_high_s16(qS1_vec);

            qT0_vec = vmull_s16(dS0_0123, alpha_vec.val[0]);
            qT1_vec = vmull_s16(dS1_0123, alpha_vec.val[0]);
            qT0_vec = vmlal_s16(qT0_vec, dS0_4567, alpha_vec.val[1]);
            qT1_vec = vmlal_s16(qT1_vec, dS1_4567, alpha_vec.val[1]);

            vst1q_s32(&D0[dx], qT0_vec);
            vst1q_s32(&D1[dx], qT1_vec);
        }

        for (; dx < dwidth; dx += 4) {
            int32_t sx = xofs[dx];

            dS0_vec = vld1_u8(&S0[sx]);
            dS1_vec = vld1_u8(&S1[sx]);

            qS0_vec = vreinterpretq_s16_u16(vmovl_u8(dS0_vec));
            qS1_vec = vreinterpretq_s16_u16(vmovl_u8(dS1_vec));

            dS0_0123 = vget_low_s16(qS0_vec);
            dS1_0123 = vget_low_s16(qS1_vec);

            qT0_vec = vmull_s16(dS0_0123, dCoeff);
            qT1_vec = vmull_s16(dS1_0123, dCoeff);

            vst1q_s32(&D0[dx], qT0_vec);
            vst1q_s32(&D1[dx], qT1_vec);
        }
    }

    //for (; k < count; k++)
    if (count == 1) {
        k = 0;
        const uint8_t* S = src[k];
        int32_t* D = dst[k];
        for (dx = 0; dx < xmax; dx += 4) {
            int32_t sx = xofs[dx];

            alpha_vec = vld2_s16(&alpha[dx * 2]);

            dS0_vec = vld1_u8(&S[sx]);
            qS0_vec = vreinterpretq_s16_u16(vmovl_u8(dS0_vec));

            dS0_0123 = vget_low_s16(qS0_vec);
            dS0_4567 = vget_high_s16(qS0_vec);

            qT0_vec = vmull_s16(dS0_0123, alpha_vec.val[0]);
            qT0_vec = vmlal_s16(qT0_vec, dS0_4567, alpha_vec.val[1]);

            vst1q_s32(&D[dx], qT0_vec);
        }

        for (; dx < dwidth; dx += 4) {
            int32_t sx = xofs[dx];

            int32_t value = *(int32_t*)(S + sx);
            int32x2_t int32_S0_vec;
            int32_S0_vec = vset_lane_s32(value, int32_S0_vec, 0);
            //uint8x8_t u8_S0_vec = vreinterpret_u8_s32(int32_S0_vec);
            uint8x8_t dS0_vec = vreinterpret_u8_s32(int32_S0_vec);

            //dS0_vec = vld1_u8 (&S[sx]);
            qS0_vec = vreinterpretq_s16_u16(vmovl_u8(dS0_vec));
            dS0_0123 = vget_low_s16(qS0_vec);
            qT0_vec = vmull_s16(dS0_0123, dCoeff);

            vst1q_s32(&D[dx], qT0_vec);
        }
    }
}

static void img_hresize_linear_c_uchar(
    const uint8_t** src,
    int32_t** dst,
    int32_t count,
    const int32_t* xofs,
    const int16_t* alpha,
    int32_t swidth,
    int32_t dwidth,
    int32_t cn,
    int32_t xmin,
    int32_t xmax)
{
    int32_t dx, k;

    int32_t k0 = 0;
    //for (k = 0; k <= count - 2; k++)
    if (count == 2) {
        k = 0;
        const uint8_t *S0 = src[k], *S1 = src[k + 1];
        int32_t *D0 = dst[k], *D1 = dst[k + 1];
        for (dx = k0; dx < xmax; dx++) {
            int32_t sx = xofs[dx];
            int32_t a0 = (int32_t)alpha[dx * 2], a1 = (int32_t)alpha[dx * 2 + 1];
            int32_t t0 = S0[sx] * a0 + S0[sx + cn] * a1;
            int32_t t1 = S1[sx] * a0 + S1[sx + cn] * a1;
            D0[dx] = t0;
            D1[dx] = t1;
        }

        for (; dx < dwidth; dx++) {
            int32_t sx = xofs[dx];
            D0[dx] = (int32_t)S0[sx] * INTER_RESIZE_COEF_SCALE;
            D1[dx] = (int32_t)S1[sx] * INTER_RESIZE_COEF_SCALE;
        }
    }

    //for (; k < count; k++)
    if (count == 1) {
        k = 0;
        const uint8_t* S = src[k];
        int32_t* D = dst[k];
        for (dx = k0; dx < xmax; dx++) {
            int32_t sx = xofs[dx];
            D[dx] = S[sx] * alpha[dx * 2] + S[sx + cn] * alpha[dx * 2 + 1];
        }

        for (; dx < dwidth; dx++)
            D[dx] = (int32_t)S[xofs[dx]] * INTER_RESIZE_COEF_SCALE;
    }
}

void img_vresize_linear_neon_uchar(const int32_t** src, uint8_t* dst, const int16_t* beta, int32_t width)
{
    const int32_t *S0 = src[0], *S1 = src[1];

    int32x4_t qS0_00, qS0_01, qS0_10, qS0_11;
    int16x8_t qS_00, qS_01, q_dst0, q_dst1;

    int16x8_t dBeta_0, dBeta_1, v_delta;
    dBeta_0 = vdupq_n_s16(beta[0]);
    dBeta_1 = vdupq_n_s16(beta[1]);
    v_delta = vdupq_n_s16(2);

    int32_t x = 0;
    for (; x <= width - 16; x += 16) {
        qS0_00 = vshrq_n_s32(vld1q_s32(S0 + x), 4);
        qS0_10 = vshrq_n_s32(vld1q_s32(S1 + x), 4);
        qS0_01 = vshrq_n_s32(vld1q_s32(S0 + x + 4), 4);
        qS0_11 = vshrq_n_s32(vld1q_s32(S1 + x + 4), 4);
        qS_00 = vcombine_s16(vmovn_s32(qS0_00), vmovn_s32(qS0_01));
        qS_01 = vcombine_s16(vmovn_s32(qS0_10), vmovn_s32(qS0_11));

        q_dst0 = vaddq_s16(vshrq_n_s16(vqdmulhq_s16(qS_00, dBeta_0), 1),
                           vshrq_n_s16(vqdmulhq_s16(qS_01, dBeta_1), 1));
        q_dst0 = vshrq_n_s16(vaddq_s16(q_dst0, v_delta), 2);

        qS0_00 = vshrq_n_s32(vld1q_s32(S0 + x + 8), 4);
        qS0_10 = vshrq_n_s32(vld1q_s32(S1 + x + 8), 4);
        qS0_01 = vshrq_n_s32(vld1q_s32(S0 + x + 12), 4);
        qS0_11 = vshrq_n_s32(vld1q_s32(S1 + x + 12), 4);
        qS_00 = vcombine_s16(vmovn_s32(qS0_00), vmovn_s32(qS0_01));
        qS_01 = vcombine_s16(vmovn_s32(qS0_10), vmovn_s32(qS0_11));

        q_dst1 = vaddq_s16(vshrq_n_s16(vqdmulhq_s16(qS_00, dBeta_0), 1),
                           vshrq_n_s16(vqdmulhq_s16(qS_01, dBeta_1), 1));
        q_dst1 = vshrq_n_s16(vaddq_s16(q_dst1, v_delta), 2);

        vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(q_dst0), vqmovun_s16(q_dst1)));
    }

    int16_t b0 = beta[0], b1 = beta[1];
    for (; x < width; x++) {
        dst[x] = (uint8_t)((((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2) >> 2);
    }
}

static void img_resize_generic_linear_neon_uchar(
    const uint8_t* src,
    uint8_t* dst,
    const int32_t* xofs,
    const int16_t* _alpha,
    const int32_t* yofs,
    const int16_t* _beta,
    int32_t xmin,
    int32_t xmax,
    int32_t ksize,
    int32_t srcw,
    int32_t srch,
    int32_t srcstep,
    int32_t dstw,
    int32_t dsth,
    int32_t dststep,
    int32_t channels)
{
    const int16_t* alpha = _alpha;
    const int16_t* beta = _beta;
    int32_t cn = channels;
    srcw *= cn;
    dstw *= cn;

    int32_t bufstep = (int32_t)align_size(dstw, 16);
    //int32_t dststep = (int32_t) align_size (dstw, 4);
    // int32_t dststep = dstw;

    int32_t* buffer_ = (int32_t*)malloc(bufstep * ksize * sizeof(int32_t));

    const uint8_t* srows[MAX_ESIZE];
    int32_t* rows[MAX_ESIZE];
    int32_t prev_sy[MAX_ESIZE];
    int32_t k, dy;
    xmin *= cn;
    xmax *= cn;

    for (k = 0; k < ksize; k++) {
        prev_sy[k] = -1;
        rows[k] = (int32_t*)buffer_ + bufstep * k;
    }

    // image resize is a separable operation. In case of not too strong
    for (dy = 0; dy < dsth; dy++, beta += ksize) {
        int32_t sy0 = yofs[dy], k, k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (k = 0; k < ksize; k++) {
            int32_t sy = img_clip(sy0 - ksize2 + 1 + k, 0, srch);
            for (k1 = FUNC_MAX(k1, k); k1 < ksize; k1++) {
                if (sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy(rows[k], rows[k1], bufstep * sizeof(rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = FUNC_MIN(k0, k); // remember the first row that needs to be computed
            srows[k] = (const uint8_t*)(src + srcstep * sy);
            prev_sy[k] = sy;
        }

        if (k0 < ksize) {
            if (cn == 4)
                img_hresize_4channels_linear_neon_uchar(srows + k0, rows + k0, ksize - k0, xofs, alpha, srcw, dstw, cn, xmin, xmax);
            else
                img_hresize_linear_c_uchar(srows + k0, rows + k0, ksize - k0, xofs, alpha, srcw, dstw, cn, xmin, xmax);
        }
        img_vresize_linear_neon_uchar((const int32_t**)rows, (uint8_t*)(dst + dststep * dy), beta, dstw);
    }

    free(buffer_);
    buffer_ = NULL;
}

void img_resize_bilinear_neon_uchar(
    uint8_t* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const uint8_t* src,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t src_stride,
    uint32_t channels)
{
    if (src_width % 2 == 0 && dst_width == src_width / 2 &&
        src_height % 2 == 0 && dst_height == src_height / 2) {
        img_resize_bilinear_neon_shrink2_u8(dst, dst_width, dst_height, dst_stride, src, src_width, src_height, src_stride, channels);
        return;
    } else if (src_width % 4 == 0 && dst_width == src_width / 4 &&
               src_height % 4 == 0 && dst_height == src_height / 4 && channels != 1) {
        img_resize_bilinear_neon_shrink4_u8(dst, dst_width, dst_height, dst_stride, src, src_width, src_height, src_stride, channels);
        return;
    } else if (src_width % 6 == 0 && dst_width == src_width / 6 &&
               src_height % 6 == 0 && dst_height == src_height / 6) {
        img_resize_bilinear_neon_shrink6_u8(dst, dst_width, dst_height, dst_stride, src, src_width, src_height, src_stride, channels);
        return;
    }
    int32_t dstw = dst_width;
    int32_t dsth = dst_height;
    int32_t srcw = src_width;
    int32_t srch = src_height;

    int32_t cn = channels;

    bool flag = src_height >= dst_height && src_height >= 2 &&
                src_width >= dst_width && src_width >= 2;
    bool flag_c1orc4 = dst_height >= 8 && dst_width >= 8;
    if (3 == cn && flag) {
        resize_linear_generic(3, src_height, src_width, src_stride, src, dst_height, dst_width, dst_stride, dst);
        return;
    } else if (4 == cn && flag_c1orc4) {
        // resize_linear_generic(4, src_height, src_width, src_stride, src, dst_height, dst_width, dst_stride, dst);
        resize_linear_u8c1orc4<4>(src_height, src_width, src_stride, src, dst_height, dst_width, dst_stride, dst);
        return;
    } else if (1 == cn && flag_c1orc4) {
        resize_linear_u8c1orc4<1>(src_height, src_width, src_stride, src, dst_height, dst_width, dst_stride, dst);
        return;
    }

    int32_t xmin = 0;
    int32_t xmax = dstw;
    int32_t width = dstw * cn;
    //float fx, fy;

    int32_t ksize = 0, ksize2;
    ksize = 2;
    ksize2 = ksize / 2;

    uint8_t* buffer_ = (uint8_t*)malloc((width + dsth) * (sizeof(int32_t) + sizeof(float) * ksize));

    int32_t* xofs = (int32_t*)buffer_;
    int32_t* yofs = xofs + width;
    int16_t* ialpha = (int16_t*)(yofs + dsth);
    int16_t* ibeta = ialpha + width * ksize;

    img_resize_cal_offset_linear_uchar(xofs, ialpha, yofs, ibeta, &xmin, &xmax, ksize, ksize2, srcw, srch, dstw, dsth, cn);

    img_resize_generic_linear_neon_uchar(src, dst, xofs, ialpha, yofs, ibeta, xmin, xmax, ksize, srcw, srch, src_stride, dstw, dsth, dst_stride, cn);
    free(buffer_);
    buffer_ = NULL;
}

struct DecimateAlpha {
    int32_t di;
    int32_t si;
    float alpha;
};

template <typename T>
T img_saturate_cast(float x);

template <>
inline float img_saturate_cast<float>(float x)
{
    return (x > FLT_MIN ? (x < FLT_MAX ? x : FLT_MAX) : FLT_MIN);
}
template <>
inline uint8_t img_saturate_cast<uint8_t>(float x)
{
    return (x > 0 ? (x < 255 ? x : 255) : 0);
}

static void img_resize_cal_offset_area_uchar(
    int32_t* xofs,
    short* ialpha,
    int32_t* yofs,
    short* ibeta,
    int32_t* xmin,
    int32_t* xmax,
    int32_t ksize,
    int32_t ksize2,
    int32_t srcw,
    int32_t srch,
    int32_t dstw,
    int32_t dsth,
    int32_t channels)
{
    double inv_scale_x = (double)dstw / srcw;
    double inv_scale_y = (double)dsth / srch;

    int32_t cn = channels;
    double scale_x = 1. / inv_scale_x;
    double scale_y = 1. / inv_scale_y;
    int32_t k, sx, sy, dx, dy;

    float fx, fy;

    float cbuf[MAX_ESIZE];

    for (dx = 0; dx < dstw; dx++) {
        //fx = (float) ( (dx + 0.5) * scale_x - 0.5);
        //sx = img_floor (fx);
        //fx -= sx;
        sx = img_floor(dx * scale_x);
        fx = (float)((dx + 1) - (sx + 1) * inv_scale_x);
        fx = fx <= 0 ? 0.f : fx - img_floor(fx);

        if (sx < ksize2 - 1) {
            *xmin = dx + 1;
            if (sx < 0)
                fx = 0, sx = 0;
        }

        if (sx + ksize2 >= srcw) {
            *xmax = FUNC_MIN(*xmax, dx);
            if (sx >= srcw - 1)
                fx = 0, sx = srcw - 1;
        }

        for (k = 0, sx *= cn; k < cn; k++)
            xofs[dx * cn + k] = sx + k;

        cbuf[0] = 1.f - fx;
        cbuf[1] = fx;

        for (k = 0; k < ksize; k++)
            ialpha[dx * cn * ksize + k] = img_saturate_cast_short(cbuf[k] * INTER_RESIZE_COEF_SCALE);
        for (; k < cn * ksize; k++)
            ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];
    }

    for (dy = 0; dy < dsth; dy++) {
        //fy = (float) ( (dy + 0.5) * scale_y - 0.5);
        //sy = img_floor (fy);
        //fy -= sy;
        sy = img_floor(dy * scale_y);
        fy = (float)((dy + 1) - (sy + 1) * inv_scale_y);
        fy = fy <= 0 ? 0.f : fy - img_floor(fy);

        yofs[dy] = sy;

        cbuf[0] = 1.f - fy;
        cbuf[1] = fy;

        for (k = 0; k < ksize; k++)
            ibeta[dy * ksize + k] = img_saturate_cast_short(cbuf[k] * INTER_RESIZE_COEF_SCALE);
    }
}

void img_resize_area_neon_uchar(
    uint8_t* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const uint8_t* src,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t src_stride,
    uint32_t channels)
{
    int32_t dstw = dst_width;
    int32_t dsth = dst_height;
    int32_t srcw = src_width;
    int32_t srch = src_height;

    int32_t cn = channels; //4;
    //int32_t src_stride = srcw * cn;

    int32_t xmin = 0;
    int32_t xmax = dstw;
    int32_t width = dstw * cn;
    //float fx, fy;

    int32_t ksize = 0, ksize2;
    ksize = 2;
    ksize2 = ksize / 2;

    uint8_t* buffer_ = (uint8_t*)malloc((width + dsth) * (sizeof(int32_t) + sizeof(float) * ksize));

    int32_t* xofs = (int32_t*)buffer_;
    int32_t* yofs = xofs + width;
    short* ialpha = (short*)(yofs + dsth);
    short* ibeta = ialpha + width * ksize;

    img_resize_cal_offset_area_uchar(xofs, ialpha, yofs, ibeta, &xmin, &xmax, ksize, ksize2, srcw, srch, dstw, dsth, cn);

    img_resize_generic_linear_neon_uchar(src, dst, xofs, ialpha, yofs, ibeta, xmin, xmax, ksize, srcw, srch, src_stride, dstw, dsth, dst_stride, cn);
    free(buffer_);
    buffer_ = NULL;
}

template <typename Tsrc, int32_t ncSrc, typename Tdst, int32_t ncDst, int32_t nc>
void resizeAreaFast(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const Tsrc* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    Tdst* outData)
{
    int32_t scale_x = inWidth / outWidth;
    int32_t scale_y = inHeight / outHeight;
    int32_t area = scale_x * scale_y;
    //size_t srcstep = inWidthStride / (sizeof(Tsrc));
    int32_t* _ofs = (int32_t*)malloc((area + outWidth * nc) * sizeof(int32_t));
    int32_t *ofs = _ofs, *xofs = ofs + area;
    for (int32_t sy = 0, k = 0; sy < scale_y; ++sy) {
        for (int32_t sx = 0; sx < scale_x; ++sx) {
            ofs[k++] = (int32_t)(sy * inWidthStride + sx * nc);
        }
    }
    for (int32_t dx = 0; dx < outWidth; ++dx) {
        int32_t j = dx * nc;
        int32_t sx = scale_x * j;
        for (int32_t k = 0; k < nc; ++k) {
            xofs[j + k] = sx + k;
        }
    }
    float scale = 1.0f / area;
    int32_t dwidth = (inWidth / scale_x) * nc;
    inWidth *= nc;
    outWidth *= nc;
    int32_t dy, dx, k = 0;
    for (dy = 0; dy < outHeight; ++dy) {
        Tdst* D = (Tdst*)(outData + outWidthStride * dy);
        int32_t sy0 = dy * scale_y;
        int32_t w = sy0 + scale_y <= inHeight ? dwidth : 0;
        if (sy0 >= inHeight) {
            for (dx = 0; dx < outWidth; ++dx)
                D[dx] = 0;
            continue;
        }
        for (dx = 0; dx < w; ++dx) {
            const Tsrc* S = (const Tsrc*)(inData + inWidthStride * sy0) + xofs[dx];
            float sum = 0;
            for (k = 0; k < area; ++k) {
                sum += S[ofs[k]];
            }
            D[dx] = img_saturate_cast<Tdst>(sum * scale);
        }
        for (; dx < outWidth; ++dx) {
            float sum = 0;
            int32_t count = 0, sx0 = xofs[dx];
            if (sx0 >= inWidth)
                D[dx] = 0;
            for (int32_t sy = 0; sy < scale_y; ++sy) {
                if (sy0 + sy <= inHeight) break;
                const Tsrc* S = (const Tsrc*)(inData + inWidthStride * (sy0 + sy)) + sx0;
                for (int32_t sx = 0; sx < scale_x * nc; sx += nc) {
                    if (sx0 + sx >= inWidth) break;
                    sum += S[sx];
                    ++count;
                }
            }
            D[dx] = img_saturate_cast<Tdst>((float)sum / count);
        }
    }
    free(_ofs);
}

static int32_t computeResizeAreaTab(int32_t src_size, int32_t dst_size, int32_t cn, double scale, DecimateAlpha* tab)
{
    int32_t k = 0;
    for (int32_t dx = 0; dx < dst_size; ++dx) {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min<double>(scale, src_size - fsx1);

        int32_t sx1 = ceil(fsx1), sx2 = floor(fsx2);
        sx2 = std::min<int32_t>(sx2, src_size - 1);
        sx1 = std::min<int32_t>(sx1, sx2);
        if (sx1 - fsx1 > 1e-3) {
            tab[k].di = dx * cn;
            tab[k].si = (sx1 - 1) * cn;
            tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth);
        }

        for (int32_t sx = sx1; sx < sx2; ++sx) {
            tab[k].di = dx * cn;
            tab[k].si = sx * cn;
            tab[k++].alpha = float(1.0 / cellWidth);
        }

        if (fsx2 - sx2 > 1e-3) {
            tab[k].di = dx * cn;
            tab[k].si = sx2 * cn;
            tab[k++].alpha = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}

template <typename Tsrc, int32_t ncSrc, typename Tdst, int32_t ncDst, int32_t nc>
void resizeAreaShrinkx8(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const Tsrc* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    Tdst* outData);

template <>
void resizeAreaShrinkx8<uint8_t, 3, uint8_t, 3, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    const int32_t nc = 3;

    DecimateAlpha* _xytab = (DecimateAlpha*)malloc((inHeight + inWidth) * 2 * sizeof(DecimateAlpha));
    DecimateAlpha *xtab = _xytab, *ytab = xtab + inWidth * 2;

    int32_t xtab_size = computeResizeAreaTab(inWidth, outWidth, nc, double(inWidth) / outWidth, xtab);
    int32_t ytab_size = computeResizeAreaTab(inHeight, outHeight, 1, double(inHeight) / outHeight, ytab);

    // create x direction histgram
    int32_t xtab_hist_num = 0;
    int32_t* xtab_hist = (int32_t*)malloc(xtab_size * sizeof(int32_t));
    for (int32_t i = 1; i < xtab_size; ++i) {
        if (xtab[i].di != xtab[i - 1].di) {
            xtab_hist[xtab_hist_num] = i;
            ++xtab_hist_num;
        }
    }
    xtab_hist[xtab_hist_num] = xtab_size;
    ++xtab_hist_num;

    int32_t* tabofs = (int32_t*)malloc((outHeight + 1) * sizeof(int32_t));
    int32_t k, dy;
    int32_t h;
    for (k = 0, dy = 0; k < ytab_size; ++k) {
        if (k == 0 || ytab[k].di != ytab[k - 1].di) {
            tabofs[dy++] = k;
        }
    }
    tabofs[dy] = ytab_size;

    outWidth *= nc;
    float* _buffer = (float*)malloc(outWidth * 2 * sizeof(float));
    float *buf = _buffer, *sum = buf + outWidth;
    int32_t j_start = tabofs[0], j_end = tabofs[outHeight], j, dx, prev_dy = ytab[j_start].di;

    for (dx = 0; dx < outWidth; ++dx) {
        sum[dx] = 0;
    }
    for (j = j_start; j < j_end; ++j) {
        float beta = ytab[j].alpha;
        int32_t dy = ytab[j].di;
        int32_t sy = ytab[j].si;

        const uint8_t* S = (const uint8_t*)(inData + inWidthStride * sy);
        for (dx = 0; dx < outWidth; ++dx)
            buf[dx] = (uint8_t)0;
        int32_t start_k = 0;
        for (h = 0; h < xtab_hist_num; ++h) {
            int32_t dxn = xtab[start_k].di;
            for (k = start_k; k < xtab_hist[h]; ++k) {
                int32_t sxn = xtab[k].si;
                float alpha = xtab[k].alpha;
                for (int32_t c = 0; c < nc; ++c) {
                    buf[dxn + c] += S[sxn + c] * alpha;
                }
            }
            start_k = xtab_hist[h];
        }
        if (dy != prev_dy) {
            uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
            for (dx = 0; dx < outWidth; ++dx) {
                D[dx] = img_saturate_cast<uint8_t>(sum[dx]);
                sum[dx] = beta * buf[dx];
            }
            prev_dy = dy;
        } else {
            for (dx = 0; dx < outWidth; dx++)
                sum[dx] += beta * buf[dx];
        }
    }

    uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
    for (dx = 0; dx < outWidth; ++dx) {
        D[dx] = (uint8_t)sum[dx];
    }

    free(xtab_hist);
    xtab_hist = NULL;
    free(_xytab);
    _xytab = NULL;
    free(tabofs);
    tabofs = NULL;
    free(_buffer);
    _buffer = NULL;
}

template <>
void resizeAreaShrinkx8<uint8_t, 4, uint8_t, 4, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    const int32_t nc = 4;

    DecimateAlpha* _xytab = (DecimateAlpha*)malloc((inHeight + inWidth) * 2 * sizeof(DecimateAlpha));
    DecimateAlpha *xtab = _xytab, *ytab = xtab + inWidth * 2;

    int32_t xtab_size = computeResizeAreaTab(inWidth, outWidth, nc, double(inWidth) / outWidth, xtab);
    int32_t ytab_size = computeResizeAreaTab(inHeight, outHeight, 1, double(inHeight) / outHeight, ytab);

    // create x direction histgram
    int32_t xtab_hist_num = 0;
    int32_t* xtab_hist = (int32_t*)malloc(xtab_size * sizeof(int32_t));
    for (int32_t i = 1; i < xtab_size; ++i) {
        if (xtab[i].di != xtab[i - 1].di) {
            xtab_hist[xtab_hist_num] = i;
            ++xtab_hist_num;
        }
    }
    xtab_hist[xtab_hist_num] = xtab_size;
    ++xtab_hist_num;

    int32_t* tabofs = (int32_t*)malloc((outHeight + 1) * sizeof(int32_t));
    int32_t k, dy;
    int32_t h;
    for (k = 0, dy = 0; k < ytab_size; ++k) {
        if (k == 0 || ytab[k].di != ytab[k - 1].di) {
            tabofs[dy++] = k;
        }
    }
    tabofs[dy] = ytab_size;

    outWidth *= nc;
    float* _buffer = (float*)malloc(outWidth * 2 * sizeof(float));
    float *buf = _buffer, *sum = buf + outWidth;
    int32_t j_start = tabofs[0], j_end = tabofs[outHeight], j, dx, prev_dy = ytab[j_start].di;

    for (dx = 0; dx < outWidth; ++dx) {
        sum[dx] = 0;
    }
    for (j = j_start; j < j_end; ++j) {
        float beta = ytab[j].alpha;
        int32_t dy = ytab[j].di;
        int32_t sy = ytab[j].si;

        const uint8_t* S = (const uint8_t*)(inData + inWidthStride * sy);
        for (dx = 0; dx < outWidth; ++dx)
            buf[dx] = (uint8_t)0;
        int32_t start_k = 0;
        for (h = 0; h < xtab_hist_num; ++h) {
            int32_t dxn = xtab[start_k].di;
            for (k = start_k; k < start_k + (xtab_hist[h] - start_k) / 4 * 4; k += 4) {
                DecimateAlpha* cur_xtab = &(xtab[k]);
                __asm__ __volatile__(
                    "ldrsw x1, [%0, #4]\n"
                    "add x9, %0, #8\n"
                    "ld1 {v0.s}[0], [x9]\n"
                    "ldrsw x3, [%0, #16]\n"
                    "add x9, %0, #20\n"
                    "ld1 {v0.s}[1], [x9]\n"
                    "ldrsw x5, [%0, #28]\n"
                    "add x9, %0, #32\n"
                    "ld1 {v0.s}[2], [x9]\n"
                    "ldrsw x7, [%0, #40]\n"
                    "add x9, %0, #44\n"
                    "ld1 {v0.s}[3], [x9]\n"
                    "add x1, %2, x1\n"
                    "add x3, %2, x3\n"
                    "add x5, %2, x5\n"
                    "add x7, %2, x7\n"
                    "ld1 {v1.s}[0], [x1]\n"
                    "ld1 {v2.s}[0], [x3]\n"
                    "ld1 {v3.s}[0], [x5]\n"
                    "ld1 {v4.s}[0], [x7]\n"
                    "ld1 {v5.4s}, [%1]\n"
                    "uxtl v1.8h, v1.8b\n"
                    "uxtl v2.8h, v2.8b\n"
                    "uxtl v3.8h, v3.8b\n"
                    "uxtl v4.8h, v4.8b\n"
                    "uxtl v1.4s, v1.4h\n"
                    "uxtl v2.4s, v2.4h\n"
                    "uxtl v3.4s, v3.4h\n"
                    "uxtl v4.4s, v4.4h\n"
                    "ucvtf v1.4s, v1.4s\n"
                    "ucvtf v2.4s, v2.4s\n"
                    "ucvtf v3.4s, v3.4s\n"
                    "ucvtf v4.4s, v4.4s\n"
                    "fmla v5.4s, v1.4s, v0.s[0]\n"
                    "fmul v6.4s, v2.4s, v0.s[1]\n"
                    "fmul v7.4s, v3.4s, v0.s[2]\n"
                    "fmul v8.4s, v4.4s, v0.s[3]\n"
                    "fadd v5.4s, v5.4s, v6.4s\n"
                    "fadd v5.4s, v5.4s, v7.4s\n"
                    "fadd v5.4s, v5.4s, v8.4s\n"
                    "st1 {v5.4s}, [%1]\n"
                    :
                    : "r"(cur_xtab), "r"(buf + dxn), "r"(S)
                    : "cc", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "memory");
            }
            for (k = start_k + (xtab_hist[h] - start_k) / 4 * 4; k < xtab_hist[h]; ++k) {
                int32_t sxn = xtab[k].si;
                float alpha = xtab[k].alpha;
                for (int32_t c = 0; c < nc; ++c) {
                    buf[dxn + c] += S[sxn + c] * alpha;
                }
            }
            start_k = xtab_hist[h];
        }
        if (dy != prev_dy) {
            uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
            for (dx = 0; dx < outWidth; ++dx) {
                D[dx] = img_saturate_cast<uint8_t>(sum[dx]);
                sum[dx] = beta * buf[dx];
            }
            prev_dy = dy;
        } else {
            for (dx = 0; dx < outWidth; dx++)
                sum[dx] += beta * buf[dx];
        }
    }

    uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
    for (dx = 0; dx < outWidth; ++dx) {
        D[dx] = (uint8_t)sum[dx];
    }

    free(xtab_hist);
    xtab_hist = NULL;
    free(_xytab);
    _xytab = NULL;
    free(tabofs);
    tabofs = NULL;
    free(_buffer);
    _buffer = NULL;
}

template <typename Tsrc, int32_t ncSrc, typename Tdst, int32_t ncDst, int32_t nc>
void resizeArea(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const Tsrc* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    Tdst* outData)
{
    if (inWidth % 2 == 0 && outWidth == inWidth / 2 &&
        inHeight % 2 == 0 && outHeight == inHeight / 2) {
        img_resize_bilinear_neon_shrink2_u8(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, nc);
        return;
    }
    if (inWidth % outWidth == 0 && inHeight % outHeight == 0) {
        resizeAreaFast<Tsrc, ncSrc, Tdst, ncDst, nc>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return;
    }

    DecimateAlpha* _xytab = (DecimateAlpha*)malloc((inHeight + inWidth) * 2 * sizeof(DecimateAlpha));
    DecimateAlpha *xtab = _xytab, *ytab = xtab + inWidth * 2;

    int32_t xtab_size = computeResizeAreaTab(inWidth, outWidth, nc, double(inWidth) / outWidth, xtab);
    int32_t ytab_size = computeResizeAreaTab(inHeight, outHeight, 1, double(inHeight) / outHeight, ytab);

    int32_t* tabofs = (int32_t*)malloc((outHeight + 1) * sizeof(int32_t));
    int32_t k, dy;
    for (k = 0, dy = 0; k < ytab_size; ++k) {
        if (k == 0 || ytab[k].di != ytab[k - 1].di) {
            tabofs[dy++] = k;
        }
    }
    tabofs[dy] = ytab_size;

    outWidth *= nc;
    float* _buffer = (float*)malloc(outWidth * 2 * sizeof(float));
    float *buf = _buffer, *sum = buf + outWidth;
    int32_t j_start = tabofs[0], j_end = tabofs[outHeight], j, dx, prev_dy = ytab[j_start].di;

    for (dx = 0; dx < outWidth; ++dx) {
        sum[dx] = 0;
    }
    for (j = j_start; j < j_end; ++j) {
        float beta = ytab[j].alpha;
        int32_t dy = ytab[j].di;
        int32_t sy = ytab[j].si;

        const Tsrc* S = (const Tsrc*)(inData + inWidthStride * sy);
        for (dx = 0; dx < outWidth; ++dx)
            buf[dx] = (Tdst)0;
        for (k = 0; k < xtab_size; ++k) {
            int32_t sxn = xtab[k].si;
            int32_t dxn = xtab[k].di;
            float alpha = xtab[k].alpha;
            for (int32_t c = 0; c < nc; ++c) {
                buf[dxn + c] += S[sxn + c] * alpha;
            }
        }

        if (dy != prev_dy) {
            Tdst* D = (Tdst*)(outData + outWidthStride * prev_dy);
            for (dx = 0; dx < outWidth; ++dx) {
                D[dx] = img_saturate_cast<Tdst>(sum[dx]);
                sum[dx] = beta * buf[dx];
            }
            prev_dy = dy;
        } else {
            for (dx = 0; dx < outWidth; dx++)
                sum[dx] += beta * buf[dx];
        }
    }

    Tdst* D = (Tdst*)(outData + outWidthStride * prev_dy);
    for (dx = 0; dx < outWidth; ++dx) {
        D[dx] = (Tdst)sum[dx];
    }

    free(_xytab);
    _xytab = NULL;
    free(tabofs);
    tabofs = NULL;

    free(_buffer);
    _buffer = NULL;
}

template <>
void resizeArea<uint8_t, 3, uint8_t, 3, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (inWidth % 2 == 0 && outWidth == inWidth / 2 &&
        inHeight % 2 == 0 && outHeight == inHeight / 2) {
        img_resize_bilinear_neon_shrink2_u8(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 3);
        return;
    }
    if (inWidth % outWidth == 0 && inHeight % outHeight == 0) {
        resizeAreaFast<uint8_t, 3, uint8_t, 3, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return;
    }
    if ((inWidth / outWidth) > 8) {
        resizeAreaShrinkx8<uint8_t, 3, uint8_t, 3, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return;
    }
    const int32_t nc = 3;

    DecimateAlpha* _xytab = (DecimateAlpha*)malloc((inHeight + inWidth) * 2 * sizeof(DecimateAlpha));
    DecimateAlpha *xtab = _xytab, *ytab = xtab + inWidth * 2;

    int32_t xtab_size = computeResizeAreaTab(inWidth, outWidth, nc, double(inWidth) / outWidth, xtab);
    int32_t ytab_size = computeResizeAreaTab(inHeight, outHeight, 1, double(inHeight) / outHeight, ytab);

    int32_t* tabofs = (int32_t*)malloc((outHeight + 1) * sizeof(int32_t));
    int32_t k, dy;
    for (k = 0, dy = 0; k < ytab_size; ++k) {
        if (k == 0 || ytab[k].di != ytab[k - 1].di) {
            tabofs[dy++] = k;
        }
    }
    tabofs[dy] = ytab_size;

    outWidth *= nc;
    float* _buffer = (float*)malloc(outWidth * 2 * sizeof(float));
    float *buf = _buffer, *sum = buf + outWidth;
    int32_t j_start = tabofs[0], j_end = tabofs[outHeight], j, dx, prev_dy = ytab[j_start].di;

    for (dx = 0; dx < outWidth; ++dx) {
        sum[dx] = 0;
    }
    for (j = j_start; j < j_end; ++j) {
        float beta = ytab[j].alpha;
        int32_t dy = ytab[j].di;
        int32_t sy = ytab[j].si;

        const uint8_t* S = (const uint8_t*)(inData + inWidthStride * sy);
        for (dx = 0; dx < outWidth; ++dx)
            buf[dx] = (uint8_t)0;
        for (k = 0; k < xtab_size; ++k) {
            int32_t sxn = xtab[k].si;
            int32_t dxn = xtab[k].di;
            float alpha = xtab[k].alpha;
            for (int32_t c = 0; c < nc; ++c) {
                buf[dxn + c] += S[sxn + c] * alpha;
            }
        }
        if (dy != prev_dy) {
            uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
            for (dx = 0; dx < outWidth; ++dx) {
                D[dx] = img_saturate_cast<uint8_t>(sum[dx]);
                sum[dx] = beta * buf[dx];
            }
            prev_dy = dy;
        } else {
            for (dx = 0; dx < outWidth; dx++)
                sum[dx] += beta * buf[dx];
        }
    }

    uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
    for (dx = 0; dx < outWidth; ++dx) {
        D[dx] = (uint8_t)sum[dx];
    }

    free(_xytab);
    _xytab = NULL;
    free(tabofs);
    tabofs = NULL;
    free(_buffer);
    _buffer = NULL;
}

template <>
void resizeArea<uint8_t, 4, uint8_t, 4, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (inWidth % 2 == 0 && outWidth == inWidth / 2 &&
        inHeight % 2 == 0 && outHeight == inHeight / 2) {
        img_resize_bilinear_neon_shrink2_u8(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 4);
        return;
    }
    if (inWidth % outWidth == 0 && inHeight % outHeight == 0) {
        resizeAreaFast<uint8_t, 4, uint8_t, 4, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return;
    }
    if ((inWidth / outWidth) > 8) {
        resizeAreaShrinkx8<uint8_t, 4, uint8_t, 4, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return;
    }

    const int32_t nc = 4;

    DecimateAlpha* _xytab = (DecimateAlpha*)malloc((inHeight + inWidth) * 2 * sizeof(DecimateAlpha));
    DecimateAlpha *xtab = _xytab, *ytab = xtab + inWidth * 2;

    int32_t xtab_size = computeResizeAreaTab(inWidth, outWidth, nc, double(inWidth) / outWidth, xtab);
    int32_t ytab_size = computeResizeAreaTab(inHeight, outHeight, 1, double(inHeight) / outHeight, ytab);

    int32_t* tabofs = (int32_t*)malloc((outHeight + 1) * sizeof(int32_t));
    int32_t k, dy;
    for (k = 0, dy = 0; k < ytab_size; ++k) {
        if (k == 0 || ytab[k].di != ytab[k - 1].di) {
            tabofs[dy++] = k;
        }
    }
    tabofs[dy] = ytab_size;

    outWidth *= nc;
    float* _buffer = (float*)malloc(outWidth * 2 * sizeof(float));
    float *buf = _buffer, *sum = buf + outWidth;
    int32_t j_start = tabofs[0], j_end = tabofs[outHeight], j, dx, prev_dy = ytab[j_start].di;

    for (dx = 0; dx < outWidth; ++dx) {
        sum[dx] = 0;
    }
    for (j = j_start; j < j_end; ++j) {
        float beta = ytab[j].alpha;
        int32_t dy = ytab[j].di;
        int32_t sy = ytab[j].si;

        const uint8_t* S = (const uint8_t*)(inData + inWidthStride * sy);
        for (dx = 0; dx < outWidth; ++dx)
            buf[dx] = (uint8_t)0;
        for (k = 0; k < xtab_size; ++k) {
            int32_t sxn = xtab[k].si;
            int32_t dxn = xtab[k].di;
            float alpha = xtab[k].alpha;
            for (int32_t c = 0; c < nc; ++c) {
                buf[dxn + c] += S[sxn + c] * alpha;
            }
        }
        if (dy != prev_dy) {
            uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
            for (dx = 0; dx < outWidth; ++dx) {
                D[dx] = img_saturate_cast<uint8_t>(sum[dx]);
                sum[dx] = beta * buf[dx];
            }
            prev_dy = dy;
        } else {
            for (dx = 0; dx < outWidth; dx++)
                sum[dx] += beta * buf[dx];
        }
    }

    uint8_t* D = (uint8_t*)(outData + outWidthStride * prev_dy);
    for (dx = 0; dx < outWidth; ++dx) {
        D[dx] = (uint8_t)sum[dx];
    }

    free(_xytab);
    _xytab = NULL;
    free(tabofs);
    tabofs = NULL;
    free(_buffer);
    _buffer = NULL;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    resizeNearestPoint<uint8_t, uint8_t, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    resizeNearestPoint<uint8_t, uint8_t, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    resizeNearestPoint<uint8_t, uint8_t, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    img_resize_bilinear_neon_uchar(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 1);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    img_resize_bilinear_neon_uchar(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 3);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    img_resize_bilinear_neon_uchar(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 4);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeArea<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (inHeight < outHeight || inWidth < outWidth) {
        img_resize_area_neon_uchar(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 1);
        return ppl::common::RC_SUCCESS;
    } else {
        resizeArea<uint8_t, 1, uint8_t, 1, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight < outHeight || inWidth < outWidth) {
        img_resize_area_neon_uchar(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 3);
        return ppl::common::RC_SUCCESS;
    } else {
        resizeArea<uint8_t, 3, uint8_t, 3, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight < outHeight || inWidth < outWidth) {
        img_resize_area_neon_uchar(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 4);
        return ppl::common::RC_SUCCESS;
    } else {
        resizeArea<uint8_t, 4, uint8_t, 4, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

}
}
} // namespace ppl::cv::arm
