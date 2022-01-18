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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <float.h>
#include <limits.h>
#include "ppl/cv/arm/resize.h"
#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"
#include "operation_utils.hpp"
#include <algorithm>

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
void resizeNearestPoint<float, float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData) // resize_nereast_f32c1
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
    for (y = 0; y < outHeight; y++) {
        float* D = outData + y * outWidthStride;
        int32_t sy = std::min(int32_t(y * ify), inHeight - 1);
        const float* S = inData + sy * inWidthStride;
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
void resizeNearestPoint<float, float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData) // resize_nereast_f32c3
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
        float* D = outData + y * outWidthStride;
        int32_t sy = std::min(int32_t(y * ify), inHeight - 1);
        const float* S = inData + sy * inWidthStride;
        for (x = 0; x + 4 <= outWidth; x += 4) {
            int32_t x0 = x_ofs[x];
            int32_t x1 = x_ofs[x + 1];
            int32_t x2 = x_ofs[x + 2];
            int32_t x3 = x_ofs[x + 3];
            int32_t xd = x * nc;
            D[xd + 0] = S[x0], D[xd + 1] = S[x0 + 1], D[xd + 2] = S[x0 + 2];
            D[xd + 3] = S[x1], D[xd + 4] = S[x1 + 1], D[xd + 5] = S[x1 + 2];
            D[xd + 6] = S[x2], D[xd + 7] = S[x2 + 1], D[xd + 8] = S[x2 + 2];
            D[xd + 9] = S[x3], D[xd + 10] = S[x3 + 1], D[xd + 11] = S[x3 + 2];
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
void resizeNearestPoint<float, float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData) // resize_nereast_f32c4
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
        float* D = outData + y * outWidthStride;
        int32_t sy = std::min(int32_t(y * ify), inHeight - 1);
        const float* S = inData + sy * inWidthStride;
        for (x = 0; x + 4 <= outWidth; x += 4) {
            int32_t x0 = x_ofs[x];
            int32_t x1 = x_ofs[x + 1];
            int32_t x2 = x_ofs[x + 2];
            int32_t x3 = x_ofs[x + 3];
            int32_t xd = x * nc;
            float32x4_t v_0 = vld1q_f32(S + x0);
            float32x4_t v_1 = vld1q_f32(S + x1);
            float32x4_t v_2 = vld1q_f32(S + x2);
            float32x4_t v_3 = vld1q_f32(S + x3);
            vst1q_f32(D + xd, v_0);
            vst1q_f32(D + xd + 4, v_1);
            vst1q_f32(D + xd + 8, v_2);
            vst1q_f32(D + xd + 12, v_3);
        }
        for (; x < outWidth; x++) {
            int32_t x0 = x_ofs[x];
            int32_t xd = x * nc;
            float32x4_t v = vld1q_f32(S + x0);
            vst1q_f32(D + xd, v);
        }
    }
    free(x_ofs);
}

static void img_resize_cal_offset_linear_f32(
    int32_t* xofs,
    float* ialpha,
    int32_t* yofs,
    float* ibeta,
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
            ialpha[dx * cn * ksize + k] = cbuf[k];
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
            ibeta[dy * ksize + k] = cbuf[k];
    }
}

void img_hresize_4channels_linear_neon_f32(
    const float** src,
    float** dst,
    int32_t count,
    const int32_t* xofs,
    const float* alpha,
    int32_t swidth,
    int32_t dwidth,
    int32_t cn,
    int32_t xmin,
    int32_t xmax)
{
    int32_t dx, k;
    int32_t dx0 = 0;

    float32x4x2_t alpha_vec;

    float32x4_t qS_00, qS_01, qS_10, qS_11;
    float32x4_t qT0, qT1;

    //for (k = 0; k <= count - 2; k++)
    if (count == 2) {
        k = 0;
        const float *S0 = src[k], *S1 = src[k + 1];
        float *D0 = dst[k], *D1 = dst[k + 1];

        for (dx = dx0; dx < xmax; dx += 4) {
            int32_t sx = xofs[dx];

            alpha_vec = vld2q_f32(&alpha[dx * 2]);

            qS_00 = vld1q_f32(&S0[sx]);
            qS_01 = vld1q_f32(&S0[sx + 4]);
            qS_10 = vld1q_f32(&S1[sx]);
            qS_11 = vld1q_f32(&S1[sx + 4]);

            qT0 = vmulq_f32(qS_00, alpha_vec.val[0]);
            qT0 = vmlaq_f32(qT0, qS_01, alpha_vec.val[1]);
            qT1 = vmulq_f32(qS_10, alpha_vec.val[0]);
            qT1 = vmlaq_f32(qT1, qS_11, alpha_vec.val[1]);

            vst1q_f32(&D0[dx], qT0);
            vst1q_f32(&D1[dx], qT1);
        }

        for (; dx < dwidth; dx += 4) {
            int32_t sx = xofs[dx];

            vst1q_f32(&D0[dx], vld1q_f32(&S0[sx]));
            vst1q_f32(&D1[dx], vld1q_f32(&S1[sx]));
        }
    }

    //for (; k < count; k++)
    if (count == 1) {
        k = 0;
        const float* S = src[k];
        float* D = dst[k];
        for (dx = 0; dx < xmax; dx += 4) {
            int32_t sx = xofs[dx];

            alpha_vec = vld2q_f32(&alpha[dx * 2]);

            qS_00 = vld1q_f32(&S[sx]);
            qS_01 = vld1q_f32(&S[sx + 4]);

            qT0 = vmulq_f32(qS_00, alpha_vec.val[0]);
            qT0 = vmlaq_f32(qT0, qS_01, alpha_vec.val[1]);

            vst1q_f32(&D[dx], qT0);
        }

        for (; dx < dwidth; dx += 4) {
            int32_t sx = xofs[dx];
            vst1q_f32(&D[dx], vld1q_f32(&S[sx]));
        }
    }
}

static void img_hresize_linear_c_f32(
    const float** src,
    float** dst,
    int32_t count,
    const int32_t* xofs,
    const float* alpha,
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
        const float *S0 = src[k], *S1 = src[k + 1];
        float *D0 = dst[k], *D1 = dst[k + 1];
        for (dx = k0; dx < xmax; dx++) {
            int32_t sx = xofs[dx];
            float a0 = alpha[dx * 2], a1 = alpha[dx * 2 + 1];
            float t0 = S0[sx] * a0 + S0[sx + cn] * a1;
            float t1 = S1[sx] * a0 + S1[sx + cn] * a1;
            D0[dx] = t0;
            D1[dx] = t1;
        }

        for (; dx < dwidth; dx++) {
            int32_t sx = xofs[dx];
            D0[dx] = S0[sx];
            D1[dx] = S1[sx];
        }
    }

    //for (; k < count; k++)
    if (count == 1) {
        k = 0;
        const float* S = src[k];
        float* D = dst[k];
        for (dx = k0; dx < xmax; dx++) {
            int32_t sx = xofs[dx];
            D[dx] = S[sx] * alpha[dx * 2] + S[sx + cn] * alpha[dx * 2 + 1];
        }

        for (; dx < dwidth; dx++)
            D[dx] = S[xofs[dx]];
    }
}

void img_vresize_linear_neon_f32(
    const float** src,
    float* dst,
    const float* beta,
    int32_t width)
{
    const float *S0 = src[0], *S1 = src[1];

    float32x4_t qS_00, qS_01, qS_02, qS_03;
    float32x4_t qS_10, qS_11, qS_12, qS_13;
    float32x4_t qT0, qT1, qT2, qT3;

    float32x4_t qv0, qv1;
    qv0 = vdupq_n_f32(beta[0]);
    qv1 = vdupq_n_f32(beta[1]);

    int32_t x = 0;
    for (; x <= width - 16; x += 16) {
        qS_00 = vld1q_f32(S0 + x);
        qS_01 = vld1q_f32(S0 + x + 4);
        qS_02 = vld1q_f32(S0 + x + 8);
        qS_03 = vld1q_f32(S0 + x + 12);

        qS_10 = vld1q_f32(S1 + x);
        qS_11 = vld1q_f32(S1 + x + 4);
        qS_12 = vld1q_f32(S1 + x + 8);
        qS_13 = vld1q_f32(S1 + x + 12);

        qT0 = vmulq_f32(qS_00, qv0);
        qT0 = vmlaq_f32(qT0, qS_10, qv1);

        qT1 = vmulq_f32(qS_01, qv0);
        qT1 = vmlaq_f32(qT1, qS_11, qv1);

        qT2 = vmulq_f32(qS_02, qv0);
        qT2 = vmlaq_f32(qT2, qS_12, qv1);

        qT3 = vmulq_f32(qS_03, qv0);
        qT3 = vmlaq_f32(qT3, qS_13, qv1);

        vst1q_f32(dst + x, qT0);
        vst1q_f32(dst + x + 4, qT1);
        vst1q_f32(dst + x + 8, qT2);
        vst1q_f32(dst + x + 12, qT3);
    }

    float b0 = beta[0], b1 = beta[1];
    for (; x < width; x++) {
        dst[x] = b0 * S0[x] + b1 * S1[x];
    }
}

static void img_resize_generic_linear_neon_f32(
    const float* src,
    float* dst,
    const int32_t* xofs,
    const float* _alpha,
    const int32_t* yofs,
    const float* _beta,
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
    const float* alpha = _alpha;
    const float* beta = _beta;
    int32_t cn = channels;
    srcw *= cn;
    dstw *= cn;

    int32_t bufstep = (int32_t)align_size(dstw, 16);
    //int32_t dststep = (int32_t) align_size (dstw, 4);
    //int32_t dststep = dstw;

    float* buffer_ = (float*)malloc(bufstep * ksize * sizeof(float));

    const float* srows[MAX_ESIZE];
    float* rows[MAX_ESIZE];
    int32_t prev_sy[MAX_ESIZE];
    int32_t k, dy;
    xmin *= cn;
    xmax *= cn;

    for (k = 0; k < ksize; k++) {
        prev_sy[k] = -1;
        rows[k] = (float*)buffer_ + bufstep * k;
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
            srows[k] = (const float*)(src + srcstep * sy);
            prev_sy[k] = sy;
        }

        //printf("--->calc %d: sy0 = %d, k0 = %d.\n", dy, sy0, k0);

        if (k0 < ksize) {
            if (cn == 4)
                img_hresize_4channels_linear_neon_f32(srows + k0, rows + k0, ksize - k0, xofs, alpha, srcw, dstw, cn, xmin, xmax);
            else
                img_hresize_linear_c_f32(srows + k0, rows + k0, ksize - k0, xofs, alpha, srcw, dstw, cn, xmin, xmax);
        }
        img_vresize_linear_neon_f32((const float**)rows, (float*)(dst + dststep * dy), beta, dstw);
    }

    free(buffer_);
    buffer_ = NULL;
}

bool img_resize_bilinear_neon_shrink2_f32(
    float* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const float* src,
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
            const float* row1 = src + (2 * i) * src_stride;
            const float* row2 = src + (2 * i + 1) * src_stride;
            int32_t j = 0;
            for (; j <= dstw - 4; j += 4) {
                float32x4x2_t q0 = vld2q_f32(row1 + 2 * j);
                prefetch_l1(row1, j * 2 + 256);
                float32x4x2_t q1 = vld2q_f32(row2 + 2 * j);
                prefetch_l1(row2, j * 2 + 256);
                float32x4_t q00 = q0.val[0];
                float32x4_t q01 = q0.val[1];
                float32x4_t q10 = q1.val[0];
                float32x4_t q11 = q1.val[1];
                float32x4_t res_f32 = vmulq_f32(vaddq_f32(vaddq_f32(q00, q01), vaddq_f32(q10, q11)), vdupq_n_f32(0.25));
                vst1q_f32(dst + i * dst_stride + j, res_f32);
            }
            for (; j < dstw; j++) {
                dst[i * dst_stride + j] = (row1[j * 2 + 0] + row1[j * 2 + 1] +
                                           row2[j * 2 + 0] + row2[j * 2 + 1] + 2) *
                                          0.25;
            }
        }
    } else if (cn == 4) {
        for (int32_t i = 0; i < dsth; i++) {
            const float* row1 = src + (2 * i) * src_stride;
            const float* row2 = src + (2 * i + 1) * src_stride;

            for (int32_t j = 0; j < dstw; j++) {
                float32x4_t q0 = vld1q_f32(row1 + j * 8);
                prefetch_l1(row1, j * 8 + 256);
                float32x4_t q1 = vld1q_f32(row2 + j * 8);
                prefetch_l1(row2, j * 8 + 256);
                float32x4_t q2 = vld1q_f32(row1 + j * 8 + 4);
                float32x4_t q3 = vld1q_f32(row2 + j * 8 + 4);
                float32x4_t res_f32 = vmulq_f32(vaddq_f32(vaddq_f32(q0, q1), vaddq_f32(q2, q3)), vdupq_n_f32(0.25));
                vst1q_f32(dst + i * dst_stride + j * 4, res_f32);
            }
        }
    } else {
        for (int32_t i = 0; i < dsth; i++) {
            const float* row1 = src + (2 * i) * src_stride;
            const float* row2 = src + (2 * i + 1) * src_stride;
            int32_t j = 0;
            for (; j < dstw; j++) {
                for (int32_t c = 0; c < cn; c++) {
                    dst[i * dst_stride + j * cn + c] = (row1[j * 2 * cn + c] + row1[j * 2 * cn + cn + c] +
                                                        row2[j * 2 * cn + c] + row2[j * 2 * cn + cn + c] + 2) *
                                                       0.25;
                }
            }
        }
    }
    return true;
}

void img_resize_bilinear_neon_f32(
    float* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const float* src,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t src_stride,
    uint32_t channels)
{
    if (src_width % 2 == 0 && dst_width == src_width / 2 &&
        src_height % 2 == 0 && dst_height == src_height / 2 && (channels == 1 || channels == 4)) {
        img_resize_bilinear_neon_shrink2_f32(dst, dst_width, dst_height, dst_stride, src, src_width, src_height, src_stride, channels);
        return;
    }
    int32_t dstw = dst_width;
    int32_t dsth = dst_height;
    int32_t srcw = src_width;
    int32_t srch = src_height;

    int32_t cn = channels; //4;
    // int32_t src_stride = srcw * cn;

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
    float* ialpha = (float*)(yofs + dsth);
    float* ibeta = ialpha + width * ksize;

    img_resize_cal_offset_linear_f32(xofs, ialpha, yofs, ibeta, &xmin, &xmax, ksize, ksize2, srcw, srch, dstw, dsth, cn);

    img_resize_generic_linear_neon_f32(src, dst, xofs, ialpha, yofs, ibeta, xmin, xmax, ksize, srcw, srch, src_stride, dstw, dsth, dst_stride, cn);

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

static void img_resize_cal_offset_area_f32(
    int32_t* xofs,
    float* ialpha,
    int32_t* yofs,
    float* ibeta,
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
            ialpha[dx * cn * ksize + k] = cbuf[k];
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
            ibeta[dy * ksize + k] = cbuf[k];
    }
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
        inHeight % 2 == 0 && outHeight == inHeight / 2 && (nc == 1 || nc == 4)) {
        img_resize_bilinear_neon_shrink2_f32(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, nc);
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

    //invoker's operator
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

void img_resize_area_neon_f32(
    float* dst,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t dst_stride,
    const float* src,
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
    // int32_t src_stride = srcw * cn;

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
    float* ialpha = (float*)(yofs + dsth);
    float* ibeta = ialpha + width * ksize;

    img_resize_cal_offset_area_f32(xofs, ialpha, yofs, ibeta, &xmin, &xmax, ksize, ksize2, srcw, srch, dstw, dsth, cn);

    img_resize_generic_linear_neon_f32(src, dst, xofs, ialpha, yofs, ibeta, xmin, xmax, ksize, srcw, srch, src_stride, dstw, dsth, dst_stride, cn);

    free(buffer_);
    buffer_ = NULL;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    resizeNearestPoint<float, float, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    resizeNearestPoint<float, float, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    resizeNearestPoint<float, float, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    img_resize_bilinear_neon_f32(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 1);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    img_resize_bilinear_neon_f32(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 3);
    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode ResizeLinear<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    img_resize_bilinear_neon_f32(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 4);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeArea<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight < outHeight || inWidth < outWidth) {
        img_resize_area_neon_f32(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 1);
        return ppl::common::RC_SUCCESS;
    } else {
        resizeArea<float, 1, float, 1, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight < outHeight || inWidth < outWidth) {
        img_resize_area_neon_f32(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 3);
        return ppl::common::RC_SUCCESS;
    } else {
        resizeArea<float, 3, float, 3, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight < outHeight || inWidth < outWidth) {
        img_resize_area_neon_f32(outData, outWidth, outHeight, outWidthStride, inData, inWidth, inHeight, inWidthStride, 4);
        return ppl::common::RC_SUCCESS;
    } else {
        resizeArea<float, 4, float, 4, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

}
}
} // namespace ppl::cv::arm
