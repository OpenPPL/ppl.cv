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

namespace ppl::cv::arm {

template <typename T, int32_t nc>
void imgRotate90degree(int32_t inHeight,
                       int32_t inWidth,
                       int32_t inWidthStride,
                       const T* inData,
                       int32_t outHeight,
                       int32_t outWidth,
                       int32_t outWidthStride,
                       T* outData)
{
    const int32_t BC = 64;
    const int32_t BR = 64;
    int32_t numColsEnd = outWidth - outWidth % BC;
    int32_t numRowsEnd = outHeight - outHeight % BR;
    for (int32_t i = 0; i < numRowsEnd; i += BR) {
        for (int32_t j = 0; j < numColsEnd; j += BC) {
            for (int32_t ii = 0; ii < BR; ii++) {
                for (int32_t jj = 0; jj < BC; jj++) {
                    for (int32_t c = 0; c < nc; c++) {
                        outData[(i + ii) * outWidthStride + (j + jj) * nc + c] =
                            inData[(inHeight - (j + jj) - 1) * inWidthStride + (i + ii) * nc + c];
                    }
                }
            }
        }
    }
    for (int32_t i = numRowsEnd; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] = inData[(inHeight - j - 1) * inWidthStride + i * nc + c];
            }
        }
    }
    for (int32_t i = 0; i < numRowsEnd; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] = inData[(inHeight - j - 1) * inWidthStride + i * nc + c];
            }
        }
    }
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

template <typename T, int32_t nc>
void imgRotate270degree(int32_t inHeight,
                        int32_t inWidth,
                        int32_t inWidthStride,
                        const T* inData,
                        int32_t outHeight,
                        int32_t outWidth,
                        int32_t outWidthStride,
                        T* outData)
{
    const int32_t BC = 64;
    const int32_t BR = 64;
    int32_t numColsEnd = outWidth - outWidth % BC;
    int32_t numRowsEnd = outHeight - outHeight % BR;
    for (int32_t i = 0; i < numRowsEnd; i += BR) {
        for (int32_t j = 0; j < numColsEnd; j += BC) {
            for (int32_t ii = 0; ii < BR; ii++) {
                for (int32_t jj = 0; jj < BC; jj++) {
                    for (int32_t c = 0; c < nc; c++) {
                        outData[(i + ii) * outWidthStride + (j + jj) * nc + c] =
                            inData[(j + jj) * inWidthStride + (inWidth - (i + ii) - 1) * nc + c];
                    }
                }
            }
        }
    }
    for (int32_t i = numRowsEnd; i < outHeight; i++) {
        for (int32_t j = 0; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] = inData[j * inWidthStride + (inWidth - i - 1) * nc + c];
            }
        }
    }
    for (int32_t i = 0; i < numRowsEnd; i++) {
        for (int32_t j = numColsEnd; j < outWidth; j++) {
            for (int32_t c = 0; c < nc; c++) {
                outData[i * outWidthStride + j * nc + c] = inData[j * inWidthStride + (inWidth - i - 1) * nc + c];
            }
        }
    }
}

template <typename T, int32_t nc>
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
    if (degree == 90) {
        imgRotate90degree<T, nc>(
            inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    } else if (degree == 180) {
        imgRotate180degree<T, nc>(
            inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    } else if (degree == 270) {
        imgRotate270degree<T, nc>(
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

} // namespace ppl::cv::arm
