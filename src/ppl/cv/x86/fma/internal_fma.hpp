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

#ifndef PPL_CV_X86_INTERNAL_FMA_H_
#define PPL_CV_X86_INTERNAL_FMA_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"
#include <stdint.h>

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

int32_t resize_linear_twoline_fp32_fma(
    int32_t max_length,
    int32_t channels,
    const float *in_data_0,
    const float *in_data_1,
    const int32_t *w_offset,
    const float *w_coeff,
    float h_coeff,
    float *row_0,
    float *row_1,
    float *out_data);

int32_t resize_linear_w_oneline_c1_u8_fma(
    int32_t in_width,
    const uint8_t *in_data,
    int32_t out_width,
    const int32_t *w_offset,
    const int16_t *w_coeff,
    int16_t COEFF_SUM,
    int32_t *row);

void resize_linear_kernel_c1_shrink_u8_fma(
    int32_t in_height,
    int32_t in_width,
    int32_t in_stride,
    const uint8_t *in_data,
    int32_t out_height,
    int32_t out_width,
    int32_t out_stride,
    const int32_t *h_offset,
    const int32_t *w_offset,
    int16_t *h_coeff,
    int16_t *w_coeff,
    int16_t INTER_RESIZE_COEF_SCALE,
    uint8_t *out_data);

int32_t resize_linear_shrink2_oneline_c1_kernel_u8_fma(
    const uint8_t *in_ptr,
    int32_t in_stride,
    int32_t out_width,
    uint8_t *out_ptr);

int32_t resize_linear_w_oneline_c3_u8_fma(
    int32_t in_width,
    const uint8_t *in_data,
    int32_t out_width,
    const int32_t *w_offset,
    const int16_t *w_coeff,
    int16_t COEFF_SUM,
    int32_t *row);

int32_t resize_linear_w_oneline_c4_u8_fma(
    int32_t in_width,
    const uint8_t *in_data,
    int32_t out_width,
    const int32_t *w_offset,
    const int16_t *w_coeff,
    int16_t COEFF_SUM,
    int32_t *row);

int32_t resize_linear_shrink2_oneline_c4_kernel_u8_fma(
    const uint8_t *in_ptr,
    int32_t in_stride,
    int32_t out_width,
    uint8_t *out_ptr);

template <int32_t dstcn, int32_t blueIdx>
::ppl::common::RetCode i420_2_rgb(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUStride,
    const uint8_t *inUV,
    int32_t inVStride,
    const uint8_t *inV,
    int32_t outWidthStride,
    uint8_t *outData);

template <int32_t channels>
::ppl::common::RetCode addWighted_fma(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData);

template <typename T, int32_t channels>
::ppl::common::RetCode Add_fma(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData);

template <typename T, int32_t channels>
::ppl::common::RetCode Mul_fma(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData,
    float alpha);

template <int32_t channels>
::ppl::common::RetCode Subtract(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData);

::ppl::common::RetCode BGR2GRAY(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    bool reverse_channel);

template <int32_t dstcn, int32_t blueIdx, bool isUV>
::ppl::common::RetCode nv_2_rgb(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData);

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T *dst,
    const T *src,
    const double *M,
    T delta);

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_nearest(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T *dst,
    const T *src,
    const double *M,
    T delta);

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpperspective_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T *dst,
    const T *src,
    const double M[][3],
    T delta);

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpperspective_nearest(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T *dst,
    const T *src,
    const double M[][3],
    T delta);

template <typename T, int32_t nc>
::ppl::common::RetCode splitAOS2SOA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T *in,
    int32_t outWidthStride,
    T **out);

template <int32_t filterSize>
void convolution_f(
    int32_t imageInSizeX,
    int32_t imageInSizeY,
    int32_t inWidthStride,
    float *imageIn, /*do copy make border inside */
    const float *filter,
    int32_t outWidthStride,
    float *imageOut,
    int32_t cn,
    const float *src,
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    BorderType border_type);

template <int32_t filterSize>
void convolution_b(
    int32_t imageInSizeX,
    int32_t imageInSizeY,
    int32_t inWidthStride,
    uint8_t *imageIn, /*do copy make border inside */
    const float *filter,
    int32_t outWidthStride,
    uint8_t *imageOut,
    int32_t cn,
    const uint8_t *src,
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    BorderType border_type);

void convolution_b_r(
    int32_t imageInSizeX,
    int32_t imageInSizeY,
    int32_t inWidthStride,
    uint8_t *imageIn,
    int32_t filterSize,
    const float *filter,
    int32_t outWidthStride,
    uint8_t *imageOut,
    int32_t cn,
    const uint8_t *src,
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    BorderType border_type);

void convolution_f_r(
    int32_t imageInSizeX,
    int32_t imageInSizeY,
    int32_t inWidthStride,
    float *imageIn,
    int32_t filterSize,
    const float *filter,
    int32_t outWidthStride,
    float *imageOut,
    int32_t cn,
    const float *src,
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcWidthStride,
    BorderType border_type);

template <typename T, int32_t nc>
void mergeSOA2AOS(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T **in,
    int32_t outWidthStride,
    T *out);

}}}} // namespace ppl::cv::x86::fma
#endif //! PPL_CV_X86_INTERNAL_FMA_H_
