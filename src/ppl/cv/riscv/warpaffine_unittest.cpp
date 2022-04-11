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

#include "ppl/cv/riscv/warpaffine.h"
#include "ppl/cv/riscv/cvtcolor.h"
#include "ppl/cv/riscv/test.h"
#include "ppl/cv/riscv/util.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

static inline int32_t saturate_cast(double value)
{
    int32_t round2zero = (int32_t)value;
    if (value >= 0) {
        return (value - round2zero != 0.5) ? (int32_t)(value + 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero + 1;
    } else {
        return (round2zero - value != 0.5) ? (int32_t)(value - 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero - 1;
    }
}
static inline int32_t floor(float value)
{
    int32_t i = (int32_t)value;
    return i - (i > value);
}

static inline short saturate_cast(int32_t v)
{
    return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX
                                                                               : SHRT_MIN);
}

// from ppl/cv/x86/warpaffine.cpp
template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
static void warpaffine_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* dst,
    const T* src,
    const double* M,
    T delta)
{
    for (int32_t i = 0; i < outHeight; i++) {
        float base_x = M[1] * i + M[2];
        float base_y = M[4] * i + M[5];
        for (int32_t j = 0; j < outWidth; j++) {
            float x = base_x + M[0] * j;
            float y = base_y + M[3] * j;
            int32_t sx0 = (int32_t)x;
            int32_t sy0 = (int32_t)y;

            float u = x - sx0;
            float v = y - sy0;

            float tab[4];
            float taby[2], tabx[2];
            float v0, v1, v2, v3;
            taby[0] = 1.0f - v;
            taby[1] = v;
            tabx[0] = 1.0f - u;
            tabx[1] = u;

            tab[0] = taby[0] * tabx[0];
            tab[1] = taby[0] * tabx[1];
            tab[2] = taby[1] * tabx[0];
            tab[3] = taby[1] * tabx[1];

            int32_t idxDst = (i * outWidthStride + j * nc);

            if (borderMode == ppl::cv::BORDER_CONSTANT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                for (int32_t k = 0; k < nc; k++) {
                    int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                    int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                    v0 = flag0 ? src[position1 + k] : delta;
                    v1 = flag1 ? src[position1 + nc + k] : delta;
                    v2 = flag2 ? src[position2 + k] : delta;
                    v3 = flag3 ? src[position2 + nc + k] : delta;
                    float sum = 0;
                    sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                    dst[idxDst + k] = static_cast<T>(sum);
                }
            } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                int32_t sx1 = sx0 + 1;
                int32_t sy1 = sy0 + 1;
                sx0 = ppl::cv::riscv::clip(sx0, 0, inWidth - 1);
                sx1 = ppl::cv::riscv::clip(sx1, 0, inWidth - 1);
                sy0 = ppl::cv::riscv::clip(sy0, 0, inHeight - 1);
                sy1 = ppl::cv::riscv::clip(sy1, 0, inHeight - 1);
                const T* t0 = src + sy0 * inWidthStride + sx0 * nc;
                const T* t1 = src + sy0 * inWidthStride + sx1 * nc;
                const T* t2 = src + sy1 * inWidthStride + sx0 * nc;
                const T* t3 = src + sy1 * inWidthStride + sx1 * nc;
                for (int32_t k = 0; k < nc; ++k) {
                    float sum = 0;
                    sum += t0[k] * tab[0] + t1[k] * tab[1] + t2[k] * tab[2] + t3[k] * tab[3];
                    dst[idxDst + k] = static_cast<T>(sum);
                }
            } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                if (flag0 && flag1 && flag2 && flag3) {
                    for (int32_t k = 0; k < nc; k++) {
                        int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                        int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                        v0 = src[position1 + k];
                        v1 = src[position1 + nc + k];
                        v2 = src[position2 + k];
                        v3 = src[position2 + nc + k];
                        float sum = 0;
                        sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                        dst[idxDst + k] = static_cast<T>(sum);
                    }
                } else {
                    continue;
                }
            }
        }
    }
}

template <typename T, int32_t nc, ppl::cv::InterpolationType inter_mode, ppl::cv::BorderType border_type>
void WarpAffineTest(int32_t height, int32_t width, float diff)
{
    int32_t input_height = height;
    int32_t input_width = width;
    int32_t output_height = height;
    int32_t output_width = width;
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    std::unique_ptr<double[]> inv_warpMat(new double[6]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<T>(dst.get(), width * height * nc, 0, 255);
    memcpy(dst_ref.get(), dst.get(), height * width * nc * sizeof(T));
    ppl::cv::debug::randomFill<double>(inv_warpMat.get(), 6, 0, 2);
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * input_width * nc);
    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * output_width * nc);
    cv::Mat inv_mat(2, 3, CV_64FC1, inv_warpMat.get());
    cv::BorderTypes cv_border_type;
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        cv_border_type = cv::BORDER_CONSTANT;
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        cv_border_type = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        cv_border_type = cv::BORDER_TRANSPARENT;
    }
    if (inter_mode == ppl::cv::INTERPOLATION_LINEAR) {
        warpaffine_linear<T, nc, border_type>(input_height, input_width, input_width * nc, output_height, output_width, output_width * nc, dst_ref.get(), src.get(), inv_warpMat.get(), border_type);
        ppl::cv::riscv::WarpAffineLinear<T, nc>(input_height, input_width, input_width * nc, src.get(), output_height, output_width, output_width * nc, dst.get(), inv_warpMat.get(), border_type);
    } else if (inter_mode == ppl::cv::INTERPOLATION_NEAREST_POINT) {
        for (int32_t i = 0; i < 16; ++i)
            cv::warpAffine(src_opencv, dst_opencv, inv_mat, dst_opencv.size(), cv::WARP_INVERSE_MAP | cv::INTER_NEAREST, cv_border_type);
        ppl::cv::riscv::WarpAffineNearestPoint<T, nc>(input_height, input_width, input_width * nc, src.get(), output_height, output_width, output_width * nc, dst.get(), inv_warpMat.get(), border_type);
    }
    checkResult<T, nc>(dst_ref.get(), dst.get(), output_height, output_width, output_width * nc, output_width * nc, diff);
}

#define R(name, dtype, nc, inter_mode, border_type, diff)                    \
    TEST(name, riscv)                                                        \
    {                                                                        \
        WarpAffineTest<dtype, nc, inter_mode, border_type>(240, 320, diff);  \
        WarpAffineTest<dtype, nc, inter_mode, border_type>(480, 640, diff);  \
        WarpAffineTest<dtype, nc, inter_mode, border_type>(720, 1280, diff); \
    }

R(WARPAFFINE_FP32_C1_NEAREST_BORDER_CONSTANT, float, 1, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_FP32_C3_NEAREST_BORDER_CONSTANT, float, 3, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_FP32_C4_NEAREST_BORDER_CONSTANT, float, 4, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_U8_C1_NEAREST_BORDER_CONSTANT, uint8_t, 1, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_U8_C3_NEAREST_BORDER_CONSTANT, uint8_t, 3, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_U8_C4_NEAREST_BORDER_CONSTANT, uint8_t, 4, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_FP32_C1_NEAREST_BORDER_REPLICATE, float, 1, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_FP32_C3_NEAREST_BORDER_REPLICATE, float, 3, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_FP32_C4_NEAREST_BORDER_REPLICATE, float, 4, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_U8_C1_NEAREST_BORDER_REPLICATE, uint8_t, 1, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_U8_C3_NEAREST_BORDER_REPLICATE, uint8_t, 3, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_U8_C4_NEAREST_BORDER_REPLICATE, uint8_t, 4, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_FP32_C1_NEAREST_BORDER_TRANSPARENT, float, 1, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_FP32_C3_NEAREST_BORDER_TRANSPARENT, float, 3, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_FP32_C4_NEAREST_BORDER_TRANSPARENT, float, 4, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_U8_C1_NEAREST_BORDER_TRANSPARENT, uint8_t, 1, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_U8_C3_NEAREST_BORDER_TRANSPARENT, uint8_t, 3, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_U8_C4_NEAREST_BORDER_TRANSPARENT, uint8_t, 4, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT, 1.01f);

R(WARPAFFINE_FP32_C1_LINEAR_BORDER_CONSTANT, float, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 1.01f); // ppl.cv is more accurate than opencv
R(WARPAFFINE_FP32_C3_LINEAR_BORDER_CONSTANT, float, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_FP32_C4_LINEAR_BORDER_CONSTANT, float, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_U8_C1_LINEAR_BORDER_CONSTANT, uint8_t, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_U8_C3_LINEAR_BORDER_CONSTANT, uint8_t, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_U8_C4_LINEAR_BORDER_CONSTANT, uint8_t, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 1.01f);
R(WARPAFFINE_FP32_C1_LINEAR_BORDER_REPLICATE, float, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_FP32_C3_LINEAR_BORDER_REPLICATE, float, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_FP32_C4_LINEAR_BORDER_REPLICATE, float, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_U8_C1_LINEAR_BORDER_REPLICATE, uint8_t, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_U8_C3_LINEAR_BORDER_REPLICATE, uint8_t, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_U8_C4_LINEAR_BORDER_REPLICATE, uint8_t, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 1.01f);
R(WARPAFFINE_FP32_C1_LINEAR_BORDER_TRANSPARENT, float, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_FP32_C3_LINEAR_BORDER_TRANSPARENT, float, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_FP32_C4_LINEAR_BORDER_TRANSPARENT, float, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_U8_C1_LINEAR_BORDER_TRANSPARENT, uint8_t, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_U8_C3_LINEAR_BORDER_TRANSPARENT, uint8_t, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
R(WARPAFFINE_U8_C4_LINEAR_BORDER_TRANSPARENT, uint8_t, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
