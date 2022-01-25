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

#include "ppl/cv/x86/warpaffine.h"
#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int32_t nc, ppl::cv::InterpolationType inter_mode, ppl::cv::BorderType border_type>
void WarpAffineTest(int32_t height, int32_t width, float diff) {
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
        for (int32_t i = 0; i < 16; ++i)    
        cv::warpAffine(src_opencv, dst_opencv, inv_mat, dst_opencv.size(), cv::WARP_INVERSE_MAP|cv::INTER_LINEAR, cv_border_type);
        ppl::cv::x86::WarpAffineLinear<T, nc>(input_height, input_width, input_width * nc,
                                src.get(), output_height, output_width, output_width * nc,
                                dst.get(), inv_warpMat.get(), border_type);
    } else if (inter_mode == ppl::cv::INTERPOLATION_NEAREST_POINT) {
        for (int32_t i = 0; i < 16; ++i) 
        cv::warpAffine(src_opencv, dst_opencv, inv_mat, dst_opencv.size(), cv::WARP_INVERSE_MAP|cv::INTER_NEAREST, cv_border_type);
        ppl::cv::x86::WarpAffineNearestPoint<T, nc>(input_height, input_width, input_width * nc,
                                src.get(), output_height, output_width, output_width * nc,
                                dst.get(), inv_warpMat.get(), border_type);
    }
    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    output_height, output_width,
                    output_width * nc, output_width * nc,
                    diff);
}

#define R(name, dtype, nc, inter_mode, border_type, diff)\
    TEST(name, x86)\
    {\
        WarpAffineTest<dtype, nc, inter_mode, border_type>(240, 320, diff); \
        WarpAffineTest<dtype, nc, inter_mode, border_type>(480, 640, diff); \
        WarpAffineTest<dtype, nc, inter_mode, border_type>(720, 1280, diff); \
    }\

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

R(WARPAFFINE_FP32_C1_LINEAR_BORDER_CONSTANT, float, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 7.01f);  // ppl.cv is more accurate than opencv 
R(WARPAFFINE_FP32_C3_LINEAR_BORDER_CONSTANT, float, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 7.01f);
R(WARPAFFINE_FP32_C4_LINEAR_BORDER_CONSTANT, float, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 7.01f);
R(WARPAFFINE_U8_C1_LINEAR_BORDER_CONSTANT, uint8_t, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 7.01f);
R(WARPAFFINE_U8_C3_LINEAR_BORDER_CONSTANT, uint8_t, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 7.01f);
R(WARPAFFINE_U8_C4_LINEAR_BORDER_CONSTANT, uint8_t, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT, 7.01f);
R(WARPAFFINE_FP32_C1_LINEAR_BORDER_REPLICATE, float, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 7.01f);
R(WARPAFFINE_FP32_C3_LINEAR_BORDER_REPLICATE, float, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 7.01f);
R(WARPAFFINE_FP32_C4_LINEAR_BORDER_REPLICATE, float, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 7.01f);
R(WARPAFFINE_U8_C1_LINEAR_BORDER_REPLICATE, uint8_t, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 7.01f);
R(WARPAFFINE_U8_C3_LINEAR_BORDER_REPLICATE, uint8_t, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 7.01f);
R(WARPAFFINE_U8_C4_LINEAR_BORDER_REPLICATE, uint8_t, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE, 7.01f);
// R(WARPAFFINE_FP32_C1_LINEAR_BORDER_TRANSPARENT, float, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
// R(WARPAFFINE_FP32_C3_LINEAR_BORDER_TRANSPARENT, float, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
// R(WARPAFFINE_FP32_C4_LINEAR_BORDER_TRANSPARENT, float, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
// R(WARPAFFINE_U8_C1_LINEAR_BORDER_TRANSPARENT, uint8_t, 1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
// R(WARPAFFINE_U8_C3_LINEAR_BORDER_TRANSPARENT, uint8_t, 3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
// R(WARPAFFINE_U8_C4_LINEAR_BORDER_TRANSPARENT, uint8_t, 4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT, 1.01f);
