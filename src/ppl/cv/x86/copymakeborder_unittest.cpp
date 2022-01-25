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

#include <opencv2/imgproc.hpp>
#include "ppl/cv/x86/copymakeborder.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/types.h"
#include "ppl/cv/x86/test.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>

template<typename T, int32_t nc, ppl::cv::BorderType border_type>
void CopymakeborderTest(int32_t height, int32_t width, int32_t padding, float diff) {
    int32_t input_height = height;
    int32_t input_width = width;
    int32_t output_height = height + 2 * padding;
    int32_t output_width = width + 2 * padding;
    std::unique_ptr<T[]> src(new T[input_height * input_width * nc]);
    std::unique_ptr<T[]> dst_ref(new T[output_height * output_width * nc]);
    std::unique_ptr<T[]> dst(new T[output_height * output_width * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), input_height * input_width * nc, 0, 255);
    ppl::cv::debug::randomFill<T>(dst.get(), output_height * output_width * nc, 0, 255);
    memcpy(dst_ref.get(), dst.get(), output_height * output_width * nc * sizeof(T));
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * input_width * nc);
    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * output_width * nc);
    cv::BorderTypes cv_border_type;
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        cv_border_type = cv::BORDER_CONSTANT;
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        cv_border_type = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_REFLECT) {
        cv_border_type = cv::BORDER_REFLECT;
    } else if (border_type == ppl::cv::BORDER_REFLECT101) {
        cv_border_type = cv::BORDER_REFLECT101;
    }
    ppl::cv::x86::CopyMakeBorder<T, nc>(input_height, input_width, input_width * nc, src.get(), output_height, 
                                          output_width, output_width * nc, dst.get(), border_type);
    cv::copyMakeBorder(src_opencv, dst_opencv, padding, padding, padding, padding, cv_border_type);                                       
    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    output_height, output_width,
                    output_width * nc, output_width * nc,
                    diff);
}

#define R(name, dtype, nc, border_type, diff) \
    TEST(name, x86) \
    { \
        CopymakeborderTest<dtype, nc, border_type>(240, 320, 1, diff); \
        CopymakeborderTest<dtype, nc, border_type>(241, 321, 2, diff); \
        CopymakeborderTest<dtype, nc, border_type>(480, 640, 3, diff); \
        CopymakeborderTest<dtype, nc, border_type>(720, 1280, 4, diff); \
    } \

R(copymakeborder_u8c1_constant_x86, uint8_t, 1, ppl::cv::BORDER_CONSTANT, 1.01f);
R(copymakeborder_u8c3_constant_x86, uint8_t, 3, ppl::cv::BORDER_CONSTANT, 1.01f);
R(copymakeborder_u8c4_constant_x86, uint8_t, 4, ppl::cv::BORDER_CONSTANT, 1.01f);
R(copymakeborder_u8c1_replicate_x86, uint8_t, 1, ppl::cv::BORDER_REPLICATE, 1.01f);
R(copymakeborder_u8c3_replicate_x86, uint8_t, 3, ppl::cv::BORDER_REPLICATE, 1.01f);
R(copymakeborder_u8c4_replicate_x86, uint8_t, 4, ppl::cv::BORDER_REPLICATE, 1.01f);
R(copymakeborder_u8c1_reflect_x86, uint8_t, 1, ppl::cv::BORDER_REFLECT, 1.01f);
R(copymakeborder_u8c3_reflect_x86, uint8_t, 3, ppl::cv::BORDER_REFLECT, 1.01f);
R(copymakeborder_u8c4_reflect_x86, uint8_t, 4, ppl::cv::BORDER_REFLECT, 1.01f);
R(copymakeborder_u8c1_reflect101_x86, uint8_t, 1, ppl::cv::BORDER_REFLECT_101, 1.01f);
R(copymakeborder_u8c3_reflect101_x86, uint8_t, 3, ppl::cv::BORDER_REFLECT_101, 1.01f);
R(copymakeborder_u8c4_reflect101_x86, uint8_t, 4, ppl::cv::BORDER_REFLECT_101, 1.01f);

R(copymakeborder_fp32c1_constant_x86, float, 1, ppl::cv::BORDER_CONSTANT, 1.01f);
R(copymakeborder_fp32c3_constant_x86, float, 3, ppl::cv::BORDER_CONSTANT, 1.01f);
R(copymakeborder_fp32c4_constant_x86, float, 4, ppl::cv::BORDER_CONSTANT, 1.01f);
R(copymakeborder_fp32c1_replicate_x86, float, 1, ppl::cv::BORDER_REPLICATE, 1.01f);
R(copymakeborder_fp32c3_replicate_x86, float, 3, ppl::cv::BORDER_REPLICATE, 1.01f);
R(copymakeborder_fp32c4_replicate_x86, float, 4, ppl::cv::BORDER_REPLICATE, 1.01f);
R(copymakeborder_fp32c1_reflect_x86, float, 1, ppl::cv::BORDER_REFLECT, 1.01f);
R(copymakeborder_fp32c3_reflect_x86, float, 3, ppl::cv::BORDER_REFLECT, 1.01f);
R(copymakeborder_fp32c4_reflect_x86, float, 4, ppl::cv::BORDER_REFLECT, 1.01f);
R(copymakeborder_fp32c1_reflect101_x86, float, 1, ppl::cv::BORDER_REFLECT_101, 1.01f);
R(copymakeborder_fp32c3_reflect101_x86, float, 3, ppl::cv::BORDER_REFLECT_101, 1.01f);
R(copymakeborder_fp32c4_reflect101_x86, float, 4, ppl::cv::BORDER_REFLECT_101, 1.01f);

