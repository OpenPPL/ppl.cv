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

#include "ppl/cv/x86/erode.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/types.h"
#include "ppl/cv/debug.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include <memory>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>


template<typename T, int32_t channels>
void ErodeTest(int32_t height, int32_t width, int32_t dilation_size, T border_value, ppl::cv::BorderType ppl_border_type, cv::BorderTypes cv_border_type) {
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                         cv::Size(dilation_size, dilation_size));
    ppl::cv::x86::Erode<T, channels>(height, width, width * channels, src.get(),
                                        dilation_size, dilation_size,
                                        element.ptr<uint8_t>(), width * channels,
                                        dst.get(), ppl_border_type, border_value);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst_ref.get());
    cv::erode(srcMat, dstMat, element, cv::Point(-1,-1), 1, cv_border_type, cv::Scalar(border_value, border_value, border_value, border_value));
    checkResult<T, channels>(dst.get(), dst_ref.get(), height, width, width * channels, width * channels, 1.01f);
}

TEST(Erode_FP32, x86)
{
    int32_t kernel_size[] = {3, 5, 7};
    float border_value[] = {0.0f, 1.0f, 2.0f, 127.0f, 254.0f};
    ppl::cv::BorderType ppl_bt[] = {
        ppl::cv::BORDER_REFLECT,
        ppl::cv::BORDER_REFLECT101,
        ppl::cv::BORDER_REPLICATE,
        ppl::cv::BORDER_CONSTANT,
        };
    cv::BorderTypes cv_bt[] = {
        cv::BORDER_REFLECT,
        cv::BORDER_REFLECT101,
        cv::BORDER_REPLICATE,
        cv::BORDER_CONSTANT,
        };
    for (uint32_t k = 0; k < sizeof(ppl_bt) / sizeof(ppl::cv::BorderType); k++) {
        for (uint32_t i = 0; i < sizeof(kernel_size) / sizeof(int32_t); ++i) {
            for (uint32_t j = 0; j < sizeof(border_value) / sizeof(float); ++j) {
                ErodeTest<float, 1>(640, 480, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<float, 3>(640, 480, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<float, 4>(640, 480, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<float, 1>(320, 240, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<float, 3>(320, 240, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<float, 4>(320, 240, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
            }
        }
    }
}

TEST(Erode_U8, x86)
{
    int32_t kernel_size[] = {3, 5, 7};
    uint8_t border_value[] = {0, 1, 4, 16, 127, 254};
    ppl::cv::BorderType ppl_bt[] = {
        ppl::cv::BORDER_REFLECT,
        ppl::cv::BORDER_REFLECT101,
        ppl::cv::BORDER_REPLICATE,
        ppl::cv::BORDER_CONSTANT,
        };
    cv::BorderTypes cv_bt[] = {
        cv::BORDER_REFLECT,
        cv::BORDER_REFLECT101,
        cv::BORDER_REPLICATE,
        cv::BORDER_CONSTANT,
        };
    for (uint32_t k = 0; k < sizeof(ppl_bt) / sizeof(ppl::cv::BorderType); k++) {
        for (uint32_t i = 0; i < sizeof(kernel_size) / sizeof(int32_t); ++i) {
            for (uint32_t j = 0; j < sizeof(border_value) / sizeof(uint8_t); ++j) {
                ErodeTest<uint8_t, 1>(640, 480, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<uint8_t, 3>(640, 480, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<uint8_t, 4>(640, 480, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<uint8_t, 1>(320, 240, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<uint8_t, 3>(320, 240, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
                ErodeTest<uint8_t, 4>(320, 240, kernel_size[i], border_value[j], ppl_bt[k], cv_bt[k]);
            }
        }
    }
}
