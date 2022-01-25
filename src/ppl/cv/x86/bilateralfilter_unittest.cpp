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

#include "ppl/cv/x86/bilateralfilter.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int32_t nc, int32_t diameter, int32_t color, int32_t space>
void BilateralFilterTest(int32_t height, int32_t width, T diff) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);

    cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);
    ppl::cv::x86::BilateralFilter<T, nc>(height, width, width * nc,
                            src.get(),
                            diameter,
                            color,
                            space, 
                            width * nc,
                            dst.get(),
                            ppl::cv::BORDER_DEFAULT);
    
    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    height, width,
                    width * nc, width * nc,
                    diff);
}


TEST(BilateralFilter_FP32, x86)
{
    BilateralFilterTest<float, 1, 9, 75, 75>(720, 1080, 2.0f);
    BilateralFilterTest<float, 3, 9, 75, 75>(720, 1080, 2.0f);
}

TEST(BilateralFilter_U8, x86)
{
    BilateralFilterTest<uint8_t, 1, 9, 75, 75>(720, 1080, 2.0f);
    BilateralFilterTest<uint8_t, 3, 9, 75, 75>(720, 1080, 2.0f);
}
