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

#include "ppl/cv/x86/medianblur.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/types.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"


template<typename T, int32_t filter_size, int32_t nc>
void MedianBlurTest(int32_t height, int32_t width, float diff) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);

    cv::medianBlur(src_opencv, dst_opencv, filter_size);
    ppl::cv::x86::MedianBlur<T, nc>(height, width, width * nc, src.get(), width * nc,
                            dst.get(), filter_size, ppl::cv::BORDER_REPLICATE);

    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    height, width,
                    width * nc, width * nc,
                    diff);
}


TEST(MEDIAN_BLUR_FP32, x86)
{
    MedianBlurTest<float, 3, 1>(720, 1080, 1e-3);
    MedianBlurTest<float, 5, 1>(720, 1080, 1e-3);
    MedianBlurTest<float, 3, 3>(720, 1080, 1e-3);
    MedianBlurTest<float, 5, 3>(720, 1080, 1e-3);
    MedianBlurTest<float, 3, 4>(720, 1080, 1e-3);
    MedianBlurTest<float, 5, 4>(720, 1080, 1e-3);
}

TEST(MEDIAN_BLUR_UINT8, x86)
{
    MedianBlurTest<uint8_t, 3, 1>(720, 1080, 1e-3);
    MedianBlurTest<uint8_t, 5, 1>(720, 1080, 1e-3);
    MedianBlurTest<uint8_t, 3, 3>(720, 1080, 1e-3);
    MedianBlurTest<uint8_t, 5, 3>(720, 1080, 1e-3);
    MedianBlurTest<uint8_t, 3, 4>(720, 1080, 1e-3);
    MedianBlurTest<uint8_t, 5, 4>(720, 1080, 1e-3);
}
