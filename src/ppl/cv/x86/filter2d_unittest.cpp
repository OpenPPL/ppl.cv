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

#include "ppl/cv/x86/filter2d.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int32_t nc, int32_t filter_size>
void Filter2DTest(int32_t height, int32_t width, T diff) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    std::unique_ptr<float[]> filter(new float[filter_size * filter_size]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<float>(filter.get(), filter_size * filter_size, 0, 1.0 / (filter_size * filter_size));

    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);
    cv::Mat filter_opencv(filter_size, filter_size, CV_32FC1, filter.get());

    cv::filter2D(src_opencv, dst_opencv, -1, filter_opencv,cv::Point(-1,-1),0,cv::BORDER_REFLECT101);
    ppl::cv::x86::Filter2D<T, nc>(height, width, width * nc,
                            src.get(), filter_size, filter.get(), width * nc,
                            dst.get(), ppl::cv::BORDER_REFLECT101);

    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    height, width,
                    width * nc, width * nc,
                    diff);
}


TEST(FILTER2D_FP32, x86)
{
    Filter2DTest<float, 3, 3>(720, 1080, 2.0);
    Filter2DTest<float, 3, 5>(720, 1080, 2.0);
    Filter2DTest<float, 3, 7>(720, 1080, 2.0);

}

TEST(FILTER2D_UINT8, x86)
{
    Filter2DTest<uint8_t, 3, 3>(720, 1080, 2.0);
    Filter2DTest<uint8_t, 3, 5>(720, 1080, 2.0);
    Filter2DTest<uint8_t, 3, 7>(720, 1080, 2.0);
}
