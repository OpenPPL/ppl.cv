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

#include "ppl/cv/x86/boxfilter.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"


template<typename T, int32_t nc, int32_t filter_size, bool normalized, ppl::cv::BorderType border_type>
void BoxFilterTest(int32_t height, int32_t width, T diff) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);

    if (border_type == ppl::cv::BORDER_REFLECT101) {
        ppl::cv::x86::BoxFilter<T, nc>(height, width,
                                width * nc, src.get(),
                                filter_size, filter_size, normalized,
                                width * nc, dst.get(), ppl::cv::BORDER_DEFAULT);
        cv::boxFilter(src_opencv, dst_opencv, -1,
                cv::Size(filter_size, filter_size), cv::Point(-1,-1), normalized, cv::BORDER_REFLECT101);
    } else if (border_type == ppl::cv::BORDER_REFLECT) {
        ppl::cv::x86::BoxFilter<T, nc>(height, width,
                                width * nc, src.get(),
                                filter_size, filter_size, normalized,
                                width * nc, dst.get(), ppl::cv::BORDER_REFLECT);
        cv::boxFilter(src_opencv, dst_opencv, -1,
                cv::Size(filter_size, filter_size), cv::Point(-1,-1), normalized, cv::BORDER_REFLECT);
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        ppl::cv::x86::BoxFilter<T, nc>(height, width,
                                width * nc, src.get(),
                                filter_size, filter_size, normalized,
                                width * nc, dst.get(), ppl::cv::BORDER_REPLICATE);
        cv::boxFilter(src_opencv, dst_opencv, -1,
                cv::Size(filter_size, filter_size), cv::Point(-1,-1), normalized, cv::BORDER_REPLICATE);
    } else if (border_type == ppl::cv::BORDER_CONSTANT) {
        ppl::cv::x86::BoxFilter<T, nc>(height, width,
                                width * nc, src.get(),
                                filter_size, filter_size, normalized,
                                width * nc, dst.get(), ppl::cv::BORDER_CONSTANT);
        cv::boxFilter(src_opencv, dst_opencv, -1,
                cv::Size(filter_size, filter_size), cv::Point(-1,-1), normalized, cv::BORDER_CONSTANT);
    }
    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    height, width,
                    width * nc, width * nc,
                    diff);
}


TEST(BoxFilter_REFELCT_FP32, x86)
{
    BoxFilterTest<float, 3, 3, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 3, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 3, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 3, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);

    BoxFilterTest<float, 3, 5, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 5, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 5, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 5, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
}

TEST(BoxFilter_REFELCT101_FP32, x86)
{
    BoxFilterTest<float, 3, 3, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 3, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 3, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 3, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);

    BoxFilterTest<float, 3, 5, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 5, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 5, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 5, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
}

TEST(BoxFilter_CONSTANT_FP32, x86)
{
    BoxFilterTest<float, 3, 3, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 3, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 3, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 3, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);

    BoxFilterTest<float, 3, 5, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 5, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 5, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 5, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
}

TEST(BoxFilter_REPLICATE_FP32, x86)
{
    BoxFilterTest<float, 3, 3, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 3, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 3, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 3, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);

    BoxFilterTest<float, 3, 5, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<float, 3, 5, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);

    BoxFilterTest<float, 1, 5, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<float, 1, 5, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
}

TEST(BoxFilter_REFELCT_UINT8, x86)
{
    BoxFilterTest<uint8_t, 3, 3, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 3, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 3, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 3, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 3, 5, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 5, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 5, false, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 5, true, ppl::cv::BORDER_REFLECT>(720, 1080, 1.0f);
}

TEST(BoxFilter_REFELCT101_UINT8, x86)
{
    BoxFilterTest<uint8_t, 3, 3, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 3, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 3, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 3, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 3, 5, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 5, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 5, false, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 5, true, ppl::cv::BORDER_REFLECT101>(720, 1080, 1.0f);
}

TEST(BoxFilter_CONSTANT_UINT8, x86)
{
    BoxFilterTest<uint8_t, 3, 3, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 3, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 3, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 3, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 3, 5, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 5, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 5, false, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 5, true, ppl::cv::BORDER_CONSTANT>(720, 1080, 1.0f);
}

TEST(BoxFilter_REPLICATE_UINT8, x86)
{
    BoxFilterTest<uint8_t, 3, 3, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 3, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 3, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 3, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 3, 5, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 3, 5, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);

    BoxFilterTest<uint8_t, 1, 5, false, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
    BoxFilterTest<uint8_t, 1, 5, true, ppl::cv::BORDER_REPLICATE>(720, 1080, 1.0f);
}
