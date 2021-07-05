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

#include "ppl/cv/x86/resize.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include "ppl/common/retcode.h"


template<typename T, int32_t nc>
void ResizeLinearTest(int32_t inHeight, int32_t inWidth,
                    int32_t outHeight, int32_t outWidth, T diff) {
    std::unique_ptr<T[]> src(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst_ref(new T[outWidth * outHeight * nc]);
    std::unique_ptr<T[]> dst(new T[outWidth * outHeight * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), inWidth * inHeight * nc, 0, 255);

    cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * inWidth * nc);
    cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * outWidth * nc);

    cv::resize(src_opencv, dst_opencv, cv::Size(outWidth, outHeight), cv::INTER_LINEAR);
    auto rst = ppl::cv::x86::ResizeLinear<T, nc>(inHeight, inWidth, inWidth * nc, src.get(),
                                                 outHeight, outWidth, outWidth * nc,
                                                 dst.get());
    EXPECT_EQ(rst, ppl::common::RC_SUCCESS);

    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    outHeight, outWidth,
                    outWidth * nc, outWidth * nc,
                    diff);
}

template<typename T, int32_t nc>
void ResizeNearestTest(int32_t inHeight, int32_t inWidth,
                    int32_t outHeight, int32_t outWidth, T diff) {
    std::unique_ptr<T[]> src(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst_ref(new T[outWidth * outHeight * nc]);
    std::unique_ptr<T[]> dst(new T[outWidth * outHeight * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), inWidth * inHeight * nc, 0, 255);
    cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * inWidth * nc);
    cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * outWidth * nc);

    cv::resize(src_opencv, dst_opencv, cv::Size(outWidth, outHeight), 0, 0, cv::INTER_NEAREST);
    auto rst = ppl::cv::x86::ResizeNearestPoint<T, nc>(inHeight, inWidth, inWidth * nc, src.get(),
                                                       outHeight, outWidth, outWidth * nc,
                                                       dst.get());

    EXPECT_EQ(rst, ppl::common::RC_SUCCESS);

    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    outHeight, outWidth,
                    outWidth * nc, outWidth * nc,
                    diff);
}

TEST(RESIZE_LINEAR_FP32, x86)
{
    ResizeLinearTest<float, 1>(360, 540, 720, 1080, 1);
    ResizeLinearTest<float, 1>(720, 1080, 360, 540, 1);
    ResizeLinearTest<float, 1>(360, 540, 640, 480, 1);
    ResizeLinearTest<float, 1>(640, 480, 360, 540, 1);

    ResizeLinearTest<float, 3>(360, 540, 720, 1080, 1);
    ResizeLinearTest<float, 3>(720, 1080, 360, 540, 1);
    ResizeLinearTest<float, 3>(360, 540, 640, 480, 1);
    ResizeLinearTest<float, 3>(640, 480, 360, 540, 1);

    ResizeLinearTest<float, 4>(360, 540, 720, 1080, 1);
    ResizeLinearTest<float, 4>(720, 1080, 360, 540, 1);
    ResizeLinearTest<float, 4>(360, 540, 640, 480, 1);
    ResizeLinearTest<float, 4>(640, 480, 360, 540, 1);
}

TEST(RESIZE_LINEAR_UINT8, x86)
{
    ResizeLinearTest<uint8_t, 1>(360, 540, 720, 1080, 1);
    ResizeLinearTest<uint8_t, 1>(720, 1080, 360, 540, 1);
    ResizeLinearTest<uint8_t, 1>(360, 540, 640, 480, 1);
    ResizeLinearTest<uint8_t, 1>(640, 480, 360, 540, 1);

    ResizeLinearTest<uint8_t, 3>(360, 540, 720, 1080, 1);
    ResizeLinearTest<uint8_t, 3>(720, 1080, 360, 540, 1);
    ResizeLinearTest<uint8_t, 3>(360, 540, 640, 480, 1);
    ResizeLinearTest<uint8_t, 3>(640, 480, 360, 540, 1);

    ResizeLinearTest<uint8_t, 4>(360, 540, 720, 1080, 1);
    ResizeLinearTest<uint8_t, 4>(720, 1080, 360, 540, 1);
    ResizeLinearTest<uint8_t, 4>(360, 540, 640, 480, 1);
    ResizeLinearTest<uint8_t, 4>(640, 480, 360, 540, 1);
}

TEST(RESIZE_NEAREST_FP32, x86)
{
    ResizeNearestTest<float, 1>(360, 540, 720, 1080, 1);
    ResizeNearestTest<float, 1>(720, 1080, 360, 540, 1);
    ResizeNearestTest<float, 1>(360, 540, 640, 480, 1);
    ResizeNearestTest<float, 1>(640, 480, 360, 540, 1);

    ResizeNearestTest<float, 3>(360, 540, 720, 1080, 1);
    ResizeNearestTest<float, 3>(720, 1080, 360, 540, 1);
    ResizeNearestTest<float, 3>(360, 540, 640, 480, 1);
    ResizeNearestTest<float, 3>(640, 480, 360, 540, 1);

    ResizeNearestTest<float, 4>(360, 540, 720, 1080, 1);
    ResizeNearestTest<float, 4>(720, 1080, 360, 540, 1);
    ResizeNearestTest<float, 4>(360, 540, 640, 480, 1);
    ResizeNearestTest<float, 4>(640, 480, 360, 540, 1);
}

TEST(RESIZE_NEAREST_UINT8, x86)
{
    ResizeNearestTest<uint8_t, 1>(360, 540, 720, 1080, 1);
    ResizeNearestTest<uint8_t, 1>(720, 1080, 360, 540, 1);
    ResizeNearestTest<uint8_t, 1>(360, 540, 640, 480, 1);
    ResizeNearestTest<uint8_t, 1>(640, 480, 360, 540, 1);

    ResizeNearestTest<uint8_t, 3>(360, 540, 720, 1080, 1);
    ResizeNearestTest<uint8_t, 3>(720, 1080, 360, 540, 1);
    ResizeNearestTest<uint8_t, 3>(360, 540, 640, 480, 1);
    ResizeNearestTest<uint8_t, 3>(640, 480, 360, 540, 1);

    ResizeNearestTest<uint8_t, 4>(360, 540, 720, 1080, 1);
    ResizeNearestTest<uint8_t, 4>(720, 1080, 360, 540, 1);
    ResizeNearestTest<uint8_t, 4>(360, 540, 640, 480, 1);
    ResizeNearestTest<uint8_t, 4>(640, 480, 360, 540, 1);
}
