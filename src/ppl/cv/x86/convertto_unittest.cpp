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

#include "ppl/cv/x86/convertto.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <int32_t nc>
void FP32_To_Uint8_Test(int32_t height, int32_t width)
{
    float scale = 255.0f;
    std::unique_ptr<float[]> src(new float[width * height * nc]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * nc]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * nc]);
    ppl::cv::debug::randomFill<float>(src.get(), width * height * nc, 0, 1);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<float>::depth, nc), src.get(), sizeof(float) * width);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, nc), dst_ref.get(), sizeof(uint8_t) * width);
    ppl::cv::x86::ConvertTo<float, nc, uint8_t>(height, width, width * nc, src.get(), scale, width * nc, dst.get());
    src_opencv.convertTo(dst_opencv, CV_8U, scale);
    checkResult<uint8_t, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

template <int32_t nc>
void Uint8_To_FP32_Test(int32_t height, int32_t width)
{
    float scale = 1.0 / 255.0f;
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * nc]);
    std::unique_ptr<float[]> dst_ref(new float[width * height * nc]);
    std::unique_ptr<float[]> dst(new float[width * height * nc]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, nc), src.get(), sizeof(uint8_t) * width);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<float>::depth, nc), dst_ref.get(), sizeof(float) * width);
    ppl::cv::x86::ConvertTo<uint8_t, nc, float>(height, width, width * nc, src.get(), scale, width * nc, dst.get());
    src_opencv.convertTo(dst_opencv, CV_32F, scale);
    checkResult<float, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1e-3);
}

TEST(CONVERT_FP32_TO_UINT8, x86)
{
    FP32_To_Uint8_Test<1>(640, 720);
    FP32_To_Uint8_Test<3>(640, 720);
    FP32_To_Uint8_Test<4>(640, 720);
}

TEST(CONVERT_UINT8_TO_FP32, x86)
{
    Uint8_To_FP32_Test<1>(640, 720);
    Uint8_To_FP32_Test<3>(640, 720);
    Uint8_To_FP32_Test<4>(640, 720);
}
