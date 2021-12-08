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
#include "ppl/cv/x86/abs.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/x86/test.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>

template<typename T, int nc>
void AbsTest(int height, int width) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, -128, 127);
    ppl::cv::x86::Abs<T, nc>(height, width, width * nc, src.get(), width * nc, dst.get());
    cv::Mat iMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get());
    cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get());
    oMat = cv::abs(iMat);
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

TEST(ABS_FP32, x86)
{
    AbsTest<float, 1>(640, 720);
    AbsTest<float, 1>(720, 1080);
    AbsTest<float, 3>(640, 720);
    AbsTest<float, 3>(720, 1080);
    AbsTest<float, 4>(640, 720);
    AbsTest<float, 4>(720, 1080);
}

TEST(ABS_INT8, x86)
{
    AbsTest<int8_t, 1>(640, 720);
    AbsTest<int8_t, 1>(720, 1080);
    AbsTest<int8_t, 3>(640, 720);
    AbsTest<int8_t, 3>(720, 1080);
    AbsTest<int8_t, 4>(640, 720);
    AbsTest<int8_t, 4>(720, 1080);
}
