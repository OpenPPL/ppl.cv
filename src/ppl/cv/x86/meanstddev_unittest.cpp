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

#include "ppl/cv/x86/meanstddev.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/types.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

#define CHECK_RESULT(a, b, diff_THR) \
        EXPECT_LT(abs(a-b), diff_THR);


template<typename T, int32_t nc, bool use_mask>
void MeanStdDevTest(int32_t height, int32_t width, float diff) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<uint8_t[]> mask(new uint8_t[width * height]);
    std::unique_ptr<float[]> mean(new float[nc]);
    std::unique_ptr<float[]> variance(new float[nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<uint8_t>(mask.get(), width * height, 0, 255);
    cv::Scalar meanScalar;
    cv::Scalar varianceScalar;
    
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get());
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), mask.get());
    if (use_mask) {
        ppl::cv::x86::MeanStdDev<T, nc>(height, width, width * nc, src.get(),
                                        mean.get(), variance.get(),
                                        width, mask.get());
        cv::meanStdDev(srcMat, meanScalar, varianceScalar, maskMat);
    } else {
        ppl::cv::x86::MeanStdDev<T, nc>(height, width, width * nc, src.get(),
                                        mean.get(), variance.get());
        cv::meanStdDev(srcMat, meanScalar, varianceScalar);
    }
    for (int32_t i = 0; i < nc; ++i) {
        CHECK_RESULT(mean.get()[i], meanScalar[i], diff);
        CHECK_RESULT(variance.get()[i], varianceScalar[i], diff);
    }
}

TEST(MeanStdDevTest_UCHAR, x86)
{
    MeanStdDevTest<uint8_t, 1, true>(480, 640, 1.01f);
    MeanStdDevTest<uint8_t, 3, true>(480, 640, 1.01f);
    MeanStdDevTest<uint8_t, 4, true>(480, 640, 1.01f);

    MeanStdDevTest<uint8_t, 1, false>(480, 640, 1.01f);
    MeanStdDevTest<uint8_t, 3, false>(480, 640, 1.01f);
    MeanStdDevTest<uint8_t, 4, false>(480, 640, 1.01f);
}

TEST(MeanStdDevTest_FP32, x86)
{
    MeanStdDevTest<float, 1, true>(480, 640, 0.1);
    MeanStdDevTest<float, 3, true>(480, 640, 0.1);
    MeanStdDevTest<float, 4, true>(480, 640, 0.1);

    MeanStdDevTest<float, 1, false>(480, 640, 0.1);
    MeanStdDevTest<float, 3, false>(480, 640, 0.1);
    MeanStdDevTest<float, 4, false>(480, 640, 0.1);
}

