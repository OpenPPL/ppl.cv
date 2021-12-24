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

#include "ppl/cv/x86/norm.h"
#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename T, int nc, bool use_mask, ppl::cv::NormTypes norm_type>
void NormTest(int height, int width)
{
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get());
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);

    std::unique_ptr<uchar[]> mask(new uchar[width * height]);
    for (int i = 0; i < height * width; ++i) {
        mask.get()[i] = (std::rand()) % 2;
    }
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<uchar>::depth, 1), mask.get());

    double result;
    double result_ref;

    if (use_mask) {
        if (norm_type == ppl::cv::NORM_L1) {
            result     = ppl::cv::x86::Norm<T, nc>(height, width, width * nc, src.get(), ppl::cv::NORM_L1, width, mask.get());
            result_ref = cv::norm(srcMat, cv::NORM_L1, maskMat);
        } else if (norm_type == ppl::cv::NORM_L2) {
            result     = ppl::cv::x86::Norm<T, nc>(height, width, width * nc, src.get(), ppl::cv::NORM_L2, width, mask.get());
            result_ref = cv::norm(srcMat, cv::NORM_L2, maskMat);
        } else if (norm_type == ppl::cv::NORM_INF) {
            result     = ppl::cv::x86::Norm<T, nc>(height, width, width * nc, src.get(), ppl::cv::NORM_INF, width, mask.get());
            result_ref = cv::norm(srcMat, cv::NORM_INF, maskMat);
        }
    } else {
        if (norm_type == ppl::cv::NORM_L1) {
            result     = ppl::cv::x86::Norm<T, nc>(height, width, width * nc, src.get(), ppl::cv::NORM_L1, 0, nullptr);
            result_ref = cv::norm(srcMat, cv::NORM_L1);
        } else if (norm_type == ppl::cv::NORM_L2) {
            result     = ppl::cv::x86::Norm<T, nc>(height, width, width * nc, src.get(), ppl::cv::NORM_L2, 0, nullptr);
            result_ref = cv::norm(srcMat, cv::NORM_L2);
        } else if (norm_type == ppl::cv::NORM_INF) {
            result     = ppl::cv::x86::Norm<T, nc>(height, width, width * nc, src.get(), ppl::cv::NORM_INF, 0, nullptr);
            result_ref = cv::norm(srcMat, cv::NORM_INF);
        }
    }

    EXPECT_LT(std::abs(result - result_ref), 1e-2);
}

TEST(NORM_L1, x86)
{
    NormTest<float, 1, false, ppl::cv::NORM_L1>(64, 72);
    NormTest<float, 3, false, ppl::cv::NORM_L1>(64, 72);
    NormTest<float, 4, false, ppl::cv::NORM_L1>(64, 72);

    NormTest<uchar, 1, false, ppl::cv::NORM_L1>(64, 72);
    NormTest<uchar, 3, false, ppl::cv::NORM_L1>(64, 72);
    NormTest<uchar, 4, false, ppl::cv::NORM_L1>(64, 72);

    NormTest<float, 1, true, ppl::cv::NORM_L1>(64, 72);
    NormTest<float, 3, true, ppl::cv::NORM_L1>(64, 72);
    NormTest<float, 4, true, ppl::cv::NORM_L1>(64, 72);

    NormTest<uchar, 1, true, ppl::cv::NORM_L1>(64, 72);
    NormTest<uchar, 3, true, ppl::cv::NORM_L1>(64, 72);
    NormTest<uchar, 4, true, ppl::cv::NORM_L1>(64, 72);
}

TEST(NORM_L2, x86)
{
    NormTest<float, 1, false, ppl::cv::NORM_L2>(64, 72);
    NormTest<float, 3, false, ppl::cv::NORM_L2>(64, 72);
    NormTest<float, 4, false, ppl::cv::NORM_L2>(64, 72);

    NormTest<uchar, 1, false, ppl::cv::NORM_L2>(64, 72);
    NormTest<uchar, 3, false, ppl::cv::NORM_L2>(64, 72);
    NormTest<uchar, 4, false, ppl::cv::NORM_L2>(64, 72);

    NormTest<float, 1, true, ppl::cv::NORM_L2>(64, 72);
    NormTest<float, 3, true, ppl::cv::NORM_L2>(64, 72);
    NormTest<float, 4, true, ppl::cv::NORM_L2>(64, 72);

    NormTest<uchar, 1, true, ppl::cv::NORM_L2>(64, 72);
    NormTest<uchar, 3, true, ppl::cv::NORM_L2>(64, 72);
    NormTest<uchar, 4, true, ppl::cv::NORM_L2>(64, 72);
}

TEST(NORM_INF, x86)
{
    NormTest<float, 1, false, ppl::cv::NORM_INF>(64, 72);
    NormTest<float, 3, false, ppl::cv::NORM_INF>(64, 72);
    NormTest<float, 4, false, ppl::cv::NORM_INF>(64, 72);

    NormTest<uchar, 1, false, ppl::cv::NORM_INF>(64, 72);
    NormTest<uchar, 3, false, ppl::cv::NORM_INF>(64, 72);
    NormTest<uchar, 4, false, ppl::cv::NORM_INF>(64, 72);

    NormTest<float, 1, true, ppl::cv::NORM_INF>(64, 72);
    NormTest<float, 3, true, ppl::cv::NORM_INF>(64, 72);
    NormTest<float, 4, true, ppl::cv::NORM_INF>(64, 72);

    NormTest<uchar, 1, true, ppl::cv::NORM_INF>(64, 72);
    NormTest<uchar, 3, true, ppl::cv::NORM_INF>(64, 72);
    NormTest<uchar, 4, true, ppl::cv::NORM_INF>(64, 72);
}
