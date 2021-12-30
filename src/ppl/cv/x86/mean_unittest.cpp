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

#include "ppl/cv/x86/mean.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/types.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int32_t nc, bool use_mask>
void MeanTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<float[]> dst(new float[nc]);
    std::unique_ptr<float[]> dst_ref(new float[nc]);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get());
    cv::Mat dstMat(1, 1, CV_MAKETYPE(cv::DataType<float>::depth, nc), dst_ref.get());
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    std::unique_ptr<uint8_t[]> mask(new uint8_t[width * height]);
    for (int32_t i = 0; i < height * width; ++i) {
        mask.get()[i] = (std::rand()) % 2;
    }
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), mask.get());
    if (use_mask) {
        ppl::cv::x86::Mean<T, nc>(height, width, width * nc, src.get(),
                                dst.get(), width, mask.get());
        dstMat = cv::mean(srcMat, maskMat);
    } else {
        ppl::cv::x86::Mean<T, nc>(height, width, width * nc, src.get(), dst.get(), width, nullptr);
        dstMat = cv::mean(srcMat);
    }
    for (int32_t c = 0; c < nc; ++c) {
        EXPECT_LT(std::abs(dst.get()[c] - dst_ref.get()[c]), 1.0);
    }
}



TEST(MEAN_FP32, x86)
{
   MeanTest<float, 1, true>(240, 320);
   MeanTest<float, 1, true>(480, 640);
   MeanTest<float, 1, true>(640, 720);
   MeanTest<float, 1, true>(720, 1080);

   MeanTest<float, 3, true>(240, 320);
   MeanTest<float, 3, true>(480, 640);
   MeanTest<float, 3, true>(640, 720);
   MeanTest<float, 3, true>(720, 1080);

   MeanTest<float, 4, true>(240, 320);
   MeanTest<float, 4, true>(480, 640);
   MeanTest<float, 4, true>(640, 720);
   MeanTest<float, 4, true>(720, 1080);

   MeanTest<float, 1, false>(240, 320);
   MeanTest<float, 1, false>(480, 640);
   MeanTest<float, 1, false>(640, 720);
   MeanTest<float, 1, false>(720, 1080);

   MeanTest<float, 3, false>(240, 320);
   MeanTest<float, 3, false>(480, 640);
   MeanTest<float, 3, false>(640, 720);
   MeanTest<float, 3, false>(720, 1080);

   MeanTest<float, 4, false>(240, 320);
   MeanTest<float, 4, false>(480, 640);
   MeanTest<float, 4, false>(640, 720);
   MeanTest<float, 4, false>(720, 1080);
}

TEST(MEAN_UCHAR, x86)
{
   MeanTest<uint8_t, 1, true>(240, 320);
   MeanTest<uint8_t, 1, true>(480, 640);
   MeanTest<uint8_t, 1, true>(640, 720);
   MeanTest<uint8_t, 1, true>(720, 1080);

   MeanTest<uint8_t, 3, true>(240, 320);
   MeanTest<uint8_t, 3, true>(480, 640);
   MeanTest<uint8_t, 3, true>(640, 720);
   MeanTest<uint8_t, 3, true>(720, 1080);

   MeanTest<uint8_t, 4, true>(240, 320);
   MeanTest<uint8_t, 4, true>(480, 640);
   MeanTest<uint8_t, 4, true>(640, 720);
   MeanTest<uint8_t, 4, true>(720, 1080);

   MeanTest<uint8_t, 1, false>(240, 320);
   MeanTest<uint8_t, 1, false>(480, 640);
   MeanTest<uint8_t, 1, false>(640, 720);
   MeanTest<uint8_t, 1, false>(720, 1080);

   MeanTest<uint8_t, 3, false>(240, 320);
   MeanTest<uint8_t, 3, false>(480, 640);
   MeanTest<uint8_t, 3, false>(640, 720);
   MeanTest<uint8_t, 3, false>(720, 1080);

   MeanTest<uint8_t, 4, false>(240, 320);
   MeanTest<uint8_t, 4, false>(480, 640);
   MeanTest<uint8_t, 4, false>(640, 720);
   MeanTest<uint8_t, 4, false>(720, 1080);
}
