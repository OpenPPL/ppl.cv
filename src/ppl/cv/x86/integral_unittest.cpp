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

#include "ppl/cv/x86/integral.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename TSrc, typename TDst, int32_t nc>
void IntegralTest(int32_t height, int32_t width) {
    int32_t outHeight = height + 1;
    int32_t outWidth = width + 1;
    std::unique_ptr<TSrc[]> src(new TSrc[width * height * nc]);
    std::unique_ptr<TDst[]> dst_ref(new TDst[outHeight * outWidth * nc]);
    std::unique_ptr<TDst[]> dst(new TDst[outHeight * outWidth * nc]);
    ppl::cv::debug::randomFill<TSrc>(src.get(), width * height * nc, 0, 255);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<TSrc>::depth, nc), src.get());
    cv::Mat dstMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<TDst>::depth, nc), dst_ref.get());
    ppl::cv::x86::Integral<TSrc, TDst, nc>(height, width, width * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get());
    cv::integral(srcMat, dstMat, CV_MAKETYPE(cv::DataType<TDst>::depth, nc));

    checkResult<TDst, nc>(dst.get(), dst_ref.get(),
                    outHeight, outWidth,
                    outWidth * nc, outWidth * nc,
                    1.0f);
}


TEST(Integral_FP32, x86)
{
    IntegralTest<float, float, 1>(64, 72);
    IntegralTest<float, float, 1>(72, 108);
    IntegralTest<float, float, 3>(64, 72);
    IntegralTest<float, float, 3>(72, 108);
    IntegralTest<float, float, 4>(64, 72);
    IntegralTest<float, float, 4>(72, 108);
}

TEST(Integral_UINT8, x86)
{
    IntegralTest<uint8_t, int32_t, 1>(64, 72);
    IntegralTest<uint8_t, int32_t, 1>(72, 108);
    IntegralTest<uint8_t, int32_t, 3>(64, 72);
    IntegralTest<uint8_t, int32_t, 3>(72, 108);
    IntegralTest<uint8_t, int32_t, 4>(64, 72);
    IntegralTest<uint8_t, int32_t, 4>(72, 108);
}
