// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for mulitional information
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

#include "ppl/cv/arm/arithmetic.h"
#include "ppl/cv/arm/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename T, int32_t nc>
void MlsTest(int32_t height, int32_t width)
{
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<T>(dst.get(), width * height * nc, 0, 255);
    memcpy(dst_ref.get(), dst.get(), height * width * nc * sizeof(T));
    
    ppl::cv::arm::Mls<T, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());

    cv::Mat iMat0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src0.get(), sizeof(T) * width * nc);
    cv::Mat iMat1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src1.get(), sizeof(T) * width * nc);
    cv::Mat temp0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc));
    cv::Mat temp1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc));
    cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);
    oMat.copyTo(temp1);

    cv::multiply(iMat0, iMat1, temp0);
    cv::subtract(temp1, temp0, oMat);
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

TEST(MLS_FP32, arm)
{
    MlsTest<float, 1>(640, 720);
    MlsTest<float, 1>(720, 1080);
    MlsTest<float, 3>(640, 720);
    MlsTest<float, 3>(720, 1080);
    MlsTest<float, 4>(640, 720);
    MlsTest<float, 4>(720, 1080);
    // Non-divsiable Cases
    MlsTest<float, 1>(640, 728);
    MlsTest<float, 1>(720, 724);
    MlsTest<float, 1>(720, 723);
}
