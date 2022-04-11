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

#include "ppl/cv/riscv/addweighted.h"
#include "ppl/cv/riscv/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename T, int32_t nc>
void AddWeightedTest(int32_t height, int32_t width)
{
    float alpha = 1.0f;
    float beta = 2.0f;
    float gamma = 3.0f;
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 0, 255);
    ppl::cv::riscv::AddWeighted<T, nc>(height, width, width * nc, src0.get(), alpha, width * nc, src1.get(), beta, gamma, width * nc, dst.get());
    cv::Mat iMat0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src0.get(), sizeof(T) * width * nc);
    cv::Mat iMat1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src1.get(), sizeof(T) * width * nc);
    cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);
    cv::addWeighted(iMat0, alpha, iMat1, beta, gamma, oMat);
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

TEST(AddWeighted_FP32, riscv)
{
    AddWeightedTest<float, 1>(640, 720);
    AddWeightedTest<float, 3>(640, 720);
    AddWeightedTest<float, 4>(640, 720);
}
TEST(AddWeighted_INT8, riscv)
{
    AddWeightedTest<uint8_t, 1>(640, 720);
    AddWeightedTest<uint8_t, 3>(640, 720);
    AddWeightedTest<uint8_t, 4>(640, 720);
}
