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

#include "ppl/cv/x86/setvalue.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/types.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int32_t nc, bool use_mask, int32_t mask_nc = 0>
void SetToTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<uint8_t[]> mask(new uint8_t[width * height * mask_nc]);
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, mask_nc), mask.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get());
    ppl::cv::debug::randomFill<T>(dst.get(), width * height * nc, 0, 255);
    memcpy(dst_ref.get(), dst.get(), sizeof(T) * height * width * nc);
    for (int32_t i = 0; i < height * width * mask_nc; ++i) {
        mask.get()[i] = std::rand() % 2;
    }
    T value = static_cast<T>(17);
    if (use_mask) {
        ppl::cv::x86::SetTo<T, nc, mask_nc>(height, width, width * nc, dst.get(), static_cast<T>(value), width * mask_nc, mask.get());
        dstMat.setTo(value, maskMat);
    } else {
        ppl::cv::x86::SetTo<T, nc>(height, width, width * nc, dst.get(), static_cast<T>(value), 0, nullptr);
        dstMat.setTo(value);
    }
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1e-3);
}



TEST(SetToUChar, x86)
{
    SetToTest<uint8_t, 1, true, 1>(480, 640);
    SetToTest<uint8_t, 3, true, 1>(480, 640);
    SetToTest<uint8_t, 3, true, 3>(480, 640);
    SetToTest<uint8_t, 4, true, 1>(480, 640);
    SetToTest<uint8_t, 4, true, 4>(480, 640);

    SetToTest<uint8_t, 1, false>(480, 640);
    SetToTest<uint8_t, 3, false>(480, 640);
    SetToTest<uint8_t, 4, false>(480, 640);
}

TEST(SetToFP32, x86)
{
    SetToTest<float, 1, true, 1>(480, 640);
    SetToTest<float, 3, true, 1>(480, 640);
    SetToTest<float, 3, true, 3>(480, 640);
    SetToTest<float, 4, true, 1>(480, 640);
    SetToTest<float, 4, true, 4>(480, 640);

    SetToTest<float, 1, false>(480, 640);
    SetToTest<float, 3, false>(480, 640);
    SetToTest<float, 4, false>(480, 640);

}

template<typename T, int32_t nc>
void OnesTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::x86::Ones<T, nc>(height, width, width * nc, dst.get());
    cv::Mat dstMat = cv::Mat::ones(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc));
    T* dst_ref = dstMat.ptr<T>(0);
    checkResult<T, nc>(dst.get(), dst_ref, height, width, width * nc, width * nc, 1e-3);
}



TEST(OnesUCHAR, x86)
{
    OnesTest<uint8_t, 1>(480, 640);
    OnesTest<uint8_t, 3>(480, 640);
    OnesTest<uint8_t, 4>(480, 640);

    OnesTest<uint8_t, 1>(480, 640);
    OnesTest<uint8_t, 3>(480, 640);
    OnesTest<uint8_t, 4>(480, 640);

}


TEST(OnesFP32, x86)
{
    OnesTest<float, 1>(480, 640);
    OnesTest<float, 3>(480, 640);
    OnesTest<float, 4>(480, 640);

    OnesTest<float, 1>(480, 640);
    OnesTest<float, 3>(480, 640);
    OnesTest<float, 4>(480, 640);
}

template<typename T, int32_t nc>
void ZerosTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::x86::Zeros<T, nc>(height, width, width * nc, dst.get());
    cv::Mat dstMat = cv::Mat::zeros(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc));
    T* dst_ref = dstMat.ptr<T>(0);
    checkResult<T, nc>(dst.get(), dst_ref, height, width, width * nc, width * nc, 1e-3);
}



TEST(ZerosUCHAR, x86)
{
    ZerosTest<uint8_t, 1>(480, 640);
    ZerosTest<uint8_t, 3>(480, 640);
    ZerosTest<uint8_t, 4>(480, 640);

    ZerosTest<uint8_t, 1>(480, 640);
    ZerosTest<uint8_t, 3>(480, 640);
    ZerosTest<uint8_t, 4>(480, 640);

}


TEST(ZerosFP32, x86)
{
    ZerosTest<float, 1>(480, 640);
    ZerosTest<float, 3>(480, 640);
    ZerosTest<float, 4>(480, 640);

    ZerosTest<float, 1>(480, 640);
    ZerosTest<float, 3>(480, 640);
    ZerosTest<float, 4>(480, 640);
}
