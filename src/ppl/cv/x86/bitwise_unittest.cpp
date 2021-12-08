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

#include "ppl/cv/x86/bitwise.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/types.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"


template<typename T, int32_t nc, bool use_mask>
void BitwiseAndTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<uint8_t[]> mask(new uint8_t[width * height]);
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), mask.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get());
    cv::Mat src0Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src0.get());
    cv::Mat src1Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src1.get());
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 0, 255);
    ppl::cv::debug::randomFill<uint8_t>(mask.get(), width * height, 0, 255);
    memcpy(dst_ref.get(), dst.get(), height * width * nc * sizeof(T));
    if (use_mask) {
        ppl::cv::x86::BitwiseAnd<T, nc>(height, width, width * nc, src0.get(),
                                                        width * nc, src1.get(),
                                                        width * nc, dst.get(),
                                                        width, mask.get());
        cv::bitwise_and(src0Mat, src1Mat, dstMat, maskMat);
    } else {
        ppl::cv::x86::BitwiseAnd<T, nc>(height, width, width * nc, src0.get(),
                                                        width * nc, src1.get(),
                                                        width * nc, dst.get());
        cv::bitwise_and(src0Mat, src1Mat, dstMat);
    }
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1e-3);
}



TEST(BitwiseAndMaskUint8, x86)
{
    BitwiseAndTest<uint8_t, 1, true>(480, 640);
    BitwiseAndTest<uint8_t, 3, true>(480, 640);
    BitwiseAndTest<uint8_t, 4, true>(480, 640);

    BitwiseAndTest<uint8_t, 1, true>(640, 720);
    BitwiseAndTest<uint8_t, 3, true>(640, 720);
    BitwiseAndTest<uint8_t, 4, true>(640, 720);

}

TEST(BitwiseAndUint8, x86)
{
    BitwiseAndTest<uint8_t, 1, false>(480, 640);
    BitwiseAndTest<uint8_t, 3, false>(480, 640);
    BitwiseAndTest<uint8_t, 4, false>(480, 640);

    BitwiseAndTest<uint8_t, 1, false>(640, 720);
    BitwiseAndTest<uint8_t, 3, false>(640, 720);
    BitwiseAndTest<uint8_t, 4, false>(640, 720);
}
