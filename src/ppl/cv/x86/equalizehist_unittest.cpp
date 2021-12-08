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

#include "ppl/cv/x86/equalizehist.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include <opencv2/imgproc.hpp>

void EqualizeHistTest(int32_t height, int32_t width) {
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height, 0, 255);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), dst_ref.get());
    cv::equalizeHist(srcMat, dstMat);
    ppl::cv::x86::EqualizeHist(height, width, width, src.get(), width, dst.get());
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), height, width, width, width, 1.01f);
}

TEST(EqualizeHist_UINT8, x86)
{
    EqualizeHistTest(640, 720);
    EqualizeHistTest(720, 1080);
    EqualizeHistTest(1080, 1920);
}

