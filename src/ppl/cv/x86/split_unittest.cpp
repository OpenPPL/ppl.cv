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

#include "ppl/cv/x86/split.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename T, int32_t nc>
void SplitTest(int32_t height, int32_t width, T diff)
{
    T *src = new T[width * height * nc];
    T *dst[nc];
    T *dst_ref[nc];
    for (int32_t i = 0; i < nc; ++i) {
        dst[i]     = new T[width * height];
        dst_ref[i] = new T[width * height];
    }
    ppl::cv::debug::randomFill<T>(src, width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src, sizeof(T) * width * nc);
    cv::Mat dst_opencv[nc];

    for (int32_t i = 0; i < nc; ++i) {
        dst_opencv[i] = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), dst_ref[i], sizeof(T) * width);
    }

    cv::split(src_opencv, dst_opencv);
    if (nc == 3) {
        ppl::cv::x86::Split3Channels(height, width, width * nc, src, width, dst[0], dst[1], dst[2]);
    } else if (nc == 4) {
        ppl::cv::x86::Split4Channels(height, width, width * nc, src, width, dst[0], dst[1], dst[2], dst[3]);
    }
    for (int32_t i = 0; i < nc; ++i) {
        checkResult<T, 1>(dst_ref[i], dst[i], height, width, width, width, diff);
    }
    delete[] src;
    for (int32_t i = 0; i < nc; ++i) {
        delete[] dst[i];
        delete[] dst_ref[i];
    }
}

TEST(SPLIT_FP32, x86)
{
    for (int32_t h = 720; h < 800; h += 15) {
        for (int32_t w = 1080; w < 1280; w += 15) {
            SplitTest<float, 3>(h, w, 0.01f);
            SplitTest<float, 4>(h, w, 0.01f);
        }
    }
}

TEST(SPLIT_UINT8, x86)
{
    for (int32_t h = 720; h < 800; h += 15) {
        for (int32_t w = 1080; w < 1280; w += 15) {
            SplitTest<uint8_t, 3>(h, w, 1.01f);
            SplitTest<uint8_t, 4>(h, w, 1.01f);
        }
    }
    // SplitTest<uint8_t, 3>(16, 33, 1.01f);
}
