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

#include "ppl/cv/x86/merge.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"


template<typename T, int32_t nc>
void MergeTest(int32_t height, int32_t width, T diff) {
    T *src[nc];
    T *dst;
    T *dst_ref;
    for (int32_t i = 0; i < nc; ++i) {
        src[i] = new T[width * height];
        ppl::cv::debug::randomFill<T>(src[i], width * height, 0, 255);
    }
    dst = new T[width * height * nc];
    dst_ref = new T[width * height * nc];

    cv::Mat src_opencv[nc];
    cv::Mat dst_opencv;

    for (int32_t i = 0; i < nc; ++i) {
        src_opencv[i] = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), src[i], sizeof(T) * width);
    }
    dst_opencv = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref, sizeof(T) * width * nc);

    cv::merge(src_opencv, nc, dst_opencv);
    if (nc == 3) {
        ppl::cv::x86::Merge3Channels(height, width, width,
                                src[0], src[1], src[2], width * nc,
                                dst);
    } else if (nc == 4) {
        ppl::cv::x86::Merge4Channels(height, width, width,
                                src[0], src[1], src[2], src[3], width * nc,
                                dst);
    }
    checkResult<T, nc>(dst, dst_ref,
                    height, width,
                    width, width,
                    diff);



    for (int32_t i = 0; i < nc; ++i) {
        delete[] src[i];
    }
    delete[] dst;
    delete[] dst_ref;
}


TEST(MERGE_FP32, x86)
{
    for (int32_t h = 720; h < 1080; h += 15) {
        for (int32_t w = 1080; w < 1280; w += 15) {
            MergeTest<float, 3>(720, 1080, 0.01f);
            MergeTest<float, 4>(720, 1080, 0.01f);
        }
    }
}

TEST(MERGE_UINT8, x86)
{
    MergeTest<uint8_t, 3>(720, 1080, 1.01f);
    MergeTest<uint8_t, 4>(720, 1080, 1.01f);
}
