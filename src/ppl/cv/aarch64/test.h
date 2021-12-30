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

#ifndef __ST_HPC_PPL_CV_AARCH64_TEST_H_
#define __ST_HPC_PPL_CV_AARCH64_TEST_H_

#include <gtest/gtest.h>
#include <cmath>
#include <float.h>
#include <random>
#include <iostream>

#define USE_QUANTIZED

template <typename T, int32_t nc>
inline void checkResult(const T *data1,
                        const T *data2,
                        const int32_t height,
                        const int32_t width,
                        int32_t dstep1,
                        int32_t dstep2,
                        const float diff_THR)
{
    double max       = DBL_MIN;
    double min       = DBL_MAX;
    double temp_diff = 0.f;

    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width * nc; j++) {
            float val1 = data1[i * dstep1 + j];
            float val2 = data2[i * dstep2 + j];
            temp_diff  = fabs(val1 - val2);
            max        = (temp_diff > max) ? temp_diff : max;
            min        = (temp_diff < min) ? temp_diff : min;
        }
    }
    EXPECT_LT(max, diff_THR);
}

struct Size {
    int width;
    int height;
};

template <typename T, int nc>
void checkResult(const T *data1,
                 const T *data2,
                 const int height,
                 const int width,
                 int dstep1,
                 int dstep2,
                 const float diff_THR,
                 const float percentage)
{
    double temp_diff = 0.f;
    int count        = 0;

    //int num = height * width * nc;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width * nc; j++) {
            float val1 = data1[i * dstep1 + j];
            float val2 = data2[i * dstep2 + j];
            temp_diff  = fabs(val1 - val2);
            if (temp_diff > diff_THR) {
#ifdef PRINT_ALL_ARM
                std::cout << "Print data12: "
                          << "h:" << i << " w:" << j << " : "
                          << (float)val1 << " : " << (float)val2 << " diff: " << temp_diff
                          << std::endl;
#endif
                count++;
            }
        }
    }
    float ratio = (float)count / (width * nc * height);
    EXPECT_LT(ratio, percentage);
}

#endif //!__ST_HPC_PPL_CV_AARCH64_TEST_H_
