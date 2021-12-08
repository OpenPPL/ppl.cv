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

#ifndef __ST_HPC_PPL_CV_X86_TEST_H_
#define __ST_HPC_PPL_CV_X86_TEST_H_

#include <gtest/gtest.h>
#include <cmath>
#include <float.h>
#include <random>

template <typename T, int32_t nc>
inline void checkResult(const T* data1,
                        const T* data2,
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

template <typename T, bool is_integral>
class RandomFillImpl;

template <typename T>
class RandomFillImpl<T, false> {
public:
    static void randomFill(T* array, size_t N, T min, T max)
    {
        std::default_random_engine eng(clock());
        std::uniform_real_distribution<T> dis(min, max);
        for (size_t i = 0; i < N; ++i) {
            array[i] = dis(eng);
        }
    }
};

template <typename T>
class RandomFillImpl<T, true> {
public:
    static void randomFill(T* array, size_t N, T min, T max)
    {
        std::default_random_engine eng(clock());
        std::uniform_int_distribution<T> dis(min, max);
        for (size_t i = 0; i < N; ++i) {
            array[i] = dis(eng);
        }
    }
};

template <typename T>
inline void randomFill(T* array, size_t N, T min, T max)
{
    RandomFillImpl<T, std::is_integral<T>::value>::randomFill(array, N, min, max);
}

template <typename T>
inline void randomFill(T* array, size_t N)
{
    RandomFillImpl<T, std::is_integral<T>::value>::randomFill(array, N, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

struct Size {
    int width;
    int height;
};
#endif //!__ST_HPC_PPL_CV_X86_TEST_H_
