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

#ifndef __ST_HPC_PPL3_CV_DEBUG_H_
#define __ST_HPC_PPL3_CV_DEBUG_H_

#include <random>

#if defined(PPLCV_UNITTEST_OPENCV) || defined(PPLCV_BENCHMARK_OPENCV)
#include <opencv2/imgproc.hpp>
template <typename T, int32_t channels>
struct T2CvType {};
template <int32_t channels>
struct T2CvType<float, channels> {
    static constexpr int32_t type = CV_MAKETYPE(CV_32F, channels);
};
template <int32_t channels>
struct T2CvType<uint8_t, channels> {
    static constexpr int32_t type = CV_MAKETYPE(CV_8U, channels);
};
template<int32_t channels>
struct T2CvType<int16_t, channels> {
    static constexpr int32_t type = CV_MAKETYPE(CV_16S, channels);
};

#define CV_GET_CHANNELS(type) (((type) >> 3) + 1)
#define CV_GET_TYPE(type) ((type)&0x07)

#endif // PPLCV_UNITTEST_OPENCV || PPLCV_BENCHMARK_OPENCV

#include <random>
namespace ppl {
namespace cv {

/**
 * \brief
 * There are some utils and tricky code for debug and benchmark in this namespace.
 * It is not allowed to use these codes in product code.
 *******************************************************/
namespace debug {

constexpr int32_t c1 = 1;   //!< One channel
constexpr int32_t c2 = 2;   //!< Two channels
constexpr int32_t c3 = 3;   //!< Three channels
constexpr int32_t c4 = 4;   //!< Four channels
constexpr int32_t k1x1 = 1; //!< kernel 1x1
constexpr int32_t k3x3 = 3; //!< kernel 3x3
constexpr int32_t k5x5 = 5; //!< kernel 5x5
constexpr int32_t k7x7 = 7; //!< kernel 7x7
constexpr int32_t k9x9 = 9; //!< kernel 9x9
constexpr int32_t k11x11 = 11; //!< kernel 11x11
constexpr int32_t k13x13 = 13; //!< kernel 13x13
constexpr int32_t k15x15 = 15; //!< kernel 15x15
constexpr int32_t k17x17 = 17; //!< kernel 17x17

constexpr int32_t connectivity4 = 4;
constexpr int32_t connectivity8 = 8;

constexpr int32_t padding1 = 1;
constexpr int32_t padding2 = 2;
constexpr int32_t padding3 = 3;
constexpr int32_t padding4 = 4;
constexpr int32_t padding5 = 5;
constexpr int32_t padding6 = 6;
constexpr int32_t padding7 = 7;
constexpr int32_t padding8 = 8;

constexpr int32_t subsample1 = 1;
constexpr int32_t subsample2 = 2;
constexpr int32_t subsample3 = 3;
constexpr int32_t subsample4 = 4;

template<typename T, bool is_integral>
class RandomFillImpl;

template<typename T>
class RandomFillImpl<T, false> {
public:
    static void randomFill(T* array, size_t N, T min, T max) {
        std::default_random_engine eng(clock());
        std::uniform_real_distribution<T> dis(min, max);
        for (size_t i = 0; i < N; ++i) {
            array[i] = dis(eng);
        }
    }
};

template<typename T>
class RandomFillImpl<T, true> {
public:
    static void randomFill(T* array, size_t N, T min, T max) {
        std::default_random_engine eng(clock());
#ifdef _MSC_VER
        // vs2015 does not support uniform_int_distribution<uint8_t>
        std::uniform_int_distribution<int64_t> dis(min, max);
#else
        std::uniform_int_distribution<T> dis(min, max);
#endif
        for (size_t i = 0; i < N; ++i) {
            array[i] = dis(eng);
        }
    }
};

template<typename T>
inline void randomFill(T* array, size_t N, T min, T max)
{
    RandomFillImpl<T, std::is_integral<T>::value>::randomFill(array, N, min, max);
}

template<typename T>
inline void randomFill(T* array, size_t N)
{
    RandomFillImpl<T, std::is_integral<T>::value>::randomFill(array, N, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

} // namespace debug
} // namespace cv
} // namespace ppl

#endif //! __ST_HPC_PPL3_CV_DEBUG_H_
