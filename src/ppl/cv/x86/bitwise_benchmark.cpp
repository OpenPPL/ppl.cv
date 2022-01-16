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

#include <benchmark/benchmark.h>
#include "ppl/cv/x86/bitwise.h"
#include "ppl/cv/debug.h"

namespace {
template<typename T, int32_t channels>
class BitwiseBenchmark {
public:
    int32_t device = 0;
    T* dev_iImage1 = nullptr;
    T* dev_iImage2 = nullptr;
    T* dev_iMask = nullptr;
    T* dev_oImage = nullptr;
    int32_t height;
    int32_t width;
    BitwiseBenchmark(int32_t height, int32_t width)
        : height(height)
        , width(width)
    {
        dev_iImage1 = (T*)malloc(height * width * channels * sizeof(T));
        dev_iImage2 = (T*)malloc(height * width * channels * sizeof(T));
        dev_iMask  = (T*)malloc(height * width * sizeof(uint8_t));
        dev_oImage = (T*)malloc(height * width * channels * sizeof(T));
    }

    void apply() {
        int32_t stride = width * channels;
        ppl::cv::x86::BitwiseAnd<T, channels>(height, width, stride, dev_iImage1, stride, dev_iImage2, stride, dev_oImage);
    }

    void apply_opencv() {
        cv::Mat iMat1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage1);
        cv::Mat iMat2(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage2);
        cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage);
        cv::bitwise_and(iMat1, iMat2, oMat);
    }

    ~BitwiseBenchmark() {
        free(this->dev_iImage1);
        free(this->dev_iImage2);
        free(this->dev_oImage);
        free(this->dev_iMask);
    }
};
}

using namespace ppl::cv::debug;
template<typename T, int32_t channels>
static void BM_BitwiseAnd_ppl_x86(benchmark::State &state) {
    BitwiseBenchmark<T, channels> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * channels);
}

BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t channels>
static void BM_BitwiseAnd_opencv_x86(benchmark::State &state) {
    BitwiseBenchmark<T, channels> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * channels);
}
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480});
#endif //! PPLCV_BENCHMARK_OPENCV

