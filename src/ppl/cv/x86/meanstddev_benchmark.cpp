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
#include "ppl/cv/x86/meanstddev.h"
#include "ppl/cv/debug.h"

namespace {
template<typename T, int32_t channels>
class MeanStdDevBenchmark {
public:
    int32_t device = 0;
    T* dev_iImage = nullptr;
    int32_t height;
    int32_t width;
    MeanStdDevBenchmark(int32_t height, int32_t width)
        : height(height)
        , width(width)
    {
        dev_iImage = (T*)malloc(height * width * channels * sizeof(T));
        memset(this->dev_iImage, 0, height * width * channels * sizeof(T));
    }

    void apply() {
        float pplcv_mean[4], pplcv_std[4];     
        int32_t stride = width * channels;    
        ppl::cv::x86::MeanStdDev<T, channels>(height, width, stride, dev_iImage, pplcv_mean, pplcv_std);
    }

    void apply_opencv() {
        cv::setNumThreads(0);
        cv::Mat iMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage);
        cv::Scalar opencv_mean, opencv_std;
        cv::meanStdDev(iMat, opencv_mean, opencv_std);
    }

    ~MeanStdDevBenchmark() {
        free(this->dev_iImage);
    }
};
}

using namespace ppl::cv::debug;
template<typename T, int32_t channels>
static void BM_MeanStdDev_ppl_x86(benchmark::State &state) {
    MeanStdDevBenchmark<T, channels> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * channels);
}

BENCHMARK_TEMPLATE(BM_MeanStdDev_ppl_x86, float, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MeanStdDev_ppl_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t channels>
static void BM_MeanStdDev_opencv_x86(benchmark::State &state) {
    MeanStdDevBenchmark<T, channels> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * channels);
}
BENCHMARK_TEMPLATE(BM_MeanStdDev_opencv_x86, float, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MeanStdDev_opencv_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MeanStdDev_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480});
#endif //! PPLCV_BENCHMARK_OPENCV

