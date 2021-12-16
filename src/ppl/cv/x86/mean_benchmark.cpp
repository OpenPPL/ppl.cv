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
#include "ppl/cv/x86/mean.h"
#include "ppl/cv/debug.h"

namespace {
template<typename T, int c>
class meanBenchmark {
public:
    int device = 0;
    T* dev_iImage = nullptr;
    int inHeight;
    int inWidth;
    int inWidthStride;
    meanBenchmark(int height, int width)
        : inHeight(height)
        , inWidth(width)
        , inWidthStride(inWidth * c)
    {
        dev_iImage = (T*)malloc(inHeight * inWidth * c * sizeof(T));
    }

    void apply() { 
        float outMean_pplcv[4] = {0,0,0,0};
        ppl::cv::x86::Mean<T, c>(inHeight, inWidth, inWidthStride, dev_iImage, outMean_pplcv);
    }
    void apply_opencv() {
        cv::setNumThreads(0);
        cv::Mat iMat_src(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), dev_iImage);
        cv::Scalar opencv_mean, opencv_std;
        cv::meanStdDev(iMat_src, opencv_mean, opencv_std);
    }

    ~meanBenchmark() {
        free(this->dev_iImage);
    }
};
}

using namespace ppl::cv::debug;
template<typename T, int c>
static void BM_Mean_ppl_x86(benchmark::State &state) {
    meanBenchmark<T, c> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * c);
}

template<typename T, int c>
static void BM_Mean_opencv_x86(benchmark::State &state) {
    meanBenchmark<T, c> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * c);
}

//ppl.cv
BENCHMARK_TEMPLATE(BM_Mean_ppl_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_Mean_ppl_x86, float, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480}); 
//opencv
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86, float, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480}); 
