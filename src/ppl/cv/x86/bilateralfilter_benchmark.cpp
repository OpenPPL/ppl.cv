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
#include <memory>
#include "ppl/cv/x86/bilateralfilter.h"
#include "ppl/cv/debug.h"
#include <opencv2/imgproc.hpp>

namespace {

template<typename T, int32_t channels, int32_t diameter, int32_t color, int32_t space>
void BM_Bilateral_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::BilateralFilter<T, channels>(height, width, width * channels,
                                src.get(),
                                diameter,
                                color,
                                space, 
                                width * channels,
                                dst.get(),
                                ppl::cv::BORDER_DEFAULT);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Bilateral_ppl_x86, float, c3, 9, 75, 75)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Bilateral_ppl_x86, uint8_t, c3, 9, 75, 75)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t channels, int32_t diameter, int32_t color, int32_t space>
void BM_Bilateral_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get(), sizeof(T) * width * channels);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst.get(), sizeof(T) * width * channels);
    for (auto _ : state) {
        cv::bilateralFilter(src_opencv, dst_opencv, diameter, color, space);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}


BENCHMARK_TEMPLATE(BM_Bilateral_opencv_x86, float, c3, 9, 75, 75)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Bilateral_opencv_x86, uint8_t, c3, 9, 75, 75)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
}
