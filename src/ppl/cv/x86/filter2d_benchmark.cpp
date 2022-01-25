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
#include "ppl/cv/x86/filter2d.h"
#include "ppl/cv/debug.h"

namespace {

template<typename T, int32_t channels, int32_t filter_size>
void BM_Filter2D_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    std::unique_ptr<float[]> filter(new float[filter_size * filter_size]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    ppl::cv::debug::randomFill<float>(filter.get(), filter_size * filter_size, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::Filter2D<T, channels>(height, width, width * channels,
                                src.get(), filter_size, filter.get(), width * channels,
                                dst.get(), ppl::cv::BORDER_DEFAULT);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;


BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c1, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c3, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, float, c4, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c1, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c3, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_x86, uint8_t, c4, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t channels, int32_t filter_size>
static void BM_Filter2D_opencv_x86(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * channels]);
    std::unique_ptr<float[]> filter(new float[filter_size * filter_size]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    ppl::cv::debug::randomFill<float>(filter.get(), filter_size * filter_size, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get(), sizeof(T) * width * channels);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst_ref.get(), sizeof(T) * width * channels);
    cv::Mat filter_opencv(filter_size, 3, CV_32FC1, filter.get());

    for (auto _ : state) {
        cv::filter2D(src_opencv, dst_opencv, -1, filter_opencv);
    }
    state.SetItemsProcessed(state.iterations() * 1);

}


BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c1, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c3, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, float, c4, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c1, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c3, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_x86, uint8_t, c4, k7x7)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
}
