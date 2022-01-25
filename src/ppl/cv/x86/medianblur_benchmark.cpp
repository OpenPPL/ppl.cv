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
#include "ppl/cv/x86/medianblur.h"
#include "ppl/cv/debug.h"

namespace {

template<typename T, int32_t channels, int32_t filter_size>
void BM_MedianBlur_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::MedianBlur<T, channels>(height, width, width * channels, src.get(), width * channels, dst.get(),
                                filter_size, ppl::cv::BORDER_DEFAULT);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_x86, float, c1, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_x86, float, c3, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_x86, float, c4, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_x86, float, c1, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_x86, float, c3, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_x86, float, c4, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});


#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t channels, int32_t filter_size>
static void BM_MedianBlur_opencv_x86(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get(), sizeof(T) * width * channels);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst.get(), sizeof(T) * width * channels);

    for (auto _ : state) {
        cv::medianBlur(src_opencv, dst_opencv, filter_size);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86, float, c1, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86, float, c3, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86, float, c4, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86, float, c1, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86, float, c3, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86, float, c4, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}
