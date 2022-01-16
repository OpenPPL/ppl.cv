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
#include "ppl/cv/x86/split.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template <typename T, int32_t nc>
void BM_Split_ppl_x86(benchmark::State &state)
{
    int32_t width  = state.range(0);
    int32_t height = state.range(1);
    T *src         = new T[width * height * nc];
    T *dst[nc];
    for (int32_t i = 0; i < nc; ++i) {
        dst[i] = new T[width * height];
    }
    ppl::cv::debug::randomFill<T>(src, width * height * nc, 0, 255);
    for (auto _ : state) {
        if (nc == 3) {
            ppl::cv::x86::Split3Channels(height, width, width * nc, src, width, dst[0], dst[1], dst[2]);
        } else if (nc == 4) {
            ppl::cv::x86::Split4Channels(height, width, width * nc, src, width, dst[0], dst[1], dst[2], dst[3]);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
    delete[] src;
    for (int32_t i = 0; i < nc; ++i) {
        delete[] dst[i];
    }
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Split_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Split_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Split_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Split_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int32_t nc>
static void BM_Split_opencv_x86(benchmark::State &state)
{
    int32_t width  = state.range(0);
    int32_t height = state.range(1);
    T *src         = new T[width * height * nc];
    T *dst[nc];
    for (int32_t i = 0; i < nc; ++i) {
        dst[i] = new T[width * height];
    }
    ppl::cv::debug::randomFill<T>(src, width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src, sizeof(T) * width * nc);
    cv::Mat dst_opencv[nc];

    for (int32_t i = 0; i < nc; ++i) {
        dst_opencv[i] = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), dst[i], sizeof(T) * width);
    }
    for (auto _ : state) {
        cv::split(src_opencv, dst_opencv);
    }
    state.SetItemsProcessed(state.iterations() * 1);
    delete[] src;
    for (int32_t i = 0; i < nc; ++i) {
        delete[] dst[i];
    }
}

BENCHMARK_TEMPLATE(BM_Split_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Split_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Split_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Split_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
} // namespace
