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
#include "ppl/cv/x86/merge.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template<typename T, int32_t nc>
void BM_Merge_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    T *src[nc];
    T *dst;
    for (int32_t i = 0; i < nc; ++i) {
        src[i] = new T[width * height];    
        ppl::cv::debug::randomFill<T>(src[i], width * height, 0, 255);
    } 
    dst = new T[width * height * nc];

    for (auto _ : state) {
        if (nc == 3) {
            ppl::cv::x86::Merge3Channels(height, width, width,
                                    src[0], src[1], src[2], width * nc,
                                    dst);
        } else if (nc == 4) {
            ppl::cv::x86::Merge4Channels(height, width, width,
                                    src[0], src[1], src[2], src[3], width * nc,
                                    dst);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
    for (int32_t i = 0; i < nc; ++i) { 
        delete[] src[i];
    }
    delete[] dst;
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Merge_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Merge_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Merge_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Merge_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t nc>
static void BM_Merge_opencv_x86(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    T *src[nc];
    T *dst;
    for (int32_t i = 0; i < nc; ++i) {
        src[i] = new T[width * height];    
        ppl::cv::debug::randomFill<T>(src[i], width * height, 0, 255);
    } 
    dst = new T[width * height * nc];

    cv::Mat src_opencv[nc];
    cv::Mat dst_opencv;
    
    for (int32_t i = 0; i < nc; ++i) {
        src_opencv[i] = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), src[i], sizeof(T) * width);
    }
    dst_opencv = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst, sizeof(T) * width * nc);

    for (auto _ : state) {
        cv::merge(src_opencv, nc, dst_opencv);
    }
    state.SetItemsProcessed(state.iterations() * 1);

    for (int32_t i = 0; i < nc; ++i) { 
        delete[] src[i];
    }
    delete[] dst;
}

BENCHMARK_TEMPLATE(BM_Merge_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Merge_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Merge_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Merge_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}
