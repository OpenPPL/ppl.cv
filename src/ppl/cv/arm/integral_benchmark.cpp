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

#include "ppl/cv/arm/integral.h"

#include <chrono>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst, int channels>
void BM_Integral_ppl_aarch64(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
    cv::Mat dst(height + 1, width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        ppl::cv::arm::Integral<Tsrc, Tdst, channels>(src.rows,
                                                     src.cols,
                                                     src.step / sizeof(Tsrc),
                                                     (Tsrc *)src.data,
                                                     dst.rows,
                                                     dst.cols,
                                                     dst.step / sizeof(Tdst),
                                                     (Tdst *)dst.data);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            ppl::cv::arm::Integral<Tsrc, Tdst, channels>(src.rows,
                                                         src.cols,
                                                         src.step / sizeof(Tsrc),
                                                         (Tsrc *)src.data,
                                                         dst.rows,
                                                         dst.cols,
                                                         dst.step / sizeof(Tdst),
                                                         (Tdst *)dst.data);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_PPL_CV_TYPE_FUNCTIONS(tsrc, tdst, channels)               \
    BENCHMARK_TEMPLATE(BM_Integral_ppl_aarch64, tsrc, tdst, channels) \
        ->Args({640, 480})                                            \
        ->UseManualTime()                                             \
        ->Iterations(10);                                             \
    BENCHMARK_TEMPLATE(BM_Integral_ppl_aarch64, tsrc, tdst, channels) \
        ->Args({1920, 1080})                                          \
        ->UseManualTime()                                             \
        ->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(float, float, c1)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, c3)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, c4)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, int32_t, c1)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, int32_t, c3)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, int32_t, c4)

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename Tsrc, typename Tdst, int32_t channels>
void BM_Integral_opencv_aarch64(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
    cv::Mat dst(height + 1, width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        cv::integral(src, dst);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            cv::integral(src, dst);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_OPENCV_TYPE_FUNCTIONS(tsrc, tdst, channels)                  \
    BENCHMARK_TEMPLATE(BM_Integral_opencv_aarch64, tsrc, tdst, channels) \
        ->Args({640, 480})                                               \
        ->UseManualTime()                                                \
        ->Iterations(10);                                                \
    BENCHMARK_TEMPLATE(BM_Integral_opencv_aarch64, tsrc, tdst, channels) \
        ->Args({1920, 1080})                                             \
        ->UseManualTime()                                                \
        ->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(float, float, c1)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, c3)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, c4)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, int32_t, c1)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, int32_t, c3)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, int32_t, c4)

#endif //! PPLCV_BENCHMARK_OPENCV
