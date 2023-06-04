// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for mulitional information
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

#include "ppl/cv/arm/arithmetic.h"

#include <chrono>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

namespace {

template <typename T, int32_t channels>
void BM_Mls_ppl_aarch64(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    cv::Mat src0 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat src1 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        ppl::cv::arm::Mls<T, channels>(src0.rows,
                                       src0.cols,
                                       src0.step / sizeof(T),
                                       (T *)src0.data,
                                       src1.step / sizeof(T),
                                       (T *)src1.data,
                                       dst.step / sizeof(T),
                                       (T *)dst.data);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            ppl::cv::arm::Mls<T, channels>(src0.rows,
                                           src0.cols,
                                           src0.step / sizeof(T),
                                           (T *)src0.data,
                                           src1.step / sizeof(T),
                                           (T *)src1.data,
                                           dst.step / sizeof(T),
                                           (T *)dst.data);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_PPL_CV_TYPE_FUNCTIONS(type)                                                                    \
    BENCHMARK_TEMPLATE(BM_Mls_ppl_aarch64, type, c1)->Args({640, 480})->UseManualTime()->Iterations(10);   \
    BENCHMARK_TEMPLATE(BM_Mls_ppl_aarch64, type, c3)->Args({640, 480})->UseManualTime()->Iterations(10);   \
    BENCHMARK_TEMPLATE(BM_Mls_ppl_aarch64, type, c4)->Args({640, 480})->UseManualTime()->Iterations(10);   \
    BENCHMARK_TEMPLATE(BM_Mls_ppl_aarch64, type, c1)->Args({1920, 1080})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Mls_ppl_aarch64, type, c3)->Args({1920, 1080})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Mls_ppl_aarch64, type, c4)->Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(float)

} // namespace
