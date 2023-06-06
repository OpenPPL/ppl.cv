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

#include "ppl/cv/arm/split.h"

#include <chrono>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels>
void BM_Split_ppl_aarch64(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst2(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst3(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        if (channels == 3) {
            ppl::cv::arm::Split3Channels<T>(src.rows,
                                            src.cols,
                                            src.step / sizeof(T),
                                            (T *)src.data,
                                            dst0.step / sizeof(T),
                                            (T *)dst0.data,
                                            (T *)dst1.data,
                                            (T *)dst2.data);
        } else { // channels == 4
            ppl::cv::arm::Split4Channels<T>(src.rows,
                                            src.cols,
                                            src.step / sizeof(T),
                                            (T *)src.data,
                                            dst0.step / sizeof(T),
                                            (T *)dst0.data,
                                            (T *)dst1.data,
                                            (T *)dst2.data,
                                            (T *)dst3.data);
        }
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            if (channels == 3) {
                ppl::cv::arm::Split3Channels<T>(src.rows,
                                                src.cols,
                                                src.step / sizeof(T),
                                                (T *)src.data,
                                                dst0.step / sizeof(T),
                                                (T *)dst0.data,
                                                (T *)dst1.data,
                                                (T *)dst2.data);
            } else { // channels == 4
                ppl::cv::arm::Split4Channels<T>(src.rows,
                                                src.cols,
                                                src.step / sizeof(T),
                                                (T *)src.data,
                                                dst0.step / sizeof(T),
                                                (T *)dst0.data,
                                                (T *)dst1.data,
                                                (T *)dst2.data,
                                                (T *)dst3.data);
            }
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, channels)                                                                 \
    BENCHMARK_TEMPLATE(BM_Split_ppl_aarch64, type, channels)->Args({320, 240})->UseManualTime()->Iterations(10);  \
    BENCHMARK_TEMPLATE(BM_Split_ppl_aarch64, type, channels)->Args({640, 480})->UseManualTime()->Iterations(10);  \
    BENCHMARK_TEMPLATE(BM_Split_ppl_aarch64, type, channels)->Args({1280, 720})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Split_ppl_aarch64, type, channels)->Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(float, c3)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c4)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, c3)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, c4)

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int32_t channels>
void BM_Split_opencv_aarch64(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst2(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst3(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));

    cv::Mat dsts0[3] = {dst0, dst1, dst2};
    cv::Mat dsts1[4] = {dst0, dst1, dst2, dst3};

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        if (channels == 3) {
            cv::split(src, dsts0);
        } else { // channels == 4
            cv::split(src, dsts1);
        }
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            if (channels == 3) {
                cv::split(src, dsts0);
            } else { // channels == 4
                cv::split(src, dsts1);
            }
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_OPENCV_TYPE_FUNCTIONS(type, channels)                                                                    \
    BENCHMARK_TEMPLATE(BM_Split_opencv_aarch64, type, channels)->Args({320, 240})->UseManualTime()->Iterations(10);  \
    BENCHMARK_TEMPLATE(BM_Split_opencv_aarch64, type, channels)->Args({640, 480})->UseManualTime()->Iterations(10);  \
    BENCHMARK_TEMPLATE(BM_Split_opencv_aarch64, type, channels)->Args({1280, 720})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Split_opencv_aarch64, type, channels)->Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(float, c3)
RUN_OPENCV_TYPE_FUNCTIONS(float, c4)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, c3)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, c4)

#endif //! PPLCV_BENCHMARK_OPENCV
