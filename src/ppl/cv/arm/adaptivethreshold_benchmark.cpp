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

#include "ppl/cv/arm/adaptivethreshold.h"

#include <chrono>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <int ksize, int adaptive_method>
void BM_AdaptiveThreshold_ppl_aarch64(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1));
    cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1));

    float max_value = 155.f;
    float delta = 10.f;
    int threshold_type = ppl::cv::THRESH_BINARY;
    ppl::cv::BorderType border_type = ppl::cv::BORDER_REPLICATE;

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        ppl::cv::arm::AdaptiveThreshold(src.rows,
                                        src.cols,
                                        src.step,
                                        (uint8_t *)src.data,
                                        dst.step,
                                        (uint8_t *)dst.data,
                                        max_value,
                                        adaptive_method,
                                        threshold_type,
                                        ksize,
                                        delta,
                                        border_type);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            ppl::cv::arm::AdaptiveThreshold(src.rows,
                                            src.cols,
                                            src.step,
                                            (uint8_t *)src.data,
                                            dst.step,
                                            (uint8_t *)dst.data,
                                            max_value,
                                            adaptive_method,
                                            threshold_type,
                                            ksize,
                                            delta,
                                            border_type);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_PPL_CV_TYPE_FUNCTIONS(ksize, adaptive_method)                        \
    BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_ppl_aarch64, ksize, adaptive_method) \
        ->Args({640, 480})                                                       \
        ->UseManualTime()                                                        \
        ->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)

#ifdef PPLCV_BENCHMARK_OPENCV
template <int ksize, int adaptive_method>
void BM_AdaptiveThreshold_opencv_aarch64(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1));
    cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1));

    float max_value = 155.f;
    float delta = 10.f;
    int threshold_type = ppl::cv::THRESH_BINARY;

    cv::AdaptiveThresholdTypes cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
    if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_MEAN_C) {
        cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
    } else if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C) {
        cv_adaptive_method = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
    } else {
    }

    cv::ThresholdTypes cv_threshold_type = cv::THRESH_BINARY;
    if (threshold_type == ppl::cv::THRESH_BINARY) {
        cv_threshold_type = cv::THRESH_BINARY;
    } else if (threshold_type == ppl::cv::THRESH_BINARY_INV) {
        cv_threshold_type = cv::THRESH_BINARY_INV;
    }

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        cv::adaptiveThreshold(src, dst, max_value, cv_adaptive_method, cv_threshold_type, ksize, delta);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            cv::adaptiveThreshold(src, dst, max_value, cv_adaptive_method, cv_threshold_type, ksize, delta);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_OPENCV_TYPE_FUNCTIONS(ksize, adaptive_method)                           \
    BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_opencv_aarch64, ksize, adaptive_method) \
        ->Args({640, 480})                                                          \
        ->UseManualTime()                                                           \
        ->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)

#endif //! PPLCV_BENCHMARK_OPENCV
