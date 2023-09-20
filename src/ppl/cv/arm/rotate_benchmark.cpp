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

#include "ppl/cv/arm/rotate.h"

#include <chrono>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, int degree>
void BM_Rotate_ppl_aarch64(benchmark::State &state)
{
    int src_width = state.range(0);
    int src_height = state.range(1);
    int dst_height, dst_width;
    if (degree == 90) {
        dst_height = src_width;
        dst_width = src_height;
    } else if (degree == 180) {
        dst_height = src_height;
        dst_width = src_width;
    } else if (degree == 270) {
        dst_height = src_width;
        dst_width = src_height;
    } else {
        return;
    }
    cv::Mat src = createSourceImage(src_height, src_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(dst_height, dst_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        ppl::cv::arm::Rotate<T, channels>(src_height,
                                          src_width,
                                          src.step / sizeof(T),
                                          (T *)src.data,
                                          dst_height,
                                          dst_width,
                                          dst.step / sizeof(T),
                                          (T *)dst.data,
                                          degree);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            ppl::cv::arm::Rotate<T, channels>(src_height,
                                              src_width,
                                              src.step / sizeof(T),
                                              (T *)src.data,
                                              dst_height,
                                              dst_width,
                                              dst.step / sizeof(T),
                                              (T *)dst.data,
                                              degree);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, degree)                                                                       \
    BENCHMARK_TEMPLATE(BM_Rotate_ppl_aarch64, type, c1, degree)->Args({640, 480})->UseManualTime()->Iterations(10);   \
    BENCHMARK_TEMPLATE(BM_Rotate_ppl_aarch64, type, c3, degree)->Args({640, 480})->UseManualTime()->Iterations(10);   \
    BENCHMARK_TEMPLATE(BM_Rotate_ppl_aarch64, type, c4, degree)->Args({640, 480})->UseManualTime()->Iterations(10);   \
    BENCHMARK_TEMPLATE(BM_Rotate_ppl_aarch64, type, c1, degree)->Args({1920, 1080})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Rotate_ppl_aarch64, type, c3, degree)->Args({1920, 1080})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Rotate_ppl_aarch64, type, c4, degree)->Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(float, 90)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 180)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 270)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, 90)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, 180)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, 270)

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int32_t channels, int32_t degree>
void BM_Rotate_opencv_aarch64(benchmark::State &state)
{
    int src_width = state.range(0);
    int src_height = state.range(1);
    int dst_height, dst_width;
    cv::RotateFlags cv_rotate_flag;
    if (degree == 90) {
        dst_height = src_width;
        dst_width = src_height;
        cv_rotate_flag = cv::ROTATE_90_CLOCKWISE;
    } else if (degree == 180) {
        dst_height = src_height;
        dst_width = src_width;
        cv_rotate_flag = cv::ROTATE_180;
    } else if (degree == 270) {
        dst_height = src_width;
        dst_width = src_height;
        cv_rotate_flag = cv::ROTATE_90_COUNTERCLOCKWISE;
    } else {
        return;
    }

    cv::Mat src = createSourceImage(src_height, src_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(dst_height, dst_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        cv::rotate(src, dst, cv_rotate_flag);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            cv::rotate(src, dst, cv_rotate_flag);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_OPENCV_TYPE_FUNCTIONS(type, degree)                                                                        \
    BENCHMARK_TEMPLATE(BM_Rotate_opencv_aarch64, type, c1, degree)->Args({640, 480})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Rotate_opencv_aarch64, type, c3, degree)->Args({640, 480})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Rotate_opencv_aarch64, type, c4, degree)->Args({640, 480})->UseManualTime()->Iterations(10); \
    BENCHMARK_TEMPLATE(BM_Rotate_opencv_aarch64, type, c1, degree)                                                     \
        ->Args({1920, 1080})                                                                                           \
        ->UseManualTime()                                                                                              \
        ->Iterations(10);                                                                                              \
    BENCHMARK_TEMPLATE(BM_Rotate_opencv_aarch64, type, c3, degree)                                                     \
        ->Args({1920, 1080})                                                                                           \
        ->UseManualTime()                                                                                              \
        ->Iterations(10);                                                                                              \
    BENCHMARK_TEMPLATE(BM_Rotate_opencv_aarch64, type, c4, degree)->Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(float, 90)
RUN_OPENCV_TYPE_FUNCTIONS(float, 180)
RUN_OPENCV_TYPE_FUNCTIONS(float, 270)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, 90)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, 180)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, 270)

#endif //! PPLCV_BENCHMARK_OPENCV
