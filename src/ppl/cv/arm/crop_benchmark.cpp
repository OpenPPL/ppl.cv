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

#include "ppl/cv/arm/crop.h"

#include <chrono>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, int left, int top, int int_scale>
void BM_Crop_ppl_aarch64(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    int src_width = width * 2;
    int src_height = height * 2;
    cv::Mat src = createSourceImage(src_height, src_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    float scale = int_scale / 10.f;

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        ppl::cv::arm::Crop<T, channels>(src.rows,
                                        src.cols,
                                        src.step / sizeof(T),
                                        (T *)src.data,
                                        dst.rows,
                                        dst.cols,
                                        dst.step / sizeof(T),
                                        (T *)dst.data,
                                        left,
                                        top,
                                        scale);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            ppl::cv::arm::Crop<T, channels>(src.rows,
                                            src.cols,
                                            src.step / sizeof(T),
                                            (T *)src.data,
                                            dst.rows,
                                            dst.cols,
                                            dst.step / sizeof(T),
                                            (T *)dst.data,
                                            left,
                                            top,
                                            scale);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, top, left, scale)               \
    BENCHMARK_TEMPLATE(BM_Crop_ppl_aarch64, type, c1, top, left, scale) \
        ->Args({640, 480})                                              \
        ->UseManualTime()                                               \
        ->Iterations(10);                                               \
    BENCHMARK_TEMPLATE(BM_Crop_ppl_aarch64, type, c3, top, left, scale) \
        ->Args({640, 480})                                              \
        ->UseManualTime()                                               \
        ->Iterations(10);                                               \
    BENCHMARK_TEMPLATE(BM_Crop_ppl_aarch64, type, c4, top, left, scale) \
        ->Args({640, 480})                                              \
        ->UseManualTime()                                               \
        ->Iterations(10);                                               \
    BENCHMARK_TEMPLATE(BM_Crop_ppl_aarch64, type, c1, top, left, scale) \
        ->Args({1920, 1080})                                            \
        ->UseManualTime()                                               \
        ->Iterations(10);                                               \
    BENCHMARK_TEMPLATE(BM_Crop_ppl_aarch64, type, c3, top, left, scale) \
        ->Args({1920, 1080})                                            \
        ->UseManualTime()                                               \
        ->Iterations(10);                                               \
    BENCHMARK_TEMPLATE(BM_Crop_ppl_aarch64, type, c4, top, left, scale) \
        ->Args({1920, 1080})                                            \
        ->UseManualTime()                                               \
        ->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, 10)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, 15)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, 16, 16, 10)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, 16, 16, 15)

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int channels, int left, int top, int int_scale>
void BM_Crop_opencv_aarch64(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    int src_width = width * 2;
    int src_height = height * 2;
    cv::Mat src = createSourceImage(src_height, src_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    float scale = int_scale / 10.f;

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        cv::Rect roi(left, top, width, height);
        cv::Mat croppedImage = src(roi);
        croppedImage.copyTo(dst);
        dst = dst * scale;
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            cv::Rect roi(left, top, width, height);
            cv::Mat croppedImage = src(roi);
            croppedImage.copyTo(dst);
            dst = dst * scale;
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_OPENCV_TYPE_FUNCTIONS(type, top, left, scale)                  \
    BENCHMARK_TEMPLATE(BM_Crop_opencv_aarch64, type, c1, top, left, scale) \
        ->Args({640, 480})                                                 \
        ->UseManualTime()                                                  \
        ->Iterations(10);                                                  \
    BENCHMARK_TEMPLATE(BM_Crop_opencv_aarch64, type, c3, top, left, scale) \
        ->Args({640, 480})                                                 \
        ->UseManualTime()                                                  \
        ->Iterations(10);                                                  \
    BENCHMARK_TEMPLATE(BM_Crop_opencv_aarch64, type, c4, top, left, scale) \
        ->Args({640, 480})                                                 \
        ->UseManualTime()                                                  \
        ->Iterations(10);                                                  \
    BENCHMARK_TEMPLATE(BM_Crop_opencv_aarch64, type, c1, top, left, scale) \
        ->Args({1920, 1080})                                               \
        ->UseManualTime()                                                  \
        ->Iterations(10);                                                  \
    BENCHMARK_TEMPLATE(BM_Crop_opencv_aarch64, type, c3, top, left, scale) \
        ->Args({1920, 1080})                                               \
        ->UseManualTime()                                                  \
        ->Iterations(10);                                                  \
    BENCHMARK_TEMPLATE(BM_Crop_opencv_aarch64, type, c4, top, left, scale) \
        ->Args({1920, 1080})                                               \
        ->UseManualTime()                                                  \
        ->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, 10)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, 15)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, 16, 16, 10)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, 16, 16, 15)

#endif //! PPLCV_BENCHMARK_OPENCV
