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

#include "ppl/cv/arm/filter2d.h"

#include <chrono>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst, int channels, int ksize, ppl::cv::BorderType border_type>
void BM_Filter2D_ppl_aarch64(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
    cv::Mat kernel = createSourceImage(1, ksize * ksize, CV_MAKETYPE(cv::DataType<float>::depth, 1));
    cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

    float delta = 0.f;

    int warmup_iters = 3;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        ppl::cv::arm::Filter2D<Tsrc, channels>(src.rows,
                                               src.cols,
                                               src.step / sizeof(Tsrc),
                                               (Tsrc *)src.data,
                                               ksize,
                                               (float *)kernel.data,
                                               dst.step / sizeof(Tdst),
                                               (Tdst *)dst.data,
                                               delta,
                                               border_type);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            ppl::cv::arm::Filter2D<Tsrc, channels>(src.rows,
                                                   src.cols,
                                                   src.step / sizeof(Tsrc),
                                                   (Tsrc *)src.data,
                                                   ksize,
                                                   (float *)kernel.data,
                                                   dst.step / sizeof(Tdst),
                                                   (Tdst *)dst.data,
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

#define RUN_PPL_CV_TYPE_FUNCTIONS(src_type, dst_type, ksize, border_type)                   \
    BENCHMARK_TEMPLATE(BM_Filter2D_ppl_aarch64, src_type, dst_type, c1, ksize, border_type) \
        ->Args({640, 480})                                                                  \
        ->UseManualTime()                                                                   \
        ->Iterations(10);                                                                   \
    BENCHMARK_TEMPLATE(BM_Filter2D_ppl_aarch64, src_type, dst_type, c3, ksize, border_type) \
        ->Args({640, 480})                                                                  \
        ->UseManualTime()                                                                   \
        ->Iterations(10);                                                                   \
    BENCHMARK_TEMPLATE(BM_Filter2D_ppl_aarch64, src_type, dst_type, c4, ksize, border_type) \
        ->Args({640, 480})                                                                  \
        ->UseManualTime()                                                                   \
        ->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 3, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 3, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 3, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 5, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 5, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 5, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 5, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 5, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 5, ppl::cv::BORDER_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 17, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 17, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uint8_t, uint8_t, 17, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 17, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 17, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 17, ppl::cv::BORDER_REFLECT_101)

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename Tsrc, typename Tdst, int channels, int ksize, ppl::cv::BorderType border_type>
void BM_Filter2D_opencv_aarch64(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
    cv::Mat kernel = createSourceImage(1, ksize * ksize, CV_MAKETYPE(cv::DataType<float>::depth, 1));
    cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

    cv::BorderTypes border = cv::BORDER_DEFAULT;
    if (border_type == ppl::cv::BORDER_REPLICATE) {
        border = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_REFLECT) {
        border = cv::BORDER_REFLECT;
    } else if (border_type == ppl::cv::BORDER_REFLECT_101) {
        border = cv::BORDER_REFLECT_101;
    } else {
    }

    float delta = 0.f;

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        cv::filter2D(src, dst, dst.depth(), kernel, cv::Point(-1, -1), delta, border);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            cv::filter2D(src, dst, dst.depth(), kernel, cv::Point(-1, -1), delta, border);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_OPENCV_TYPE_FUNCTIONS(src_type, dst_type, ksize, border_type)                      \
    BENCHMARK_TEMPLATE(BM_Filter2D_opencv_aarch64, src_type, dst_type, c1, ksize, border_type) \
        ->Args({640, 480})                                                                     \
        ->UseManualTime()                                                                      \
        ->Iterations(10);                                                                      \
    BENCHMARK_TEMPLATE(BM_Filter2D_opencv_aarch64, src_type, dst_type, c3, ksize, border_type) \
        ->Args({640, 480})                                                                     \
        ->UseManualTime()                                                                      \
        ->Iterations(10);                                                                      \
    BENCHMARK_TEMPLATE(BM_Filter2D_opencv_aarch64, src_type, dst_type, c4, ksize, border_type) \
        ->Args({640, 480})                                                                     \
        ->UseManualTime()                                                                      \
        ->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 3, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 3, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 3, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT_101)

RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 5, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 5, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 5, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 5, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 5, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 5, ppl::cv::BORDER_REFLECT_101)

RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 17, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 17, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uint8_t, uint8_t, 17, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 17, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 17, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 17, ppl::cv::BORDER_REFLECT_101)

#endif //! PPLCV_BENCHMARK_OPENCV
