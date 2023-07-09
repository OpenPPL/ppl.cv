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

#include "ppl/cv/arm/warpperspective.h"

#include <chrono>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

cv::Mat getRandomPerspectiveMat(cv::Mat src, cv::Mat dst)
{
    constexpr int offset = 16;
    cv::Point2f srcPoints[4];
    cv::Point2f dstPoints[4];

    srcPoints[0] = cv::Point2f(0 + offset, 0 + offset);
    srcPoints[1] = cv::Point2f(src.rows - 1 - offset, 0);
    srcPoints[2] = cv::Point2f(src.rows - 1 - offset, src.cols - 1 - offset);
    srcPoints[3] = cv::Point2f(0 + offset, src.cols - 1 - offset);

    dstPoints[0] = cv::Point2f(0 + offset, 0 + offset);
    dstPoints[1] = cv::Point2f(dst.rows - 1 - offset, 0);
    dstPoints[2] = cv::Point2f(dst.rows - 1 - offset, dst.cols - 1 - offset);
    dstPoints[3] = cv::Point2f(0 + offset, dst.cols - 1 - offset);

    // inverse map
    cv::Mat M = getPerspectiveTransform(dstPoints, srcPoints);
    return M;
}

template <typename T, int channels, ppl::cv::InterpolationType inter_type, ppl::cv::BorderType border_type>
void BM_WarpPerspective_ppl_aarch64(benchmark::State &state)
{
    int src_width = state.range(0);
    int src_height = state.range(1);
    int dst_width = state.range(2);
    int dst_height = state.range(3);
    cv::Mat src = createSourceImage(src_height, src_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(dst_height, dst_width, src.type());
    cv::Mat M = getRandomPerspectiveMat(src, dst);

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        ppl::cv::arm::WarpPerspective<T, channels>(src.rows,
                                                   src.cols,
                                                   src.step / sizeof(T),
                                                   (T *)src.data,
                                                   dst_height,
                                                   dst_width,
                                                   dst.step / sizeof(T),
                                                   (T *)dst.data,
                                                   (double *)M.data,
                                                   inter_type,
                                                   border_type);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            ppl::cv::arm::WarpPerspective<T, channels>(src.rows,
                                                       src.cols,
                                                       src.step / sizeof(T),
                                                       (T *)src.data,
                                                       dst_height,
                                                       dst_width,
                                                       dst.step / sizeof(T),
                                                       (T *)dst.data,
                                                       (double *)M.data,
                                                       inter_type,
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

#define RUN_PPL_CV_TYPE_FUNCTIONS(inter_type, border_type)                                 \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, uchar, c1, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                       \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, uchar, c1, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                      \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, uchar, c3, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                       \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, uchar, c3, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                      \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, uchar, c4, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                       \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, uchar, c4, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                      \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, float, c1, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                       \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, float, c1, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                      \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, float, c3, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                       \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, float, c3, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                      \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, float, c4, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                       \
        ->UseManualTime()                                                                  \
        ->Iterations(10);                                                                  \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_ppl_aarch64, float, c4, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                      \
        ->UseManualTime()                                                                  \
        ->Iterations(10);

RUN_PPL_CV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT)
RUN_PPL_CV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT)

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int channels, ppl::cv::InterpolationType inter_type, ppl::cv::BorderType border_type>
void BM_WarpPerspective_opencv_aarch64(benchmark::State &state)
{
    int src_width = state.range(0);
    int src_height = state.range(1);
    int dst_width = state.range(2);
    int dst_height = state.range(3);
    cv::Mat src = createSourceImage(src_height, src_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(dst_height, dst_width, src.type());
    cv::Mat M = getRandomPerspectiveMat(src, dst);

    int cv_iterpolation;
    if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
        cv_iterpolation = cv::INTER_LINEAR;
    } else {
        cv_iterpolation = cv::INTER_NEAREST;
    }

    cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        cv_border = cv::BORDER_CONSTANT;
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        cv_border = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        cv_border = cv::BORDER_TRANSPARENT;
    } else {
    }

    int warmup_iters = 5;
    int perf_iters = 50;

    // Warm up the CPU.
    for (int i = 0; i < warmup_iters; i++) {
        cv::warpPerspective(
            src, dst, M, cv::Size(dst_width, dst_height), cv_iterpolation | cv::WARP_INVERSE_MAP, cv_border);
    }

    for (auto _ : state) {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iters; i++) {
            cv::warpPerspective(
                src, dst, M, cv::Size(dst_width, dst_height), cv_iterpolation | cv::WARP_INVERSE_MAP, cv_border);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto duration = time_end - time_start;
        auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        double time = overall_time * 1.0 / perf_iters;
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_OPENCV_TYPE_FUNCTIONS(inter_type, border_type)                                    \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, uchar, c1, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                          \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, uchar, c1, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                         \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, uchar, c3, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                          \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, uchar, c3, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                         \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, uchar, c4, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                          \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, uchar, c4, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                         \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, float, c1, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                          \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, float, c1, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                         \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, float, c3, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                          \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, float, c3, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                         \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, float, c4, inter_type, border_type) \
        ->Args({640, 480, 320, 240})                                                          \
        ->UseManualTime()                                                                     \
        ->Iterations(10);                                                                     \
    BENCHMARK_TEMPLATE(BM_WarpPerspective_opencv_aarch64, float, c4, inter_type, border_type) \
        ->Args({640, 480, 1280, 960})                                                         \
        ->UseManualTime()                                                                     \
        ->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT)
RUN_OPENCV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT)

#endif //! PPLCV_BENCHMARK_OPENCV
