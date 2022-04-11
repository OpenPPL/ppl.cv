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
#include "ppl/cv/riscv/flip.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template <typename T, int32_t nc, int32_t flip_mode>
void BM_Flip_ppl_riscv(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);

    for (auto _ : state) {
        ppl::cv::riscv::Flip<T, nc>(height, width, width, src.get(), width * nc, dst.get(), flip_mode);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c1, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c1, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c1, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c3, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c3, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c3, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c4, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c4, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, uint8_t, c4, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c1, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c1, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c1, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c3, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c3, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c3, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c4, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c4, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_ppl_riscv, float, c4, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int32_t nc, int32_t flip_mode>
static void BM_Flip_opencv_riscv(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    std::unique_ptr<T[]> dst_opencv(new T[width * height * nc]);
    cv::Mat iMat(height, width, T2CvType<T, nc>::type, src.get());
    cv::Mat oMat(height, width, T2CvType<T, nc>::type, dst_opencv.get());
    for (auto _ : state) {
        cv::flip(iMat, oMat, flip_mode);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c1, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c1, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c1, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c3, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c3, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c3, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c4, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c4, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, uint8_t, c4, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c1, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c1, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c1, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c3, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c3, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c3, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c4, -1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c4, 0)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Flip_opencv_riscv, float, c4, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
} // namespace
