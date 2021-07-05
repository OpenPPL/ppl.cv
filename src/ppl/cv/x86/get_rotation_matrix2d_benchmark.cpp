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

#include "ppl/cv/x86/get_rotation_matrix2d.h"
#include "ppl/cv/debug.h"
#include <memory>
#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>

namespace {

void BM_GetRotationMatrix2D_ppl_x86(benchmark::State &state) {
    double angle = state.range(0) / 2;
    double scale = state.range(1) / 2;
    std::unique_ptr<double[]> dst(new double[6]);
    for (auto _ : state) {
        ppl::cv::x86::GetRotationMatrix2D(320, 360, angle, scale, dst.get());
    
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK(BM_GetRotationMatrix2D_ppl_x86)->Args({30, 1})->Args({45, 1})->Args({90, 1})->Args({-30, 1})->Args({-45, 1})->Args({-90, 1});

#ifdef PPLCV_BENCHMARK_OPENCV
void BM_GetRotationMatrix2D_opencv_x86(benchmark::State &state) {
    double angle = state.range(0) / 2;
    double scale = state.range(1) / 2;
    cv::Point2f center;
    center.x = 320;
    center.y = 360;
    for (auto _ : state) {
        cv::getRotationMatrix2D(center, angle, scale);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK(BM_GetRotationMatrix2D_opencv_x86)->Args({30, 1})->Args({45, 1})->Args({90, 1})->Args({-30, 1})->Args({-45, 1})->Args({-90, 1});
#endif //! PPLCV_BENCHMARK_OPENCV
}
