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
#include "ppl/cv/debug.h"
#include <memory>
#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>

namespace {

template <typename T, int32_t nc>
void BM_Mla_ppl_aarch64(benchmark::State &state)
{
    int32_t width  = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 0, 1);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 0, 1);
    for (auto _ : state) {
        ppl::cv::arm::Mla<T, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Mla_ppl_aarch64, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Mla_ppl_aarch64, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Mla_ppl_aarch64, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

} // namespace
