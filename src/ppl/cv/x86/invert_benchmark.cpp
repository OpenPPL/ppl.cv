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
#include "ppl/cv/x86/invert.h"
#include "ppl/cv/debug.h"

namespace {
template<typename T, int32_t c>
class vecOperatorBenchmark {
public:
    int32_t device = 0;
    T* input = nullptr;
    T* output = nullptr;
    int32_t height;
    int32_t width;
    int32_t inWidthStride;
    int32_t outWidthStride;
    int32_t method;
    vecOperatorBenchmark(int32_t height, int32_t width, int32_t method)
        : height(height)
        , width(width)
        , inWidthStride(width * c)
        , outWidthStride(width * c)
        , method(method)
    {
        input  = (T*)malloc(height * width * c * sizeof(T));
        output = (T*)malloc(height * width * c * sizeof(T));
        ppl::cv::debug::randomFill<T>(input, width * height * c, -255, 255);

        if(method == ppl::cv::DECOMP_CHOLESKY)
        for(int32_t i = 0; i < height; i++){
            for(int32_t j = 0; j < width - i; j++){
                input[i * width + width - 1 - j] = input[(width - 1 - j) * width + i];
            }
        }
    }

    void apply_opencv() {
        cv::Mat src_opencv(height, width,
            CV_MAKETYPE(cv::DataType<T>::depth, c), input);
        cv::Mat dst_opencv(height, width,
                CV_MAKETYPE(cv::DataType<T>::depth, c), output);
        switch(method) {
            case ppl::cv::DECOMP_CHOLESKY:
                //opencv
                cv::invert(src_opencv, dst_opencv, 2/*DECOMP_CHOLESKY*/);
                break;
            default:
                return;
        }
    }
    void apply() {
        //ppl.cv
        ppl::cv::x86::Invert<T>(height, width, inWidthStride, input, outWidthStride, output, ppl::cv::DECOMP_CHOLESKY);
    }

    ~vecOperatorBenchmark() {
        free(this->input);
        free(this->output);
    }
};
}

using namespace ppl::cv::debug;
template<typename T, int32_t c>
static void BM_Invert_ppl_x86(benchmark::State &state) {
    vecOperatorBenchmark<T, c> bm(state.range(1), state.range(0), state.range(2));
    for (auto _: state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * c);
}

template<typename T, int32_t c>
static void BM_Invert_opencv_x86(benchmark::State &state) {
    vecOperatorBenchmark<T, c> bm(state.range(1), state.range(0), state.range(2));
    for (auto _: state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * c);
}

//ppl.cv
BENCHMARK_TEMPLATE(BM_Invert_ppl_x86, float, c1)->Args({320, 320, ppl::cv::DECOMP_CHOLESKY})->Args({640, 640, ppl::cv::DECOMP_CHOLESKY});
BENCHMARK_TEMPLATE(BM_Invert_ppl_x86, double, c1)->Args({320, 320, ppl::cv::DECOMP_CHOLESKY})->Args({640, 640, ppl::cv::DECOMP_CHOLESKY});

//opencv
BENCHMARK_TEMPLATE(BM_Invert_opencv_x86, float, c1)->Args({320, 320, ppl::cv::DECOMP_CHOLESKY})->Args({640, 640, ppl::cv::DECOMP_CHOLESKY});
BENCHMARK_TEMPLATE(BM_Invert_opencv_x86, double, c1)->Args({320, 320, ppl::cv::DECOMP_CHOLESKY})->Args({640, 640, ppl::cv::DECOMP_CHOLESKY});
