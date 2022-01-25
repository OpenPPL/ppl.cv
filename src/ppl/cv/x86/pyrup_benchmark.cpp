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
#include "ppl/cv/x86/pyrup.h"
#include "ppl/cv/debug.h"

namespace {
template<typename T, int32_t channels>
class PyrUpBenchmark {
public:
    int32_t height;
    int32_t width;
    T *inData;
    T *outData;

    PyrUpBenchmark(int32_t height, int32_t width)
        : height(height)
        , width(width)
    {
        inData = NULL;
        outData = NULL;

        inData = (T*)malloc(height * width * channels * sizeof(T));
        memset(inData, 0, height * width * channels * sizeof(T));
        outData = (T*)malloc(height*2 * width*2 * channels * sizeof(T));

        if (!inData || !outData) {
            if (inData) {
                free(inData);
            }
            if (outData) {
                free(outData);
            }
            return;
        }
    }
    
    void apply() {
        ppl::cv::x86::PyrUp<T, channels>(height, width, width * channels,
            inData, width*2*channels, outData, ppl::cv::BORDER_DEFAULT);
    }
     void apply_opencv() {
        cv::Mat iMat(height, width, T2CvType<T, channels>::type, inData);
        cv::Mat oMat(height, width, T2CvType<T, channels>::type, outData);
        int32_t oheight = height * 2;
        int32_t owidth = width * 2;
        cv::pyrUp(iMat, oMat, cv::Size(owidth, oheight));
    }

    ~PyrUpBenchmark() {
        free(inData);
        free(outData);
    }
};
}

using namespace ppl::cv::debug;

template<typename T, int32_t channels>
static void BM_PyrUp_ppl_x86(benchmark::State &state) {
    PyrUpBenchmark<T, channels> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * channels);
}

template<typename T, int32_t channels>
static void BM_PyrUp_opencv_x86(benchmark::State &state) {
    PyrUpBenchmark<T, channels> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply_opencv();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * channels);
}
//pplcv
BENCHMARK_TEMPLATE(BM_PyrUp_ppl_x86, float, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_PyrUp_ppl_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480});

//opencv
BENCHMARK_TEMPLATE(BM_PyrUp_opencv_x86, float, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_PyrUp_opencv_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PyrUp_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480});
