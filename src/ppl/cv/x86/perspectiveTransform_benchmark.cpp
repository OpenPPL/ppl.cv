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
#include "ppl/cv/x86/perspectivetransform.h"
#include "ppl/cv/debug.h"

namespace {
template<typename T, int32_t ncSrc, int32_t ncDst>
class PerspectiveTransformBenchmark {
public:
    int32_t height;
    int32_t width;
    T *inData;
    T *outData;
    int32_t mHeight;
    int32_t mWidth;
    double* mData;

    PerspectiveTransformBenchmark(int32_t height, int32_t width)
        : height(height)
        , width(width)
        , mHeight(ncDst + 1)
        , mWidth(ncSrc + 1)
    {
        inData = NULL;
        outData = NULL;

        inData = (T*)malloc(height * width * ncSrc * sizeof(T));
        outData = (T*)malloc(height * width * ncDst * sizeof(T));
        mData = (double*)malloc(mHeight * mWidth * sizeof(double));
        ppl::cv::debug::randomFill<double>(mData, mHeight * mWidth, 0, 255);
    }
    
    void apply() {
        //pplcv
        int32_t iStride = width * ncSrc;
        int32_t oStride = width * ncDst;
        ppl::cv::x86::PerspectiveTransform<T, ncSrc, ncDst>(height, width, iStride, inData, oStride, outData, (float*)mData);
    }
    void apply_opencv() {
        cv::Mat iMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, ncSrc), inData);
        cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, ncDst), outData);
        cv::Mat mMat(mHeight, mWidth, CV_MAKETYPE(cv::DataType<double>::depth, 1), mData);
        cv::perspectiveTransform(iMat, oMat, mMat);
    }

    ~PerspectiveTransformBenchmark() {
        free(inData);
        free(outData);
        free(mData);
    }
};
}

using namespace ppl::cv::debug;

template<typename T, int32_t ncSrc, int32_t ncDst>
static void BM_PerspectiveTransform_ppl_x86(benchmark::State &state) {
    PerspectiveTransformBenchmark<T, ncSrc, ncDst> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply();
    }

    state.SetItemsProcessed(state.iterations());
}

template<typename T, int32_t ncSrc, int32_t ncDst>
static void BM_PerspectiveTransform_opencv_x86(benchmark::State &state) {
    PerspectiveTransformBenchmark<T, ncSrc, ncDst> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply_opencv();
    }

    state.SetItemsProcessed(state.iterations());
}

//pplcv
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_x86, float, c2, c2)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_x86, float, c2, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_x86, float, c3, c2)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_x86, float, c3, c3)->Args({320, 240})->Args({640, 480});

//opencv
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86, float, c2, c2)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86, float, c2, c3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86, float, c3, c2)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86, float, c3, c3)->Args({320, 240})->Args({640, 480});
