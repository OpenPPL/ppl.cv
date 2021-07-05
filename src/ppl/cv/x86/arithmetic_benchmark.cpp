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

#include "ppl/cv/x86/arithmetic.h"
#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include <memory>
#include "ppl/cv/debug.h"

namespace {

enum MATH_OP {ADD, SUB, MUL, DIV, MLA, MLS};

template<typename T, int32_t nc, MATH_OP op>
void BM_MATH_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 1, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 1, 255);
    if (op == ADD) {
        for (auto _ : state) {
            ppl::cv::x86::Add<T, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());
        }
    } else if (op == MUL) {
        for (auto _ : state) {
            ppl::cv::x86::Mul<T, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());
        }
    } else if (op == DIV) {
        for (auto _ : state) {
            ppl::cv::x86::Div<T, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());
        }
    } else if (op == SUB) {
        for (auto _ : state) {
            std::unique_ptr<T[]> scl(new T[nc]);
            ppl::cv::debug::randomFill<T>(scl.get(), nc, 1, 255);
            ppl::cv::x86::Subtract<T, nc>(height, width, width * nc, src0.get(), scl.get(), width * nc, dst.get());
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 1, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 3, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 4, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 1, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 3, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 4, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 1, DIV)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 3, DIV)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 4, DIV)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 1, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 3, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, float, 4, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 1, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 3, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 4, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 1, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 3, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 4, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 1, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 3, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_ppl_x86, uint8_t, 4, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t nc, MATH_OP op>
static void BM_MATH_opencv_x86(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 1, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 1, 255);
    cv::Mat src0Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src0.get());
    cv::Mat src1Mat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src1.get());
    cv::Mat resultMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst.get());
    if (op == ADD) {
        for (auto _ : state) {
            cv::add(src0Mat, src1Mat, resultMat);
        }
    } else if (op == MUL) {
        for (auto _ : state) {
            cv::multiply(src0Mat, src1Mat, resultMat);
        }
    } else if (op == DIV) {
        for (auto _ : state) {
            cv::divide(src0Mat, src1Mat, resultMat);
        }
    } else if (op == SUB) {
        for ( auto _ : state) {
            std::unique_ptr<T[]> inScalar(new T[nc]);
            ppl::cv::debug::randomFill<T>(inScalar.get(), nc, 1, 255);
            cv::Scalar scl;
            for (int32_t i = 0; i < nc; i++)
            {
                scl[i] = inScalar.get()[i];
            }
            cv::subtract(src0Mat, scl, resultMat);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 1, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 3, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 4, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 1, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 3, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 4, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 1, DIV)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 3, DIV)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 4, DIV)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 1, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 3, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, float, 4, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 1, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 3, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 4, ADD)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 1, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 3, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 4, SUB)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 1, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 3, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_MATH_opencv_x86, uint8_t, 4, MUL)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
}
