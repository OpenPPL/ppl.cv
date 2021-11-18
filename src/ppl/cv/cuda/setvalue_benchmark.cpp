/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "ppl/cv/cuda/setvalue.h"
#include "ppl/cv/cuda/zeros.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility.hpp"
#include "infrastructure.hpp"

using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

enum Functions {
  OnesFunc,
  ZerosFunc,
};

template <typename T, int channels, Functions function>
void BM_SetValue_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat dst;
  dst = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 3000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    Zeros<T, channels>(0, gpu_dst.rows, gpu_dst.cols, gpu_dst.step / sizeof(T),
                       (T*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (function == OnesFunc) {
        Ones<T, channels>(0, gpu_dst.rows, gpu_dst.cols,
                            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else {
        Zeros<T, channels>(0, gpu_dst.rows, gpu_dst.cols,
                              gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, Functions function>
void BM_SetValue_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat dst;
  dst = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_dst(dst);

  cv::Scalar scalar_one(1, 1, 1, 1);
  cv::Scalar scalar_zero(0, 0, 0, 0);

  int iterations = 3000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    gpu_dst.setTo(scalar_zero);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (function == OnesFunc) {
        gpu_dst.setTo(scalar_one);
      }
      else {
        gpu_dst.setTo(scalar_zero);
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, Functions function>
void BM_SetValue_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat dst;
  dst = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));

  for (auto _ : state) {
    if (function == OnesFunc) {
      dst = cv::Mat::ones(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
    }
    else {
      dst = cv::Mat::zeros(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(function, width, height)                                \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c1, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c3, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c4, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c1, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c3, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c4, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_BENCHMARK0(OnesFunc, 320, 240)
RUN_BENCHMARK0(OnesFunc, 640, 480)
RUN_BENCHMARK0(OnesFunc, 1280, 720)
RUN_BENCHMARK0(OnesFunc, 1920, 1080)

// #define RUN_BENCHMARK1(function, width, height)                                \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c1, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, function)->                \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c3, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, function)->                \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c4, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, function)->                \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c1, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, function)->                \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c3, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, function)->                \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c4, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, function)->                \
//                    Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(OnesFunc, 320, 240)
// RUN_BENCHMARK1(OnesFunc, 640, 480)
// RUN_BENCHMARK1(OnesFunc, 1280, 720)
// RUN_BENCHMARK1(OnesFunc, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(function)                                    \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c1, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c3, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c4, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c1, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c3, function)->         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c4, function)->         \
                   Args({width, height});

// #define RUN_OPENCV_TYPE_FUNCTIONS(function)                                    \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c1, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c3, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c4, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c1, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c3, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);     \
// BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c4, function)->             \
//                    Args({width, height})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(function)                                    \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, function)->                \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_OPENCV_TYPE_FUNCTIONS(OnesFunc)
// RUN_PPL_CV_TYPE_FUNCTIONS(OnesFunc)
