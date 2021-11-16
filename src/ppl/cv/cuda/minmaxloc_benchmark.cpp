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

#include "ppl/cv/cuda/minmaxloc.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

template <typename T>
void BM_MinMaxLoc_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  cv::cuda::GpuMat gpu_src(src);

  T minVal;
  T maxVal;
  int minIdxX;
  int minIdxY;
  int maxIdxX;
  int maxIdxY;

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    MinMaxLoc<T>(0, gpu_src.rows, gpu_src.cols,
                 gpu_src.step / sizeof(T), (T*)gpu_src.data,
                 &minVal, &maxVal, &minIdxX, &minIdxY, &maxIdxX, &maxIdxY);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      MinMaxLoc<T>(0, gpu_src.rows, gpu_src.cols,
                  gpu_src.step / sizeof(T), (T*)gpu_src.data,
                  &minVal, &maxVal, &minIdxX, &minIdxY, &maxIdxX, &maxIdxY);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T>
void BM_MinMaxLoc_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  cv::cuda::GpuMat gpu_src(src);

  double minVal, maxVal;
  cv::Point minLoc, maxLoc;

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::minMaxLoc(gpu_src, &minVal, &maxVal, &minLoc, &maxLoc);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::minMaxLoc(gpu_src, &minVal, &maxVal, &minLoc, &maxLoc);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T>
void BM_MinMaxLoc_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          1));

  double minVal, maxVal;
  cv::Point minLoc, maxLoc;

  for (auto _ : state) {
    cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(width, height)                                          \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, uchar)->Args({width, height})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, uchar)->Args({width, height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, float)->Args({width, height})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, float)->Args({width, height})->      \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(320, 240)
// RUN_BENCHMARK0(640, 480)
// RUN_BENCHMARK0(1280, 720)
// RUN_BENCHMARK0(1920, 1080)

#define RUN_BENCHMARK1(width, height)                                          \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, uchar)->Args({width, height});\
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, uchar)->Args({width, height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, float)->Args({width, height});\
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, float)->Args({width, height})->      \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(320, 240)
// RUN_BENCHMARK1(640, 480)
// RUN_BENCHMARK1(1280, 720)
// RUN_BENCHMARK1(1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type)                                        \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, type)->Args({320, 240})->         \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, type)->Args({640, 480})->         \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, type)->Args({1280, 720})->        \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, type)->Args({1920, 1080})->       \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(type)                                        \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type)->Args({320, 240})->            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type)->Args({640, 480})->            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type)->Args({1280, 720})->           \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type)->Args({1920, 1080})->          \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar)
RUN_OPENCV_TYPE_FUNCTIONS(float)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar)
RUN_PPL_CV_TYPE_FUNCTIONS(float)
