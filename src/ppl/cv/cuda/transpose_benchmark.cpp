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

#include "ppl/cv/cuda/transpose.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

template <typename T, int channels>
void BM_Transpose_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(width, height, src.type());

  int iterations = 3000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    Transpose<T, channels>(0, gpu_src.rows, gpu_src.cols,
                           gpu_src.step / sizeof(T), (T*)gpu_src.data,
                           gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      Transpose<T, channels>(0, gpu_src.rows, gpu_src.cols,
                             gpu_src.step / sizeof(T), (T*)gpu_src.data,
                             gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels>
void BM_Transpose_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(width, height, src.type());

  int iterations = 3000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::transpose(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::transpose(gpu_src, gpu_dst);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels>
void BM_Transpose_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(width, height, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  for (auto _ : state) {
    cv::transpose(src, dst);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, width, height)                                 \
BENCHMARK_TEMPLATE(BM_Transpose_opencv_x86_cuda, uchar, channels)->            \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Transpose_ppl_cuda, uchar, channels)->                   \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Transpose_opencv_x86_cuda, float, channels)->            \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Transpose_ppl_cuda, float, channels)->                   \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK(c1, 640, 480)
// RUN_BENCHMARK(c3, 640, 480)
// RUN_BENCHMARK(c4, 640, 480)

// RUN_BENCHMARK(c1, 1920, 1080)
// RUN_BENCHMARK(c3, 1920, 1080)
// RUN_BENCHMARK(c4, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type)                                        \
BENCHMARK_TEMPLATE(BM_Transpose_opencv_x86_cuda, type, c1)->Args({640, 480});  \
BENCHMARK_TEMPLATE(BM_Transpose_opencv_x86_cuda, type, c3)->Args({640, 480});  \
BENCHMARK_TEMPLATE(BM_Transpose_opencv_x86_cuda, type, c4)->Args({640, 480});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type)                                        \
BENCHMARK_TEMPLATE(BM_Transpose_ppl_cuda, type, c1)->Args({640, 480})->        \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Transpose_ppl_cuda, type, c3)->Args({640, 480})->        \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Transpose_ppl_cuda, type, c4)->Args({640, 480})->        \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar)
RUN_OPENCV_TYPE_FUNCTIONS(float)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar)
RUN_PPL_CV_TYPE_FUNCTIONS(float)
