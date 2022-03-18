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

#include "ppl/cv/cuda/integral.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst>
void BM_Integral_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          1));
  cv::Mat dst(height + 1, width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Integral<Tsrc, Tdst, 1>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data, gpu_dst.rows,
        gpu_dst.cols, gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::Integral<Tsrc, Tdst, 1>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data, gpu_dst.rows,
          gpu_dst.cols, gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename Tsrc, typename Tdst>
void BM_Integral_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          1));
  cv::Mat dst(height + 1, width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::integral(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::integral(gpu_src, gpu_dst);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename Tsrc, typename Tdst>
void BM_Integral_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          1));
  cv::Mat dst(height + 1, width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, 1));

  for (auto _ : state) {
    cv::integral(src, dst, dst.depth());
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(width, height)                                          \
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86_cuda, uchar, int)->                  \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Integral_ppl_cuda, uchar, int)->Args({width, height})->  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86_cuda, float, float)->                \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Integral_ppl_cuda, float, float)->Args({width, height})->\
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(320, 240)
// RUN_BENCHMARK0(640, 480)
// RUN_BENCHMARK0(1280, 720)
// RUN_BENCHMARK0(1920, 1080)

#define RUN_BENCHMARK1(width, height)                                          \
BENCHMARK_TEMPLATE(BM_Integral_opencv_cuda, uchar, int)->                      \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Integral_ppl_cuda, uchar, int)->Args({width, height})->  \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(320, 240)
// RUN_BENCHMARK1(640, 480)
// RUN_BENCHMARK1(1280, 720)
// RUN_BENCHMARK1(1920, 1080)

#define RUN_OPENCV_X86_FUNCTIONS(tsrc, tdst)                                   \
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86_cuda, tsrc, tdst)->Args({320, 240}); \
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86_cuda, tsrc, tdst)->Args({640, 480}); \
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86_cuda, tsrc, tdst)->Args({1280, 720});\
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86_cuda, tsrc, tdst)->Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(tsrc, tdst)                                  \
BENCHMARK_TEMPLATE(BM_Integral_ppl_cuda, tsrc, tdst)->Args({320, 240})->       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Integral_ppl_cuda, tsrc, tdst)->Args({640, 480})->       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Integral_ppl_cuda, tsrc, tdst)->Args({1280, 720})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Integral_ppl_cuda, tsrc, tdst)->Args({1920, 1080})->     \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_X86_FUNCTIONS(uchar, int)
RUN_OPENCV_X86_FUNCTIONS(float, float)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, int)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float)
