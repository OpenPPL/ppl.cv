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

#include "ppl/cv/cuda/pyrdown.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, ppl::cv::BorderType border_type>
void BM_PyrDown_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst((height + 1) / 2, (width + 1) / 2,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::PyrDown<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::PyrDown<T, channels>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
          (T*)gpu_dst.data, border_type);
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

template <typename T, int channels, ppl::cv::BorderType border_type>
void BM_PyrDown_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst((height + 1) / 2, (width + 1) / 2,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::pyrDown(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::pyrDown(gpu_src, gpu_dst);
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

template <typename T, int channels, ppl::cv::BorderType border_type>
void BM_PyrDown_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst((height + 1) / 2, (width + 1) / 2,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));

  cv::BorderTypes border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  for (auto _ : state) {
    cv::pyrDown(src, dst, cv::Size((width + 1) / 2, (height + 1) / 2), border);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(type, border_type, width, height)                       \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c1, border_type)->        \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c1, border_type)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c3, border_type)->        \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c3, border_type)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c4, border_type)->        \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c4, border_type)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(uchar, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(float, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, ppl::cv::BORDER_REFLECT_101, 640, 480)

// RUN_BENCHMARK0(uchar, ppl::cv::BORDER_REPLICATE, 1920, 1080)
// RUN_BENCHMARK0(uchar, ppl::cv::BORDER_REFLECT, 1920, 1080)
// RUN_BENCHMARK0(uchar, ppl::cv::BORDER_REFLECT_101, 1920, 1080)
// RUN_BENCHMARK0(float, ppl::cv::BORDER_REPLICATE, 1920, 1080)
// RUN_BENCHMARK0(float, ppl::cv::BORDER_REFLECT, 1920, 1080)
// RUN_BENCHMARK0(float, ppl::cv::BORDER_REFLECT_101, 1920, 1080)

#define RUN_BENCHMARK1(type, border_type, width, height)                       \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_cuda, type, c1, border_type)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c1, border_type)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_cuda, type, c3, border_type)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c3, border_type)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_cuda, type, c4, border_type)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c4, border_type)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(uchar, ppl::cv::BORDER_REFLECT_101, 320, 240)
// RUN_BENCHMARK1(uchar, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, ppl::cv::BORDER_REFLECT_101, 1280, 720)
// RUN_BENCHMARK1(uchar, ppl::cv::BORDER_REFLECT_101, 1920, 1080)
// RUN_BENCHMARK1(float, ppl::cv::BORDER_REFLECT_101, 320, 240)
// RUN_BENCHMARK1(float, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(float, ppl::cv::BORDER_REFLECT_101, 1280, 720)
// RUN_BENCHMARK1(float, ppl::cv::BORDER_REFLECT_101, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, border_type)                           \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c1, border_type)->        \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c3, border_type)->        \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c4, border_type)->        \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c1, border_type)->        \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c3, border_type)->        \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_PyrDown_opencv_x86_cuda, type, c4, border_type)->        \
                   Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, border_type)                           \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c1, border_type)->               \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c3, border_type)->               \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c4, border_type)->               \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c1, border_type)->               \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c3, border_type)->               \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_PyrDown_ppl_cuda, type, c4, border_type)->               \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::BORDER_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::BORDER_REFLECT_101)
