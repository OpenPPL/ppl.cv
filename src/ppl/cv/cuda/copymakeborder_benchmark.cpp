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

#include "ppl/cv/cuda/copymakeborder.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, int top, int left,
          ppl::cv::BorderType border_type>
void BM_CopyMakeBorder_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(src.rows + top * 2, src.cols + left * 2,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int bottom = top;
  int right  = left;

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::CopyMakeBorder<T, channels>(0, src.rows, src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, top, bottom, left, right, border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::CopyMakeBorder<T, channels>(0, src.rows, src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
          (T*)gpu_dst.data, top, bottom, left, right, border_type);
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

template <typename T, int channels, int top, int left,
          ppl::cv::BorderType border_type>
void BM_CopyMakeBorder_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(src.rows + top * 2, src.cols + left * 2,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int bottom = top;
  int right  = left;

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_WRAP) {
    cv_border = cv::BORDER_WRAP;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::copyMakeBorder(gpu_src, gpu_dst, top, bottom, left, right,
                             cv_border);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::copyMakeBorder(gpu_src, gpu_dst, top, bottom, left, right,
                               cv_border);
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

template <typename T, int channels, int top, int left,
          ppl::cv::BorderType border_type>
void BM_CopyMakeBorder_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(src.rows + top * 2, src.cols + left * 2,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int bottom = top;
  int right  = left;

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_WRAP) {
    cv_border = cv::BORDER_WRAP;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  for (auto _ : state) {
    cv::copyMakeBorder(src, dst, top, bottom, left, right, cv_border);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, top, left, border_type, width, height)        \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_x86_cuda, uchar, channels, top,    \
                   left, border_type)->Args({width, height});                  \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, uchar, channels, top, left,     \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_x86_cuda, float, channels, top,    \
                   left, border_type)->Args({width, height});                  \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, float, channels, top, left,     \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);

// RUN_BENCHMARK0(c1, 16, 16, ppl::cv::BORDER_CONSTANT, 640, 480)
// RUN_BENCHMARK0(c3, 16, 16, ppl::cv::BORDER_CONSTANT, 640, 480)
// RUN_BENCHMARK0(c4, 16, 16, ppl::cv::BORDER_CONSTANT, 640, 480)

// RUN_BENCHMARK0(c1, 16, 16, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(c3, 16, 16, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(c4, 16, 16, ppl::cv::BORDER_REPLICATE, 640, 480)

// RUN_BENCHMARK0(c1, 16, 16, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(c3, 16, 16, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(c4, 16, 16, ppl::cv::BORDER_REFLECT, 640, 480)

// RUN_BENCHMARK0(c1, 16, 16, ppl::cv::BORDER_WRAP, 640, 480)
// RUN_BENCHMARK0(c3, 16, 16, ppl::cv::BORDER_WRAP, 640, 480)
// RUN_BENCHMARK0(c4, 16, 16, ppl::cv::BORDER_WRAP, 640, 480)

// RUN_BENCHMARK0(c1, 16, 16, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(c3, 16, 16, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(c4, 16, 16, ppl::cv::BORDER_REFLECT_101, 640, 480)

#define RUN_BENCHMARK1(top, left, border_type, width, height)                  \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_cuda, uchar, 1, top,               \
                   left, border_type)->Args({width, height})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, uchar, 1, top, left,            \
                   border_type)->Args({width, height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_cuda, uchar, 4, top,               \
                   left, border_type)->Args({width, height})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, uchar, 4, top, left,            \
                   border_type)->Args({width, height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_cuda, float, 1, top,               \
                   left, border_type)->Args({width, height})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, float, 1, top, left,            \
                   border_type)->Args({width, height})->                       \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(16, 16, ppl::cv::BORDER_CONSTANT, 640, 480)
// RUN_BENCHMARK1(16, 16, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(16, 16, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(16, 16, ppl::cv::BORDER_WRAP, 640, 480)
// RUN_BENCHMARK1(16, 16, ppl::cv::BORDER_REFLECT_101, 640, 480)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, top, left, border_type)                \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_cuda, type, c1, top, left,         \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_x86_cuda, type, c3, top, left,     \
                   border_type)->Args({640, 480});                             \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_opencv_cuda, type, c4, top, left,         \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, top, left, border_type)                \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, type, c1, top, left,            \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, type, c3, top, left,            \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CopyMakeBorder_ppl_cuda, type, c4, top, left,            \
                   border_type)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_WRAP)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_WRAP)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_WRAP)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 16, 16, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_WRAP)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, ppl::cv::BORDER_REFLECT_101)
