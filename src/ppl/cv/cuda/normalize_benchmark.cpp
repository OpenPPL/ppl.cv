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

#include "ppl/cv/cuda/normalize.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, ppl::cv::NormTypes norm_type,
          MaskType mask_type>
void BM_Normalize_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<float>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  float alpha = 3.f;
  float beta  = 10.f;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  ppl::cv::cuda::activateGpuMemoryPool(4096);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "activateGpuMemoryPool() time: " << elapsed_time * 1000000
            << " ns" << std::endl;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data,
        gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
        norm_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        ppl::cv::cuda::Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
            norm_type);
      }
      else {
        ppl::cv::cuda::Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
            norm_type, gpu_mask.step, (uchar*)gpu_mask.data);
      }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  cudaEventRecord(start, 0);
  ppl::cv::cuda::shutDownGpuMemoryPool();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "shutDownGpuMemoryPool() time: " << elapsed_time * 1000000
            << " ns" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename T, int channels, ppl::cv::NormTypes norm_type,
          MaskType mask_type>
void BM_Normalize_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<float>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  float alpha = 3.f;
  float beta  = 10.f;

  cv::NormTypes cv_norm_type;
  if (norm_type == ppl::cv::NORM_INF) {
    cv_norm_type = cv::NORM_INF;
  }
  else if (norm_type == ppl::cv::NORM_L1) {
    cv_norm_type = cv::NORM_L1;
  }
  else if (norm_type == ppl::cv::NORM_L2) {
    cv_norm_type = cv::NORM_L2;
  }
  else {  // norm_type == ppl::cv::NORM_MINMAX
    cv_norm_type = cv::NORM_MINMAX;
  }

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::normalize(gpu_src, gpu_dst, alpha, beta, cv_norm_type,
                        CV_MAT_DEPTH(dst.type()));
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        cv::cuda::normalize(gpu_src, gpu_dst, alpha, beta, cv_norm_type,
                            CV_MAT_DEPTH(dst.type()));
      }
      else {
        cv::cuda::normalize(gpu_src, gpu_dst, alpha, beta, cv_norm_type,
                            CV_MAT_DEPTH(dst.type()), gpu_mask);
      }
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

template <typename T, int channels, ppl::cv::NormTypes norm_type,
          MaskType mask_type>
void BM_Normalize_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<float>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));

  float alpha = 3.f;
  float beta  = 10.f;

  cv::NormTypes cv_norm_type;
  if (norm_type == ppl::cv::NORM_INF) {
    cv_norm_type = cv::NORM_INF;
  }
  else if (norm_type == ppl::cv::NORM_L1) {
    cv_norm_type = cv::NORM_L1;
  }
  else if (norm_type == ppl::cv::NORM_L2) {
    cv_norm_type = cv::NORM_L2;
  }
  else {  // norm_type == ppl::cv::NORM_MINMAX
    cv_norm_type = cv::NORM_MINMAX;
  }

  for (auto _ : state) {
    if (mask_type == kUnmasked) {
      cv::normalize(src, dst, alpha, beta, cv_norm_type,
                    CV_MAT_DEPTH(dst.type()));
    }
    else {
      cv::normalize(src, dst, alpha, beta, cv_norm_type,
                    CV_MAT_DEPTH(dst.type()), mask);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, norm_type, mask_type, width, height)          \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_cuda, uchar, channels, norm_type,       \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, uchar, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_cuda, float, channels, norm_type,       \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, float, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

#define RUN_BENCHMARK1(channels, norm_type, mask_type, width, height)          \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_cuda, uchar, channels, norm_type,       \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, uchar, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

// RUN_BENCHMARK0(c1, ppl::cv::NORM_INF, kUnmasked, 640, 480)
// RUN_BENCHMARK0(c1, ppl::cv::NORM_INF, kUnmasked, 1920, 1080)
// RUN_BENCHMARK0(c1, ppl::cv::NORM_L1, kUnmasked, 640, 480)
// RUN_BENCHMARK0(c1, ppl::cv::NORM_L1, kUnmasked, 1920, 1080)
// RUN_BENCHMARK0(c1, ppl::cv::NORM_L2, kUnmasked, 640, 480)
// RUN_BENCHMARK0(c1, ppl::cv::NORM_L2, kUnmasked, 1920, 1080)
// RUN_BENCHMARK0(c1, ppl::cv::NORM_MINMAX, kUnmasked, 640, 480)
// RUN_BENCHMARK0(c1, ppl::cv::NORM_MINMAX, kUnmasked, 1920, 1080)

// RUN_BENCHMARK1(c1, ppl::cv::NORM_INF, kMasked, 640, 480)
// RUN_BENCHMARK1(c1, ppl::cv::NORM_INF, kMasked, 1920, 1080)
// RUN_BENCHMARK1(c1, ppl::cv::NORM_L1, kMasked, 640, 480)
// RUN_BENCHMARK1(c1, ppl::cv::NORM_L1, kMasked, 1920, 1080)
// RUN_BENCHMARK1(c1, ppl::cv::NORM_L2, kMasked, 640, 480)
// RUN_BENCHMARK1(c1, ppl::cv::NORM_L2, kMasked, 1920, 1080)
// RUN_BENCHMARK1(c1, ppl::cv::NORM_MINMAX, kMasked, 640, 480)
// RUN_BENCHMARK1(c1, ppl::cv::NORM_MINMAX, kMasked, 1920, 1080)

#define RUN_BENCHMARK2(channels, norm_type, mask_type, width, height)          \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, uchar, channels, norm_type,   \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, uchar, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, float, channels, norm_type,   \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, float, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

// RUN_BENCHMARK2(c1, ppl::cv::NORM_INF, kMasked, 640, 480)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_INF, kMasked, 640, 480)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_INF, kMasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_INF, kMasked, 1920, 1080)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_INF, kMasked, 1920, 1080)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_INF, kMasked, 1920, 1080)

// RUN_BENCHMARK2(c1, ppl::cv::NORM_L1, kMasked, 640, 480)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L1, kMasked, 640, 480)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L1, kMasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_L1, kMasked, 1920, 1080)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L1, kMasked, 1920, 1080)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L1, kMasked, 1920, 1080)

// RUN_BENCHMARK2(c1, ppl::cv::NORM_L2, kMasked, 640, 480)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L2, kMasked, 640, 480)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L2, kMasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_L2, kMasked, 1920, 1080)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L2, kMasked, 1920, 1080)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L2, kMasked, 1920, 1080)

// RUN_BENCHMARK2(c1, ppl::cv::NORM_MINMAX, kMasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_MINMAX, kMasked, 1920, 1080)

// RUN_BENCHMARK2(c1, ppl::cv::NORM_INF, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_INF, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_INF, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_INF, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_INF, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_INF, kUnmasked, 1920, 1080)

// RUN_BENCHMARK2(c1, ppl::cv::NORM_L1, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L1, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L1, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_L1, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L1, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L1, kUnmasked, 1920, 1080)

// RUN_BENCHMARK2(c1, ppl::cv::NORM_L2, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L2, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L2, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_L2, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_L2, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_L2, kUnmasked, 1920, 1080)

// RUN_BENCHMARK2(c1, ppl::cv::NORM_MINMAX, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_MINMAX, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_MINMAX, kUnmasked, 640, 480)
// RUN_BENCHMARK2(c1, ppl::cv::NORM_MINMAX, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c3, ppl::cv::NORM_MINMAX, kUnmasked, 1920, 1080)
// RUN_BENCHMARK2(c4, ppl::cv::NORM_MINMAX, kUnmasked, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS0(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({640, 480});                               \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c3, norm_type,          \
                   mask_type)->Args({640, 480});                               \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c4, norm_type,          \
                   mask_type)->Args({640, 480});                               \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({1920, 1080});                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c3, norm_type,          \
                   mask_type)->Args({1920, 1080});                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c4, norm_type,          \
                   mask_type)->Args({1920, 1080});

#define RUN_OPENCV_TYPE_FUNCTIONS1(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({640, 480});                               \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS0(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c3, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c4, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c3, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c4, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS1(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_INF, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L1, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L2, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS1(uchar, ppl::cv::NORM_MINMAX, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_INF, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L1, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L2, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS1(float, ppl::cv::NORM_MINMAX, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_INF, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L1, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L2, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_MINMAX, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_INF, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L1, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L2, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_MINMAX, kUnmasked)

RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_INF, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L1, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L2, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS1(uchar, ppl::cv::NORM_MINMAX, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_INF, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L1, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L2, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS1(float, ppl::cv::NORM_MINMAX, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_INF, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L1, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_L2, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, ppl::cv::NORM_MINMAX, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_INF, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L1, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_L2, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, ppl::cv::NORM_MINMAX, kUnmasked)
