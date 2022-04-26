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

#include "ppl/cv/cuda/bitwise.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, MaskType mask_type>
void BM_BitwiseAnd_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src0, src1, dst, mask;
  src0 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  src1 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  dst = cv::Mat::zeros(height, width,
                       CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src0(src0);
  cv::cuda::GpuMat gpu_src1(src1);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::BitwiseAnd<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
        gpu_src0.step / sizeof(T), (T*)gpu_src0.data, gpu_src1.step / sizeof(T),
        (T*)gpu_src1.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        ppl::cv::cuda::BitwiseAnd<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
            gpu_src0.step / sizeof(T), (T*)gpu_src0.data,
            gpu_src1.step / sizeof(T), (T*)gpu_src1.data,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else {
        ppl::cv::cuda::BitwiseAnd<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
            gpu_src0.step / sizeof(T), (T*)gpu_src0.data,
            gpu_src1.step / sizeof(T), (T*)gpu_src1.data,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data,
            gpu_mask.step / sizeof(uchar), (uchar*)gpu_mask.data);
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

template <typename T, int channels, MaskType mask_type>
void BM_BitwiseAnd_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src0, src1, dst, mask;
  src0 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  src1 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  dst = cv::Mat::zeros(height, width,
                       CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src0(src0);
  cv::cuda::GpuMat gpu_src1(src1);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::bitwise_and(gpu_src0, gpu_src1, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        cv::cuda::bitwise_and(gpu_src0, gpu_src1, gpu_dst);
      }
      else {
        cv::cuda::bitwise_and(gpu_src0, gpu_src1, gpu_dst, gpu_mask);
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

template <typename T, int channels, MaskType mask_type>
void BM_BitwiseAnd_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src0, src1, dst, mask;
  src0 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  src1 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  dst = cv::Mat::zeros(height, width,
                       CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));

  for (auto _ : state) {
    if (mask_type == kUnmasked) {
      cv::bitwise_and(src0, src1, dst);
    }
    else {
      cv::bitwise_and(src0, src1, dst, mask);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(mask_type, width, height)                               \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c1, mask_type)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c1, mask_type)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c3, mask_type)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c3, mask_type)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c4, mask_type)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c4, mask_type)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(kUnmasked, 320, 240)
// RUN_BENCHMARK0(kUnmasked, 640, 480)
// RUN_BENCHMARK0(kUnmasked, 1280, 720)
// RUN_BENCHMARK0(kUnmasked, 1920, 1080)

#define RUN_BENCHMARK1(mask_type, width, height)                               \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c1, mask_type)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c1, mask_type)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(kMasked, 320, 240)
// RUN_BENCHMARK1(kMasked, 640, 480)
// RUN_BENCHMARK1(kMasked, 1280, 720)
// RUN_BENCHMARK1(kMasked, 1920, 1080)

#define RUN_BENCHMARK2(mask_type, width, height)                               \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86_cuda, uchar, c1, mask_type)->      \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c1, mask_type)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86_cuda, uchar, c3, mask_type)->      \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c3, mask_type)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86_cuda, uchar, c4, mask_type)->      \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c4, mask_type)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK2(kUnmasked, 320, 240)
// RUN_BENCHMARK2(kUnmasked, 640, 480)
// RUN_BENCHMARK2(kUnmasked, 1280, 720)
// RUN_BENCHMARK2(kUnmasked, 1920, 1080)

// RUN_BENCHMARK2(kMasked, 320, 240)
// RUN_BENCHMARK2(kMasked, 640, 480)
// RUN_BENCHMARK2(kMasked, 1280, 720)
// RUN_BENCHMARK2(kMasked, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(width, height)                               \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c1, kUnmasked)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c3, kUnmasked)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c4, kUnmasked)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_cuda, uchar, c1, kMasked)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86_cuda, uchar, c3, kMasked)->        \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_opencv_x86_cuda, uchar, c4, kMasked)->        \
                   Args({width, height});

#define RUN_PPL_CV_TYPE_FUNCTIONS(width, height)                               \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c1, kUnmasked)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c3, kUnmasked)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c4, kUnmasked)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c1, kMasked)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c3, kMasked)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_BitwiseAnd_ppl_cuda, uchar, c4, kMasked)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(320, 240)
RUN_OPENCV_TYPE_FUNCTIONS(640, 480)
RUN_OPENCV_TYPE_FUNCTIONS(1280, 720)
RUN_OPENCV_TYPE_FUNCTIONS(1920, 1080)

RUN_PPL_CV_TYPE_FUNCTIONS(320, 240)
RUN_PPL_CV_TYPE_FUNCTIONS(640, 480)
RUN_PPL_CV_TYPE_FUNCTIONS(1280, 720)
RUN_PPL_CV_TYPE_FUNCTIONS(1920, 1080)
