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

#include "ppl/cv/cuda/calchist.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, MaskType mask_type>
void BM_CalcHist_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask, dst;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask(mask);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::CalcHist<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, (int*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        ppl::cv::cuda::CalcHist<T>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, (int*)gpu_dst.data);
      }
      else {
        ppl::cv::cuda::CalcHist<T>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, (int*)gpu_dst.data,
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
void BM_CalcHist_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask, dst;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask(mask);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::calcHist(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        cv::cuda::calcHist(gpu_src, gpu_dst);
      }
      else {
        cv::cuda::calcHist(gpu_src, gpu_mask, gpu_dst);
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
void BM_CalcHist_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask, dst;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));

  int channel[] = {0};
  int hist_size = 256;
  float data_range[2] = {0, 256};
  const float* ranges[1] = {data_range};

  for (auto _ : state) {
    if (mask_type == kUnmasked) {
      cv::calcHist(&src, 1, channel, cv::Mat(), dst, 1, &hist_size, ranges,
                   true, false);
    }
    else {
      cv::calcHist(&src, 1, channel, mask, dst, 1, &hist_size, ranges, true,
                   false);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, mask_type, width, height)                     \
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_cuda, uchar, channels,  mask_type)->     \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_cuda, uchar, channels, mask_type)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(c1, kUnmasked, 320, 240)
// RUN_BENCHMARK0(c1, kUnmasked, 640, 480)
// RUN_BENCHMARK0(c1, kUnmasked, 1280, 720)
// RUN_BENCHMARK0(c1, kUnmasked, 1920, 1080)
// RUN_BENCHMARK0(c1, kMasked, 320, 240)
// RUN_BENCHMARK0(c1, kMasked, 640, 480)
// RUN_BENCHMARK0(c1, kMasked, 1280, 720)
// RUN_BENCHMARK0(c1, kMasked, 1920, 1080)

#define RUN_BENCHMARK1(channels, mask_type, width, height)                     \
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_x86_cuda, uchar, channels, mask_type)->  \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_cuda, uchar, channels, mask_type)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(c1, kUnmasked, 320, 240)
// RUN_BENCHMARK1(c1, kUnmasked, 640, 480)
// RUN_BENCHMARK1(c1, kUnmasked, 1280, 720)
// RUN_BENCHMARK1(c1, kUnmasked, 1920, 1080)
// RUN_BENCHMARK1(c1, kMasked, 320, 240)
// RUN_BENCHMARK1(c1, kMasked, 640, 480)
// RUN_BENCHMARK1(c1, kMasked, 1280, 720)
// RUN_BENCHMARK1(c1, kMasked, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, mask_type)                             \
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_cuda, type, c1, mask_type)->             \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_cuda, type, c1, mask_type)->             \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_cuda, type, c1, mask_type)->             \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_cuda, type, c1, mask_type)->             \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, mask_type)                             \
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_cuda, type, c1, mask_type)->                \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_cuda, type, c1, mask_type)->                \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_cuda, type, c1, mask_type)->                \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_cuda, type, c1, mask_type)->                \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, kMasked)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kMasked)
