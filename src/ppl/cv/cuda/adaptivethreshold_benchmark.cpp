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

#include "ppl/cv/cuda/adaptivethreshold.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <int ksize, int adaptive_method>
void BM_AdaptiveThreshold_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  float max_value = 155.f;
  float delta = 10.f;
  int threshold_type = ppl::cv::THRESH_BINARY;
  ppl::cv::BorderType border_type = ppl::cv::BORDER_REPLICATE;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (ksize > 32) {
    cudaEventRecord(start, 0);
    size_t size_width = width * sizeof(float);
    size_t ceiled_volume = ppl::cv::cuda::ceil2DVolume(size_width, height);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "activateGpuMemoryPool() time: " << elapsed_time * 1000000
              << " ns" << std::endl;
  }

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::AdaptiveThreshold(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step, (uchar*)gpu_src.data, gpu_dst.step, (uchar*)gpu_dst.data,
        max_value, adaptive_method, threshold_type, ksize, delta, border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::AdaptiveThreshold(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step, (uchar*)gpu_src.data, gpu_dst.step,
          (uchar*)gpu_dst.data, max_value, adaptive_method, threshold_type,
          ksize, delta, border_type);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  if (ksize > 32) {
    cudaEventRecord(start, 0);
    ppl::cv::cuda::shutDownGpuMemoryPool();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "shutDownGpuMemoryPool() time: " << elapsed_time * 1000000
              << " ns" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <int ksize, int adaptive_method>
void BM_AdaptiveThreshold_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<uchar>::depth, 1));

  float max_value = 155.f;
  float delta = 10.f;
  int threshold_type = ppl::cv::THRESH_BINARY;

  cv::AdaptiveThresholdTypes cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
  if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_MEAN_C) {
    cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
  }
  else if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C) {
    cv_adaptive_method = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
  }
  else {
  }

  cv::ThresholdTypes cv_threshold_type = cv::THRESH_BINARY;
  if (threshold_type == ppl::cv::THRESH_BINARY) {
    cv_threshold_type = cv::THRESH_BINARY;
  }
  else if (threshold_type == ppl::cv::THRESH_BINARY_INV) {
    cv_threshold_type = cv::THRESH_BINARY_INV;
  }

  for (auto _ : state) {
    cv::adaptiveThreshold(src, dst, max_value, cv_adaptive_method,
                          cv_threshold_type, ksize, delta);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(ksize, adaptive_method, width, height)                   \
BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_opencv_x86_cuda, ksize,                \
                   adaptive_method)->Args({width, height});                    \
BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_ppl_cuda, ksize, adaptive_method)->    \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK(3, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 640, 480)
// RUN_BENCHMARK(3, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
// RUN_BENCHMARK(7, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 640, 480)
// RUN_BENCHMARK(7, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
// RUN_BENCHMARK(13, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 640, 480)
// RUN_BENCHMARK(13, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
// RUN_BENCHMARK(25, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 640, 480)
// RUN_BENCHMARK(25, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
// RUN_BENCHMARK(31, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 640, 480)
// RUN_BENCHMARK(31, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
// RUN_BENCHMARK(43, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 640, 480)
// RUN_BENCHMARK(43, ppl::cv::ADAPTIVE_THRESH_MEAN_C, 1920, 1080)

// RUN_BENCHMARK(3, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
// RUN_BENCHMARK(3, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
// RUN_BENCHMARK(7, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
// RUN_BENCHMARK(7, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
// RUN_BENCHMARK(13, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
// RUN_BENCHMARK(13, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
// RUN_BENCHMARK(25, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
// RUN_BENCHMARK(25, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
// RUN_BENCHMARK(31, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
// RUN_BENCHMARK(31, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
// RUN_BENCHMARK(43, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
// RUN_BENCHMARK(43, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(ksize, adaptive_method)                      \
BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_opencv_x86_cuda, ksize,                \
                   adaptive_method)->Args({640, 480});

#define RUN_PPL_CV_TYPE_FUNCTIONS(ksize, adaptive_method)                      \
BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_ppl_cuda, ksize, adaptive_method)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_OPENCV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)

RUN_PPL_CV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_MEAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(3, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(7, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(13, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(25, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(31, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(35, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
RUN_PPL_CV_TYPE_FUNCTIONS(43, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C)
