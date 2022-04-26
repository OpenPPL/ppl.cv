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
#include "ppl/cv/cuda/use_memory_pool.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, MaskType mask_type>
void BM_MinMaxLoc_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask(mask);

  T min_val;
  T max_val;
  int min_loc_x;
  int min_loc_y;
  int max_loc_x;
  int max_loc_y;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  ppl::cv::cuda::activateGpuMemoryPool(6144);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "activateGpuMemoryPool() time: " << elapsed_time * 1000000
            << " ns" << std::endl;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::MinMaxLoc<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, &min_val, &max_val,
        &min_loc_x, &min_loc_y, &max_loc_x, &max_loc_y);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        ppl::cv::cuda::MinMaxLoc<T>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, &min_val, &max_val,
            &min_loc_x, &min_loc_y, &max_loc_x, &max_loc_y);
      }
      else {
        ppl::cv::cuda::MinMaxLoc<T>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, &min_val, &max_val,
            &min_loc_x, &min_loc_y, &max_loc_x, &max_loc_y, gpu_mask.step,
            (uchar*)gpu_mask.data);
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

template <typename T, MaskType mask_type>
void BM_MinMaxLoc_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask(mask);

  double min_val, max_val;
  cv::Point min_loc, max_loc;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::minMaxLoc(gpu_src, &min_val, &max_val, &min_loc, &max_loc);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        cv::cuda::minMaxLoc(gpu_src, &min_val, &max_val, &min_loc, &max_loc);
      }
      else {
        cv::cuda::minMaxLoc(gpu_src, &min_val, &max_val, &min_loc, &max_loc,
                            gpu_mask);
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

template <typename T, MaskType mask_type>
void BM_MinMaxLoc_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));

  double min_val, max_val;
  cv::Point min_loc, max_loc;

  for (auto _ : state) {
    if (mask_type == kUnmasked) {
      cv::minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);
    }
    else {
      cv::minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc, mask);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(width, height)                                          \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, uchar, kUnmasked)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, uchar, kUnmasked)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, uchar, kMasked)->                 \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, uchar, kMasked)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, float, kUnmasked)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, float, kUnmasked)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_cuda, float, kMasked)->                 \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, float, kMasked)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(320, 240)
// RUN_BENCHMARK0(640, 480)
// RUN_BENCHMARK0(1280, 720)
// RUN_BENCHMARK0(1920, 1080)

#define RUN_BENCHMARK1(width, height)                                          \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, uchar, kUnmasked)->           \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, uchar, kUnmasked)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, uchar, kMasked)->             \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, uchar, kMasked)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, float, kUnmasked)->           \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, float, kUnmasked)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, float, kMasked)->             \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, float, kMasked)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(320, 240)
// RUN_BENCHMARK1(640, 480)
// RUN_BENCHMARK1(1280, 720)
// RUN_BENCHMARK1(1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, mask_type)                             \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, type, mask_type)->            \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, type, mask_type)->            \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, type, mask_type)->            \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_opencv_x86_cuda, type, mask_type)->            \
                   Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, mask_type)                                             \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type, mask_type)->                   \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type, mask_type)->                   \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type, mask_type)->                   \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_MinMaxLoc_ppl_cuda, type, mask_type)->                   \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, kMasked)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kMasked)
