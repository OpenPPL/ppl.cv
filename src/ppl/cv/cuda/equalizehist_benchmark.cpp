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

#include "ppl/cv/cuda/equalizehist.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T>
void BM_EqualizeHist_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          1));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  ppl::cv::cuda::activateGpuMemoryPool(1024);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "activateGpuMemoryPool() time: " << elapsed_time * 1000000
            << " ns" << std::endl;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::EqualizeHist(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::EqualizeHist(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
          (T*)gpu_dst.data);
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

template <typename T>
void BM_EqualizeHist_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          1));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::equalizeHist(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::equalizeHist(gpu_src, gpu_dst);
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

template <typename T>
void BM_EqualizeHist_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          1));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));

  for (auto _ : state) {
    cv::equalizeHist(src, dst);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(width, height)                                          \
BENCHMARK_TEMPLATE(BM_EqualizeHist_opencv_cuda, uchar)->Args({width, height})->\
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_ppl_cuda, uchar)->Args({width, height})->   \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(320, 240)
// RUN_BENCHMARK0(640, 480)
// RUN_BENCHMARK0(1280, 720)
// RUN_BENCHMARK0(1920, 1080)

#define RUN_BENCHMARK1(width, height)                                          \
BENCHMARK_TEMPLATE(BM_EqualizeHist_opencv_x86_cuda, uchar)->                   \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_EqualizeHist_ppl_cuda, uchar)->Args({width, height})->   \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(320, 240)
// RUN_BENCHMARK1(640, 480)
// RUN_BENCHMARK1(1280, 720)
// RUN_BENCHMARK1(1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS()                                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_opencv_cuda, uchar)->Args({320, 240})->     \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_opencv_cuda, uchar)->Args({640, 480})->     \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_opencv_cuda, uchar)->Args({1280, 720})->    \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_opencv_cuda, uchar)->Args({1920, 1080})->   \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS()                                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_ppl_cuda, uchar)->Args({320, 240})->        \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_ppl_cuda, uchar)->Args({640, 480})->        \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_ppl_cuda, uchar)->Args({1280, 720})->       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_EqualizeHist_ppl_cuda, uchar)->Args({1920, 1080})->      \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS()
RUN_PPL_CV_TYPE_FUNCTIONS()
