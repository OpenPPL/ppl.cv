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

#include "ppl/cv/cuda/split.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels>
void BM_Split_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst0(dst);
  cv::cuda::GpuMat gpu_dst1(dst);
  cv::cuda::GpuMat gpu_dst2(dst);
  cv::cuda::GpuMat gpu_dst3(dst);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Split3Channels<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst0.step / sizeof(T),
        (T*)gpu_dst0.data, (T*)gpu_dst1.data, (T*)gpu_dst2.data);
}
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (channels == 3) {
        ppl::cv::cuda::Split3Channels<T>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst0.step / sizeof(T), (T*)gpu_dst0.data,
            (T*)gpu_dst1.data, (T*)gpu_dst2.data);
      }
      else {  // channels == 4
        ppl::cv::cuda::Split4Channels<T>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst0.step / sizeof(T), (T*)gpu_dst0.data,
            (T*)gpu_dst1.data, (T*)gpu_dst2.data, (T*)gpu_dst3.data);
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

template <typename T, int channels>
void BM_Split_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst0(dst);
  cv::cuda::GpuMat gpu_dst1(dst);
  cv::cuda::GpuMat gpu_dst2(dst);
  cv::cuda::GpuMat gpu_dst3(dst);

  cv::cuda::GpuMat gpu_dsts0[3] = {gpu_dst0, gpu_dst1, gpu_dst2};
  cv::cuda::GpuMat gpu_dsts1[4] = {gpu_dst0, gpu_dst1, gpu_dst2, gpu_dst3};

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::split(gpu_src, gpu_dsts0);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (channels == 3) {
        cv::cuda::split(gpu_src, gpu_dsts0);
      }
      else {  // channels == 4
        cv::cuda::split(gpu_src, gpu_dsts1);
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

template <typename T, int channels>
void BM_Split_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst2(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst3(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));

  cv::Mat dsts0[3] = {dst0, dst1, dst2};
  cv::Mat dsts1[4] = {dst0, dst1, dst2, dst3};

  for (auto _ : state) {
    if (channels == 3) {
      cv::split(src, dsts0);
    }
    else {  // channels == 4
      cv::split(src, dsts1);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, width, height)                                \
BENCHMARK_TEMPLATE(BM_Split_opencv_x86_cuda, uchar, channels)->                \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, uchar, channels)->                       \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Split_opencv_x86_cuda, float, channels)->                \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, float, channels)->                       \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(c3, 320, 240)
// RUN_BENCHMARK0(c4, 320, 240)
// RUN_BENCHMARK0(c3, 640, 480)
// RUN_BENCHMARK0(c4, 640, 480)
// RUN_BENCHMARK0(c3, 1280, 720)
// RUN_BENCHMARK0(c4, 1280, 720)
// RUN_BENCHMARK0(c3, 1920, 1080)
// RUN_BENCHMARK0(c4, 1920, 1080)

#define RUN_BENCHMARK1(channels, width, height)                                \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, uchar, channels)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, uchar, channels)->                       \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, float, channels)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, float, channels)->                       \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(c3, 320, 240)
// RUN_BENCHMARK1(c4, 320, 240)
// RUN_BENCHMARK1(c3, 640, 480)
// RUN_BENCHMARK1(c4, 640, 480)
// RUN_BENCHMARK1(c3, 1280, 720)
// RUN_BENCHMARK1(c4, 1280, 720)
// RUN_BENCHMARK1(c3, 1920, 1080)
// RUN_BENCHMARK1(c4, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(channels)                                    \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, uchar, channels)->                    \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, float, channels)->                    \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, uchar, channels)->                    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, float, channels)->                    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, uchar, channels)->                    \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, float, channels)->                    \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, uchar, channels)->                    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Split_opencv_cuda, float, channels)->                    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(channels)                                    \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, uchar, channels)->                       \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, float, channels)->                       \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, uchar, channels)->                       \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, float, channels)->                       \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, uchar, channels)->                       \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, float, channels)->                       \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, uchar, channels)->                       \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Split_ppl_cuda, float, channels)->                       \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(c3)
RUN_OPENCV_TYPE_FUNCTIONS(c4)

RUN_PPL_CV_TYPE_FUNCTIONS(c3)
RUN_PPL_CV_TYPE_FUNCTIONS(c4)
