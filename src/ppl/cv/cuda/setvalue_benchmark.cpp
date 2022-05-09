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

#include "ppl/cv/cuda/setvalue.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

enum SetValueFunctions {
  kUnmaskedSetTo,
  kMaskedSetTo,
  kOnes,
  kZeros,
};

template <typename T, int outChannels, int maskChannels,
          SetValueFunctions function>
void BM_SetValue_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat dst, mask;
  dst = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          outChannels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth,
                           maskChannels));
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  int value = 5;

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Zeros<T, outChannels>(0, gpu_dst.rows, gpu_dst.cols,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (function == kUnmaskedSetTo) {
        ppl::cv::cuda::SetTo<T, outChannels, maskChannels>(0, gpu_dst.rows,
            gpu_dst.cols, gpu_dst.step / sizeof(T), (T*)gpu_dst.data, value);
      }
      else if (function == kMaskedSetTo) {
        ppl::cv::cuda::SetTo<T, outChannels, maskChannels>(0, gpu_dst.rows,
            gpu_dst.cols, gpu_dst.step / sizeof(T), (T*)gpu_dst.data, value,
            gpu_mask.step, gpu_mask.data);
      }
      else if (function == kOnes) {
        ppl::cv::cuda::Ones<T, outChannels>(0, gpu_dst.rows, gpu_dst.cols,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else {
        ppl::cv::cuda::Zeros<T, outChannels>(0, gpu_dst.rows, gpu_dst.cols,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
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

template <typename T, int outChannels, int maskChannels,
          SetValueFunctions function>
void BM_SetValue_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat dst, mask;
  dst = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          outChannels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  cv::Scalar scalar_zero(0, 0, 0, 0);
  cv::Scalar scalar_five(5, 9, 2, 37);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    gpu_dst.setTo(scalar_zero);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (function == kUnmaskedSetTo) {
        gpu_dst.setTo(scalar_five);
      }
      else if (function == kMaskedSetTo) {
        gpu_dst.setTo(scalar_five, gpu_mask);
      }
      else {
        gpu_dst.setTo(scalar_zero);
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

template <typename T, int outChannels, int maskChannels,
          SetValueFunctions function>
void BM_SetValue_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat dst, mask;
  dst = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          outChannels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth,
                           maskChannels));

  int value = 5;

  for (auto _ : state) {
    if (function == kUnmaskedSetTo) {
      dst.setTo(value);
    }
    else if (function == kMaskedSetTo) {
      dst.setTo(value, mask);
    }
    else if (function == kOnes) {
      dst = cv::Mat::ones(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          outChannels));
    }
    else {
      dst = cv::Mat::zeros(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           outChannels));
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK00(function, width, height)                               \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c1, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c3, c3, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, c3, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c4, c4, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, c4, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c3, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c4, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c1, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c3, c3, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, c3, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c4, c4, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, c4, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c3, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c4, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK00(kUnmaskedSetTo, 320, 240)
// RUN_BENCHMARK00(kUnmaskedSetTo, 640, 480)
// RUN_BENCHMARK00(kUnmaskedSetTo, 1280, 720)
// RUN_BENCHMARK00(kUnmaskedSetTo, 1920, 1080)

// RUN_BENCHMARK00(kMaskedSetTo, 320, 240)
// RUN_BENCHMARK00(kMaskedSetTo, 640, 480)
// RUN_BENCHMARK00(kMaskedSetTo, 1280, 720)
// RUN_BENCHMARK00(kMaskedSetTo, 1920, 1080)

#define RUN_BENCHMARK01(function, width, height)                               \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c1, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c3, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c4, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c1, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c3, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c4, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK01(kOnes, 320, 240)
// RUN_BENCHMARK01(kOnes, 640, 480)
// RUN_BENCHMARK01(kOnes, 1280, 720)
// RUN_BENCHMARK01(kOnes, 1920, 1080)

// RUN_BENCHMARK01(kZeros, 320, 240)
// RUN_BENCHMARK01(kZeros, 640, 480)
// RUN_BENCHMARK01(kZeros, 1280, 720)
// RUN_BENCHMARK01(kZeros, 1920, 1080)

#define RUN_BENCHMARK1(function, width, height)                                \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c1, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c3, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c4, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c1, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c3, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c4, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(kUnmaskedSetTo, 320, 240)
// RUN_BENCHMARK1(kUnmaskedSetTo, 640, 480)
// RUN_BENCHMARK1(kUnmaskedSetTo, 1280, 720)
// RUN_BENCHMARK1(kUnmaskedSetTo, 1920, 1080)

// RUN_BENCHMARK1(kMaskedSetTo, 320, 240)
// RUN_BENCHMARK1(kMaskedSetTo, 640, 480)
// RUN_BENCHMARK1(kMaskedSetTo, 1280, 720)
// RUN_BENCHMARK1(kMaskedSetTo, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS0(function, width, height)                    \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c1, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c3, c3, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c4, c4, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c3, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c4, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c1, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c3, c3, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c4, c4, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c3, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c4, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);

#define RUN_OPENCV_TYPE_FUNCTIONS1(function, width, height)                    \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c1, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c3, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, uchar, c4, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c1, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c3, c1, function)->     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_x86_cuda, float, c4, c1, function)->     \
                   Args({width, height});

#define RUN_OPENCV_TYPE_FUNCTIONS2(function, width, height)                    \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c1, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c3, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, uchar, c4, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c1, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c3, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_opencv_cuda, float, c4, c1, function)->         \
                   Args({width, height})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS0(function, width, height)                    \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, c3, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, c4, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, c3, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, c4, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS1(function, width, height)                    \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, uchar, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c1, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c3, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_SetValue_ppl_cuda, float, c4, c1, function)->            \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 320, 240)
RUN_OPENCV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 640, 480)
RUN_OPENCV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 1280, 720)
RUN_OPENCV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 1920, 1080)

RUN_OPENCV_TYPE_FUNCTIONS0(kMaskedSetTo, 320, 240)
RUN_OPENCV_TYPE_FUNCTIONS0(kMaskedSetTo, 640, 480)
RUN_OPENCV_TYPE_FUNCTIONS0(kMaskedSetTo, 1280, 720)
RUN_OPENCV_TYPE_FUNCTIONS0(kMaskedSetTo, 1920, 1080)

RUN_OPENCV_TYPE_FUNCTIONS1(kOnes, 320, 240)
RUN_OPENCV_TYPE_FUNCTIONS1(kOnes, 640, 480)
RUN_OPENCV_TYPE_FUNCTIONS1(kOnes, 1280, 720)
RUN_OPENCV_TYPE_FUNCTIONS1(kOnes, 1920, 1080)

RUN_OPENCV_TYPE_FUNCTIONS2(kZeros, 320, 240)
RUN_OPENCV_TYPE_FUNCTIONS2(kZeros, 640, 480)
RUN_OPENCV_TYPE_FUNCTIONS2(kZeros, 1280, 720)
RUN_OPENCV_TYPE_FUNCTIONS2(kZeros, 1920, 1080)

RUN_PPL_CV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 320, 240)
RUN_PPL_CV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 640, 480)
RUN_PPL_CV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 1280, 720)
RUN_PPL_CV_TYPE_FUNCTIONS0(kUnmaskedSetTo, 1920, 1080)

RUN_PPL_CV_TYPE_FUNCTIONS0(kMaskedSetTo, 320, 240)
RUN_PPL_CV_TYPE_FUNCTIONS0(kMaskedSetTo, 640, 480)
RUN_PPL_CV_TYPE_FUNCTIONS0(kMaskedSetTo, 1280, 720)
RUN_PPL_CV_TYPE_FUNCTIONS0(kMaskedSetTo, 1920, 1080)

RUN_PPL_CV_TYPE_FUNCTIONS1(kOnes, 320, 240)
RUN_PPL_CV_TYPE_FUNCTIONS1(kOnes, 640, 480)
RUN_PPL_CV_TYPE_FUNCTIONS1(kOnes, 1280, 720)
RUN_PPL_CV_TYPE_FUNCTIONS1(kOnes, 1920, 1080)

RUN_PPL_CV_TYPE_FUNCTIONS1(kZeros, 320, 240)
RUN_PPL_CV_TYPE_FUNCTIONS1(kZeros, 640, 480)
RUN_PPL_CV_TYPE_FUNCTIONS1(kZeros, 1280, 720)
RUN_PPL_CV_TYPE_FUNCTIONS1(kZeros, 1920, 1080)