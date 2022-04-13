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

#include "ppl/cv/cuda/arithmetic.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

double double_alpha = 0.1;
double double_beta  = 0.2;
double double_gamma = 0.3;

enum ArithFunctions {
  kADD,
  kADDWEITHTED,
  kSUBTRACT,
  kMUL,
  kDIV,
  kMLA,
  kMLS,
};

template <typename T, int channels, ArithFunctions function>
void BM_Arith_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, dst;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(src.rows, src.cols, src.type());

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Add<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_src.step / sizeof(T),
        (T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (function == kADD) {
        ppl::cv::cuda::Add<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else if (function == kADDWEITHTED) {
        ppl::cv::cuda::AddWeighted<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, double_alpha,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, double_beta,
            double_gamma, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else if (function == kSUBTRACT) {
        T scalars[4];
        for (int i = 0; i < channels; i++) {
          scalars[i] = ((T*)(src.data))[i];
        }
        ppl::cv::cuda::Subtract<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, (T*)scalars,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else if (function == kMUL) {
        ppl::cv::cuda::Mul<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else if (function == kDIV) {
        ppl::cv::cuda::Div<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else if (function == kMLA && sizeof(T) == 4) {
        ppl::cv::cuda::Mla<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else if (function == kMLS && sizeof(T) == 4) {
        ppl::cv::cuda::Mls<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else {
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

template <typename T, int channels, ArithFunctions function>
void BM_Arith_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, dst;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(src.rows, src.cols, src.type());

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::add(gpu_src, gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (function == kADD) {
        cv::cuda::add(gpu_src, gpu_src, gpu_dst);
      }
      else if (function == kADDWEITHTED) {
        cv::cuda::addWeighted(gpu_src, double_alpha, gpu_src, double_beta,
                              double_gamma, gpu_dst);
      }
      else if (function == kSUBTRACT) {
        cv::Scalar scalar;
        for (int i = 0; i < channels; i++) {
          scalar[i] = ((T*)(src.data))[i];
        }
        cv::cuda::subtract(gpu_src, scalar, gpu_dst);
      }
      else if (function == kMUL) {
        cv::cuda::multiply(gpu_src, gpu_src, gpu_dst);
      }
      else if (function == kDIV) {
        cv::cuda::divide(gpu_src, gpu_src, gpu_dst);
      }
      else {
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

template <typename T, int channels, ArithFunctions function>
void BM_Arith_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(src.rows, src.cols, src.type());

  for (auto _ : state) {
    if (function == kADD) {
      cv::add(src, src, dst);
    }
    else if (function == kADDWEITHTED) {
      cv::addWeighted(src, double_alpha, src, double_beta, double_gamma, dst);
    }
    else if (function == kSUBTRACT) {
      cv::Scalar scalar;
      for (int i = 0; i < channels; i++) {
        scalar[i] = ((T*)(src.data))[i];
      }
      cv::subtract(src, scalar, dst);
    }
    else if (function == kMUL) {
      cv::multiply(src, src, dst);
    }
    else if (function == kDIV) {
      cv::divide(src, src, dst);
    }
    else {
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, function, width, height)                       \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, uchar, channels, function)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, uchar, channels, function)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, float, channels, function)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, float, channels, function)->             \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK(c1, kADD, 640, 480)
// RUN_BENCHMARK(c3, kADD, 640, 480)
// RUN_BENCHMARK(c4, kADD, 640, 480)

// RUN_BENCHMARK(c1, kADD, 1920, 1080)
// RUN_BENCHMARK(c3, kADD, 1920, 1080)
// RUN_BENCHMARK(c4, kADD, 1920, 1080)

// RUN_BENCHMARK(c1, kADDWEITHTED, 640, 480)
// RUN_BENCHMARK(c3, kADDWEITHTED, 640, 480)
// RUN_BENCHMARK(c4, kADDWEITHTED, 640, 480)

// RUN_BENCHMARK(c1, kADDWEITHTED, 1920, 1080)
// RUN_BENCHMARK(c3, kADDWEITHTED, 1920, 1080)
// RUN_BENCHMARK(c4, kADDWEITHTED, 1920, 1080)

// RUN_BENCHMARK(c1, kSUBTRACT, 640, 480)
// RUN_BENCHMARK(c3, kSUBTRACT, 640, 480)
// RUN_BENCHMARK(c4, kSUBTRACT, 640, 480)

// RUN_BENCHMARK(c1, kSUBTRACT, 1920, 1080)
// RUN_BENCHMARK(c3, kSUBTRACT, 1920, 1080)
// RUN_BENCHMARK(c4, kSUBTRACT, 1920, 1080)

// RUN_BENCHMARK(c1, kMUL, 640, 480)
// RUN_BENCHMARK(c3, kMUL, 640, 480)
// RUN_BENCHMARK(c4, kMUL, 640, 480)

// RUN_BENCHMARK(c1, kMUL, 1920, 1080)
// RUN_BENCHMARK(c3, kMUL, 1920, 1080)
// RUN_BENCHMARK(c4, kMUL, 1920, 1080)

// RUN_BENCHMARK(c1, kDIV, 640, 480)
// RUN_BENCHMARK(c3, kDIV, 640, 480)
// RUN_BENCHMARK(c4, kDIV, 640, 480)

// RUN_BENCHMARK(c1, kDIV, 1920, 1080)
// RUN_BENCHMARK(c3, kDIV, 1920, 1080)
// RUN_BENCHMARK(c4, kDIV, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, function)                              \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, type, c1, function)->                 \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, type, c3, function)->                 \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, type, c4, function)->                 \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, type, c1, function)->                 \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, type, c3, function)->                 \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Arith_opencv_cuda, type, c4, function)->                 \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, function)                              \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, type, c1, function)->                    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, type, c3, function)->                    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, type, c4, function)->                    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, type, c1, function)->                    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, type, c3, function)->                    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Arith_ppl_cuda, type, c4, function)->                    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kADD)
RUN_OPENCV_TYPE_FUNCTIONS(float, kADD)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kADD)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kADD)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kADDWEITHTED)
RUN_OPENCV_TYPE_FUNCTIONS(float, kADDWEITHTED)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kADDWEITHTED)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kADDWEITHTED)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kSUBTRACT)
RUN_OPENCV_TYPE_FUNCTIONS(float, kSUBTRACT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kSUBTRACT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kSUBTRACT)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kMUL)
RUN_OPENCV_TYPE_FUNCTIONS(float, kMUL)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kMUL)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kMUL)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kDIV)
RUN_OPENCV_TYPE_FUNCTIONS(float, kDIV)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, kDIV)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kDIV)

RUN_PPL_CV_TYPE_FUNCTIONS(float, kMLA)
RUN_PPL_CV_TYPE_FUNCTIONS(float, kMLS)
