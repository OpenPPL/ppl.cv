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

#include "ppl/cv/ocl/arithmetic.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

float falpha = 0.1f;
float fbeta  = 0.2f;
float fgamma = 0.3f;

enum ArithFunctions {
  kADD,
  kADDWEITHTED,
  kSUBTRACT,
  kMUL0,
  kMUL1,
  kDIV0,
  kDIV1,
};

template <typename T, int channels, ArithFunctions function>
void BM_Arith_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src0, src1;
  src0 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  src1 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes0 = src0.rows * src0.step;
  int src_bytes1 = src1.rows * src1.step;
  int dst_bytes = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src0 = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                   src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_src1 = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                   src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src0, CL_TRUE, 0, src_bytes0,
                                    src0.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src1, CL_TRUE, 0, src_bytes1,
                                    src1.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int iterations = 100;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    if (function == kADD) {
      ppl::cv::ocl::Add<T, channels>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
          dst.step / sizeof(T), gpu_dst);
    }
    else if (function == kADDWEITHTED) {
      ppl::cv::ocl::AddWeighted<T, channels>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, falpha, src0.step / sizeof(T),
          gpu_src1, fbeta, fgamma, dst.step / sizeof(T), gpu_dst);
    }
    else if (function == kSUBTRACT) {
      ppl::cv::ocl::Subtract<T, channels>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
          dst.step / sizeof(T), gpu_dst);
    }
    else if (function == kMUL0) {
      ppl::cv::ocl::Mul<T, channels>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
          dst.step / sizeof(T), gpu_dst);
    }
    else if (function == kMUL1) {
      ppl::cv::ocl::Mul<T, channels>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
          dst.step / sizeof(T), gpu_dst, falpha);
    }
    else if (function == kDIV0) {
      ppl::cv::ocl::Div<T, channels>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
          dst.step / sizeof(T), gpu_dst);
    }
    else if (function == kDIV1) {
      ppl::cv::ocl::Div<T, channels>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
          dst.step / sizeof(T), gpu_dst, falpha);
    }
    else {
    }
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (function == kADD) {
        ppl::cv::ocl::Add<T, channels>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
            dst.step / sizeof(T), gpu_dst);
      }
      else if (function == kADDWEITHTED) {
        ppl::cv::ocl::AddWeighted<T, channels>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, falpha, src0.step / sizeof(T),
            gpu_src1, fbeta, fgamma, dst.step / sizeof(T), gpu_dst);
      }
      else if (function == kSUBTRACT) {
        ppl::cv::ocl::Subtract<T, channels>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
            dst.step / sizeof(T), gpu_dst);
      }
      else if (function == kMUL0) {
        ppl::cv::ocl::Mul<T, channels>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
            dst.step / sizeof(T), gpu_dst);
      }
      else if (function == kMUL1) {
        ppl::cv::ocl::Mul<T, channels>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
            dst.step / sizeof(T), gpu_dst, falpha);
      }
      else if (function == kDIV0) {
        ppl::cv::ocl::Div<T, channels>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
            dst.step / sizeof(T), gpu_dst);
      }
      else if (function == kDIV1) {
        ppl::cv::ocl::Div<T, channels>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, src0.step / sizeof(T), gpu_src1,
            dst.step / sizeof(T), gpu_dst, falpha);
      }
      else {
      }
    }
    clFinish(queue);
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  clReleaseMemObject(gpu_src0);
  clReleaseMemObject(gpu_src1);
  clReleaseMemObject(gpu_dst);
}

template <typename T, int channels, ArithFunctions function>
void BM_Arith_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src0, src1, dst;
  src0 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  src1 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));

  for (auto _ : state) {
    if (function == kADD) {
      cv::add(src0, src1, dst);
    }
    else if (function == kADDWEITHTED) {
      cv::addWeighted(src0, falpha, src1, fbeta, fgamma, dst);
    }
    else if (function == kSUBTRACT) {
      cv::subtract(src0, src1, dst);
    }
    else if (function == kMUL0) {
      cv::multiply(src0, src1, dst);
    }
    else if (function == kMUL1) {
      cv::multiply(src0, src1, dst, falpha);
    }
    else if (function == kDIV0) {
      cv::divide(src0, src1, dst);
    }
    else if (function == kDIV1) {
      cv::divide(src0, src1, dst, falpha);
    }
    else {
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, function, width, height)                       \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, uchar, channels, function)->           \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, uchar, channels, function)->              \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, float, channels, function)->           \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, float, channels, function)->              \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_BENCHMARK(c1, kADD, 320, 240)
RUN_BENCHMARK(c3, kADD, 320, 240)
RUN_BENCHMARK(c4, kADD, 320, 240)

RUN_BENCHMARK(c1, kADD, 640, 480)
RUN_BENCHMARK(c3, kADD, 640, 480)
RUN_BENCHMARK(c4, kADD, 640, 480)

RUN_BENCHMARK(c1, kADD, 1920, 1080)
RUN_BENCHMARK(c3, kADD, 1920, 1080)
RUN_BENCHMARK(c4, kADD, 1920, 1080)

RUN_BENCHMARK(c1, kADDWEITHTED, 320, 240)
RUN_BENCHMARK(c3, kADDWEITHTED, 320, 240)
RUN_BENCHMARK(c4, kADDWEITHTED, 320, 240)

RUN_BENCHMARK(c1, kADDWEITHTED, 640, 480)
RUN_BENCHMARK(c3, kADDWEITHTED, 640, 480)
RUN_BENCHMARK(c4, kADDWEITHTED, 640, 480)

RUN_BENCHMARK(c1, kADDWEITHTED, 1920, 1080)
RUN_BENCHMARK(c3, kADDWEITHTED, 1920, 1080)
RUN_BENCHMARK(c4, kADDWEITHTED, 1920, 1080)

RUN_BENCHMARK(c1, kSUBTRACT, 320, 240)
RUN_BENCHMARK(c3, kSUBTRACT, 320, 240)
RUN_BENCHMARK(c4, kSUBTRACT, 320, 240)

RUN_BENCHMARK(c1, kSUBTRACT, 640, 480)
RUN_BENCHMARK(c3, kSUBTRACT, 640, 480)
RUN_BENCHMARK(c4, kSUBTRACT, 640, 480)

RUN_BENCHMARK(c1, kSUBTRACT, 1920, 1080)
RUN_BENCHMARK(c3, kSUBTRACT, 1920, 1080)
RUN_BENCHMARK(c4, kSUBTRACT, 1920, 1080)

RUN_BENCHMARK(c1, kMUL0, 320, 240)
RUN_BENCHMARK(c3, kMUL0, 320, 240)
RUN_BENCHMARK(c4, kMUL0, 320, 240)

RUN_BENCHMARK(c1, kMUL0, 640, 480)
RUN_BENCHMARK(c3, kMUL0, 640, 480)
RUN_BENCHMARK(c4, kMUL0, 640, 480)

RUN_BENCHMARK(c1, kMUL0, 1920, 1080)
RUN_BENCHMARK(c3, kMUL0, 1920, 1080)
RUN_BENCHMARK(c4, kMUL0, 1920, 1080)

RUN_BENCHMARK(c1, kMUL1, 320, 240)
RUN_BENCHMARK(c3, kMUL1, 320, 240)
RUN_BENCHMARK(c4, kMUL1, 320, 240)

RUN_BENCHMARK(c1, kMUL1, 640, 480)
RUN_BENCHMARK(c3, kMUL1, 640, 480)
RUN_BENCHMARK(c4, kMUL1, 640, 480)

RUN_BENCHMARK(c1, kMUL1, 1920, 1080)
RUN_BENCHMARK(c3, kMUL1, 1920, 1080)
RUN_BENCHMARK(c4, kMUL1, 1920, 1080)

RUN_BENCHMARK(c1, kDIV0, 320, 240)
RUN_BENCHMARK(c3, kDIV0, 320, 240)
RUN_BENCHMARK(c4, kDIV0, 320, 240)

RUN_BENCHMARK(c1, kDIV0, 640, 480)
RUN_BENCHMARK(c3, kDIV0, 640, 480)
RUN_BENCHMARK(c4, kDIV0, 640, 480)

RUN_BENCHMARK(c1, kDIV0, 1920, 1080)
RUN_BENCHMARK(c3, kDIV0, 1920, 1080)
RUN_BENCHMARK(c4, kDIV0, 1920, 1080)

RUN_BENCHMARK(c1, kDIV1, 320, 240)
RUN_BENCHMARK(c3, kDIV1, 320, 240)
RUN_BENCHMARK(c4, kDIV1, 320, 240)

RUN_BENCHMARK(c1, kDIV1, 640, 480)
RUN_BENCHMARK(c3, kDIV1, 640, 480)
RUN_BENCHMARK(c4, kDIV1, 640, 480)

RUN_BENCHMARK(c1, kDIV1, 1920, 1080)
RUN_BENCHMARK(c3, kDIV1, 1920, 1080)
RUN_BENCHMARK(c4, kDIV1, 1920, 1080)

#define RUN_OPENCV_FUNCTIONS(type, function)                                   \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c1, function)->Args({320, 240}); \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c3, function)->Args({320, 240}); \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c4, function)->Args({320, 240}); \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c1, function)->Args({640, 480}); \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c3, function)->Args({640, 480}); \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c4, function)->Args({640, 480}); \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c1, function)->                  \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c3, function)->                  \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Arith_opencv_ocl, type, c4, function)->                  \
                   Args({1920, 1080});

#define RUN_PPL_CV_FUNCTIONS(type, function)                                   \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c1, function)->Args({320, 240})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c3, function)->Args({320, 240})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c4, function)->Args({320, 240})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c1, function)->Args({640, 480})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c3, function)->Args({640, 480})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c4, function)->Args({640, 480})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c1, function)->Args({1920, 1080})-> \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c3, function)->Args({1920, 1080})-> \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Arith_ppl_ocl, type, c4, function)->Args({1920, 1080})-> \
                   UseManualTime()->Iterations(10);

// RUN_OPENCV_FUNCTIONS(uchar, kADD)
// RUN_OPENCV_FUNCTIONS(float, kADD)
// RUN_OPENCV_FUNCTIONS(uchar, kADDWEITHTED)
// RUN_OPENCV_FUNCTIONS(float, kADDWEITHTED)
// RUN_OPENCV_FUNCTIONS(uchar, kSUBTRACT)
// RUN_OPENCV_FUNCTIONS(float, kSUBTRACT)
// RUN_OPENCV_FUNCTIONS(uchar, kMUL0)
// RUN_OPENCV_FUNCTIONS(float, kMUL0)
// RUN_OPENCV_FUNCTIONS(uchar, kMUL1)
// RUN_OPENCV_FUNCTIONS(float, kMUL1)
// RUN_OPENCV_FUNCTIONS(uchar, kDIV0)
// RUN_OPENCV_FUNCTIONS(float, kDIV0)
// RUN_OPENCV_FUNCTIONS(uchar, kDIV1)
// RUN_OPENCV_FUNCTIONS(float, kDIV1)

// RUN_PPL_CV_FUNCTIONS(uchar, kADD)
// RUN_PPL_CV_FUNCTIONS(float, kADD)
// RUN_PPL_CV_FUNCTIONS(uchar, kADDWEITHTED)
// RUN_PPL_CV_FUNCTIONS(float, kADDWEITHTED)
// RUN_PPL_CV_FUNCTIONS(uchar, kSUBTRACT)
// RUN_PPL_CV_FUNCTIONS(float, kSUBTRACT)
// RUN_PPL_CV_FUNCTIONS(uchar, kMUL0)
// RUN_PPL_CV_FUNCTIONS(float, kMUL0)
// RUN_PPL_CV_FUNCTIONS(uchar, kMUL1)
// RUN_PPL_CV_FUNCTIONS(float, kMUL1)
// RUN_PPL_CV_FUNCTIONS(uchar, kDIV0)
// RUN_PPL_CV_FUNCTIONS(float, kDIV0)
// RUN_PPL_CV_FUNCTIONS(uchar, kDIV1)
// RUN_PPL_CV_FUNCTIONS(float, kDIV1)
