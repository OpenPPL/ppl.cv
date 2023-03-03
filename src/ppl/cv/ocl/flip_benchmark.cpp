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

#include "ppl/cv/ocl/flip.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename T, int channels, int flip_code>
void BM_Flip_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes = src.rows * src.step;
  int dst_bytes = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_TRUE, 0, src_bytes,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int iterations = 100;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::ocl::Flip<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, dst.step / sizeof(T), gpu_dst,
        flip_code);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Flip<T, channels>(queue, src.rows, src.cols,
          src.step / sizeof(T), gpu_src, dst.step / sizeof(T), gpu_dst,
          flip_code);
    }
    clFinish(queue);
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
}

template <typename T, int channels, int flip_code>
void BM_Flip_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(src.rows, src.cols, src.type());

  for (auto _ : state) {
    cv::flip(src, dst, flip_code);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, flip_code)                                     \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, uchar, channels, flip_code)->           \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, uchar, channels, flip_code)->              \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, float, channels, flip_code)->           \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, float, channels, flip_code)->              \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, uchar, channels, flip_code)->           \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, uchar, channels, flip_code)->              \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, float, channels, flip_code)->           \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, float, channels, flip_code)->              \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, uchar, channels, flip_code)->           \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, uchar, channels, flip_code)->              \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, float, channels, flip_code)->           \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, float, channels, flip_code)->              \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, uchar, channels, flip_code)->           \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, uchar, channels, flip_code)->              \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, float, channels, flip_code)->           \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, float, channels, flip_code)->              \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_BENCHMARK(c1, 0)
RUN_BENCHMARK(c3, 0)
RUN_BENCHMARK(c4, 0)

RUN_BENCHMARK(c1, 1)
RUN_BENCHMARK(c3, 1)
RUN_BENCHMARK(c4, 1)

RUN_BENCHMARK(c1, -1)
RUN_BENCHMARK(c3, -1)
RUN_BENCHMARK(c4, -1)

#define RUN_OPENCV_FUNCTIONS(type, flip_code)                                  \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c1, flip_code)->                  \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c3, flip_code)->                  \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c4, flip_code)->                  \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c1, flip_code)->                  \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c3, flip_code)->                  \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c4, flip_code)->                  \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c1, flip_code)->                  \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c3, flip_code)->                  \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c4, flip_code)->                  \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c1, flip_code)->                  \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c3, flip_code)->                  \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Flip_opencv_ocl, type, c4, flip_code)->                  \
                   Args({1920, 1080});

#define RUN_PPL_CV_FUNCTIONS(type, flip_code)                                  \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c1, flip_code)->                     \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c3, flip_code)->                     \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c4, flip_code)->                     \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c1, flip_code)->                     \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c3, flip_code)->                     \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c4, flip_code)->                     \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c1, flip_code)->                     \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c3, flip_code)->                     \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c4, flip_code)->                     \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c1, flip_code)->                     \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c3, flip_code)->                     \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Flip_ppl_ocl, type, c4, flip_code)->                     \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

// RUN_OPENCV_FUNCTIONS(uchar, 0)
// RUN_OPENCV_FUNCTIONS(uchar, 1)
// RUN_OPENCV_FUNCTIONS(uchar, -1)
// RUN_OPENCV_FUNCTIONS(float, 0)
// RUN_OPENCV_FUNCTIONS(float, 1)
// RUN_OPENCV_FUNCTIONS(float, -1)

// RUN_PPL_CV_FUNCTIONS(uchar, 0)
// RUN_PPL_CV_FUNCTIONS(uchar, 1)
// RUN_PPL_CV_FUNCTIONS(uchar, -1)
// RUN_PPL_CV_FUNCTIONS(float, 0)
// RUN_PPL_CV_FUNCTIONS(float, 1)
// RUN_PPL_CV_FUNCTIONS(float, -1)
