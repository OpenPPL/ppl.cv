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

#include "ppl/cv/ocl/rotate.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename T, int channels, int degree>
void BM_Rotate_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_height, dst_width;
  if (degree == 90) {
    dst_height = src_width;
    dst_width  = src_height;
  }
  else if (degree == 180) {
    dst_height = src_height;
    dst_width  = src_width;
  }
  else if (degree == 270) {
    dst_height = src_width;
    dst_width  = src_height;
  }
  else {
    return;
  }

  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

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
    ppl::cv::ocl::Rotate<T, channels>(queue, src.rows, src.cols,
      src.step / sizeof(T), gpu_src, dst.rows, dst.cols, dst.step / sizeof(T), gpu_dst, degree);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Rotate<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, dst.rows, dst.cols, dst.step / sizeof(T), gpu_dst, degree);
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

template <typename T, int channels, int degree>
void BM_Rotate_opencv_ocl(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_height, dst_width;
  cv::RotateFlags cv_rotate_flag;
  if (degree == 90) {
    dst_height = src_width;
    dst_width  = src_height;
    cv_rotate_flag = cv::ROTATE_90_CLOCKWISE;
  }
  else if (degree == 180) {
    dst_height = src_height;
    dst_width  = src_width;
    cv_rotate_flag = cv::ROTATE_180;
  }
  else if (degree == 270) {
    dst_height = src_width;
    dst_width  = src_height;
    cv_rotate_flag = cv::ROTATE_90_COUNTERCLOCKWISE;
  }
  else {
    return;
  }

  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());

  for (auto _ : state) {
    cv::rotate(src, dst, cv_rotate_flag);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK1(channels, degree, width, height)                        \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_ocl, uchar, channels, degree)->            \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_ocl, uchar, channels, degree)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_ocl, float, channels, degree)->            \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_ocl, float, channels, degree)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_BENCHMARK1(c1, 90, 640, 480)
RUN_BENCHMARK1(c3, 90, 640, 480)
RUN_BENCHMARK1(c4, 90, 640, 480)
RUN_BENCHMARK1(c1, 90, 1920, 1080)
RUN_BENCHMARK1(c3, 90, 1920, 1080)
RUN_BENCHMARK1(c4, 90, 1920, 1080)

RUN_BENCHMARK1(c1, 180, 640, 480)
RUN_BENCHMARK1(c3, 180, 640, 480)
RUN_BENCHMARK1(c4, 180, 640, 480)
RUN_BENCHMARK1(c1, 180, 1920, 1080)
RUN_BENCHMARK1(c3, 180, 1920, 1080)
RUN_BENCHMARK1(c4, 180, 1920, 1080)

RUN_BENCHMARK1(c1, 270, 640, 480)
RUN_BENCHMARK1(c3, 270, 640, 480)
RUN_BENCHMARK1(c4, 270, 640, 480)
RUN_BENCHMARK1(c1, 270, 1920, 1080)
RUN_BENCHMARK1(c3, 270, 1920, 1080)
RUN_BENCHMARK1(c4, 270, 1920, 1080)