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

#include "ppl/cv/ocl/adaptivethreshold.h"
#include "ppl/cv/ocl/use_memory_pool.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <int ksize, int adaptive_method>
void BM_AdaptiveThreshold_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<uchar>::depth, 1));

  float max_value = 155.f;
  float delta = 10.f;
  int threshold_type = THRESH_BINARY;
  BorderType border_type = BORDER_REPLICATE;

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

  size_t size_width = width * sizeof(float);
  size_t ceiled_volume = ppl::cv::ocl::ceil2DVolume(size_width, height);
  ppl::cv::ocl::activateGpuMemoryPool(ceiled_volume + ppl::cv::ocl::ceil1DVolume(ksize * sizeof(float)));

  int iterations = 100;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::ocl::AdaptiveThreshold(
        queue, src.rows, src.cols, src.step / sizeof(uchar), gpu_src, dst.step / sizeof(uchar), gpu_dst, 
        max_value, adaptive_method, threshold_type, ksize, delta, border_type);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::AdaptiveThreshold(
          queue, src.rows, src.cols, src.step / sizeof(uchar), gpu_src, dst.step / sizeof(uchar), gpu_dst, 
          max_value, adaptive_method, threshold_type, ksize, delta, border_type);
    }
    clFinish(queue);
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  ppl::cv::ocl::shutDownGpuMemoryPool();
  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
}

template <int ksize, int adaptive_method>
void BM_AdaptiveThreshold_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<uchar>::depth, 1));

  float max_value = 155.f;
  float delta = 10.f;
  int threshold_type = THRESH_BINARY;

  cv::AdaptiveThresholdTypes cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
  if (adaptive_method == ADAPTIVE_THRESH_MEAN_C) {
    cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
  }
  else if (adaptive_method == ADAPTIVE_THRESH_GAUSSIAN_C) {
    cv_adaptive_method = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
  }
  else {
  }

  cv::ThresholdTypes cv_threshold_type = cv::THRESH_BINARY;
  if (threshold_type == THRESH_BINARY) {
    cv_threshold_type = cv::THRESH_BINARY;
  }
  else if (threshold_type == THRESH_BINARY_INV) {
    cv_threshold_type = cv::THRESH_BINARY_INV;
  }

  for (auto _ : state) {
    cv::adaptiveThreshold(src, dst, max_value, cv_adaptive_method,
                          cv_threshold_type, ksize, delta);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(ksize, adaptive_method, width, height)                   \
BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_opencv_ocl, ksize,                     \
                   adaptive_method)->Args({width, height});                    \
BENCHMARK_TEMPLATE(BM_AdaptiveThreshold_ppl_ocl, ksize, adaptive_method)->     \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_BENCHMARK(3, ADAPTIVE_THRESH_MEAN_C, 640, 480)
RUN_BENCHMARK(3, ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
RUN_BENCHMARK(7, ADAPTIVE_THRESH_MEAN_C, 640, 480)
RUN_BENCHMARK(7, ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
RUN_BENCHMARK(13, ADAPTIVE_THRESH_MEAN_C, 640, 480)
RUN_BENCHMARK(13, ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
RUN_BENCHMARK(25, ADAPTIVE_THRESH_MEAN_C, 640, 480)
RUN_BENCHMARK(25, ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
RUN_BENCHMARK(31, ADAPTIVE_THRESH_MEAN_C, 640, 480)
RUN_BENCHMARK(31, ADAPTIVE_THRESH_MEAN_C, 1920, 1080)
RUN_BENCHMARK(43, ADAPTIVE_THRESH_MEAN_C, 640, 480)
RUN_BENCHMARK(43, ADAPTIVE_THRESH_MEAN_C, 1920, 1080)

RUN_BENCHMARK(3, ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
RUN_BENCHMARK(3, ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
RUN_BENCHMARK(7, ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
RUN_BENCHMARK(7, ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
RUN_BENCHMARK(13, ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
RUN_BENCHMARK(13, ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
RUN_BENCHMARK(25, ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
RUN_BENCHMARK(25, ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
RUN_BENCHMARK(31, ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
RUN_BENCHMARK(31, ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
RUN_BENCHMARK(43, ADAPTIVE_THRESH_GAUSSIAN_C, 640, 480)
RUN_BENCHMARK(43, ADAPTIVE_THRESH_GAUSSIAN_C, 1920, 1080)
