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

#include "ppl/cv/ocl/crop.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename T, int channels, int left, int top, int int_scale>
void BM_Crop_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  int src_width  = width * 2;
  int src_height = height * 2;
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  float scale = int_scale / 10.f;

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
    ppl::cv::ocl::Crop<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, dst.rows, dst.cols,
        dst.step / sizeof(T), gpu_dst, left, top, scale);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Crop<T, channels>(queue, src.rows, src.cols,
          src.step / sizeof(T), gpu_src, dst.rows, dst.cols,
          dst.step / sizeof(T), gpu_dst, left, top, scale);
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

template <typename T, int channels, int left, int top, int int_scale>
void BM_Crop_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  int src_width  = width * 2;
  int src_height = height * 2;
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  float scale = int_scale / 10.f;

  for (auto _ : state) {
    cv::Rect roi(left, top, width, height);
    cv::Mat croppedImage = src(roi);
    croppedImage.copyTo(dst);
    dst = dst * scale;
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, top, left, scale, width, height)              \
BENCHMARK_TEMPLATE(BM_Crop_opencv_ocl, uchar, channels, top, left,             \
                   scale)->Args({width, height});                              \
BENCHMARK_TEMPLATE(BM_Crop_ppl_ocl, uchar, channels, top, left, scale)->       \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Crop_opencv_ocl, float, channels, top, left,             \
                   scale)->Args({width, height});                              \
BENCHMARK_TEMPLATE(BM_Crop_ppl_ocl, float, channels, top, left, scale)->       \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_BENCHMARK0(c1, 16, 16, 10, 640, 480)
RUN_BENCHMARK0(c3, 16, 16, 10, 640, 480)
RUN_BENCHMARK0(c4, 16, 16, 10, 640, 480)
RUN_BENCHMARK0(c1, 16, 16, 10, 1920, 1080)
RUN_BENCHMARK0(c3, 16, 16, 10, 1920, 1080)
RUN_BENCHMARK0(c4, 16, 16, 10, 1920, 1080)

RUN_BENCHMARK0(c1, 16, 16, 15, 640, 480)
RUN_BENCHMARK0(c3, 16, 16, 15, 640, 480)
RUN_BENCHMARK0(c4, 16, 16, 15, 640, 480)
RUN_BENCHMARK0(c1, 16, 16, 15, 1920, 1080)
RUN_BENCHMARK0(c3, 16, 16, 15, 1920, 1080)
RUN_BENCHMARK0(c4, 16, 16, 15, 1920, 1080)