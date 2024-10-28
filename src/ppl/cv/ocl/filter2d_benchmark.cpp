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

#include "ppl/cv/ocl/filter2d.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst, int channels, int ksize,
          BorderType border_type>
void BM_Filter2D_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, kernel;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel = createSourceImage(1, ksize * ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  float delta = 0.f;

  int src_bytes = src.rows * src.step;
  int kernel_bytes = kernel.rows * kernel.step;
  int dst_bytes = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_kernel = clCreateBuffer(context,
                                     CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                     kernel_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_TRUE, 0, src_bytes,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_kernel, CL_TRUE, 0, kernel_bytes,
                                    kernel.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int iterations = 100;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::ocl::Filter2D<Tsrc, channels>(
        queue, src.rows, src.cols, src.step / sizeof(Tsrc), gpu_src, ksize,
        gpu_kernel, dst.step / sizeof(Tsrc), gpu_dst, delta,
        border_type);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Filter2D<Tsrc, channels>(
          queue, src.rows, src.cols, src.step / sizeof(Tsrc), gpu_src, ksize,
          gpu_kernel, dst.step / sizeof(Tsrc), gpu_dst, delta,
          border_type);
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

template <typename Tsrc, typename Tdst, int channels, int ksize,
          BorderType border_type>
void BM_Filter2D_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, kernel;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel = createSourceImage(1, ksize * ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

  cv::BorderTypes border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_REPLICATE) {
    border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_REFLECT) {
    border = cv::BORDER_REFLECT;
  }
  else if (border_type == BORDER_REFLECT_101) {
    border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  for (auto _ : state) {
    cv::filter2D(src, dst, dst.depth(), kernel, cv::Point(-1, -1), 0,
                 border);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK1(src_type, dst_type, ksize, border_type, width, height)  \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_ocl, src_type, dst_type, c1,             \
                   ksize, border_type)->Args({width, height});                 \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_ocl, src_type, dst_type, c1, ksize,         \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_ocl, src_type, dst_type, c3,             \
                   ksize, border_type)->Args({width, height});                 \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_ocl, src_type, dst_type, c3, ksize,         \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Filter2D_opencv_ocl, src_type, dst_type, c4,             \
                   ksize, border_type)->Args({width, height});                 \
BENCHMARK_TEMPLATE(BM_Filter2D_ppl_ocl, src_type, dst_type, c4, ksize,         \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);

RUN_BENCHMARK1(uchar, uchar, 5, BORDER_REPLICATE, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 5, BORDER_REFLECT, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 5, BORDER_REFLECT_101, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 17, BORDER_REPLICATE, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 17, BORDER_REFLECT, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 17, BORDER_REFLECT_101, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 31, BORDER_REPLICATE, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 31, BORDER_REFLECT, 640, 480)
RUN_BENCHMARK1(uchar, uchar, 31, BORDER_REFLECT_101, 640, 480)

RUN_BENCHMARK1(float, float, 5, BORDER_REPLICATE, 640, 480)
RUN_BENCHMARK1(float, float, 5, BORDER_REFLECT, 640, 480)
RUN_BENCHMARK1(float, float, 5, BORDER_REFLECT_101, 640, 480)
RUN_BENCHMARK1(float, float, 17, BORDER_REPLICATE, 640, 480)
RUN_BENCHMARK1(float, float, 17, BORDER_REFLECT, 640, 480)
RUN_BENCHMARK1(float, float, 17, BORDER_REFLECT_101, 640, 480)
RUN_BENCHMARK1(float, float, 31, BORDER_REPLICATE, 640, 480)
RUN_BENCHMARK1(float, float, 31, BORDER_REFLECT, 640, 480)
RUN_BENCHMARK1(float, float, 31, BORDER_REFLECT_101, 640, 480)
