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

#include "ppl/cv/ocl/resize.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename T, int channels, ppl::cv::InterpolationType inter_type>
void BM_Resize_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());

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
    ppl::cv::ocl::Resize<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, dst_height, dst_width,
        dst.step / sizeof(T), gpu_dst, inter_type);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Resize<T, channels>(queue, src.rows, src.cols,
          src.step / sizeof(T), gpu_src, dst_height, dst_width,
          dst.step / sizeof(T), gpu_dst, inter_type);
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

template <typename T, int channels, ppl::cv::InterpolationType inter_type>
void BM_Resize_opencv_ocl(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());

  int cv_iterpolation;
  if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
    cv_iterpolation = cv::INTER_LINEAR;
  }
  else if (inter_type == ppl::cv::INTERPOLATION_NEAREST_POINT) {
    cv_iterpolation = cv::INTER_NEAREST;
  }
  else if (inter_type == ppl::cv::INTERPOLATION_AREA) {
    cv_iterpolation = cv::INTER_AREA;
  }
  else {
  }

  for (auto _ : state) {
    cv::resize(src, dst, cv::Size(dst_width, dst_height), 0, 0,
               cv_iterpolation);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(inter_type, src_width, src_height, dst_width, dst_height)\
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, uchar, c1, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, uchar, c1, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, uchar, c3, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, uchar, c3, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, uchar, c4, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, uchar, c4, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, float, c1, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, float, c1, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, float, c3, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, float, c3, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, float, c4, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, float, c4, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);

RUN_BENCHMARK(ppl::cv::INTERPOLATION_LINEAR, 320, 240, 640, 480)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_LINEAR, 640, 480, 1280, 960)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_LINEAR, 1280, 960, 640, 480)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_LINEAR, 640, 480, 320, 240)

RUN_BENCHMARK(ppl::cv::INTERPOLATION_NEAREST_POINT, 320, 240, 640, 480)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_NEAREST_POINT, 640, 480, 1280, 960)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_NEAREST_POINT, 1280, 960, 640, 480)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_NEAREST_POINT, 640, 480, 320, 240)

RUN_BENCHMARK(ppl::cv::INTERPOLATION_AREA, 320, 240, 640, 480)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_AREA, 640, 480, 1280, 960)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_AREA, 1280, 960, 640, 480)
RUN_BENCHMARK(ppl::cv::INTERPOLATION_AREA, 640, 480, 320, 240)

#define RUN_OPENCV_FUNCTIONS(inter_type, src_width, src_height, dst_width,     \
                             dst_height)                                       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, uchar, c1, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, uchar, c3, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, uchar, c4, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, float, c1, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, float, c3, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_ocl, float, c4, inter_type)->              \
                   Args({src_width, src_height, dst_width, dst_height});

#define RUN_PPL_CV_FUNCTIONS(inter_type, src_width, src_height, dst_width,     \
                             dst_height)                                       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, uchar, c1, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, uchar, c3, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, uchar, c4, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, float, c1, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, float, c3, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_ocl, float, c4, inter_type)->                 \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);

// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 320, 240, 640, 480)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 640, 480, 1280, 960)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 1280, 960, 640, 480)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 640, 480, 320, 240)

// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 320, 240,
//                      640, 480)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 640, 480,
//                      1280, 960)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 1280, 960,
//                      640, 480)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 640, 480,
//                      320, 240)

// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 320, 240, 640, 480)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 640, 480, 1280, 960)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 1280, 960, 640, 480)
// RUN_OPENCV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 640, 480, 320, 240)

// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 320, 240, 640, 480)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 640, 480, 1280, 960)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 1280, 960, 640, 480)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_LINEAR, 640, 480, 320, 240)

// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 320, 240,
//                      640, 480)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 640, 480,
//                      1280, 960)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 1280, 960,
//                      640, 480)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_NEAREST_POINT, 640, 480,
//                      320, 240)

// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 320, 240, 640, 480)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 640, 480, 1280, 960)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 1280, 960, 640, 480)
// RUN_PPL_CV_FUNCTIONS(ppl::cv::INTERPOLATION_AREA, 640, 480, 320, 240)
