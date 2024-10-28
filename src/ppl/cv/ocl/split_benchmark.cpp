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

#include "ppl/cv/ocl/split.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename T, int channels>
void BM_Split_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst0(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst1(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst2(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst3(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));

  int src_bytes = src.rows * src.step;
  int dst_bytes = dst0.rows * dst0.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                   src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst0 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst1 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst2 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst3 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int iterations = 100;
  struct timeval start, end;

  if (channels == 3){
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Split3Channels<T>(queue, src.rows, src.cols,
          src.step / sizeof(T), gpu_src, dst0.step / sizeof(T), 
          gpu_dst0, gpu_dst1, gpu_dst2);
    }
    clFinish(queue);
    for (auto _ : state) {
      gettimeofday(&start, NULL);
      for (int i = 0; i < iterations; i++) {
        ppl::cv::ocl::Split3Channels<T>(queue, src.rows, src.cols,
            src.step / sizeof(T), gpu_src, dst0.step / sizeof(T), 
            gpu_dst0, gpu_dst1, gpu_dst2);
      }
      clFinish(queue);
      gettimeofday(&end, NULL);
      int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                  (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
      state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
  } 
  else {
  if (channels == 4){
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Split4Channels<T>(queue, src.rows, src.cols,
          src.step / sizeof(T), gpu_src, dst0.step / sizeof(T), 
          gpu_dst0, gpu_dst1, gpu_dst2, gpu_dst3);
    }
    clFinish(queue);
    for (auto _ : state) {
      gettimeofday(&start, NULL);
      for (int i = 0; i < iterations; i++) {
        ppl::cv::ocl::Split4Channels<T>(queue, src.rows, src.cols,
            src.step / sizeof(T), gpu_src, dst0.step / sizeof(T), 
            gpu_dst0, gpu_dst1, gpu_dst2, gpu_dst3);
      }
      clFinish(queue);
      gettimeofday(&end, NULL);
      int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                  (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
      state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
  }
  }

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst0);
  clReleaseMemObject(gpu_dst1);
  clReleaseMemObject(gpu_dst2);
  clReleaseMemObject(gpu_dst3);
}

template <typename T, int channels>
void BM_Split_opencv_ocl(benchmark::State &state) {
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

#define RUN_BENCHMARK0(T, channels, width, height)                             \
BENCHMARK_TEMPLATE(BM_Split_opencv_ocl, T, channels)->                         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Split_ppl_ocl, T, channels)->                            \
                   Args({width, height})->UseManualTime()->Iterations(10);     

RUN_BENCHMARK0(uchar, c3, 320, 240)
RUN_BENCHMARK0(float, c3, 320, 240)
RUN_BENCHMARK0(uchar, c4, 320, 240)
RUN_BENCHMARK0(float, c4, 320, 240)

RUN_BENCHMARK0(uchar, c3, 640, 480)
RUN_BENCHMARK0(float, c3, 640, 480)
RUN_BENCHMARK0(uchar, c4, 640, 480)
RUN_BENCHMARK0(float, c4, 640, 480)

RUN_BENCHMARK0(uchar, c3, 1280, 720)
RUN_BENCHMARK0(float, c3, 1280, 720)
RUN_BENCHMARK0(uchar, c4, 1280, 720)
RUN_BENCHMARK0(float, c4, 1280, 720)

RUN_BENCHMARK0(uchar, c3, 1920, 1080)
RUN_BENCHMARK0(float, c3, 1920, 1080)
RUN_BENCHMARK0(uchar, c4, 1920, 1080)
RUN_BENCHMARK0(float, c4, 1920, 1080)

