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

#include "ppl/cv/ocl/merge.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename T, int channels>
void BM_Merge_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src0, src1, src2, src3;
  src0 = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src1 = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src2 = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src3 = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes = src0.rows * src0.step;
  int dst_bytes = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src0 = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_src1 = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_src2 = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_src3 = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src0, CL_TRUE, 0, src_bytes,
                                    src0.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src1, CL_TRUE, 0, src_bytes,
                                    src1.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src2, CL_TRUE, 0, src_bytes,
                                    src2.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src3, CL_TRUE, 0, src_bytes,
                                    src3.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int iterations = 100;
  struct timeval start, end;

  if (channels == 3){
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Merge3Channels<T>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, gpu_src1, gpu_src2, 
          dst.step / sizeof(T), gpu_dst);
    }
    clFinish(queue);
    for (auto _ : state) {
      gettimeofday(&start, NULL);
      for (int i = 0; i < iterations; i++) {
        ppl::cv::ocl::Merge3Channels<T>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, gpu_src1, gpu_src2, 
            dst.step / sizeof(T), gpu_dst);
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
      ppl::cv::ocl::Merge4Channels<T>(queue, src0.rows, src0.cols,
          src0.step / sizeof(T), gpu_src0, gpu_src1, gpu_src2, gpu_src3, 
          dst.step / sizeof(T), gpu_dst);
    }
    clFinish(queue);
    for (auto _ : state) {
      gettimeofday(&start, NULL);
      for (int i = 0; i < iterations; i++) {
        ppl::cv::ocl::Merge4Channels<T>(queue, src0.rows, src0.cols,
            src0.step / sizeof(T), gpu_src0, gpu_src1, gpu_src2, gpu_src3, 
            dst.step / sizeof(T), gpu_dst);
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

  clReleaseMemObject(gpu_src0);
  clReleaseMemObject(gpu_src1);
  clReleaseMemObject(gpu_src2);
  clReleaseMemObject(gpu_src3);
  clReleaseMemObject(gpu_dst);
}

template <typename T, int channels>
void BM_Merge_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src0, src1, src2, src3;
  src0 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  src1 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  src2 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  src3 = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           1));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  cv::Mat srcs0[3] = {src0, src1, src2};
  cv::Mat srcs1[4] = {src0, src1, src2, src3};

  for (auto _ : state) {
    if (channels == 3) {
      cv::merge(srcs0, 3, dst);
    }
    else { // channels == 4
      cv::merge(srcs1, 4, dst);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(T, channels, width, height)                             \
BENCHMARK_TEMPLATE(BM_Merge_opencv_ocl, T, channels)->                         \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Merge_ppl_ocl, T, channels)->                            \
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

