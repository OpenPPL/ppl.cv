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

#include "ppl/cv/ocl/calchist.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename T, int channels, MaskType mask_type>
void BM_CalcHist_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask, dst;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));

  int src_bytes = src.rows * src.step;
  int mask_bytes = mask.rows * mask.step;
  int dst_bytes = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_mask = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  mask_bytes, NULL, &error_code);
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
      if (mask_type == kUnmasked) {
        ppl::cv::ocl::CalcHist(queue, src.rows, src.cols,
            src.step / sizeof(uchar), gpu_src, gpu_dst, mask.step / sizeof(uchar), nullptr);
      }
      else {
        ppl::cv::ocl::CalcHist(queue, src.rows, src.cols,
            src.step / sizeof(uchar), gpu_src, gpu_dst, mask.step / sizeof(uchar), gpu_mask);
      }
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
        if (mask_type == kUnmasked) {
          ppl::cv::ocl::CalcHist(queue, src.rows, src.cols,
              src.step / sizeof(uchar), gpu_src, gpu_dst, mask.step / sizeof(uchar), nullptr);
        }
        else {
          ppl::cv::ocl::CalcHist(queue, src.rows, src.cols,
              src.step / sizeof(uchar), gpu_src, gpu_dst, mask.step / sizeof(uchar), gpu_mask);
        }
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

template <typename T, int channels, MaskType mask_type>
void BM_CalcHist_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask, dst;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));

  int channel[] = {0};
  int hist_size = 256;
  float data_range[2] = {0, 256};
  const float* ranges[1] = {data_range};

  for (auto _ : state) {
    if (mask_type == kUnmasked) {
      cv::calcHist(&src, 1, channel, cv::Mat(), dst, 1, &hist_size, ranges,
                   true, false);
    }
    else {
      cv::calcHist(&src, 1, channel, mask, dst, 1, &hist_size, ranges, true,
                   false);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK1(channels, mask_type, width, height)                     \
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_ocl, uchar, channels, mask_type)->       \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_ocl, uchar, channels, mask_type)->          \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_BENCHMARK1(c1, kUnmasked, 320, 240)
RUN_BENCHMARK1(c1, kUnmasked, 640, 480)
RUN_BENCHMARK1(c1, kUnmasked, 1280, 720)
RUN_BENCHMARK1(c1, kUnmasked, 1920, 1080)
RUN_BENCHMARK1(c1, kMasked, 320, 240)
RUN_BENCHMARK1(c1, kMasked, 640, 480)
RUN_BENCHMARK1(c1, kMasked, 1280, 720)
RUN_BENCHMARK1(c1, kMasked, 1920, 1080)