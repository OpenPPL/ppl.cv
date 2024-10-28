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

#include "ppl/cv/ocl/integral.h"
#include "ppl/cv/ocl/use_memory_pool.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst>
void BM_Integral_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          1));
  cv::Mat dst(height + 1, width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, 1));

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

  size_t size_width = (width + 1) * sizeof(float);
  size_t ceiled_volume = ppl::cv::ocl::ceil2DVolume(size_width, (height + 1));
  ppl::cv::ocl::activateGpuMemoryPool(ceiled_volume);

  int iterations = 100;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::ocl::Integral<Tsrc, Tdst>(queue, src.rows, src.cols,
        src.step / sizeof(Tsrc), gpu_src, dst.rows, dst.cols,
        dst.step / sizeof(Tdst), gpu_dst);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::ocl::Integral<Tsrc, Tdst>(queue, src.rows, src.cols,
          src.step / sizeof(Tsrc), gpu_src, dst.rows, dst.cols,
          dst.step / sizeof(Tdst), gpu_dst);
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

template <typename Tsrc, typename Tdst>
void BM_Integral_opencv_ocl(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          1));
  cv::Mat dst(height + 1, width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, 1));

  for (auto _ : state) {
    cv::integral(src, dst, dst.depth());
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK1(width, height)                                          \
BENCHMARK_TEMPLATE(BM_Integral_opencv_ocl, uchar, int)->                       \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Integral_ppl_ocl, uchar, int)->Args({width, height})->   \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Integral_opencv_ocl, float, float)->                     \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Integral_ppl_ocl, float, float)->Args({width, height})-> \
                   UseManualTime()->Iterations(10); 

RUN_BENCHMARK1(320, 240)
RUN_BENCHMARK1(640, 480)
RUN_BENCHMARK1(1280, 720)
RUN_BENCHMARK1(1920, 1080)
