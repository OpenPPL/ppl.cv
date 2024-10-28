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

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

#define BASE 50

using Parameters = std::tuple<cv::Size>;
inline std::string mergeToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclMergeToTest: public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclMergeToTest() {
    const Parameters& parameters = GetParam();
    size = std::get<0>(parameters);

    ppl::common::ocl::createSharedFrameChain(false);
    context = ppl::common::ocl::getSharedFrameChain()->getContext();
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();

    bool status = ppl::common::ocl::initializeKernelBinariesManager(
                      ppl::common::ocl::BINARIES_RETRIEVE);
    if (status) {
      ppl::common::ocl::FrameChain* frame_chain =
          ppl::common::ocl::getSharedFrameChain();
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);
    }
  }

  ~PplCvOclMergeToTest() {
  }

  bool apply();

 private:
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclMergeToTest<T, channels>::apply() {
  cv::Mat src0, src1, src2, src3;
  src0 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src1 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src2 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src3 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes0 = src0.rows * src0.step;
  int dst_bytes0 = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src0 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_src1 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_src2 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_src3 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src0, CL_FALSE, 0, src_bytes0,
                                    src0.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src1, CL_FALSE, 0, src_bytes0,
                                    src1.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src2, CL_FALSE, 0, src_bytes0,
                                    src2.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src3, CL_FALSE, 0, src_bytes0,
                                    src3.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int src_bytes1 = size.height * size.width * sizeof(T);
  int dst_bytes1 = (size.height) * (size.width) * channels * sizeof(T);
  cl_mem gpu_input0 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_input1 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_input2 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_input3 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  T* input0 = (T*)clEnqueueMapBuffer(queue, gpu_input0, CL_TRUE, CL_MAP_WRITE,
                                     0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  T* input1 = (T*)clEnqueueMapBuffer(queue, gpu_input1, CL_TRUE, CL_MAP_WRITE,
                                     0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  T* input2 = (T*)clEnqueueMapBuffer(queue, gpu_input2, CL_TRUE, CL_MAP_WRITE,
                                     0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  T* input3 = (T*)clEnqueueMapBuffer(queue, gpu_input3, CL_TRUE, CL_MAP_WRITE,
                                     0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  copyMatToArray(src0, input0);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_input0, input0, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  copyMatToArray(src1, input1);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_input1, input1, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  copyMatToArray(src2, input2);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_input2, input2, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  copyMatToArray(src3, input3);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_input3, input3, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  if (channels == 3) {
    cv::Mat srcs[3] = {src0, src1, src2};
    cv::merge(srcs, 3, cv_dst);
    ppl::cv::ocl::Merge3Channels<T>(queue, src0.rows, src0.cols,
                                    src0.step / sizeof(T), gpu_src0, gpu_src1,
                                    gpu_src2, dst.step / sizeof(T), gpu_dst);
    ppl::cv::ocl::Merge3Channels<T>(queue, size.height, size.width, size.width,
                                    gpu_input0, gpu_input1, gpu_input2,
                                    size.width * channels, gpu_output);
  }
  else {
    cv::Mat srcs[4] = {src0, src1, src2, src3};
    cv::merge(srcs, 4, cv_dst);
    ppl::cv::ocl::Merge4Channels<T>(
        queue, src0.rows, src0.cols, src0.step / sizeof(T), gpu_src0, gpu_src1,
        gpu_src2, gpu_src3, dst.step / sizeof(T), gpu_dst);
    ppl::cv::ocl::Merge4Channels<T>(
        queue, size.height, size.width, size.width, gpu_input0, gpu_input1,
        gpu_input2, gpu_input3, size.width * channels, gpu_output);
  }

  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);

  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ, 0,
                                     dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }

  bool identity0 = checkMatricesIdentity<T>(
      (const T*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, (const T*)dst.data, dst.step, epsilon);
  bool identity1 = checkMatricesIdentity<T>(
      (const T*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, output, dst.cols * channels * sizeof(T), epsilon);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  clReleaseMemObject(gpu_src0);
  clReleaseMemObject(gpu_src1);
  clReleaseMemObject(gpu_src2);
  clReleaseMemObject(gpu_src3);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input0);
  clReleaseMemObject(gpu_input1);
  clReleaseMemObject(gpu_input2);
  clReleaseMemObject(gpu_input3);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvOclMergeToTest ## T ## channels =                                   \
        PplCvOclMergeToTest<T, channels>;                                      \
TEST_P(PplCvOclMergeToTest ## T ## channels, Standard) {                       \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvOclMergeToTest ## T ## channels,                                        \
  ::testing::Combine(                                                          \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclMergeToTest ## T ## channels::ParamType>&                        \
        info) {                                                                \
    return mergeToString(info.param);                                          \
  }                                                                            \
);

UNITTEST(uchar, 3)
UNITTEST(float, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 4)