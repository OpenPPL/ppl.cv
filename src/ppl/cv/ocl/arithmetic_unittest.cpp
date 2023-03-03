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

#include "ppl/cv/ocl/arithmetic.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

enum ArithFunctions {
  kADD,
  kADDWEITHTED,
  kSUBTRACT,
  kMUL0,
  kMUL1,
  kDIV0,
  kDIV1,
};

using Parameters = std::tuple<ArithFunctions, cv::Size>;
inline std::string convertToStringArith(const Parameters& parameters) {
  std::ostringstream formatted;

  ArithFunctions function = std::get<0>(parameters);
  if (function == kADD) {
    formatted << "Add" << "_";
  }
  else if (function == kADDWEITHTED) {
    formatted << "AddWeighted" << "_";
  }
  else if (function == kSUBTRACT) {
    formatted << "Subtract" << "_";
  }
  else if (function == kMUL0) {
    formatted << "Mul0" << "_";
  }
  else if (function == kMUL1) {
    formatted << "Mul1" << "_";
  }
  else if (function == kDIV0) {
    formatted << "Div0" << "_";
  }
  else if (function == kDIV1) {
    formatted << "Div1" << "_";
  }
  else {
  }

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclArithmeticTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclArithmeticTest() {
    const Parameters& parameters = GetParam();
    function = std::get<0>(parameters);
    size     = std::get<1>(parameters);

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

  ~PplCvOclArithmeticTest() {
    ppl::common::ocl::shutDownKernelBinariesManager(
        ppl::common::ocl::BINARIES_RETRIEVE);
  }

  bool apply();

 private:
  ArithFunctions function;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclArithmeticTest<T, channels>::apply() {
  cv::Mat src0, src1, dst, cv_dst;
  src0 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  src1 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  src0.copyTo(dst);
  src0.copyTo(cv_dst);

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
  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src0, CL_FALSE, 0, src_bytes,
                                    src0.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src1, CL_FALSE, 0, src_bytes,
                                    src1.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int data_size = size.height * size.width * channels * sizeof(T);
  cl_mem gpu_input0 = clCreateBuffer(context,
                                     CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     data_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_input1 = clCreateBuffer(context,
                                     CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     data_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_output = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     data_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  T* input0 = (T*)clEnqueueMapBuffer(queue, gpu_input0, CL_TRUE, CL_MAP_WRITE,
                                     0, data_size, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  T* input1 = (T*)clEnqueueMapBuffer(queue, gpu_input1, CL_TRUE, CL_MAP_WRITE,
                                     0, data_size, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(src0, input0);
  copyMatToArray(src1, input1);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input0, input0, 0, NULL,
                                       NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input1, input1, 0, NULL,
                                       NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  if (function == kADD) {
    cv::add(src0, src1, cv_dst);

    ppl::cv::ocl::Add<T, channels>(queue, src0.rows, src0.cols,
        src0.step / sizeof(T), gpu_src0, src1.step / sizeof(T), gpu_src1,
        dst.step / sizeof(T), gpu_dst);

    ppl::cv::ocl::Add<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output);
  }
  else if (function == kADDWEITHTED) {
    float alpha = 0.1f;
    float beta  = 0.2f;
    float gamma = 0.3f;
    cv::addWeighted(src0, alpha, src1, beta, gamma, cv_dst);

    ppl::cv::ocl::AddWeighted<T, channels>(queue, src0.rows, src0.cols,
        src0.step / sizeof(T), gpu_src0, alpha, src1.step / sizeof(T), gpu_src1,
        beta, gamma, dst.step / sizeof(T), gpu_dst);
    ppl::cv::ocl::AddWeighted<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input0, alpha, size.width * channels,
        gpu_input1, beta, gamma, size.width * channels, gpu_output);
  }
  else if (function == kSUBTRACT) {
    cv::subtract(src0, src1, cv_dst);

    ppl::cv::ocl::Subtract<T, channels>(queue, src0.rows, src0.cols,
        src0.step / sizeof(T), gpu_src0, src1.step / sizeof(T), gpu_src1,
        dst.step / sizeof(T), gpu_dst);

    ppl::cv::ocl::Subtract<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output);
  }
  else if (function == kMUL0) {
    cv::multiply(src0, src1, cv_dst);

    ppl::cv::ocl::Mul<T, channels>(queue, src0.rows, src0.cols,
        src0.step / sizeof(T), gpu_src0, src1.step / sizeof(T), gpu_src1,
        dst.step / sizeof(T), gpu_dst);

    ppl::cv::ocl::Mul<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output);
  }
  else if (function == kMUL1) {
    float scale = 0.1f;
    cv::multiply(src0, src1, cv_dst, scale);

    ppl::cv::ocl::Mul<T, channels>(queue, src0.rows, src0.cols,
        src0.step / sizeof(T), gpu_src0, src1.step / sizeof(T), gpu_src1,
        dst.step / sizeof(T), gpu_dst, scale);

    ppl::cv::ocl::Mul<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output, scale);
  }
  else if (function == kDIV0) {
    cv::divide(src0, src1, cv_dst);

    ppl::cv::ocl::Div<T, channels>(queue, src0.rows, src0.cols,
        src0.step / sizeof(T), gpu_src0, src1.step / sizeof(T), gpu_src1,
        dst.step / sizeof(T), gpu_dst);

    ppl::cv::ocl::Div<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output);
  }
  else if (function == kDIV1) {
    float scale = 0.1f;
    cv::divide(src0, src1, cv_dst, scale);

    ppl::cv::ocl::Div<T, channels>(queue, src0.rows, src0.cols,
        src0.step / sizeof(T), gpu_src0, src1.step / sizeof(T), gpu_src1,
        dst.step / sizeof(T), gpu_dst, scale);

    ppl::cv::ocl::Div<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output, scale);
  }
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,
                                     0, data_size, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    if (function == kDIV0 || function == kDIV1) {
      epsilon = EPSILON_E3;
    }
    else {
      epsilon = EPSILON_E6;
    }
  }
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data, dst.step,
      epsilon);
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,
      size.width * channels * sizeof(T), epsilon);
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,
                                       NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  clReleaseMemObject(gpu_src0);
  clReleaseMemObject(gpu_src1);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input0);
  clReleaseMemObject(gpu_input1);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvOclArithmeticTest ## T ## channels =                                \
        PplCvOclArithmeticTest<T, channels>;                                   \
TEST_P(PplCvOclArithmeticTest ## T ## channels, Standard) {                    \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclArithmeticTest ## T ## channels,      \
  ::testing::Combine(                                                          \
    ::testing::Values(kADD, kADDWEITHTED, kSUBTRACT, kMUL0, kMUL1, kDIV0,      \
                      kDIV1),                                                  \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclArithmeticTest ## T ## channels::ParamType>& info) {             \
    return convertToStringArith(info.param);                                   \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
