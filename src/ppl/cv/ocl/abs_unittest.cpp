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

#include "ppl/cv/ocl/abs.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

using Parameters = std::tuple<cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclAbsTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclAbsTest() {
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

  ~PplCvOclAbsTest() {
    ppl::common::ocl::shutDownKernelBinariesManager(
        ppl::common::ocl::BINARIES_RETRIEVE);
  }

  bool apply();

 private:
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclAbsTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));

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
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int data_size = size.height * size.width * channels * sizeof(T);
  T* input  = (T*)malloc(data_size);
  T* output = (T*)malloc(data_size);
  cl_mem gpu_input = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    data_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_output = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                     data_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  copyMatToArray(src, input);
  error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0, data_size,
                                    input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  cv_dst = cv::abs(src);
  ppl::cv::ocl::Abs<T, channels>(queue, src.rows, src.cols,
      src.step / sizeof(T), gpu_src, dst.step / sizeof(T), gpu_dst);
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);

  ppl::cv::ocl::Abs<T, channels>(queue, size.height, size.width,
      size.width * channels, gpu_input, size.width * channels, gpu_output);
  error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
                                   output, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data, dst.step,
      epsilon);
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,
      size.width * channels * sizeof(T), epsilon);

  free(input);
  free(output);
  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvOclAbsTest ## T ## channels = PplCvOclAbsTest<T, channels>;         \
TEST_P(PplCvOclAbsTest ## T ## channels, Standard) {                           \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclAbsTest ## T ## channels,             \
  ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                    \
                    cv::Size{1283, 720}, cv::Size{1934, 1080},                 \
                    cv::Size{320, 240}, cv::Size{640, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1920, 1080}),                \
  [](const testing::TestParamInfo<                                             \
      PplCvOclAbsTest ## T ## channels::ParamType>& info) {                    \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(schar, 1)
UNITTEST(schar, 3)
UNITTEST(schar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
