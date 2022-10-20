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

// #include <iostream>  // debug

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/framechain.h"
#include "utility/infrastructure.hpp"

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

    ppl::common::ocl::FrameChain frame_chain;
    frame_chain.createDefaultOclFrame(false);
    context = frame_chain.getContext();
    queue   = frame_chain.getQueue();
  }

  ~PplCvOclAbsTest() {
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
  // std::cout << "src_bytes: " << src_bytes << std::endl;
  // std::cout << "dst_bytes: " << dst_bytes << std::endl;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context, CL_MEM_READ_ONLY, src_bytes, NULL,
                                  &error_code);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clCreateBuffer() failed with code: " << error_code;
  }
  cl_mem gpu_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_bytes, NULL,
                                  &error_code);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clCreateBuffer() failed with code: " << error_code;
  }
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,
                                    src.data, 0, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clEnqueueWriteBuffer() failed with code: " << error_code;
  }

  int data_size = size.height * size.width * channels * sizeof(T);
  T* input  = (T*)malloc(data_size);
  T* output = (T*)malloc(data_size);
  cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL,
                                    &error_code);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clCreateBuffer() failed with code: " << error_code;
  }
  cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL,
                                     &error_code);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clCreateBuffer() failed with code: " << error_code;
  }
  copyMatToArray(src, input);
  error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0, data_size,
                                    input, 0, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clEnqueueWriteBuffer() failed with code: " << error_code;
  }

  cv_dst = cv::abs(src);
  // std::cout << "before Abs()" << std::endl;
  ppl::cv::ocl::Abs<T, channels>(queue, src.rows, src.cols,
      src.step / sizeof(T), gpu_src, dst.step / sizeof(T), gpu_dst);
  // std::cout << "after Abs()" << std::endl;
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes,
                                   dst.data, 0, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clEnqueueReadBuffer() failed with code: " << error_code;
  }
  // std::cout << "after clEnqueueReadBuffer()" << std::endl;

  ppl::cv::ocl::Abs<T, channels>(queue, size.height, size.width,
      size.width * channels, gpu_input, size.width * channels, gpu_output);
  error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
                                   output, 0, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    LOG(ERROR) << "Call clEnqueueReadBuffer() failed with code: " << error_code;
  }

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);

  free(input);
  free(output);
  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

// #define UNITTEST(T, channels)                                                  \
// using PplCvOclAbsTest ## T ## channels = PplCvOclAbsTest<T, channels>;         \
// TEST_P(PplCvOclAbsTest ## T ## channels, Standard) {                           \
//   bool identity = this->apply();                                               \
//   EXPECT_TRUE(identity);                                                       \
// }                                                                              \
//                                                                                \
// INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclAbsTest ## T ## channels,             \
//   ::testing::Values(cv::Size{3, 3}),                \
//   [](const testing::TestParamInfo<                                             \
//       PplCvOclAbsTest ## T ## channels::ParamType>& info) {                    \
//     return convertToString(info.param);                                        \
//   }                                                                            \
// );

// UNITTEST(schar, 1)
// // UNITTEST(schar, 3)
// // UNITTEST(schar, 4)
// // UNITTEST(float, 1)
// // UNITTEST(float, 3)
// // UNITTEST(float, 4)

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
