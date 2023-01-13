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

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/oclcommon.h"
#include "utility/infrastructure.h"

enum Scaling {
  kHalfSize,
  kSmallerSize,
  kSameSize,
  kBiggerSize,
  kDoubleSize,
};

using Parameters = std::tuple<Scaling, ppl::cv::InterpolationType, cv::Size>;
inline std::string convertToStringResize(const Parameters& parameters) {
  std::ostringstream formatted;

  Scaling scale = std::get<0>(parameters);
  if (scale == kHalfSize) {
    formatted << "HalfSize" << "_";
  }
  else if (scale == kSmallerSize) {
    formatted << "SmallerSize" << "_";
  }
  else if (scale == kSameSize) {
    formatted << "SameSize" << "_";
  }
  else if (scale == kBiggerSize) {
    formatted << "BiggerSize" << "_";
  }
  else if (scale == kDoubleSize) {
    formatted << "DoubleSize" << "_";
  }
  else {
  }

  ppl::cv::InterpolationType inter_type = std::get<1>(parameters);
  if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
    formatted << "InterLinear" << "_";
  }
  else if (inter_type == ppl::cv::INTERPOLATION_NEAREST_POINT) {
    formatted << "InterNearest" << "_";
  }
  else if (inter_type == ppl::cv::INTERPOLATION_AREA) {
    formatted << "InterArea" << "_";
  }
  else {
  }

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclResizeTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclResizeTest() {
    const Parameters& parameters = GetParam();
    scale      = std::get<0>(parameters);
    inter_type = std::get<1>(parameters);
    size       = std::get<2>(parameters);

    ppl::common::ocl::createSharedFrameChain(false);
    context = ppl::common::ocl::getSharedFrameChain()->getContext();
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();
  }

  ~PplCvOclResizeTest() {
  }

  bool apply();

 private:
  Scaling scale;
  ppl::cv::InterpolationType inter_type;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclResizeTest<T, channels>::apply() {
  float scale_coeff;
  if (scale == kHalfSize) {
    scale_coeff = 0.5f;
  }
  else if (scale == kSmallerSize) {
    scale_coeff = 0.7f;
  }
  else if (scale == kSameSize) {
    scale_coeff = 1.0f;
  }
  else if (scale == kBiggerSize) {
    scale_coeff = 1.4f;
  }
  else {  // scale == kDoubleSize
    scale_coeff = 2.0f;
  }
  int dst_height = size.height * scale_coeff;
  int dst_width  = size.width * scale_coeff;
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(dst_height, dst_width,
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

  int input_size  = size.height * size.width * channels * sizeof(T);
  int output_size = dst_height * dst_width * channels * sizeof(T);
  cl_mem gpu_input = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                    input_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_output = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     output_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,
                                    0, input_size, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  int cv_iterpolation = cv::INTER_LINEAR;
  if (inter_type == ppl::cv::INTERPOLATION_NEAREST_POINT) {
    cv_iterpolation = cv::INTER_NEAREST;
  }
  else if (inter_type == ppl::cv::INTERPOLATION_AREA) {
    cv_iterpolation = cv::INTER_AREA;
  }
  else {
  }
  cv::resize(src, cv_dst, cv::Size(dst_width, dst_height), 0, 0,
             cv_iterpolation);

  ppl::cv::ocl::Resize<T, channels>(queue, src.rows, src.cols,
      src.step / sizeof(T), gpu_src, dst_height, dst_width,
      dst.step / sizeof(T), gpu_dst, inter_type);
  ppl::cv::ocl::Resize<T, channels>(queue, size.height, size.width,
      size.width * channels, gpu_input, dst_height, dst_width,
      dst_width * channels, gpu_output, inter_type);
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,
                                     0, output_size, 0, NULL, NULL,
                                     &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_2F;
  }
  else {
    epsilon = EPSILON_E2;
  }
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data, dst.step,
      epsilon);
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,
      dst_width * channels * sizeof(T), epsilon);
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,
                                       NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvOclResizeTest ## T ## channels = PplCvOclResizeTest<T, channels>;   \
TEST_P(PplCvOclResizeTest ## T ## channels, Standard) {                        \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclResizeTest ## T ## channels,          \
  ::testing::Combine(                                                          \
    ::testing::Values(kHalfSize, kSmallerSize, kSameSize, kBiggerSize,         \
                      kDoubleSize),                                            \
    ::testing::Values(ppl::cv::INTERPOLATION_LINEAR,                           \
                      ppl::cv::INTERPOLATION_NEAREST_POINT,                    \
                      ppl::cv::INTERPOLATION_AREA),                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclResizeTest ## T ## channels::ParamType>& info) {                 \
    return convertToStringResize(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
