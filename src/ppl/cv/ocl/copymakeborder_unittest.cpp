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

#include "ppl/cv/ocl/copymakeborder.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

using Parameters = std::tuple<int, int, ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringBorder(const Parameters& parameters) {
  std::ostringstream formatted;

  int top = std::get<0>(parameters);
  formatted << "TopBottom" << top << "_";

  int left = std::get<1>(parameters);
  formatted << "LeftRight" << left << "_";

  ppl::cv::BorderType border_type = std::get<2>(parameters);
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    formatted << "BORDER_CONSTANT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    formatted << "BORDER_REFLECT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_WRAP) {
    formatted << "BORDER_WRAP" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    formatted << "BORDER_REFLECT_101" << "_";
  }
  else {
  }

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclCopyMakeBorderTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclCopyMakeBorderTest() {
    const Parameters& parameters = GetParam();
    top         = std::get<0>(parameters);
    left        = std::get<1>(parameters);
    border_type = std::get<2>(parameters);
    size        = std::get<3>(parameters);

    bottom = top;
    right  = left;

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

  ~PplCvOclCopyMakeBorderTest() {
    ppl::common::ocl::shutDownKernelBinariesManager(
        ppl::common::ocl::BINARIES_RETRIEVE);
  }

  bool apply();

 private:
  int top;
  int bottom;
  int left;
  int right;
  ppl::cv::BorderType border_type;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclCopyMakeBorderTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst((size.height + top + bottom), (size.width + left + right),
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst((size.height + top + bottom), (size.width + left + right),
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes0 = src.rows * src.step;
  int dst_bytes0 = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes0,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int src_bytes1 = size.height * size.width * channels * sizeof(T);
  int dst_bytes1 = (size.height + top + bottom) * (size.width + left + right) *
                   channels * sizeof(T);
  cl_mem gpu_input = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                    src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_output = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,
                                    0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_WRAP) {
    cv_border = cv::BORDER_WRAP;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }
  cv::copyMakeBorder(src, cv_dst, top, bottom, left, right, cv_border);

  ppl::cv::ocl::CopyMakeBorder<T, channels>(queue, src.rows, src.cols,
      src.step / sizeof(T), gpu_src, dst.step / sizeof(T), gpu_dst, top, bottom,
      left, right, border_type);

  ppl::cv::ocl::CopyMakeBorder<T, channels>(queue, size.height, size.width,
      size.width * channels, gpu_input, dst.cols * channels, gpu_output, top,
      bottom, left, right, border_type);
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,
                                     0, dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

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
      dst.cols * channels * sizeof(T), epsilon);
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
using PplCvOclCopyMakeBorderTest ## T ## channels =                            \
        PplCvOclCopyMakeBorderTest<T, channels>;                               \
TEST_P(PplCvOclCopyMakeBorderTest ## T ## channels, Standard) {                \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclCopyMakeBorderTest ## T ## channels,  \
  ::testing::Combine(                                                          \
    ::testing::Values(0, 11, 17),                                              \
    ::testing::Values(0, 11, 17),                                              \
    ::testing::Values(ppl::cv::BORDER_CONSTANT, ppl::cv::BORDER_REPLICATE,     \
                      ppl::cv::BORDER_REFLECT, ppl::cv::BORDER_WRAP,           \
                      ppl::cv::BORDER_REFLECT_101),                            \
    ::testing::Values(cv::Size{11, 11}, cv::Size{25, 17},                      \
                      cv::Size{320, 240}, cv::Size{647, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclCopyMakeBorderTest ## T ## channels::ParamType>& info) {         \
    return convertToStringBorder(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
