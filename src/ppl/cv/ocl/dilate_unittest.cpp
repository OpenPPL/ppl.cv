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

#include "ppl/cv/ocl/dilate.h"
#include "ppl/cv/ocl/erode.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

enum Functions {
  kFullyMaskedDilate,
  kPartiallyMaskedDilate,
  kFullyMaskedErode,
  kPartiallyMaskedErode,
};

using Parameters = std::tuple<Functions, ppl::cv::BorderType, int, cv::Size>;
inline std::string convertToStringDilate(const Parameters& parameters) {
  std::ostringstream formatted;

  Functions function = std::get<0>(parameters);
  if (function == kFullyMaskedDilate) {
    formatted << "FullyMaskedDilate" << "_";
  }
  else if (function == kPartiallyMaskedDilate) {
    formatted << "PartiallyMaskedDilate" << "_";
  }
  else if (function == kFullyMaskedErode) {
    formatted << "FullyMaskedErode" << "_";
  }
  else if (function == kPartiallyMaskedErode) {
    formatted << "PartiallyMaskedErode" << "_";
  }
  else {
  }

  ppl::cv::BorderType border_type = std::get<1>(parameters);
  if (border_type == ppl::cv::BORDER_DEFAULT) {
    formatted << "BORDER_DEFAULT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_CONSTANT) {
    formatted << "BORDER_CONSTANT" << "_";
  }
  else {
  }

  int ksize = std::get<2>(parameters);
  formatted << "Ksize" << ksize << "_";

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclDilateTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclDilateTest() {
    const Parameters& parameters = GetParam();
    function    = std::get<0>(parameters);
    border_type = std::get<1>(parameters);
    ksize       = std::get<2>(parameters);
    size        = std::get<3>(parameters);

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

  ~PplCvOclDilateTest() {
    ppl::common::ocl::shutDownKernelBinariesManager(
        ppl::common::ocl::BINARIES_RETRIEVE);
  }

  bool apply();

 private:
  Functions function;
  ppl::cv::BorderType border_type;
  int ksize;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclDilateTest<T, channels>::apply() {
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
  cl_mem gpu_input = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                    data_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_output = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     data_size, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,
                                    0, data_size, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  cv::Size kSize(ksize, ksize);
  cv::Mat kernel0 = cv::getStructuringElement(cv::MORPH_RECT, kSize);
  cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, kSize);
  uchar* mask = (uchar*)malloc(ksize * ksize * sizeof(uchar));
  int index = 0;
  for (int i = 0; i < ksize; i++) {
    const uchar* data = kernel1.ptr<const uchar>(i);
    for (int j = 0; j < ksize; j++) {
      mask[index++] = data[j];
    }
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_DEFAULT) {
    cv_border = cv::BORDER_DEFAULT;
  }
  else if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else {
  }

  int constant_border;
  if (function == kFullyMaskedDilate) {
    constant_border = 253;
    cv::Scalar border(constant_border, constant_border, constant_border,
                      constant_border);
    cv::dilate(src, cv_dst, kernel0, cv::Point(-1, -1), 1, cv_border,
               border);
    ppl::cv::ocl::Dilate<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, ksize, ksize, nullptr,
        dst.step / sizeof(T), gpu_dst, border_type, constant_border);
    ppl::cv::ocl::Dilate<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input, ksize, ksize, nullptr,
        size.width * channels, gpu_output, border_type, constant_border);
  }
  else if (function == kPartiallyMaskedDilate) {
    constant_border = 253;
    cv::Scalar border(constant_border, constant_border, constant_border,
                      constant_border);
    cv::dilate(src, cv_dst, kernel1, cv::Point(-1, -1), 1, cv_border,
               border);
    ppl::cv::ocl::Dilate<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, ksize, ksize, mask,
        dst.step / sizeof(T), gpu_dst, border_type, constant_border);
    ppl::cv::ocl::Dilate<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input, ksize, ksize, mask,
        size.width * channels, gpu_output, border_type, constant_border);
  }
  else if (function == kFullyMaskedErode) {
    constant_border = 1;
    cv::Scalar border(constant_border, constant_border, constant_border,
                      constant_border);
    cv::erode(src, cv_dst, kernel0, cv::Point(-1, -1), 1, cv_border,
              border);
    ppl::cv::ocl::Erode<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, ksize, ksize, nullptr,
        dst.step / sizeof(T), gpu_dst, border_type, constant_border);
    ppl::cv::ocl::Erode<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input, ksize, ksize, nullptr,
        size.width * channels, gpu_output, border_type, constant_border);
  }
  else {
    constant_border = 1;
    cv::Scalar border(constant_border, constant_border, constant_border,
                      constant_border);
    cv::erode(src, cv_dst, kernel1, cv::Point(-1, -1), 1, cv_border,
              border);
    ppl::cv::ocl::Erode<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, ksize, ksize, mask,
        dst.step / sizeof(T), gpu_dst, border_type, constant_border);
    ppl::cv::ocl::Erode<T, channels>(queue, size.height, size.width,
        size.width * channels, gpu_input, ksize, ksize, mask,
        size.width * channels, gpu_output, border_type, constant_border);
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
    epsilon = EPSILON_E6;
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

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvOclDilateTest ## T ## channels = PplCvOclDilateTest<T, channels>;   \
TEST_P(PplCvOclDilateTest ## T ## channels, Standard) {                        \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclDilateTest ## T ## channels,          \
  ::testing::Combine(                                                          \
    ::testing::Values(kFullyMaskedDilate, kPartiallyMaskedDilate,              \
                      kFullyMaskedErode, kPartiallyMaskedErode),               \
    ::testing::Values(ppl::cv::BORDER_DEFAULT, ppl::cv::BORDER_CONSTANT),      \
    ::testing::Values(1, 3, 5, 7, 11, 15),                                     \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclDilateTest ## T ## channels::ParamType>& info) {                 \
    return convertToStringDilate(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
