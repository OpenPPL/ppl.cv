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

#include "ppl/cv/ocl/split.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

#define BASE 50

using Parameters = std::tuple<cv::Size>;
inline std::string splitToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclSplitToTest: public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclSplitToTest() {
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

  ~PplCvOclSplitToTest() {
  }

  bool apply();

 private:
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclSplitToTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst0(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst0(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, 1));

  cv::Mat dst1(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst1(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, 1));

  cv::Mat dst2(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst2(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, 1));

  cv::Mat dst3(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst3(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, 1));

  int src_bytes0 = src.rows * src.step;
  int dst_bytes0 = dst0.rows * dst0.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                   src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst0 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst1 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst2 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst3 = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes0,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int src_bytes1 = size.height * size.width * channels * sizeof(T);
  int dst_bytes1 = (size.height) * (size.width) * sizeof(T);
  cl_mem gpu_input = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                    src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output0 = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output1 = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output2 = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output3 = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,
                                    0, src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);
  
  if (channels == 3) {
    cv::Mat cv_dsts[3] = {cv_dst0, cv_dst1, cv_dst2};
    cv::split(src, cv_dsts);
    ppl::cv::ocl::Split3Channels<T>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, dst0.step / sizeof(T), 
        gpu_dst0, gpu_dst1, gpu_dst2);
    ppl::cv::ocl::Split3Channels<T>(queue, size.height, size.width,
        size.width * channels, gpu_input, size.width, 
        gpu_output0, gpu_output1, gpu_output2);
  }
  else if (channels == 4) {
    cv::Mat cv_dsts[4] = {cv_dst0, cv_dst1, cv_dst2, cv_dst3};
    cv::split(src, cv_dsts);
    ppl::cv::ocl::Split4Channels<T>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, dst0.step / sizeof(T), 
        gpu_dst0, gpu_dst1, gpu_dst2, gpu_dst3);
    ppl::cv::ocl::Split4Channels<T>(queue, size.height, size.width,
        size.width * channels, gpu_input, size.width, gpu_output0,
        gpu_output1, gpu_output2, gpu_output3);
  }

  error_code = clEnqueueReadBuffer(queue, gpu_dst0, CL_TRUE, 0, dst_bytes0,
                                   dst0.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  error_code = clEnqueueReadBuffer(queue, gpu_dst1, CL_TRUE, 0, dst_bytes0,
                                   dst1.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  error_code = clEnqueueReadBuffer(queue, gpu_dst2, CL_TRUE, 0, dst_bytes0,
                                   dst2.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);

  T* output0 = (T*)clEnqueueMapBuffer(queue, gpu_output0, CL_TRUE, CL_MAP_READ,
                                     0, dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  T* output1 = (T*)clEnqueueMapBuffer(queue, gpu_output1, CL_TRUE, CL_MAP_READ,
                                     0, dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  T* output2 = (T*)clEnqueueMapBuffer(queue, gpu_output2, CL_TRUE, CL_MAP_READ,
                                     0, dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  T* output3;
  if (channels == 4) {
    error_code = clEnqueueReadBuffer(queue, gpu_dst3, CL_TRUE, 0, dst_bytes0,
                                    dst3.data, 0, NULL, NULL);
    CHECK_ERROR(error_code, clEnqueueReadBuffer);
    output3 = (T*)clEnqueueMapBuffer(queue, gpu_output3, CL_TRUE, CL_MAP_READ,
                                      0, dst_bytes1, 0, NULL, NULL, &error_code);
    CHECK_ERROR(error_code, clEnqueueMapBuffer);
  }

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }

  bool identity0 = checkMatricesIdentity<T>(
      (const T*)cv_dst0.data, cv_dst0.rows, cv_dst0.cols, cv_dst0.channels(),
      cv_dst0.step, (const T*)dst0.data, dst0.step, epsilon);
  bool identity1 = checkMatricesIdentity<T>(
      (const T*)cv_dst1.data, cv_dst1.rows, cv_dst1.cols, cv_dst1.channels(),
      cv_dst1.step, (const T*)dst1.data, dst1.step, epsilon);
  bool identity2 = checkMatricesIdentity<T>(
      (const T*)cv_dst2.data, cv_dst2.rows, cv_dst2.cols, cv_dst2.channels(),
      cv_dst2.step, (const T*)dst2.data, dst2.step, epsilon);

  bool identity4 = checkMatricesIdentity<T>(
      (const T*)cv_dst0.data, cv_dst0.rows, cv_dst0.cols, cv_dst0.channels(),
      cv_dst0.step, output0, dst0.cols * sizeof(T), epsilon);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_output0, output0, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);
  bool identity5 = checkMatricesIdentity<T>(
      (const T*)cv_dst1.data, cv_dst1.rows, cv_dst1.cols, cv_dst1.channels(),
      cv_dst1.step, output1, dst1.cols * sizeof(T), epsilon);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_output1, output1, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);
  bool identity6 = checkMatricesIdentity<T>(
      (const T*)cv_dst2.data, cv_dst2.rows, cv_dst2.cols, cv_dst2.channels(),
      cv_dst2.step, output2, dst2.cols * sizeof(T), epsilon);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_output2, output2, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  bool identity3, identity7;
  if (channels == 4) {
    identity3 = checkMatricesIdentity<T>(
        (const T*)cv_dst3.data, cv_dst3.rows, cv_dst3.cols, cv_dst3.channels(),
        cv_dst3.step, (const T*)dst3.data, dst3.step, epsilon);
    identity7 = checkMatricesIdentity<T>(
        (const T*)cv_dst3.data, cv_dst3.rows, cv_dst3.cols, cv_dst3.channels(),
        cv_dst3.step, output3, dst3.cols * sizeof(T), epsilon);
    error_code =
        clEnqueueUnmapMemObject(queue, gpu_output3, output3, 0, NULL, NULL);
    CHECK_ERROR(error_code, clEnqueueUnmapMemObject);
  }

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst0);
  clReleaseMemObject(gpu_dst1);
  clReleaseMemObject(gpu_dst2);
  clReleaseMemObject(gpu_dst3);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output0);
  clReleaseMemObject(gpu_output1);
  clReleaseMemObject(gpu_output2);
  clReleaseMemObject(gpu_output3);

  if (channels == 3)
    return (identity0 && identity1 && identity2 && identity4 && identity5 &&
            identity6);
  if (channels == 4)
    return (identity0 && identity1 && identity2 && identity3 && identity4 &&
            identity5 && identity6 && identity7);
}

#define UNITTEST(T, channels)                                                  \
using PplCvOclSplitToTest ## T ## channels =                                   \
        PplCvOclSplitToTest<T, channels>;                                      \
TEST_P(PplCvOclSplitToTest ## T ## channels, Standard) {                       \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvOclSplitToTest ## T ## channels,                                        \
  ::testing::Combine(                                                          \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclSplitToTest ## T ## channels::ParamType>&                        \
        info) {                                                                \
    return splitToString(info.param);                                          \
  }                                                                            \
);

UNITTEST(uchar, 3)
UNITTEST(float, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 4)

