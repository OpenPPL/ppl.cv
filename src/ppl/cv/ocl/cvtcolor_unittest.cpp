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

#include "ppl/cv/ocl/cvtcolor.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/oclcommon.h"
#include "utility/infrastructure.h"

using Parameters = std::tuple<cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

#define UNITTEST_CLASS_DECLARATION(Function)                                   \
template <typename T, int src_channels, int dst_channels>                      \
class PplCvOclCvtColor ## Function :                                           \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvOclCvtColor ## Function() {                                             \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
                                                                               \
    ppl::common::ocl::createSharedFrameChain(false);                           \
    context = ppl::common::ocl::getSharedFrameChain()->getContext();           \
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();             \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
  cl_context context;                                                          \
  cl_command_queue queue;                                                      \
};                                                                             \
                                                                               \
template <typename T, int src_channels, int dst_channels>                      \
bool PplCvOclCvtColor ## Function<T, src_channels, dst_channels>::apply() {    \
  cv::Mat src;                                                                 \
  src = createSourceImage(size.height, size.width,                             \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(size.height, size.width,                                         \
              CV_MAKETYPE(cv::DataType<T>::depth, dst_channels));              \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channels));           \
                                                                               \
  int src_bytes = src.rows * src.step;                                         \
  int dst_bytes = dst.rows * dst.step;                                         \
  cl_int error_code = 0;                                                       \
  cl_mem gpu_src = clCreateBuffer(context,                                     \
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,   \
                                  src_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_dst = clCreateBuffer(context,                                     \
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,   \
                                  dst_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,    \
                                    src.data, 0, NULL, NULL);                  \
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);                               \
                                                                               \
  int src_size = size.height * size.width * src_channels * sizeof(T);          \
  int dst_size = size.height * size.width * dst_channels * sizeof(T);          \
  cl_mem gpu_input = clCreateBuffer(context,                                   \
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  \
                                    src_size, NULL, &error_code);              \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_output = clCreateBuffer(context,                                  \
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,\
                                     dst_size, NULL, &error_code);             \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,   \
                                    0, src_size, 0, NULL, NULL, &error_code);  \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
  copyMatToArray(src, input);                                                  \
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);\
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
                                                                               \
  cv::cvtColor(src, cv_dst, cv::COLOR_ ## Function);                           \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, src.rows, src.cols,                         \
      src.step / sizeof(T), gpu_src, dst.step / sizeof(T), gpu_dst);           \
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes,      \
                                   dst.data, 0, NULL, NULL);                   \
  CHECK_ERROR(error_code, clEnqueueReadBuffer);                                \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, size.height, size.width,                    \
      size.width * src_channels, gpu_input, size.width * dst_channels,         \
      gpu_output);                                                             \
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,  \
                                     0, dst_size, 0, NULL, NULL, &error_code); \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
                                                                               \
  float epsilon;                                                               \
  if (sizeof(T) == 1) {                                                        \
    epsilon = EPSILON_1F;                                                      \
  }                                                                            \
  else {                                                                       \
    epsilon = EPSILON_E6;                                                      \
  }                                                                            \
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data,         \
      dst.step, epsilon);                                                      \
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,                     \
      size.width * dst_channels * sizeof(T), epsilon);                         \
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,     \
                                       NULL);                                  \
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
                                                                               \
  clReleaseMemObject(gpu_src);                                                 \
  clReleaseMemObject(gpu_dst);                                                 \
  clReleaseMemObject(gpu_input);                                               \
  clReleaseMemObject(gpu_output);                                              \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define UNITTEST_TEST_SUITE(Function, T, src_channel, dst_channel)             \
using PplCvOclCvtColor ## Function ## T =                                      \
        PplCvOclCvtColor ## Function<T, src_channel, dst_channel>;             \
TEST_P(PplCvOclCvtColor ## Function ## T, Standard) {                          \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclCvtColor ## Function ## T,            \
  ::testing::Values(cv::Size{320, 240}, cv::Size{321, 240},                    \
                    cv::Size{640, 480}, cv::Size{642, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1283, 720},                  \
                    cv::Size{1920, 1080}, cv::Size{1934, 1080}),               \
  [](const testing::TestParamInfo<                                             \
      PplCvOclCvtColor ## Function ## T::ParamType>& info) {                   \
    return convertToString(info.param);                                        \
  }                                                                            \
);

#define UNITTEST(Function, src_channel, dst_channel)                           \
UNITTEST_CLASS_DECLARATION(Function)                                           \
UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)                 \
UNITTEST_TEST_SUITE(Function, float, src_channel, dst_channel)

/**************************** Indirect unittest *****************************/

#define INDIRECT_UNITTEST_CLASS_DECLARATION(F1, F2, Function, float_diff)      \
template <typename T, int src_channels, int dst_channels>                      \
class PplCvOclCvtColor ## Function :                                           \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvOclCvtColor ## Function() {                                             \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
                                                                               \
    ppl::common::ocl::createSharedFrameChain(false);                           \
    context = ppl::common::ocl::getSharedFrameChain()->getContext();           \
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();             \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
  cl_context context;                                                          \
  cl_command_queue queue;                                                      \
};                                                                             \
                                                                               \
template <typename T, int src_channels, int dst_channels>                      \
bool PplCvOclCvtColor ## Function<T, src_channels, dst_channels>::apply() {    \
  cv::Mat src;                                                                 \
  src = createSourceImage(size.height, size.width,                             \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat src1(size.height, size.width,                                        \
               CV_MAKETYPE(cv::DataType<T>::depth, (src_channels - 1)));       \
  cv::Mat dst(size.height, size.width,                                         \
              CV_MAKETYPE(cv::DataType<T>::depth, dst_channels));              \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channels));           \
                                                                               \
  int src_bytes = src.rows * src.step;                                         \
  int dst_bytes = dst.rows * dst.step;                                         \
  cl_int error_code = 0;                                                       \
  cl_mem gpu_src = clCreateBuffer(context,                                     \
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,   \
                                  src_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_dst = clCreateBuffer(context,                                     \
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,   \
                                  dst_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,    \
                                    src.data, 0, NULL, NULL);                  \
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);                               \
                                                                               \
  int src_size = size.height * size.width * src_channels * sizeof(T);          \
  int dst_size = size.height * size.width * dst_channels * sizeof(T);          \
  cl_mem gpu_input = clCreateBuffer(context,                                   \
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  \
                                    src_size, NULL, &error_code);              \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_output = clCreateBuffer(context,                                  \
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,\
                                     dst_size, NULL, &error_code);             \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,   \
                                    0, src_size, 0, NULL, NULL, &error_code);  \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
  copyMatToArray(src, input);                                                  \
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);\
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
                                                                               \
  cv::cvtColor(src, src1, cv::COLOR_ ## F1);                                   \
  cv::cvtColor(src1, cv_dst, cv::COLOR_ ## F2);                                \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, src.rows, src.cols,                         \
      src.step / sizeof(T), gpu_src, dst.step / sizeof(T), gpu_dst);           \
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes,      \
                                   dst.data, 0, NULL, NULL);                   \
  CHECK_ERROR(error_code, clEnqueueReadBuffer);                                \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, size.height, size.width,                    \
      size.width * src_channels, gpu_input, size.width * dst_channels,         \
      gpu_output);                                                             \
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,  \
                                     0, dst_size, 0, NULL, NULL, &error_code); \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
                                                                               \
  float epsilon;                                                               \
  if (sizeof(T) == 1) {                                                        \
    epsilon = EPSILON_1F;                                                      \
  }                                                                            \
  else {                                                                       \
    epsilon = EPSILON_E6;                                                      \
  }                                                                            \
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data,         \
      dst.step, epsilon);                                                      \
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,                     \
      size.width * dst_channels * sizeof(T), epsilon);                         \
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,     \
                                       NULL);                                  \
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
                                                                               \
  clReleaseMemObject(gpu_src);                                                 \
  clReleaseMemObject(gpu_dst);                                                 \
  clReleaseMemObject(gpu_input);                                               \
  clReleaseMemObject(gpu_output);                                              \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define INDIRECT_UNITTEST(F1, F2, Function, src_channel, dst_channel,          \
                          float_diff)                                          \
INDIRECT_UNITTEST_CLASS_DECLARATION(F1, F2, Function, float_diff)              \
UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)                 \
UNITTEST_TEST_SUITE(Function, float, src_channel, dst_channel)

// // BGR(RBB) <-> BGRA(RGBA)
// UNITTEST(BGR2BGRA, 3, 4)
// UNITTEST(RGB2RGBA, 3, 4)
// UNITTEST(BGRA2BGR, 4, 3)
// UNITTEST(RGBA2RGB, 4, 3)
// UNITTEST(BGR2RGBA, 3, 4)
// UNITTEST(RGB2BGRA, 3, 4)
// UNITTEST(BGRA2RGB, 4, 3)
// UNITTEST(RGBA2BGR, 4, 3)

// // BGR <-> RGB
// UNITTEST(BGR2RGB, 3, 3)
// UNITTEST(RGB2BGR, 3, 3)

// // BGRA <-> RGBA
// UNITTEST(BGRA2RGBA, 4, 4)
// UNITTEST(RGBA2BGRA, 4, 4)

// // BGR/RGB/BGRA/RGBA <-> Gray
// UNITTEST(BGR2GRAY, 3, 1)
// UNITTEST(RGB2GRAY, 3, 1)
// UNITTEST(BGRA2GRAY, 4, 1)
// UNITTEST(RGBA2GRAY, 4, 1)
// UNITTEST(GRAY2BGR, 1, 3)
// UNITTEST(GRAY2RGB, 1, 3)
// UNITTEST(GRAY2BGRA, 1, 4)
// UNITTEST(GRAY2RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> YCrCb
UNITTEST(BGR2YCrCb, 3, 3)
UNITTEST(RGB2YCrCb, 3, 3)
INDIRECT_UNITTEST(BGRA2BGR, BGR2YCrCb, BGRA2YCrCb, 4, 3, 1e-6)
INDIRECT_UNITTEST(RGBA2RGB, RGB2YCrCb, RGBA2YCrCb, 4, 3, 1e-6)
UNITTEST(YCrCb2BGR, 3, 3)
UNITTEST(YCrCb2RGB, 3, 3)
INDIRECT_UNITTEST(YCrCb2BGR, BGR2BGRA, YCrCb2BGRA, 3, 4, 1e-6)
INDIRECT_UNITTEST(YCrCb2RGB, RGB2RGBA, YCrCb2RGBA, 3, 4, 1e-6)
