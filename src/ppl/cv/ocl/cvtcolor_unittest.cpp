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

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"
#ifdef __x86_64__
#include "ppl/cv/x86/cvtcolor.h"
using namespace ppl::cv::x86;
#else
#include "ppl/cv/arm/cvtcolor.h"
using namespace ppl::cv::arm;
#endif

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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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

#define UNITTEST_TEST_SUITE(Function, T, src_channels, dst_channels)           \
using PplCvOclCvtColor ## Function ## T =                                      \
        PplCvOclCvtColor ## Function<T, src_channels, dst_channels>;           \
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

#define UNITTEST(Function, src_channels, dst_channels)                         \
UNITTEST_CLASS_DECLARATION(Function)                                           \
UNITTEST_TEST_SUITE(Function, uchar, src_channels, dst_channels)               \
UNITTEST_TEST_SUITE(Function, float, src_channels, dst_channels)

#define UNITTEST_CLASS_DECLARATION1(Function, float_epsilon)                   \
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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
    epsilon = float_epsilon;                                                   \
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

#define UNITTEST1(Function, src_channels, dst_channels, float_epsilon)         \
UNITTEST_CLASS_DECLARATION1(Function, float_epsilon)                           \
UNITTEST_TEST_SUITE(Function, uchar, src_channels, dst_channels)               \
UNITTEST_TEST_SUITE(Function, float, src_channels, dst_channels)

/***************************** LAB unittest ********************************/

enum LabFunctions {
  kBGR2LAB,
  kRGB2LAB,
  kLAB2BGR,
  kLAB2RGB,
  kLAB2BGRA,
  kLAB2RGBA,
};

#define LAB_UNITTEST_CLASS_DECLARATION(Function)                               \
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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
  LabFunctions ppl_function = k ## Function;                                   \
  if (ppl_function == kBGR2LAB || ppl_function == kRGB2LAB) {                  \
    src = createSourceImage(size.height, size.width,                           \
                            CV_MAKETYPE(cv::DataType<T>::depth, src_channels));\
  }                                                                            \
  else {                                                                       \
    cv::Mat temp0 = createSourceImage(size.height, size.width,                 \
                                      CV_MAKETYPE(cv::DataType<T>::depth, 3)); \
    cv::Mat temp1(size.height, size.width,                                     \
                  CV_MAKETYPE(cv::DataType<T>::depth, 3));                     \
    cv::cvtColor(temp0, temp1, cv::COLOR_RGB2Lab);                             \
    src = temp1.clone();                                                       \
  }                                                                            \
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
  if (ppl_function == kBGR2LAB) {                                              \
    cv::cvtColor(src, cv_dst, cv::COLOR_BGR2Lab);                              \
  }                                                                            \
  else if (ppl_function == kRGB2LAB) {                                         \
    cv::cvtColor(src, cv_dst, cv::COLOR_RGB2Lab);                              \
  }                                                                            \
  else if (ppl_function == kLAB2BGR) {                                         \
    cv::cvtColor(src, cv_dst, cv::COLOR_Lab2BGR);                              \
  }                                                                            \
  else if (ppl_function == kLAB2RGB) {                                         \
    cv::cvtColor(src, cv_dst, cv::COLOR_Lab2RGB);                              \
  }                                                                            \
  else if (ppl_function == kLAB2BGRA) {                                        \
    cv::Mat temp(size.height, size.width,                                      \
                CV_MAKETYPE(cv::DataType<T>::depth, (dst_channels - 1)));      \
    cv::cvtColor(src, temp, cv::COLOR_Lab2BGR);                                \
    cv::cvtColor(temp, cv_dst, cv::COLOR_BGR2BGRA);                            \
  }                                                                            \
  else if (ppl_function == kLAB2RGBA) {                                        \
    cv::Mat temp(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, (dst_channels - 1)));     \
    cv::cvtColor(src, temp, cv::COLOR_Lab2RGB);                                \
    cv::cvtColor(temp, cv_dst, cv::COLOR_RGB2RGBA);                            \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
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
    if (ppl_function == kBGR2LAB || ppl_function == kRGB2LAB) {                \
      epsilon = EPSILON_1F;                                                    \
    }                                                                          \
    else {                                                                     \
      epsilon = 2.1f;                                                          \
    }                                                                          \
  }                                                                            \
  else {                                                                       \
    if (ppl_function == kBGR2LAB || ppl_function == kRGB2LAB) {                \
      epsilon = 0.67f;                                                         \
    }                                                                          \
    else {                                                                     \
      epsilon = 0.0022f;                                                       \
    }                                                                          \
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

#define LAB_UNITTEST(Function, src_channels, dst_channels)                     \
LAB_UNITTEST_CLASS_DECLARATION(Function)                                       \
UNITTEST_TEST_SUITE(Function, uchar, src_channels, dst_channels)               \
UNITTEST_TEST_SUITE(Function, float, src_channels, dst_channels)

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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
    epsilon = float_diff;                                                      \
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

#define INDIRECT_UNITTEST(F1, F2, Function, src_channels, dst_channels,        \
                          float_diff)                                          \
INDIRECT_UNITTEST_CLASS_DECLARATION(F1, F2, Function, float_diff)              \
UNITTEST_TEST_SUITE(Function, uchar, src_channels, dst_channels)               \
UNITTEST_TEST_SUITE(Function, float, src_channels, dst_channels)

/****************** NVXX's comparison with ppl.cv.x86/arm *******************/

#define NVXX_PPL_UNITTEST_CLASS_DECLARATION(Function)                          \
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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(dst_height, width,                                               \
              CV_MAKETYPE(cv::DataType<T>::depth, dst_channels));              \
  cv::Mat cv_dst(dst_height, width,                                            \
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
  int src_size = src_height * width * src_channels * sizeof(T);                \
  int dst_size = dst_height * width * dst_channels * sizeof(T);                \
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
  Function<T>(height, width, src.step / sizeof(T), src.data,                   \
              cv_dst.step / sizeof(T), cv_dst.data);                           \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, height, width, src.step / sizeof(T),        \
                            gpu_src, dst.step / sizeof(T), gpu_dst);           \
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes,      \
                                   dst.data, 0, NULL, NULL);                   \
  CHECK_ERROR(error_code, clEnqueueReadBuffer);                                \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, height, width, width * src_channels,        \
                            gpu_input, width * dst_channels, gpu_output);      \
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,  \
                                     0, dst_size, 0, NULL, NULL, &error_code); \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
                                                                               \
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data,         \
      dst.step, EPSILON_1F);                                                   \
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,                     \
      size.width * dst_channels * sizeof(T), EPSILON_1F);                      \
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

#define UNITTEST_NVXX_TEST_SUITE(Function, T, src_channels, dst_channels)      \
using PplCvOclCvtColor ## Function ## T =                                      \
        PplCvOclCvtColor ## Function<T, src_channels, dst_channels>;           \
TEST_P(PplCvOclCvtColor ## Function ## T, Standard) {                          \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclCvtColor ## Function ## T,            \
  ::testing::Values(cv::Size{320, 240}, cv::Size{322, 240},                    \
                    cv::Size{640, 480}, cv::Size{644, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1296, 720},                  \
                    cv::Size{1920, 1080}, cv::Size{1978, 1080}),               \
  [](const testing::TestParamInfo<                                             \
      PplCvOclCvtColor ## Function ## T::ParamType>& info) {                   \
    return convertToString(info.param);                                        \
  }                                                                            \
);

#define NVXX_PPL_UNITTEST(Function, src_channels, dst_channels)                \
NVXX_PPL_UNITTEST_CLASS_DECLARATION(Function)                                  \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channels, dst_channels)

/************** Discrete NVXX's comparison with ppl.cv.x86/arm ***************/

#define DISCRETE_NVXX_PPL_UNITTEST_CLASS_DECLARATION(Function)                 \
template <typename T, int src_channels, int dst_channels>                      \
class PplCvOclCvtColorDescrete ## Function :                                   \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvOclCvtColorDescrete ## Function() {                                     \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
                                                                               \
    ppl::common::ocl::createSharedFrameChain(false);                           \
    context = ppl::common::ocl::getSharedFrameChain()->getContext();           \
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();             \
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColorDescrete ## Function() {                                    \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
bool PplCvOclCvtColorDescrete ## Function<T, src_channels,                     \
                                          dst_channels>::apply() {             \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat cpu_dst(dst_height, width,                                           \
                  CV_MAKETYPE(cv::DataType<T>::depth, dst_channels));          \
                                                                               \
  int src_size = src_height * width * src_channels * sizeof(T);                \
  int dst_size = dst_height * width * dst_channels * sizeof(T);                \
  int uv_size  = (height >> 1) * width * sizeof(T);                            \
  cl_int error_code = 0;                                                       \
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
  cl_mem gpu_uv = clCreateBuffer(context,                                      \
                                 CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,    \
                                 uv_size, NULL, &error_code);                  \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  if (src_channels == 1) {                                                     \
    input = (T*)clEnqueueMapBuffer(queue, gpu_uv, CL_TRUE, CL_MAP_WRITE,       \
                                   0, uv_size, 0, NULL, NULL, &error_code);    \
    CHECK_ERROR(error_code, clEnqueueMapBuffer);                               \
    cv::Mat uv((height >> 1), width, CV_MAKETYPE(cv::DataType<T>::depth, 1),   \
                src.data + height * src.step, src.step);                       \
    copyMatToArray(uv, input);                                                 \
    error_code = clEnqueueUnmapMemObject(queue, gpu_uv, input, 0, NULL, NULL); \
  }                                                                            \
                                                                               \
  if (src_channels == 1) {                                                     \
    Function<T>(height, width, src.step / sizeof(T), src.data,                 \
                src.step / sizeof(T), src.data + height * src.step,            \
                cpu_dst.step / sizeof(T), cpu_dst.data);                       \
    ppl::cv::ocl::Function<T>(queue, height, width, width, gpu_input, width,   \
                              gpu_uv, width * dst_channels, gpu_output);       \
  }                                                                            \
  else {                                                                       \
    Function<T>(height, width, src.step / sizeof(T), src.data,                 \
                cpu_dst.step / sizeof(T), cpu_dst.data,                        \
                cpu_dst.step / sizeof(T), cpu_dst.data +                       \
                height * cpu_dst.step);                                        \
    ppl::cv::ocl::Function<T>(queue, height, width, width * src_channels,      \
                              gpu_input, width, gpu_output, width, gpu_uv);    \
  }                                                                            \
                                                                               \
  float epsilon;                                                               \
  if (src_channels == 1) {                                                     \
    epsilon = EPSILON_2F;                                                      \
  }                                                                            \
  else {                                                                       \
    epsilon = EPSILON_1F;                                                      \
  }                                                                            \
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,  \
                                     0, dst_size, 0, NULL, NULL, &error_code); \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
  bool identity0 = checkMatricesIdentity<T>((const T*)cpu_dst.data, height,    \
      width, cpu_dst.channels(), cpu_dst.step, output,                         \
      size.width * dst_channels * sizeof(T), epsilon);                         \
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,     \
                                       NULL);                                  \
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
                                                                               \
  bool identity1 = true;                                                       \
  if (dst_channels == 1) {                                                     \
    T* uv1 = (T*)clEnqueueMapBuffer(queue, gpu_uv, CL_TRUE, CL_MAP_READ, 0,    \
                                    uv_size, 0, NULL, NULL, &error_code);      \
    identity1 = checkMatricesIdentity<T>((const T*)cpu_dst.data +              \
        height * cpu_dst.step, (height >> 1), width, cpu_dst.channels(),       \
        cpu_dst.step, uv1, size.width * dst_channels * sizeof(T), epsilon);    \
    error_code = clEnqueueUnmapMemObject(queue, gpu_uv, uv1, 0, NULL, NULL);   \
    CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                          \
  }                                                                            \
                                                                               \
  clReleaseMemObject(gpu_input);                                               \
  clReleaseMemObject(gpu_output);                                              \
  clReleaseMemObject(gpu_uv);                                                  \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define UNITTEST_DESCRETE_NVXX_TEST_SUITE(Function, T, src_channels,           \
                                          dst_channels)                        \
using PplCvOclCvtColorDescrete ## Function ## T =                              \
        PplCvOclCvtColorDescrete ## Function<T, src_channels, dst_channels>;   \
TEST_P(PplCvOclCvtColorDescrete ## Function ## T, Standard) {                  \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclCvtColorDescrete ## Function ## T,    \
  ::testing::Values(cv::Size{320, 240}, cv::Size{322, 240},                    \
                    cv::Size{640, 480}, cv::Size{644, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1296, 720},                  \
                    cv::Size{1920, 1080}, cv::Size{1978, 1080}),               \
  [](const testing::TestParamInfo<                                             \
      PplCvOclCvtColorDescrete ## Function ## T::ParamType>& info) {           \
    return convertToString(info.param);                                        \
  }                                                                            \
);

#define DISCRETE_NVXX_PPL_UNITTEST(Function, src_channels, dst_channels)       \
DISCRETE_NVXX_PPL_UNITTEST_CLASS_DECLARATION(Function)                         \
UNITTEST_DESCRETE_NVXX_TEST_SUITE(Function, uchar, src_channels, dst_channels)

/***************************** NV12 unittest ********************************/

enum NV12Functions {
  kNV122BGR,
  kNV122RGB,
  kNV122BGRA,
  kNV122RGBA,
};

#define UNITTEST_NV12_CLASS_DECLARATION(Function)                              \
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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
  int src_height = size.height;                                                \
  int dst_height = size.height;                                                \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = size.height + (size.height >> 1);                             \
  }                                                                            \
  else {                                                                       \
    dst_height = size.height + (size.height >> 1);                             \
  }                                                                            \
  src = createSourceImage(src_height, size.width,                              \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth,     \
              dst_channels));                                                  \
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
  int src_size = src_height * size.width * src_channels * sizeof(T);           \
  int dst_size = dst_height * size.width * dst_channels * sizeof(T);           \
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
  NV12Functions ppl_function = k ## Function;                                  \
  if (ppl_function == kNV122BGR) {                                             \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGR_NV12);                         \
  }                                                                            \
  else if (ppl_function == kNV122RGB) {                                        \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2RGB_NV12);                         \
  }                                                                            \
  else if (ppl_function == kNV122BGRA) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGRA_NV12);                        \
  }                                                                            \
  else if (ppl_function == kNV122RGBA) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2RGBA_NV12);                        \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, size.height, size.width,                    \
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
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data,         \
      dst.step, EPSILON_1F);                                                   \
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,                     \
      size.width * dst_channels * sizeof(T), EPSILON_1F);                      \
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

#define NV12_UNITTEST(Function, src_channels, dst_channels)                    \
UNITTEST_NV12_CLASS_DECLARATION(Function)                                      \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channels, dst_channels)

/***************************** NV21 unittest ********************************/

enum NV21Functions {
  kNV212BGR,
  kNV212RGB,
  kNV212BGRA,
  kNV212RGBA,
};

#define UNITTEST_NV21_CLASS_DECLARATION(Function)                              \
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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
  int src_height = size.height;                                                \
  int dst_height = size.height;                                                \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = size.height + (size.height >> 1);                             \
  }                                                                            \
  else {                                                                       \
    dst_height = size.height + (size.height >> 1);                             \
  }                                                                            \
  src = createSourceImage(src_height, size.width,                              \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth,     \
              dst_channels));                                                  \
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
  int src_size = src_height * size.width * src_channels * sizeof(T);           \
  int dst_size = dst_height * size.width * dst_channels * sizeof(T);           \
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
  NV21Functions ppl_function = k ## Function;                                  \
  if (ppl_function == kNV212BGR) {                                             \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGR_NV21);                         \
  }                                                                            \
  else if (ppl_function == kNV212RGB) {                                        \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2RGB_NV21);                         \
  }                                                                            \
  else if (ppl_function == kNV212BGRA) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGRA_NV21);                        \
  }                                                                            \
  else if (ppl_function == kNV212RGBA) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2RGBA_NV21);                        \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, size.height, size.width,                    \
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
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data,         \
      dst.step, EPSILON_1F);                                                   \
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,                     \
      size.width * dst_channels * sizeof(T), EPSILON_1F);                      \
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

#define NV21_UNITTEST(Function, src_channels, dst_channels)                    \
UNITTEST_NV21_CLASS_DECLARATION(Function)                                      \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channels, dst_channels)

/***************************** I420 unittest ********************************/

enum I420Functions {
  kBGR2I420,
  kRGB2I420,
  kBGRA2I420,
  kRGBA2I420,
  kI4202BGR,
  kI4202RGB,
  kI4202BGRA,
  kI4202RGBA,
  kYUV2GRAY,
};

#define UNITTEST_I420_CLASS_DECLARATION(Function)                              \
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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
  int src_height = size.height;                                                \
  int dst_height = size.height;                                                \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = size.height + (size.height >> 1);                             \
  }                                                                            \
  else {                                                                       \
    dst_height = size.height + (size.height >> 1);                             \
  }                                                                            \
  src = createSourceImage(src_height, size.width,                              \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(dst_height, size.width, CV_MAKETYPE(cv::DataType<T>::depth,      \
              dst_channels));                                                  \
  cv::Mat cv_dst(dst_height, size.width,                                       \
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
  int src_size = src_height * size.width * src_channels * sizeof(T);           \
  int dst_size = dst_height * size.width * dst_channels * sizeof(T);           \
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
  I420Functions ppl_function = k ## Function;                                  \
  if (ppl_function == kBGR2I420) {                                             \
    cv::cvtColor(src, cv_dst, cv::COLOR_BGR2YUV_I420);                         \
  }                                                                            \
  else if (ppl_function == kRGB2I420) {                                        \
    cv::cvtColor(src, cv_dst, cv::COLOR_RGB2YUV_I420);                         \
  }                                                                            \
  else if (ppl_function == kBGRA2I420) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_BGRA2YUV_I420);                        \
  }                                                                            \
  else if (ppl_function == kRGBA2I420) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_RGBA2YUV_I420);                        \
  }                                                                            \
  else if (ppl_function == kI4202BGR) {                                        \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGR_I420);                         \
  }                                                                            \
  else if (ppl_function == kI4202RGB) {                                        \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2RGB_I420);                         \
  }                                                                            \
  else if (ppl_function == kI4202BGRA) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGRA_I420);                        \
  }                                                                            \
  else if (ppl_function == kI4202RGBA) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2RGBA_I420);                        \
  }                                                                            \
  else if (ppl_function == kYUV2GRAY) {                                        \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2GRAY_420);                         \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, size.height, size.width,                    \
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
  bool identity0 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, (const T*)dst.data,         \
      dst.step, EPSILON_1F);                                                   \
  bool identity1 = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows,\
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,                     \
      size.width * dst_channels * sizeof(T), EPSILON_1F);                      \
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

#define I420_UNITTEST(Function, src_channels, dst_channels)                    \
UNITTEST_I420_CLASS_DECLARATION(Function)                                      \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channels, dst_channels)

/************** Discrete I420's comparison with ppl.cv.x86/arm ***************/

#define DISCRETE_I420_PPL_UNITTEST_CLASS_DECLARATION(Function)                 \
template <typename T, int src_channels, int dst_channels>                      \
class PplCvOclCvtColorDescrete ## Function :                                   \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvOclCvtColorDescrete ## Function() {                                     \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
                                                                               \
    ppl::common::ocl::createSharedFrameChain(false);                           \
    context = ppl::common::ocl::getSharedFrameChain()->getContext();           \
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();             \
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColorDescrete ## Function() {                                    \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
bool PplCvOclCvtColorDescrete ## Function<T, src_channels,                     \
                                          dst_channels>::apply() {             \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
                                                                               \
  int src_size = src_height * width * src_channels * sizeof(T);                \
  int dst_size = dst_height * width * dst_channels * sizeof(T);                \
  int uv_size  = (height >> 1) * (width >> 1) * sizeof(T);                     \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* output_cpu = (T*)malloc(dst_size);                                        \
  copyMatToArray(src, input);                                                  \
  cl_int error_code = 0;                                                       \
  cl_mem gpu_input = clCreateBuffer(context,                                   \
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  \
                                    src_size, NULL, &error_code);              \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_output = clCreateBuffer(context,                                  \
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,\
                                     dst_size, NULL, &error_code);             \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  T* map = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,     \
                                  0, src_size, 0, NULL, NULL, &error_code);    \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
  copyMatToArray(src, map);                                                    \
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, map, 0, NULL, NULL);  \
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
  cl_mem gpu_u = clCreateBuffer(context,                                       \
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,     \
                                uv_size, NULL, &error_code);                   \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_v = clCreateBuffer(context,                                       \
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,     \
                                uv_size, NULL, &error_code);                   \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  if (src_channels == 1) {                                                     \
    map = (T*)clEnqueueMapBuffer(queue, gpu_u, CL_TRUE, CL_MAP_WRITE, 0,       \
                                 uv_size, 0, NULL, NULL, &error_code);         \
    CHECK_ERROR(error_code, clEnqueueMapBuffer);                               \
    memcpy(map, input + height * width * sizeof(T), uv_size);                  \
    error_code = clEnqueueUnmapMemObject(queue, gpu_u, map, 0, NULL, NULL);    \
    map = (T*)clEnqueueMapBuffer(queue, gpu_v, CL_TRUE, CL_MAP_WRITE, 0,       \
                                 uv_size, 0, NULL, NULL, &error_code);         \
    CHECK_ERROR(error_code, clEnqueueMapBuffer);                               \
    memcpy(map, input + height * 5 / 4 * width * sizeof(T), uv_size);          \
    error_code = clEnqueueUnmapMemObject(queue, gpu_v, map, 0, NULL, NULL);    \
  }                                                                            \
                                                                               \
  if (src_channels == 1) {                                                     \
    Function<T>(height, width, width, input, width / 2,                        \
        input + height * width * sizeof(T), width / 2,                         \
        input + height * 5 / 4 * width * sizeof(T), width * dst_channels,      \
        output_cpu);                                                           \
    ppl::cv::ocl::Function<T>(queue, height, width, width, gpu_input,          \
                              width / 2, gpu_u, width / 2, gpu_v,              \
                              width * dst_channels, gpu_output);               \
  }                                                                            \
  else {                                                                       \
    Function<T>(height, width, width * src_channels, input, width, output_cpu, \
        width / 2, output_cpu + height * width * sizeof(T), width / 2,         \
        output_cpu + height * 5 / 4 * width * sizeof(T));                      \
    ppl::cv::ocl::Function<T>(queue, height, width, width * src_channels,      \
                              gpu_input, width, gpu_output, width / 2, gpu_u,  \
                              width / 2, gpu_v);                               \
  }                                                                            \
                                                                               \
  map = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ, 0,     \
                               dst_size, 0, NULL, NULL, &error_code);          \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
  bool identity0 = checkMatricesIdentity<T>(output_cpu, height, width,         \
      dst_channels, width * dst_channels * sizeof(T), map,                     \
      width * dst_channels * sizeof(T), EPSILON_3F);                           \
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, map, 0, NULL, NULL); \
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
                                                                               \
  bool identity1 = true;                                                       \
  bool identity2 = true;                                                       \
  if (dst_channels == 1) {                                                     \
    T* uv = (T*)clEnqueueMapBuffer(queue, gpu_u, CL_TRUE, CL_MAP_READ, 0,      \
                                   uv_size, 0, NULL, NULL, &error_code);       \
    identity1 = checkMatricesIdentity<T>(output_cpu +                          \
        height * width * sizeof(T), (height >> 1), (width >> 1), 1, width / 2, \
        uv, width / 2 * sizeof(T), EPSILON_3F);                                \
    error_code = clEnqueueUnmapMemObject(queue, gpu_u, uv, 0, NULL, NULL);     \
    CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                          \
    uv = (T*)clEnqueueMapBuffer(queue, gpu_v, CL_TRUE, CL_MAP_READ, 0, uv_size,\
                                0, NULL, NULL, &error_code);                   \
    identity2 = checkMatricesIdentity<T>(output_cpu +                          \
        height * 5 / 4 * width * sizeof(T), (height >> 1), (width >> 1), 1,    \
        width / 2, uv, width / 2 * sizeof(T), EPSILON_3F);                     \
    error_code = clEnqueueUnmapMemObject(queue, gpu_v, uv, 0, NULL, NULL);     \
    CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                          \
  }                                                                            \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  free(output_cpu);                                                            \
  clReleaseMemObject(gpu_input);                                               \
  clReleaseMemObject(gpu_output);                                              \
  clReleaseMemObject(gpu_u);                                                   \
  clReleaseMemObject(gpu_v);                                                   \
                                                                               \
  return (identity0 && identity1 && identity2);                                                           \
}

#define DISCRETE_I420_PPL_UNITTEST(Function, src_channels, dst_channels)       \
DISCRETE_I420_PPL_UNITTEST_CLASS_DECLARATION(Function)                         \
UNITTEST_DESCRETE_NVXX_TEST_SUITE(Function, uchar, src_channels, dst_channels)

/***************************** YUV422 unittest ********************************/

enum YUV422Functions {
  kUYVY2BGR,
  kUYVY2GRAY,
  kYUYV2BGR,
  kYUYV2GRAY,
};

#define UNITTEST_FROM_YUV422_CLASS_DECLARATION(Function)                       \
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
                                                                               \
    bool status = ppl::common::ocl::initializeKernelBinariesManager(           \
                      ppl::common::ocl::BINARIES_RETRIEVE);                    \
    if (status) {                                                              \
      ppl::common::ocl::FrameChain* frame_chain =                              \
          ppl::common::ocl::getSharedFrameChain();                             \
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  ~PplCvOclCvtColor ## Function() {                                            \
    ppl::common::ocl::shutDownKernelBinariesManager(                           \
        ppl::common::ocl::BINARIES_RETRIEVE);                                  \
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
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  cv::Mat src = createSourceImage(height, width,                               \
                  CV_MAKETYPE(cv::DataType<T>::depth, src_channels), 16, 235); \
  cv::Mat cv_dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth,            \
                 dst_channels));                                               \
                                                                               \
  int src_size = height * width * src_channels * sizeof(T);                    \
  int dst_size = height * width * dst_channels * sizeof(T);                    \
  cl_int error_code = 0;                                                       \
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
  YUV422Functions ppl_function = k ## Function;                                \
  if (ppl_function == kUYVY2BGR) {                                             \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGR_UYVY);                         \
  }                                                                            \
  else if (ppl_function == kUYVY2GRAY) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2GRAY_UYVY);                        \
  }                                                                            \
  else if (ppl_function == kYUYV2BGR) {                                        \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2BGR_YUYV);                         \
  }                                                                            \
  else if (ppl_function == kYUYV2GRAY) {                                       \
    cv::cvtColor(src, cv_dst, cv::COLOR_YUV2GRAY_YUYV);                        \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  ppl::cv::ocl::Function<T>(queue, size.height, size.width,                    \
      size.width * src_channels, gpu_input, size.width * dst_channels,         \
      gpu_output);                                                             \
                                                                               \
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ,  \
                                     0, dst_size, 0, NULL, NULL, &error_code); \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
  bool identity = checkMatricesIdentity<T>((const T*)cv_dst.data, cv_dst.rows, \
      cv_dst.cols, cv_dst.channels(), cv_dst.step, output,                     \
      size.width * dst_channels * sizeof(T), EPSILON_1F);                      \
  error_code = clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL,     \
                                       NULL);                                  \
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
                                                                               \
  clReleaseMemObject(gpu_input);                                               \
  clReleaseMemObject(gpu_output);                                              \
                                                                               \
  return (identity);                                                           \
}

#define FROM_YUV422_UNITTEST(Function, src_channels, dst_channels)             \
UNITTEST_FROM_YUV422_CLASS_DECLARATION(Function)                               \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channels, dst_channels)

// BGR(RBB) <-> BGRA(RGBA)
UNITTEST(BGR2BGRA, 3, 4)
UNITTEST(RGB2RGBA, 3, 4)
UNITTEST(BGRA2BGR, 4, 3)
UNITTEST(RGBA2RGB, 4, 3)
UNITTEST(BGR2RGBA, 3, 4)
UNITTEST(RGB2BGRA, 3, 4)
UNITTEST(BGRA2RGB, 4, 3)
UNITTEST(RGBA2BGR, 4, 3)

// BGR <-> RGB
UNITTEST(BGR2RGB, 3, 3)
UNITTEST(RGB2BGR, 3, 3)

// BGRA <-> RGBA
UNITTEST(BGRA2RGBA, 4, 4)
UNITTEST(RGBA2BGRA, 4, 4)

// BGR/RGB/BGRA/RGBA <-> Gray
UNITTEST(BGR2GRAY, 3, 1)
UNITTEST(RGB2GRAY, 3, 1)
UNITTEST(BGRA2GRAY, 4, 1)
UNITTEST(RGBA2GRAY, 4, 1)
UNITTEST(GRAY2BGR, 1, 3)
UNITTEST(GRAY2RGB, 1, 3)
UNITTEST(GRAY2BGRA, 1, 4)
UNITTEST(GRAY2RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> YCrCb
UNITTEST(BGR2YCrCb, 3, 3)
UNITTEST(RGB2YCrCb, 3, 3)
INDIRECT_UNITTEST(BGRA2BGR, BGR2YCrCb, BGRA2YCrCb, 4, 3, 1e-6)
INDIRECT_UNITTEST(RGBA2RGB, RGB2YCrCb, RGBA2YCrCb, 4, 3, 1e-6)
UNITTEST(YCrCb2BGR, 3, 3)
UNITTEST(YCrCb2RGB, 3, 3)
INDIRECT_UNITTEST(YCrCb2BGR, BGR2BGRA, YCrCb2BGRA, 3, 4, 1e-6)
INDIRECT_UNITTEST(YCrCb2RGB, RGB2RGBA, YCrCb2RGBA, 3, 4, 1e-6)

// // BGR/RGB/BGRA/RGBA <-> HSV
UNITTEST1(BGR2HSV, 3, 3, 0.0001)
UNITTEST1(RGB2HSV, 3, 3, 0.0001)
INDIRECT_UNITTEST(BGRA2BGR, BGR2HSV, BGRA2HSV, 4, 3, 0.001)
INDIRECT_UNITTEST(RGBA2RGB, RGB2HSV, RGBA2HSV, 4, 3, 0.001)
UNITTEST(HSV2BGR, 3, 3)
UNITTEST(HSV2RGB, 3, 3)
INDIRECT_UNITTEST(HSV2BGR, BGR2BGRA, HSV2BGRA, 3, 4, 1e-6)
INDIRECT_UNITTEST(HSV2RGB, RGB2RGBA, HSV2RGBA, 3, 4, 1e-6)

// // BGR/RGB/BGRA/RGBA <-> LAB
LAB_UNITTEST(BGR2LAB, 3, 3)
LAB_UNITTEST(RGB2LAB, 3, 3)
INDIRECT_UNITTEST(BGRA2BGR, BGR2Lab, BGRA2LAB, 4, 3, 0.67)
INDIRECT_UNITTEST(RGBA2RGB, RGB2Lab, RGBA2LAB, 4, 3, 0.67)
LAB_UNITTEST(LAB2BGR, 3, 3)
LAB_UNITTEST(LAB2RGB, 3, 3)
LAB_UNITTEST(LAB2BGRA, 3, 4)
LAB_UNITTEST(LAB2RGBA, 3, 4)

// BGR/RGB/BGRA/RGBA <-> NV12
NVXX_PPL_UNITTEST(BGR2NV12, 3, 1)
NVXX_PPL_UNITTEST(RGB2NV12, 3, 1)
NVXX_PPL_UNITTEST(BGRA2NV12, 4, 1)
NVXX_PPL_UNITTEST(RGBA2NV12, 4, 1)
NV12_UNITTEST(NV122BGR, 1, 3)
NV12_UNITTEST(NV122RGB, 1, 3)
NV12_UNITTEST(NV122BGRA, 1, 4)
NV12_UNITTEST(NV122RGBA, 1, 4)

DISCRETE_NVXX_PPL_UNITTEST(BGR2NV12, 3, 1)
DISCRETE_NVXX_PPL_UNITTEST(RGB2NV12, 3, 1)
DISCRETE_NVXX_PPL_UNITTEST(BGRA2NV12, 4, 1)
DISCRETE_NVXX_PPL_UNITTEST(RGBA2NV12, 4, 1)
DISCRETE_NVXX_PPL_UNITTEST(NV122BGR, 1, 3)
DISCRETE_NVXX_PPL_UNITTEST(NV122RGB, 1, 3)
DISCRETE_NVXX_PPL_UNITTEST(NV122BGRA, 1, 4)
DISCRETE_NVXX_PPL_UNITTEST(NV122RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> NV21
NVXX_PPL_UNITTEST(BGR2NV21, 3, 1)
NVXX_PPL_UNITTEST(RGB2NV21, 3, 1)
NVXX_PPL_UNITTEST(BGRA2NV21, 4, 1)
NVXX_PPL_UNITTEST(RGBA2NV21, 4, 1)
NV21_UNITTEST(NV212BGR, 1, 3)
NV21_UNITTEST(NV212RGB, 1, 3)
NV21_UNITTEST(NV212BGRA, 1, 4)
NV21_UNITTEST(NV212RGBA, 1, 4)

DISCRETE_NVXX_PPL_UNITTEST(BGR2NV21, 3, 1)
DISCRETE_NVXX_PPL_UNITTEST(RGB2NV21, 3, 1)
DISCRETE_NVXX_PPL_UNITTEST(BGRA2NV21, 4, 1)
DISCRETE_NVXX_PPL_UNITTEST(RGBA2NV21, 4, 1)
DISCRETE_NVXX_PPL_UNITTEST(NV212BGR, 1, 3)
DISCRETE_NVXX_PPL_UNITTEST(NV212RGB, 1, 3)
DISCRETE_NVXX_PPL_UNITTEST(NV212BGRA, 1, 4)
DISCRETE_NVXX_PPL_UNITTEST(NV212RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> I420
I420_UNITTEST(BGR2I420, 3, 1)
I420_UNITTEST(RGB2I420, 3, 1)
I420_UNITTEST(BGRA2I420, 4, 1)
I420_UNITTEST(RGBA2I420, 4, 1)
I420_UNITTEST(I4202BGR, 1, 3)
I420_UNITTEST(I4202RGB, 1, 3)
I420_UNITTEST(I4202BGRA, 1, 4)
I420_UNITTEST(I4202RGBA, 1, 4)

DISCRETE_I420_PPL_UNITTEST(BGR2I420, 3, 1)
DISCRETE_I420_PPL_UNITTEST(RGB2I420, 3, 1)
DISCRETE_I420_PPL_UNITTEST(BGRA2I420, 4, 1)
DISCRETE_I420_PPL_UNITTEST(RGBA2I420, 4, 1)
DISCRETE_I420_PPL_UNITTEST(I4202BGR, 1, 3)
DISCRETE_I420_PPL_UNITTEST(I4202RGB, 1, 3)
DISCRETE_I420_PPL_UNITTEST(I4202BGRA, 1, 4)
DISCRETE_I420_PPL_UNITTEST(I4202RGBA, 1, 4)

I420_UNITTEST(YUV2GRAY, 1, 1)

// BGR <-> UYVY
FROM_YUV422_UNITTEST(UYVY2BGR, 2, 3)
FROM_YUV422_UNITTEST(UYVY2GRAY, 2, 1)

// BGR <-> YUYV, epsilon: 1.1f
FROM_YUV422_UNITTEST(YUYV2BGR, 2, 3)
FROM_YUV422_UNITTEST(YUYV2GRAY, 2, 1)
