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

#include "ppl/cv/cuda/cvtcolor.h"

#include <cstdlib>
#include <cmath>
#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"
#include "ppl/cv/x86/cvtcolor.h"

template <typename T>
bool checkArraysIdentity(const T* image0, const T* image1, int rows, int cols,
                         int channels, float epsilon, bool display = false) {
  T value0, value1;
  int index = 0;
  float difference, max = 0.0f;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      for (int channel = 0; channel < channels; channel++) {
        value0 = image0[index];
        value1 = image1[index];
        index++;

        difference = fabs((float) value0 - (float) value1);
        findMax(max, difference);
        if (difference > epsilon || display) {
          std::cout << "[" << row << ", " << col << "]." << channel << ": "
                    << (float)value0 << ", " << (float)value1 << std::endl;
        }
      }
    }
  }

  if (max <= epsilon) {
    return true;
  }
  else {
    std::cout << "Max difference between elements of the two images: "
              << max << std::endl;
    return false;
  }
}

using Parameters = std::tuple<cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

#define UNITTEST_CLASS_DECLARATION(Function)                                   \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  cv::Mat src;                                                                 \
  src = createSourceImage(size.height, size.width,                             \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(size.height, size.width,                                         \
              CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));               \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));            \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = size.height * size.width * src_channel * sizeof(T);           \
  int dst_size = size.height * size.width * dst_channel * sizeof(T);           \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  cv::cvtColor(src, cv_dst, cv::COLOR_ ## Function);                           \
                                                                               \
  ppl::cv::cuda::Function<T>(0, gpu_src.rows, gpu_src.cols,                    \
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),    \
      (T*)gpu_dst.data);                                                       \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, size.height, size.width,                       \
      size.width * src_channel, gpu_input, size.width * dst_channel,           \
      gpu_output);                                                             \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  float epsilon;                                                               \
  if (sizeof(T) == 1) {                                                        \
    epsilon = EPSILON_1F;                                                      \
  }                                                                            \
  else {                                                                       \
    epsilon = EPSILON_E6;                                                      \
  }                                                                            \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon, false);      \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);          \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define UNITTEST_TEST_SUITE(Function, T, src_channel, dst_channel)             \
using PplCvCudaCvtColor ## Function ## T =                                     \
        PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>;            \
TEST_P(PplCvCudaCvtColor ## Function ## T, Standard) {                         \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaCvtColor ## Function ## T,           \
  ::testing::Values(cv::Size{320, 240}, cv::Size{321, 240},                    \
                    cv::Size{640, 480}, cv::Size{648, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1293, 720},                  \
                    cv::Size{1920, 1080}, cv::Size{1976, 1080}),               \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaCvtColor ## Function ## T::ParamType>& info) {                  \
    return convertToString(info.param);                                        \
  }                                                                            \
);

#define UNITTEST(Function, src_channel, dst_channel)                           \
UNITTEST_CLASS_DECLARATION(Function)                                           \
UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)                 \
UNITTEST_TEST_SUITE(Function, float, src_channel, dst_channel)

/**************************** Indirect unittest *****************************/

#define INDIRECT_UNITTEST_CLASS_DECLARATION(F1, F2, Function, float_diff)      \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  cv::Mat src;                                                                 \
  src = createSourceImage(size.height, size.width,                             \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat src1(size.height, size.width,                                        \
              CV_MAKETYPE(cv::DataType<T>::depth, (src_channel - 1)));         \
  cv::Mat dst(size.height, size.width,                                         \
              CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));               \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));            \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = size.height * size.width * src_channel * sizeof(T);           \
  int dst_size = size.height * size.width * dst_channel * sizeof(T);           \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  cv::cvtColor(src, src1, cv::COLOR_ ## F1);                                   \
  cv::cvtColor(src1, cv_dst, cv::COLOR_ ## F2);                                \
                                                                               \
  ppl::cv::cuda::Function<T>(0, gpu_src.rows, gpu_src.cols,                    \
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),    \
      (T*)gpu_dst.data);                                                       \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, size.height, size.width,                       \
      size.width * src_channel, gpu_input, size.width * dst_channel,           \
      gpu_output);                                                             \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  float epsilon;                                                               \
  if (sizeof(T) == 1) {                                                        \
    epsilon = EPSILON_1F;                                                      \
  }                                                                            \
  else {                                                                       \
    epsilon = float_diff;                                                      \
  }                                                                            \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon, false);      \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);          \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define INDIRECT_UNITTEST(F1, F2, Function, src_channel, dst_channel,          \
                          float_diff)                                          \
INDIRECT_UNITTEST_CLASS_DECLARATION(F1, F2, Function, float_diff)              \
UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)                 \
UNITTEST_TEST_SUITE(Function, float, src_channel, dst_channel)

/***************************** HSV unittest ********************************/

#define HSV_UNITTEST_CLASS_DECLARATION(Function)                               \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  cv::Mat src;                                                                 \
  src = createSourceImage(size.height, size.width,                             \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(size.height, size.width,                                         \
              CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));               \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));            \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = size.height * size.width * src_channel * sizeof(T);           \
  int dst_size = size.height * size.width * dst_channel * sizeof(T);           \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  cv::cvtColor(src, cv_dst, cv::COLOR_ ## Function);                           \
                                                                               \
  ppl::cv::cuda::Function<T>(0, gpu_src.rows, gpu_src.cols,                    \
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),    \
      (T*)gpu_dst.data);                                                       \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, size.height, size.width,                       \
      size.width * src_channel, gpu_input, size.width * dst_channel,           \
      gpu_output);                                                             \
                                                                               \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  float epsilon;                                                               \
  if (sizeof(T) == 1) {                                                        \
    epsilon = EPSILON_1F;                                                      \
  }                                                                            \
  else {                                                                       \
    epsilon = 0.0002;                                                          \
  }                                                                            \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon, false);      \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);          \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define HSV_UNITTEST(Function, src_channel, dst_channel)                       \
HSV_UNITTEST_CLASS_DECLARATION(Function)                                       \
UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)                 \
UNITTEST_TEST_SUITE(Function, float, src_channel, dst_channel)

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
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  cv::Mat src;                                                                 \
  LabFunctions ppl_function = k ## Function;                                   \
  if (ppl_function == kBGR2LAB || ppl_function == kRGB2LAB) {                  \
    src = createSourceImage(size.height, size.width,                           \
                            CV_MAKETYPE(cv::DataType<T>::depth, src_channel)); \
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
              CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));               \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));            \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = size.height * size.width * src_channel * sizeof(T);           \
  int dst_size = size.height * size.width * dst_channel * sizeof(T);           \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
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
                CV_MAKETYPE(cv::DataType<T>::depth, (dst_channel - 1)));       \
    cv::cvtColor(src, temp, cv::COLOR_Lab2BGR);                                \
    cv::cvtColor(temp, cv_dst, cv::COLOR_BGR2BGRA);                            \
  }                                                                            \
  else if (ppl_function == kLAB2RGBA) {                                        \
    cv::Mat temp(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, (dst_channel - 1)));      \
    cv::cvtColor(src, temp, cv::COLOR_Lab2RGB);                                \
    cv::cvtColor(temp, cv_dst, cv::COLOR_RGB2RGBA);                            \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  ppl::cv::cuda::Function<T>(0, gpu_src.rows, gpu_src.cols,                    \
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),    \
      (T*)gpu_dst.data);                                                       \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, size.height, size.width,                       \
      size.width * src_channel, gpu_input, size.width * dst_channel,           \
      gpu_output);                                                             \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
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
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon, false);      \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);          \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define LAB_UNITTEST(Function, src_channel, dst_channel)                       \
LAB_UNITTEST_CLASS_DECLARATION(Function)                                       \
UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)                 \
UNITTEST_TEST_SUITE(Function, float, src_channel, dst_channel)

/***************************** NV12 unittest ********************************/

enum NV12Functions {
  kNV122BGR,
  kNV122RGB,
  kNV122BGRA,
  kNV122RGBA,
};

#define UNITTEST_NV12_CLASS_DECLARATION(Function)                              \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
              dst_channel));                                                   \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));            \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
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
  ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),       \
      (T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);           \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * dst_channel, gpu_output);                                        \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, EPSILON_1F);          \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, EPSILON_1F);       \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define UNITTEST_NVXX_TEST_SUITE(Function, T, src_channel, dst_channel)        \
using PplCvCudaCvtColor ## Function ## T =                                     \
        PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>;            \
TEST_P(PplCvCudaCvtColor ## Function ## T, Standard) {                         \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaCvtColor ## Function ## T,           \
  ::testing::Values(cv::Size{320, 240}, cv::Size{322, 240},                    \
                    cv::Size{640, 480}, cv::Size{644, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1296, 720},                  \
                    cv::Size{1920, 1080}, cv::Size{1978, 1080}),               \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaCvtColor ## Function ## T::ParamType>& info) {                  \
    return convertToString(info.param);                                        \
  }                                                                            \
);

#define NV12_UNITTEST(Function, src_channel, dst_channel)                      \
UNITTEST_NV12_CLASS_DECLARATION(Function)                                      \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/***************************** NV21 unittest ********************************/

enum NV21Functions {
  kNV212BGR,
  kNV212RGB,
  kNV212BGRA,
  kNV212RGBA,
};

#define UNITTEST_NV21_CLASS_DECLARATION(Function)                              \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
              dst_channel));                                                   \
  cv::Mat cv_dst(size.height, size.width,                                      \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));            \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
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
  ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),       \
      (T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);           \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * dst_channel, gpu_output);                                        \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, EPSILON_1F);          \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, EPSILON_1F);       \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define NV21_UNITTEST(Function, src_channel, dst_channel)                      \
UNITTEST_NV21_CLASS_DECLARATION(Function)                                      \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channel, dst_channel)

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
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
              dst_channel));                                                   \
  cv::Mat cv_dst(dst_height, width,                                            \
                 CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));            \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
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
  ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),       \
      (T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);           \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * dst_channel, gpu_output);                                        \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, EPSILON_1F);          \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, EPSILON_1F);       \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define I420_UNITTEST(Function, src_channel, dst_channel)                      \
UNITTEST_I420_CLASS_DECLARATION(Function)                                      \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/******************** NVXX's comparison with ppl.cv.x86 *********************/

#define NVXX_X86_UNITTEST_CLASS_DECLARATION(Function)                          \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* output_x86 = (T*)malloc(dst_size);                                        \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  ppl::cv::x86::Function<T>(height, width, width * src_channel, input,         \
      width * dst_channel, output_x86);                                        \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * dst_channel, gpu_output);                                        \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity = checkArraysIdentity<T>(output_x86, output, dst_height,       \
                                         width, dst_channel, EPSILON_1F);      \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  free(output_x86);                                                            \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return identity;                                                             \
}

#define NVXX_X86_UNITTEST(Function, src_channel, dst_channel)                  \
NVXX_X86_UNITTEST_CLASS_DECLARATION(Function)                                  \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/**************************** discrete NV12/21 ****************************/

#define DISCRETE_NVXX_UNITTEST_CLASS_DECLARATION(Function)                     \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColorDisc ## Function :                                      \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColorDisc ## Function() {                                        \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColorDisc ## Function() {                                       \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColorDisc ## Function<T, src_channel, dst_channel>::apply() { \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* output_x86 = (T*)malloc(dst_size);                                        \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  if (src_channel == 1) {                                                      \
    ppl::cv::x86::Function<T>(height, width, width * src_channel, input,       \
        width * src_channel, input + height * width, width * dst_channel,      \
        output_x86);                                                           \
    ppl::cv::cuda::Function<T>(0, height, width, width * src_channel,          \
        gpu_input, width * src_channel, gpu_input + height * width,            \
        width * dst_channel, gpu_output);                                      \
  }                                                                            \
  else {                                                                       \
    ppl::cv::x86::Function<T>(height, width, width * src_channel, input,       \
        width * dst_channel, output_x86, width * dst_channel,                  \
        output_x86 + height * width);                                          \
    ppl::cv::cuda::Function<T>(0, height, width, width * src_channel,          \
        gpu_input, width * dst_channel, gpu_output, width * dst_channel,       \
        gpu_output + height * width);                                          \
  }                                                                            \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity = checkArraysIdentity<T>(output_x86, output, dst_height,       \
                                         width, dst_channel, EPSILON_1F);      \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  free(output_x86);                                                            \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return identity;                                                             \
}

#define DISCRETE_NVXX_UNITTEST_TEST_SUITE(Function, T, src_channel,            \
                                          dst_channel)                         \
using PplCvCudaCvtColorDisc ## Function ## T =                                 \
        PplCvCudaCvtColorDisc ## Function<T, src_channel, dst_channel>;        \
TEST_P(PplCvCudaCvtColorDisc ## Function ## T, Standard) {                     \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaCvtColorDisc ## Function ## T,       \
  ::testing::Values(cv::Size{320, 240}, cv::Size{322, 240},                    \
                    cv::Size{640, 480}, cv::Size{644, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1296, 720},                  \
                    cv::Size{1920, 1080}, cv::Size{1978, 1080}),               \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaCvtColorDisc ## Function ## T::ParamType>& info) {              \
    return convertToString(info.param);                                        \
  }                                                                            \
);

#define DISCRETE_NVXX_X86_UNITTEST(Function, src_channel, dst_channel)         \
DISCRETE_NVXX_UNITTEST_CLASS_DECLARATION(Function)                             \
DISCRETE_NVXX_UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/***************************** discrete I420 *****************************/

#define DISCRETE_I420_UNITTEST_CLASS_DECLARATION(Function)                     \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColorDisc ## Function :                                      \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColorDisc ## Function() {                                        \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColorDisc ## Function() {                                       \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColorDisc ## Function<T, src_channel, dst_channel>::apply() { \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* output_x86   = (T*)malloc(dst_size);                                      \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  int stride0 = height * width;                                                \
  int stride1 = stride0 + ((height * width) >> 2);                             \
  if (src_channel == 1) {                                                      \
    ppl::cv::x86::Function<T>(height, width, width * src_channel, input,       \
        width * src_channel / 2, input + stride0, width * src_channel / 2,     \
        input + stride1, width * dst_channel, output_x86);                     \
    ppl::cv::cuda::Function<T>(0, height, width, width * src_channel,          \
        gpu_input, width * src_channel / 2, gpu_input + stride0,               \
        width * src_channel / 2, gpu_input + stride1, width * dst_channel,     \
        gpu_output);                                                           \
  }                                                                            \
  else {                                                                       \
    ppl::cv::x86::Function<T>(height, width, width * src_channel, input,       \
        width * dst_channel, output_x86, width * dst_channel / 2,              \
        output_x86 + stride0, width * dst_channel / 2, output_x86 + stride1);  \
    ppl::cv::cuda::Function<T>(0, height, width, width * src_channel,          \
        gpu_input, width * dst_channel, gpu_output,                            \
        width * dst_channel / 2, gpu_output + stride0,                         \
        width * dst_channel / 2, gpu_output + stride1);                        \
  }                                                                            \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity = checkArraysIdentity<T>(output_x86, output, dst_height,       \
                                         width, dst_channel, EPSILON_1F);      \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  free(output_x86);                                                            \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return identity;                                                             \
}

#define DISCRETE_I420_X86_UNITTEST(Function, src_channel, dst_channel)         \
DISCRETE_I420_UNITTEST_CLASS_DECLARATION(Function)                             \
DISCRETE_NVXX_UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/************************* YUV422 **************************/

enum YUV422Functions {
  kUYVY2BGR,
  kUYVY2GRAY,
  kYUYV2BGR,
  kYUYV2GRAY,
};

#define TO_YUV422_UNITTEST_CLASS_DECLARATION(Function)                         \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  cv::Mat src = createSourceImage(height, width,                               \
                  CV_MAKETYPE(cv::DataType<T>::depth, src_channel), 16, 235);  \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
  cv::Mat cv_dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth,            \
                 dst_channel));                                                \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = height * width * src_channel * sizeof(T);                     \
  int dst_size = height * width * dst_channel * sizeof(T);                     \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  ppl::cv::x86::Function<T>(height, width, src.step / sizeof(T), (T*)src.data, \
      cv_dst.step / sizeof(T), (T*)cv_dst.data);                               \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),       \
      (T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);           \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * dst_channel, gpu_output);                                        \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, EPSILON_1F, false);   \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, EPSILON_1F);       \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define UNITTEST_FROM_YUV422_CLASS_DECLARATION(Function)                       \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColor ## Function :                                          \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColor ## Function() {                                            \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColor ## Function() {                                           \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColor ## Function<T, src_channel, dst_channel>::apply() {     \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  cv::Mat src = createSourceImage(height, width,                               \
                  CV_MAKETYPE(cv::DataType<T>::depth, src_channel), 16, 235);  \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
  cv::Mat cv_dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth,            \
                 dst_channel));                                                \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int src_size = height * width * src_channel * sizeof(T);                     \
  int dst_size = height * width * dst_channel * sizeof(T);                     \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
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
  ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),       \
      (T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);           \
  gpu_dst.download(dst);                                                       \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * dst_channel, gpu_output);                                        \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, EPSILON_1F);          \
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, EPSILON_1F);       \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return (identity0 && identity1);                                             \
}

#define TO_YUV422_X86_UNITTEST(Function, src_channel, dst_channel)             \
TO_YUV422_UNITTEST_CLASS_DECLARATION(Function)                                 \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channel, dst_channel)

#define FROM_YUV422_UNITTEST(Function, src_channel, dst_channel)               \
UNITTEST_FROM_YUV422_CLASS_DECLARATION(Function)                               \
UNITTEST_NVXX_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/***************************** NVXX to YUV2 *****************************/

#define DISCRETE_YUY2_UNITTEST_CLASS_DECLARATION(Function)                     \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColorDisc ## Function :                                      \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColorDisc ## Function() {                                        \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColorDisc ## Function() {                                       \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColorDisc ## Function<T, src_channel, dst_channel>::apply() { \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  cv::Mat src;                                                                 \
  int src_height = height + (height >> 1);                                     \
  int dst_width  = width << 1;                                                 \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = height * dst_width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* output_x86 = (T*)malloc(dst_size);                                        \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  int stride = height * width;                                                 \
  ppl::cv::x86::Function<T>(height, width, width * src_channel, input,         \
      width * src_channel, input + stride, dst_width * dst_channel,            \
      output_x86);                                                             \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * src_channel, gpu_input + stride, dst_width * dst_channel,        \
      gpu_output);                                                             \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity = checkArraysIdentity<T>(output_x86, output, height,           \
                                         width, 2, EPSILON_1F);                \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  free(output_x86);                                                            \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return identity;                                                             \
}

#define DISCRETE_YUY2_X86_UNITTEST(Function, src_channel, dst_channel)         \
DISCRETE_YUY2_UNITTEST_CLASS_DECLARATION(Function)                             \
DISCRETE_NVXX_UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/***************************** NVXX to I420 *****************************/

#define DISCRETE_NVXX_TO_I420_UNITTEST_CLASS_DECLARATION(Function)             \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColorDisc ## Function :                                      \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColorDisc ## Function() {                                        \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColorDisc ## Function() {                                       \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColorDisc ## Function<T, src_channel, dst_channel>::apply() { \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  cv::Mat src;                                                                 \
  int src_height = height + (height >> 1);                                     \
  int dst_height = height + (height >> 1);                                     \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* output_x86 = (T*)malloc(dst_size);                                        \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  int stride0 = height * width;                                                \
  int stride1 = height * width * 5 / 4;                                        \
  ppl::cv::x86::Function<T>(height, width, width * src_channel, input,         \
      width * src_channel, input + stride0, width * dst_channel, output_x86,   \
      width * dst_channel / 2, output_x86 + stride0, width * dst_channel / 2,  \
      output_x86 + stride1);                                                   \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * src_channel, gpu_input + stride0, width * dst_channel,           \
      gpu_output, width * dst_channel / 2, gpu_output + stride0,               \
      width * dst_channel / 2, gpu_output + stride1);                          \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity = checkArraysIdentity<T>(output_x86, output, dst_height,       \
                                         width, 1, EPSILON_1F);                \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  free(output_x86);                                                            \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return identity;                                                             \
}

#define DISCRETE_NVXX_TO_I420_X86_UNITTEST(Function, src_channel, dst_channel) \
DISCRETE_NVXX_TO_I420_UNITTEST_CLASS_DECLARATION(Function)                     \
DISCRETE_NVXX_UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)

/***************************** I420 to NVXX *****************************/

#define DISCRETE_I420_TO_NVXX_UNITTEST_CLASS_DECLARATION(Function)             \
template <typename T, int src_channel, int dst_channel>                        \
class PplCvCudaCvtColorDisc ## Function :                                      \
  public ::testing::TestWithParam<Parameters> {                                \
 public:                                                                       \
  PplCvCudaCvtColorDisc ## Function() {                                        \
    const Parameters& parameters = GetParam();                                 \
    size = std::get<0>(parameters);                                            \
  }                                                                            \
                                                                               \
  ~PplCvCudaCvtColorDisc ## Function() {                                       \
  }                                                                            \
                                                                               \
  bool apply();                                                                \
                                                                               \
 private:                                                                      \
  cv::Size size;                                                               \
};                                                                             \
                                                                               \
template <typename T, int src_channel, int dst_channel>                        \
bool PplCvCudaCvtColorDisc ## Function<T, src_channel, dst_channel>::apply() { \
  int width  = size.width;                                                     \
  int height = size.height;                                                    \
  cv::Mat src;                                                                 \
  int src_height = height + (height >> 1);                                     \
  int dst_height = height + (height >> 1);                                     \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
                                                                               \
  int src_size = src_height * width * src_channel * sizeof(T);                 \
  int dst_size = dst_height * width * dst_channel * sizeof(T);                 \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  T* output_x86 = (T*)malloc(dst_size);                                        \
  T* gpu_input;                                                                \
  T* gpu_output;                                                               \
  cudaMalloc((void**)&gpu_input, src_size);                                    \
  cudaMalloc((void**)&gpu_output, dst_size);                                   \
  copyMatToArray(src, input);                                                  \
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);              \
                                                                               \
  int stride0 = height * width;                                                \
  int stride1 = height * width * 5 / 4;                                        \
  ppl::cv::x86::Function<T>(height, width, width * src_channel, input,         \
      width * src_channel / 2, input + stride0, width * src_channel / 2,       \
      input + stride1, width * dst_channel, output_x86, width * dst_channel,   \
      output_x86 + stride0);                                                   \
                                                                               \
  ppl::cv::cuda::Function<T>(0, height, width, width * src_channel, gpu_input, \
      width * src_channel / 2, gpu_input + stride0, width * src_channel / 2,   \
      gpu_input + stride1, width * dst_channel, gpu_output,                    \
      width * dst_channel, gpu_output + stride0);                              \
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);            \
                                                                               \
  bool identity = checkArraysIdentity<T>(output_x86, output, dst_height,       \
                                         width, 1, EPSILON_1F);                \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  free(output_x86);                                                            \
  cudaFree(gpu_input);                                                         \
  cudaFree(gpu_output);                                                        \
                                                                               \
  return identity;                                                             \
}

#define DISCRETE_I420_TO_NVXX_X86_UNITTEST(Function, src_channel, dst_channel) \
DISCRETE_I420_TO_NVXX_UNITTEST_CLASS_DECLARATION(Function)                     \
DISCRETE_NVXX_UNITTEST_TEST_SUITE(Function, uchar, src_channel, dst_channel)

// BGR(RBB) <-> BGRA(RGBA), epsilon: 1.1f(uchar), 1e-6(float)
UNITTEST(BGR2BGRA, 3, 4)
UNITTEST(RGB2RGBA, 3, 4)
UNITTEST(BGRA2BGR, 4, 3)
UNITTEST(RGBA2RGB, 4, 3)
UNITTEST(BGR2RGBA, 3, 4)
UNITTEST(RGB2BGRA, 3, 4)
UNITTEST(RGBA2BGR, 4, 3)
UNITTEST(BGRA2RGB, 4, 3)

// BGR <-> RGB, epsilon: 1.1f(uchar), 1e-6(float)
UNITTEST(BGR2RGB, 3, 3)
UNITTEST(RGB2BGR, 3, 3)

// BGRA <-> RGBA, epsilon: 1.1f(uchar), 1e-6(float)
UNITTEST(BGRA2RGBA, 4, 4)
UNITTEST(RGBA2BGRA, 4, 4)

// BGR/RGB/BGRA/RGBA <-> Gray, epsilon: 1.1f(uchar), 1e-6(float)
UNITTEST(BGR2GRAY, 3, 1)
UNITTEST(RGB2GRAY, 3, 1)
UNITTEST(BGRA2GRAY, 4, 1)
UNITTEST(RGBA2GRAY, 4, 1)
UNITTEST(GRAY2BGR, 1, 3)
UNITTEST(GRAY2RGB, 1, 3)
UNITTEST(GRAY2BGRA, 1, 4)
UNITTEST(GRAY2RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> YCrCb, epsilon: 1.1f(uchar), 1e-6(float)
UNITTEST(BGR2YCrCb, 3, 3)
UNITTEST(RGB2YCrCb, 3, 3)
INDIRECT_UNITTEST(BGRA2BGR, BGR2YCrCb, BGRA2YCrCb, 4, 3, 1e-6)
INDIRECT_UNITTEST(RGBA2RGB, RGB2YCrCb, RGBA2YCrCb, 4, 3, 1e-6)
UNITTEST(YCrCb2BGR, 3, 3)
UNITTEST(YCrCb2RGB, 3, 3)
INDIRECT_UNITTEST(YCrCb2BGR, BGR2BGRA, YCrCb2BGRA, 3, 4, 1e-6)
INDIRECT_UNITTEST(YCrCb2RGB, RGB2RGBA, YCrCb2RGBA, 3, 4, 1e-6)

// BGR/RGB/BGRA/RGBA <-> HSV, epsilon: 1.1f(uchar), 0.001(float)
HSV_UNITTEST(BGR2HSV, 3, 3)
HSV_UNITTEST(RGB2HSV, 3, 3)
INDIRECT_UNITTEST(BGRA2BGR, BGR2HSV, BGRA2HSV, 4, 3, 0.001)
INDIRECT_UNITTEST(RGBA2RGB, RGB2HSV, RGBA2HSV, 4, 3, 0.001)
HSV_UNITTEST(HSV2BGR, 3, 3)
HSV_UNITTEST(HSV2RGB, 3, 3)
INDIRECT_UNITTEST(HSV2BGR, BGR2BGRA, HSV2BGRA, 3, 4, 0.001)
INDIRECT_UNITTEST(HSV2RGB, RGB2RGBA, HSV2RGBA, 3, 4, 0.001)

// BGR/RGB/BGRA/RGBA <-> LAB, epsilon: 2.1f(uchar), 0.67(float)
LAB_UNITTEST(BGR2LAB, 3, 3)
LAB_UNITTEST(RGB2LAB, 3, 3)
INDIRECT_UNITTEST(BGRA2BGR, BGR2Lab, BGRA2LAB, 4, 3, 0.67)
INDIRECT_UNITTEST(RGBA2RGB, RGB2Lab, RGBA2LAB, 4, 3, 0.67)
LAB_UNITTEST(LAB2BGR, 3, 3)
LAB_UNITTEST(LAB2RGB, 3, 3)
LAB_UNITTEST(LAB2BGRA, 3, 4)
LAB_UNITTEST(LAB2RGBA, 3, 4)

// BGR/RGB/BGRA/RGBA <-> NV12, epsilon: 1.1f
NVXX_X86_UNITTEST(BGR2NV12, 3, 1)
NVXX_X86_UNITTEST(RGB2NV12, 3, 1)
NVXX_X86_UNITTEST(BGRA2NV12, 4, 1)
NVXX_X86_UNITTEST(RGBA2NV12, 4, 1)
NV12_UNITTEST(NV122BGR, 1, 3)
NV12_UNITTEST(NV122RGB, 1, 3)
NV12_UNITTEST(NV122BGRA, 1, 4)
NV12_UNITTEST(NV122RGBA, 1, 4)
DISCRETE_NVXX_X86_UNITTEST(BGR2NV12, 3, 1)
DISCRETE_NVXX_X86_UNITTEST(RGB2NV12, 3, 1)
DISCRETE_NVXX_X86_UNITTEST(BGRA2NV12, 4, 1)
DISCRETE_NVXX_X86_UNITTEST(RGBA2NV12, 4, 1)
DISCRETE_NVXX_X86_UNITTEST(NV122BGR, 1, 3)
DISCRETE_NVXX_X86_UNITTEST(NV122RGB, 1, 3)
DISCRETE_NVXX_X86_UNITTEST(NV122BGRA, 1, 4)
DISCRETE_NVXX_X86_UNITTEST(NV122RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> NV21, epsilon: 1.1f
NVXX_X86_UNITTEST(BGR2NV21, 3, 1)
NVXX_X86_UNITTEST(RGB2NV21, 3, 1)
NVXX_X86_UNITTEST(BGRA2NV21, 4, 1)
NVXX_X86_UNITTEST(RGBA2NV21, 4, 1)
NV21_UNITTEST(NV212BGR, 1, 3)
NV21_UNITTEST(NV212RGB, 1, 3)
NV21_UNITTEST(NV212BGRA, 1, 4)
NV21_UNITTEST(NV212RGBA, 1, 4)
DISCRETE_NVXX_X86_UNITTEST(BGR2NV21, 3, 1)
DISCRETE_NVXX_X86_UNITTEST(RGB2NV21, 3, 1)
DISCRETE_NVXX_X86_UNITTEST(BGRA2NV21, 4, 1)
DISCRETE_NVXX_X86_UNITTEST(RGBA2NV21, 4, 1)
DISCRETE_NVXX_X86_UNITTEST(NV212BGR, 1, 3)
DISCRETE_NVXX_X86_UNITTEST(NV212RGB, 1, 3)
DISCRETE_NVXX_X86_UNITTEST(NV212BGRA, 1, 4)
DISCRETE_NVXX_X86_UNITTEST(NV212RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> I420, epsilon: 1.1f
I420_UNITTEST(BGR2I420, 3, 1)
I420_UNITTEST(RGB2I420, 3, 1)
I420_UNITTEST(BGRA2I420, 4, 1)
I420_UNITTEST(RGBA2I420, 4, 1)
I420_UNITTEST(I4202BGR, 1, 3)
I420_UNITTEST(I4202RGB, 1, 3)
I420_UNITTEST(I4202BGRA, 1, 4)
I420_UNITTEST(I4202RGBA, 1, 4)
DISCRETE_I420_X86_UNITTEST(BGR2I420, 3, 1)
DISCRETE_I420_X86_UNITTEST(RGB2I420, 3, 1)
DISCRETE_I420_X86_UNITTEST(BGRA2I420, 4, 1)
DISCRETE_I420_X86_UNITTEST(RGBA2I420, 4, 1)
DISCRETE_I420_X86_UNITTEST(I4202BGR, 1, 3)
DISCRETE_I420_X86_UNITTEST(I4202RGB, 1, 3)
DISCRETE_I420_X86_UNITTEST(I4202BGRA, 1, 4)
DISCRETE_I420_X86_UNITTEST(I4202RGBA, 1, 4)

// epsilon: 1.1f
I420_UNITTEST(YUV2GRAY, 1, 1)

// BGR <-> UYVY, epsilon: 1.1f
FROM_YUV422_UNITTEST(UYVY2BGR, 2, 3)
FROM_YUV422_UNITTEST(UYVY2GRAY, 2, 1)

// BGR <-> YUYV, epsilon: 1.1f
FROM_YUV422_UNITTEST(YUYV2BGR, 2, 3)
FROM_YUV422_UNITTEST(YUYV2GRAY, 2, 1)

// NV12/21 <-> I420
DISCRETE_NVXX_TO_I420_X86_UNITTEST(NV122I420, 1, 1)
DISCRETE_NVXX_TO_I420_X86_UNITTEST(NV212I420, 1, 1)
DISCRETE_I420_TO_NVXX_X86_UNITTEST(I4202NV12, 1, 1)
DISCRETE_I420_TO_NVXX_X86_UNITTEST(I4202NV21, 1, 1)
