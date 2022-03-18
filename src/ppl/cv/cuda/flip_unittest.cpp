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

#include "ppl/cv/cuda/flip.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

enum FlipFunctions {
  kFlipX,
  kFlipY,
  kFlipXY,
};

using Parameters = std::tuple<FlipFunctions, cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  FlipFunctions function = std::get<0>(parameters);
  if (function == kFlipX) {
    formatted << "FlipX" << "_";
  }
  else if (function == kFlipY) {
    formatted << "FlipY" << "_";
  }
  else if (function == kFlipXY) {
    formatted << "FlipXY" << "_";
  }
  else {
  }

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaFlipTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaFlipTest() {
    const Parameters& parameters = GetParam();
    function = std::get<0>(parameters);
    size     = std::get<1>(parameters);
  }

  ~PplCvCudaFlipTest() {
  }

  bool apply();

 private:
  FlipFunctions function;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaFlipTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(src.rows, src.cols,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(src.rows, src.cols,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  T* input = (T*)malloc(src_size);
  T* output = (T*)malloc(src_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, src_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  if (function == kFlipX) {
    cv::flip(src, cv_dst, 0);
    ppl::cv::cuda::Flip<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, 0);
    ppl::cv::cuda::Flip<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, size.width * channels, gpu_output, 0);
  }
  else if (function == kFlipY) {
    cv::flip(src, cv_dst, 1);
    ppl::cv::cuda::Flip<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, 1);
    ppl::cv::cuda::Flip<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, size.width * channels, gpu_output, 1);
  }
  else if (function == kFlipXY) {
    cv::flip(src, cv_dst, -1);
    ppl::cv::cuda::Flip<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, -1);
    ppl::cv::cuda::Flip<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, size.width * channels, gpu_output,
        -1);
  }
  else {
  }
  gpu_dst.download(dst);
  cudaMemcpy(output, gpu_output, src_size, cudaMemcpyDeviceToHost);

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
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaFlipTest ## T ## channels = PplCvCudaFlipTest<T, channels>;     \
TEST_P(PplCvCudaFlipTest ## T ## channels, Standard) {                         \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaFlipTest ## T ## channels,           \
  ::testing::Combine(                                                          \
    ::testing::Values(kFlipX, kFlipY, kFlipXY),                                \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080},               \
                      cv::Size{450, 660}, cv::Size{660, 450})),                \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaFlipTest ## T ## channels::ParamType>& info) {                  \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
