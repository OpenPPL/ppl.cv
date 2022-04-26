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

#include "ppl/cv/cuda/meanstddev.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MaskType, cv::Size>;
inline std::string convertToStringMean(const Parameters& parameters) {
  std::ostringstream formatted;

  MaskType is_masked = std::get<0>(parameters);
  if (is_masked == kUnmasked) {
    formatted << "Unmasked" << "_";
  }
  else {
    formatted << "Masked" << "_";
  }

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaMeanStdDevTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaMeanStdDevTest() {
    const Parameters& parameters = GetParam();
    is_masked = std::get<0>(parameters);
    size      = std::get<1>(parameters);
  }

  ~PplCvCudaMeanStdDevTest() {
  }

  bool apply();

 private:
  MaskType is_masked;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaMeanStdDevTest<T, channels>::apply() {
  cv::Mat src, mask0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask0(mask0);
  int dst_size = channels * sizeof(float);
  float* mean0 = (float*)malloc(dst_size);
  float* stddev0 = (float*)malloc(dst_size);
  float* gpu_mean0;
  float* gpu_stddev0;
  cudaMalloc((void**)&gpu_mean0, dst_size);
  cudaMalloc((void**)&gpu_stddev0, dst_size);

  int src_size = size.height * size.width * channels * sizeof(T);
  int mask_size = size.height * size.width * sizeof(uchar);
  T* input = (T*)malloc(src_size);
  uchar* mask1 = (uchar*)malloc(mask_size);
  float* mean1 = (float*)malloc(dst_size);
  float* stddev1 = (float*)malloc(dst_size);
  T* gpu_input;
  uchar* gpu_mask1;
  float* gpu_mean1;
  float* gpu_stddev1;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_mask1, mask_size);
  cudaMalloc((void**)&gpu_mean1, dst_size);
  cudaMalloc((void**)&gpu_stddev1, dst_size);
  copyMatToArray(src, input);
  copyMatToArray(mask0, mask1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mask1, mask1, mask_size, cudaMemcpyHostToDevice);

  cv::Scalar cv_mean;
  cv::Scalar cv_stddev;
  if (is_masked == kUnmasked) {
    cv::meanStdDev(src, cv_mean, cv_stddev);
    ppl::cv::cuda::MeanStdDev<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_mean0, gpu_stddev0);
    cudaMemcpy(mean0, gpu_mean0, dst_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(stddev0, gpu_stddev0, dst_size, cudaMemcpyDeviceToHost);

    ppl::cv::cuda::MeanStdDev<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, gpu_mean1, gpu_stddev1);
    cudaMemcpy(mean1, gpu_mean1, dst_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(stddev1, gpu_stddev1, dst_size, cudaMemcpyDeviceToHost);
  }
  else {
    cv::meanStdDev(src, cv_mean, cv_stddev, mask0);
    ppl::cv::cuda::MeanStdDev<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_mean0, gpu_stddev0,
        gpu_mask0.step, (uchar*)gpu_mask0.data);
    cudaMemcpy(mean0, gpu_mean0, dst_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(stddev0, gpu_stddev0, dst_size, cudaMemcpyDeviceToHost);

    ppl::cv::cuda::MeanStdDev<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, gpu_mean1, gpu_stddev1, size.width,
        gpu_mask1);
    cudaMemcpy(mean1, gpu_mean1, dst_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(stddev1, gpu_stddev1, dst_size, cudaMemcpyDeviceToHost);
  }

  float epsilon = EPSILON_E3;
  bool identity = true;
  for (int i = 0; i < channels; i++) {
    if (fabs(cv_mean.val[i] - mean0[i]) > epsilon ||
        fabs(cv_mean.val[i] - mean1[i]) > epsilon ||
        fabs(cv_stddev.val[i] - stddev0[i]) > epsilon ||
        fabs(cv_stddev.val[i] - stddev1[i]) > epsilon) {
      identity = false;
    }
  }

  free(input);
  free(mask1);
  free(mean0);
  free(mean1);
  free(stddev0);
  free(stddev1);
  cudaFree(gpu_input);
  cudaFree(gpu_mask1);
  cudaFree(gpu_mean0);
  cudaFree(gpu_mean1);
  cudaFree(gpu_stddev0);
  cudaFree(gpu_stddev1);

  return identity;
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaMeanStdDevTest ## T ## channels =                               \
        PplCvCudaMeanStdDevTest<T, channels>;                                  \
TEST_P(PplCvCudaMeanStdDevTest ## T ## channels, Standard) {                   \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaMeanStdDevTest ## T ## channels,     \
  ::testing::Combine(                                                          \
    ::testing::Values(kUnmasked, kMasked),                                     \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaMeanStdDevTest ## T ## channels::ParamType>& info) {            \
    return convertToStringMean(info.param);                                    \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
