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

#include "ppl/cv/cuda/gaussianblur.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MemoryPool, int, int, ppl::cv::BorderType,
                              cv::Size>;
inline std::string convertToStringGaussianBlur(const Parameters& parameters) {
  std::ostringstream formatted;

  MemoryPool memory_pool = std::get<0>(parameters);
  if (memory_pool == kActivated) {
    formatted << "MemoryPoolUsed" << "_";
  }
  else {
    formatted << "MemoryPoolUnused" << "_";
  }

  int ksize = std::get<1>(parameters);
  formatted << "Ksize" << ksize << "_";

  int int_sigma = std::get<2>(parameters);
  formatted << "Sigma" << int_sigma << "_";

  ppl::cv::BorderType border_type = std::get<3>(parameters);
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    formatted << "BORDER_REFLECT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    formatted << "BORDER_REFLECT_101" << "_";
  }
  else {  // border_type == ppl::cv::BORDER_DEFAULT
    formatted << "BORDER_DEFAULT" << "_";
  }

  cv::Size size = std::get<4>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaGaussianBlurTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaGaussianBlurTest() {
    const Parameters& parameters = GetParam();
    memory_pool = std::get<0>(parameters);
    ksize       = std::get<1>(parameters);
    sigma       = std::get<2>(parameters) / 10.f;
    border_type = std::get<3>(parameters);
    size        = std::get<4>(parameters);
  }

  ~PplCvCudaGaussianBlurTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  int ksize;
  float sigma;
  ppl::cv::BorderType border_type;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaGaussianBlurTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(src_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, src_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }
  cv::GaussianBlur(src, cv_dst, cv::Size(ksize, ksize), sigma, sigma,
                   cv_border);

  if (memory_pool == kActivated) {
    size_t volume = ksize * sizeof(float);
    size_t ceiled_volume = ppl::cv::cuda::ceil1DVolume(volume);
    volume = ppl::cv::cuda::ceil2DVolume(size.width * channels * sizeof(float),
                                         size.height) * 2;
    ceiled_volume += volume;
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  ppl::cv::cuda::GaussianBlur<T, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, sigma,
      gpu_dst.step / sizeof(T), (T*)gpu_dst.data, border_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::GaussianBlur<T, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, ksize, sigma, size.width * channels,
      gpu_output, border_type);
  cudaMemcpy(output, gpu_output, src_size, cudaMemcpyDeviceToHost);

  if (memory_pool == kActivated) {
    ppl::cv::cuda::shutDownGpuMemoryPool();
  }

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_2F;
  }
  else {
    epsilon = EPSILON_E4;
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
using PplCvCudaGaussianBlurTest ## T ## channels =                             \
        PplCvCudaGaussianBlurTest<T, channels>;                                \
TEST_P(PplCvCudaGaussianBlurTest ## T ## channels, Standard) {                 \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaGaussianBlurTest ## T ## channels,                                  \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(1, 5, 13, 31, 43),                                       \
    ::testing::Values(0, 1, 7, 10, 43),                                        \
    ::testing::Values(ppl::cv::BORDER_REPLICATE, ppl::cv::BORDER_REFLECT,      \
                      ppl::cv::BORDER_REFLECT_101),                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaGaussianBlurTest ## T ## channels::ParamType>& info) {          \
    return convertToStringGaussianBlur(info.param);                            \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
