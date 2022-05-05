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

#include "ppl/cv/cuda/medianblur.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MemoryPool, int, cv::Size>;
inline std::string convertToStringMedianblur(const Parameters& parameters) {
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

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaMedianBlurTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaMedianBlurTest() {
    const Parameters& parameters = GetParam();
    memory_pool = std::get<0>(parameters);
    ksize       = std::get<1>(parameters);
    size        = std::get<2>(parameters);
  }

  ~PplCvCudaMedianBlurTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  int ksize;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaMedianBlurTest<T, channels>::apply() {
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
  int dst_size = size.height * size.width * channels * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(dst_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  if (sizeof(T) == 1 || (sizeof(T) == 4 && (ksize == 3 || ksize == 5))) {
    cv::medianBlur(src, cv_dst, ksize);
  }

  if (memory_pool == kActivated && sizeof(T) == 1 &&
      ((channels == 1 && ksize > 7) ||
       ((channels == 3 || channels == 4) && ksize > 5))) {
    size_t volume = size.width * channels * (size.height + 255) / 256 * 272 *
                    sizeof(ushort);
    size_t ceiled_volume = ppl::cv::cuda::ceil1DVolume(volume);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  ppl::cv::cuda::MedianBlur<T, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
      (T*)gpu_dst.data, ksize, ppl::cv::BORDER_REPLICATE);
  gpu_dst.download(dst);

  ppl::cv::cuda::MedianBlur<T, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, size.width * channels, gpu_output,
      ksize, ppl::cv::BORDER_REPLICATE);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  if (memory_pool == kActivated && sizeof(T) == 1 &&
      ((channels == 1 && ksize > 7) ||
       ((channels == 3 || channels == 4) && ksize > 5))) {
    ppl::cv::cuda::shutDownGpuMemoryPool();
  }

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0, identity1;
  if (sizeof(T) == 1 || (sizeof(T) == 4 && (ksize == 3 || ksize == 5))) {
    identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);
    identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);
  }
  else {
    identity0 = true;
    identity1 = true;
  }

  free(input);
  free(output);
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaMedianBlurTest ## T ## channels =                               \
        PplCvCudaMedianBlurTest<T, channels>;                                  \
TEST_P(PplCvCudaMedianBlurTest ## T ## channels, Standard) {                   \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaMedianBlurTest ## T ## channels,     \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(3, 5, 7, 15, 33, 43),                                    \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaMedianBlurTest ## T ## channels::ParamType>& info) {            \
    return convertToStringMedianblur(info.param);                              \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
