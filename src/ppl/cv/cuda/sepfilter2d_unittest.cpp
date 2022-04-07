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

#include "ppl/cv/cuda/sepfilter2d.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MemoryPool, int, int, ppl::cv::BorderType,
                              cv::Size>;
inline std::string convertToStringFilter2D(const Parameters& parameters) {
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

  int int_delta = std::get<2>(parameters);
  formatted << "Delta" << int_delta << "_";

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

template <typename Tsrc, typename Tdst, int channels>
class PplCvCudaSepFilter2DTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaSepFilter2DTest() {
    const Parameters& parameters = GetParam();
    memory_pool = std::get<0>(parameters);
    ksize       = std::get<1>(parameters);
    delta       = std::get<2>(parameters) / 10.f;
    border_type = std::get<3>(parameters);
    size        = std::get<4>(parameters);
  }

  ~PplCvCudaSepFilter2DTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  int ksize;
  float delta;
  ppl::cv::BorderType border_type;
  cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvCudaSepFilter2DTest<Tsrc, Tdst, channels>::apply() {
  cv::Mat src, kernel0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel0 = createSourceImage(1, ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(Tsrc);
  int dst_size = size.height * size.width * channels * sizeof(Tdst);
  int kernel_size = ksize * sizeof(float);
  Tsrc* input  = (Tsrc*)malloc(src_size);
  Tdst* output = (Tdst*)malloc(dst_size);
  float* kernel1 = (float*)malloc(kernel_size);
  Tsrc* gpu_input;
  Tdst* gpu_output;
  float* gpu_kernel;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  cudaMalloc((void**)&gpu_kernel, kernel_size);
  copyMatToArray(src, input);
  copyMatToArray(kernel0, kernel1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_kernel, kernel1, kernel_size, cudaMemcpyHostToDevice);

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
  cv::sepFilter2D(src, cv_dst, cv_dst.depth(), kernel0, kernel0,
                  cv::Point(-1, -1), delta, cv_border);

  if (memory_pool == kActivated) {
    size_t width = size.width * channels * sizeof(float);
    size_t ceiled_volume = ppl::cv::cuda::ceil2DVolume(width, size.height);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  ppl::cv::cuda::SepFilter2D<Tsrc, Tdst, channels>(0, gpu_src.rows,
      gpu_src.cols, gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data, ksize,
      gpu_kernel, gpu_kernel, gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data,
      delta, border_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::SepFilter2D<Tsrc, Tdst, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, ksize, gpu_kernel, gpu_kernel,
      size.width * channels, gpu_output, delta, border_type);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  if (memory_pool == kActivated) {
    ppl::cv::cuda::shutDownGpuMemoryPool();
  }

  float epsilon;
  if (sizeof(Tdst) <= 2) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E4;
  }
  bool identity0 = checkMatricesIdentity<Tdst>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<Tdst>(cv_dst, output, epsilon);

  free(input);
  free(output);
  free(kernel1);
  cudaFree(gpu_input);
  cudaFree(gpu_output);
  cudaFree(gpu_kernel);

  return (identity0 && identity1);
}

#define UNITTEST(Tsrc, Tdst, channels)                                         \
using PplCvCudaSepFilter2DTest ## Tdst ## channels =                           \
        PplCvCudaSepFilter2DTest<Tsrc, Tdst, channels>;                        \
TEST_P(PplCvCudaSepFilter2DTest ## Tdst ## channels, Standard) {               \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaSepFilter2DTest ## Tdst ## channels,                                \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(1, 4, 5, 13, 28, 43),                                    \
    ::testing::Values(0, 10, 43),                                              \
    ::testing::Values(ppl::cv::BORDER_REPLICATE, ppl::cv::BORDER_REFLECT,      \
                      ppl::cv::BORDER_REFLECT_101),                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaSepFilter2DTest ## Tdst ## channels::ParamType>& info) {        \
    return convertToStringFilter2D(info.param);                                \
  }                                                                            \
);

UNITTEST(uchar, uchar, 1)
UNITTEST(uchar, uchar, 3)
UNITTEST(uchar, uchar, 4)
UNITTEST(uchar, short, 1)
UNITTEST(uchar, short, 3)
UNITTEST(uchar, short, 4)
UNITTEST(float, float, 1)
UNITTEST(float, float, 3)
UNITTEST(float, float, 4)
