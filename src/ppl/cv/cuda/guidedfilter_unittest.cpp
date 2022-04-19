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

#include "ppl/cv/cuda/guidedfilter.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/ximgproc/edge_filter.hpp"
#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

struct Config {
  int radius;
  float eps;
};

using Parameters = std::tuple<MemoryPool, Config, cv::Size>;
inline std::string convertToStringGuidedFilter(const Parameters& parameters) {
  std::ostringstream formatted;

  MemoryPool memory_pool = std::get<0>(parameters);
  if (memory_pool == kActivated) {
    formatted << "MemoryPoolUsed" << "_";
  }
  else {
    formatted << "MemoryPoolUnused" << "_";
  }

  Config config = std::get<1>(parameters);
  formatted << "Radius" << config.radius << "_";
  formatted << "Eps" << config.eps << "_";

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int srcCns, int guideCns>
class PplCvCudaGuidedFilterTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaGuidedFilterTest() {
    const Parameters& parameters = GetParam();
    memory_pool = std::get<0>(parameters);
    config      = std::get<1>(parameters);
    size        = std::get<2>(parameters);
  }

  ~PplCvCudaGuidedFilterTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  Config config;
  cv::Size size;
};

template <typename T, int srcCns, int guideCns>
bool PplCvCudaGuidedFilterTest<T, srcCns, guideCns>::apply() {
  cv::Mat src, guide;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  guide = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<T>::depth, guideCns));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_guide(guide);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * srcCns * sizeof(T);
  int guide_size = size.height * size.width * guideCns * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* guide1 = (T*)malloc(guide_size);
  T* output = (T*)malloc(src_size);
  T* gpu_input;
  T* gpu_guide1;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_guide1, guide_size);
  cudaMalloc((void**)&gpu_output, src_size);
  copyMatToArray(src, input);
  copyMatToArray(guide, guide1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_guide1, guide1, guide_size, cudaMemcpyHostToDevice);

  cv::ximgproc::guidedFilter(guide, src, cv_dst, config.radius, config.eps, -1);

  if (memory_pool == kActivated) {
    size_t size_width = size.width * sizeof(float);
    size_t size_height = size.height * (srcCns * 2 + guideCns + 28);
    size_t ceiled_volume = ppl::cv::cuda::ceil2DVolume(size_width, size_height);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  ppl::cv::cuda::GuidedFilter<T, srcCns, guideCns>(0, gpu_src.rows,
      gpu_src.cols, gpu_src.step / sizeof(T), (T*)gpu_src.data,
      gpu_guide.step / sizeof(T), (T*)gpu_guide.data, gpu_dst.step / sizeof(T),
      (T*)gpu_dst.data, config.radius, config.eps, ppl::cv::BORDER_REFLECT);
  gpu_dst.download(dst);

  ppl::cv::cuda::GuidedFilter<T, srcCns, guideCns>(0, size.height, size.width,
      size.width * srcCns, gpu_input, size.width * guideCns, gpu_guide1,
      size.width * srcCns, gpu_output, config.radius, config.eps,
      ppl::cv::BORDER_REFLECT);
  cudaMemcpy(output, gpu_output, src_size, cudaMemcpyDeviceToHost);

  if (memory_pool == kActivated) {
    ppl::cv::cuda::shutDownGpuMemoryPool();
  }

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
  free(guide1);
  free(output);
  cudaFree(gpu_input);
  cudaFree(gpu_guide1);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, srcCns, guideCns)                                          \
using PplCvCudaGuidedFilterTest ## T ## srcCns ## guideCns =                   \
        PplCvCudaGuidedFilterTest<T, srcCns, guideCns>;                        \
TEST_P(PplCvCudaGuidedFilterTest ## T ## srcCns ## guideCns, Standard) {       \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaGuidedFilterTest ## T ## srcCns ## guideCns,                        \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(Config{3, 26.f}, Config{7, 59.f}, Config{8, 11.f},       \
                      Config{15, 9.f}, Config{22, 64.f}),                      \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaGuidedFilterTest ## T ## srcCns ## guideCns::ParamType>& info) {\
    return convertToStringGuidedFilter(info.param);                            \
  }                                                                            \
);

UNITTEST(uchar, 1, 1)
UNITTEST(uchar, 3, 1)
UNITTEST(uchar, 4, 1)
UNITTEST(uchar, 1, 3)
UNITTEST(uchar, 3, 3)
UNITTEST(uchar, 4, 3)
UNITTEST(float, 1, 1)
UNITTEST(float, 3, 1)
UNITTEST(float, 4, 1)
UNITTEST(float, 1, 3)
UNITTEST(float, 3, 3)
UNITTEST(float, 4, 3)
