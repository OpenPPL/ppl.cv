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

#include "ppl/cv/cuda/laplacian.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringLaplacian(const Parameters& parameters) {
  std::ostringstream formatted;

  int ksize = std::get<0>(parameters);
  formatted << "Ksize" << ksize << "_";

  ppl::cv::BorderType border_type = std::get<1>(parameters);
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

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename Tsrc, typename Tdst, int channels>
class PplCvCudaLaplacianTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaLaplacianTest() {
    const Parameters& parameters = GetParam();
    ksize       = std::get<0>(parameters);
    border_type = std::get<1>(parameters);
    size        = std::get<2>(parameters);
  }

  ~PplCvCudaLaplacianTest() {
  }

  bool apply();

 private:
  int ksize;
  ppl::cv::BorderType border_type;
  cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvCudaLaplacianTest<Tsrc, Tdst, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  cv::Mat dst(src.rows, src.cols,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::Mat cv_dst(src.rows, src.cols,
                 CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(Tsrc);
  int dst_size = size.height * size.width * channels * sizeof(Tdst);
  Tsrc* input  = (Tsrc*)malloc(src_size);
  Tdst* output = (Tdst*)malloc(dst_size);
  Tsrc* gpu_input;
  Tdst* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  float scale = 1.5f;
  float delta = 1.5f;

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
  cv::Laplacian(src, cv_dst, cv_dst.depth(), ksize, scale, delta, cv_border);

  ppl::cv::cuda::Laplacian<Tsrc, Tdst, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data,
      gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data, ksize, scale, delta,
      border_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::Laplacian<Tsrc, Tdst, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, size.width * channels, gpu_output,
      ksize, scale, delta, border_type);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

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
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(Tsrc, Tdst, channels)                                         \
using PplCvCudaLaplacianTest ## Tdst ## channels =                             \
        PplCvCudaLaplacianTest<Tsrc, Tdst, channels>;                          \
TEST_P(PplCvCudaLaplacianTest ## Tdst ## channels, Standard) {                 \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaLaplacianTest ## Tdst ## channels,   \
  ::testing::Combine(                                                          \
    ::testing::Values(1, 3, 5),                                                \
    ::testing::Values(ppl::cv::BORDER_REPLICATE, ppl::cv::BORDER_REFLECT,      \
                      ppl::cv::BORDER_REFLECT_101),                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaLaplacianTest ## Tdst ## channels::ParamType>& info) {          \
    return convertToStringLaplacian(info.param);                               \
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
