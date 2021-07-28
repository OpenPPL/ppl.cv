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

#include "filter2d.h"

#include <tuple>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;

using Parameters = std::tuple<int, BorderType, cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  int ksize = std::get<0>(parameters);
  formatted << "Ksize" << ksize << "_";

  BorderType border_type = (BorderType)std::get<1>(parameters);
  formatted << (border_type == BORDER_TYPE_REPLICATE ? "BORDER_REPLICATE" :
                              "BORDER_DEFAULT") << "_";

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename Tsrc, typename Tdst, int channels>
class PplCvCudaFilter2DTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaFilter2DTest() {
    const Parameters& parameters = GetParam();
    ksize       = std::get<0>(parameters);
    border_type = std::get<1>(parameters);
    size        = std::get<2>(parameters);
  }

  ~PplCvCudaFilter2DTest() {
  }

  bool apply();

 private:
  int ksize;
  BorderType border_type;
  cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvCudaFilter2DTest<Tsrc, Tdst, channels>::apply() {
  cv::Mat src, kernel0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  kernel0 = createSourceImage(ksize, ksize,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(Tsrc);
  int dst_size = size.height * size.width * channels * sizeof(Tdst);
  int kernel_size = ksize * ksize * sizeof(float);
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

  cv::BorderTypes border;
  if (border_type == BORDER_TYPE_REPLICATE) {
    border = cv::BORDER_REPLICATE;
  }
  else {
    border = cv::BORDER_DEFAULT;
  }
  cv::filter2D(src, cv_dst, cv_dst.depth(), kernel0, cv::Point(-1,-1), 0,
               border);

  Filter2D<Tsrc, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data, ksize, gpu_kernel,
      gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data, border_type);
  Filter2D<Tsrc, channels>(0, size.height, size.width, size.width * channels,
      gpu_input, ksize, gpu_kernel, size.width * channels, gpu_output,
      border_type);
  gpu_dst.download(dst);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(Tdst) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E3;
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
using PplCvCudaFilter2DTest ## Tsrc ## channels =                              \
        PplCvCudaFilter2DTest<Tsrc, Tdst, channels>;                           \
TEST_P(PplCvCudaFilter2DTest ## Tsrc ## channels, Standard) {                  \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaFilter2DTest ## Tsrc ## channels,                                   \
  ::testing::Combine(                                                          \
    ::testing::Values(1, 3, 5, 13, 27, 43),                                    \
    ::testing::Values(BORDER_TYPE_DEFAULT, BORDER_TYPE_REPLICATE),             \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaFilter2DTest ## Tsrc ## channels::ParamType>& info) {           \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, uchar, 1)
UNITTEST(uchar, uchar, 3)
UNITTEST(uchar, uchar, 4)
UNITTEST(float, float, 1)
UNITTEST(float, float, 3)
UNITTEST(float, float, 4)
