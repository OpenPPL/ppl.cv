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

#include "ppl/cv/cuda/integral.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename Tsrc, typename Tdst, int channels>
class PplCvCudaIntegralTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaIntegralTest() {
    const Parameters& parameters = GetParam();
    size = std::get<0>(parameters);
  }

  ~PplCvCudaIntegralTest() {
  }

  bool apply();

 private:
  cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvCudaIntegralTest<Tsrc, Tdst, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  // cv::Mat dst(size.height, size.width,
  //             CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::Mat dst(size.height + 1, size.width + 1,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::Mat cv_dst(size.height + 1, size.width + 1,
                 CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * sizeof(Tsrc);
  int dst_size = dst.rows * dst.cols * sizeof(Tdst);
  Tsrc* input  = (Tsrc*)malloc(src_size);
  Tdst* output = (Tdst*)malloc(dst_size);
  Tsrc* gpu_input;
  Tdst* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::integral(src, cv_dst, cv_dst.depth());
  if (dst.rows == src.rows && dst.cols == src.cols) {
    cv::Rect roi(1, 1, size.width, size.height);
    cv::Mat croppedImage = cv_dst(roi);
    cv::Mat tmp;
    croppedImage.copyTo(tmp);
    cv_dst = tmp;
  }

  ppl::cv::cuda::Integral<Tsrc, Tdst, 1>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data, gpu_dst.rows,
      gpu_dst.cols, gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data);
  gpu_dst.download(dst);

  ppl::cv::cuda::Integral<Tsrc, Tdst, 1>(0, size.height, size.width, size.width,
      gpu_input, dst.rows, dst.cols, dst.cols, gpu_output);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(Tdst) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_4F;
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
using PplCvCudaIntegralTest ## Tsrc ## Tdst ## channels =                      \
        PplCvCudaIntegralTest<Tsrc, Tdst, channels>;                           \
TEST_P(PplCvCudaIntegralTest ## Tsrc ## Tdst ## channels, Standard) {          \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaIntegralTest ## Tsrc ## Tdst ## channels,                           \
  ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                    \
                    cv::Size{1283, 720}, cv::Size{1934, 1080},                 \
                    cv::Size{320, 240}, cv::Size{640, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1920, 1080}),                \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaIntegralTest ## Tsrc ## Tdst ## channels::ParamType>& info) {   \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, int, 1)
UNITTEST(float, float, 1)
