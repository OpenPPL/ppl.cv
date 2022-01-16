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

#include <tuple>
#include <sstream>

#include "opencv2/ximgproc/edge_filter.hpp"
#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;

struct Config {
  int radius;
  float eps;
};

using Parameters = std::tuple<Config, cv::Size>;
inline std::string convertToStringGuidedFilter(const Parameters& parameters) {
  std::ostringstream formatted;

  Config config = std::get<0>(parameters);
  formatted << "Radius" << config.radius << "_";
  formatted << "Eps" << config.eps << "_";

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int srcCns, int guideCns>
class PplCvCudaGuidedFilterTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaGuidedFilterTest() {
    const Parameters& parameters = GetParam();
    config = std::get<0>(parameters);
    size   = std::get<1>(parameters);
  }

  ~PplCvCudaGuidedFilterTest() {
  }

  bool apply();

 private:
  Config config;
  cv::Size size;
};

template <typename T, int srcCns, int guideCns>
bool PplCvCudaGuidedFilterTest<T, srcCns, guideCns>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  cv::Mat guide(size.height, size.width,
                CV_MAKETYPE(cv::DataType<T>::depth, guideCns));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  if (srcCns == 1) {
    src.copyTo(guide);
  }
  else if (srcCns == 3) {
    cv::cvtColor(src, guide, cv::COLOR_BGR2GRAY);
  }
  else {
    cv::cvtColor(src, guide, cv::COLOR_BGRA2GRAY);
  }
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_guide(guide);
  cv::cuda::GpuMat gpu_dst(dst);

  cv::ximgproc::guidedFilter(guide, src, cv_dst, config.radius, config.eps, -1);

  GuidedFilter<T, srcCns, guideCns>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_guide.step / sizeof(T),
      (T*)gpu_guide.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data,
      config.radius, config.eps, ppl::cv::BORDER_TYPE_REFLECT);
  gpu_dst.download(dst);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

  return identity;
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
UNITTEST(float, 1, 1)
UNITTEST(float, 3, 1)
UNITTEST(float, 4, 1)
