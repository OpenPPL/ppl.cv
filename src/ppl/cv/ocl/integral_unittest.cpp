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

#include "ppl/cv/ocl/integral.h"
#include "ppl/cv/ocl/use_memory_pool.h"

#include <tuple>
#include <sstream>
#include <iostream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

using Parameters = std::tuple<cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename Tsrc, typename Tdst>
class PplCvOclIntegralToTest: public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclIntegralToTest() {
    const Parameters& parameters = GetParam();
    size = std::get<0>(parameters);

    ppl::common::ocl::createSharedFrameChain(false);
    context = ppl::common::ocl::getSharedFrameChain()->getContext();
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();

    bool status = ppl::common::ocl::initializeKernelBinariesManager(
                      ppl::common::ocl::BINARIES_RETRIEVE);
    if (status) {
      ppl::common::ocl::FrameChain* frame_chain =
          ppl::common::ocl::getSharedFrameChain();
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);
    }
  }

  ~PplCvOclIntegralToTest() {
  }

  bool apply();

 private:
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename Tsrc, typename Tdst>
bool PplCvOclIntegralToTest<Tsrc, Tdst>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<Tsrc>::depth, 1));
  cv::Mat dst(size.height + 1, size.width + 1,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, 1));
  cv::Mat cv_dst(size.height + 1, size.width + 1,
                 CV_MAKETYPE(cv::DataType<Tdst>::depth, 1));

  int src_bytes0 = src.rows * src.step;
  int dst_bytes0 = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes0,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int src_bytes1 = size.height * size.width * sizeof(Tsrc);
  int dst_bytes1 = (size.height + 1) * (size.width + 1) * sizeof(Tdst);
  cl_mem gpu_input =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  Tsrc* input =
      (Tsrc*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE, 0,
                                src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  size_t size_width = (size.width + 1) * sizeof(float);
  size_t ceiled_volume =
      ppl::cv::ocl::ceil2DVolume(size_width, (size.height + 1));
  ppl::cv::ocl::activateGpuMemoryPool(ceiled_volume);

  cv::integral(src, cv_dst, cv_dst.depth());
  if (dst.rows == src.rows && dst.cols == src.cols) {
    cv::Rect roi(1, 1, size.width, size.height);
    cv::Mat croppedImage = cv_dst(roi);
    cv::Mat tmp;
    croppedImage.copyTo(tmp);
    cv_dst = tmp;
  }

  ppl::cv::ocl::Integral<Tsrc, Tdst>(
      queue, src.rows, src.cols, src.step / sizeof(Tsrc), gpu_src, dst.rows,
      dst.cols, dst.step / sizeof(Tdst), gpu_dst);
  ppl::cv::ocl::Integral<Tsrc, Tdst>(queue, size.height, size.width, size.width,
                                     gpu_input, size.height + 1, size.width + 1,
                                     size.width + 1, gpu_output);

  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);

  Tdst* output =
      (Tdst*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ, 0,
                                dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  float epsilon;
  if (sizeof(Tdst) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_4F;
  }

  bool identity0 = checkMatricesIdentity<Tdst>(
      (const Tdst*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, (const Tdst*)dst.data, dst.step, epsilon);
  bool identity1 = checkMatricesIdentity<Tdst>(
      (const Tdst*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, output, dst.cols * sizeof(Tdst), epsilon);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL, NULL);

  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  ppl::cv::ocl::shutDownGpuMemoryPool();
  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(Tsrc, Tdst)                                                   \
using PplCvOclIntegralToTest ## Tsrc ## Tdst =                                 \
        PplCvOclIntegralToTest<Tsrc, Tdst>;                                    \
TEST_P(PplCvOclIntegralToTest ## Tsrc ## Tdst, Standard) {                     \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvOclIntegralToTest ## Tsrc ## Tdst,                                      \
  ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                    \
                    cv::Size{1283, 720}, cv::Size{1934, 1080},                 \
                    cv::Size{320, 240}, cv::Size{640, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1920, 1080}),                \
  [](const testing::TestParamInfo<                                             \
      PplCvOclIntegralToTest ## Tsrc ## Tdst::ParamType>&                      \
        info) {                                                                \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, int)
UNITTEST(float, float)