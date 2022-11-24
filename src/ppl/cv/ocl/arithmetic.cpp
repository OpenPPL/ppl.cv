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

#include "ppl/cv/ocl/arithmetic.h"

// #include <iostream>> // debug

#include "utility/utility.hpp"
#include "ppl/common/ocl/framechain.h"
#include "ppl/common/ocl/kernel.h"

#include "kernels/arithmetic.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

/******************************* add operation *******************************/

RetCode addU8(const cl_mem src0, int rows, int cols, int channels,
              int src0_stride, const cl_mem src1, int src1_stride, cl_mem dst,
              int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain);

  int columns = cols * channels;
  if (src0_stride == columns && src1_stride == columns &&
      dst_stride == columns) {
    // std::cout << "came in if()" << std::endl;
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    size_t local_size[]  = {512, 1};
    size_t global_size[] = {(size_t)roundUp(cols, 512, 9), 1};

    frame_chain->setCompileOptions("-D ADD_U81D");
    runOclKernel(frame_chain, "addU8Kernel0", 2, global_size, local_size, src0,
                 columns, src1, dst);
  }
  else {
    // std::cout << "came in else()" << std::endl;
    columns = cols * channels;
    cols = divideUp(columns, 4, 2);
    size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
    size_t global_size[] = {(size_t)cols, (size_t)rows};

    frame_chain->setCompileOptions("-D ADD_U82D");
    runOclKernel(frame_chain, "addU8Kernel1", 2, global_size, local_size, src0,
                 rows, columns, src0_stride, src1, src1_stride, dst,
                 dst_stride);
  }

  return RC_SUCCESS;
}

RetCode addF32(const cl_mem src0, int rows, int cols, int channels,
               int src0_stride, const cl_mem src1, int src1_stride, cl_mem dst,
               int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain);

  int columns = cols * channels;
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)divideUp(columns, 2, 1), (size_t)rows};

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    // std::cout << "came in if()" << std::endl;
    cols = divideUp(columns, 2, 1);
    frame_chain->setCompileOptions("-D ADD_F32ALIGNED");
    runOclKernel(frame_chain, "addF32Kernel0", 2, global_size, local_size, src0,
                 rows, cols, src0_stride, src1, src1_stride, dst, dst_stride);
  }
  else {
    // std::cout << "came in else()" << std::endl;
    frame_chain->setCompileOptions("-D ADD_F32UNALIGNED");
    runOclKernel(frame_chain, "addF32Kernel1", 2, global_size, local_size, src0,
                 rows, columns, src0_stride, src1, src1_stride, dst,
                 dst_stride);
  }

  return RC_SUCCESS;
}

template <>
RetCode Add<uchar, 1>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride0,
                      const cl_mem inData0,
                      int inWidthStride1,
                      const cl_mem inData1,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = addU8(inData0, height, width, 1, inWidthStride0, inData1,
                       inWidthStride1, outData, outWidthStride, queue);

  return code;
}

template <>
RetCode Add<uchar, 3>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride0,
                      const cl_mem inData0,
                      int inWidthStride1,
                      const cl_mem inData1,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = addU8(inData0, height, width, 3, inWidthStride0, inData1,
                       inWidthStride1, outData, outWidthStride, queue);

  return code;
}

template <>
RetCode Add<uchar, 4>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride0,
                      const cl_mem inData0,
                      int inWidthStride1,
                      const cl_mem inData1,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = addU8(inData0, height, width, 4, inWidthStride0, inData1,
                       inWidthStride1, outData, outWidthStride, queue);

  return code;
}

template <>
RetCode Add<float, 1>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride0,
                      const cl_mem inData0,
                      int inWidthStride1,
                      const cl_mem inData1,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addF32(inData0, height, width, 1, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, queue);

  return code;
}

template <>
RetCode Add<float, 3>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride0,
                      const cl_mem inData0,
                      int inWidthStride1,
                      const cl_mem inData1,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addF32(inData0, height, width, 3, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, queue);

  return code;
}

template <>
RetCode Add<float, 4>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride0,
                      const cl_mem inData0,
                      int inWidthStride1,
                      const cl_mem inData1,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addF32(inData0, height, width, 4, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, queue);

  return code;
}

/*************************** addWeighted operation ***************************/

RetCode addWeightedU8(const cl_mem src0, int rows, int cols, int channels,
                      int src0_stride, float alpha, const cl_mem src1,
                      int src1_stride, float beta, float gamma, cl_mem dst,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain);

  int columns = cols * channels;
  if (src0_stride == columns && src1_stride == columns &&
      dst_stride == columns) {
    // std::cout << "came in if()" << std::endl;
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    size_t local_size[]  = {512, 1};
    size_t global_size[] = {(size_t)roundUp(cols, 512, 9), 1};

    frame_chain->setCompileOptions("-D ADDWEIGHTED_U81D");
    runOclKernel(frame_chain, "addWeightedU8Kernel0", 2, global_size,
                 local_size, src0, columns, alpha, src1, beta, gamma, dst);
  }
  else {
    // std::cout << "came in else()" << std::endl;
    columns = cols * channels;
    cols = divideUp(columns, 4, 2);
    size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
    size_t global_size[] = {(size_t)cols, (size_t)rows};

    frame_chain->setCompileOptions("-D ADDWEIGHTED_U82D");
    runOclKernel(frame_chain, "addWeightedU8Kernel1", 2, global_size,
                 local_size, src0, rows, columns, src0_stride, alpha, src1,
                 src1_stride, beta, gamma, dst, dst_stride);
  }

  return RC_SUCCESS;
}

RetCode addWeightedF32(const cl_mem src0, int rows, int cols, int channels,
                       int src0_stride, float alpha, const cl_mem src1,
                       int src1_stride, float beta, float gamma, cl_mem dst,
                       int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain);

  int columns = cols * channels;
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)divideUp(columns, 2, 1), (size_t)rows};

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    // std::cout << "came in if()" << std::endl;
    cols = divideUp(columns, 2, 1);
    frame_chain->setCompileOptions("-D ADDWEIGHTED_F32ALIGNED");
    runOclKernel(frame_chain, "addWeightedF32Kernel0", 2, global_size,
                 local_size, src0, rows, cols, src0_stride, alpha, src1,
                 src1_stride, beta, gamma, dst, dst_stride);
  }
  else {
    // std::cout << "came in else()" << std::endl;
    frame_chain->setCompileOptions("-D ADDWEIGHTED_F32UNALIGNED");
    runOclKernel(frame_chain, "addWeightedF32Kernel1", 2, global_size,
                 local_size, src0, rows, columns, src0_stride, alpha, src1,
                 src1_stride, beta, gamma, dst, dst_stride);
  }

  return RC_SUCCESS;
}

template <>
RetCode AddWeighted<uchar, 1>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride0,
                              const cl_mem inData0,
                              float alpha,
                              int inWidthStride1,
                              const cl_mem inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              cl_mem outData) {
  RetCode code = addWeightedU8(inData0, height, width, 1, inWidthStride0,
                               alpha, inData1, inWidthStride1, beta, gamma,
                               outData, outWidthStride, queue);

  return code;
}

template <>
RetCode AddWeighted<uchar, 3>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride0,
                              const cl_mem inData0,
                              float alpha,
                              int inWidthStride1,
                              const cl_mem inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              cl_mem outData) {
  RetCode code = addWeightedU8(inData0, height, width, 3, inWidthStride0,
                               alpha, inData1, inWidthStride1, beta, gamma,
                               outData, outWidthStride, queue);

  return code;
}

template <>
RetCode AddWeighted<uchar, 4>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride0,
                              const cl_mem inData0,
                              float alpha,
                              int inWidthStride1,
                              const cl_mem inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              cl_mem outData) {
  RetCode code = addWeightedU8(inData0, height, width, 4, inWidthStride0,
                               alpha, inData1, inWidthStride1, beta, gamma,
                               outData, outWidthStride, queue);

  return code;
}

template <>
RetCode AddWeighted<float, 1>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride0,
                              const cl_mem inData0,
                              float alpha,
                              int inWidthStride1,
                              const cl_mem inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              cl_mem outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addWeightedF32(inData0, height, width, 1, inWidthStride0,
                                alpha, inData1, inWidthStride1, beta, gamma,
                                outData, outWidthStride, queue);

  return code;
}

template <>
RetCode AddWeighted<float, 3>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride0,
                              const cl_mem inData0,
                              float alpha,
                              int inWidthStride1,
                              const cl_mem inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              cl_mem outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addWeightedF32(inData0, height, width, 3, inWidthStride0,
                                alpha, inData1, inWidthStride1, beta, gamma,
                                outData, outWidthStride, queue);

  return code;
}

template <>
RetCode AddWeighted<float, 4>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride0,
                              const cl_mem inData0,
                              float alpha,
                              int inWidthStride1,
                              const cl_mem inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              cl_mem outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addWeightedF32(inData0, height, width, 4, inWidthStride0,
                                alpha, inData1, inWidthStride1, beta, gamma,
                                outData, outWidthStride, queue);

  return code;
}


}  // namespace ocl
}  // namespace cv
}  // namespace ppl
