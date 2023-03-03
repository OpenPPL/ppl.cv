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

#include "ppl/cv/ocl/abs.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/abs.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode absU8(const cl_mem src, int rows, int cols, int channels,
              int src_stride, cl_mem dst, int dst_stride,
              cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(schar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(schar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, abs);

  int columns = cols * channels;
  cols = divideUp(columns, 4, 2);
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};

  if ((src_stride & 3) == 0 && (dst_stride & 3) == 0) {
    frame_chain->setCompileOptions("-D U8 -D U8ALIGNED");
    runOclKernel(frame_chain, "absU8Kernel0", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride);
  }
  else if (src_stride == columns && dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    local_size[0]  = 512;
    local_size[1]  = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D U8 -D U81D");
    runOclKernel(frame_chain, "absU8Kernel1", 2, global_size, local_size, src,
                 columns, dst);
  }
  else {
    frame_chain->setCompileOptions("-D U8 -D U8UNALIGNED");
    runOclKernel(frame_chain, "absU8Kernel2", 2, global_size, local_size, src,
                 rows, columns, src_stride, dst, dst_stride);
  }

  return RC_SUCCESS;
}

RetCode absF32(const cl_mem src, int rows, int cols, int channels,
               int src_stride, cl_mem dst, int dst_stride,
               cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, abs);

  int columns = cols * channels;
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)divideUp(columns, 2, 1), (size_t)rows};

  if ((src_stride & 7) == 0 && (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    frame_chain->setCompileOptions("-D F32ALIGNED");
    runOclKernel(frame_chain, "absF32Kernel0", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride);
  }
  else {
    frame_chain->setCompileOptions("-D F32UNALIGNED");
    runOclKernel(frame_chain, "absF32Kernel1", 2, global_size, local_size, src,
                 rows, columns, src_stride, dst, dst_stride);
  }

  return RC_SUCCESS;
}

template <>
RetCode Abs<schar, 1>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = absU8(inData, height, width, 1, inWidthStride, outData,
                       outWidthStride, queue);

  return code;
}

template <>
RetCode Abs<schar, 3>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = absU8(inData, height, width, 3, inWidthStride, outData,
                       outWidthStride, queue);

  return code;
}

template <>
RetCode Abs<schar, 4>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = absU8(inData, height, width, 4, inWidthStride, outData,
                       outWidthStride, queue);

  return code;
}

template <>
RetCode Abs<float, 1>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = absF32(inData, height, width, 1, inWidthStride, outData,
                        outWidthStride, queue);

  return code;
}

template <>
RetCode Abs<float, 3>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = absF32(inData, height, width, 3, inWidthStride, outData,
                        outWidthStride, queue);

  return code;
}

template <>
RetCode Abs<float, 4>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = absF32(inData, height, width, 4, inWidthStride, outData,
                        outWidthStride, queue);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
