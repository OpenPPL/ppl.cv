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

#include "ppl/cv/ocl/split.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/split.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode split3U8(const cl_mem src, int rows, int cols, int src_stride,
                 cl_mem dst0, cl_mem dst1, cl_mem dst2, int dst_stride,
                 cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 3 * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, split);

  int columns = cols;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == 3 * columns * (int)sizeof(uchar) &&
      dst_stride == columns * (int)sizeof(uchar)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D SPLIT3_U81D");
    runOclKernel(frame_chain, "split3U8Kernel0", 2, global_size, local_size,
                 src, columns, dst0, dst1, dst2);
  }
  else {
    frame_chain->setCompileOptions("-D SPLIT3_U82D");
    runOclKernel(frame_chain, "split3U8Kernel1", 2, global_size, local_size,
                 src, rows, columns, src_stride, dst0, dst1, dst2, dst_stride);
  }
  return RC_SUCCESS;
}

RetCode split4U8(const cl_mem src, int rows, int cols, int src_stride,
                 cl_mem dst0, cl_mem dst1, cl_mem dst2, cl_mem dst3,
                 int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(dst3 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 4 * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, split);

  int columns = cols;
  cols = columns;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == 4 * columns * (int)sizeof(uchar) &&
      dst_stride == columns * (int)sizeof(uchar)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D SPLIT4_U81D");
    runOclKernel(frame_chain, "split4U8Kernel0", 2, global_size, local_size,
                 src, columns, dst0, dst1, dst2, dst3);
  }
  else {
    frame_chain->setCompileOptions("-D SPLIT4_U82D");
    runOclKernel(frame_chain, "split4U8Kernel1", 2, global_size, local_size,
                 src, rows, columns, src_stride, dst0, dst1, dst2, dst3,
                 dst_stride);
  }
  return RC_SUCCESS;
}

RetCode split3F32(const cl_mem src, int rows, int cols, int src_stride,
                  cl_mem dst0, cl_mem dst1, cl_mem dst2, int dst_stride,
                  cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 3 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, split);
  int columns = cols;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == 3 * columns * (int)sizeof(float) &&
      dst_stride == columns * (int)sizeof(float)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D SPLIT3_F321D");
    runOclKernel(frame_chain,
                 "split3F32Kernel0",
                 2, global_size, local_size, src, columns, dst0, dst1, dst2);
  }
  else {
    frame_chain->setCompileOptions("-D SPLIT3_F322D");
    runOclKernel(frame_chain,
                 "split3F32Kernel1",
                 2, global_size, local_size, src, rows, columns, src_stride,
                 dst0, dst1, dst2, dst_stride);
  }
  return RC_SUCCESS;
}

RetCode split4F32(const cl_mem src, int rows, int cols, int src_stride,
                  cl_mem dst0, cl_mem dst1, cl_mem dst2, cl_mem dst3,
                  int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(dst3 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 4 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, split);
  int columns = cols;
  cols = columns;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == 4 * columns * (int)sizeof(float) &&
      dst_stride == columns * (int)sizeof(float)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D SPLIT4_F321D");
    runOclKernel(frame_chain,
                 "split4F32Kernel0",
                 2, global_size, local_size, src, columns, dst0, dst1, dst2,
                 dst3);
  }
  else {
    frame_chain->setCompileOptions("-D SPLIT4_F322D");
    runOclKernel(frame_chain,
                 "split4F32Kernel1",
                 2, global_size, local_size, src, rows, columns, src_stride,
                 dst0, dst1, dst2, dst3, dst_stride);
  }
  return RC_SUCCESS;
}

template <>
RetCode Split3Channels<uchar>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData0,
                              cl_mem outData1,
                              cl_mem outData2) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = split3U8(inData, height, width, inWidthStride, outData0,
                          outData1, outData2, outWidthStride, queue);
  return code;
}

template <>
RetCode Split4Channels<uchar>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData0,
                              cl_mem outData1,
                              cl_mem outData2,
                              cl_mem outData3) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = split4U8(inData, height, width, inWidthStride, outData0,
                          outData1, outData2, outData3, outWidthStride, queue);
  return code;
}

template <>
RetCode Split3Channels<float>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData0,
                              cl_mem outData1,
                              cl_mem outData2) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = split3F32(inData, height, width, inWidthStride, outData0,
                           outData1, outData2, outWidthStride, queue);
  return code;
}

template <>
RetCode Split4Channels<float>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData0,
                              cl_mem outData1,
                              cl_mem outData2,
                              cl_mem outData3) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = split4F32(inData, height, width, inWidthStride, outData0,
                           outData1, outData2, outData3, outWidthStride, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
