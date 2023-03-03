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

#include "ppl/cv/ocl/flip.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/flip.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode flipU8(const cl_mem src, int rows, int cols, int channels,
               int src_stride, cl_mem dst, int dst_stride, int flip_code,
               cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, flip);

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};

  if (channels == 1) {
    global_size[0] = divideUp(global_size[0], 4, 2);
    frame_chain->setCompileOptions("-D FLIP_U8C1");
    runOclKernel(frame_chain, "flipU8Kernel0", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride, flip_code);
  }
  else if (channels == 3) {
    frame_chain->setCompileOptions("-D FLIP_U8C3");
    runOclKernel(frame_chain, "flipU8Kernel1", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride, flip_code);
  }
  else {  // channels == 4
    frame_chain->setCompileOptions("-D FLIP_U8C4");
    runOclKernel(frame_chain, "flipU8Kernel2", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride, flip_code);
  }

  return RC_SUCCESS;
}

RetCode flipF32(const cl_mem src, int rows, int cols, int channels,
                int src_stride, cl_mem dst, int dst_stride, int flip_code,
                cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, flip);

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};

  if (channels == 1) {
    frame_chain->setCompileOptions("-D FLIP_F32C1");
    runOclKernel(frame_chain, "flipF32Kernel0", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride, flip_code);
  }
  else if (channels == 3) {
    frame_chain->setCompileOptions("-D FLIP_F32C3");
    runOclKernel(frame_chain, "flipF32Kernel1", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride, flip_code);
  }
  else {  // channels == 4
    frame_chain->setCompileOptions("-D FLIP_F32C4");
    runOclKernel(frame_chain, "flipF32Kernel2", 2, global_size, local_size, src,
                 rows, cols, src_stride, dst, dst_stride, flip_code);
  }

  return RC_SUCCESS;
}

template <>
RetCode Flip<uchar, 1>(cl_command_queue queue,
                       int height,
                       int width,
                       int inWidthStride,
                       const cl_mem inData,
                       int outWidthStride,
                       cl_mem outData,
                       int flipCode) {
  RetCode code = flipU8(inData, height, width, 1, inWidthStride, outData,
                        outWidthStride, flipCode, queue);

  return code;
}

template <>
RetCode Flip<uchar, 3>(cl_command_queue queue,
                       int height,
                       int width,
                       int inWidthStride,
                       const cl_mem inData,
                       int outWidthStride,
                       cl_mem outData,
                       int flipCode) {
  RetCode code = flipU8(inData, height, width, 3, inWidthStride, outData,
                        outWidthStride, flipCode, queue);

  return code;
}

template <>
RetCode Flip<uchar, 4>(cl_command_queue queue,
                       int height,
                       int width,
                       int inWidthStride,
                       const cl_mem inData,
                       int outWidthStride,
                       cl_mem outData,
                       int flipCode) {
  RetCode code = flipU8(inData, height, width, 4, inWidthStride, outData,
                        outWidthStride, flipCode, queue);

  return code;
}

template <>
RetCode Flip<float, 1>(cl_command_queue queue,
                       int height,
                       int width,
                       int inWidthStride,
                       const cl_mem inData,
                       int outWidthStride,
                       cl_mem outData,
                       int flipCode) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = flipF32(inData, height, width, 1, inWidthStride, outData,
                         outWidthStride, flipCode, queue);

  return code;
}

template <>
RetCode Flip<float, 3>(cl_command_queue queue,
                       int height,
                       int width,
                       int inWidthStride,
                       const cl_mem inData,
                       int outWidthStride,
                       cl_mem outData,
                       int flipCode) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = flipF32(inData, height, width, 3, inWidthStride, outData,
                         outWidthStride, flipCode, queue);

  return code;
}

template <>
RetCode Flip<float, 4>(cl_command_queue queue,
                       int height,
                       int width,
                       int inWidthStride,
                       const cl_mem inData,
                       int outWidthStride,
                       cl_mem outData,
                       int flipCode) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = flipF32(inData, height, width, 4, inWidthStride, outData,
                         outWidthStride, flipCode, queue);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
