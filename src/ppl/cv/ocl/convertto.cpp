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

#include "ppl/cv/ocl/convertto.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/convertto.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode converttoF32_2_F32(const cl_mem src0, int rows, int cols, int channels,
                           int src0_stride, cl_mem dst, int dst_stride,
                           float scale, float delta, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, convertto);
  int columns = cols * channels;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)divideUp(columns, 2, 1), (size_t)rows};
  if ((src0_stride & 7) == 0 && (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    frame_chain->setCompileOptions("-D CONVERTTO_F32_2_F32ALIGNED");
    runOclKernel(frame_chain,
                 "converttoF32_2_F32Kernel0",
                 2, global_size, local_size, src0, rows, cols, src0_stride, dst,
                 dst_stride, scale, delta);
  }
  else {
    frame_chain->setCompileOptions("-D CONVERTTO_F32_2_F32UNALIGNED");
    runOclKernel(frame_chain,
                 "converttoF32_2_F32Kernel1",
                 2, global_size, local_size, src0, rows, columns, src0_stride,
                 dst, dst_stride, scale, delta);
  }
  return RC_SUCCESS;
}

RetCode converttoU8_2_F32(const cl_mem src0, int rows, int cols, int channels,
                          int src0_stride, cl_mem dst, int dst_stride,
                          float scale, float delta, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, convertto);
  int columns = cols * channels;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)divideUp(columns, 2, 1), (size_t)rows};
  if ((src0_stride & 1) == 0 && (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    frame_chain->setCompileOptions("-D CONVERTTO_U8_2_F32ALIGNED");
    runOclKernel(frame_chain,
                 "converttoU8_2_F32Kernel0",
                 2, global_size, local_size, src0, rows, cols, src0_stride, dst,
                 dst_stride, scale, delta);
  }
  else {
    frame_chain->setCompileOptions("-D CONVERTTO_U8_2_F32UNALIGNED");
    runOclKernel(frame_chain,
                 "converttoU8_2_F32Kernel1",
                 2, global_size, local_size, src0, rows, columns, src0_stride,
                 dst, dst_stride, scale, delta);
  }
  return RC_SUCCESS;
}

RetCode converttoF32_2_U8(const cl_mem src0, int rows, int cols, int channels,
                          int src0_stride, cl_mem dst, int dst_stride,
                          float scale, float delta, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, convertto);
  int columns = cols * channels;
  if (src0_stride * (int)sizeof(float) == columns && dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    size_t local_size[] = {512, 1};
    size_t global_size[] = {(size_t)roundUp(cols, 512, 9), 1};
    frame_chain->setCompileOptions("-D CONVERTTO_F32_2_U81D");
    runOclKernel(frame_chain,
                 "converttoF32_2_U8Kernel0",
                 2, global_size, local_size, src0, columns, dst, scale, delta);
  }
  else {
    columns = cols * channels;
    cols = divideUp(columns, 4, 2);
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    size_t global_size[] = {(size_t)cols, (size_t)rows};
    frame_chain->setCompileOptions("-D CONVERTTO_F32_2_U82D");
    runOclKernel(frame_chain,
                 "converttoF32_2_U8Kernel1",
                 2, global_size, local_size, src0, rows, columns, src0_stride,
                 dst, dst_stride, scale, delta);
  }
  return RC_SUCCESS;
}

RetCode converttoU8_2_U8(const cl_mem src0, int rows, int cols, int channels,
                         int src0_stride, cl_mem dst, int dst_stride,
                         float scale, float delta, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, convertto);
  int columns = cols * channels;
  if (src0_stride * (int)sizeof(uchar) == columns && dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    size_t local_size[] = {512, 1};
    size_t global_size[] = {(size_t)roundUp(cols, 512, 9), 1};
    frame_chain->setCompileOptions("-D CONVERTTO_U8_2_U81D");
    runOclKernel(frame_chain,
                 "converttoU8_2_U8Kernel0",
                 2, global_size, local_size, src0, columns, dst, scale, delta);
  }
  else {
    columns = cols * channels;
    cols = divideUp(columns, 4, 2);
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    size_t global_size[] = {(size_t)cols, (size_t)rows};
    frame_chain->setCompileOptions("-D CONVERTTO_U8_2_U82D");
    runOclKernel(frame_chain,
                 "converttoU8_2_U8Kernel1",
                 2, global_size, local_size, src0, rows, columns, src0_stride,
                 dst, dst_stride, scale, delta);
  }
  return RC_SUCCESS;
}

template <>
RetCode ConvertTo<float, float, 1>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      converttoF32_2_F32(inData0, height, width, 1, inWidthStride0, outData,
                         outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<float, float, 3>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      converttoF32_2_F32(inData0, height, width, 3, inWidthStride0, outData,
                         outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<float, float, 4>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      converttoF32_2_F32(inData0, height, width, 4, inWidthStride0, outData,
                         outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<uchar, float, 1>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(uchar);
  outWidthStride *= sizeof(float);
  RetCode code =
      converttoU8_2_F32(inData0, height, width, 1, inWidthStride0, outData,
                        outWidthStride, scale, delta, queue);
  return code;
}
template <>
RetCode ConvertTo<uchar, float, 3>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(uchar);
  outWidthStride *= sizeof(float);
  RetCode code =
      converttoU8_2_F32(inData0, height, width, 3, inWidthStride0, outData,
                        outWidthStride, scale, delta, queue);
  return code;
}
template <>
RetCode ConvertTo<uchar, float, 4>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(uchar);
  outWidthStride *= sizeof(float);
  RetCode code =
      converttoU8_2_F32(inData0, height, width, 4, inWidthStride0, outData,
                        outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<float, uchar, 1>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(float);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      converttoF32_2_U8(inData0, height, width, 1, inWidthStride0, outData,
                        outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<float, uchar, 3>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(float);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      converttoF32_2_U8(inData0, height, width, 3, inWidthStride0, outData,
                        outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<float, uchar, 4>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(float);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      converttoF32_2_U8(inData0, height, width, 4, inWidthStride0, outData,
                        outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<uchar, uchar, 1>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = converttoU8_2_U8(inData0, height, width, 1, inWidthStride0,
                                  outData, outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<uchar, uchar, 3>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = converttoU8_2_U8(inData0, height, width, 3, inWidthStride0,
                                  outData, outWidthStride, scale, delta, queue);
  return code;
}

template <>
RetCode ConvertTo<uchar, uchar, 4>(cl_command_queue queue,
                                   int height,
                                   int width,
                                   int inWidthStride0,
                                   const cl_mem inData0,
                                   int outWidthStride,
                                   cl_mem outData,
                                   float scale,
                                   float delta) {
  inWidthStride0 *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = converttoU8_2_U8(inData0, height, width, 4, inWidthStride0,
                                  outData, outWidthStride, scale, delta, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
