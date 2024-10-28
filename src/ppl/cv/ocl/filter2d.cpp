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

#include "ppl/cv/ocl/filter2d.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/filter2d.cl"
#include "kerneltypes.h"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode filter2DU8(const cl_mem src, int rows, int cols, int channels,
                   int src_stride, const cl_mem kernel, int ksize, cl_mem dst,
                   int dst_stride, float delta, BorderType border_type,
                   cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(kernel != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101);
  int radius = ksize >> 1;
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, filter2d);
  int global_cols;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  if (channels == 1)
    global_cols = divideUp(cols, 4, 2);
  else
    global_cols = cols;
  size_t global_size[] = {(size_t)global_cols, (size_t)rows};
  ksize = ksize >> 1;
  if (channels == 1) {
    frame_chain->setCompileOptions("-D FILTER2D_C1U8");
    runOclKernel(frame_chain, "filter2DU8C1Kernel", 2, global_size, local_size,
                 src, rows, cols, src_stride, kernel, radius, dst, dst_stride,
                 delta, border_type);
  }
  else {
    if (channels == 3) {
      frame_chain->setCompileOptions("-D FILTER2D_C3U8");
      runOclKernel(frame_chain, "filter2DU8C3Kernel", 2, global_size,
                   local_size, src, rows, cols, src_stride, kernel, radius, dst,
                   dst_stride, delta, border_type);
    }
    else {
      frame_chain->setCompileOptions("-D FILTER2D_C4U8");
      runOclKernel(frame_chain, "filter2DU8C4Kernel", 2, global_size,
                   local_size, src, rows, cols, src_stride, kernel, radius, dst,
                   dst_stride, delta, border_type);
    }
  }
  return RC_SUCCESS;
}

RetCode filter2DF32(const cl_mem src, int rows, int cols, int channels,
                    int src_stride, const cl_mem kernel, int ksize, cl_mem dst,
                    int dst_stride, float delta, BorderType border_type,
                    cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(kernel != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101);
  int radius = ksize >> 1;
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, filter2d);
  int global_cols;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  if (channels == 1)
    global_cols = divideUp(cols, 4, 2);
  else
    global_cols = cols;
  size_t global_size[] = {(size_t)global_cols, (size_t)rows};
  ksize = ksize >> 1;
  if (channels == 1) {
    frame_chain->setCompileOptions("-D FILTER2D_C1F32");
    runOclKernel(frame_chain, "filter2DF32C1Kernel", 2, global_size, local_size,
                 src, rows, cols, src_stride, kernel, radius, dst, dst_stride,
                 delta, border_type);
  }
  else {
    if (channels == 3) {
      frame_chain->setCompileOptions("-D FILTER2D_C3F32");
      runOclKernel(frame_chain, "filter2DF32C3Kernel", 2, global_size,
                   local_size, src, rows, cols, src_stride, kernel, radius, dst,
                   dst_stride, delta, border_type);
    }
    else {
      frame_chain->setCompileOptions("-D FILTER2D_C4F32");
      runOclKernel(frame_chain, "filter2DF32C4Kernel", 2, global_size,
                   local_size, src, rows, cols, src_stride, kernel, radius, dst,
                   dst_stride, delta, border_type);
    }
  }

  return RC_SUCCESS;
}

template <>
RetCode Filter2D<uchar, 1>(cl_command_queue queue,
                           int height,
                           int width,
                           int inWidthStride,
                           const cl_mem inData,
                           int ksize,
                           const cl_mem kernel,
                           int outWidthStride,
                           cl_mem outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      filter2DU8(inData, height, width, 1, inWidthStride, kernel, ksize,
                 outData, outWidthStride, delta, border_type, queue);
  return code;
}

template <>
RetCode Filter2D<uchar, 3>(cl_command_queue queue,
                           int height,
                           int width,
                           int inWidthStride,
                           const cl_mem inData,
                           int ksize,
                           const cl_mem kernel,
                           int outWidthStride,
                           cl_mem outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      filter2DU8(inData, height, width, 3, inWidthStride, kernel, ksize,
                 outData, outWidthStride, delta, border_type, queue);
  return code;
}

template <>
RetCode Filter2D<uchar, 4>(cl_command_queue queue,
                           int height,
                           int width,
                           int inWidthStride,
                           const cl_mem inData,
                           int ksize,
                           const cl_mem kernel,
                           int outWidthStride,
                           cl_mem outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      filter2DU8(inData, height, width, 4, inWidthStride, kernel, ksize,
                 outData, outWidthStride, delta, border_type, queue);
  return code;
}

template <>
RetCode Filter2D<float, 1>(cl_command_queue queue,
                           int height,
                           int width,
                           int inWidthStride,
                           const cl_mem inData,
                           int ksize,
                           const cl_mem kernel,
                           int outWidthStride,
                           cl_mem outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      filter2DF32(inData, height, width, 1, inWidthStride, kernel, ksize,
                  outData, outWidthStride, delta, border_type, queue);
  return code;
}

template <>
RetCode Filter2D<float, 3>(cl_command_queue queue,
                           int height,
                           int width,
                           int inWidthStride,
                           const cl_mem inData,
                           int ksize,
                           const cl_mem kernel,
                           int outWidthStride,
                           cl_mem outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      filter2DF32(inData, height, width, 3, inWidthStride, kernel, ksize,
                  outData, outWidthStride, delta, border_type, queue);
  return code;
}

template <>
RetCode Filter2D<float, 4>(cl_command_queue queue,
                           int height,
                           int width,
                           int inWidthStride,
                           const cl_mem inData,
                           int ksize,
                           const cl_mem kernel,
                           int outWidthStride,
                           cl_mem outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      filter2DF32(inData, height, width, 4, inWidthStride, kernel, ksize,
                  outData, outWidthStride, delta, border_type, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
