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

#include "ppl/cv/ocl/warpperspective.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/warpperspective.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode warpPerspective(const cl_mem src, int src_rows, int src_cols, int channels,
                   int src_stride, const float* perspective, cl_mem dst,
                   int dst_rows, int dst_cols, int dst_stride,
                   InterpolationType interpolation, BorderType border_type,
                   uchar border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(perspective != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_TRANSPARENT);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, warpperspective);

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};

  if (interpolation == INTERPOLATION_LINEAR) {
    frame_chain->setCompileOptions("-D WARPPERSPECTIVE_LINEAR_U8");
    runOclKernel(frame_chain, "warpperspectiveLinearU8Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 perspective[0], perspective[1], perspective[2],
                 perspective[3], perspective[4], perspective[5],
                 perspective[6], perspective[7], perspective[8], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    frame_chain->setCompileOptions("-D WARPPERSPECTIVE_NP_U8");
    runOclKernel(frame_chain, "warpperspectiveNPU8Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 perspective[0], perspective[1], perspective[2],
                 perspective[3], perspective[4], perspective[5],
                 perspective[6], perspective[7], perspective[8], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else {
  }

  return RC_SUCCESS;
}

RetCode warpPerspective(const cl_mem src, int src_rows, int src_cols, int channels,
                   int src_stride, const float* perspective, cl_mem dst,
                   int dst_rows, int dst_cols, int dst_stride,
                   InterpolationType interpolation, BorderType border_type,
                   float border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(perspective != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(float));
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_TRANSPARENT);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, warpperspective);

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};

  if (interpolation == INTERPOLATION_LINEAR) {
    frame_chain->setCompileOptions("-D WARPPERSPECTIVE_LINEAR_F32");
    runOclKernel(frame_chain, "warpperspectiveLinearF32Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 perspective[0], perspective[1], perspective[2],
                 perspective[3], perspective[4], perspective[5],
                 perspective[6], perspective[7], perspective[8], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    frame_chain->setCompileOptions("-D WARPPERSPECTIVE_NP_F32");
    runOclKernel(frame_chain, "warpperspectiveNPF32Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 perspective[0], perspective[1], perspective[2],
                 perspective[3], perspective[4], perspective[5],
                 perspective[6], perspective[7], perspective[8], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else {
  }

  return RC_SUCCESS;
}

template <>
RetCode WarpPerspective<uchar, 1>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* perspectiveMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 1, inWidthStride,
                            perspectiveMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpPerspective<uchar, 3>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* perspectiveMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 3, inWidthStride,
                            perspectiveMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpPerspective<uchar, 4>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* perspectiveMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 4, inWidthStride,
                            perspectiveMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpPerspective<float, 1>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* perspectiveMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpPerspective(inData, inHeight, inWidth, 1, inWidthStride,
                            perspectiveMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpPerspective<float, 3>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* perspectiveMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpPerspective(inData, inHeight, inWidth, 3, inWidthStride,
                            perspectiveMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpPerspective<float, 4>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* perspectiveMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpPerspective(inData, inHeight, inWidth, 4, inWidthStride,
                            perspectiveMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
