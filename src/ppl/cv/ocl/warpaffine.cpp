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

#include "ppl/cv/ocl/warpaffine.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/warpaffine.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode warpAffine(const cl_mem src, int src_rows, int src_cols, int channels,
                   int src_stride, const float* affine_matrix, cl_mem dst,
                   int dst_rows, int dst_cols, int dst_stride,
                   InterpolationType interpolation, BorderType border_type,
                   uchar border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(affine_matrix != nullptr);
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
  SET_PROGRAM_SOURCE(frame_chain, warpaffine);

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};

  if (interpolation == INTERPOLATION_LINEAR) {
    frame_chain->setCompileOptions("-D WARPAFFINE_LINEAR_U8");
    runOclKernel(frame_chain, "warpaffineLinearU8Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 affine_matrix[0], affine_matrix[1], affine_matrix[2],
                 affine_matrix[3], affine_matrix[4], affine_matrix[5], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    frame_chain->setCompileOptions("-D WARPAFFINE_NP_U8");
    runOclKernel(frame_chain, "warpaffineNPU8Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 affine_matrix[0], affine_matrix[1], affine_matrix[2],
                 affine_matrix[3], affine_matrix[4], affine_matrix[5], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else {
  }

  return RC_SUCCESS;
}

RetCode warpAffine(const cl_mem src, int src_rows, int src_cols, int channels,
                   int src_stride, const float* affine_matrix, cl_mem dst,
                   int dst_rows, int dst_cols, int dst_stride,
                   InterpolationType interpolation, BorderType border_type,
                   float border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(affine_matrix != nullptr);
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
  SET_PROGRAM_SOURCE(frame_chain, warpaffine);

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};

  if (interpolation == INTERPOLATION_LINEAR) {
    frame_chain->setCompileOptions("-D WARPAFFINE_LINEAR_F32");
    runOclKernel(frame_chain, "warpaffineLinearF32Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 affine_matrix[0], affine_matrix[1], affine_matrix[2],
                 affine_matrix[3], affine_matrix[4], affine_matrix[5], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    frame_chain->setCompileOptions("-D WARPAFFINE_NP_F32");
    runOclKernel(frame_chain, "warpaffineNPF32Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 affine_matrix[0], affine_matrix[1], affine_matrix[2],
                 affine_matrix[3], affine_matrix[4], affine_matrix[5], dst,
                 dst_rows, dst_cols, dst_stride, border_type, border_value);
  }
  else {
  }

  return RC_SUCCESS;
}

template <>
RetCode WarpAffine<uchar, 1>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpAffine(inData, inHeight, inWidth, 1, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpAffine<uchar, 3>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpAffine(inData, inHeight, inWidth, 3, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpAffine<uchar, 4>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpAffine(inData, inHeight, inWidth, 4, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpAffine<float, 1>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpAffine(inData, inHeight, inWidth, 1, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpAffine<float, 3>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpAffine(inData, inHeight, inWidth, 3, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

template <>
RetCode WarpAffine<float, 4>(cl_command_queue queue,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const cl_mem inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             cl_mem outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpAffine(inData, inHeight, inWidth, 4, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, queue);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
