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

#include "ppl/cv/ocl/copymakeborder.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/copymakeborder.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode copyMakeBorder(const cl_mem src, int rows, int cols, int channels,
                       int src_stride, cl_mem dst, int dst_stride, int top,
                       int bottom, int left, int right, BorderType border_type,
                       uchar border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(top >= 0);
  PPL_ASSERT(bottom >= 0);
  PPL_ASSERT(left >= 0);
  PPL_ASSERT(right >= 0);
  PPL_ASSERT(top == bottom);
  PPL_ASSERT(left == right);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_WRAP ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, copymakeborder);

  cl_int error_code = 0;
  if (top == 0 && bottom == 0 && left == 0 && right == 0 &&
      src_stride == dst_stride) {
    if (src != dst) {
      error_code = clEnqueueCopyBuffer(queue, src, dst, 0, 0, rows * src_stride,
                                       0, NULL, NULL);
      CHECK_ERROR(error_code, clEnqueueCopyBuffer);
      if (error_code != CL_SUCCESS) {
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)(cols + left + right),
                          (size_t)(rows + top + bottom)};

  if (channels == 1) {
    global_size[0] = divideUp(global_size[0], 4, 2);
    frame_chain->setCompileOptions("-D COPYMAKEBORDER_U8C1");
    runOclKernel(frame_chain, "copyMakeBorderU8Kernel0", 2, global_size,
                 local_size, src, rows, cols, src_stride, dst, dst_stride, top,
                 left, border_type, border_value);
  }
  else if (channels == 3) {
    frame_chain->setCompileOptions("-D COPYMAKEBORDER_U8C3");
    runOclKernel(frame_chain, "copyMakeBorderU8Kernel1", 2, global_size,
                 local_size, src, rows, cols, src_stride, dst, dst_stride, top,
                 left, border_type, border_value);
  }
  else {
    frame_chain->setCompileOptions("-D COPYMAKEBORDER_U8C4");
    runOclKernel(frame_chain, "copyMakeBorderU8Kernel2", 2, global_size,
                 local_size, src, rows, cols, src_stride, dst, dst_stride, top,
                 left, border_type, border_value);
  }

  return RC_SUCCESS;
}

RetCode copyMakeBorder(const cl_mem src, int rows, int cols, int channels,
                       int src_stride, cl_mem dst, int dst_stride, int top,
                       int bottom, int left, int right, BorderType border_type,
                       float border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(top >= 0);
  PPL_ASSERT(bottom >= 0);
  PPL_ASSERT(left >= 0);
  PPL_ASSERT(right >= 0);
  PPL_ASSERT(top == bottom);
  PPL_ASSERT(left == right);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_WRAP ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, copymakeborder);

  cl_int error_code = 0;
  if (top == 0 && bottom == 0 && left == 0 && right == 0 &&
      src_stride == dst_stride) {
    if (src != dst) {
      error_code = clEnqueueCopyBuffer(queue, src, dst, 0, 0, rows * src_stride,
                                       0, NULL, NULL);
      CHECK_ERROR(error_code, clEnqueueCopyBuffer);
      if (error_code != CL_SUCCESS) {
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)(cols + left + right),
                          (size_t)(rows + top + bottom)};

  if (channels == 1) {
    frame_chain->setCompileOptions("-D COPYMAKEBORDER_F32C1");
    runOclKernel(frame_chain, "copyMakeBorderF32Kernel0", 2, global_size,
                 local_size, src, rows, cols, src_stride, dst, dst_stride, top,
                 left, border_type, border_value);
  }
  else if (channels == 3) {
    frame_chain->setCompileOptions("-D COPYMAKEBORDER_F32C3");
    runOclKernel(frame_chain, "copyMakeBorderF32Kernel1", 2, global_size,
                 local_size, src, rows, cols, src_stride, dst, dst_stride, top,
                 left, border_type, border_value);
  }
  else {
    frame_chain->setCompileOptions("-D COPYMAKEBORDER_F32C4");
    runOclKernel(frame_chain, "copyMakeBorderF32Kernel2", 2, global_size,
                 local_size, src, rows, cols, src_stride, dst, dst_stride, top,
                 left, border_type, border_value);
  }

  return RC_SUCCESS;
}

template <>
RetCode CopyMakeBorder<uchar, 1>(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const cl_mem inData,
                                 int outWidthStride,
                                 cl_mem outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 uchar border_value) {
  RetCode code = copyMakeBorder(inData, height, width, 1, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, queue);

  return code;
}

template <>
RetCode CopyMakeBorder<uchar, 3>(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const cl_mem inData,
                                 int outWidthStride,
                                 cl_mem outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 uchar border_value) {
  RetCode code = copyMakeBorder(inData, height, width, 3, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, queue);

  return code;
}

template <>
RetCode CopyMakeBorder<uchar, 4>(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const cl_mem inData,
                                 int outWidthStride,
                                 cl_mem outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 uchar border_value) {
  RetCode code = copyMakeBorder(inData, height, width, 4, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, queue);

  return code;
}

template <>
RetCode CopyMakeBorder<float, 1>(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const cl_mem inData,
                                 int outWidthStride,
                                 cl_mem outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = copyMakeBorder(inData, height, width, 1, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, queue);

  return code;
}

template <>
RetCode CopyMakeBorder<float, 3>(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const cl_mem inData,
                                 int outWidthStride,
                                 cl_mem outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = copyMakeBorder(inData, height, width, 3, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, queue);

  return code;
}

template <>
RetCode CopyMakeBorder<float, 4>(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const cl_mem inData,
                                 int outWidthStride,
                                 cl_mem outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = copyMakeBorder(inData, height, width, 4, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, queue);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
