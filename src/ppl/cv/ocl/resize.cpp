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

#include "ppl/cv/ocl/resize.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/resize.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode resizeU8(const cl_mem src, int src_rows, int src_cols, int channels,
                 int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                 int dst_stride, InterpolationType interpolation,
                 cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT ||
             interpolation == INTERPOLATION_AREA);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, resize);

  cl_int error_code = 0;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      error_code = clEnqueueCopyBuffer(queue, src, dst, 0, 0,
                                       src_rows * src_stride * sizeof(uchar), 0,
                                       NULL, NULL);
      CHECK_ERROR(error_code, clEnqueueCopyBuffer);
      if (error_code != CL_SUCCESS) {
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};

  float col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;
  float inv_col_scale = 1.f / col_scale;
  float inv_row_scale = 1.f / row_scale;

  if (interpolation == INTERPOLATION_LINEAR) {
    frame_chain->setCompileOptions("-D RESIZE_LINEAR_U8");
    runOclKernel(frame_chain, "resizeLinearU8Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    frame_chain->setCompileOptions("-D RESIZE_NP_U8");
    runOclKernel(frame_chain, "resizeNPU8Kernel", 2, global_size, local_size,
                 src, src_rows, src_cols, channels, src_stride, dst, dst_rows,
                 dst_cols, dst_stride, col_scale, row_scale);
  }
  else if (interpolation == INTERPOLATION_AREA) {
    if (src_cols > dst_cols && src_rows > dst_rows) {
      if (src_cols % dst_cols == 0 && src_rows % dst_rows == 0) {
        frame_chain->setCompileOptions("-D RESIZE_AREA0_U8");
        runOclKernel(frame_chain, "resizeAreaU8Kernel0", 2, global_size,
                     local_size, src, src_rows, src_cols, channels, src_stride,
                     dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale);
      }
      else {
        frame_chain->setCompileOptions("-D RESIZE_AREA1_U8");
        runOclKernel(frame_chain, "resizeAreaU8Kernel1", 2, global_size,
                     local_size, src, src_rows, src_cols, channels, src_stride,
                     dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale);
      }
    }
    else {
      frame_chain->setCompileOptions("-D RESIZE_AREA2_U8");
      runOclKernel(frame_chain, "resizeAreaU8Kernel2", 2, global_size,
                   local_size, src, src_rows, src_cols, channels, src_stride,
                   dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale,
                   inv_col_scale, inv_row_scale);
    }
  }
  else {
  }

  return RC_SUCCESS;
}

RetCode resizeF32(const cl_mem src, int src_rows, int src_cols, int channels,
                  int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                  int dst_stride, InterpolationType interpolation,
                  cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT ||
             interpolation == INTERPOLATION_AREA);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, resize);

  cl_int error_code = 0;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      error_code = clEnqueueCopyBuffer(queue, src, dst, 0, 0,
                                       src_rows * src_stride * sizeof(float), 0,
                                       NULL, NULL);
      CHECK_ERROR(error_code, clEnqueueCopyBuffer);
      if (error_code != CL_SUCCESS) {
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};

  float col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;
  float inv_col_scale = 1.f / col_scale;
  float inv_row_scale = 1.f / row_scale;

  if (interpolation == INTERPOLATION_LINEAR) {
    frame_chain->setCompileOptions("-D RESIZE_LINEAR_F32");
    runOclKernel(frame_chain, "resizeLinearF32Kernel", 2, global_size,
                 local_size, src, src_rows, src_cols, channels, src_stride,
                 dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    frame_chain->setCompileOptions("-D RESIZE_NP_F32");
    runOclKernel(frame_chain, "resizeNPF32Kernel", 2, global_size, local_size,
                 src, src_rows, src_cols, channels, src_stride, dst, dst_rows,
                 dst_cols, dst_stride, col_scale, row_scale);
  }
  else if (interpolation == INTERPOLATION_AREA) {
    if (src_cols > dst_cols && src_rows > dst_rows) {
      if (src_cols % dst_cols == 0 && src_rows % dst_rows == 0) {
        frame_chain->setCompileOptions("-D RESIZE_AREA0_F32");
        runOclKernel(frame_chain, "resizeAreaF32Kernel0", 2, global_size,
                     local_size, src, src_rows, src_cols, channels, src_stride,
                     dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale);
      }
      else {
        frame_chain->setCompileOptions("-D RESIZE_AREA1_F32");
        runOclKernel(frame_chain, "resizeAreaF32Kernel1", 2, global_size,
                     local_size, src, src_rows, src_cols, channels, src_stride,
                     dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale);
      }
    }
    else {
      frame_chain->setCompileOptions("-D RESIZE_AREA2_F32");
      runOclKernel(frame_chain, "resizeAreaF32Kernel2", 2, global_size,
                   local_size, src, src_rows, src_cols, channels, src_stride,
                   dst, dst_rows, dst_cols, dst_stride, col_scale, row_scale,
                   inv_col_scale, inv_row_scale);
    }
  }
  else {
  }

  return RC_SUCCESS;
}

template <>
RetCode Resize<uchar, 1>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         InterpolationType interpolation) {
  RetCode code = resizeU8(inData, inHeight, inWidth, 1, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, interpolation,
                          queue);

  return code;
}

template <>
RetCode Resize<uchar, 3>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         InterpolationType interpolation) {
  RetCode code = resizeU8(inData, inHeight, inWidth, 3, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, interpolation,
                          queue);

  return code;
}

template <>
RetCode Resize<uchar, 4>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         InterpolationType interpolation) {
  RetCode code = resizeU8(inData, inHeight, inWidth, 4, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, interpolation,
                          queue);

  return code;
}

template <>
RetCode Resize<float, 1>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         InterpolationType interpolation) {
  RetCode code = resizeF32(inData, inHeight, inWidth, 1, inWidthStride, outData,
                           outHeight, outWidth, outWidthStride, interpolation,
                           queue);

  return code;
}

template <>
RetCode Resize<float, 3>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         InterpolationType interpolation) {
  RetCode code = resizeF32(inData, inHeight, inWidth, 3, inWidthStride, outData,
                           outHeight, outWidth, outWidthStride, interpolation,
                           queue);

  return code;
}

template <>
RetCode Resize<float, 4>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         InterpolationType interpolation) {
  RetCode code = resizeF32(inData, inHeight, inWidth, 4, inWidthStride, outData,
                           outHeight, outWidth, outWidthStride, interpolation,
                           queue);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
