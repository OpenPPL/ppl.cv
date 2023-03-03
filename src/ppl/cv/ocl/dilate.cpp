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

#include "ppl/cv/ocl/dilate.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/dilate.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode dilate(const cl_mem src, int rows, int cols, int channels,
               int src_stride, const uchar* kernel, int kernel_y, int kernel_x,
               cl_mem dst, int dst_stride, BorderType border_type,
               const uchar border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(kernel_y > 0 && kernel_y < rows);
  PPL_ASSERT(kernel_x > 0 && kernel_x < cols);
  PPL_ASSERT((kernel_y & 1) == 1 && (kernel_x & 1) == 1);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_WRAP ||
             border_type == BORDER_REFLECT_101);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, dilate);

  cl_int error_code = 0;
  if (kernel_y == 1 && kernel_x == 1 && src_stride == dst_stride) {
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

  size_t local_size[]   = {kBlockDimX0, kBlockDimY0};
  size_t global_size0[] = {(size_t)cols, (size_t)rows};

  int radius_x = kernel_x >> 1;
  int radius_y = kernel_y >> 1;

  bool all_masked = true;
  if (kernel != nullptr) {
    int count = kernel_y * kernel_x;
    for (int index = 0; index < count; index++) {
      if (kernel[index] != 1) {
        all_masked = false;
        break;
      }
    }
  }

  if (all_masked) {
    if (kernel_y <= 5 && kernel_x <= 5) {
      if (channels == 1) {
        global_size0[0] = divideUp(global_size0[0], 4, 2);
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_2D_U8C1");
        runOclKernel(frame_chain, "dilate2DU8Kernel0", 2, global_size0,
                     local_size, src, rows, cols, src_stride, radius_x,
                     radius_y, dst, dst_stride, border_type, border_value);
      }
      else if (channels == 3) {
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_2D_U8C3");
        runOclKernel(frame_chain, "dilate2DU8Kernel1", 2, global_size0,
                     local_size, src, rows, cols, src_stride, radius_x,
                     radius_y, dst, dst_stride, border_type, border_value);
      }
      else {  // channels == 4
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_2D_U8C4");
        runOclKernel(frame_chain, "dilate2DU8Kernel2", 2, global_size0,
                     local_size, src, rows, cols, src_stride, radius_x,
                     radius_y, dst, dst_stride, border_type, border_value);
      }
    }
    else {
      cl_context context = frame_chain->getContext();
      cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     rows * src_stride, NULL, &error_code);
      CHECK_ERROR(error_code, clCreateBuffer);
      if (error_code != CL_SUCCESS) {
        return RC_DEVICE_MEMORY_ERROR;
      }

      size_t global_size1[] = {(size_t)divideUp(cols, 4, 2), (size_t)rows};
      size_t global_size2[] = {(size_t)cols, (size_t)divideUp(rows, 4, 2)};
      if (channels == 1) {
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_SEP2D_U8_C1");
        runOclKernel(frame_chain, "dilateRowU8Kernel0", 2, global_size1,
                     local_size, src, rows, cols, src_stride, radius_x, buffer,
                     src_stride, border_type, border_value);
        runOclKernel(frame_chain, "dilateColU8Kernel0", 2, global_size1,
                     local_size, buffer, rows, cols, src_stride, radius_y, dst,
                     dst_stride, border_type, border_value);
      }
      else if (channels == 3) {
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_SEP2D_U8_C3");
        runOclKernel(frame_chain, "dilateRowU8Kernel1", 2, global_size1,
                     local_size, src, rows, cols, src_stride, radius_x, buffer,
                     src_stride, border_type, border_value);
        runOclKernel(frame_chain, "dilateColU8Kernel1", 2, global_size2,
                     local_size, buffer, rows, cols, src_stride, radius_y, dst,
                     dst_stride, border_type, border_value);
      }
      else {  // channels == 4
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_SEP2D_U8_C4");
        runOclKernel(frame_chain, "dilateRowU8Kernel2", 2, global_size1,
                     local_size, src, rows, cols, src_stride, radius_x, buffer,
                     src_stride, border_type, border_value);
        runOclKernel(frame_chain, "dilateColU8Kernel2", 2, global_size2,
                     local_size, buffer, rows, cols, src_stride, radius_y, dst,
                     dst_stride, border_type, border_value);
      }

      clReleaseMemObject(buffer);
    }
  }
  else {
    size_t size = kernel_y * kernel_x * sizeof(uchar);
    cl_context context = frame_chain->getContext();
    cl_mem mask = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL,
                                 &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    if (error_code != CL_SUCCESS) {
      return RC_DEVICE_MEMORY_ERROR;
    }
    cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->
                             getQueue();
    error_code = clEnqueueWriteBuffer(queue, mask, CL_FALSE, 0, size,
                                      kernel, 0, NULL, NULL);
    CHECK_ERROR(error_code, clEnqueueWriteBuffer);
    if (error_code != CL_SUCCESS) {
      clReleaseMemObject(mask);
      return RC_DEVICE_MEMORY_ERROR;
    }

    if (channels == 1) {
      frame_chain->setCompileOptions("-D DILATE_PARTIALLY_MASKED_2D_U8C1");
      runOclKernel(frame_chain, "dilate2DU8Kernel3", 2, global_size0,
                   local_size, src, rows, cols, src_stride, mask, radius_x,
                   radius_y, kernel_x, kernel_y, dst, dst_stride, border_type,
                   border_value);
    }
    else if (channels == 3) {
      frame_chain->setCompileOptions("-D DILATE_PARTIALLY_MASKED_2D_U8C3");
      runOclKernel(frame_chain, "dilate2DU8Kernel4", 2, global_size0,
                   local_size, src, rows, cols, src_stride, mask, radius_x,
                   radius_y, kernel_x, kernel_y, dst, dst_stride, border_type,
                   border_value);
    }
    else {  // channels == 4
      frame_chain->setCompileOptions("-D DILATE_PARTIALLY_MASKED_2D_U8C4");
      runOclKernel(frame_chain, "dilate2DU8Kernel5", 2, global_size0,
                   local_size, src, rows, cols, src_stride, mask, radius_x,
                   radius_y, kernel_x, kernel_y, dst, dst_stride, border_type,
                   border_value);
    }

    clReleaseMemObject(mask);
  }

  return RC_SUCCESS;
}

RetCode dilate(const cl_mem src, int rows, int cols, int channels,
               int src_stride, const uchar* kernel, int kernel_y, int kernel_x,
               cl_mem dst, int dst_stride, BorderType border_type,
               const float border_value, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(kernel_y > 0 && kernel_y < rows);
  PPL_ASSERT(kernel_x > 0 && kernel_x < cols);
  PPL_ASSERT((kernel_y & 1) == 1 && (kernel_x & 1) == 1);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_WRAP ||
             border_type == BORDER_REFLECT_101);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, dilate);

  cl_int error_code = 0;
  if (kernel_y == 1 && kernel_x == 1 && src_stride == dst_stride) {
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

  size_t local_size[]   = {kBlockDimX0, kBlockDimY0};
  size_t global_size0[] = {(size_t)cols, (size_t)rows};

  int radius_x = kernel_x >> 1;
  int radius_y = kernel_y >> 1;

  bool all_masked = true;
  if (kernel != nullptr) {
    int count = kernel_y * kernel_x;
    for (int index = 0; index < count; index++) {
      if (kernel[index] != 1) {
        all_masked = false;
        break;
      }
    }
  }

  if (all_masked) {
    if (kernel_y <= 5 && kernel_x <= 5) {
      if (channels == 1) {
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_2D_F32C1");
        runOclKernel(frame_chain, "dilate2DF32Kernel0", 2, global_size0,
                     local_size, src, rows, cols, src_stride, radius_x,
                     radius_y, dst, dst_stride, border_type, border_value);
      }
      else if (channels == 3) {
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_2D_F32C3");
        runOclKernel(frame_chain, "dilate2DF32Kernel1", 2, global_size0,
                     local_size, src, rows, cols, src_stride, radius_x,
                     radius_y, dst, dst_stride, border_type, border_value);
      }
      else {  // channels == 4
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_2D_F32C4");
        runOclKernel(frame_chain, "dilate2DF32Kernel2", 2, global_size0,
                     local_size, src, rows, cols, src_stride, radius_x,
                     radius_y, dst, dst_stride, border_type, border_value);
      }
    }
    else {
      cl_context context = frame_chain->getContext();
      cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     rows * src_stride, NULL, &error_code);
      CHECK_ERROR(error_code, clCreateBuffer);
      if (error_code != CL_SUCCESS) {
        return RC_DEVICE_MEMORY_ERROR;
      }

      size_t global_size1[] = {(size_t)divideUp(cols, 4, 2), (size_t)rows};
      size_t global_size2[] = {(size_t)cols, (size_t)divideUp(rows, 4, 2)};
      if (channels == 1) {
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_SEP2D_F32_C1");
        runOclKernel(frame_chain, "dilateRowF32Kernel0", 2, global_size1,
                     local_size, src, rows, cols, src_stride, radius_x, buffer,
                     src_stride, border_type, border_value);
        runOclKernel(frame_chain, "dilateColF32Kernel0", 2, global_size2,
                     local_size, buffer, rows, cols, src_stride, radius_y, dst,
                     dst_stride, border_type, border_value);
      }
      else if (channels == 3) {
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_SEP2D_F32_C3");
        runOclKernel(frame_chain, "dilateRowF32Kernel1", 2, global_size1,
                     local_size, src, rows, cols, src_stride, radius_x, buffer,
                     src_stride, border_type, border_value);
        runOclKernel(frame_chain, "dilateColF32Kernel1", 2, global_size2,
                     local_size, buffer, rows, cols, src_stride, radius_y, dst,
                     dst_stride, border_type, border_value);
      }
      else {  // channels == 4
        frame_chain->setCompileOptions("-D DILATE_FULLLY_MASKED_SEP2D_F32_C4");
        runOclKernel(frame_chain, "dilateRowF32Kernel2", 2, global_size1,
                     local_size, src, rows, cols, src_stride, radius_x, buffer,
                     src_stride, border_type, border_value);
        runOclKernel(frame_chain, "dilateColF32Kernel2", 2, global_size2,
                     local_size, buffer, rows, cols, src_stride, radius_y, dst,
                     dst_stride, border_type, border_value);
      }

      clReleaseMemObject(buffer);
    }
  }
  else {
    size_t size = kernel_y * kernel_x * sizeof(uchar);
    cl_context context = frame_chain->getContext();
    cl_mem mask = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL,
                                 &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    if (error_code != CL_SUCCESS) {
      return RC_DEVICE_MEMORY_ERROR;
    }
    cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->
                             getQueue();
    error_code = clEnqueueWriteBuffer(queue, mask, CL_FALSE, 0, size,
                                      kernel, 0, NULL, NULL);
    CHECK_ERROR(error_code, clEnqueueWriteBuffer);
    if (error_code != CL_SUCCESS) {
      clReleaseMemObject(mask);
      return RC_DEVICE_MEMORY_ERROR;
    }

    if (channels == 1) {
      frame_chain->setCompileOptions("-D DILATE_PARTIALLY_MASKED_2D_F32C1");
      runOclKernel(frame_chain, "dilate2DF32Kernel3", 2, global_size0,
                   local_size, src, rows, cols, src_stride, mask, radius_x,
                   radius_y, kernel_x, kernel_y, dst, dst_stride, border_type,
                   border_value);
    }
    else if (channels == 3) {
      frame_chain->setCompileOptions("-D DILATE_PARTIALLY_MASKED_2D_F32C3");
      runOclKernel(frame_chain, "dilate2DF32Kernel4", 2, global_size0,
                   local_size, src, rows, cols, src_stride, mask, radius_x,
                   radius_y, kernel_x, kernel_y, dst, dst_stride, border_type,
                   border_value);
    }
    else {  // channels == 4
      frame_chain->setCompileOptions("-D DILATE_PARTIALLY_MASKED_2D_F32C4");
      runOclKernel(frame_chain, "dilate2DF32Kernel5", 2, global_size0,
                   local_size, src, rows, cols, src_stride, mask, radius_x,
                   radius_y, kernel_x, kernel_y, dst, dst_stride, border_type,
                   border_value);
    }

    clReleaseMemObject(mask);
  }

  return RC_SUCCESS;
}

template <>
RetCode Dilate<uchar, 1>(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int kernelx_len,
                         int kernely_len,
                         const uchar* kernel,
                         int outWidthStride,
                         cl_mem outData,
                         BorderType border_type,
                         const uchar border_value) {
  RetCode code = dilate(inData, height, width, 1, inWidthStride, kernel,
                        kernely_len, kernelx_len, outData, outWidthStride,
                        border_type, border_value, queue);

  return code;
}

template <>
RetCode Dilate<uchar, 3>(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int kernelx_len,
                         int kernely_len,
                         const uchar* kernel,
                         int outWidthStride,
                         cl_mem outData,
                         BorderType border_type,
                         const uchar border_value) {
  RetCode code = dilate(inData, height, width, 3, inWidthStride, kernel,
                        kernely_len, kernelx_len, outData, outWidthStride,
                        border_type, border_value, queue);

  return code;
}

template <>
RetCode Dilate<uchar, 4>(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int kernelx_len,
                         int kernely_len,
                         const uchar* kernel,
                         int outWidthStride,
                         cl_mem outData,
                         BorderType border_type,
                         const uchar border_value) {
  RetCode code = dilate(inData, height, width, 4, inWidthStride, kernel,
                        kernely_len, kernelx_len, outData, outWidthStride,
                        border_type, border_value, queue);

  return code;
}

template <>
RetCode Dilate<float, 1>(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int kernelx_len,
                         int kernely_len,
                         const uchar* kernel,
                         int outWidthStride,
                         cl_mem outData,
                         BorderType border_type,
                         const float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = dilate(inData, height, width, 1, inWidthStride, kernel,
                        kernely_len, kernelx_len, outData, outWidthStride,
                        border_type, border_value, queue);

  return code;
}

template <>
RetCode Dilate<float, 3>(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int kernelx_len,
                         int kernely_len,
                         const uchar* kernel,
                         int outWidthStride,
                         cl_mem outData,
                         BorderType border_type,
                         const float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = dilate(inData, height, width, 3, inWidthStride, kernel,
                        kernely_len, kernelx_len, outData, outWidthStride,
                        border_type, border_value, queue);

  return code;
}

template <>
RetCode Dilate<float, 4>(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride,
                         const cl_mem inData,
                         int kernelx_len,
                         int kernely_len,
                         const uchar* kernel,
                         int outWidthStride,
                         cl_mem outData,
                         BorderType border_type,
                         const float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = dilate(inData, height, width, 4, inWidthStride, kernel,
                        kernely_len, kernelx_len, outData, outWidthStride,
                        border_type, border_value, queue);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
