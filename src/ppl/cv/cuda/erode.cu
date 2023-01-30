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

#include "ppl/cv/cuda/erode.h"
#include "morphology.hpp"

#include <cfloat>

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

struct MinSwap {
  __DEVICE__
  void initialize(uchar &value0, uchar &value1, uchar &value2, uchar &value3) {
    value0 = 255;
    value1 = 255;
    value2 = 255;
    value3 = 255;
  }

  __DEVICE__
  void initialize(uchar &value) {
    value = 255;
  }

  __DEVICE__
  void initialize(uchar3 &value) {
    value.x = 255;
    value.y = 255;
    value.z = 255;
  }

  __DEVICE__
  void initialize(uchar4 &value) {
    value.x = 255;
    value.y = 255;
    value.z = 255;
    value.w = 255;
  }

  __DEVICE__
  void initialize(float &value) {
    value = FLT_MAX;
  }

  __DEVICE__
  void initialize(float3 &value) {
    value.x = FLT_MAX;
    value.y = FLT_MAX;
    value.z = FLT_MAX;
  }

  __DEVICE__
  void initialize(float4 &value) {
    value.x = FLT_MAX;
    value.y = FLT_MAX;
    value.z = FLT_MAX;
    value.w = FLT_MAX;
  }

  __DEVICE__
  void operator()(uchar &value, uchar &target) {
    value = value > target ? target : value;
  }

  __DEVICE__
  void operator()(uchar3 &value, uchar3 &target) {
    value.x = value.x > target.x ? target.x : value.x;
    value.y = value.y > target.y ? target.y : value.y;
    value.z = value.z > target.z ? target.z : value.z;
  }

  __DEVICE__
  void operator()(uchar4 &value, uchar4 &target) {
    value.x = value.x > target.x ? target.x : value.x;
    value.y = value.y > target.y ? target.y : value.y;
    value.z = value.z > target.z ? target.z : value.z;
    value.w = value.w > target.w ? target.w : value.w;
  }

  __DEVICE__
  void operator()(float &value, float &target) {
    value = value > target ? target : value;
  }

  __DEVICE__
  void operator()(float3 &value, float3 &target) {
    value.x = value.x > target.x ? target.x : value.x;
    value.y = value.y > target.y ? target.y : value.y;
    value.z = value.z > target.z ? target.z : value.z;
  }

  __DEVICE__
  void operator()(float4 &value, float4 &target) {
    value.x = value.x > target.x ? target.x : value.x;
    value.y = value.y > target.y ? target.y : value.y;
    value.z = value.z > target.z ? target.z : value.z;
    value.w = value.w > target.w ? target.w : value.w;
  }

  __DEVICE__
  void checkConstantResult(uchar &result, uchar border_value) {
    result = result > border_value ? border_value : result;
  }

  __DEVICE__
  void checkConstantResult(uchar3 &result, uchar border_value) {
    result.x = result.x > border_value ? border_value : result.x;
    result.y = 0;
    result.z = 0;
  }

  __DEVICE__
  void checkConstantResult(uchar4 &result, uchar border_value) {
    result.x = result.x > border_value ? border_value : result.x;
    result.y = 0;
    result.z = 0;
    result.w = 0;
  }

  __DEVICE__
  void checkConstantResult(float &result, float border_value) {
    result = result > border_value ? border_value : result;
  }

  __DEVICE__
  void checkConstantResult(float3 &result, float border_value) {
    result.x = result.x > border_value ? border_value : result.x;
    result.y = 0;
    result.z = 0;
  }

  __DEVICE__
  void checkConstantResult(float4 &result, float border_value) {
    result.x = result.x > border_value ? border_value : result.x;
    result.y = 0;
    result.z = 0;
    result.w = 0;
  }

  __DEVICE__
  void checkU8C1ConstantResult(uchar4 &result, uchar border_value,
                               bool constant_border0, bool constant_border1,
                               bool constant_border2, bool constant_border3) {
    if (constant_border0) {
      result.x = result.x > border_value ? border_value : result.x;
    }
    if (constant_border1) {
      result.y = result.y > border_value ? border_value : result.y;
    }
    if (constant_border2) {
      result.z = result.z > border_value ? border_value : result.z;
    }
    if (constant_border3) {
      result.w = result.w > border_value ? border_value : result.w;
    }
  }
};

RetCode erode(const uchar* src, int rows, int cols, int channels,
              int src_stride, uchar* dst, int dst_stride, const uchar* kernel,
              int kernel_y, int kernel_x, BorderType border_type,
              const uchar border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 0 && cols > 0);
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

  cudaError_t code;
  if (kernel_x == 1 && kernel_y == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, src_stride * rows, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int radius_x = kernel_x >> 1;
  int radius_y = kernel_y >> 1;

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  bool all_masked = true;
  if (kernel != NULL) {
    int count = kernel_y * kernel_x;
    for (int index = 0; index < count; index++) {
      if (kernel[index] != 1) {
        all_masked = false;
        break;
      }
    }
  }

  MinSwap morphology_swap;

  if (all_masked) {
    uchar* buffer;
    size_t pitch;

    if (channels == 1) {
      int left_threads = divideUp(radius_x, 4, 2);
      int remainders = cols & 3;
      remainders = remainders > radius_x ? remainders : radius_x;
      int aligned_columns = (cols - remainders) >> 2;
      int right_threads = cols - (aligned_columns << 2);
      int columns = aligned_columns + right_threads;

      if ((left_threads << 2) + right_threads <= cols) {
        dim3 block0, grid0;
        block0.x = kBlockDimX0;
        block0.y = kBlockDimY0;
        grid0.x  = divideUp(columns, kBlockDimX0, kBlockShiftX0);
        grid0.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

        if (rows >= 480 && cols >= 640 && kernel_y >= 7 && kernel_x >= 7) {
          GpuMemoryBlock buffer_block;
          if (memoryPoolUsed()) {
            pplCudaMallocPitch(cols * channels * sizeof(uchar), rows,
                               buffer_block);
            buffer = (uchar*)(buffer_block.data);
            pitch  = buffer_block.pitch;
          }
          else {
            code = cudaMallocPitch(&buffer, &pitch,
                                   cols * channels * sizeof(uchar), rows);
            if (code != cudaSuccess) {
              LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
              return RC_DEVICE_MEMORY_ERROR;
            }
          }

          morphRowU8C1Kernel0<MinSwap><<<grid0, block0, 0, stream>>>(src, rows,
              cols, columns, src_stride, left_threads, aligned_columns,
              radius_x, buffer, pitch, morphology_swap);
          morphColKernel0<uchar, uchar, MinSwap><<<grid, block, 0, stream>>>(
              buffer, rows, cols, pitch, radius_x, radius_y, dst, dst_stride,
              border_type, border_value, morphology_swap);

          if (memoryPoolUsed()) {
            pplCudaFree(buffer_block);
          }
          else {
            cudaFree(buffer);
          }
        }
        else {
          morph2DU8C1Kernel0<MinSwap><<<grid0, block0, 0, stream>>>(src, rows,
              cols, columns, src_stride, left_threads, aligned_columns,
              radius_x, radius_y, dst, dst_stride, border_type, border_value,
              morphology_swap);
        }
      }
      else {
        morph2DKernel0<uchar, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);
      }
    }
    else if (channels == 3) {
      if (rows >= 480 && cols >= 640 && kernel_y >= 7 && kernel_x >= 7) {
        GpuMemoryBlock buffer_block;
        if (memoryPoolUsed()) {
          pplCudaMallocPitch(cols * channels * sizeof(uchar), rows,
                             buffer_block);
          buffer = (uchar*)(buffer_block.data);
          pitch  = buffer_block.pitch;
        }
        else {
          code = cudaMallocPitch(&buffer, &pitch,
                                 cols * channels * sizeof(uchar), rows);
          if (code != cudaSuccess) {
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
            return RC_DEVICE_MEMORY_ERROR;
          }
        }

        morphRowKernel0<uchar3, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, buffer, pitch, morphology_swap);
        morphColKernel0<uchar3, uchar, MinSwap><<<grid, block, 0, stream>>>(
            buffer, rows, cols, pitch, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);

        if (memoryPoolUsed()) {
          pplCudaFree(buffer_block);
        }
        else {
          cudaFree(buffer);
        }
      }
      else {
        morph2DKernel0<uchar3, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);
      }
    }
    else {  // channels == 4
      if (rows >= 780 && cols >= 1024 && kernel_y >= 7 && kernel_x >= 7) {
        GpuMemoryBlock buffer_block;
        if (memoryPoolUsed()) {
          pplCudaMallocPitch(cols * channels * sizeof(uchar), rows,
                             buffer_block);
          buffer = (uchar*)(buffer_block.data);
          pitch  = buffer_block.pitch;
        }
        else {
          code = cudaMallocPitch(&buffer, &pitch,
                                 cols * channels * sizeof(uchar), rows);
          if (code != cudaSuccess) {
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
            return RC_DEVICE_MEMORY_ERROR;
          }
        }

        morphRowKernel0<uchar4, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, buffer, pitch, morphology_swap);
        morphColKernel0<uchar4, uchar, MinSwap><<<grid, block, 0, stream>>>(
            buffer, rows, cols, pitch, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);

        if (memoryPoolUsed()) {
          pplCudaFree(buffer_block);
        }
        else {
          cudaFree(buffer);
        }
      }
      else {
        morph2DKernel0<uchar4, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);
      }
    }
  }
  else {
    uchar* mask;
    size_t size = kernel_y * kernel_x * sizeof(uchar);

    GpuMemoryBlock buffer_block;
    if (memoryPoolUsed()) {
      pplCudaMalloc(size, buffer_block);
      mask = buffer_block.data;
    }
    else {
      code = cudaMalloc(&mask, size);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }

    code = cudaMemcpy(mask, kernel, size, cudaMemcpyHostToDevice);
    if (code != cudaSuccess) {
      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block);
      }
      else {
        cudaFree(mask);
      }
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }

    if (channels == 1) {
      int left_threads = divideUp(radius_x, 4, 2);
      int remainders = cols & 3;
      remainders = remainders > radius_x ? remainders : radius_x;
      int aligned_columns = (cols - remainders) >> 2;
      int right_threads = cols - (aligned_columns << 2);
      int columns = aligned_columns + right_threads;

      if ((left_threads << 2) + right_threads <= cols) {
        dim3 block0, grid0;
        block0.x = kBlockDimX0;
        block0.y = kBlockDimY0;
        grid0.x  = divideUp(columns, kBlockDimX0, kBlockShiftX0);
        grid0.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

        morph2DU8C1Kernel1<MinSwap><<<grid0, block0, 0, stream>>>(src, rows,
            cols, columns, src_stride, mask, left_threads, aligned_columns,
            radius_x, radius_y, kernel_x, kernel_y, dst, dst_stride,
            border_type, border_value, morphology_swap);
      }
      else {
        morph2DKernel1<uchar, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, mask, radius_x, radius_y, kernel_x,
            kernel_y, dst, dst_stride, border_type, border_value,
            morphology_swap);
      }
    }
    else if (channels == 3) {
      morph2DKernel1<uchar3, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
          rows, cols, src_stride, mask, radius_x, radius_y, kernel_x, kernel_y,
          dst, dst_stride, border_type, border_value, morphology_swap);
    }
    else {
      morph2DKernel1<uchar4, uchar, MinSwap><<<grid, block, 0, stream>>>(src,
          rows, cols, src_stride, mask, radius_x, radius_y, kernel_x, kernel_y,
          dst, dst_stride, border_type, border_value, morphology_swap);
    }

    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(mask);
    }
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }
  else {
    return RC_SUCCESS;
  }
}

RetCode erode(const float* src, int rows, int cols, int channels,
              int src_stride, float* dst, int dst_stride, const uchar* kernel,
              int kernel_y, int kernel_x, BorderType border_type,
              const float border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 0 && cols > 0);
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

  cudaError_t code;
  if (kernel_x == 1 && kernel_y == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, src_stride * rows, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int radius_x = kernel_x >> 1;
  int radius_y = kernel_y >> 1;

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  bool all_masked = true;
  if (kernel != NULL) {
    int count = kernel_y * kernel_x;
    for (int index = 0; index < count; index++) {
      if (kernel[index] != 1) {
        all_masked = false;
        break;
      }
    }
  }

  MinSwap morphology_swap;

  if (all_masked) {
    float* buffer;
    size_t pitch;

    if (channels == 1) {
      if (rows >= 480 && cols >= 640 && kernel_y >= 7 && kernel_x >= 7) {
        GpuMemoryBlock buffer_block;
        if (memoryPoolUsed()) {
          pplCudaMallocPitch(cols * channels * sizeof(float), rows,
                             buffer_block);
          buffer = (float*)(buffer_block.data);
          pitch  = buffer_block.pitch;
        }
        else {
          code = cudaMallocPitch(&buffer, &pitch,
                                 cols * channels * sizeof(float), rows);
          if (code != cudaSuccess) {
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
            return RC_DEVICE_MEMORY_ERROR;
          }
        }

        morphRowKernel0<float, float, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, buffer, pitch, morphology_swap);
        morphColKernel0<float, float, MinSwap><<<grid, block, 0, stream>>>(
            buffer, rows, cols, pitch, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);

        if (memoryPoolUsed()) {
          pplCudaFree(buffer_block);
        }
        else {
          cudaFree(buffer);
        }
      }
      else {
        morph2DKernel0<float, float, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);
      }
    }
    else if (channels == 3) {
      if (rows >= 480 && cols >= 640 && kernel_y >= 7 && kernel_x >= 7) {
        GpuMemoryBlock buffer_block;
        if (memoryPoolUsed()) {
          pplCudaMallocPitch(cols * channels * sizeof(float), rows,
                             buffer_block);
          buffer = (float*)(buffer_block.data);
          pitch  = buffer_block.pitch;
        }
        else {
          code = cudaMallocPitch(&buffer, &pitch,
                                 cols * channels * sizeof(float), rows);
          if (code != cudaSuccess) {
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
            return RC_DEVICE_MEMORY_ERROR;
          }
        }

        morphRowKernel0<float3, float, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, buffer, pitch, morphology_swap);
        morphColKernel0<float3, float, MinSwap><<<grid, block, 0, stream>>>(
            buffer, rows, cols, pitch, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);

        if (memoryPoolUsed()) {
          pplCudaFree(buffer_block);
        }
        else {
          cudaFree(buffer);
        }
      }
      else {
        morph2DKernel0<float3, float, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);
      }
    }
    else {  // channels == 4
      if (rows >= 480 && cols >= 640 && kernel_y >= 7 && kernel_x >= 7) {
        GpuMemoryBlock buffer_block;
        if (memoryPoolUsed()) {
          pplCudaMallocPitch(cols * channels * sizeof(float), rows,
                             buffer_block);
          buffer = (float*)(buffer_block.data);
          pitch  = buffer_block.pitch;
        }
        else {
          code = cudaMallocPitch(&buffer, &pitch,
                                 cols * channels * sizeof(float), rows);
          if (code != cudaSuccess) {
            LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
            return RC_DEVICE_MEMORY_ERROR;
          }
        }

        morphRowKernel0<float4, float, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, buffer, pitch, morphology_swap);
        morphColKernel0<float4, float, MinSwap><<<grid, block, 0, stream>>>(
            buffer, rows, cols, pitch, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);

        if (memoryPoolUsed()) {
          pplCudaFree(buffer_block);
        }
        else {
          cudaFree(buffer);
        }
      }
      else {
        morph2DKernel0<float4, float, MinSwap><<<grid, block, 0, stream>>>(src,
            rows, cols, src_stride, radius_x, radius_y, dst, dst_stride,
            border_type, border_value, morphology_swap);
      }
    }
  }
  else {
    uchar* mask;
    size_t size = kernel_y * kernel_x * sizeof(uchar);

    GpuMemoryBlock buffer_block;
    if (memoryPoolUsed()) {
      pplCudaMalloc(size, buffer_block);
      mask = buffer_block.data;
    }
    else {
      code = cudaMalloc(&mask, size);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }

    code = cudaMemcpy(mask, kernel, size, cudaMemcpyHostToDevice);
    if (code != cudaSuccess) {
      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block);
      }
      else {
        cudaFree(mask);
      }
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }

    if (channels == 1) {
      morph2DKernel1<float, float, MinSwap><<<grid, block, 0, stream>>>(src,
          rows, cols, src_stride, mask, radius_x, radius_y, kernel_x, kernel_y,
          dst, dst_stride, border_type, border_value, morphology_swap);
    }
    else if (channels == 3) {
      morph2DKernel1<float3, float, MinSwap><<<grid, block, 0, stream>>>(src,
          rows, cols, src_stride, mask, radius_x, radius_y, kernel_x, kernel_y,
          dst, dst_stride, border_type, border_value, morphology_swap);
    }
    else {
      morph2DKernel1<float4, float, MinSwap><<<grid, block, 0, stream>>>(src,
          rows, cols, src_stride, mask, radius_x, radius_y, kernel_x, kernel_y,
          dst, dst_stride, border_type, border_value, morphology_swap);
    }

    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(mask);
    }
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }
  else {
    return RC_SUCCESS;
  }
}

template <>
RetCode Erode<uchar, 1>(cudaStream_t stream,
                        int height,
                        int width,
                        int inWidthStride,
                        const uchar* inData,
                        int kernelx_len,
                        int kernely_len,
                        const uchar* kernel,
                        int outWidthStride,
                        uchar* outData,
                        BorderType border_type,
                        const uchar border_value) {
  RetCode code = erode(inData, height, width, 1, inWidthStride, outData,
                       outWidthStride, kernel, kernely_len, kernelx_len,
                       border_type, border_value, stream);

  return code;
}

template <>
RetCode Erode<uchar, 3>(cudaStream_t stream,
                        int height,
                        int width,
                        int inWidthStride,
                        const uchar* inData,
                        int kernelx_len,
                        int kernely_len,
                        const uchar* kernel,
                        int outWidthStride,
                        uchar* outData,
                        BorderType border_type,
                        const uchar border_value) {
  RetCode code = erode(inData, height, width, 3, inWidthStride, outData,
                       outWidthStride, kernel, kernely_len, kernelx_len,
                       border_type, border_value, stream);

  return code;
}

template <>
RetCode Erode<uchar, 4>(cudaStream_t stream,
                        int height,
                        int width,
                        int inWidthStride,
                        const uchar* inData,
                        int kernelx_len,
                        int kernely_len,
                        const uchar* kernel,
                        int outWidthStride,
                        uchar* outData,
                        BorderType border_type,
                        const uchar border_value) {
  RetCode code = erode(inData, height, width, 4, inWidthStride, outData,
                       outWidthStride, kernel, kernely_len, kernelx_len,
                       border_type, border_value, stream);

  return code;
}

template <>
RetCode Erode<float, 1>(cudaStream_t stream,
                        int height,
                        int width,
                        int inWidthStride,
                        const float* inData,
                        int kernelx_len,
                        int kernely_len,
                        const uchar* kernel,
                        int outWidthStride,
                        float* outData,
                        BorderType border_type,
                        const float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = erode(inData, height, width, 1, inWidthStride, outData,
                       outWidthStride, kernel, kernely_len, kernelx_len,
                       border_type, border_value, stream);

  return code;
}

template <>
RetCode Erode<float, 3>(cudaStream_t stream,
                        int height,
                        int width,
                        int inWidthStride,
                        const float* inData,
                        int kernelx_len,
                        int kernely_len,
                        const uchar* kernel,
                        int outWidthStride,
                        float* outData,
                        BorderType border_type,
                        const float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = erode(inData, height, width, 3, inWidthStride, outData,
                       outWidthStride, kernel, kernely_len, kernelx_len,
                       border_type, border_value, stream);

  return code;
}

template <>
RetCode Erode<float, 4>(cudaStream_t stream,
                        int height,
                        int width,
                        int inWidthStride,
                        const float* inData,
                        int kernelx_len,
                        int kernely_len,
                        const uchar* kernel,
                        int outWidthStride,
                        float* outData,
                        BorderType border_type,
                        const float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = erode(inData, height, width, 4, inWidthStride, outData,
                       outWidthStride, kernel, kernely_len, kernelx_len,
                       border_type, border_value, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
