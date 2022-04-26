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

#include "ppl/cv/cuda/norm.h"

#include "utility/utility.hpp"
#include "norm.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

RetCode norm(const uchar* src, int rows, int cols, int channels, int src_stride,
             NormTypes norm_type, const uchar* mask, int mask_stride,
             double* norm_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(norm_value != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(norm_type == NORM_INF || norm_type == NORM_L1 ||
             norm_type == NORM_L2);
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  cols *= channels;
  int grid_y, columns = divideUp(cols, 4, 2);
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(columns, BLOCK_SIZE, BLOCK_SHIFT);
  // Launchs about MAX_BLOCKS thread blocks on a GPU.
  grid_y  = MAX_BLOCKS / grid.x;
  grid.y  = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  size_t norms_size = blocks * sizeof(long long);
  long long* norm_values;
  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMalloc(norms_size, buffer_block);
    norm_values = (long long*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&norm_values, norms_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  if (norm_type == NORM_INF) {
    normLinfKernel<uchar, long long><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks, norm_values);
  }
  else if (norm_type == NORM_L1) {
    normL1Kernel<uchar, uint, long long><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks, norm_values);
  }
  else {  // norm_type == NORM_L2
    normL2Kernel<uchar, long long, long long><<<grid, block, 0, stream>>>(src,
        rows, cols, channels, src_stride, mask, mask_stride, blocks,
        norm_values);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(norm_values);
    }
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  long long value;
  code = cudaMemcpy(&value, norm_values, sizeof(long long),
                    cudaMemcpyDeviceToHost);
  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(norm_values);
  }
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }

  if (norm_type == NORM_L2) {
    *norm_value = sqrt(value);
  }
  else {
    *norm_value = value;
  }

  return RC_SUCCESS;
}

RetCode norm(const float* src, int rows, int cols, int channels, int src_stride,
             NormTypes norm_type, const uchar* mask, int mask_stride,
             double* norm_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(norm_value != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(norm_type == NORM_INF || norm_type == NORM_L1 ||
             norm_type == NORM_L2);
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  cols *= channels;
  int grid_y, columns = divideUp(cols, 4, 2);
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(columns, BLOCK_SIZE, BLOCK_SHIFT);
  // Launchs about MAX_BLOCKS thread blocks on a GPU.
  grid_y  = MAX_BLOCKS / grid.x;
  grid.y  = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  size_t norms_size = blocks * sizeof(double);
  double* norm_values;
  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMalloc(norms_size, buffer_block);
    norm_values = (double*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&norm_values, norms_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  if (norm_type == NORM_INF) {
    normLinfKernel<float, double><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, blocks, norm_values);
  }
  else if (norm_type == NORM_L1) {
    normL1Kernel<float, float, double><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks,
        norm_values);
  }
  else {  // norm_type == NORM_L2
    normL2Kernel<float, float, double><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks,
        norm_values);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(norm_values);
    }
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  double value;
  code = cudaMemcpy(&value, norm_values, sizeof(double),
                    cudaMemcpyDeviceToHost);
  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(norm_values);
  }
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }

  if (norm_type == NORM_L2) {
    *norm_value = sqrt(value);
  }
  else {
    *norm_value = value;
  }

  return RC_SUCCESS;
}

template <>
RetCode Norm<uchar, 1>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       double* normValue,
                       NormTypes normType,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = norm(inData, inHeight, inWidth, 1, inWidthStride, normType,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<uchar, 3>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       double* normValue,
                       NormTypes normType,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = norm(inData, inHeight, inWidth, 3, inWidthStride, normType,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<uchar, 4>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       double* normValue,
                       NormTypes normType,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = norm(inData, inHeight, inWidth, 4, inWidthStride, normType,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<float, 1>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       double* normValue,
                       NormTypes normType,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = norm(inData, inHeight, inWidth, 1, inWidthStride, normType,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<float, 3>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       double* normValue,
                       NormTypes normType,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = norm(inData, inHeight, inWidth, 3, inWidthStride, normType,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<float, 4>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       double* normValue,
                       NormTypes normType,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = norm(inData, inHeight, inWidth, 4, inWidthStride, normType,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
