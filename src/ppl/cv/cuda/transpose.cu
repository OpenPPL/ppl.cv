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

#include "ppl/cv/cuda/transpose.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define LENGTH 16
#define SHIFT  4

template <typename T>
__global__
void transposeKernel(const T* src, int rows, int cols, int src_stride, T* dst,
                     int dst_stride) {
  __shared__ T data[LENGTH][LENGTH + 1];
  int element_x = (blockIdx.x << SHIFT) + threadIdx.x;
  int element_y = (blockIdx.y << SHIFT) + threadIdx.y;

  if (blockIdx.x < gridDim.x - 1 && blockIdx.y < gridDim.y - 1) {
    T* input = (T*)((uchar*)src + element_y * src_stride);
    T value = input[element_x];
    data[threadIdx.x][threadIdx.y] = value;
    __syncthreads();

    int dst_x = (blockIdx.y << SHIFT) + threadIdx.x;
    int dst_y = (blockIdx.x << SHIFT) + threadIdx.y;
    T* output = (T*)((uchar*)dst + dst_y * dst_stride);
    output[dst_x] = data[threadIdx.y][threadIdx.x];
  }
  else {
    if (element_x < cols && element_y < rows) {
      T* input = (T*)((uchar*)src + element_y * src_stride);
      T value = input[element_x];
      data[threadIdx.x][threadIdx.y] = value;
    }
    __syncthreads();

    int threadIdx_x = blockIdx.y == gridDim.y - 1 ?
                        rows - (blockIdx.y << SHIFT) : LENGTH;
    int threadIdx_y = blockIdx.x == gridDim.x - 1 ?
                        cols - (blockIdx.x << SHIFT) : LENGTH;
    if (threadIdx.x < threadIdx_x && threadIdx.y < threadIdx_y) {
      int dst_x = (blockIdx.y << SHIFT) + threadIdx.x;
      int dst_y = (blockIdx.x << SHIFT) + threadIdx.y;

      T* output = (T*)((uchar*)dst + dst_y * dst_stride);
      output[dst_x] = data[threadIdx.y][threadIdx.x];
    }
  }
}

RetCode transpose(const uchar* src, int rows, int cols, int channels,
                  int src_stride, uchar* dst, int dst_stride,
                  cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= rows * channels * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = LENGTH;
  block.y = LENGTH;
  grid.x  = divideUp(cols, LENGTH, SHIFT);
  grid.y  = divideUp(rows, LENGTH, SHIFT);

  if (channels == 1) {
    transposeKernel<<<grid, block, 0, stream>>>((uchar*)src, rows, cols,
        src_stride, (uchar*)dst, dst_stride);
  }
  else if (channels == 3) {
    transposeKernel<<<grid, block, 0, stream>>>((uchar3*)src, rows, cols,
        src_stride, (uchar3*)dst, dst_stride);
  }
  else {
    transposeKernel<<<grid, block, 0, stream>>>((uchar4*)src, rows, cols,
        src_stride, (uchar4*)dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode transpose(const float* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride,
                  cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= rows * channels * (int)sizeof(float));

  dim3 block, grid;
  block.x = LENGTH;
  block.y = LENGTH;
  grid.x  = divideUp(cols, LENGTH, SHIFT);
  grid.y  = divideUp(rows, LENGTH, SHIFT);

  if (channels == 1) {
    transposeKernel<<<grid, block, 0, stream>>>((float*)src, rows, cols,
        src_stride, (float*)dst, dst_stride);
  }
  else if (channels == 3) {
    transposeKernel<<<grid, block, 0, stream>>>((float3*)src, rows, cols,
        src_stride, (float3*)dst, dst_stride);
  }
  else {
    transposeKernel<<<grid, block, 0, stream>>>((float4*)src, rows, cols,
        src_stride, (float4*)dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Transpose<uchar, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int outWidthStride,
                            uchar* outData) {
  RetCode code = transpose(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, stream);

  return code;
}

template <>
RetCode Transpose<uchar, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int outWidthStride,
                            uchar* outData) {
  RetCode code = transpose(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, stream);

  return code;
}

template <>
RetCode Transpose<uchar, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int outWidthStride,
                            uchar* outData) {
  RetCode code = transpose(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, stream);

  return code;
}

template <>
RetCode Transpose<float, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int outWidthStride,
                            float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = transpose(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, stream);

  return code;
}

template <>
RetCode Transpose<float, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int outWidthStride,
                            float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = transpose(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, stream);

  return code;
}

template <>
RetCode Transpose<float, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int outWidthStride,
                            float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = transpose(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
