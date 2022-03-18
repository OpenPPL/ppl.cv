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

#include "ppl/cv/cuda/integral.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define BLOCK_X 128
#define BLOCK_Y 8

template <typename Tsrc, typename Tdst>
__global__
void verticalKernel(const Tsrc* src, int src_rows, int src_cols, int src_stride,
                    Tdst* dst, int dst_rows, int dst_stride) {
  __shared__ Tsrc data[BLOCK_Y][BLOCK_X * 2];
  __shared__ Tdst block_sum[BLOCK_X * 2];

  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 1;
  int element_y = threadIdx.y;
  if (element_x >= src_cols) {
    return;
  }

  if (sizeof(Tsrc) != 1) {
    src_stride = src_stride >> 2;
  }
  Tsrc* input = (Tsrc*)src + element_y * src_stride;
  Tdst* output;
  dst_stride = dst_stride >> 2;
  if (src_rows == dst_rows) {
    output = (Tdst*)dst + element_y * dst_stride;
  }
  else {
    output = (Tdst*)dst + (element_y + 1) * dst_stride;
  }

  if (src_rows != dst_rows && threadIdx.y == 0) {
    dst[element_x] = 0;
    dst[element_x + 1] = 0;
    if (element_x == src_cols - 1 || element_x == src_cols - 2) {
      dst[src_cols] = 0;
    }
  }

  int threadIdx_x = (threadIdx.x << 1);
  if (threadIdx.y == 0) {
    block_sum[threadIdx_x] = 0;
    block_sum[threadIdx_x + 1] = 0;
  }

  Tdst element_sum0, element_sum1;
  while (element_y < src_rows) {
    data[threadIdx.y][threadIdx_x]     = input[element_x];
    data[threadIdx.y][threadIdx_x + 1] = input[element_x + 1];
    __syncthreads();

    element_sum0 = block_sum[threadIdx_x] + data[threadIdx.y][threadIdx_x];
    element_sum1 = block_sum[threadIdx_x + 1] +
                   data[threadIdx.y][threadIdx_x + 1];
    for (int i = 0; i < threadIdx.y; i++) {
      element_sum0 += data[i][threadIdx_x];
      element_sum1 += data[i][threadIdx_x + 1];
    }
    __syncthreads();

    if (src_rows == dst_rows) {
      if (element_x < src_cols - 1) {
        output[element_x]     = element_sum0;
        output[element_x + 1] = element_sum1;
      }
      else {
        output[element_x] = element_sum0;
      }
    }
    else {
      if (element_x == 0 && src_rows != dst_rows) {
        output[0] = 0;
      }
      if (element_x < src_cols - 1) {
        output[element_x + 1] = element_sum0;
        output[element_x + 2] = element_sum1;
      }
      else {
        output[element_x + 1] = element_sum0;
      }
    }

    if (threadIdx.y == blockDim.y - 1) {
      block_sum[threadIdx_x]     = element_sum0;
      block_sum[threadIdx_x + 1] = element_sum1;
    }

    element_y += blockDim.y;
    input  += blockDim.y * src_stride;
    output += blockDim.y * dst_stride;
  }
}

template <typename T>
__global__
void horizontalKernel(int src_rows, int src_cols, T* dst, int dst_rows,
                      int dst_stride) {
  __shared__ T data[BLOCK_X];
  __shared__ T block_sum;

  int threadIdx_x = threadIdx.x;
  int element_x = threadIdx_x;
  int element_y = blockIdx.x;
  if (element_y >= src_rows) {
    return;
  }

  T* output;
  if (src_rows == dst_rows) {
    output = (T*)((uchar*)dst + element_y * dst_stride);
  }
  else {
    output = (T*)((uchar*)dst + (element_y + 1) * dst_stride);
  }

  if (element_x == 0) {
    block_sum = 0;
  }
  __syncthreads();

  T element_sum;
  while (element_x < src_cols) {
    if (src_rows == dst_rows) {
      data[threadIdx_x] = output[element_x];
    }
    else {
      data[threadIdx_x] = output[element_x + 1];
    }
    __syncthreads();

    element_sum = block_sum + data[threadIdx_x];
    for (int i = 0; i < threadIdx_x; i++) {
      element_sum += data[i];
    }
    __syncthreads();

    if (src_rows == dst_rows) {
      output[element_x] = element_sum;
    }
    else {
      if (element_x == 0 && src_rows != dst_rows) {
        output[0] = 0;
      }
      output[element_x + 1] = element_sum;
    }

    if (threadIdx_x == BLOCK_X - 1) {
      block_sum = element_sum;
    }
    element_x += BLOCK_X;
  }
}

RetCode integral(const uchar* src, int src_rows, int src_cols, int channels,
                 int src_stride, int* dst, int dst_rows, int dst_cols,
                 int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(dst_rows == src_rows || dst_rows == src_rows + 1);
  PPL_ASSERT(dst_cols == src_cols || dst_cols == src_cols + 1);
  PPL_ASSERT(channels == 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(uchar));

  dim3 block0, grid0;
  block0.x = kBlockDimX1;
  block0.y = kBlockDimY1;
  grid0.x  = divideUp(divideUp(src_cols, 2, 1), kBlockDimX1, kBlockShiftX1);
  grid0.y  = 1;
  verticalKernel<uchar, int><<<grid0, block0, 0, stream>>>(src, src_rows,
      src_cols, src_stride, dst, dst_rows, dst_stride);

  int block1 = BLOCK_X;
  int grid1  = src_rows;
  horizontalKernel<int><<<grid1, block1, 0, stream>>>(src_rows, src_cols, dst,
                                                      dst_rows, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode integral(const float* src, int src_rows, int src_cols, int channels,
                 int src_stride, float* dst, int dst_rows, int dst_cols,
                 int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(dst_rows == src_rows || dst_rows == src_rows + 1);
  PPL_ASSERT(dst_cols == src_cols || dst_cols == src_cols + 1);
  PPL_ASSERT(channels == 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(float));

  dim3 block0, grid0;
  block0.x = kBlockDimX1;
  block0.y = kBlockDimY1;
  grid0.x  = divideUp(divideUp(src_cols, 2, 1), kBlockDimX1, kBlockShiftX1);
  grid0.y  = 1;
  verticalKernel<float, float><<<grid0, block0, 0, stream>>>(src, src_rows,
      src_cols, src_stride, dst, dst_rows, dst_stride);

  int block1 = BLOCK_X;
  int grid1  = src_rows;
  horizontalKernel<float><<<grid1, block1, 0, stream>>>(src_rows, src_cols, dst,
                                                        dst_rows, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Integral<uchar, int, 1>(cudaStream_t stream,
                                int inHeight,
                                int inWidth,
                                int inWidthStride,
                                const uchar* inData,
                                int outHeight,
                                int outWidth,
                                int outWidthStride,
                                int* outData) {
  outWidthStride *= sizeof(int);
  RetCode code = integral(inData, inHeight, inWidth, 1, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, stream);

  return code;
}

template <>
RetCode Integral<float, float, 1>(cudaStream_t stream,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const float* inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = integral(inData, inHeight, inWidth, 1, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
