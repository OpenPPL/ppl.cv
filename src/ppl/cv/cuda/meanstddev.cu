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

#include "ppl/cv/cuda/meanstddev.h"
#include "mean.hpp"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename Tsrc>
__global__
void unmaskedDevC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                         uint blocks, float weight, float* mean_values,
                         float* stddev_values) {
  __shared__ float partial_sums[BLOCK_SIZE];

  int threadIdx_x = threadIdx.x;
  int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  int element_y = blockIdx.y;
  partial_sums[threadIdx_x] = 0;

  Tsrc* input;
  Tsrc value0, value1, value2, value3;
  float mean = mean_values[0];

  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      input = (Tsrc*)((uchar*)src + element_y * src_stride);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (element_x < cols - 3) {
        partial_sums[threadIdx_x] += (value0 - mean) * (value0 - mean);
        partial_sums[threadIdx_x] += (value1 - mean) * (value1 - mean);
        partial_sums[threadIdx_x] += (value2 - mean) * (value2 - mean);
        partial_sums[threadIdx_x] += (value3 - mean) * (value3 - mean);
      }
      else {
        partial_sums[threadIdx_x] += (value0 - mean) * (value0 - mean);
        if (element_x < cols - 1) {
          partial_sums[threadIdx_x] += (value1 - mean) * (value1 - mean);
        }
        if (element_x < cols - 2) {
          partial_sums[threadIdx_x] += (value2 - mean) * (value2 - mean);
        }
      }
    }
  }
  __syncthreads();

#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 256];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 128];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 64];
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 32];
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 16];
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 8];
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 4];
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 2];
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 1];
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    atomicAdd(stddev_values, partial_sums[0]);

    uint local_count = atomicInc(&block_count, blocks);
    bool is_last_block_done = (local_count == (blocks - 1));
    if (is_last_block_done) {
      float square = stddev_values[0] * weight;
      stddev_values[0] = sqrtf(square);

      block_count = 0;
    }
  }
}

template <typename Tsrc, typename Tsrcn, typename Tsumn>
__global__
void unmaskedDevCnKernel(const Tsrc* src, int rows, int cols, int channels,
                         int src_stride, uint blocks, float weight,
                         float* mean_values, float* stddev_values) {
  __shared__ Tsumn partial_sums[BLOCK_SIZE];

  int threadIdx_x = threadIdx.x;
  int element_x = (blockIdx.x << BLOCK_SHIFT) + threadIdx_x;
  int element_y = blockIdx.y;
  setZeroVector(partial_sums[threadIdx_x]);

  Tsrcn* input;
  Tsrcn value0;
  Tsumn mean, value1;
  readVector(mean, mean_values);

  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      input = (Tsrcn*)((uchar*)src + element_y * src_stride);
      value0 = input[element_x];
      assignVector(value1, value0);
      value1 -= mean;
      mulAdd(partial_sums[threadIdx_x], value1, value1);
    }
  }
  __syncthreads();

#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 256];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 128];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 64];
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 32];
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 16];
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 8];
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 4];
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 2];
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 1];
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    atomicAddVector(stddev_values, partial_sums[0]);
    __threadfence();

    uint local_count = atomicInc(&block_count, blocks);
    bool is_last_block_done = (local_count == (blocks - 1));
    if (is_last_block_done) {
      float square = stddev_values[0] * weight;
      stddev_values[0] = sqrtf(square);
      if (channels > 2) {
        square = stddev_values[1] * weight;
        stddev_values[1] = sqrtf(square);
        square = stddev_values[2] * weight;
        stddev_values[2] = sqrtf(square);
      }
      if (channels > 3) {
        square = stddev_values[3] * weight;
        stddev_values[3] = sqrtf(square);
      }

      block_count = 0;
    }
  }
}

template <typename Tsrc>
__global__
void maskedDevC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                       const uchar* mask, int mask_stride, uint blocks,
                       float* mean_values, float* stddev_values) {
  __shared__ float partial_sums[BLOCK_SIZE];
  __shared__ uint partial_counts[BLOCK_SIZE];

  int threadIdx_x = threadIdx.x;
  int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  int element_y = blockIdx.y;
  partial_sums[threadIdx_x] = 0;
  partial_counts[threadIdx_x] = 0;

  Tsrc* input;
  uchar* mask_row;
  Tsrc value0, value1, value2, value3;
  uchar mvalue0, mvalue1, mvalue2, mvalue3;
  float mean = mean_values[0];

  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      input  = (Tsrc*)((uchar*)src + element_y * src_stride);
      mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      mvalue0 = mask_row[element_x];
      mvalue1 = mask_row[element_x + 1];
      mvalue2 = mask_row[element_x + 2];
      mvalue3 = mask_row[element_x + 3];
      if (mvalue0 > 0) {
        partial_sums[threadIdx_x] += (value0 - mean) * (value0 - mean);
        partial_counts[threadIdx_x] += 1;
      }
      if (mvalue1 > 0 && element_x < cols - 1) {
        partial_sums[threadIdx_x] += (value1 - mean) * (value1 - mean);
        partial_counts[threadIdx_x] += 1;
      }
      if (mvalue2 > 0 && element_x < cols - 2) {
        partial_sums[threadIdx_x] += (value2 - mean) * (value2 - mean);
        partial_counts[threadIdx_x] += 1;
      }
      if (mvalue3 > 0 && element_x < cols - 3) {
        partial_sums[threadIdx_x] += (value3 - mean) * (value3 - mean);
        partial_counts[threadIdx_x] += 1;
      }
    }
  }
  __syncthreads();

#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 256];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 256];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 128];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 128];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 64];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 64];
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 32];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 32];
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 16];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 16];
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 8];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 8];
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 4];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 4];
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 2];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 2];
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 1];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 1];
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    atomicAdd(stddev_values, partial_sums[0]);
    atomicAdd(&mask_count, partial_counts[0]);
    __threadfence();

    uint local_count = atomicInc(&block_count, blocks);
    bool is_last_block_done = (local_count == (blocks - 1));
    if (is_last_block_done) {
      float weight = 1.f / mask_count;
      float square = stddev_values[0] * weight;
      stddev_values[0] = sqrtf(square);

      block_count = 0;
      mask_count = 0;
    }
  }
}

template <typename Tsrc, typename Tsrcn, typename Tsumn>
__global__
void maskedDevCnKernel(const Tsrc* src, int rows, int cols, int channels,
                       int src_stride, const uchar* mask, int mask_stride,
                       uint blocks, float* mean_values, float* stddev_values) {
  __shared__ Tsumn partial_sums[BLOCK_SIZE];
  __shared__ uint partial_counts[BLOCK_SIZE];

  int threadIdx_x = threadIdx.x;
  int element_x = (blockIdx.x << BLOCK_SHIFT) + threadIdx_x;
  int element_y = blockIdx.y;
  setZeroVector(partial_sums[threadIdx_x]);
  partial_counts[threadIdx_x] = 0;

  Tsrcn* input;
  uchar* mask_row;
  Tsrcn value0;
  uchar mvalue;
  Tsumn mean, value1;
  readVector(mean, mean_values);

  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      input  = (Tsrcn*)((uchar*)src + element_y * src_stride);
      mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
      value0  = input[element_x];
      mvalue = mask_row[element_x];

      if (mvalue > 0) {
        assignVector(value1, value0);
        value1 -= mean;
        mulAdd(partial_sums[threadIdx_x], value1, value1);
        partial_counts[threadIdx_x] += 1;
      }
    }
  }
  __syncthreads();

#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 256];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 256];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 128];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 128];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 64];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 64];
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 32];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 32];
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 16];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 16];
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 8];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 8];
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 4];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 4];
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 2];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 2];
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    partial_sums[threadIdx_x] += partial_sums[threadIdx_x + 1];
    partial_counts[threadIdx_x] += partial_counts[threadIdx_x + 1];
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    atomicAddVector(stddev_values, partial_sums[0]);
    atomicAdd(&mask_count, partial_counts[0]);
    __threadfence();

    uint local_count = atomicInc(&block_count, blocks);
    bool is_last_block_done = (local_count == (blocks - 1));
    if (is_last_block_done) {
      float weight = 1.f / mask_count;
      float square = stddev_values[0] * weight;
      stddev_values[0] = sqrtf(square);
      if (channels > 2) {
        square = stddev_values[1] * weight;
        stddev_values[1] = sqrtf(square);
        square = stddev_values[2] * weight;
        stddev_values[2] = sqrtf(square);
      }
      if (channels > 3) {
        square = stddev_values[3] * weight;
        stddev_values[3] = sqrtf(square);
      }

      block_count = 0;
      mask_count = 0;
    }
  }
}

RetCode meanStdDev(const uchar* src, int rows, int cols, int channels,
                   int src_stride, const uchar* mask, int mask_stride,
                   float* mean_values, float* stddev_values,
                   cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(mean_values != nullptr);
  PPL_ASSERT(stddev_values != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  int columns, grid_y;
  if (channels == 1) {
    columns = divideUp(cols, 4, 2);
  }
  else {
    columns = cols;
  }
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(columns, BLOCK_SIZE, BLOCK_SHIFT);
  grid_y  = MAX_BLOCKS / grid.x;
  grid.y  = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  if (mask == nullptr) {
    float weight = 1.f / (rows * cols);
    if (channels == 1) {
      unmaskedMeanC1Kernel<uchar, uint><<<grid, block, 0, stream>>>(src, rows,
          cols, src_stride, blocks, weight, mean_values);
      unmaskedDevC1Kernel<uchar><<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, blocks, weight, mean_values, stddev_values);
    }
    else if (channels == 3) {
      unmaskedMeanCnKernel<uchar, uchar3, uint3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
      unmaskedDevCnKernel<uchar, uchar3, float3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values,
          stddev_values);
    }
    else {  //  channels == 4
      unmaskedMeanCnKernel<uchar, uchar4, uint4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
      unmaskedDevCnKernel<uchar, uchar4, float4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values,
          stddev_values);
    }
  }
  else {
    if (channels == 1) {
      maskedMeanC1Kernel<uchar, uint><<<grid, block, 0, stream>>>(src, rows,
          cols, src_stride, mask, mask_stride, blocks, mean_values);
      maskedDevC1Kernel<uchar><<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, mask, mask_stride, blocks, mean_values, stddev_values);
    }
    else if (channels == 3) {
      maskedMeanCnKernel<uchar, uchar3, uint3><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
      maskedDevCnKernel<uchar, uchar3, float3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values, stddev_values);
    }
    else {  //  channels == 4
      maskedMeanCnKernel<uchar, uchar4, uint4><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
      maskedDevCnKernel<uchar, uchar4, float4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values, stddev_values);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode meanStdDev(const float* src, int rows, int cols, int channels,
                   int src_stride, const uchar* mask, int mask_stride,
                   float* mean_values, float* stddev_values,
                   cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(mean_values != nullptr);
  PPL_ASSERT(stddev_values != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  int columns, grid_y;
  if (channels == 1) {
    columns = divideUp(cols, 4, 2);
  }
  else {
    columns = cols;
  }
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(columns, BLOCK_SIZE, BLOCK_SHIFT);
  grid_y  = MAX_BLOCKS / grid.x;
  grid.y  = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  if (mask == nullptr) {
    float weight = 1.f / (rows * cols);
    if (channels == 1) {
      unmaskedMeanC1Kernel<float, float><<<grid, block, 0, stream>>>(src, rows,
          cols, src_stride, blocks, weight, mean_values);
      unmaskedDevC1Kernel<float><<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, blocks, weight, mean_values, stddev_values);
    }
    else if (channels == 3) {
      unmaskedMeanCnKernel<float, float3, float3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
      unmaskedDevCnKernel<float, float3, float3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values,
          stddev_values);
    }
    else {  //  channels == 4
      unmaskedMeanCnKernel<float, float4, float4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
      unmaskedDevCnKernel<float, float4, float4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values,
          stddev_values);
    }
  }
  else {
    if (channels == 1) {
      maskedMeanC1Kernel<float, float><<<grid, block, 0, stream>>>(src, rows,
          cols, src_stride, mask, mask_stride, blocks, mean_values);
      maskedDevC1Kernel<float><<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, mask, mask_stride, blocks, mean_values, stddev_values);
    }
    else if (channels == 3) {
      maskedMeanCnKernel<float, float3, float3><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
      maskedDevCnKernel<float, float3, float3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values, stddev_values);
    }
    else {  //  channels == 4
      maskedMeanCnKernel<float, float4, float4><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
      maskedDevCnKernel<float, float4, float4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values, stddev_values);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode MeanStdDev<uchar, 1>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const uchar* inData,
                             float* outMean,
                             float* outStdDev,
                             int maskWidthStride,
                             const uchar* mask) {
  RetCode code = meanStdDev(inData, height, width, 1, inWidthStride, mask,
                            maskWidthStride, outMean, outStdDev, stream);

  return code;
}

template <>
RetCode MeanStdDev<uchar, 3>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const uchar* inData,
                             float* outMean,
                             float* outStdDev,
                             int maskWidthStride,
                             const uchar* mask) {
  RetCode code = meanStdDev(inData, height, width, 3, inWidthStride, mask,
                            maskWidthStride, outMean, outStdDev, stream);

  return code;
}

template <>
RetCode MeanStdDev<uchar, 4>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const uchar* inData,
                             float* outMean,
                             float* outStdDev,
                             int maskWidthStride,
                             const uchar* mask) {
  RetCode code = meanStdDev(inData, height, width, 4, inWidthStride, mask,
                            maskWidthStride, outMean, outStdDev, stream);

  return code;
}

template <>
RetCode MeanStdDev<float, 1>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const float* inData,
                             float* outMean,
                             float* outStdDev,
                             int maskWidthStride,
                             const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = meanStdDev(inData, height, width, 1, inWidthStride, mask,
                            maskWidthStride, outMean, outStdDev, stream);

  return code;
}

template <>
RetCode MeanStdDev<float, 3>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const float* inData,
                             float* outMean,
                             float* outStdDev,
                             int maskWidthStride,
                             const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = meanStdDev(inData, height, width, 3, inWidthStride, mask,
                            maskWidthStride, outMean, outStdDev, stream);

  return code;
}

template <>
RetCode MeanStdDev<float, 4>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const float* inData,
                             float* outMean,
                             float* outStdDev,
                             int maskWidthStride,
                             const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = meanStdDev(inData, height, width, 4, inWidthStride, mask,
                            maskWidthStride, outMean, outStdDev, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
