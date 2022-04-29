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

#include "ppl/cv/cuda/normalize.h"

#include "utility/utility.hpp"
#include "norm.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename Tsrc, typename Tdst>
__global__
void convertKernel(const Tsrc* src, int rows, int cols, int channels,
                   int src_stride, const uchar* mask, int mask_stride,
                   float* dst, int dst_stride, Tdst* norms_values, float alpha,
                   float beta, NormTypes norm_type) {
  int threadIdx_x = threadIdx.x;
  int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  int element_y = blockIdx.y;
  if (element_x >= cols) {
    return;
  }

  float scale, shift;
  if (norm_type == NORM_L1 || norm_type == NORM_INF) {
    scale = norms_values[0];
    scale = scale > FLT_EPSILON ? alpha / scale : 0.f;
  }
  else if (norm_type == NORM_L2) {
    scale = sqrtf(norms_values[0]);
    scale = scale > FLT_EPSILON ? alpha / scale : 0.f;
  }
  else {  // norm_type == NORM_MINMAX
    float src_max = norms_values[0];
    float src_min = norms_values[1];
    scale = src_max - src_min;
    scale = scale > FLT_EPSILON ? (beta - alpha) / scale : 0.f;
    shift = alpha - src_min * scale;
  }

  int offset;
  Tsrc* input;
  float* output;
  uchar* mask_row;
  float value0, value1, value2, value3;
  uchar mvalue0, mvalue1, mvalue2, mvalue3;

  for (; element_y < rows; element_y += gridDim.y) {
    offset = element_y * src_stride;
    input = (Tsrc*)((uchar*)src + offset);
    value0 = input[element_x];
    value1 = input[element_x + 1];
    value2 = input[element_x + 2];
    value3 = input[element_x + 3];

    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;

    if (norm_type == NORM_MINMAX) {
      value0 += shift;
      value1 += shift;
      value2 += shift;
      value3 += shift;
    }

    offset = element_y * dst_stride;
    output = (float*)((uchar*)dst + offset);
    if (element_x < cols - 3) {
      if (mask != nullptr) {
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_row[element_x / channels];
        mvalue1 = mask_row[(element_x + 1) / channels];
        mvalue2 = mask_row[(element_x + 2) / channels];
        mvalue3 = mask_row[(element_x + 3) / channels];
        output[element_x] = mvalue0 > 0 ? value0 : 0;
        output[element_x + 1] = mvalue1 > 0 ? value1 : 0;
        output[element_x + 2] = mvalue2 > 0 ? value2 : 0;
        output[element_x + 3] = mvalue3 > 0 ? value3 : 0;
      }
      else {
        output[element_x] = value0;
        output[element_x + 1] = value1;
        output[element_x + 2] = value2;
        output[element_x + 3] = value3;
      }
    }
    else {
      if (mask != nullptr) {
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_row[element_x / channels];
        mvalue1 = mask_row[(element_x + 1) / channels];
        mvalue2 = mask_row[(element_x + 2) / channels];
        output[element_x] = mvalue0 > 0 ? value0 : 0;
        if (element_x < cols - 1) {
          output[element_x + 1] = mvalue1 > 0 ? value1 : 0;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = mvalue2 > 0 ? value2 : 0;
        }
      }
      else {
        output[element_x] = value0;
        if (element_x < cols - 1) {
          output[element_x + 1] = value1;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = value2;
        }
      }
    }
  }
}

inline void swap(float& alpha, float& beta) {
  float temp = alpha;
  alpha = beta;
  beta  = temp;
}

RetCode normalize(const uchar* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride, float alpha,
                  float beta, NormTypes norm_type, const uchar* mask,
                  int mask_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(norm_type == NORM_INF || norm_type == NORM_L1 ||
             norm_type == NORM_L2 || norm_type == NORM_MINMAX);
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
  long long* norms_values;
  cudaError_t code;

  size_t norms_size;;
  if (norm_type == NORM_MINMAX) {
    norms_size = blocks * 2 * sizeof(long long);
  }
  else {
    norms_size = blocks * sizeof(long long);
  }
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMalloc(norms_size, buffer_block);
    norms_values = (long long*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&norms_values, norms_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  if (norm_type == NORM_INF) {
    normLinfKernel<uchar, long long><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks, norms_values);
  }
  else if (norm_type == NORM_L1) {
    normL1Kernel<uchar, uint, long long><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks, norms_values);
  }
  else if (norm_type == NORM_L2) {
    normL2Kernel<uchar, long long, long long><<<grid, block, 0, stream>>>(src,
        rows, cols, channels, src_stride, mask, mask_stride, blocks,
        norms_values);
  }
  else {  // norm_type == NORM_MINMAX
    if (alpha > beta) {
      swap(alpha, beta);
    }
    MinMaxKernel<uchar, long long><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, blocks, norms_values);
  }
  convertKernel<uchar, long long><<<grid, block, 0, stream>>>(src, rows, cols,
      channels, src_stride, mask, mask_stride, dst, dst_stride, norms_values,
      alpha, beta, norm_type);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(norms_values);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  else {
    return RC_SUCCESS;
  }
}

RetCode normalize(const float* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride, float alpha,
                  float beta, NormTypes norm_type, const uchar* mask,
                  int mask_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(norm_type == NORM_INF || norm_type == NORM_L1 ||
             norm_type == NORM_L2 || norm_type == NORM_MINMAX);
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
  double* norms_values;
  cudaError_t code;

  size_t norms_size;;
  if (norm_type == NORM_MINMAX) {
    norms_size = blocks * 2 * sizeof(double);
  }
  else {
    norms_size = blocks * sizeof(double);
  }
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMalloc(norms_size, buffer_block);
    norms_values = (double*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&norms_values, norms_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  if (norm_type == NORM_INF) {
    normLinfKernel<float, double><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, blocks, norms_values);
  }
  else if (norm_type == NORM_L1) {
    normL1Kernel<float, float, double><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks, norms_values);
  }
  else if (norm_type == NORM_L2) {
    normL2Kernel<float, float, double><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, blocks, norms_values);
  }
  else {  // norm_type == NORM_MINMAX
    if (alpha > beta) {
      swap(alpha, beta);
    }
    MinMaxKernel<float, double><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, blocks, norms_values);
  }
  convertKernel<float, double><<<grid, block, 0, stream>>>(src, rows, cols,
      channels, src_stride, mask, mask_stride, dst, dst_stride, norms_values,
      alpha, beta, norm_type);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(norms_values);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  else {
    return RC_SUCCESS;
  }
}

template <>
RetCode Normalize<uchar, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int outWidthStride,
                            float* outData,
                            float alpha,
                            float beta,
                            NormTypes normType,
                            int maskWidthStride,
                            const uchar* mask) {
  outWidthStride *= sizeof(float);
  RetCode code = normalize(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, alpha, beta, normType, mask,
                           maskWidthStride, stream);

  return code;
}

template <>
RetCode Normalize<uchar, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int outWidthStride,
                            float* outData,
                            float alpha,
                            float beta,
                            NormTypes normType,
                            int maskWidthStride,
                            const uchar* mask) {
  outWidthStride *= sizeof(float);
  RetCode code = normalize(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, alpha, beta, normType, mask,
                           maskWidthStride, stream);

  return code;
}

template <>
RetCode Normalize<uchar, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int outWidthStride,
                            float* outData,
                            float alpha,
                            float beta,
                            NormTypes normType,
                            int maskWidthStride,
                            const uchar* mask) {
  outWidthStride *= sizeof(float);
  RetCode code = normalize(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, alpha, beta, normType, mask,
                           maskWidthStride, stream);

  return code;
}

template <>
RetCode Normalize<float, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int outWidthStride,
                            float* outData,
                            float alpha,
                            float beta,
                            NormTypes normType,
                            int maskWidthStride,
                            const uchar* mask) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = normalize(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, alpha, beta, normType, mask,
                           maskWidthStride, stream);

  return code;
}

template <>
RetCode Normalize<float, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int outWidthStride,
                            float* outData,
                            float alpha,
                            float beta,
                            NormTypes normType,
                            int maskWidthStride,
                            const uchar* mask) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = normalize(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, alpha, beta, normType, mask,
                           maskWidthStride, stream);

  return code;
}

template <>
RetCode Normalize<float, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int outWidthStride,
                            float* outData,
                            float alpha,
                            float beta,
                            NormTypes normType,
                            int maskWidthStride,
                            const uchar* mask) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = normalize(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, alpha, beta, normType, mask,
                           maskWidthStride, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
