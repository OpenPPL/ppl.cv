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

#include "ppl/cv/cuda/mean.h"
#include "mean.hpp"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

RetCode mean(const uchar* src, int rows, int cols, int channels, int src_stride,
             const uchar* mask, int mask_stride, float* mean_values,
             cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(mean_values != nullptr);
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
    }
    else if (channels == 3) {
      unmaskedMeanCnKernel<uchar, uchar3, uint3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
    }
    else {  //  channels == 4
      unmaskedMeanCnKernel<uchar, uchar4, uint4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
    }
  }
  else {
    if (channels == 1) {
      maskedMeanC1Kernel<uchar, uint><<<grid, block, 0, stream>>>(src, rows,
          cols, src_stride, mask, mask_stride, blocks, mean_values);
    }
    else if (channels == 3) {
      maskedMeanCnKernel<uchar, uchar3, uint3><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
    }
    else {  //  channels == 4
      maskedMeanCnKernel<uchar, uchar4, uint4><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode mean(const float* src, int rows, int cols, int channels, int src_stride,
             const uchar* mask, int mask_stride, float* mean_values,
             cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(mean_values != nullptr);
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
    }
    else if (channels == 3) {
      unmaskedMeanCnKernel<float, float3, float3><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
    }
    else {  //  channels == 4
      unmaskedMeanCnKernel<float, float4, float4><<<grid, block, 0, stream>>>(
          src, rows, cols, channels, src_stride, blocks, weight, mean_values);
    }
  }
  else {
    if (channels == 1) {
      maskedMeanC1Kernel<float, float><<<grid, block, 0, stream>>>(src, rows,
          cols, src_stride, mask, mask_stride, blocks, mean_values);
    }
    else if (channels == 3) {
      maskedMeanCnKernel<float, float3, float3><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
    }
    else {  //  channels == 4
      maskedMeanCnKernel<float, float4, float4><<<grid, block, 0, stream>>>(src,
          rows, cols, channels, src_stride, mask, mask_stride, blocks,
          mean_values);
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
RetCode Mean<uchar, 1>(cudaStream_t stream,
                       int height,
                       int width,
                       int inWidthStride,
                       const uchar* inData,
                       float* outMean,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = mean(inData, height, width, 1, inWidthStride, mask,
                      maskWidthStride, outMean, stream);

  return code;
}

template <>
RetCode Mean<uchar, 3>(cudaStream_t stream,
                       int height,
                       int width,
                       int inWidthStride,
                       const uchar* inData,
                       float* outMean,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = mean(inData, height, width, 3, inWidthStride, mask,
                      maskWidthStride, outMean, stream);

  return code;
}

template <>
RetCode Mean<uchar, 4>(cudaStream_t stream,
                       int height,
                       int width,
                       int inWidthStride,
                       const uchar* inData,
                       float* outMean,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = mean(inData, height, width, 4, inWidthStride, mask,
                      maskWidthStride, outMean, stream);

  return code;
}

template <>
RetCode Mean<float, 1>(cudaStream_t stream,
                       int height,
                       int width,
                       int inWidthStride,
                       const float* inData,
                       float* outMean,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = mean(inData, height, width, 1, inWidthStride, mask,
                      maskWidthStride, outMean, stream);

  return code;
}

template <>
RetCode Mean<float, 3>(cudaStream_t stream,
                       int height,
                       int width,
                       int inWidthStride,
                       const float* inData,
                       float* outMean,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = mean(inData, height, width, 3, inWidthStride, mask,
                      maskWidthStride, outMean, stream);

  return code;
}

template <>
RetCode Mean<float, 4>(cudaStream_t stream,
                       int height,
                       int width,
                       int inWidthStride,
                       const float* inData,
                       float* outMean,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = mean(inData, height, width, 4, inWidthStride, mask,
                      maskWidthStride, outMean, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
