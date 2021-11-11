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

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define BLOCK_SIZE 128
#define BLOCK_SHIFT 7

__global__
void verticalKernel1(const uchar* src, int* dst, int src_rows, int src_cols,
                     int src_stride, int dst_stride) {
  int src_x0 = ((blockIdx.x << BLOCK_SHIFT) + threadIdx.x) << 1;
  if (src_x0 >= src_cols) {
    return;
  }

  int src_x1 = src_x0 + 1;
  int dst_x0 = src_x0;
  int dst_x1 = src_x0 + 1;
  int value0 = 0, value1 = 0;
  uchar src_value0, src_value1;

  dst_stride = dst_stride >> 2;
  uchar* src_row = (uchar*)src;
  int* dst_row = dst;

  if (blockIdx.x < gridDim.x - 1) {
    for (int row = 0; row < src_rows; row++) {
      src_value0 = src_row[src_x0];
      src_value1 = src_row[src_x1];

      value0 += src_value0;
      value1 += src_value1;

      dst_row[dst_x0] = value0;
      dst_row[dst_x1] = value1;

      src_row += src_stride;
      dst_row += dst_stride;
    }
  }
  else {
    for (int row = 0; row < src_rows; row++) {
      src_value0 = src_row[src_x0];
      if (src_x0 != src_cols - 1) {
        src_value1 = src_row[src_x1];
      }

      value0 += src_value0;
      if (src_x0 != src_cols - 1) {
        value1 += src_value1;
      }

      dst_row[dst_x0] = value0;
      if (src_x0 != src_cols - 1) {
        dst_row[dst_x1] = value1;
      }

      src_row += src_stride;
      dst_row += dst_stride;
    }
  }
}

__global__
void verticalKernel2(const uchar* src, int* dst, int src_rows, int src_cols,
                     int src_stride, int dst_stride) {
  int src_x0 = ((blockIdx.x << BLOCK_SHIFT) + threadIdx.x) << 1;
  if (src_x0 >= src_cols) {
    return;
  }

  int src_x1 = src_x0 + 1;
  int dst_x0 = src_x0 + 1;
  int dst_x1 = src_x0 + 2;
  int value0 = 0, value1 = 0;
  uchar src_value0, src_value1;

  if (src_x0 == 0) {
    dst[0] = 0;
  }

  dst[dst_x0] = 0;
  if (dst_x1 < src_cols + 1) {
    dst[dst_x1] = 0;
  }

  dst_stride = dst_stride >> 2;
  uchar* src_row = (uchar*)src;
  int* dst_row = dst + dst_stride;

  if (blockIdx.x < gridDim.x - 1) {
    for (int row = 0; row < src_rows; row++) {
      src_value0 = src_row[src_x0];
      src_value1 = src_row[src_x1];

      value0 += src_value0;
      value1 += src_value1;

      dst_row[dst_x0] = value0;
      dst_row[dst_x1] = value1;

      src_row += src_stride;
      dst_row += dst_stride;
    }
  }
  else {
    for (int row = 0; row < src_rows; row++) {
      src_value0 = src_row[src_x0];
      if (src_x0 != src_cols - 1) {
        src_value1 = src_row[src_x1];
      }

      value0 += src_value0;
      if (src_x0 != src_cols - 1) {
        value1 += src_value1;
      }

      dst_row[dst_x0] = value0;
      if (src_x0 != src_cols - 1) {
        dst_row[dst_x1] = value1;
      }

      src_row += src_stride;
      dst_row += dst_stride;
    }
  }
}

__global__
void verticalKernel1(const float* src, float* dst, int src_rows, int src_cols,
                     int src_stride, int dst_stride) {
  int src_x0 = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  if (src_x0 >= src_cols) {
    return;
  }

  int dst_x0 = src_x0;
  double value0 = 0.0;
  float src_value0;

  float* src_row = (float*)src;
  float* dst_row = dst;
  src_stride = src_stride >> 2;
  dst_stride = dst_stride >> 2;

  for (int row = 0; row < src_rows; row++) {
    src_value0 = src_row[src_x0];
    value0 += src_value0;
    dst_row[dst_x0] = value0;

    src_row += src_stride;
    dst_row += dst_stride;
  }
}

__global__
void verticalKernel2(const float* src, float* dst, int src_rows, int src_cols,
                     int src_stride, int dst_stride) {
  int src_x0 = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  if (src_x0 >= src_cols) {
    return;
  }

  int dst_x0 = src_x0 + 1;
  double value0 = 0.0;
  float src_value0;

  if (src_x0 == 0) {
    dst[0] = 0.f;
  }
  dst[dst_x0] = 0.f;

  src_stride = src_stride >> 2;
  dst_stride = dst_stride >> 2;
  float* src_row = (float*)src;
  float* dst_row = dst + dst_stride;

  for (int row = 0; row < src_rows; row++) {
    src_value0 = src_row[src_x0];
    value0 += src_value0;
    dst_row[dst_x0] = value0;

    src_row += src_stride;
    dst_row += dst_stride;
  }
}

__global__
void horizontalKernel1(int* dst, int src_rows, int src_cols, int dst_stride) {
  int row = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  if (row >= src_rows) {
    return;
  }

  int value0, value1, value;
  int* dst_row = (int*)((uchar*)dst + row * dst_stride);

  value0 = dst_row[0];
  value1 = dst_row[1];

  value1 += value0;
  value   = value1;

  dst_row[1] = value1;

  for (int col = 2; col < src_cols; col += 2) {
    value0 = dst_row[col];
    if (col + 1 < src_cols) {
      value1 = dst_row[col + 1];
    }

    value0 += value;
    value1 += value0;
    value   = value1;

    dst_row[col] = value0;
    if (col + 1 < src_cols) {
      dst_row[col + 1] = value1;
    }
  }
}

__global__
void horizontalKernel2(int* dst, int src_rows, int src_cols, int dst_stride) {
  int row = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  if (row >= src_rows) {
    return;
  }

  int value0, value1, value;
  int* dst_row = (int*)((uchar*)dst + (row + 1) * dst_stride);

  value0 = 0;
  value1 = dst_row[1];

  value = value1;

  dst_row[0] = value0;

  for (int col = 2; col <= src_cols; col += 2) {
    value0 = dst_row[col];
    if (col + 1 <= src_cols) {
      value1 = dst_row[col + 1];
    }

    value0 += value;
    value1 += value0;
    value   = value1;

    dst_row[col] = value0;
    if (col + 1 <= src_cols) {
      dst_row[col + 1] = value1;
    }
  }
}

__global__
void horizontalKernel1(float* dst, int src_rows, int src_cols, int dst_stride) {
  int row = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  if (row >= src_rows) {
    return;
  }

  double value0, value;
  float* dst_row = (float*)((uchar*)dst + row * dst_stride);

  value0 = dst_row[0];
  value  = value0;

  for (int col = 1; col < src_cols; col++) {
    value0 = dst_row[col];
    value += value0;

    dst_row[col] = value;
  }
}

__global__
void horizontalKernel2(float* dst, int src_rows, int src_cols, int dst_stride) {
  int row = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  if (row >= src_rows) {
    return;
  }

  double value0, value;
  float* dst_row = (float*)((uchar*)dst + (row + 1) * dst_stride);

  value = dst_row[1];

  dst_row[0] = 0.0;

  for (int col = 2; col <= src_cols; col++) {
    value0 = dst_row[col];
    value += value0;

    dst_row[col] = value;
  }
}

RetCode integral(const uchar* src, int* dst, int src_rows, int src_cols,
                 int src_stride, int dst_rows, int dst_cols, int dst_stride,
                 cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 1 && src_cols > 1);
  PPL_ASSERT(dst_rows > 1 && dst_cols > 1);
  PPL_ASSERT(dst_rows == src_rows || dst_rows == src_rows + 1);
  PPL_ASSERT(dst_cols == src_cols || dst_cols == src_cols + 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(uchar));

  int threads, block, grid;
  threads = divideUp(src_cols, 2, 1);
  block   = BLOCK_SIZE;
  grid    = divideUp(threads, BLOCK_SIZE, BLOCK_SHIFT);
  if (dst_cols == src_cols) {
    verticalKernel1<<<grid, block, 0, stream>>>(src, dst, src_rows, src_cols,
                                                src_stride, dst_stride);
  }
  else {  // dst_cols == src_cols + 1
    verticalKernel2<<<grid, block, 0, stream>>>(src, dst, src_rows, src_cols,
                                                src_stride, dst_stride);
  }

  threads = src_rows;
  grid = divideUp(threads, BLOCK_SIZE, BLOCK_SHIFT);
  if (dst_rows == src_rows) {
    horizontalKernel1<<<grid, block, 0, stream>>>(dst, src_rows, src_cols,
                                                  dst_stride);
  }
  else {  // dst_rows == src_rows + 1
    horizontalKernel2<<<grid, block, 0, stream>>>(dst, src_rows, src_cols,
                                                  dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode integral(const float* src, float* dst, int src_rows, int src_cols,
                 int src_stride, int dst_rows, int dst_cols, int dst_stride,
                 cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 1 && src_cols > 1);
  PPL_ASSERT(dst_rows > 1 && dst_cols > 1);
  PPL_ASSERT(dst_rows == src_rows || dst_rows == src_rows + 1);
  PPL_ASSERT(dst_cols == src_cols || dst_cols == src_cols + 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * (int)sizeof(float));

  int threads, block, grid;
  threads = src_cols;
  block = BLOCK_SIZE;
  grid  = divideUp(threads, BLOCK_SIZE, BLOCK_SHIFT);
  if (dst_cols == src_cols) {
    verticalKernel1<<<grid, block, 0, stream>>>(src, dst, src_rows, src_cols,
                                                src_stride, dst_stride);
  }
  else {  // dst_cols == src_cols + 1
    verticalKernel2<<<grid, block, 0, stream>>>(src, dst, src_rows, src_cols,
                                                src_stride, dst_stride);
  }

  threads = src_rows;
  grid = divideUp(threads, BLOCK_SIZE, BLOCK_SHIFT);
  if (dst_rows == src_rows) {
    horizontalKernel1<<<grid, block, 0, stream>>>(dst, src_rows, src_cols,
                                                  dst_stride);
  }
  else {  // dst_rows == src_rows + 1
    horizontalKernel2<<<grid, block, 0, stream>>>(dst, src_rows, src_cols,
                                                  dst_stride);
  }

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
  RetCode code = integral(inData, outData, inHeight, inWidth, inWidthStride,
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
  RetCode code = integral(inData, outData, inHeight, inWidth, inWidthStride,
                          outHeight, outWidth, outWidthStride, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
