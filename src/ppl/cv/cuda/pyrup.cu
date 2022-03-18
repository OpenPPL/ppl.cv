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

#include "ppl/cv/cuda/pyrup.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define RADIUS 2
#define BlockDimX 32
#define BlockDimY 8
#define BlockShiftX 5
#define BlockShiftY 3

template <typename Tn, typename T>
__DEVICE__
Tn setValue(T value);

template <>
__DEVICE__
uchar3 setValue(uchar value) {
  uchar3 result;
  result.x = value;
  result.y = value;
  result.z = value;

  return result;
}

template <>
__DEVICE__
uchar4 setValue(uchar value) {
  uchar4 result;
  result.x = value;
  result.y = value;
  result.z = value;
  result.w = value;

  return result;
}

template <>
__DEVICE__
float3 setValue(float value) {
  float3 result;
  result.x = value;
  result.y = value;
  result.z = value;

  return result;
}

template <>
__DEVICE__
float4 setValue(float value) {
  float4 result;
  result.x = value;
  result.y = value;
  result.z = value;
  result.w = value;

  return result;
}

template <typename T, typename BorderInterpolation>
__global__
void pyrupC1Kernel(const T* src, int src_stride, T* dst, int dst_rows,
                   int dst_cols, int dst_stride,
                   BorderInterpolation interpolation) {
  __shared__ T raw_data[BlockDimY + (RADIUS << 1)][BlockDimX + (RADIUS << 1)];
  __shared__ float col_data[BlockDimY + (RADIUS << 1)][BlockDimX];

  int element_x = (blockIdx.x << BlockShiftX) + threadIdx.x;
  int element_y = (blockIdx.y << BlockShiftY) + threadIdx.y;

  int y_index   = threadIdx.y;
  int row_index = element_y - RADIUS;
  while (row_index < (int)(((blockIdx.y + 1) << BlockShiftY) + RADIUS) &&
         row_index < dst_rows + RADIUS) {
    int index = interpolation(dst_rows, RADIUS, row_index);
    if ((index & 1) == 0) {
      T* input = (T*)((uchar*)src + (index >> 1) * src_stride);

      int x_index   = threadIdx.x, x_index1;
      int col_index = element_x - RADIUS;
      while (col_index < (int)(((blockIdx.x + 1) << BlockShiftX) + RADIUS) &&
             col_index < dst_cols + RADIUS) {
        x_index1 = interpolation(dst_cols, RADIUS, col_index);
        if ((x_index1 & 1) == 0) {
          raw_data[y_index][x_index] = input[x_index1 >> 1];
        }
        else {
          raw_data[y_index][x_index] = 0;
        }
        x_index   += BlockDimX;
        col_index += BlockDimX;
      }
    }
    else {
      int x_index   = threadIdx.x;
      int col_index = element_x - RADIUS;
      while (col_index < (int)(((blockIdx.x + 1) << BlockShiftX) + RADIUS) &&
             col_index < dst_cols + RADIUS) {
        raw_data[y_index][x_index] = 0;
        x_index   += BlockDimX;
        col_index += BlockDimX;
      }
    }

    y_index   += BlockDimY;
    row_index += BlockDimY;
  }
  __syncthreads();

  float sum;
  y_index   = threadIdx.y;
  row_index = element_y - RADIUS;
  while (row_index < (int)(((blockIdx.y + 1) << BlockShiftY) + RADIUS) &&
         row_index < dst_rows + RADIUS && element_x < dst_cols) {
    sum = 0.f;
    sum += raw_data[y_index][threadIdx.x] * 0.0625f;
    sum += raw_data[y_index][threadIdx.x + 1] * 0.25f;
    sum += raw_data[y_index][threadIdx.x + 2] * 0.375f;
    sum += raw_data[y_index][threadIdx.x + 3] * 0.25f;
    sum += raw_data[y_index][threadIdx.x + 4] * 0.0625f;

    col_data[y_index][threadIdx.x] = sum;
    y_index   += BlockDimY;
    row_index += BlockDimY;
  }
  __syncthreads();

  if (element_y < dst_rows && element_x < dst_cols) {
    sum = 0.f;
    sum += col_data[threadIdx.y][threadIdx.x] * 0.0625f;
    sum += col_data[threadIdx.y + 1][threadIdx.x] * 0.25f;
    sum += col_data[threadIdx.y + 2][threadIdx.x] * 0.375f;
    sum += col_data[threadIdx.y + 3][threadIdx.x] * 0.25f;
    sum += col_data[threadIdx.y + 4][threadIdx.x] * 0.0625f;
    sum *= 4;

    T* output = (T*)((uchar*)dst + element_y * dst_stride);
    if (sizeof(T) == 1) {
      output[element_x] = saturateCast(sum);
    }
    else {
      output[element_x] = sum;
    }
  }
}

template <typename T>
__global__
void pyrupC1Bat4Ref101Kernel(const T* src, int src_stride, T* dst, int dst_rows,
                             int dst_cols, int dst_stride) {
  __shared__ T raw_data[(BlockDimY >> 1) + RADIUS][(BlockDimX << 1) + 8];
  __shared__ float4 col_data[BlockDimY + (RADIUS << 1)][BlockDimX];

  int element_x = ((blockIdx.x << BlockShiftX) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << BlockShiftY) + threadIdx.y;

  int index0, index1;
  int src_x = (blockIdx.x << (BlockShiftX + 1)) + (threadIdx.x << 2) - 4;
  int src_y = (blockIdx.y << (BlockShiftY - 1)) + threadIdx.y - 1;
  if (threadIdx.y < (blockDim.y >> 1) + RADIUS &&
      threadIdx.x < (blockDim.x >> 1) + RADIUS &&
      src_y < (dst_rows >> 1) + 1 && src_x < (dst_cols >> 1) + 1) {
    int src_x1 = src_x + 1;
    int src_x2 = src_x + 2;
    int src_x3 = src_x + 3;

    src_x = src_x < 0 ? 0 - src_x : src_x;
    src_x1 = src_x1 < 0 ? 0 - src_x1 : src_x1;
    src_x2 = src_x2 < 0 ? 0 - src_x2 : src_x2;
    src_x3 = src_x3 < 0 ? 0 - src_x3 : src_x3;
    src_y = src_y < 0 ? 0 - src_y : src_y;
    int top = (dst_cols >> 1) - 1;
    src_x = src_x > top ? top : src_x;
    src_x1 = src_x1 > top ? top : src_x1;
    src_x2 = src_x2 > top ? top : src_x2;
    src_x3 = src_x3 > top ? top : src_x3;
    top = (dst_rows >> 1) - 1;
    src_y = src_y > top ? top : src_y;

    index0 = threadIdx.x << 2;
    T* input = (T*)((uchar*)src + src_y * src_stride);
    raw_data[threadIdx.y][index0] = input[src_x];
    raw_data[threadIdx.y][index0 + 1] = input[src_x1];
    raw_data[threadIdx.y][index0 + 2] = input[src_x2];
    raw_data[threadIdx.y][index0 + 3] = input[src_x3];
  }
  __syncthreads();

  int y_index   = threadIdx.y;
  int row_index = element_y - RADIUS;
  float4 sum;
  while (row_index < (int)(((blockIdx.y + 1) << BlockShiftY) + RADIUS) &&
         row_index < dst_rows + RADIUS && element_x < dst_cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    if ((y_index & 1) == 0) {
      index0 = y_index >> 1;
      index1 = threadIdx.x << 1;
      sum.x += raw_data[index0][index1 + 3] * 0.0625f;
      sum.z += raw_data[index0][index1 + 4] * 0.0625f;

      sum.y += raw_data[index0][index1 + 4] * 0.25f;
      sum.w += raw_data[index0][index1 + 5] * 0.25f;

      sum.x += raw_data[index0][index1 + 4] * 0.375f;
      sum.z += raw_data[index0][index1 + 5] * 0.375f;

      sum.y += raw_data[index0][index1 + 5] * 0.25f;
      sum.w += raw_data[index0][index1 + 6] * 0.25f;

      sum.x += raw_data[index0][index1 + 5] * 0.0625f;
      sum.z += raw_data[index0][index1 + 6] * 0.0625f;
    }

    col_data[y_index][threadIdx.x] = sum;
    y_index   += BlockDimY;
    row_index += BlockDimY;
  }
  __syncthreads();

  if (element_y < dst_rows && element_x < dst_cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    mulAdd(sum, col_data[threadIdx.y][threadIdx.x], 0.0625f);
    mulAdd(sum, col_data[threadIdx.y + 1][threadIdx.x], 0.25f);
    mulAdd(sum, col_data[threadIdx.y + 2][threadIdx.x], 0.375f);
    mulAdd(sum, col_data[threadIdx.y + 3][threadIdx.x], 0.25f);
    mulAdd(sum, col_data[threadIdx.y + 4][threadIdx.x], 0.0625f);

    sum.x *= 4;
    sum.y *= 4;
    sum.z *= 4;
    sum.w *= 4;

    T* output = (T*)((uchar*)dst + element_y * dst_stride);
    if (sizeof(T) == 1) {
      if (element_x < dst_cols - 3) {
        output[element_x]     = saturateCast(sum.x);
        output[element_x + 1] = saturateCast(sum.y);
        output[element_x + 2] = saturateCast(sum.z);
        output[element_x + 3] = saturateCast(sum.w);
      }
      else {
        output[element_x] = saturateCast(sum.x);
        if (element_x < dst_cols - 1) {
          output[element_x + 1] = saturateCast(sum.y);
        }
        if (element_x < dst_cols - 2) {
          output[element_x + 2] = saturateCast(sum.z);
        }
      }
    }
    else {
      if (element_x < dst_cols - 3) {
        output[element_x]     = sum.x;
        output[element_x + 1] = sum.y;
        output[element_x + 2] = sum.z;
        output[element_x + 3] = sum.w;
      }
      else {
        output[element_x] = sum.x;
        if (element_x < dst_cols - 1) {
          output[element_x + 1] = sum.y;
        }
        if (element_x < dst_cols - 2) {
          output[element_x + 2] = sum.z;
        }
      }
    }
  }
}

template <typename T, typename Tn, typename BorderInterpolation>
__global__
void pyrupCnKernel(const T* src, int src_stride, T* dst, int dst_rows,
                   int dst_cols, int dst_stride,
                   BorderInterpolation interpolation) {
  __shared__ Tn raw_data[BlockDimY + (RADIUS << 1)][BlockDimX + (RADIUS << 1)];
  __shared__ float4 col_data[BlockDimY + (RADIUS << 1)][BlockDimX];

  int element_x = (blockIdx.x << BlockShiftX) + threadIdx.x;
  int element_y = (blockIdx.y << BlockShiftY) + threadIdx.y;

  int y_index   = threadIdx.y;
  int row_index = element_y - RADIUS;
  while (row_index < (int)(((blockIdx.y + 1) << BlockShiftY) + RADIUS) &&
         row_index < dst_rows + RADIUS) {
    int index = interpolation(dst_rows, RADIUS, row_index);
    if ((index & 1) == 0) {
      Tn* input = (Tn*)((uchar*)src + (index >> 1) * src_stride);

      int x_index   = threadIdx.x, x_index1;
      int col_index = element_x - RADIUS;
      while (col_index < (int)(((blockIdx.x + 1) << BlockShiftX) + RADIUS) &&
             col_index < dst_cols + RADIUS) {
        x_index1 = interpolation(dst_cols, RADIUS, col_index);
        if ((x_index1 & 1) == 0) {
          raw_data[y_index][x_index] = input[x_index1 >> 1];
        }
        else {
          raw_data[y_index][x_index] = setValue<Tn, T>(0);
        }
        x_index   += BlockDimX;
        col_index += BlockDimX;
      }
    }
    else {
      int x_index   = threadIdx.x;
      int col_index = element_x - RADIUS;
      while (col_index < (int)(((blockIdx.x + 1) << BlockShiftX) + RADIUS) &&
             col_index < dst_cols + RADIUS) {
        raw_data[y_index][x_index] = setValue<Tn, T>(0);
        x_index   += BlockDimX;
        col_index += BlockDimX;
      }
    }

    y_index   += BlockDimY;
    row_index += BlockDimY;
  }
  __syncthreads();

  float4 sum;
  y_index   = threadIdx.y;
  row_index = element_y - RADIUS;
  while (row_index < (int)(((blockIdx.y + 1) << BlockShiftY) + RADIUS) &&
         row_index < dst_rows + RADIUS && element_x < dst_cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    mulAdd(sum, raw_data[y_index][threadIdx.x], 0.0625f);
    mulAdd(sum, raw_data[y_index][threadIdx.x + 1], 0.25f);
    mulAdd(sum, raw_data[y_index][threadIdx.x + 2], 0.375f);
    mulAdd(sum, raw_data[y_index][threadIdx.x + 3], 0.25f);
    mulAdd(sum, raw_data[y_index][threadIdx.x + 4], 0.0625f);

    col_data[y_index][threadIdx.x] = sum;
    y_index   += BlockDimY;
    row_index += BlockDimY;
  }
  __syncthreads();

  if (element_y < dst_rows && element_x < dst_cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    mulAdd(sum, col_data[threadIdx.y][threadIdx.x], 0.0625f);
    mulAdd(sum, col_data[threadIdx.y + 1][threadIdx.x], 0.25f);
    mulAdd(sum, col_data[threadIdx.y + 2][threadIdx.x], 0.375f);
    mulAdd(sum, col_data[threadIdx.y + 3][threadIdx.x], 0.25f);
    mulAdd(sum, col_data[threadIdx.y + 4][threadIdx.x], 0.0625f);

    sum.x *= 4;
    sum.y *= 4;
    sum.z *= 4;
    sum.w *= 4;

    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<Tn, float4>(sum);
  }
}

template <typename T, typename Tn>
__global__
void pyrupCnRef101Kernel(const T* src, int src_stride, T* dst, int dst_rows,
                         int dst_cols, int dst_stride) {
  __shared__ Tn raw_data[(BlockDimY >> 1) + RADIUS][(BlockDimX >> 1) + RADIUS];
  __shared__ float4 col_data[BlockDimY + (RADIUS << 1)][BlockDimX];

  int element_x = (blockIdx.x << BlockShiftX) + threadIdx.x;
  int element_y = (blockIdx.y << BlockShiftY) + threadIdx.y;

  int src_x = (blockIdx.x << (BlockShiftX - 1)) + threadIdx.x - 1;
  int src_y = (blockIdx.y << (BlockShiftY - 1)) + threadIdx.y - 1;
  if (threadIdx.y < (blockDim.y >> 1) + RADIUS &&
      threadIdx.x < (blockDim.x >> 1) + RADIUS &&
      src_y < (dst_rows >> 1) + 1 && src_x < (dst_cols >> 1) + 1) {
    src_x = src_x < 0 ? 0 - src_x : src_x;
    src_y = src_y < 0 ? 0 - src_y : src_y;
    int top = (dst_cols >> 1) - 1;
    src_x = src_x > top ? top : src_x;
    top = (dst_rows >> 1) - 1;
    src_y = src_y > top ? top : src_y;

    Tn* input = (Tn*)((uchar*)src + src_y * src_stride);
    raw_data[threadIdx.y][threadIdx.x] = input[src_x];
  }
  __syncthreads();

  int y_index   = threadIdx.y, index;
  int row_index = element_y - RADIUS;
  float4 sum;
  bool even_x = (threadIdx.x & 1) == 0 ? true : false;
  while (row_index < (int)(((blockIdx.y + 1) << BlockShiftY) + RADIUS) &&
         row_index < dst_rows + RADIUS && element_x < dst_cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    if ((y_index & 1) == 0) {
      index = y_index >> 1;
      mulAdd(sum, raw_data[index][(threadIdx.x >> 1)], even_x * 0.0625f);
      mulAdd(sum, raw_data[index][(threadIdx.x + 1) >> 1], (!even_x) * 0.25f);
      mulAdd(sum, raw_data[index][(threadIdx.x + 2) >> 1], even_x * 0.375f);
      mulAdd(sum, raw_data[index][(threadIdx.x + 3) >> 1], (!even_x) * 0.25f);
      mulAdd(sum, raw_data[index][(threadIdx.x + 4) >> 1], even_x * 0.0625f);
    }

    col_data[y_index][threadIdx.x] = sum;
    y_index   += BlockDimY;
    row_index += BlockDimY;
  }
  __syncthreads();

  if (element_y < dst_rows && element_x < dst_cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    mulAdd(sum, col_data[threadIdx.y][threadIdx.x], 0.0625f);
    mulAdd(sum, col_data[threadIdx.y + 1][threadIdx.x], 0.25f);
    mulAdd(sum, col_data[threadIdx.y + 2][threadIdx.x], 0.375f);
    mulAdd(sum, col_data[threadIdx.y + 3][threadIdx.x], 0.25f);
    mulAdd(sum, col_data[threadIdx.y + 4][threadIdx.x], 0.0625f);

    sum.x *= 4;
    sum.y *= 4;
    sum.z *= 4;
    sum.w *= 4;

    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<Tn, float4>(sum);
  }
}

#define RUN_C1_BATCH1_KERNELS(T, Interpolation)                                \
Interpolation interpolation;                                                   \
pyrupC1Kernel<T, Interpolation><<<grid, block, 0, stream>>>(src, src_stride,   \
    dst, dst_rows, dst_cols, dst_stride, interpolation);

#define RUN_C1_BATCH4_Reflect101_KERNELS(T)                                    \
pyrupC1Bat4Ref101Kernel<T><<<grid, block, 0, stream>>>(src, src_stride, dst,   \
    dst_rows, dst_cols, dst_stride);

#define RUN_CN_BATCH1_KERNELS(T, Interpolation)                                \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  pyrupCnKernel<T, T ## 3, Interpolation><<<grid, block, 0, stream>>>(src,     \
      src_stride, dst, dst_rows, dst_cols, dst_stride, interpolation);         \
}                                                                              \
else {                                                                         \
  pyrupCnKernel<T, T ## 4, Interpolation><<<grid, block, 0, stream>>>(src,     \
      src_stride, dst, dst_rows, dst_cols, dst_stride, interpolation);         \
}

#define RUN_CN_BATCH1_Reflect101_KERNELS(T)                                    \
if (channels == 3) {                                                           \
  pyrupCnRef101Kernel<T, T ## 3><<<grid, block, 0, stream>>>(src, src_stride,  \
      dst, dst_rows, dst_cols, dst_stride);                                    \
}                                                                              \
else {                                                                         \
  pyrupCnRef101Kernel<T, T ## 4><<<grid, block, 0, stream>>>(src, src_stride,  \
      dst, dst_rows, dst_cols, dst_stride);                                    \
}

RetCode pyrup(const uchar* src, int rows, int cols, int channels,
              int src_stride, uchar* dst, int dst_stride,
              BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * 2 * channels * (int)sizeof(uchar));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  int dst_rows = rows << 1;
  int dst_cols = cols << 1;
  dim3 block, grid;
  block.x = BlockDimX;
  block.y = BlockDimY;
  grid.x = divideUp(dst_cols, BlockDimX, BlockShiftX);
  grid.y = divideUp(dst_rows, BlockDimY, BlockShiftY);

  if (channels == 1) {
    if (border_type == BORDER_REPLICATE) {
      RUN_C1_BATCH1_KERNELS(uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_C1_BATCH1_KERNELS(uchar, ReflectBorder);
    }
    else {
      grid.x = divideUp(divideUp(dst_cols, 4, 2), BlockDimX, BlockShiftX);
      RUN_C1_BATCH4_Reflect101_KERNELS(uchar);
    }
  }
  else {
    if (border_type == BORDER_REPLICATE) {
      RUN_CN_BATCH1_KERNELS(uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CN_BATCH1_KERNELS(uchar, ReflectBorder);
    }
    else {
      RUN_CN_BATCH1_Reflect101_KERNELS(uchar);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode pyrup(const float* src, int rows, int cols, int channels,
              int src_stride, float* dst, int dst_stride,
              BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 2 * channels * (int)sizeof(float));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  int dst_rows = rows << 1;
  int dst_cols = cols << 1;
  dim3 block, grid;
  block.x = BlockDimX;
  block.y = BlockDimY;
  grid.x = divideUp(dst_cols, BlockDimX, BlockShiftX);
  grid.y = divideUp(dst_rows, BlockDimY, BlockShiftY);

  if (channels == 1) {
    if (border_type == BORDER_REPLICATE) {
      RUN_C1_BATCH1_KERNELS(float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_C1_BATCH1_KERNELS(float, ReflectBorder);
    }
    else {
      grid.x = divideUp(divideUp(dst_cols, 4, 2), BlockDimX, BlockShiftX);
      RUN_C1_BATCH4_Reflect101_KERNELS(float);
    }
  }
  else {
    if (border_type == BORDER_REPLICATE) {
      RUN_CN_BATCH1_KERNELS(float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CN_BATCH1_KERNELS(float, ReflectBorder);
    }
    else {
      RUN_CN_BATCH1_Reflect101_KERNELS(float);
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
RetCode PyrUp<uchar, 1>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const uchar* inData,
                        int outWidthStride,
                        uchar* outData,
                        BorderType border_type) {
  RetCode code = pyrup(inData, inHeight, inWidth, 1, inWidthStride, outData,
                       outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrUp<uchar, 3>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const uchar* inData,
                        int outWidthStride,
                        uchar* outData,
                        BorderType border_type) {
  RetCode code = pyrup(inData, inHeight, inWidth, 3, inWidthStride, outData,
                       outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrUp<uchar, 4>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const uchar* inData,
                        int outWidthStride,
                        uchar* outData,
                        BorderType border_type) {
  RetCode code = pyrup(inData, inHeight, inWidth, 4, inWidthStride, outData,
                       outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrUp<float, 1>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const float* inData,
                        int outWidthStride,
                        float* outData,
                        BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = pyrup(inData, inHeight, inWidth, 1, inWidthStride, outData,
                       outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrUp<float, 3>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const float* inData,
                        int outWidthStride,
                        float* outData,
                        BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = pyrup(inData, inHeight, inWidth, 3, inWidthStride, outData,
                       outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrUp<float, 4>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const float* inData,
                        int outWidthStride,
                        float* outData,
                        BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = pyrup(inData, inHeight, inWidth, 4, inWidthStride, outData,
                       outWidthStride, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
