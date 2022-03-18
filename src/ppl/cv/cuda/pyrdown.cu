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

#include "ppl/cv/cuda/pyrdown.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define RADIUS 2
#define BlockDimX 128
#define BlockDimY 2
#define BlockShiftX 7
#define BlockShiftY 1

template <typename T, typename BorderInterpolation>
__global__
void colRowC1Kernel0(const T* src, int rows, int cols, int src_stride, T* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  __shared__ float row_data[BlockDimY][(BlockDimX << 2) + RADIUS * 2];

  int element_x = ((blockIdx.x << BlockShiftX) + threadIdx.x) << 2;
  int element_y = ((blockIdx.y << BlockShiftY) + threadIdx.y) << 1;

  int y_index, row_index;
  int x_index = threadIdx.x << 2;
  int col_index0 = element_x - RADIUS;
  int col_index1, col_index2, col_index3, col_index4;
  T* input;
  float4 sum;

  if (element_y != 0 && element_y < ((rows + 1) >> 1) - 1) {
    while (col_index0 < (int)(((blockIdx.x + 1) << (BlockShiftX + 2)) + RADIUS)
           && col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);
      col_index2 = interpolation(cols, RADIUS, col_index0 + 1);
      col_index3 = interpolation(cols, RADIUS, col_index0 + 2);
      col_index4 = interpolation(cols, RADIUS, col_index0 + 3);

      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      sum.z += input[col_index3] * 0.0625f;
      sum.w += input[col_index4] * 0.0625f;
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      sum.z += input[col_index3] * 0.25f;
      sum.w += input[col_index4] * 0.25f;
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.375f;
      sum.y += input[col_index2] * 0.375f;
      sum.z += input[col_index3] * 0.375f;
      sum.w += input[col_index4] * 0.375f;
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      sum.z += input[col_index3] * 0.25f;
      sum.w += input[col_index4] * 0.25f;
      input = (T*)((uchar*)src + (row_index) * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      sum.z += input[col_index3] * 0.0625f;
      sum.w += input[col_index4] * 0.0625f;
      row_data[threadIdx.y][x_index] = sum.x;
      row_data[threadIdx.y][x_index + 1] = sum.y;
      row_data[threadIdx.y][x_index + 2] = sum.z;
      row_data[threadIdx.y][x_index + 3] = sum.w;

      x_index    += (BlockDimX << 2);
      col_index0 += (BlockDimX << 2);
    }
  }
  else {
    while (col_index0 < (int)(((blockIdx.x + 1) << (BlockShiftX + 2)) + RADIUS)
           && col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);
      col_index2 = interpolation(cols, RADIUS, col_index0 + 1);
      col_index3 = interpolation(cols, RADIUS, col_index0 + 2);
      col_index4 = interpolation(cols, RADIUS, col_index0 + 3);

      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      sum.z += input[col_index3] * 0.0625f;
      sum.w += input[col_index4] * 0.0625f;
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      sum.z += input[col_index3] * 0.25f;
      sum.w += input[col_index4] * 0.25f;
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.375f;
      sum.y += input[col_index2] * 0.375f;
      sum.z += input[col_index3] * 0.375f;
      sum.w += input[col_index4] * 0.375f;
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      sum.z += input[col_index3] * 0.25f;
      sum.w += input[col_index4] * 0.25f;
      y_index = interpolation(rows, RADIUS, row_index);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      sum.z += input[col_index3] * 0.0625f;
      sum.w += input[col_index4] * 0.0625f;
      row_data[threadIdx.y][x_index] = sum.x;
      row_data[threadIdx.y][x_index + 1] = sum.y;
      row_data[threadIdx.y][x_index + 2] = sum.z;
      row_data[threadIdx.y][x_index + 3] = sum.w;

      x_index    += (BlockDimX << 2);
      col_index0 += (BlockDimX << 2);
    }
  }
  __syncthreads();

  y_index = threadIdx.y;
  x_index = threadIdx.x << 2;
  col_index0 = x_index + 2;
  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    sum.x += row_data[y_index][x_index++] * 0.0625f;
    sum.y += row_data[y_index][col_index0++] * 0.0625f;
    sum.x += row_data[y_index][x_index++] * 0.25f;
    sum.y += row_data[y_index][col_index0++] * 0.25f;
    sum.x += row_data[y_index][x_index++] * 0.375f;
    sum.y += row_data[y_index][col_index0++] * 0.375f;
    sum.x += row_data[y_index][x_index++] * 0.25f;
    sum.y += row_data[y_index][col_index0++] * 0.25f;
    sum.x += row_data[y_index][x_index] * 0.0625f;
    sum.y += row_data[y_index][col_index0] * 0.0625f;

    T* output = (T*)((uchar*)dst + (element_y >> 1) * dst_stride);
    x_index = element_x >> 1;
    if (sizeof(T) == 1) {
      if (element_x < cols - 3) {
        output[x_index]     = saturateCast(sum.x);
        output[x_index + 1] = saturateCast(sum.y);
      }
      else {
        output[x_index] = saturateCast(sum.x);
        if (element_x < cols - 2) {
          output[x_index + 1] = saturateCast(sum.y);
        }
      }
    }
    else {
      if (element_x < cols - 3) {
        output[x_index]     = sum.x;
        output[x_index + 1] = sum.y;
      }
      else {
        output[x_index] = sum.x;
        if (element_x < cols - 2) {
          output[x_index + 1] = sum.y;
        }
      }
    }
  }
}

template <typename T, typename BorderInterpolation>
__global__
void colRowC1Kernel1(const T* src, int rows, int cols, int src_stride, T* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  __shared__ float row_data[BlockDimY][(BlockDimX << 1) + RADIUS * 2];

  int element_x = ((blockIdx.x << BlockShiftX) + threadIdx.x) << 1;
  int element_y = ((blockIdx.y << BlockShiftY) + threadIdx.y) << 1;

  int y_index, row_index;
  int x_index = threadIdx.x << 1;
  int col_index0 = element_x - RADIUS;
  int col_index1, col_index2;
  T* input;
  float4 sum;

  if (element_y != 0 && element_y < ((rows + 1) >> 1) - 1) {
    while (col_index0 < (int)(((blockIdx.x + 1) << (BlockShiftX + 1)) + RADIUS)
           && col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);
      col_index2 = interpolation(cols, RADIUS, col_index0 + 1);

      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.375f;
      sum.y += input[col_index2] * 0.375f;
      input = (T*)((uchar*)src + (row_index++) * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      input = (T*)((uchar*)src + (row_index) * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      row_data[threadIdx.y][x_index] = sum.x;
      row_data[threadIdx.y][x_index + 1] = sum.y;

      x_index    += (BlockDimX << 1);
      col_index0 += (BlockDimX << 1);
    }
  }
  else {
    while (col_index0 < (int)(((blockIdx.x + 1) << (BlockShiftX + 1)) + RADIUS)
           && col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);
      col_index2 = interpolation(cols, RADIUS, col_index0 + 1);

      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.375f;
      sum.y += input[col_index2] * 0.375f;
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.25f;
      sum.y += input[col_index2] * 0.25f;
      y_index = interpolation(rows, RADIUS, row_index);
      input = (T*)((uchar*)src + y_index * src_stride);
      sum.x += input[col_index1] * 0.0625f;
      sum.y += input[col_index2] * 0.0625f;
      row_data[threadIdx.y][x_index] = sum.x;
      row_data[threadIdx.y][x_index + 1] = sum.y;

      x_index    += (BlockDimX << 1);
      col_index0 += (BlockDimX << 1);
    }
  }
  __syncthreads();

  y_index = threadIdx.y;
  x_index = threadIdx.x << 1;
  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    sum.x += row_data[y_index][x_index++] * 0.0625f;
    sum.x += row_data[y_index][x_index++] * 0.25f;
    sum.x += row_data[y_index][x_index++] * 0.375f;
    sum.x += row_data[y_index][x_index++] * 0.25f;
    sum.x += row_data[y_index][x_index] * 0.0625f;

    T* output = (T*)((uchar*)dst + (element_y >> 1) * dst_stride);
    x_index = element_x >> 1;
    if (sizeof(T) == 1) {
      output[x_index] = saturateCast(sum.x);
    }
    else {
      output[x_index] = sum.x;
    }
  }
}

template <typename T, typename Tn, typename BorderInterpolation>
__global__
void colRowCnKernel0(const T* src, int rows, int cols, int src_stride, T* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  __shared__ float4 row_data[BlockDimY][BlockDimX + RADIUS * 2];

  int element_x = (blockIdx.x << BlockShiftX) + threadIdx.x;
  int element_y = ((blockIdx.y << BlockShiftY) + threadIdx.y) << 1;

  int y_index, row_index;
  int x_index = threadIdx.x;
  int col_index0 = element_x - RADIUS, col_index1;
  Tn* input;
  float4 sum;

  if (element_y != 0 && element_y < ((rows + 1) >> 1) - 1) {
    while (col_index0 < (int)(((blockIdx.x + 1) << BlockShiftX) + RADIUS) &&
           col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);

      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum, input[col_index1], 0.0625f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum, input[col_index1], 0.25f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum, input[col_index1], 0.375f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum, input[col_index1], 0.25f);
      input = (Tn*)((uchar*)src + (row_index) * src_stride);
      mulAdd(sum, input[col_index1], 0.0625f);
      row_data[threadIdx.y][x_index] = sum;

      x_index    += BlockDimX;
      col_index0 += BlockDimX;
    }
  }
  else {
    while (col_index0 < (int)(((blockIdx.x + 1) << BlockShiftX) + RADIUS) &&
           col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);

      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum, input[col_index1], 0.0625f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum, input[col_index1], 0.25f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum, input[col_index1], 0.375f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum, input[col_index1], 0.25f);
      y_index = interpolation(rows, RADIUS, row_index);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum, input[col_index1], 0.0625f);
      row_data[threadIdx.y][x_index] = sum;

      x_index    += BlockDimX;
      col_index0 += BlockDimX;
    }
  }
  __syncthreads();

  element_x = (blockIdx.x << (BlockShiftX - 1)) + threadIdx.x;
  y_index = threadIdx.y;
  x_index = threadIdx.x << 1;
  if (element_y < rows && element_x < ((cols + 1) >> 1) &&
      x_index < BlockDimX) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    mulAdd(sum, row_data[y_index][x_index++], 0.0625f);
    mulAdd(sum, row_data[y_index][x_index++], 0.25f);
    mulAdd(sum, row_data[y_index][x_index++], 0.375f);
    mulAdd(sum, row_data[y_index][x_index++], 0.25f);
    mulAdd(sum, row_data[y_index][x_index], 0.0625f);

    Tn* output = (Tn*)((uchar*)dst + (element_y >> 1) * dst_stride);
    output[element_x] = saturateCastVector<Tn, float4>(sum);
  }
}

template <typename T, typename Tn, typename BorderInterpolation>
__global__
void colRowCnKernel1(const T* src, int rows, int cols, int src_stride, T* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  __shared__ float4 row_data[BlockDimY][(BlockDimX << 1) + RADIUS * 2];

  int element_x = ((blockIdx.x << BlockShiftX) + threadIdx.x) << 1;
  int element_y = ((blockIdx.y << BlockShiftY) + threadIdx.y) << 1;

  int y_index, row_index;
  int x_index = threadIdx.x << 1;
  int col_index0 = element_x - RADIUS, col_index1, col_index2;
  Tn* input;
  float4 sum0, sum1;

  if (element_y != 0 && element_y < ((rows + 1) >> 1) - 1) {
    while (col_index0 < (int)(((blockIdx.x + 1) << (BlockShiftX + 1)) + RADIUS)
           && col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);
      col_index2 = interpolation(cols, RADIUS, col_index0 + 1);

      sum0 = make_float4(0.f, 0.f, 0.f, 0.f);
      sum1 = make_float4(0.f, 0.f, 0.f, 0.f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum0, input[col_index1], 0.0625f);
      mulAdd(sum1, input[col_index2], 0.0625f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum0, input[col_index1], 0.25f);
      mulAdd(sum1, input[col_index2], 0.25f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum0, input[col_index1], 0.375f);
      mulAdd(sum1, input[col_index2], 0.375f);
      input = (Tn*)((uchar*)src + (row_index++) * src_stride);
      mulAdd(sum0, input[col_index1], 0.25f);
      mulAdd(sum1, input[col_index2], 0.25f);
      input = (Tn*)((uchar*)src + (row_index) * src_stride);
      mulAdd(sum0, input[col_index1], 0.0625f);
      mulAdd(sum1, input[col_index2], 0.0625f);
      row_data[threadIdx.y][x_index] = sum0;
      row_data[threadIdx.y][x_index + 1] = sum1;

      x_index    += (BlockDimX << 1);
      col_index0 += (BlockDimX << 1);
    }
  }
  else {
    while (col_index0 < (int)(((blockIdx.x + 1) << (BlockShiftX + 1)) + RADIUS)
           && col_index0 < cols + RADIUS && element_y < rows) {
      row_index = element_y - RADIUS;
      col_index1 = interpolation(cols, RADIUS, col_index0);
      col_index2 = interpolation(cols, RADIUS, col_index0 + 1);

      sum0 = make_float4(0.f, 0.f, 0.f, 0.f);
      sum1 = make_float4(0.f, 0.f, 0.f, 0.f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum0, input[col_index1], 0.0625f);
      mulAdd(sum1, input[col_index2], 0.0625f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum0, input[col_index1], 0.25f);
      mulAdd(sum1, input[col_index2], 0.25f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum0, input[col_index1], 0.375f);
      mulAdd(sum1, input[col_index2], 0.375f);
      y_index = interpolation(rows, RADIUS, row_index++);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum0, input[col_index1], 0.25f);
      mulAdd(sum1, input[col_index2], 0.25f);
      y_index = interpolation(rows, RADIUS, row_index);
      input = (Tn*)((uchar*)src + y_index * src_stride);
      mulAdd(sum0, input[col_index1], 0.0625f);
      mulAdd(sum1, input[col_index2], 0.0625f);
      row_data[threadIdx.y][x_index] = sum0;
      row_data[threadIdx.y][x_index + 1] = sum1;

      x_index    += (BlockDimX << 1);
      col_index0 += (BlockDimX << 1);
    }
  }
  __syncthreads();

  y_index = threadIdx.y;
  x_index = threadIdx.x << 1;
  if (element_y < rows && element_x < cols) {
    sum0 = make_float4(0.f, 0.f, 0.f, 0.f);
    mulAdd(sum0, row_data[y_index][x_index++], 0.0625f);
    mulAdd(sum0, row_data[y_index][x_index++], 0.25f);
    mulAdd(sum0, row_data[y_index][x_index++], 0.375f);
    mulAdd(sum0, row_data[y_index][x_index++], 0.25f);
    mulAdd(sum0, row_data[y_index][x_index], 0.0625f);

    Tn* output = (Tn*)((uchar*)dst + (element_y >> 1) * dst_stride);
    x_index = element_x >> 1;
    output[x_index] = saturateCastVector<Tn, float4>(sum0);
  }
}

#define RUN_C1_BATCH4_KERNELS(T, Interpolation)                                \
Interpolation interpolation;                                                   \
colRowC1Kernel0<T, Interpolation><<<grid, block, 0, stream>>>(src, rows, cols, \
    src_stride, dst, dst_stride, interpolation);

#define RUN_C1_BATCH2_KERNELS(T, Interpolation)                                \
Interpolation interpolation;                                                   \
colRowC1Kernel1<T, Interpolation><<<grid, block, 0, stream>>>(src, rows, cols, \
    src_stride, dst, dst_stride, interpolation);

#define RUN_CN_BATCH1_KERNELS(T, Interpolation)                                \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  colRowCnKernel0<T, T ## 3, Interpolation><<<grid, block, 0, stream>>>(src,   \
      rows, cols, src_stride, dst, dst_stride, interpolation);                 \
}                                                                              \
else {                                                                         \
  colRowCnKernel0<T, T ## 4, Interpolation><<<grid, block, 0, stream>>>(src,   \
      rows, cols, src_stride, dst, dst_stride, interpolation);                 \
}

#define RUN_CN_BATCH2_KERNELS(T, Interpolation)                                \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  colRowCnKernel1<T, T ## 3, Interpolation><<<grid, block, 0, stream>>>(src,   \
      rows, cols, src_stride, dst, dst_stride, interpolation);                 \
}                                                                              \
else {                                                                         \
  colRowCnKernel1<T, T ## 4, Interpolation><<<grid, block, 0, stream>>>(src,   \
      rows, cols, src_stride, dst, dst_stride, interpolation);                 \
}

RetCode pyrdown(const uchar* src, int rows, int cols, int channels,
                int src_stride, uchar* dst, int dst_stride,
                BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= ((cols + 1) >> 1) * channels * (int)sizeof(uchar));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  int dst_rows = (rows + 1) >> 1;
  dim3 block, grid;
  block.x = BlockDimX;
  block.y = BlockDimY;
  grid.x = divideUp(divideUp(cols, 4, 2), BlockDimX, BlockShiftX);
  grid.y = divideUp(dst_rows, BlockDimY, BlockShiftY);

  if (channels == 1) {
    if (border_type == BORDER_REPLICATE) {
      RUN_C1_BATCH4_KERNELS(uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_C1_BATCH4_KERNELS(uchar, ReflectBorder);
    }
    else {
      RUN_C1_BATCH4_KERNELS(uchar, Reflect101Border);
    }
  }
  else if (channels == 3) {
    grid.x = divideUp(cols, BlockDimX, BlockShiftX);

    if (border_type == BORDER_REPLICATE) {
      RUN_CN_BATCH1_KERNELS(uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CN_BATCH1_KERNELS(uchar, ReflectBorder);
    }
    else {
      RUN_CN_BATCH1_KERNELS(uchar, Reflect101Border);
    }
  }
  else {
    grid.x = divideUp(divideUp(cols, 2, 1), BlockDimX, BlockShiftX);

    if (border_type == BORDER_REPLICATE) {
      RUN_CN_BATCH2_KERNELS(uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CN_BATCH2_KERNELS(uchar, ReflectBorder);
    }
    else {
      RUN_CN_BATCH2_KERNELS(uchar, Reflect101Border);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode pyrdown(const float* src, int rows, int cols, int channels,
                int src_stride, float* dst, int dst_stride,
                BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= ((cols + 1) >> 1) * channels * (int)sizeof(float));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  int dst_rows = (rows + 1) >> 1;
  dim3 block, grid;
  block.x = BlockDimX;
  block.y = BlockDimY;
  grid.x = divideUp(divideUp(cols, 2, 1), BlockDimX, BlockShiftX);
  grid.y = divideUp(dst_rows, BlockDimY, BlockShiftY);

  if (channels == 1) {
    if (border_type == BORDER_REPLICATE) {
      RUN_C1_BATCH2_KERNELS(float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_C1_BATCH2_KERNELS(float, ReflectBorder);
    }
    else {
      RUN_C1_BATCH2_KERNELS(float, Reflect101Border);
    }
  }
  else {
    grid.x = divideUp(cols, BlockDimX, BlockShiftX);

    if (border_type == BORDER_REPLICATE) {
      RUN_CN_BATCH1_KERNELS(float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CN_BATCH1_KERNELS(float, ReflectBorder);
    }
    else {
      RUN_CN_BATCH1_KERNELS(float, Reflect101Border);
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
RetCode PyrDown<uchar, 1>(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const uchar* inData,
                          int outWidthStride,
                          uchar* outData,
                          BorderType border_type) {
  RetCode code = pyrdown(inData, inHeight, inWidth, 1, inWidthStride, outData,
                         outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrDown<uchar, 3>(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const uchar* inData,
                          int outWidthStride,
                          uchar* outData,
                          BorderType border_type) {
  RetCode code = pyrdown(inData, inHeight, inWidth, 3, inWidthStride, outData,
                         outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrDown<uchar, 4>(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const uchar* inData,
                          int outWidthStride,
                          uchar* outData,
                          BorderType border_type) {
  RetCode code = pyrdown(inData, inHeight, inWidth, 4, inWidthStride, outData,
                         outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrDown<float, 1>(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const float* inData,
                          int outWidthStride,
                          float* outData,
                          BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = pyrdown(inData, inHeight, inWidth, 1, inWidthStride, outData,
                         outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrDown<float, 3>(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const float* inData,
                          int outWidthStride,
                          float* outData,
                          BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = pyrdown(inData, inHeight, inWidth, 3, inWidthStride, outData,
                         outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode PyrDown<float, 4>(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const float* inData,
                          int outWidthStride,
                          float* outData,
                          BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = pyrdown(inData, inHeight, inWidth, 4, inWidthStride, outData,
                         outWidthStride, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
