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

#include "ppl/cv/cuda/boxfilter.h"

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define RADIUS 8
#define SMALL_KSIZE RADIUS * 2 + 1

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void rowColC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                    int radius_x, int radius_y,  bool is_x_symmetric,
                    bool is_y_symmetric, bool normalize, float weight,
                    Tdst* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 * 3][(kDimX0 << 2)];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int bottom = element_x - radius_x;
  int top    = element_x + radius_x;
  if (!is_x_symmetric) {
    top -= 1;
  }

  int data_index, row_index;
  Tsrc* input;
  float4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius_x >> (kShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius_x) >> (kShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (threadIdx.y < radius_y && element_x < cols) {
    row_index = interpolation(rows, radius_y, element_y - radius_y);
    input = (Tsrc*)((uchar*)src + row_index * src_stride);
    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        sum += value;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius_x, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius_x, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius_x, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius_x, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }
    data_index = threadIdx.x << 2;
    data[threadIdx.y][data_index] = sum.x;
    data[threadIdx.y][data_index + 1] = sum.y;
    data[threadIdx.y][data_index + 2] = sum.z;
    data[threadIdx.y][data_index + 3] = sum.w;
  }

  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    input = (Tsrc*)((uchar*)src + element_y * src_stride);

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        sum += value;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius_x, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius_x, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius_x, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius_x, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }
    data_index = threadIdx.x << 2;
    data[radius_y + threadIdx.y][data_index] = sum.x;
    data[radius_y + threadIdx.y][data_index + 1] = sum.y;
    data[radius_y + threadIdx.y][data_index + 2] = sum.z;
    data[radius_y + threadIdx.y][data_index + 3] = sum.w;
  }

  if (threadIdx.y < radius_y && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.y != gridDim.y - 1) {
      row_index = interpolation(rows, radius_y,
                                ((blockIdx.y + 1) << kShiftY0) + threadIdx.y);
    }
    else {
      row_index = interpolation(rows, radius_y, rows + threadIdx.y);
    }
    input = (Tsrc*)((uchar*)src + row_index * src_stride);

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        sum += value;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius_x, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius_x, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius_x, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius_x, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }

    data_index = threadIdx.x << 2;
    if (blockIdx.y != gridDim.y - 1) {
      row_index = radius_y + kDimY0 + threadIdx.y;
    }
    else {
      row_index = radius_y + (rows - (blockIdx.y << kShiftY0)) + threadIdx.y;
    }
    data[row_index][data_index] = sum.x;
    data[row_index][data_index + 1] = sum.y;
    data[row_index][data_index + 2] = sum.z;
    data[row_index][data_index + 3] = sum.w;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    top = (radius_y << 1) + 1;
    if (!is_y_symmetric) {
      top -= 1;
    }
    sum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int i = 0; i < top; i++) {
      data_index = threadIdx.x << 2;
      value.x = data[i + threadIdx.y][data_index];
      value.y = data[i + threadIdx.y][data_index + 1];
      value.z = data[i + threadIdx.y][data_index + 2];
      value.w = data[i + threadIdx.y][data_index + 3];
      sum += value;
    }

    if (normalize) {
      sum.x *= weight;
      sum.y *= weight;
      sum.z *= weight;
      sum.w *= weight;
    }

    Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
    if (sizeof(Tdst) == 1) {
      if (element_x < cols - 3) {
        output[element_x]     = saturateCast(sum.x);
        output[element_x + 1] = saturateCast(sum.y);
        output[element_x + 2] = saturateCast(sum.z);
        output[element_x + 3] = saturateCast(sum.w);
      }
      else {
        output[element_x] = saturateCast(sum.x);
        if (element_x < cols - 1) {
          output[element_x + 1] = saturateCast(sum.y);
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = saturateCast(sum.z);
        }
      }
    }
    else {
      if (element_x < cols - 3) {
        output[element_x]     = sum.x;
        output[element_x + 1] = sum.y;
        output[element_x + 2] = sum.z;
        output[element_x + 3] = sum.w;
      }
      else {
        output[element_x] = sum.x;
        if (element_x < cols - 1) {
          output[element_x + 1] = sum.y;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = sum.z;
        }
      }
    }
  }
}

template <typename Tsrc, typename Tsrcn, typename Tbufn, typename Tdst,
          typename Tdstn, typename BorderInterpolation>
__global__
void rowColCnKernel(const Tsrc* src, int rows, int cols, int src_stride,
                    int radius_x, int radius_y, bool is_x_symmetric,
                    bool is_y_symmetric, bool normalize, float weight,
                    Tdst* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  __shared__ Tsrcn row_data[kDimY0 + RADIUS * 2][kDimX0 + RADIUS * 2];
  __shared__ Tbufn col_data[kDimY0 + RADIUS * 2][kDimX0];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int index, y_index, row_index;
  int ksize_x = (radius_x << 1) + 1;
  int ksize_y = (radius_y << 1) + 1;
  if (!is_x_symmetric) {
    ksize_x -= 1;
  }
  if (!is_y_symmetric) {
    ksize_y -= 1;
  }
  Tsrcn* input;
  float4 sum;

  y_index   = threadIdx.y;
  row_index = element_y - radius_y;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius_y) &&
         row_index < rows + radius_y) {
    index = interpolation(rows, radius_y, row_index);
    input = (Tsrcn*)((uchar*)src + index * src_stride);

    int x_index   = threadIdx.x;
    int col_index = element_x - radius_x;
    while (col_index < (int)(((blockIdx.x + 1) << kShiftX0) + radius_x) &&
           col_index < cols + radius_x) {
      index = interpolation(cols, radius_x, col_index);
      row_data[y_index][x_index] = input[index];
      x_index   += kDimX0;
      col_index += kDimX0;
    }

    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  y_index   = threadIdx.y;
  row_index = element_y - radius_y;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius_y) &&
         row_index < rows + radius_y && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (index = 0; index < ksize_x; index++) {
      sum += row_data[y_index][threadIdx.x + index];
    }

    col_data[y_index][threadIdx.x] = transform<Tbufn>(sum);
    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (index = 0; index < ksize_y; index++) {
      sum += col_data[threadIdx.y + index][threadIdx.x];
    }

    if (normalize) {
      sum.x *= weight;
      sum.y *= weight;
      sum.z *= weight;
      sum.w *= weight;
    }

    Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<Tdstn, float4>(sum);
  }
}

template <typename Tsrc, typename Tsrc4, typename BorderInterpolation>
__global__
void rowBatch4Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                     int radius_x, bool is_x_symmetric, float* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius_x;
  int top_x    = element_x + radius_x;
  if (!is_x_symmetric) {
    top_x -= 1;
  }

  int data_index;
  Tsrc* input;
  Tsrc4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius_x >> (kBlockShiftX1 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius_x) >> (kBlockShiftX1 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (Tsrc*)((uchar*)src + element_y * src_stride);
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value.x = input[i];
      value.y = input[i + 1];
      value.z = input[i + 2];
      value.w = input[i + 3];
      sum += value;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius_x, i);
      value.x = input[data_index];
      data_index = interpolation(cols, radius_x, i + 1);
      value.y = input[data_index];
      data_index = interpolation(cols, radius_x, i + 2);
      value.z = input[data_index];
      data_index = interpolation(cols, radius_x, i + 3);
      value.w = input[data_index];
      sum += value;
    }
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 3) {
    output[element_x]     = sum.x;
    output[element_x + 1] = sum.y;
    output[element_x + 2] = sum.z;
    output[element_x + 3] = sum.w;
  }
  else {
    output[element_x] = sum.x;
    if (element_x < cols - 1) {
      output[element_x + 1] = sum.y;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = sum.z;
    }
  }
}

template <typename Tsrc, typename Tsrc4, typename Tdst4,
          typename BorderInterpolation>
__global__
void rowBatch2Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                     int radius_x, bool is_x_symmetric, float* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius_x;
  int top_x    = element_x + radius_x;
  if (!is_x_symmetric) {
    top_x -= 1;
  }

  int data_index;
  Tsrc4* input;
  Tsrc4 value0;
  Tsrc4 value1;
  float4 sum0 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 sum1 = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius_x >> (kBlockShiftX1 + 1);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius_x) >> (kBlockShiftX1 + 1);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (Tsrc4*)((uchar*)src + element_y * src_stride);
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value0 = input[i];
      value1 = input[i + 1];
      sum0 += value0;
      sum1 += value1;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius_x, i);
      value0 = input[data_index];
      data_index = interpolation(cols, radius_x, i + 1);
      value1 = input[data_index];
      sum0 += value0;
      sum1 += value1;
    }
  }

  Tdst4* output = (Tdst4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tdst4, float4>(sum0);
  if (element_x < cols - 1) {
    output[element_x + 1] = saturateCastVector<Tdst4, float4>(sum1);
  }
}

template <typename Tsrc, typename Tsrcn, typename Tdstn,
          typename BorderInterpolation>
__global__
void rowSharedKernel(const Tsrc* src, int rows, int cols, int src_stride,
                     int radius_x, bool is_x_symmetric, float* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  __shared__ Tsrcn data[kBlockDimY1][(kBlockDimX1 << 1)];

  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows) {
    return;
  }

  Tsrcn* input = (Tsrcn*)((uchar*)src + element_y * src_stride);
  Tsrcn value;
  int index;
  int ksize_x = (radius_x << 1) + 1;
  if (!is_x_symmetric) {
    ksize_x -= 1;
  }

  if (threadIdx.x < radius_x) {
    if (blockIdx.x == 0) {
      index = interpolation(cols, radius_x, element_x - radius_x);
    }
    else {
      index = element_x - radius_x;
    }
    value = input[index];
    data[threadIdx.y][threadIdx.x] = value;
  }

  if (element_x < cols) {
    value = input[element_x];
    data[threadIdx.y][radius_x + threadIdx.x] = value;
  }

  if (threadIdx.x < radius_x) {
    index = (cols - radius_x) >> kBlockShiftX1;
    if (blockIdx.x >= index) {
      if (blockIdx.x != gridDim.x - 1) {
        index = interpolation(cols, radius_x, element_x + kBlockDimX1);
        value = input[index];
        data[threadIdx.y][radius_x + kBlockDimX1 + threadIdx.x] = value;
      }
      else {
        index = interpolation(cols, radius_x, cols + threadIdx.x);
        value = input[index];
        index = cols - (blockIdx.x << kBlockShiftX1);
        data[threadIdx.y][radius_x + index + threadIdx.x] = value;
      }
    }
    else {
      index = element_x + kBlockDimX1;
      value = input[index];
      data[threadIdx.y][radius_x + kBlockDimX1 + threadIdx.x] = value;
    }
  }
  __syncthreads();

  if (element_x >= cols) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (index = 0; index < ksize_x; index++) {
    sum += data[threadIdx.y][threadIdx.x + index];
  }

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tdstn, float4>(sum);
}

template <typename Tsrc, typename Tsrcn, typename Tdstn,
          typename BorderInterpolation>
__global__
void rowFilterKernel(const Tsrc* src, int rows, int cols, int src_stride,
                     int radius_x, bool is_x_symmetric, float* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius_x;
  int top_x    = element_x + radius_x;
  if (!is_x_symmetric) {
    top_x -= 1;
  }

  int data_index;
  Tsrcn* input;
  Tsrcn value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius_x >> kBlockShiftX1;
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius_x) >> kBlockShiftX1;
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (Tsrcn*)((uchar*)src + element_y * src_stride);
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value = input[i];
      sum += value;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius_x, i);
      value = input[data_index];
      sum += value;
    }
  }

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tdstn, float4>(sum);
}

template <typename Tdst, typename BorderInterpolation>
__global__
void colSharedKernel(const float* src, int rows, int cols4, int cols,
                     int src_stride, int radius_y, bool is_y_symmetric,
                     bool normalize, float weight, Tdst* dst, int dst_stride,
                     BorderInterpolation interpolation) {
  __shared__ float4 data[kDimY0 * 3][kDimX0];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;
  if (element_x >= cols4) {
    return;
  }

  float4* input;
  float4 value;
  int index;
  int ksize_y = (radius_y << 1) + 1;
  if (!is_y_symmetric) {
    ksize_y -= 1;
  }

  if (threadIdx.y < radius_y) {
    if (blockIdx.y == 0) {
      index = interpolation(rows, radius_y, element_y - radius_y);
    }
    else {
      index = element_y - radius_y;
    }
    input = (float4*)((uchar*)src + index * src_stride);
    value = input[element_x];
    data[threadIdx.y][threadIdx.x] = value;
  }

  if (element_y < rows) {
    input = (float4*)((uchar*)src + element_y * src_stride);
    value = input[element_x];
    data[radius_y + threadIdx.y][threadIdx.x] = value;
  }

  if (threadIdx.y < radius_y) {
    index = (rows - radius_y) >> kShiftY0;
    if (blockIdx.y >= index) {
      if (blockIdx.y != gridDim.y - 1) {
        index = interpolation(rows, radius_y, element_y + kDimY0);
        input = (float4*)((uchar*)src + index * src_stride);
        value = input[element_x];
        data[radius_y + kDimY0 + threadIdx.y][threadIdx.x] = value;
      }
      else {
        index = interpolation(rows, radius_y, rows + threadIdx.y);
        input = (float4*)((uchar*)src + index * src_stride);
        value = input[element_x];
        index = rows - (blockIdx.y << kShiftY0);
        data[radius_y + index + threadIdx.y][threadIdx.x] = value;
      }
    }
    else {
      index = element_y + kDimY0;
      input = (float4*)((uchar*)src + index * src_stride);
      value = input[element_x];
      data[radius_y + kDimY0 + threadIdx.y][threadIdx.x] = value;
    }
  }
  __syncthreads();

  if (element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (index = 0; index < ksize_y; index++) {
    sum += data[threadIdx.y + index][threadIdx.x];
  }

  if (normalize) {
    sum.x *= weight;
    sum.y *= weight;
    sum.z *= weight;
    sum.w *= weight;
  }

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  index = element_x << 2;
  if (element_x < cols4 - 1) {
    if (sizeof(Tdst) == 1) {
      output[index] = saturateCast(sum.x);
      output[index + 1] = saturateCast(sum.y);
      output[index + 2] = saturateCast(sum.z);
      output[index + 3] = saturateCast(sum.w);
    }
    else {
      output[index] = sum.x;
      output[index + 1] = sum.y;
      output[index + 2] = sum.z;
      output[index + 3] = sum.w;
    }
  }
  else {
    if (sizeof(Tdst) == 1) {
      output[index] = saturateCast(sum.x);
      if (index < cols - 1) {
        output[index + 1] = saturateCast(sum.y);
      }
      if (index < cols - 2) {
        output[index + 2] = saturateCast(sum.z);
      }
      if (index < cols - 3) {
        output[index + 3] = saturateCast(sum.w);
      }
    }
    else {
      output[index] = sum.x;
      if (index < cols - 1) {
        output[index + 1] = sum.y;
      }
      if (index < cols - 2) {
        output[index + 2] = sum.z;
      }
      if (index < cols - 3) {
        output[index + 3] = sum.w;
      }
    }
  }
}

template <typename Tdst, typename BorderInterpolation>
__global__
void colBatch4Kernel(const float* src, int rows, int cols, int src_stride,
                     int radius_y, bool is_y_symmetric, bool normalize,
                     float weight, Tdst* dst, int dst_stride,
                     BorderInterpolation interpolation) {
  __shared__ Tdst data[kBlockDimY1][kBlockDimX1 << 2];

  int element_x = (blockIdx.x << (kBlockShiftX1 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_y = element_y - radius_y;
  int top_y    = element_y + radius_y;
  if (!is_y_symmetric) {
    top_y -= 1;
  }

  int data_index;
  float* input;
  float value;
  float sum = 0.f;

  bool isnt_border_block = true;
  data_index = radius_y >> kBlockShiftY1;
  if (blockIdx.y <= data_index) isnt_border_block = false;
  data_index = (rows - radius_y) >> kBlockShiftY1;
  if (blockIdx.y >= data_index) isnt_border_block = false;

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      input = (float*)((uchar*)src + i * src_stride);
      value = input[element_x];
      sum += value;
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius_y, i);
      input = (float*)((uchar*)src + data_index * src_stride);
      value = input[element_x];
      sum += value;
    }
  }

  if (normalize) {
    sum *= weight;
  }

  if (sizeof(Tdst) == 1) {
    data[threadIdx.y][threadIdx.x] = saturateCast(sum);
  }
  __syncthreads();

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  if (sizeof(Tdst) == 1) {
    if (threadIdx.x < kBlockDimX1) {
      element_x = (((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2);
      data_index = threadIdx.x << 2;
      if (element_x < cols - 3) {
        output[element_x]     = data[threadIdx.y][data_index];
        output[element_x + 1] = data[threadIdx.y][data_index + 1];
        output[element_x + 2] = data[threadIdx.y][data_index + 2];
        output[element_x + 3] = data[threadIdx.y][data_index + 3];
      }
      else if (element_x < cols) {
        output[element_x] = data[threadIdx.y][data_index];
        if (element_x < cols - 1) {
          output[element_x + 1] = data[threadIdx.y][data_index + 1];
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = data[threadIdx.y][data_index + 2];
        }
      }
      else {
      }
    }
  }
  else {
    output[element_x] = sum;
  }
}

template <typename Tsrcn, typename Tdst, typename Tdstn,
          typename BorderInterpolation>
__global__
void colFilterKernel(const float* src, int rows, int cols, int src_stride,
                     int radius_y, bool is_y_symmetric, bool normalize,
                     float weight, Tdst* dst, int dst_stride,
                     BorderInterpolation interpolation) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_y = element_y - radius_y;
  int top_y    = element_y + radius_y;
  if (!is_y_symmetric) {
    top_y -= 1;
  }

  int data_index;
  Tsrcn* input;
  Tsrcn value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int i = origin_y; i <= top_y; i++) {
    data_index = interpolation(rows, radius_y, i);
    input = (Tsrcn*)((uchar*)src + data_index * src_stride);
    value = input[element_x];
    sum += value;
  }

  if (normalize) {
    sum.x *= weight;
    sum.y *= weight;
    sum.z *= weight;
    sum.w *= weight;
  }

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tdstn, float4>(sum);
}

#define RUN_CHANNEL1_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
rowColC1Kernel<Tsrc, Tdst, Interpolation><<<grid, block, 0, stream>>>(src,     \
    rows, cols, src_stride, radius_x, radius_y, is_x_symmetric, is_y_symmetric,\
    normalize, weight, dst, dst_stride, interpolation);

#define RUN_CHANNELN_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  rowColCnKernel<Tsrc, Tsrc ## 3, float ## 3, Tdst, Tdst ## 3, Interpolation>  \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, radius_x,      \
      radius_y, is_x_symmetric, is_y_symmetric, normalize, weight, dst,        \
      dst_stride, interpolation);                                              \
}                                                                              \
else {                                                                         \
  rowColCnKernel<Tsrc, Tsrc ## 4, float ## 4, Tdst, Tdst ## 4, Interpolation>  \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, radius_x,      \
      radius_y, is_x_symmetric, is_y_symmetric, normalize, weight, dst,        \
      dst_stride, interpolation);                                              \
}

#define RUN_KERNELS(Tsrc, Tdst, Interpolation)                                 \
Interpolation interpolation;                                                   \
if (channels == 1) {                                                           \
  rowBatch4Kernel<Tsrc, Tsrc ## 4, Interpolation><<<grid1, block, 0, stream    \
      >>>(src, rows, cols, src_stride, radius_x, is_x_symmetric, buffer, pitch,\
      interpolation);                                                          \
  if (ksize_x <= 33 && ksize_y <= 33) {                                        \
    colSharedKernel<Tdst, Interpolation><<<grid3, block3, 0, stream>>>(buffer, \
        rows, columns4, columns, pitch, radius_y, is_y_symmetric, normalize,   \
        weight, dst, dst_stride, interpolation);                               \
  }                                                                            \
  else {                                                                       \
    colBatch4Kernel<Tdst, Interpolation><<<grid4, block4, 0, stream>>>(buffer, \
        rows, columns, pitch, radius_y, is_y_symmetric, normalize, weight, dst,\
        dst_stride, interpolation);                                            \
  }                                                                            \
}                                                                              \
else if (channels == 3) {                                                      \
  if (ksize_x <= 33 && ksize_y <= 33) {                                        \
    rowSharedKernel<Tsrc, Tsrc ## 3, float ## 3, Interpolation><<<grid, block, \
        0, stream>>>(src, rows, cols, src_stride, radius_x, is_x_symmetric,    \
        buffer, pitch, interpolation);                                         \
    colSharedKernel<Tdst, Interpolation><<<grid3, block3, 0, stream>>>(buffer, \
        rows, columns4, columns, pitch, radius_y, is_y_symmetric, normalize,   \
        weight, dst, dst_stride, interpolation);                               \
  }                                                                            \
  else {                                                                       \
    rowFilterKernel<Tsrc, Tsrc ## 3, float ## 3, Interpolation><<<grid, block, \
        0, stream>>>(src, rows, cols, src_stride, radius_x, is_x_symmetric,    \
        buffer, pitch, interpolation);                                         \
    colBatch4Kernel<Tdst, Interpolation><<<grid4, block4, 0, stream>>>(buffer, \
        rows, columns, pitch, radius_y, is_y_symmetric, normalize, weight, dst,\
        dst_stride, interpolation);                                            \
  }                                                                            \
}                                                                              \
else {                                                                         \
  if (ksize_x <= 33 && ksize_y <= 33) {                                        \
    rowSharedKernel<Tsrc, Tsrc ## 4, float ## 4, Interpolation><<<grid, block, \
        0, stream>>>(src, rows, cols, src_stride, radius_x, is_x_symmetric,    \
        buffer, pitch, interpolation);                                         \
    colSharedKernel<Tdst, Interpolation><<<grid3, block3, 0, stream>>>(buffer, \
        rows, columns4, columns, pitch, radius_y, is_y_symmetric, normalize,   \
        weight, dst, dst_stride, interpolation);                               \
  }                                                                            \
  else {                                                                       \
    rowBatch2Kernel<Tsrc, Tsrc ## 4, float ## 4, Interpolation><<<grid2, block,\
        0, stream>>>(src, rows, cols, src_stride, radius_x, is_x_symmetric,    \
        buffer, pitch, interpolation);                                         \
    colFilterKernel<float ## 4, Tdst, Tdst ## 4, Interpolation><<<grid, block, \
        0, stream>>>(buffer, rows, cols, pitch, radius_y, is_y_symmetric,      \
        normalize, weight, dst, dst_stride, interpolation);                    \
  }                                                                            \
}

RetCode boxFilter(const uchar* src, int rows, int cols, int channels,
                  int src_stride, int ksize_x, int ksize_y, bool normalize,
                  uchar* dst, int dst_stride, BorderType border_type,
                  cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize_x > 0);
  PPL_ASSERT(ksize_y > 0);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize_x == 1 && ksize_y == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, rows * src_stride, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int radius_x = ksize_x >> 1;
  int radius_y = ksize_y >> 1;
  bool is_x_symmetric = ksize_x & 1;
  bool is_y_symmetric = ksize_y & 1;
  float weight = 1.f / (ksize_x * ksize_y);

  if (ksize_x <= SMALL_KSIZE && ksize_y <= SMALL_KSIZE && channels == 1) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, uchar, ReflectBorder);
    }
    else {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, uchar, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  if (ksize_x <= SMALL_KSIZE && ksize_y <= SMALL_KSIZE &&
      (channels == 3 || channels == 4)) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(cols, kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNELN_SMALL_KERNELS(uchar, uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS(uchar, uchar, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS(uchar, uchar, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid1;
  grid1.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid1.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid2;
  grid2.x = divideUp(divideUp(cols, 2, 1), kBlockDimX1, kBlockShiftX1);
  grid2.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block3, grid3;
  block3.x = kDimX0;
  block3.y = kDimY0;
  int columns = cols * channels;
  int columns4 = divideUp(columns, 4, 2);
  grid3.x = divideUp(columns4, kDimX0, kShiftX0);
  grid3.y = divideUp(rows, kDimY0, kShiftY0);

  dim3 block4, grid4;
  block4.x = (kBlockDimX1 << 2);
  block4.y = kBlockDimY1;
  grid4.x  = divideUp(columns, (kBlockDimX1 << 2), (kBlockShiftX1 + 2));
  grid4.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  float* buffer;
  size_t pitch;

  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * channels * sizeof(float), rows, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch  = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch, cols * channels * sizeof(float),
                           rows);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  if (border_type == BORDER_REPLICATE) {
    RUN_KERNELS(uchar, uchar, ReplicateBorder);
  }
  else if (border_type == BORDER_REFLECT) {
    RUN_KERNELS(uchar, uchar, ReflectBorder);
  }
  else {
    RUN_KERNELS(uchar, uchar, Reflect101Border);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
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

RetCode boxFilter(const float* src, int rows, int cols, int channels,
                  int src_stride, int ksize_x, int ksize_y, bool normalize,
                  float* dst, int dst_stride, BorderType border_type,
                  cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize_x > 0);
  PPL_ASSERT(ksize_y > 0);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize_x == 1 && ksize_y == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, rows * src_stride, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int radius_x = ksize_x >> 1;
  int radius_y = ksize_y >> 1;
  bool is_x_symmetric = ksize_x & 1;
  bool is_y_symmetric = ksize_y & 1;
  float weight = 1.f / (ksize_x * ksize_y);

  if (ksize_x <= SMALL_KSIZE && ksize_y <= SMALL_KSIZE && channels == 1) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNEL1_SMALL_KERNELS(float, float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNEL1_SMALL_KERNELS(float, float, ReflectBorder);
    }
    else {
      RUN_CHANNEL1_SMALL_KERNELS(float, float, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  if (ksize_x <= SMALL_KSIZE && ksize_y <= SMALL_KSIZE &&
      (channels == 3 || channels == 4)) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(cols, kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNELN_SMALL_KERNELS(float, float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS(float, float, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS(float, float, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid1;
  grid1.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid1.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid2;
  grid2.x = divideUp(divideUp(cols, 2, 1), kBlockDimX1, kBlockShiftX1);
  grid2.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block3, grid3;
  block3.x = kDimX0;
  block3.y = kDimY0;
  int columns = cols * channels;
  int columns4 = divideUp(columns, 4, 2);
  grid3.x = divideUp(columns4, kDimX0, kShiftX0);
  grid3.y = divideUp(rows, kDimY0, kShiftY0);

  dim3 block4, grid4;
  block4.x = (kBlockDimX1 << 2);
  block4.y = kBlockDimY1;
  grid4.x  = divideUp(columns, (kBlockDimX1 << 2), (kBlockShiftX1 + 2));
  grid4.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  float* buffer;
  size_t pitch;

  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * channels * sizeof(float), rows, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch  = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch, cols * channels * sizeof(float),
                           rows);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  if (border_type == BORDER_REPLICATE) {
    RUN_KERNELS(float, float, ReplicateBorder);
  }
  else if (border_type == BORDER_REFLECT) {
    RUN_KERNELS(float, float, ReflectBorder);
  }
  else {
    RUN_KERNELS(float, float, Reflect101Border);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
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
RetCode BoxFilter<uchar, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            uchar* outData,
                            BorderType border_type) {
  RetCode code = boxFilter(inData, height, width, 1, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<uchar, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            uchar* outData,
                            BorderType border_type) {
  RetCode code = boxFilter(inData, height, width, 3, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<uchar, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            uchar* outData,
                            BorderType border_type) {
  RetCode code = boxFilter(inData, height, width, 4, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<float, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            float* outData,
                            BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = boxFilter(inData, height, width, 1, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<float, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            float* outData,
                            BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = boxFilter(inData, height, width, 3, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<float, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            float* outData,
                            BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = boxFilter(inData, height, width, 4, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
