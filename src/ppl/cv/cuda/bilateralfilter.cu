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

#include "ppl/cv/cuda/bilateralfilter.h"

#include <cmath>

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define RADIUS0 16
#define SMALL_KSIZE0 RADIUS0 * 2 + 1

#define RADIUS1 8
#define SMALL_KSIZE1 RADIUS1 * 2 + 1

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void filter2DC1SharedKernel(const Tsrc* src, int rows, int cols, int src_stride,
                            int radius, float radius_sqr, float color_coeff,
                            float space_coeff, Tdst* dst, int dst_stride,
                            BorderInterpolation interpolation) {
  __shared__ Tsrc data[kDimY0 + RADIUS0 * 2][(kDimX0 << 2) + RADIUS0 * 2];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int index, y_index, row_index, col_index;
  float space_sqr;
  Tsrc* input;
  Tsrc value0, value1, value2, value3;

  y_index   = threadIdx.y;
  row_index = element_y - radius;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
         row_index < rows + radius) {
    index = interpolation(rows, radius, row_index);
    input = (Tsrc*)((uchar*)src + index * src_stride);

    if (threadIdx.x < radius) {
      if (blockIdx.x == 0) {
        index = interpolation(cols, radius, threadIdx.x - radius);
      }
      else {
        index = (blockIdx.x << (kShiftX0 + 2)) + threadIdx.x - radius;
      }
      value0 = input[index];
      data[y_index][threadIdx.x] = value0;
    }

    if (element_x < cols) {
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];
      index = radius + (threadIdx.x << 2);
      data[y_index][index] = value0;
      data[y_index][index + 1] = value1;
      data[y_index][index + 2] = value2;
      data[y_index][index + 3] = value3;
    }

    if (threadIdx.x < radius) {
      index = (cols - radius) >> (kShiftX0 + 2);
      if (blockIdx.x >= index) {
        if (blockIdx.x != gridDim.x - 1) {
          index = ((blockIdx.x + 1) << (kShiftX0 + 2)) + threadIdx.x;
          index = interpolation(cols, radius, index);
          value0 = input[index];
          index = radius + (kDimX0 << 2) + threadIdx.x;
          data[y_index][index] = value0;
        }
        else {
          index = interpolation(cols, radius, cols + threadIdx.x);
          value0 = input[index];
          index = cols - (blockIdx.x << (kShiftX0 + 2));
          index += (radius + threadIdx.x);
          data[y_index][index] = value0;
        }
      }
      else {
        index = ((blockIdx.x + 1) << (kShiftX0 + 2)) + threadIdx.x;
        value0 = input[index];
        index = radius + (kDimX0 << 2) + threadIdx.x;
        data[y_index][index] = value0;
      }
    }

    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 weight_sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 weight;
  float4 center;
  center.x = data[threadIdx.y + radius][(threadIdx.x << 2) + radius];
  center.y = data[threadIdx.y + radius][(threadIdx.x << 2) + radius + 1];
  center.z = data[threadIdx.y + radius][(threadIdx.x << 2) + radius + 2];
  center.w = data[threadIdx.y + radius][(threadIdx.x << 2) + radius + 3];

  y_index = threadIdx.y;
  for (row_index = -radius; row_index <= radius; row_index++) {
    for (col_index = -radius; col_index <= radius; col_index++) {
      space_sqr = row_index * row_index + col_index * col_index;
      if (space_sqr <= radius_sqr) {
        space_sqr *= space_coeff;
        index = (threadIdx.x << 2) + radius + col_index;
        value0 = data[y_index][index];
        value1 = data[y_index][index + 1];
        value2 = data[y_index][index + 2];
        value3 = data[y_index][index + 3];
        weight.x = expf(space_sqr + (value0 - center.x) * (value0 - center.x) *
                        color_coeff);
        weight.y = expf(space_sqr + (value1 - center.y) * (value1 - center.y) *
                        color_coeff);
        weight.z = expf(space_sqr + (value2 - center.z) * (value2 - center.z) *
                        color_coeff);
        weight.w = expf(space_sqr + (value3 - center.w) * (value3 - center.w) *
                        color_coeff);
        sum.x += weight.x * value0;
        sum.y += weight.y * value1;
        sum.z += weight.z * value2;
        sum.w += weight.w * value3;
        weight_sum += weight;
      }
    }
    y_index++;
  }
  sum /= weight_sum;

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  if (sizeof(Tsrc) == 1) {
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

template <typename Tsrc, typename Tsrcn, typename Tdst, typename Tdstn,
          typename BorderInterpolation>
__global__
void filter2DCnSharedKernel0(const Tsrc* src, int rows, int cols,
                             int src_stride, int radius, float radius_sqr,
                             float color_coeff, float space_coeff, Tdst* dst,
                             int dst_stride,
                             BorderInterpolation interpolation) {
  __shared__ Tsrcn data[kDimY0 + RADIUS1 * 2][kDimX0 + RADIUS1 * 2];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int index, y_index, row_index, col_index;
  float space_sqr, color_tmp;
  Tsrcn* input;

  y_index   = threadIdx.y;
  row_index = element_y - radius;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
         row_index < rows + radius) {
    index = interpolation(rows, radius, row_index);
    input = (Tsrcn*)((uchar*)src + index * src_stride);

    int x_index = threadIdx.x;
    col_index = element_x - radius;
    while (col_index < (int)(((blockIdx.x + 1) << kShiftX0) + radius) &&
           col_index < cols + radius) {
      index = interpolation(cols, radius, col_index);
      data[y_index][x_index] = input[index];
      x_index   += kDimX0;
      col_index += kDimX0;
    }

    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float weight_sum = 0.f;
  float weight;
  Tsrcn center = data[threadIdx.y + radius][threadIdx.x + radius];
  Tsrcn value;

  y_index = threadIdx.y;
  for (row_index = -radius; row_index <= radius; row_index++) {
    for (col_index = -radius; col_index <= radius; col_index++) {
      space_sqr = row_index * row_index + col_index * col_index;
      if (space_sqr <= radius_sqr) {
        space_sqr *= space_coeff;
        value = data[y_index][threadIdx.x + radius + col_index];
        color_tmp = fabsf(value.x - center.x) + fabsf(value.y - center.y) +
                    fabsf(value.z - center.z);
        weight = expf(space_sqr + color_tmp * color_tmp * color_coeff);
        sum.x += weight * value.x;
        sum.y += weight * value.y;
        sum.z += weight * value.z;
        weight_sum += weight;
      }
    }
    y_index++;
  }
  sum /= weight_sum;

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tdstn, float4>(sum);
}

template <typename Tsrc, typename Tsrc4, typename Tdst, typename Tdst4,
          typename BorderInterpolation>
__global__
void filter2DC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                      int radius, float radius_sqr, float color_coeff,
                      float space_coeff, Tdst* dst, int dst_stride,
                      BorderInterpolation interpolation) {
  int element_x, element_y;
  if (sizeof(Tsrc) == 1) {
    element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int data_index;
  bool isnt_border_block = true;
  if (sizeof(Tsrc) == 1) {
    data_index = radius >> (kBlockShiftX0 + 2);
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> (kBlockShiftX0 + 2);
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }
  else {
    data_index = radius >> (kBlockShiftX1 + 2);
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> (kBlockShiftX1 + 2);
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }

  Tsrc* input = (Tsrc*)((uchar*)src + element_y * src_stride);
  Tsrc4 value;

  float space_sqr;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 weight_sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 weight;
  Tsrc4 center;
  center.x = input[element_x];
  center.y = input[element_x + 1];
  center.z = input[element_x + 2];
  center.w = input[element_x + 3];

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrc*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        space_sqr = (i - element_y) * (i - element_y) +
                    (j - element_x) * (j - element_x);
        if (space_sqr <= radius_sqr) {
          space_sqr *= space_coeff;
          value.x = input[j];
          value.y = input[j + 1];
          value.z = input[j + 2];
          value.w = input[j + 3];
          weight.x = expf(space_sqr + (value.x - center.x) *
                          (value.x - center.x) * color_coeff);
          weight.y = expf(space_sqr + (value.y - center.y) *
                          (value.y - center.y) * color_coeff);
          weight.z = expf(space_sqr + (value.z - center.z) *
                          (value.z - center.z) * color_coeff);
          weight.w = expf(space_sqr + (value.w - center.w) *
                          (value.w - center.w) * color_coeff);
          sum.x += weight.x * value.x;
          sum.y += weight.y * value.y;
          sum.z += weight.z * value.z;
          sum.w += weight.w * value.w;
          weight_sum += weight;
        }
      }
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrc*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        space_sqr = (i - element_y) * (i - element_y) +
                    (j - element_x) * (j - element_x);
        if (space_sqr <= radius_sqr) {
          space_sqr *= space_coeff;
          data_index = interpolation(cols, radius, j);
          value.x = input[data_index];
          data_index = interpolation(cols, radius, j + 1);
          value.y = input[data_index];
          data_index = interpolation(cols, radius, j + 2);
          value.z = input[data_index];
          data_index = interpolation(cols, radius, j + 3);
          value.w = input[data_index];
          weight.x = expf(space_sqr + (value.x - center.x) *
                          (value.x - center.x) * color_coeff);
          weight.y = expf(space_sqr + (value.y - center.y) *
                          (value.y - center.y) * color_coeff);
          weight.z = expf(space_sqr + (value.z - center.z) *
                          (value.z - center.z) * color_coeff);
          weight.w = expf(space_sqr + (value.w - center.w) *
                          (value.w - center.w) * color_coeff);
          sum.x += weight.x * value.x;
          sum.y += weight.y * value.y;
          sum.z += weight.z * value.z;
          sum.w += weight.w * value.w;
          weight_sum += weight;
        }
      }
    }
  }
  sum /= weight_sum;

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  if (sizeof(Tsrc) == 1) {
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

template <typename Tsrc, typename Tsrcn, typename Tdst, typename Tdstn,
          typename BorderInterpolation>
__global__
void filter2DCnSharedKernel1(const Tsrc* src, int rows, int cols,
                             int src_stride, int radius, float radius_sqr,
                             float color_coeff, float space_coeff, Tdst* dst,
                             int dst_stride,
                             BorderInterpolation interpolation) {
  __shared__ Tsrcn data[kBlockDimY1][(kBlockDimX1 << 1)];

  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;

  int origin_y = element_y - radius;
  int top_y    = element_y + radius;
  int index, row_index, prev_index = -2;
  float space_sqr, color_tmp;
  Tsrcn* input = (Tsrcn*)((uchar*)src + element_y * src_stride);
  Tsrcn value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float weight_sum = 0.f;
  float weight;
  Tsrcn center = input[element_x];

  for (int i = origin_y; i <= top_y; i++) {
    if (element_y < rows) {
      row_index = interpolation(rows, radius, i);
      if (row_index != prev_index) {
        input = (Tsrcn*)((uchar*)src + row_index * src_stride);
        if (threadIdx.x < radius) {
          if (blockIdx.x == 0) {
            index = interpolation(cols, radius, element_x - radius);
          }
          else {
            index = element_x - radius;
          }
          value = input[index];
          data[threadIdx.y][threadIdx.x] = value;
        }

        if (element_x < cols) {
          value = input[element_x];
          data[threadIdx.y][radius + threadIdx.x] = value;
        }

        if (threadIdx.x < radius) {
          index = (cols - radius) >> kBlockShiftX1;
          if (blockIdx.x >= index) {
            if (blockIdx.x != gridDim.x - 1) {
              index = interpolation(cols, radius, element_x + kBlockDimX1);
              value = input[index];
              data[threadIdx.y][radius + kBlockDimX1 + threadIdx.x] = value;
            }
            else {
              index = interpolation(cols, radius, cols + threadIdx.x);
              value = input[index];
              index = cols - (blockIdx.x << kBlockShiftX1);
              data[threadIdx.y][radius + index + threadIdx.x] = value;
            }
          }
          else {
            index = element_x + kBlockDimX1;
            value = input[index];
            data[threadIdx.y][radius + kBlockDimX1 + threadIdx.x] = value;
          }
        }

        prev_index = row_index;
      }
    }
    __syncthreads();

    if (element_x < cols && element_y < rows) {
      for (index = 0; index < (radius << 1) + 1; index++) {
        space_sqr = (i - element_y) * (i - element_y) +
                    (index - radius) * (index - radius);
        if (space_sqr <= radius_sqr) {
          space_sqr *= space_coeff;
          value = data[threadIdx.y][threadIdx.x + index];
          color_tmp = fabsf(value.x - center.x) + fabsf(value.y - center.y) +
                      fabsf(value.z - center.z);
          weight = expf(space_sqr + color_tmp * color_tmp * color_coeff);
          sum.x += weight * value.x;
          sum.y += weight * value.y;
          sum.z += weight * value.z;
          weight_sum += weight;
        }
      }
    }
  }

  if (element_x < cols && element_y < rows) {
    sum /= weight_sum;

    Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<Tdstn, float4>(sum);
  }
}

template <typename Tsrc, typename Tsrcn, typename Tdst, typename Tdst4,
          typename BorderInterpolation>
__global__
void filter2DCnKernel(const Tsrc* src, int rows, int cols, int src_stride,
                      int radius, float radius_sqr, float color_coeff,
                      float space_coeff, Tdst* dst, int dst_stride,
                      BorderInterpolation interpolation) {
  int element_x, element_y;
  if (sizeof(Tsrc) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int data_index;
  float space_sqr, color_tmp;
  Tsrcn* input = (Tsrcn*)((uchar*)src + element_y * src_stride);
  Tsrcn value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float weight_sum = 0.f;
  float weight;
  Tsrcn center = input[element_x];

  bool isnt_border_block = true;
  if (sizeof(Tsrc) == 1) {
    data_index = radius >> kBlockShiftX0;
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> kBlockShiftX0;
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }
  else {
    data_index = radius >> kBlockShiftX1;
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> kBlockShiftX1;
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrcn*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        space_sqr = (i - element_y) * (i - element_y) +
                    (j - element_x) * (j - element_x);
        if (space_sqr <= radius_sqr) {
          space_sqr *= space_coeff;
          value = input[j];
          color_tmp = fabsf(value.x - center.x) + fabsf(value.y - center.y) +
                      fabsf(value.z - center.z);
          weight = expf(space_sqr + color_tmp * color_tmp * color_coeff);
          sum.x += weight * value.x;
          sum.y += weight * value.y;
          sum.z += weight * value.z;
          weight_sum += weight;
        }
      }
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrcn*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        space_sqr = (i - element_y) * (i - element_y) +
                    (j - element_x) * (j - element_x);
        if (space_sqr <= radius_sqr) {
          space_sqr *= space_coeff;
          data_index = interpolation(cols, radius, j);
          value = input[data_index];
          color_tmp = fabsf(value.x - center.x) + fabsf(value.y - center.y) +
                      fabsf(value.z - center.z);
          weight = expf(space_sqr + color_tmp * color_tmp * color_coeff);
          sum.x += weight * value.x;
          sum.y += weight * value.y;
          sum.z += weight * value.z;
          weight_sum += weight;
        }
      }
    }
  }
  sum /= weight_sum;

  Tdst4* output = (Tdst4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tdst4, float4>(sum);
}

#define RUN_CHANNEL1_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
filter2DC1SharedKernel<Tsrc, Tdst, Interpolation><<<grid, block, 0, stream>>>( \
    src, rows, cols, src_stride, radius, radius_sqr, color_coeff, space_coeff, \
    dst, dst_stride, interpolation);

#define RUN_CHANNELN_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
filter2DCnSharedKernel0<Tsrc, Tsrc ## 3, Tdst, Tdst ## 3, Interpolation>       \
    <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, radius,          \
    radius_sqr, color_coeff, space_coeff, dst, dst_stride, interpolation);

#define RUN_KERNELS(T, grid_x, Interpolation)                                  \
Interpolation interpolation;                                                   \
if (channels == 1) {                                                           \
  grid0.x = grid_x;                                                            \
  filter2DC1Kernel<T, T ## 4, T, T ## 4, Interpolation><<<grid0, block0, 0,    \
      stream>>>(src, rows, cols, src_stride, radius, radius_sqr, color_coeff,  \
      space_coeff, dst, dst_stride, interpolation);                            \
}                                                                              \
else if (channels == 3) {                                                      \
  if (diameter <= 33) {                                                        \
    filter2DCnSharedKernel1<T, T ## 3, T, T ## 3, Interpolation><<<grid1,      \
        block1, 0, stream>>>(src, rows, cols, src_stride, radius, radius_sqr,  \
        color_coeff, space_coeff, dst, dst_stride, interpolation);             \
  }                                                                            \
  else {                                                                       \
    filter2DCnKernel<T, T ## 3, T, T ## 3, Interpolation><<<grid0, block0, 0,  \
        stream>>>(src, rows, cols, src_stride, radius, radius_sqr, color_coeff,\
        space_coeff, dst, dst_stride, interpolation);                          \
  }                                                                            \
}                                                                              \
else {                                                                         \
}

RetCode bilateralFilter(const uchar* src, int rows, int cols, int channels,
                        int src_stride, int diameter, float sigma_color,
                        float sigma_space, uchar* dst, int dst_stride,
                        BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(!(diameter <= 0 && sigma_space <= 1));
  PPL_ASSERT(channels == 1 || channels == 3);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  if (sigma_color <= 0) sigma_color = 1;
  if (sigma_space <= 0) sigma_space = 1;
  float color_coeff = -0.5f / (sigma_color * sigma_color);
  float space_coeff = -0.5f / (sigma_space * sigma_space);
  int radius;
  if (diameter <= 0) {
    radius = rint(sigma_space * 1.5);
  }
  else {
    radius = diameter >> 1;
  }
  radius = radius > 1 ? radius : 1;
  diameter = (radius << 1) + 1;
  float radius_sqr = radius * radius;

  cudaError_t code;
  if (diameter <= SMALL_KSIZE0 && channels == 1) {
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

  if (diameter <= SMALL_KSIZE1 && (channels == 3)) {
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

  dim3 block0, grid0;
  block0.x = kBlockDimX0;
  block0.y = kBlockDimY0;
  grid0.x  = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid0.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  dim3 block1, grid1;
  block1.x = kBlockDimX1;
  block1.y = kBlockDimY1;
  grid1.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid1.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  int grid_x = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);

  if (border_type == BORDER_REPLICATE) {
    RUN_KERNELS(uchar, grid_x, ReplicateBorder);
  }
  else if (border_type == BORDER_REFLECT) {
    RUN_KERNELS(uchar, grid_x, ReflectBorder);
  }
  else {
    RUN_KERNELS(uchar, grid_x, Reflect101Border);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode bilateralFilter(const float* src, int rows, int cols, int channels,
                        int src_stride, int diameter, float sigma_color,
                        float sigma_space, float* dst, int dst_stride,
                        BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(!(diameter <= 0 && sigma_space <= 1));
  PPL_ASSERT(channels == 1 || channels == 3);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  if (sigma_color <= 0) sigma_color = 1;
  if (sigma_space <= 0) sigma_space = 1;
  float color_coeff = -0.5 / (sigma_color * sigma_color);
  float space_coeff = -0.5 / (sigma_space * sigma_space);
  int radius;
  if (diameter <= 0) {
    radius = rint(sigma_space * 1.5);
  }
  else {
    radius = diameter >> 1;
  }
  radius = radius > 1 ? radius : 1;
  diameter = (radius << 1) + 1;
  float radius_sqr = radius * radius;

  cudaError_t code;
  if (diameter <= SMALL_KSIZE0 && channels == 1) {
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

  if (diameter <= SMALL_KSIZE1 && (channels == 3)) {
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

  dim3 block0, grid0;
  block0.x = kBlockDimX1;
  block0.y = kBlockDimY1;
  grid0.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid0.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block1 = block0, grid1 = grid0;

  int grid_x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);

  if (border_type == BORDER_REPLICATE) {
    RUN_KERNELS(float, grid_x, ReplicateBorder);
  }
  else if (border_type == BORDER_REFLECT) {
    RUN_KERNELS(float, grid_x, ReflectBorder);
  }
  else {
    RUN_KERNELS(float, grid_x, Reflect101Border);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode BilateralFilter<uchar, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int diameter,
                                  float sigma_color,
                                  float sigma_space,
                                  int outWidthStride,
                                  uchar* outData,
                                  BorderType border_type) {
  RetCode code = bilateralFilter(inData, height, width, 1, inWidthStride,
                                 diameter, sigma_color, sigma_space, outData,
                                 outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode BilateralFilter<uchar, 3>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int diameter,
                                  float sigma_color,
                                  float sigma_space,
                                  int outWidthStride,
                                  uchar* outData,
                                  BorderType border_type) {
  RetCode code = bilateralFilter(inData, height, width, 3, inWidthStride,
                                 diameter, sigma_color, sigma_space, outData,
                                 outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode BilateralFilter<float, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int diameter,
                                  float sigma_color,
                                  float sigma_space,
                                  int outWidthStride,
                                  float* outData,
                                  BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = bilateralFilter(inData, height, width, 1, inWidthStride,
                                 diameter, sigma_color, sigma_space, outData,
                                 outWidthStride, border_type, stream);

  return code;
}

template <>
RetCode BilateralFilter<float, 3>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int diameter,
                                  float sigma_color,
                                  float sigma_space,
                                  int outWidthStride,
                                  float* outData,
                                  BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = bilateralFilter(inData, height, width, 3, inWidthStride,
                                 diameter, sigma_color, sigma_space, outData,
                                 outWidthStride, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
