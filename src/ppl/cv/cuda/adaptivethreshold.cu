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

#include "ppl/cv/cuda/adaptivethreshold.h"
#include "ppl/cv/cuda/setvalue.h"

#include <cmath>

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define SMALL_SIZE 7
#define SMALL_MAX_KSIZE 32
#define LARGE_MAX_KSIZE 256

__DEVICE__
void createGaussianKernel(float* coefficients, float sigma, int ksize) {
  bool fixed_kernel = false;
  if ((ksize & 1) == 1 && ksize <= SMALL_SIZE && sigma <= 0) {
    if (ksize == 1) {
      coefficients[0] = 1.f;
    }
    else if (ksize == 3) {
      coefficients[0] = 0.25f;
      coefficients[1] = 0.5f;
      coefficients[2] = 0.25f;
    }
    else if (ksize == 5) {
      coefficients[0] = 0.0625f;
      coefficients[1] = 0.25f;
      coefficients[2] = 0.375f;
      coefficients[3] = 0.25f;
      coefficients[4] = 0.0625f;
    }
    else {
      coefficients[0] = 0.03125f;
      coefficients[1] = 0.109375f;
      coefficients[2] = 0.21875f;
      coefficients[3] = 0.28125f;
      coefficients[4] = 0.21875f;
      coefficients[5] = 0.109375f;
      coefficients[6] = 0.03125f;
    }
    fixed_kernel = true;
  }

  float value = sigma > 0 ? sigma : ((ksize - 1) * 0.5f - 1) * 0.3f + 0.8f;
  float scale_2x = -0.5f / (value * value);
  float sum = 0.f;

  int i;
  float x;
  for (i = 0; i < ksize; i++) {
    x = i - (ksize - 1) * 0.5f;
    value = fixed_kernel ? coefficients[i] : std::exp(scale_2x * x * x);
    if (!fixed_kernel) {
      coefficients[i] = value;
    }
    sum += value;
  }

  sum = 1.f / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] *= sum;
  }
}

template <typename BorderInterpolation>
__global__
void rowColC1Kernel0(const uchar* src, int rows, int cols, int src_stride,
                     int radius, float weight, int threshold_type,
                     uchar setted_value, int delta, uchar* dst, int dst_stride,
                     BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 * 3][(kDimX0 << 2)];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int bottom = element_x - radius;
  int top    = element_x + radius;

  int data_index, row_index;
  uchar* input;
  float4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> (kShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (threadIdx.y < radius && element_x < cols) {
    row_index = interpolation(rows, radius, element_y - radius);
    input = (uchar*)src + row_index * src_stride;
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
        data_index = interpolation(cols, radius, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius, i + 3);
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
    input = (uchar*)src + element_y * src_stride;

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
        data_index = interpolation(cols, radius, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }
    data_index = threadIdx.x << 2;
    data[radius + threadIdx.y][data_index] = sum.x;
    data[radius + threadIdx.y][data_index + 1] = sum.y;
    data[radius + threadIdx.y][data_index + 2] = sum.z;
    data[radius + threadIdx.y][data_index + 3] = sum.w;
  }

  if (threadIdx.y < radius && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.y != gridDim.y - 1) {
      row_index = interpolation(rows, radius,
                                ((blockIdx.y + 1) << kShiftY0) + threadIdx.y);
    }
    else {
      row_index = interpolation(rows, radius, rows + threadIdx.y);
    }
    input = (uchar*)src + row_index * src_stride;

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
        data_index = interpolation(cols, radius, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }

    data_index = threadIdx.x << 2;
    if (blockIdx.y != gridDim.y - 1) {
      row_index = radius + kDimY0 + threadIdx.y;
    }
    else {
      row_index = radius + (rows - (blockIdx.y << kShiftY0)) + threadIdx.y;
    }
    data[row_index][data_index] = sum.x;
    data[row_index][data_index + 1] = sum.y;
    data[row_index][data_index + 2] = sum.z;
    data[row_index][data_index + 3] = sum.w;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    top = (radius << 1) + 1;
    sum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int i = 0; i < top; i++) {
      data_index = threadIdx.x << 2;
      value.x = data[i + threadIdx.y][data_index];
      value.y = data[i + threadIdx.y][data_index + 1];
      value.z = data[i + threadIdx.y][data_index + 2];
      value.w = data[i + threadIdx.y][data_index + 3];
      sum += value;
    }

    sum.x *= weight;
    sum.y *= weight;
    sum.z *= weight;
    sum.w *= weight;

    int4 threshold;
    threshold.x = saturateCast(sum.x) - delta;
    threshold.y = saturateCast(sum.y) - delta;
    threshold.z = saturateCast(sum.z) - delta;
    threshold.w = saturateCast(sum.w) - delta;

    input = (uchar*)src + element_y * src_stride;
    value.x = input[element_x];
    value.y = input[element_x + 1];
    value.z = input[element_x + 2];
    value.w = input[element_x + 3];

    if (threshold_type == THRESH_BINARY) {
      value.x = value.x > threshold.x ? setted_value : 0;
      value.y = value.y > threshold.y ? setted_value : 0;
      value.z = value.z > threshold.z ? setted_value : 0;
      value.w = value.w > threshold.w ? setted_value : 0;
    }
    else {
      value.x = value.x > threshold.x ? 0 : setted_value;
      value.y = value.y > threshold.y ? 0 : setted_value;
      value.z = value.z > threshold.z ? 0 : setted_value;
      value.w = value.w > threshold.w ? 0 : setted_value;
    }

    uchar* output = dst + element_y * dst_stride;
    if (element_x < cols - 3) {
      output[element_x]     = saturateCast(value.x);
      output[element_x + 1] = saturateCast(value.y);
      output[element_x + 2] = saturateCast(value.z);
      output[element_x + 3] = saturateCast(value.w);
    }
    else {
      output[element_x] = saturateCast(value.x);
      if (element_x < cols - 1) {
        output[element_x + 1] = saturateCast(value.y);
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = saturateCast(value.z);
      }
    }
  }
}

template <typename BorderInterpolation>
__global__
void rowColC1Kernel1(const uchar* src, int rows, int cols, int src_stride,
                     int ksize, int threshold_type, uchar setted_value,
                     int delta, uchar* dst, int dst_stride,
                     BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 * 3][(kDimX0 << 2)];
  __shared__ float kernel[SMALL_MAX_KSIZE];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    createGaussianKernel(kernel, 0, ksize);
  }
  __syncthreads();

  int radius = ksize >> 1;
  int bottom = element_x - radius;
  int top    = element_x + radius;

  int data_index, row_index, kernel_index = 0;
  uchar* input;
  float4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> (kShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (threadIdx.y < radius && element_x < cols) {
    row_index = interpolation(rows, radius, element_y - radius);
    input = (uchar*)src + row_index * src_stride;
    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius, i + 3);
        value.w = input[data_index];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
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
    input = (uchar*)src + element_y * src_stride;
    kernel_index = 0;

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius, i + 3);
        value.w = input[data_index];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
    data_index = threadIdx.x << 2;
    data[radius + threadIdx.y][data_index] = sum.x;
    data[radius + threadIdx.y][data_index + 1] = sum.y;
    data[radius + threadIdx.y][data_index + 2] = sum.z;
    data[radius + threadIdx.y][data_index + 3] = sum.w;
  }

  if (threadIdx.y < radius && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.y != gridDim.y - 1) {
      row_index = interpolation(rows, radius,
                                ((blockIdx.y + 1) << kShiftY0) + threadIdx.y);
    }
    else {
      row_index = interpolation(rows, radius, rows + threadIdx.y);
    }
    input = (uchar*)src + row_index * src_stride;
    kernel_index = 0;

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius, i + 3);
        value.w = input[data_index];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }

    data_index = threadIdx.x << 2;
    if (blockIdx.y != gridDim.y - 1) {
      row_index = radius + kDimY0 + threadIdx.y;
    }
    else {
      row_index = radius + (rows - (blockIdx.y << kShiftY0)) + threadIdx.y;
    }
    data[row_index][data_index] = sum.x;
    data[row_index][data_index + 1] = sum.y;
    data[row_index][data_index + 2] = sum.z;
    data[row_index][data_index + 3] = sum.w;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    top = (radius << 1) + 1;
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    kernel_index = 0;

    for (int i = 0; i < top; i++) {
      data_index = threadIdx.x << 2;
      value.x = data[i + threadIdx.y][data_index];
      value.y = data[i + threadIdx.y][data_index + 1];
      value.z = data[i + threadIdx.y][data_index + 2];
      value.w = data[i + threadIdx.y][data_index + 3];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
    }

    int4 threshold;
    threshold.x = saturateCast(sum.x) - delta;
    threshold.y = saturateCast(sum.y) - delta;
    threshold.z = saturateCast(sum.z) - delta;
    threshold.w = saturateCast(sum.w) - delta;

    input = (uchar*)src + element_y * src_stride;
    value.x = input[element_x];
    value.y = input[element_x + 1];
    value.z = input[element_x + 2];
    value.w = input[element_x + 3];

    if (threshold_type == THRESH_BINARY) {
      value.x = value.x > threshold.x ? setted_value : 0;
      value.y = value.y > threshold.y ? setted_value : 0;
      value.z = value.z > threshold.z ? setted_value : 0;
      value.w = value.w > threshold.w ? setted_value : 0;
    }
    else {
      value.x = value.x > threshold.x ? 0 : setted_value;
      value.y = value.y > threshold.y ? 0 : setted_value;
      value.z = value.z > threshold.z ? 0 : setted_value;
      value.w = value.w > threshold.w ? 0 : setted_value;
    }

    uchar* output = dst + element_y * dst_stride;
    if (element_x < cols - 3) {
      output[element_x]     = saturateCast(value.x);
      output[element_x + 1] = saturateCast(value.y);
      output[element_x + 2] = saturateCast(value.z);
      output[element_x + 3] = saturateCast(value.w);
    }
    else {
      output[element_x] = saturateCast(value.x);
      if (element_x < cols - 1) {
        output[element_x + 1] = saturateCast(value.y);
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = saturateCast(value.z);
      }
    }
  }
}

template <typename BorderInterpolation>
__global__
void rowBatch4Kernel0(const uchar* src, int rows, int cols, int src_stride,
                      int radius, float* dst, int dst_stride,
                      BorderInterpolation interpolation) {
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int top_x    = element_x + radius;

  int data_index;
  uchar* input;
  uchar4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> (kBlockShiftX1 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kBlockShiftX1 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (uchar*)src + element_y * src_stride;
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
      data_index = interpolation(cols, radius, i);
      value.x = input[data_index];
      data_index = interpolation(cols, radius, i + 1);
      value.y = input[data_index];
      data_index = interpolation(cols, radius, i + 2);
      value.z = input[data_index];
      data_index = interpolation(cols, radius, i + 3);
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

template <typename BorderInterpolation>
__global__
void rowBatch4Kernel1(const uchar* src, int rows, int cols, int src_stride,
                      int ksize, float* dst, int dst_stride,
                      BorderInterpolation interpolation) {
  __shared__ float kernel[LARGE_MAX_KSIZE];

  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    createGaussianKernel(kernel, 0, ksize);
  }
  __syncthreads();

  int radius = ksize >> 1;
  int origin_x = element_x - radius;
  int top_x    = element_x + radius;

  int data_index, kernel_index = 0;
  uchar* input;
  uchar4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> (kBlockShiftX1 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kBlockShiftX1 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (uchar*)src + element_y * src_stride;
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value.x = input[i];
      value.y = input[i + 1];
      value.z = input[i + 2];
      value.w = input[i + 3];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius, i);
      value.x = input[data_index];
      data_index = interpolation(cols, radius, i + 1);
      value.y = input[data_index];
      data_index = interpolation(cols, radius, i + 2);
      value.z = input[data_index];
      data_index = interpolation(cols, radius, i + 3);
      value.w = input[data_index];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
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

template <typename BorderInterpolation>
__global__
void colBatch4Kernel0(const float* buffer, int rows, int cols,
                      int buffer_stride, const uchar* src, int src_stride,
                      int radius, float weight, int threshold_type,
                      uchar setted_value, int delta, uchar* dst, int dst_stride,
                      BorderInterpolation interpolation) {
  __shared__ uchar data[kBlockDimY1][kBlockDimX1 << 2];

  int element_x = (blockIdx.x << (kBlockShiftX1 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_y = element_y - radius;
  int top_y    = element_y + radius;

  int data_index;
  float* input0;
  float value;
  float sum = 0.f;

  bool isnt_border_block = true;
  data_index = radius >> kBlockShiftY1;
  if (blockIdx.y <= data_index) isnt_border_block = false;
  data_index = (rows - radius) >> kBlockShiftY1;
  if (blockIdx.y >= data_index) isnt_border_block = false;

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      input0 = (float*)((uchar*)buffer + i * buffer_stride);
      value = input0[element_x];
      sum += value;
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input0 = (float*)((uchar*)buffer + data_index * buffer_stride);
      value = input0[element_x];
      sum += value;
    }
  }

  sum *= weight;
  data[threadIdx.y][threadIdx.x] = saturateCast(sum);
  __syncthreads();

  if (threadIdx.x < kBlockDimX1) {
    int4 value;
    uchar* input1 = (uchar*)src + element_y * src_stride;
    element_x = (blockIdx.x << (kBlockShiftX1 + 2)) + (threadIdx.x << 2);
    value.x = input1[element_x];
    value.y = input1[element_x + 1];
    value.z = input1[element_x + 2];
    value.w = input1[element_x + 3];

    int4 threshold;
    threshold.x = data[threadIdx.y][(threadIdx.x << 2)] - delta;
    threshold.y = data[threadIdx.y][(threadIdx.x << 2) + 1] - delta;
    threshold.z = data[threadIdx.y][(threadIdx.x << 2) + 2] - delta;
    threshold.w = data[threadIdx.y][(threadIdx.x << 2) + 3] - delta;

    if (threshold_type == THRESH_BINARY) {
      value.x = value.x > threshold.x ? setted_value : 0;
      value.y = value.y > threshold.y ? setted_value : 0;
      value.z = value.z > threshold.z ? setted_value : 0;
      value.w = value.w > threshold.w ? setted_value : 0;
    }
    else {
      value.x = value.x > threshold.x ? 0 : setted_value;
      value.y = value.y > threshold.y ? 0 : setted_value;
      value.z = value.z > threshold.z ? 0 : setted_value;
      value.w = value.w > threshold.w ? 0 : setted_value;
    }

    uchar* output = (uchar*)dst + element_y * dst_stride;
    element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
    data_index = threadIdx.x << 2;
    if (element_x < cols - 3) {
      output[element_x]     = clip(value.x, 0, 255);
      output[element_x + 1] = clip(value.y, 0, 255);
      output[element_x + 2] = clip(value.z, 0, 255);
      output[element_x + 3] = clip(value.w, 0, 255);
    }
    else if (element_x < cols) {
      output[element_x] = clip(value.x, 0, 255);
      if (element_x < cols - 1) {
        output[element_x + 1] = clip(value.y, 0, 255);
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = clip(value.z, 0, 255);
      }
    }
    else {
    }
  }
}

template <typename BorderInterpolation>
__global__
void colBatch4Kernel1(const float* buffer, int rows, int cols,
                      int buffer_stride, const uchar* src, int src_stride,
                      int ksize, int threshold_type, uchar setted_value,
                      int delta, uchar* dst, int dst_stride,
                      BorderInterpolation interpolation) {
  __shared__ uchar data[kBlockDimY1][kBlockDimX1 << 2];
  __shared__ float kernel[LARGE_MAX_KSIZE];

  int element_x = (blockIdx.x << (kBlockShiftX1 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    createGaussianKernel(kernel, 0, ksize);
  }
  __syncthreads();

  int radius = ksize >> 1;
  int origin_y = element_y - radius;
  int top_y    = element_y + radius;

  int data_index, kernel_index = 0;
  float* input0;
  float value;
  float sum = 0.f;

  bool isnt_border_block = true;
  data_index = radius >> kBlockShiftY1;
  if (blockIdx.y <= data_index) isnt_border_block = false;
  data_index = (rows - radius) >> kBlockShiftY1;
  if (blockIdx.y >= data_index) isnt_border_block = false;

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      input0 = (float*)((uchar*)buffer + i * buffer_stride);
      value = input0[element_x];
      sum += value * kernel[kernel_index];
      kernel_index++;
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input0 = (float*)((uchar*)buffer + data_index * buffer_stride);
      value = input0[element_x];
      sum += value * kernel[kernel_index];
      kernel_index++;
    }
  }

  data[threadIdx.y][threadIdx.x] = saturateCast(sum);
  __syncthreads();

  if (threadIdx.x < kBlockDimX1) {
    uchar4 value;
    uchar* input1 = (uchar*)src + element_y * src_stride;
    element_x = (blockIdx.x << (kBlockShiftX1 + 2)) + (threadIdx.x << 2);
    value.x = input1[element_x];
    value.y = input1[element_x + 1];
    value.z = input1[element_x + 2];
    value.w = input1[element_x + 3];

    int4 threshold;
    threshold.x = data[threadIdx.y][(threadIdx.x << 2)] - delta;
    threshold.y = data[threadIdx.y][(threadIdx.x << 2) + 1] - delta;
    threshold.z = data[threadIdx.y][(threadIdx.x << 2) + 2] - delta;
    threshold.w = data[threadIdx.y][(threadIdx.x << 2) + 3] - delta;

    if (threshold_type == THRESH_BINARY) {
      value.x = value.x > threshold.x ? setted_value : 0;
      value.y = value.y > threshold.y ? setted_value : 0;
      value.z = value.z > threshold.z ? setted_value : 0;
      value.w = value.w > threshold.w ? setted_value : 0;
    }
    else {
      value.x = value.x > threshold.x ? 0 : setted_value;
      value.y = value.y > threshold.y ? 0 : setted_value;
      value.z = value.z > threshold.z ? 0 : setted_value;
      value.w = value.w > threshold.w ? 0 : setted_value;
    }

    uchar* output = (uchar*)dst + element_y * dst_stride;
    element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
    data_index = threadIdx.x << 2;
    if (element_x < cols - 3) {
      output[element_x]     = value.x;
      output[element_x + 1] = value.y;
      output[element_x + 2] = value.z;
      output[element_x + 3] = value.w;
    }
    else if (element_x < cols) {
      output[element_x] = value.x;
      if (element_x < cols - 1) {
        output[element_x + 1] = value.y;
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = value.z;
      }
    }
    else {
    }
  }
}

#define RUN_SMALL_KERNELS(Interpolation)                                       \
Interpolation interpolation;                                                   \
if (adaptive_method == ADAPTIVE_THRESH_MEAN_C) {                               \
  rowColC1Kernel0<Interpolation><<<grid, block, 0, stream>>>(src, rows, cols,  \
      src_stride, radius, weight, threshold_type, setted_value, int_delta, dst,\
      dst_stride, interpolation);                                              \
}                                                                              \
else {                                                                         \
  rowColC1Kernel1<Interpolation><<<grid, block, 0, stream>>>(src, rows, cols,  \
      src_stride, ksize, threshold_type, setted_value, int_delta, dst,         \
      dst_stride, interpolation);                                              \
}

#define RUN_LARGE_KERNELS0(Interpolation)                                      \
Interpolation interpolation;                                                   \
rowBatch4Kernel0<Interpolation><<<grid1, block1, 0, stream>>>(src, rows, cols, \
    src_stride, radius, buffer, pitch, interpolation);                         \
colBatch4Kernel0<Interpolation><<<grid2, block2, 0, stream>>>(buffer, rows,    \
    cols, pitch, src, src_stride, radius, weight, threshold_type, setted_value,\
    int_delta, dst, dst_stride, interpolation);

#define RUN_LARGE_KERNELS1(Interpolation)                                      \
Interpolation interpolation;                                                   \
rowBatch4Kernel1<Interpolation><<<grid1, block1, 0, stream>>>(src, rows, cols, \
    src_stride, ksize, buffer, pitch, interpolation);                          \
colBatch4Kernel1<Interpolation><<<grid2, block2, 0, stream>>>(buffer, rows,    \
    cols, pitch, src, src_stride, ksize, threshold_type, setted_value,         \
    int_delta, dst, dst_stride, interpolation);

RetCode
AdaptiveThreshold(cudaStream_t stream, int rows, int cols, int src_stride,
                  const uchar* src, int dst_stride, uchar* dst, float max_value,
                  int adaptive_method, int threshold_type, int ksize,
                  float delta, BorderType border_type) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(max_value != 0);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(adaptive_method == ADAPTIVE_THRESH_MEAN_C ||
             adaptive_method == ADAPTIVE_THRESH_GAUSSIAN_C);
  PPL_ASSERT(threshold_type == THRESH_BINARY ||
             threshold_type == THRESH_BINARY_INV);
  PPL_ASSERT((ksize & 1) == 1 && ksize > 1 && ksize < LARGE_MAX_KSIZE);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  uchar setted_value = 0;
  if (max_value < 0) {
    Zeros<uchar, 1>(stream, rows, cols, dst_stride, dst);
    return RC_SUCCESS;
  }
  else if (max_value < 255.f) {
    setted_value = rintf(max_value);
  }
  else {
    setted_value = 255;
  }

  int int_delta = 0;
  if (threshold_type == THRESH_BINARY) {
    int_delta = std::ceil(delta);
  }
  else {
    int_delta = std::floor(delta);
  }

  int radius = ksize >> 1;
  float weight = 1.f / (ksize * ksize);

  cudaError_t code;
  if (ksize < SMALL_MAX_KSIZE) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_SMALL_KERNELS(ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_SMALL_KERNELS(ReflectBorder);
    }
    else {
      RUN_SMALL_KERNELS(Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  dim3 block1, grid1;
  block1.x = kBlockDimX1;
  block1.y = kBlockDimY1;
  grid1.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid1.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block2, grid2;
  block2.x = (kBlockDimX1 << 2);
  block2.y = kBlockDimY1;
  grid2.x  = divideUp(cols, (kBlockDimX1 << 2), (kBlockShiftX1 + 2));
  grid2.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  float* buffer;
  size_t pitch;

  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * sizeof(float), rows, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch  = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  if (adaptive_method == ADAPTIVE_THRESH_MEAN_C) {
    if (border_type == BORDER_REPLICATE) {
      RUN_LARGE_KERNELS0(ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_LARGE_KERNELS0(ReflectBorder);
    }
    else {
      RUN_LARGE_KERNELS0(Reflect101Border);
    }
  }
  else {  // adaptive_method == ADAPTIVE_THRESH_GAUSSIAN_C
    if (border_type == BORDER_REPLICATE) {
      RUN_LARGE_KERNELS1(ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_LARGE_KERNELS1(ReflectBorder);
    }
    else {
      RUN_LARGE_KERNELS1(Reflect101Border);
    }
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

}  // cuda
}  // cv
}  // ppl
