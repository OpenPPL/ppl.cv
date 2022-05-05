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

#include "ppl/cv/cuda/medianblur.h"

#include <cfloat>

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define RADIUS0 16
#define SMALL_KSIZE0 RADIUS0 * 2 + 1

#define RADIUS1 8
#define SMALL_KSIZE1 RADIUS1 * 2 + 1

#define HIST_SIZE (16 + 256)

#define VERTICAL_ELEMENTS 256
#define VERTICAL_SHIFT 8

#define BLOCK_WIDTH 8
#define BLOCK_SHIFT 3

template <typename BorderInterpolation>
__global__
void medianC1SharedKernel(const uchar* src, int rows, int cols, int src_stride,
                          int median_index, int radius, uchar* dst,
                          int dst_stride, BorderInterpolation interpolation) {
  __shared__ uchar data[kDimY0 + RADIUS0 * 2][(kDimX0 << 2) + RADIUS0 * 2];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  {
    int index, y_index, row_index;
    uchar* input;
    uchar value0, value1, value2, value3;

    y_index   = threadIdx.y;
    row_index = element_y - radius;
    while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
           row_index < rows + radius) {
      index = interpolation(rows, radius, row_index);
      input = (uchar*)((uchar*)src + index * src_stride);

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
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int ksize = (radius << 1) + 1;
  int threadIdx_x = threadIdx.x << 2;
  bool unchecked0, unchecked1, unchecked2, unchecked3;
  uint4 local_count  = make_uint4(0, 0, 0, 0);
  uint4 global_count = make_uint4(0, 0, 0, 0);
  uchar4 value;
  short4 max;
  short4 top = make_short4(256, 256, 256, 256);

  for (int index = 0; index <= median_index; index++) {
    max = make_short4(-1, -1, -1, -1);
    unchecked0 = true;
    unchecked1 = true;
    unchecked2 = true;
    unchecked3 = true;
    for (int i = 0; i < ksize; i++) {
      for (int j = 0; j < ksize; j++) {
        value.x = data[threadIdx.y + i][threadIdx_x + j];
        value.y = data[threadIdx.y + i][threadIdx_x + j + 1];
        value.z = data[threadIdx.y + i][threadIdx_x + j + 2];
        value.w = data[threadIdx.y + i][threadIdx_x + j + 3];
        if ((!unchecked0) && max.x == value.x) unchecked0 = false;
        if ((!unchecked1) && max.y == value.y) unchecked1 = false;
        if ((!unchecked2) && max.z == value.z) unchecked2 = false;
        if ((!unchecked3) && max.w == value.w) unchecked3 = false;

        if (unchecked0 && max.x == value.x) local_count.x++;
        if (unchecked1 && max.y == value.y) local_count.y++;
        if (unchecked2 && max.z == value.z) local_count.z++;
        if (unchecked3 && max.w == value.w) local_count.w++;

        if (index + global_count.x <= median_index && max.x < value.x &&
            value.x < top.x) {
          max.x = value.x;
          local_count.x = 0;
        }
        if (index + global_count.y <= median_index && max.y < value.y &&
            value.y < top.y) {
          max.y = value.y;
          local_count.y = 0;
        }
        if (index + global_count.z <= median_index && max.z < value.z &&
            value.z < top.z) {
          max.z = value.z;
          local_count.z = 0;
        }
        if (index + global_count.w <= median_index && max.w < value.w &&
            value.w < top.w) {
          max.w = value.w;
          local_count.w = 0;
        }
      }
    }
    global_count.x += local_count.x;
    global_count.y += local_count.y;
    global_count.z += local_count.z;
    global_count.w += local_count.w;
    if (max.x != -1) top.x = max.x;
    if (max.y != -1) top.y = max.y;
    if (max.z != -1) top.z = max.z;
    if (max.w != -1) top.w = max.w;
  }

  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 3) {
    output[element_x]     = saturateCast(top.x);
    output[element_x + 1] = saturateCast(top.y);
    output[element_x + 2] = saturateCast(top.z);
    output[element_x + 3] = saturateCast(top.w);
  }
  else {
    output[element_x] = saturateCast(top.x);
    if (element_x < cols - 1) {
      output[element_x + 1] = saturateCast(top.y);
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = saturateCast(top.z);
    }
  }
}

template <typename BorderInterpolation>
__global__
void medianC1SharedKernel(const float* src, int rows, int cols, int src_stride,
                          int median_index, int radius, float* dst,
                          int dst_stride, BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 + RADIUS0 * 2][(kDimX0 << 2) + RADIUS0 * 2];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  {
    int index, y_index, row_index;
    float* input;
    float value0, value1, value2, value3;

    y_index   = threadIdx.y;
    row_index = element_y - radius;
    while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
           row_index < rows + radius) {
      index = interpolation(rows, radius, row_index);
      input = (float*)((uchar*)src + index * src_stride);

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
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int ksize = (radius << 1) + 1;
  int threadIdx_x = threadIdx.x << 2;
  bool unchecked0, unchecked1, unchecked2, unchecked3;
  uint4 local_count  = make_uint4(0, 0, 0, 0);
  uint4 global_count = make_uint4(0, 0, 0, 0);
  float4 value;
  float4 max;
  float4 top = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  for (int index = 0; index <= median_index; index++) {
    max = make_float4(FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);
    unchecked0 = true;
    unchecked1 = true;
    unchecked2 = true;
    unchecked3 = true;
    for (int i = 0; i < ksize; i++) {
      for (int j = 0; j < ksize; j++) {
        value.x = data[threadIdx.y + i][threadIdx_x + j];
        value.y = data[threadIdx.y + i][threadIdx_x + j + 1];
        value.z = data[threadIdx.y + i][threadIdx_x + j + 2];
        value.w = data[threadIdx.y + i][threadIdx_x + j + 3];
        if ((!unchecked0) && max.x == value.x) unchecked0 = false;
        if ((!unchecked1) && max.y == value.y) unchecked1 = false;
        if ((!unchecked2) && max.z == value.z) unchecked2 = false;
        if ((!unchecked3) && max.w == value.w) unchecked3 = false;

        if (unchecked0 && max.x == value.x) local_count.x++;
        if (unchecked1 && max.y == value.y) local_count.y++;
        if (unchecked2 && max.z == value.z) local_count.z++;
        if (unchecked3 && max.w == value.w) local_count.w++;

        if (index + global_count.x <= median_index && max.x < value.x &&
            value.x < top.x) {
          max.x = value.x;
          local_count.x = 0;
        }
        if (index + global_count.y <= median_index && max.y < value.y &&
            value.y < top.y) {
          max.y = value.y;
          local_count.y = 0;
        }
        if (index + global_count.z <= median_index && max.z < value.z &&
            value.z < top.z) {
          max.z = value.z;
          local_count.z = 0;
        }
        if (index + global_count.w <= median_index && max.w < value.w &&
            value.w < top.w) {
          max.w = value.w;
          local_count.w = 0;
        }
      }
    }
    global_count.x += local_count.x;
    global_count.y += local_count.y;
    global_count.z += local_count.z;
    global_count.w += local_count.w;
    if (max.x != FLT_MIN) top.x = max.x;
    if (max.y != FLT_MIN) top.y = max.y;
    if (max.z != FLT_MIN) top.z = max.z;
    if (max.w != FLT_MIN) top.w = max.w;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 3) {
    output[element_x]     = top.x;
    output[element_x + 1] = top.y;
    output[element_x + 2] = top.z;
    output[element_x + 3] = top.w;
  }
  else {
    output[element_x] = top.x;
    if (element_x < cols - 1) {
      output[element_x + 1] = top.y;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = top.z;
    }
  }
}

template <typename BorderInterpolation>
__global__
void medianC3SharedKernel(const uchar* src, int rows, int cols, int src_stride,
                          int median_index, int radius, uchar* dst,
                          int dst_stride, BorderInterpolation interpolation) {
  __shared__ uchar3 data[kDimY0 + RADIUS1 * 2][kDimX0 + RADIUS1 * 2];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  {
    int index, y_index, row_index, col_index;
    uchar3* input;

    y_index   = threadIdx.y;
    row_index = element_y - radius;
    while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
           row_index < rows + radius) {
      index = interpolation(rows, radius, row_index);
      input = (uchar3*)((uchar*)src + index * src_stride);

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
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int ksize = (radius << 1) + 1;
  bool unchecked0, unchecked1, unchecked2;
  uint3 local_count  = make_uint3(0, 0, 0);
  uint3 global_count = make_uint3(0, 0, 0);
  uchar3 value;
  short3 max;
  short3 top = make_short3(256, 256, 256);

  for (int index = 0; index <= median_index; index++) {
    max = make_short3(-1, -1, -1);
    unchecked0 = true;
    unchecked1 = true;
    unchecked2 = true;
    for (int i = 0; i < ksize; i++) {
      for (int j = 0; j < ksize; j++) {
        value = data[threadIdx.y + i][threadIdx.x + j];
        if ((!unchecked0) && max.x == value.x) unchecked0 = false;
        if ((!unchecked1) && max.y == value.y) unchecked1 = false;
        if ((!unchecked2) && max.z == value.z) unchecked2 = false;

        if (unchecked0 && max.x == value.x) local_count.x++;
        if (unchecked1 && max.y == value.y) local_count.y++;
        if (unchecked2 && max.z == value.z) local_count.z++;

        if (index + global_count.x <= median_index && max.x < value.x &&
            value.x < top.x) {
          max.x = value.x;
          local_count.x = 0;
        }
        if (index + global_count.y <= median_index && max.y < value.y &&
            value.y < top.y) {
          max.y = value.y;
          local_count.y = 0;
        }
        if (index + global_count.z <= median_index && max.z < value.z &&
            value.z < top.z) {
          max.z = value.z;
          local_count.z = 0;
        }
      }
    }
    global_count.x += local_count.x;
    global_count.y += local_count.y;
    global_count.z += local_count.z;
    if (max.x != -1) top.x = max.x;
    if (max.y != -1) top.y = max.y;
    if (max.z != -1) top.z = max.z;
  }

  uchar3* output = (uchar3*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<uchar3, short3>(top);
}

template <typename BorderInterpolation>
__global__
void medianC3SharedKernel(const float* src, int rows, int cols, int src_stride,
                          int median_index, int radius, float* dst,
                          int dst_stride, BorderInterpolation interpolation) {
  __shared__ float3 data[kDimY0 + RADIUS1 * 2][kDimX0 + RADIUS1 * 2];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  {
    int index, y_index, row_index, col_index;
    float3* input;

    y_index   = threadIdx.y;
    row_index = element_y - radius;
    while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
           row_index < rows + radius) {
      index = interpolation(rows, radius, row_index);
      input = (float3*)((uchar*)src + index * src_stride);

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
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int ksize = (radius << 1) + 1;
  bool unchecked0, unchecked1, unchecked2;
  uint3 local_count  = make_uint3(0, 0, 0);
  uint3 global_count = make_uint3(0, 0, 0);
  float3 value;
  float3 max;
  float3 top = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);

  for (int index = 0; index <= median_index; index++) {
    max = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);
    unchecked0 = true;
    unchecked1 = true;
    unchecked2 = true;
    for (int i = 0; i < ksize; i++) {
      for (int j = 0; j < ksize; j++) {
        value = data[threadIdx.y + i][threadIdx.x + j];
        if ((!unchecked0) && max.x == value.x) unchecked0 = false;
        if ((!unchecked1) && max.y == value.y) unchecked1 = false;
        if ((!unchecked2) && max.z == value.z) unchecked2 = false;

        if (unchecked0 && max.x == value.x) local_count.x++;
        if (unchecked1 && max.y == value.y) local_count.y++;
        if (unchecked2 && max.z == value.z) local_count.z++;

        if (index + global_count.x <= median_index && max.x < value.x &&
            value.x < top.x) {
          max.x = value.x;
          local_count.x = 0;
        }
        if (index + global_count.y <= median_index && max.y < value.y &&
            value.y < top.y) {
          max.y = value.y;
          local_count.y = 0;
        }
        if (index + global_count.z <= median_index && max.z < value.z &&
            value.z < top.z) {
          max.z = value.z;
          local_count.z = 0;
        }
      }
    }
    global_count.x += local_count.x;
    global_count.y += local_count.y;
    global_count.z += local_count.z;
    if (max.x != FLT_MIN) top.x = max.x;
    if (max.y != FLT_MIN) top.y = max.y;
    if (max.z != FLT_MIN) top.z = max.z;
  }

  float3* output = (float3*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = top;
}

template <typename BorderInterpolation>
__global__
void medianC4SharedKernel(const uchar* src, int rows, int cols, int src_stride,
                          int median_index, int radius, uchar* dst,
                          int dst_stride, BorderInterpolation interpolation) {
  __shared__ uchar4 data[kDimY0 + RADIUS1 * 2][kDimX0 + RADIUS1 * 2];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  {
    int index, y_index, row_index, col_index;
    uchar4* input;

    y_index   = threadIdx.y;
    row_index = element_y - radius;
    while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
          row_index < rows + radius) {
      index = interpolation(rows, radius, row_index);
      input = (uchar4*)((uchar*)src + index * src_stride);

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
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int ksize = (radius << 1) + 1;
  bool unchecked0, unchecked1, unchecked2, unchecked3;
  uint4 local_count  = make_uint4(0, 0, 0, 0);
  uint4 global_count = make_uint4(0, 0, 0, 0);
  uchar4 value;
  short4 max;
  short4 top = make_short4(256, 256, 256, 256);

  for (int index = 0; index <= median_index; index++) {
    max = make_short4(-1, -1, -1, -1);
    unchecked0 = true;
    unchecked1 = true;
    unchecked2 = true;
    unchecked3 = true;
    for (int i = 0; i < ksize; i++) {
      for (int j = 0; j < ksize; j++) {
        value = data[threadIdx.y + i][threadIdx.x + j];
        if ((!unchecked0) && max.x == value.x) unchecked0 = false;
        if ((!unchecked1) && max.y == value.y) unchecked1 = false;
        if ((!unchecked2) && max.z == value.z) unchecked2 = false;
        if ((!unchecked3) && max.w == value.w) unchecked3 = false;

        if (unchecked0 && max.x == value.x) local_count.x++;
        if (unchecked1 && max.y == value.y) local_count.y++;
        if (unchecked2 && max.z == value.z) local_count.z++;
        if (unchecked3 && max.w == value.w) local_count.w++;

        if (index + global_count.x <= median_index && max.x < value.x &&
            value.x < top.x) {
          max.x = value.x;
          local_count.x = 0;
        }
        if (index + global_count.y <= median_index && max.y < value.y &&
            value.y < top.y) {
          max.y = value.y;
          local_count.y = 0;
        }
        if (index + global_count.z <= median_index && max.z < value.z &&
            value.z < top.z) {
          max.z = value.z;
          local_count.z = 0;
        }
        if (index + global_count.w <= median_index && max.w < value.w &&
            value.w < top.w) {
          max.w = value.w;
          local_count.w = 0;
        }
      }
    }
    global_count.x += local_count.x;
    global_count.y += local_count.y;
    global_count.z += local_count.z;
    global_count.w += local_count.w;
    if (max.x != -1) top.x = max.x;
    if (max.y != -1) top.y = max.y;
    if (max.z != -1) top.z = max.z;
    if (max.w != -1) top.w = max.w;
  }

  uchar4* output = (uchar4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<uchar4, short4>(top);
}

template <typename BorderInterpolation>
__global__
void medianC4SharedKernel(const float* src, int rows, int cols, int src_stride,
                          int median_index, int radius, float* dst,
                          int dst_stride, BorderInterpolation interpolation) {
  __shared__ float4 data[kDimY0 + RADIUS1 * 2][kDimX0 + RADIUS1 * 2];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  {
    int index, y_index, row_index, col_index;
    float4* input;

    y_index   = threadIdx.y;
    row_index = element_y - radius;
    while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
          row_index < rows + radius) {
      index = interpolation(rows, radius, row_index);
      input = (float4*)((uchar*)src + index * src_stride);

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
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int ksize = (radius << 1) + 1;
  bool unchecked0, unchecked1, unchecked2, unchecked3;
  uint4 local_count  = make_uint4(0, 0, 0, 0);
  uint4 global_count = make_uint4(0, 0, 0, 0);
  float4 value;
  float4 max;
  float4 top = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  for (int index = 0; index <= median_index; index++) {
    max = make_float4(FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);
    unchecked0 = true;
    unchecked1 = true;
    unchecked2 = true;
    unchecked3 = true;
    for (int i = 0; i < ksize; i++) {
      for (int j = 0; j < ksize; j++) {
        value = data[threadIdx.y + i][threadIdx.x + j];
        if ((!unchecked0) && max.x == value.x) unchecked0 = false;
        if ((!unchecked1) && max.y == value.y) unchecked1 = false;
        if ((!unchecked2) && max.z == value.z) unchecked2 = false;
        if ((!unchecked3) && max.w == value.w) unchecked3 = false;

        if (unchecked0 && max.x == value.x) local_count.x++;
        if (unchecked1 && max.y == value.y) local_count.y++;
        if (unchecked2 && max.z == value.z) local_count.z++;
        if (unchecked3 && max.w == value.w) local_count.w++;

        if (index + global_count.x <= median_index && max.x < value.x &&
            value.x < top.x) {
          max.x = value.x;
          local_count.x = 0;
        }
        if (index + global_count.y <= median_index && max.y < value.y &&
            value.y < top.y) {
          max.y = value.y;
          local_count.y = 0;
        }
        if (index + global_count.z <= median_index && max.z < value.z &&
            value.z < top.z) {
          max.z = value.z;
          local_count.z = 0;
        }
        if (index + global_count.w <= median_index && max.w < value.w &&
            value.w < top.w) {
          max.w = value.w;
          local_count.w = 0;
        }
      }
    }
    global_count.x += local_count.x;
    global_count.y += local_count.y;
    global_count.z += local_count.z;
    global_count.w += local_count.w;
    if (max.x != FLT_MIN) top.x = max.x;
    if (max.y != FLT_MIN) top.y = max.y;
    if (max.z != FLT_MIN) top.z = max.z;
    if (max.w != FLT_MIN) top.w = max.w;
  }

  float4* output = (float4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = top;
}

template <typename BorderInterpolation>
__global__
void medianC1HistKernel(const uchar* src, int rows, int cols, int src_stride,
                        ushort* histograms, int median_index, int radius,
                        uchar* dst, int dst_stride,
                        BorderInterpolation interpolation) {
  int element_x = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  int element_y = blockIdx.y << VERTICAL_SHIFT;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  uchar* input;
  ushort* coarse_histogram = (ushort*)((uchar*)histograms + (cols * blockIdx.y +
                                       element_x) * HIST_SIZE * sizeof(ushort));
  ushort* dense_histogram = coarse_histogram + 16;
  int row_end = element_y + VERTICAL_ELEMENTS < rows ?
                element_y + VERTICAL_ELEMENTS : rows;

  // processes the 1st row of each thread block and initializes the histograms.
  uchar value;
  int row, row_index, index, i;
  for (row = element_y; row < element_y + radius * 2 + 1; row++) {
    row_index = row - radius;
    index = interpolation(rows, radius, row_index);
    input = (uchar*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
    }
  }

  int count = 0;
  for (index = 0; index < 16; index++) {
    count += coarse_histogram[index];
    if (count - 1 >= median_index) {
      count -= coarse_histogram[index];
      for (int i = 0; i < 16; i++) {
        count += dense_histogram[(index << 4) + i];
        if (count - 1 >= median_index) {
          count = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }

  uchar* output;
  output = (uchar*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCast(count);

  // processes the rest rows of each thread block.
  for (row = element_y + 1; row < row_end; row++) {
    row_index = row - radius - 1;
    index = interpolation(rows, radius, row_index);
    input = (uchar*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram[(value >> 4)]--;
        dense_histogram[value]--;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram[(value >> 4)]--;
        dense_histogram[value]--;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram[(value >> 4)]--;
        dense_histogram[value]--;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram[(value >> 4)]--;
        dense_histogram[value]--;
      }
    }

    row_index = row + radius;
    index = interpolation(rows, radius, row_index);
    input = (uchar*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram[(value >> 4)]++;
        dense_histogram[value]++;
      }
    }

    count = 0;
    for (index = 0; index < 16; index++) {
      count += coarse_histogram[index];
      if (count - 1 >= median_index) {
        count -= coarse_histogram[index];
        for (int i = 0; i < 16; i++) {
          count += dense_histogram[(index << 4) + i];
          if (count - 1 >= median_index) {
            count = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }

    output = (uchar*)((uchar*)dst + row * dst_stride);
    output[element_x] = saturateCast(count);
  }
}

template <typename BorderInterpolation>
__global__
void medianC3HistKernel(const uchar* src, int rows, int cols, int src_stride,
                        ushort* histograms, int median_index, int radius,
                        uchar* dst, int dst_stride,
                        BorderInterpolation interpolation) {
  int element_x = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  int element_y = blockIdx.y << VERTICAL_SHIFT;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  uchar3* input;
  ushort* coarse_histogram0 = (ushort*)((uchar*)histograms + (cols * blockIdx.y
                              + element_x) * 3 * HIST_SIZE * sizeof(ushort));
  ushort* dense_histogram0  = coarse_histogram0 + 16;
  ushort* dense_histogram1  = dense_histogram0 + HIST_SIZE;
  ushort* dense_histogram2  = dense_histogram1 + HIST_SIZE;
  ushort* coarse_histogram1 = coarse_histogram0 + HIST_SIZE;
  ushort* coarse_histogram2 = coarse_histogram1 + HIST_SIZE;
  int row_end = element_y + VERTICAL_ELEMENTS < rows ?
                element_y + VERTICAL_ELEMENTS : rows;

  // processes the 1st row of each thread block and initializes the histograms.
  uchar3 value;
  int row, row_index, index, i;
  for (row = element_y; row < element_y + radius * 2 + 1; row++) {
    row_index = row - radius;
    index = interpolation(rows, radius, row_index);
    input = (uchar3*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
    }
  }

  int3 count = make_int3(0, 0, 0);
  for (index = 0; index < 16; index++) {
    count.x += coarse_histogram0[index];
    if (count.x - 1 >= median_index) {
      count.x -= coarse_histogram0[index];
      for (int i = 0; i < 16; i++) {
        count.x += dense_histogram0[(index << 4) + i];
        if (count.x - 1 >= median_index) {
          count.x = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }
  for (index = 0; index < 16; index++) {
    count.y += coarse_histogram1[index];
    if (count.y - 1 >= median_index) {
      count.y -= coarse_histogram1[index];
      for (int i = 0; i < 16; i++) {
        count.y += dense_histogram1[(index << 4) + i];
        if (count.y - 1 >= median_index) {
          count.y = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }
  for (index = 0; index < 16; index++) {
    count.z += coarse_histogram2[index];
    if (count.z - 1 >= median_index) {
      count.z -= coarse_histogram2[index];
      for (int i = 0; i < 16; i++) {
        count.z += dense_histogram2[(index << 4) + i];
        if (count.z - 1 >= median_index) {
          count.z = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }

  uchar3* output;
  output = (uchar3*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<uchar3, int3>(count);

  // processes the rest rows of each thread block.
  for (row = element_y + 1; row < row_end; row++) {
    row_index = row - radius - 1;
    index = interpolation(rows, radius, row_index);
    input = (uchar3*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
      }
    }

    row_index = row + radius;
    index = interpolation(rows, radius, row_index);
    input = (uchar3*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
      }
    }

    count = make_int3(0, 0, 0);
    for (index = 0; index < 16; index++) {
      count.x += coarse_histogram0[index];
      if (count.x - 1 >= median_index) {
        count.x -= coarse_histogram0[index];
        for (int i = 0; i < 16; i++) {
          count.x += dense_histogram0[(index << 4) + i];
          if (count.x - 1 >= median_index) {
            count.x = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }
    for (index = 0; index < 16; index++) {
      count.y += coarse_histogram1[index];
      if (count.y - 1 >= median_index) {
        count.y -= coarse_histogram1[index];
        for (int i = 0; i < 16; i++) {
          count.y += dense_histogram1[(index << 4) + i];
          if (count.y - 1 >= median_index) {
            count.y = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }
    for (index = 0; index < 16; index++) {
      count.z += coarse_histogram2[index];
      if (count.z - 1 >= median_index) {
        count.z -= coarse_histogram2[index];
        for (int i = 0; i < 16; i++) {
          count.z += dense_histogram2[(index << 4) + i];
          if (count.z - 1 >= median_index) {
            count.z = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }

    output = (uchar3*)((uchar*)dst + row * dst_stride);
    output[element_x] = saturateCastVector<uchar3, int3>(count);
  }
}

template <typename BorderInterpolation>
__global__
void medianC4HistKernel(const uchar* src, int rows, int cols, int src_stride,
                        ushort* histograms, int median_index, int radius,
                        uchar* dst, int dst_stride,
                        BorderInterpolation interpolation) {
  int element_x = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;
  int element_y = blockIdx.y << VERTICAL_SHIFT;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  uchar4* input;
  ushort* coarse_histogram0 = (ushort*)((uchar*)histograms + ((cols * blockIdx.y
                              + element_x) << 2) * HIST_SIZE * sizeof(ushort));
  ushort* dense_histogram0  = coarse_histogram0 + 16;
  ushort* dense_histogram1  = dense_histogram0 + HIST_SIZE;
  ushort* dense_histogram2  = dense_histogram1 + HIST_SIZE;
  ushort* coarse_histogram1 = coarse_histogram0 + HIST_SIZE;
  ushort* coarse_histogram2 = coarse_histogram1 + HIST_SIZE;
  ushort* coarse_histogram3 = coarse_histogram2 + HIST_SIZE;
  ushort* dense_histogram3 = dense_histogram2 + HIST_SIZE;
  int row_end = element_y + VERTICAL_ELEMENTS < rows ?
                element_y + VERTICAL_ELEMENTS : rows;

  // processes the 1st row of each thread block and initializes the histograms.
  uchar4 value;
  int row, row_index, index, i;
  for (row = element_y; row < element_y + radius * 2 + 1; row++) {
    row_index = row - radius;
    index = interpolation(rows, radius, row_index);
    input = (uchar4*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
    }
  }

  int4 count = make_int4(0, 0, 0, 0);
  for (index = 0; index < 16; index++) {
    count.x += coarse_histogram0[index];
    if (count.x - 1 >= median_index) {
      count.x -= coarse_histogram0[index];
      for (int i = 0; i < 16; i++) {
        count.x += dense_histogram0[(index << 4) + i];
        if (count.x - 1 >= median_index) {
          count.x = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }
  for (index = 0; index < 16; index++) {
    count.y += coarse_histogram1[index];
    if (count.y - 1 >= median_index) {
      count.y -= coarse_histogram1[index];
      for (int i = 0; i < 16; i++) {
        count.y += dense_histogram1[(index << 4) + i];
        if (count.y - 1 >= median_index) {
          count.y = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }
  for (index = 0; index < 16; index++) {
    count.z += coarse_histogram2[index];
    if (count.z - 1 >= median_index) {
      count.z -= coarse_histogram2[index];
      for (int i = 0; i < 16; i++) {
        count.z += dense_histogram2[(index << 4) + i];
        if (count.z - 1 >= median_index) {
          count.z = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }
  for (index = 0; index < 16; index++) {
    count.w += coarse_histogram3[index];
    if (count.w - 1 >= median_index) {
      count.w -= coarse_histogram3[index];
      for (int i = 0; i < 16; i++) {
        count.w += dense_histogram3[(index << 4) + i];
        if (count.w - 1 >= median_index) {
          count.w = (index << 4) + i;
          break;
        }
      }
      break;
    }
  }

  uchar4* output;
  output = (uchar4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<uchar4, int4>(count);

  // processes the rest rows of each thread block.
  for (row = element_y + 1; row < row_end; row++) {
    row_index = row - radius - 1;
    index = interpolation(rows, radius, row_index);
    input = (uchar4*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
        coarse_histogram3[(value.w >> 4)]--;
        dense_histogram3[value.w]--;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
        coarse_histogram3[(value.w >> 4)]--;
        dense_histogram3[value.w]--;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
        coarse_histogram3[(value.w >> 4)]--;
        dense_histogram3[value.w]--;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]--;
        dense_histogram0[value.x]--;
        coarse_histogram1[(value.y >> 4)]--;
        dense_histogram1[value.y]--;
        coarse_histogram2[(value.z >> 4)]--;
        dense_histogram2[value.z]--;
        coarse_histogram3[(value.w >> 4)]--;
        dense_histogram3[value.w]--;
      }
    }

    row_index = row + radius;
    index = interpolation(rows, radius, row_index);
    input = (uchar4*)((uchar*)src + index * src_stride);

    for (i = 0; i <= radius; i++) {
      if (element_x < radius) {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
      else {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
    }

    for (; i < radius * 2 + 1; i++) {
      if (element_x + radius < cols) {
        index = element_x - radius + i;
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
      else {
        index = element_x - radius + i;
        index = interpolation(cols, radius, index);
        value = input[index];
        coarse_histogram0[(value.x >> 4)]++;
        dense_histogram0[value.x]++;
        coarse_histogram1[(value.y >> 4)]++;
        dense_histogram1[value.y]++;
        coarse_histogram2[(value.z >> 4)]++;
        dense_histogram2[value.z]++;
        coarse_histogram3[(value.w >> 4)]++;
        dense_histogram3[value.w]++;
      }
    }

    count = make_int4(0, 0, 0, 0);
    for (index = 0; index < 16; index++) {
      count.x += coarse_histogram0[index];
      if (count.x - 1 >= median_index) {
        count.x -= coarse_histogram0[index];
        for (int i = 0; i < 16; i++) {
          count.x += dense_histogram0[(index << 4) + i];
          if (count.x - 1 >= median_index) {
            count.x = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }
    for (index = 0; index < 16; index++) {
      count.y += coarse_histogram1[index];
      if (count.y - 1 >= median_index) {
        count.y -= coarse_histogram1[index];
        for (int i = 0; i < 16; i++) {
          count.y += dense_histogram1[(index << 4) + i];
          if (count.y - 1 >= median_index) {
            count.y = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }
    for (index = 0; index < 16; index++) {
      count.z += coarse_histogram2[index];
      if (count.z - 1 >= median_index) {
        count.z -= coarse_histogram2[index];
        for (int i = 0; i < 16; i++) {
          count.z += dense_histogram2[(index << 4) + i];
          if (count.z - 1 >= median_index) {
            count.z = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }
    for (index = 0; index < 16; index++) {
      count.w += coarse_histogram3[index];
      if (count.w - 1 >= median_index) {
        count.w -= coarse_histogram3[index];
        for (int i = 0; i < 16; i++) {
          count.w += dense_histogram3[(index << 4) + i];
          if (count.w - 1 >= median_index) {
            count.w = (index << 4) + i;
            break;
          }
        }
        break;
      }
    }

    output = (uchar4*)((uchar*)dst + row * dst_stride);
    output[element_x] = saturateCastVector<uchar4, int4>(count);
  }
}

template <typename BorderInterpolation>
__global__
void medianC1Kernel(const float* src, int rows, int cols, int src_stride,
                    int median_index, int radius, float* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int data_index;
  bool unchecked0, unchecked1, unchecked2, unchecked3;
  uint4 local_count  = make_uint4(0, 0, 0, 0);
  uint4 global_count = make_uint4(0, 0, 0, 0);
  float* input;
  float4 value;
  float4 max;
  float4 top = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  bool isnt_border_block = true;
  data_index = radius >> (kBlockShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kBlockShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (isnt_border_block) {
    for (int index = 0; index <= median_index; index++) {
      max = make_float4(FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);
      unchecked0 = true;
      unchecked1 = true;
      unchecked2 = true;
      unchecked3 = true;
      for (int i = origin_y; i <= top_y; i++) {
        data_index = interpolation(rows, radius, i);
        input = (float*)((uchar*)src + data_index * src_stride);
        for (int j = origin_x; j <= top_x; j++) {
          value.x = input[j];
          value.y = input[j + 1];
          value.z = input[j + 2];
          value.w = input[j + 3];
          if ((!unchecked0) && max.x == value.x) unchecked0 = false;
          if ((!unchecked1) && max.y == value.y) unchecked1 = false;
          if ((!unchecked2) && max.z == value.z) unchecked2 = false;
          if ((!unchecked3) && max.w == value.w) unchecked3 = false;

          if (unchecked0 && max.x == value.x) local_count.x++;
          if (unchecked1 && max.y == value.y) local_count.y++;
          if (unchecked2 && max.z == value.z) local_count.z++;
          if (unchecked3 && max.w == value.w) local_count.w++;

          if (index + global_count.x <= median_index && max.x < value.x &&
              value.x < top.x) {
            max.x = value.x;
            local_count.x = 0;
          }
          if (index + global_count.y <= median_index && max.y < value.y &&
              value.y < top.y) {
            max.y = value.y;
            local_count.y = 0;
          }
          if (index + global_count.z <= median_index && max.z < value.z &&
              value.z < top.z) {
            max.z = value.z;
            local_count.z = 0;
          }
          if (index + global_count.w <= median_index && max.w < value.w &&
              value.w < top.w) {
            max.w = value.w;
            local_count.w = 0;
          }
        }
      }
      global_count.x += local_count.x;
      global_count.y += local_count.y;
      global_count.z += local_count.z;
      global_count.w += local_count.w;
      if (max.x != FLT_MIN) top.x = max.x;
      if (max.y != FLT_MIN) top.y = max.y;
      if (max.z != FLT_MIN) top.z = max.z;
      if (max.w != FLT_MIN) top.w = max.w;
    }
  }
  else {
    for (int index = 0; index <= median_index; index++) {
      max = make_float4(FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);
      unchecked0 = true;
      unchecked1 = true;
      unchecked2 = true;
      unchecked3 = true;
      for (int i = origin_y; i <= top_y; i++) {
        data_index = interpolation(rows, radius, i);
        input = (float*)((uchar*)src + data_index * src_stride);
        for (int j = origin_x; j <= top_x; j++) {
          data_index = interpolation(cols, radius, j);
          value.x = input[data_index];
          data_index = interpolation(cols, radius, j + 1);
          value.y = input[data_index];
          data_index = interpolation(cols, radius, j + 2);
          value.z = input[data_index];
          data_index = interpolation(cols, radius, j + 3);
          value.w = input[data_index];
          if ((!unchecked0) && max.x == value.x) unchecked0 = false;
          if ((!unchecked1) && max.y == value.y) unchecked1 = false;
          if ((!unchecked2) && max.z == value.z) unchecked2 = false;
          if ((!unchecked3) && max.w == value.w) unchecked3 = false;

          if (unchecked0 && max.x == value.x) local_count.x++;
          if (unchecked1 && max.y == value.y) local_count.y++;
          if (unchecked2 && max.z == value.z) local_count.z++;
          if (unchecked3 && max.w == value.w) local_count.w++;

          if (index + global_count.x <= median_index && max.x < value.x &&
              value.x < top.x) {
            max.x = value.x;
            local_count.x = 0;
          }
          if (index + global_count.y <= median_index && max.y < value.y &&
              value.y < top.y) {
            max.y = value.y;
            local_count.y = 0;
          }
          if (index + global_count.z <= median_index && max.z < value.z &&
              value.z < top.z) {
            max.z = value.z;
            local_count.z = 0;
          }
          if (index + global_count.w <= median_index && max.w < value.w &&
              value.w < top.w) {
            max.w = value.w;
            local_count.w = 0;
          }
        }
      }
      global_count.x += local_count.x;
      global_count.y += local_count.y;
      global_count.z += local_count.z;
      global_count.w += local_count.w;
      if (max.x != FLT_MIN) top.x = max.x;
      if (max.y != FLT_MIN) top.y = max.y;
      if (max.z != FLT_MIN) top.z = max.z;
      if (max.w != FLT_MIN) top.w = max.w;
    }
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 3) {
    output[element_x]     = top.x;
    output[element_x + 1] = top.y;
    output[element_x + 2] = top.z;
    output[element_x + 3] = top.w;
  }
  else {
    output[element_x] = top.x;
    if (element_x < cols - 1) {
      output[element_x + 1] = top.y;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = top.z;
    }
  }
}

template <typename BorderInterpolation>
__global__
void medianC3Kernel(const float* src, int rows, int cols, int src_stride,
                    int median_index, int radius, float* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int data_index;
  bool unchecked0, unchecked1, unchecked2;
  uint3 local_count  = make_uint3(0, 0, 0);
  uint3 global_count = make_uint3(0, 0, 0);
  float3* input;
  float3 value;
  float3 max;
  float3 top = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);

  bool isnt_border_block = true;
  data_index = radius >> (kBlockShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kBlockShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (isnt_border_block) {
    for (int index = 0; index <= median_index; index++) {
      max = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);
      unchecked0 = true;
      unchecked1 = true;
      unchecked2 = true;
      for (int i = origin_y; i <= top_y; i++) {
        data_index = interpolation(rows, radius, i);
        input = (float3*)((uchar*)src + data_index * src_stride);
        for (int j = origin_x; j <= top_x; j++) {
          value = input[j];
          if ((!unchecked0) && max.x == value.x) unchecked0 = false;
          if ((!unchecked1) && max.y == value.y) unchecked1 = false;
          if ((!unchecked2) && max.z == value.z) unchecked2 = false;

          if (unchecked0 && max.x == value.x) local_count.x++;
          if (unchecked1 && max.y == value.y) local_count.y++;
          if (unchecked2 && max.z == value.z) local_count.z++;

          if (index + global_count.x <= median_index && max.x < value.x &&
              value.x < top.x) {
            max.x = value.x;
            local_count.x = 0;
          }
          if (index + global_count.y <= median_index && max.y < value.y &&
              value.y < top.y) {
            max.y = value.y;
            local_count.y = 0;
          }
          if (index + global_count.z <= median_index && max.z < value.z &&
              value.z < top.z) {
            max.z = value.z;
            local_count.z = 0;
          }
        }
      }
      global_count.x += local_count.x;
      global_count.y += local_count.y;
      global_count.z += local_count.z;
      if (max.x != FLT_MIN) top.x = max.x;
      if (max.y != FLT_MIN) top.y = max.y;
      if (max.z != FLT_MIN) top.z = max.z;
    }
  }
  else {
    for (int index = 0; index <= median_index; index++) {
      max = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);
      unchecked0 = true;
      unchecked1 = true;
      unchecked2 = true;
      for (int i = origin_y; i <= top_y; i++) {
        data_index = interpolation(rows, radius, i);
        input = (float3*)((uchar*)src + data_index * src_stride);
        for (int j = origin_x; j <= top_x; j++) {
          data_index = interpolation(cols, radius, j);
          value = input[data_index];
          if ((!unchecked0) && max.x == value.x) unchecked0 = false;
          if ((!unchecked1) && max.y == value.y) unchecked1 = false;
          if ((!unchecked2) && max.z == value.z) unchecked2 = false;

          if (unchecked0 && max.x == value.x) local_count.x++;
          if (unchecked1 && max.y == value.y) local_count.y++;
          if (unchecked2 && max.z == value.z) local_count.z++;

          if (index + global_count.x <= median_index && max.x < value.x &&
              value.x < top.x) {
            max.x = value.x;
            local_count.x = 0;
          }
          if (index + global_count.y <= median_index && max.y < value.y &&
              value.y < top.y) {
            max.y = value.y;
            local_count.y = 0;
          }
          if (index + global_count.z <= median_index && max.z < value.z &&
              value.z < top.z) {
            max.z = value.z;
            local_count.z = 0;
          }
        }
      }
      global_count.x += local_count.x;
      global_count.y += local_count.y;
      global_count.z += local_count.z;
      if (max.x != FLT_MIN) top.x = max.x;
      if (max.y != FLT_MIN) top.y = max.y;
      if (max.z != FLT_MIN) top.z = max.z;
    }
  }

  float3* output = (float3*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = top;
}

template <typename BorderInterpolation>
__global__
void medianC4Kernel(const float* src, int rows, int cols, int src_stride,
                    int median_index, int radius, float* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int data_index;
  bool unchecked0, unchecked1, unchecked2, unchecked3;
  uint4 local_count  = make_uint4(0, 0, 0, 0);
  uint4 global_count = make_uint4(0, 0, 0, 0);
  float4* input;
  float4 value;
  float4 max;
  float4 top = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  bool isnt_border_block = true;
  data_index = radius >> (kBlockShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kBlockShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (isnt_border_block) {
    for (int index = 0; index <= median_index; index++) {
      max = make_float4(FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);
      unchecked0 = true;
      unchecked1 = true;
      unchecked2 = true;
      unchecked3 = true;
      for (int i = origin_y; i <= top_y; i++) {
        data_index = interpolation(rows, radius, i);
        input = (float4*)((uchar*)src + data_index * src_stride);
        for (int j = origin_x; j <= top_x; j++) {
          value = input[j];
          if ((!unchecked0) && max.x == value.x) unchecked0 = false;
          if ((!unchecked1) && max.y == value.y) unchecked1 = false;
          if ((!unchecked2) && max.z == value.z) unchecked2 = false;
          if ((!unchecked3) && max.w == value.w) unchecked3 = false;

          if (unchecked0 && max.x == value.x) local_count.x++;
          if (unchecked1 && max.y == value.y) local_count.y++;
          if (unchecked2 && max.z == value.z) local_count.z++;
          if (unchecked3 && max.w == value.w) local_count.w++;

          if (index + global_count.x <= median_index && max.x < value.x &&
              value.x < top.x) {
            max.x = value.x;
            local_count.x = 0;
          }
          if (index + global_count.y <= median_index && max.y < value.y &&
              value.y < top.y) {
            max.y = value.y;
            local_count.y = 0;
          }
          if (index + global_count.z <= median_index && max.z < value.z &&
              value.z < top.z) {
            max.z = value.z;
            local_count.z = 0;
          }
          if (index + global_count.w <= median_index && max.w < value.w &&
              value.w < top.w) {
            max.w = value.w;
            local_count.w = 0;
          }
        }
      }
      global_count.x += local_count.x;
      global_count.y += local_count.y;
      global_count.z += local_count.z;
      global_count.w += local_count.w;
      if (max.x != FLT_MIN) top.x = max.x;
      if (max.y != FLT_MIN) top.y = max.y;
      if (max.z != FLT_MIN) top.z = max.z;
      if (max.w != FLT_MIN) top.w = max.w;
    }
  }
  else {
    for (int index = 0; index <= median_index; index++) {
      max = make_float4(FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);
      unchecked0 = true;
      unchecked1 = true;
      unchecked2 = true;
      unchecked3 = true;
      for (int i = origin_y; i <= top_y; i++) {
        data_index = interpolation(rows, radius, i);
        input = (float4*)((uchar*)src + data_index * src_stride);
        for (int j = origin_x; j <= top_x; j++) {
          data_index = interpolation(cols, radius, j);
          value = input[data_index];
          if ((!unchecked0) && max.x == value.x) unchecked0 = false;
          if ((!unchecked1) && max.y == value.y) unchecked1 = false;
          if ((!unchecked2) && max.z == value.z) unchecked2 = false;
          if ((!unchecked3) && max.w == value.w) unchecked3 = false;

          if (unchecked0 && max.x == value.x) local_count.x++;
          if (unchecked1 && max.y == value.y) local_count.y++;
          if (unchecked2 && max.z == value.z) local_count.z++;
          if (unchecked3 && max.w == value.w) local_count.w++;

          if (index + global_count.x <= median_index && max.x < value.x &&
              value.x < top.x) {
            max.x = value.x;
            local_count.x = 0;
          }
          if (index + global_count.y <= median_index && max.y < value.y &&
              value.y < top.y) {
            max.y = value.y;
            local_count.y = 0;
          }
          if (index + global_count.z <= median_index && max.z < value.z &&
              value.z < top.z) {
            max.z = value.z;
            local_count.z = 0;
          }
          if (index + global_count.w <= median_index && max.w < value.w &&
              value.w < top.w) {
            max.w = value.w;
            local_count.w = 0;
          }
        }
      }
      global_count.x += local_count.x;
      global_count.y += local_count.y;
      global_count.z += local_count.z;
      global_count.w += local_count.w;
      if (max.x != FLT_MIN) top.x = max.x;
      if (max.y != FLT_MIN) top.y = max.y;
      if (max.z != FLT_MIN) top.z = max.z;
      if (max.w != FLT_MIN) top.w = max.w;
    }
  }

  float4* output = (float4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = top;
}

#define RUN_C1_SMALL_KERNELS(Interpolation)                                    \
Interpolation interpolation;                                                   \
medianC1SharedKernel<Interpolation><<<grid, block, 0, stream>>>(src, rows,     \
    cols, src_stride, median_index, radius, dst, dst_stride, interpolation);

#define RUN_CN_SMALL_KERNELS(Interpolation)                                    \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  medianC3SharedKernel<Interpolation><<<grid, block, 0, stream>>>(src, rows,   \
      cols, src_stride, median_index, radius, dst, dst_stride, interpolation); \
}                                                                              \
else {                                                                         \
  medianC4SharedKernel<Interpolation><<<grid, block, 0, stream>>>(src, rows,   \
      cols, src_stride, median_index, radius, dst, dst_stride, interpolation); \
}

#define RUN_HISTOGRAM_C1_SMALL_KERNELS(Interpolation)                          \
Interpolation interpolation;                                                   \
medianC1HistKernel<Interpolation><<<grid, block, 0, stream>>>(src, rows, cols, \
    src_stride, histograms, median_index, radius, dst, dst_stride,             \
    interpolation);

#define RUN_HISTOGRAM_CN_SMALL_KERNELS(Interpolation)                          \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  medianC3HistKernel<Interpolation><<<grid, block, 0, stream>>>(src, rows,     \
      cols, src_stride, histograms, median_index, radius, dst, dst_stride,     \
      interpolation);                                                          \
}                                                                              \
else {                                                                         \
  medianC4HistKernel<Interpolation><<<grid, block, 0, stream>>>(src, rows,     \
      cols, src_stride, histograms, median_index, radius, dst, dst_stride,     \
      interpolation);                                                          \
}

#define RUN_FLOAT_KERNELS(grid_x, Interpolation)                               \
Interpolation interpolation;                                                   \
if (channels == 1) {                                                           \
  grid.x = grid_x;                                                             \
  medianC1Kernel<Interpolation><<<grid, block, 0, stream>>>(src, rows, cols,   \
      src_stride, median_index, radius, dst, dst_stride, interpolation);       \
}                                                                              \
else if (channels == 3) {                                                      \
  medianC3Kernel<Interpolation><<<grid, block, 0, stream>>>(src, rows, cols,   \
      src_stride, median_index, radius, dst, dst_stride, interpolation);       \
}                                                                              \
else {                                                                         \
  medianC4Kernel<Interpolation><<<grid, block, 0, stream>>>(src, rows, cols,   \
      src_stride, median_index, radius, dst, dst_stride, interpolation);       \
}

RetCode medainblur(const uchar* src, int rows, int cols, int channels,
                   int src_stride, uchar* dst, int dst_stride, int ksize,
                   BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize > 1);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  uint radius = ksize >> 1;
  uint median_index = ksize * ksize >> 1;
  dim3 block, grid;

  cudaError_t code;
  if (channels == 1) {
    if (ksize <= 7) {
      block.x = kDimX0;
      block.y = kDimY0;
      grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
      grid.y = divideUp(rows, kDimY0, kShiftY0);

      if (border_type == BORDER_REPLICATE) {
        RUN_C1_SMALL_KERNELS(ReplicateBorder);
      }
      else if (border_type == BORDER_REFLECT) {
        RUN_C1_SMALL_KERNELS(ReflectBorder);
      }
      else {
        RUN_C1_SMALL_KERNELS(Reflect101Border);
      }

      code = cudaGetLastError();
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_RUNTIME_ERROR;
      }
      else {
        return RC_SUCCESS;
      }
    }
    else {
      block.x = BLOCK_WIDTH;
      block.y = 1;
      grid.x = divideUp(cols, BLOCK_WIDTH, BLOCK_SHIFT);
      grid.y = divideUp(rows, VERTICAL_ELEMENTS, VERTICAL_SHIFT);

      ushort* histograms;
      size_t hist_size = cols * grid.y * HIST_SIZE * sizeof(ushort);
      GpuMemoryBlock buffer_block;
      if (memoryPoolUsed()) {
        pplCudaMalloc(hist_size, buffer_block);
        histograms = (ushort*)(buffer_block.data);
      }
      else {
        code = cudaMalloc(&histograms, hist_size);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
          return RC_DEVICE_MEMORY_ERROR;
        }
      }

      code = cudaMemsetAsync(histograms, 0, hist_size, stream);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }

      if (border_type == BORDER_REPLICATE) {
        RUN_HISTOGRAM_C1_SMALL_KERNELS(ReplicateBorder);
      }
      else if (border_type == BORDER_REFLECT) {
        RUN_HISTOGRAM_C1_SMALL_KERNELS(ReflectBorder);
      }
      else {
        RUN_HISTOGRAM_C1_SMALL_KERNELS(Reflect101Border);
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
        cudaFree(histograms);
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
  }
  else {  // channels == 3 || channels == 4
    if (ksize <= 5) {
      block.x = kDimX0;
      block.y = kDimY0;
      grid.x = divideUp(cols, kDimX0, kShiftX0);
      grid.y = divideUp(rows, kDimY0, kShiftY0);

      if (border_type == BORDER_REPLICATE) {
        RUN_CN_SMALL_KERNELS(ReplicateBorder);
      }
      else if (border_type == BORDER_REFLECT) {
        RUN_CN_SMALL_KERNELS(ReflectBorder);
      }
      else {
        RUN_CN_SMALL_KERNELS(Reflect101Border);
      }

      code = cudaGetLastError();
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_RUNTIME_ERROR;
      }
      else {
        return RC_SUCCESS;
      }
    }
    else {
      block.x = BLOCK_WIDTH;
      block.y = 1;
      grid.x = divideUp(cols, BLOCK_WIDTH, BLOCK_SHIFT);
      grid.y = divideUp(rows, VERTICAL_ELEMENTS, VERTICAL_SHIFT);

      ushort* histograms;
      size_t hist_size = cols * channels * grid.y * HIST_SIZE * sizeof(ushort);
      GpuMemoryBlock buffer_block;
      if (memoryPoolUsed()) {
        pplCudaMalloc(hist_size, buffer_block);
        histograms = (ushort*)(buffer_block.data);
      }
      else {
        code = cudaMalloc(&histograms, hist_size);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
          return RC_DEVICE_MEMORY_ERROR;
        }
      }

      code = cudaMemsetAsync(histograms, 0, hist_size, stream);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }

      if (border_type == BORDER_REPLICATE) {
        RUN_HISTOGRAM_CN_SMALL_KERNELS(ReplicateBorder);
      }
      else if (border_type == BORDER_REFLECT) {
        RUN_HISTOGRAM_CN_SMALL_KERNELS(ReflectBorder);
      }
      else {
        RUN_HISTOGRAM_CN_SMALL_KERNELS(Reflect101Border);
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
        cudaFree(histograms);
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
  }
}

RetCode medainblur(const float* src, int rows, int cols, int channels,
                   int src_stride, float* dst, int dst_stride, int ksize,
                   BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize > 1);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  uint radius = ksize >> 1;
  uint median_index = ksize * ksize >> 1;
  dim3 block, grid;

  cudaError_t code;
  if (ksize <= SMALL_KSIZE0 && channels == 1) {
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_C1_SMALL_KERNELS(ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_C1_SMALL_KERNELS(ReflectBorder);
    }
    else {
      RUN_C1_SMALL_KERNELS(Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  if (ksize <= SMALL_KSIZE1 && (channels == 3 || channels == 4)) {
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(cols, kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CN_SMALL_KERNELS(ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CN_SMALL_KERNELS(ReflectBorder);
    }
    else {
      RUN_CN_SMALL_KERNELS(Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  int grid_x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  if (border_type == BORDER_REPLICATE) {
    RUN_FLOAT_KERNELS(grid_x, ReplicateBorder);
  }
  else if (border_type == BORDER_REFLECT) {
    RUN_FLOAT_KERNELS(grid_x, ReflectBorder);
  }
  else {
    RUN_FLOAT_KERNELS(grid_x, Reflect101Border);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode MedianBlur<uchar, 1>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const uchar* inData,
                             int outWidthStride,
                             uchar* outData,
                             int ksize,
                             BorderType border_type) {
  RetCode code = medainblur(inData, height, width, 1, inWidthStride, outData,
                            outWidthStride, ksize, border_type, stream);

  return code;
}

template <>
RetCode MedianBlur<uchar, 3>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const uchar* inData,
                             int outWidthStride,
                             uchar* outData,
                             int ksize,
                             BorderType border_type) {
  RetCode code = medainblur(inData, height, width, 3, inWidthStride, outData,
                            outWidthStride, ksize, border_type, stream);

  return code;
}

template <>
RetCode MedianBlur<uchar, 4>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const uchar* inData,
                             int outWidthStride,
                             uchar* outData,
                             int ksize,
                             BorderType border_type) {
  RetCode code = medainblur(inData, height, width, 4, inWidthStride, outData,
                            outWidthStride, ksize, border_type, stream);

  return code;
}

template <>
RetCode MedianBlur<float, 1>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const float* inData,
                             int outWidthStride,
                             float* outData,
                             int ksize,
                             BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = medainblur(inData, height, width, 1, inWidthStride, outData,
                            outWidthStride, ksize, border_type, stream);

  return code;
}

template <>
RetCode MedianBlur<float, 3>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const float* inData,
                             int outWidthStride,
                             float* outData,
                             int ksize,
                             BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = medainblur(inData, height, width, 3, inWidthStride, outData,
                            outWidthStride, ksize, border_type, stream);

  return code;
}

template <>
RetCode MedianBlur<float, 4>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const float* inData,
                             int outWidthStride,
                             float* outData,
                             int ksize,
                             BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = medainblur(inData, height, width, 4, inWidthStride, outData,
                            outWidthStride, ksize, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
