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

#include "ppl/cv/cuda/resize.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define INCREASE(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))
#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

cudaTextureObject_t uchar_c1_tex = 0;
cudaTextureObject_t uchar_c4_tex = 0;
cudaTextureObject_t float_c1_tex = 0;

template <typename T>
__DEVICE__
T bilinearSample(T values[][2], int x0, int x1, int y0, int y1);

template <>
__DEVICE__
uchar2 bilinearSample(uchar2 values[][2], int x0, int x1, int y0, int y1) {
  int a0 = y0 * x0;
  int a1 = y0 * x1;
  int a2 = y1 * x0;
  int a3 = y1 * x1;

  int2 value;
  uchar2 result;
  value.x  = values[0][0].x * a0 + values[0][1].x * a1 + values[1][0].x * a2 +
             values[1][1].x * a3;
  result.x = (value.x + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  value.y  = values[0][0].y * a0 + values[0][1].y * a1 + values[1][0].y * a2 +
             values[1][1].y * a3;
  result.y = (value.y + (1 << (CAST_BITS - 1))) >> CAST_BITS;

  return result;
}

template <>
__DEVICE__
uchar3 bilinearSample(uchar3 values[][2], int x0, int x1, int y0, int y1) {
  int a0 = y0 * x0;
  int a1 = y0 * x1;
  int a2 = y1 * x0;
  int a3 = y1 * x1;

  int3 value;
  uchar3 result;
  value.x  = values[0][0].x * a0 + values[0][1].x * a1 + values[1][0].x * a2 +
             values[1][1].x * a3;
  result.x = (value.x + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  value.y  = values[0][0].y * a0 + values[0][1].y * a1 + values[1][0].y * a2 +
             values[1][1].y * a3;
  result.y = (value.y + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  value.z  = values[0][0].z * a0 + values[0][1].z * a1 + values[1][0].z * a2 +
             values[1][1].z * a3;
  result.z = (value.z + (1 << (CAST_BITS - 1))) >> CAST_BITS;

  return result;
}

template <>
__DEVICE__
uchar4 bilinearSample(uchar4 values[][2], int x0, int x1, int y0, int y1) {
  int a0 = y0 * x0;
  int a1 = y0 * x1;
  int a2 = y1 * x0;
  int a3 = y1 * x1;

  int4 value;
  uchar4 result;
  value.x  = values[0][0].x * a0 + values[0][1].x * a1 + values[1][0].x * a2 +
             values[1][1].x * a3;
  result.x = (value.x + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  value.y  = values[0][0].y * a0 + values[0][1].y * a1 + values[1][0].y * a2 +
             values[1][1].y * a3;
  result.y = (value.y + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  value.z  = values[0][0].z * a0 + values[0][1].z * a1 + values[1][0].z * a2 +
             values[1][1].z * a3;
  result.z = (value.z + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  value.w  = values[0][0].w * a0 + values[0][1].w * a1 + values[1][0].w * a2 +
             values[1][1].w * a3;
  result.w = (value.w + (1 << (CAST_BITS - 1))) >> CAST_BITS;

  return result;
}

__global__
void resizeLinearTextureKernel(uchar* dst, int dst_rows, int dst_cols,
                               int channels, int dst_stride, float col_scale,
                               float row_scale) {
  cudaTextureObject_t uchar_c1_tex = 0;
  cudaTextureObject_t uchar_c4_tex = 0;
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float coordinate_x = (element_x + 0.5f) * col_scale;
  float coordinate_y = (element_y + 0.5f) * row_scale;

  if (channels == 1) {
    float value = tex2D<float>(uchar_c1_tex, coordinate_x, coordinate_y);
    value *= 255.0f;

    uchar* output = (uchar*)(dst + element_y * dst_stride);
    output[element_x] = saturateCast(value);
  }
  else {  // channels == 4
    float4 value = tex2D<float4>(uchar_c4_tex, coordinate_x, coordinate_y);
    value.x *= 255.0f;
    value.y *= 255.0f;
    value.z *= 255.0f;
    value.w *= 255.0f;

    uchar4* output = (uchar4*)(dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<uchar4, float4>(value);
  }
}

__global__
void resizeLinearTextureKernel(float* dst, int dst_rows, int dst_cols,
                               int channels, int dst_stride, float col_scale,
                               float row_scale) {
  cudaTextureObject_t float_c1_tex = 0;
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float coordinate_x = (element_x + 0.5f) * col_scale;
  float coordinate_y = (element_y + 0.5f) * row_scale;

  if (channels == 1) {
    float value = tex2D<float>(float_c1_tex, coordinate_x, coordinate_y);

    float* output = (float*)(dst + element_y * dst_stride);
    output[element_x] = value;
  }
}

__global__
void resizeLinearKernel(const uchar* src, int src_rows, int src_cols,
                        int channels, int src_stride, uchar* dst, int dst_rows,
                        int dst_cols, int dst_stride, float col_scale,
                        float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_y = ((element_y + 0.5f) * row_scale - 0.5f);
  float float_x = ((element_x + 0.5f) * col_scale - 0.5f);
  int int_y0 = floor(float_y);
  int int_x0 = floor(float_x);
  float_y -= int_y0;
  float_x -= int_x0;
  if (int_y0 < 0) {
    int_y0 = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0 = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0 = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0 = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0, src_rows);
  int buf_y[2];
  float_y = float_y * INTER_RESIZE_COEF_SCALE;
  buf_y[0] = rint(INTER_RESIZE_COEF_SCALE - float_y);
  buf_y[1] = rint(float_y);

  int int_x1 = INCREASE(int_x0, src_cols);
  int buf_x[2];
  float_x = float_x * INTER_RESIZE_COEF_SCALE;
  buf_x[0] = rint(INTER_RESIZE_COEF_SCALE - rint(float_x));
  buf_x[1] = rint(float_x);

  if (channels == 1) {
    int src_index0 = int_y0 * src_stride + int_x0;
    int src_index1 = int_y0 * src_stride + int_x1;
    int src_index2 = int_y1 * src_stride + int_x0;
    int src_index3 = int_y1 * src_stride + int_x1;

    int sum = 0;
    sum = buf_y[0] * buf_x[0] * src[src_index0] +
          buf_y[0] * buf_x[1] * src[src_index1] +
          buf_y[1] * buf_x[0] * src[src_index2] +
          buf_y[1] * buf_x[1] * src[src_index3];

    uchar* output = (uchar*)(dst + element_y * dst_stride);
    output[element_x] = (sum + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  }
  else if (channels == 3) {
    uchar3* input0 = (uchar3*)((uchar*)src + int_y0 * src_stride);
    uchar3* input1 = (uchar3*)((uchar*)src + int_y1 * src_stride);

    uchar3 values[2][2];
    values[0][0] = input0[int_x0];
    values[0][1] = input0[int_x1];
    values[1][0] = input1[int_x0];
    values[1][1] = input1[int_x1];

    uchar3* output = (uchar3*)(dst + element_y * dst_stride);
    output[element_x] = bilinearSample(values, buf_x[0], buf_x[1], buf_y[0],
                                       buf_y[1]);
  }
  else {
    uchar4* input0 = (uchar4*)((uchar*)src + int_y0 * src_stride);
    uchar4* input1 = (uchar4*)((uchar*)src + int_y1 * src_stride);

    uchar4 values[2][2];
    values[0][0] = input0[int_x0];
    values[0][1] = input0[int_x1];
    values[1][0] = input1[int_x0];
    values[1][1] = input1[int_x1];

    uchar4* output = (uchar4*)(dst + element_y * dst_stride);
    output[element_x] = bilinearSample(values, buf_x[0], buf_x[1], buf_y[0],
                                       buf_y[1]);
  }
}

__global__
void resizeLinearKernel(const float* src, int src_rows, int src_cols,
                        int channels, int src_stride, float* dst, int dst_rows,
                        int dst_cols, int dst_stride, double col_scale,
                        float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_x = ((element_x + 0.5f) * col_scale - 0.5f);
  float float_y = ((element_y + 0.5f) * row_scale - 0.5f);
  int int_x0 = floor(float_x);
  int int_y0 = floor(float_y);
  float_x -= int_x0;
  float_y -= int_y0;
  if (int_y0 < 0) {
    int_y0 = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0 = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0 = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0 = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0, src_rows);
  float buf_y[2];
  buf_y[0] = 1.f - float_y;
  buf_y[1] = 1.f - buf_y[0];

  int int_x1 = INCREASE(int_x0, src_cols);
  float buf_x[2];
  buf_x[0] = 1.f - float_x;
  buf_x[1] = 1.f - buf_x[0];

  if (channels == 1) {
    int index = int_y0 * src_stride;
    float src0 = src[index + int_x0];
    float src1 = src[index + int_x1];
    float value0 = buf_y[0] * buf_x[0] * src0;
    float value1 = buf_y[0] * buf_x[1] * src1;
    float sum = 0.f;
    sum += value0 + value1;

    index = int_y1 * src_stride;
    src0 = src[index + int_x0];
    src1 = src[index + int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0 + value1;

    float* output = (float*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
  else if (channels == 3) {
    int index = int_y0 * src_stride;
    float3 src0 = ((float3*)(src + index))[int_x0];
    float3 src1 = ((float3*)(src + index))[int_x1];
    float3 value0 = buf_y[0] * buf_x[0] * src0;
    float3 value1 = buf_y[0] * buf_x[1] * src1;
    float3 sum = make_float3(0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = ((float3*)(src + index))[int_x0];
    src1 = ((float3*)(src + index))[int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    float3* output = (float3*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
  else {
    int index = int_y0 * src_stride;
    float4 src0 = ((float4*)(src + index))[int_x0];
    float4 src1 = ((float4*)(src + index))[int_x1];
    float4 value0 = buf_y[0] * buf_x[0] * src0;
    float4 value1 = buf_y[0] * buf_x[1] * src1;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = ((float4*)(src + index))[int_x0];
    src1 = ((float4*)(src + index))[int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    float4* output = (float4*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
}

template <typename T, typename Tn>
__global__
void resizeNearestPointKernel(const T* src, int src_rows, int src_cols,
                              int channels, int src_stride, T* dst,
                              int dst_rows, int dst_cols, int dst_stride,
                              float col_scale, float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int int_y = element_y * row_scale;
  int_y = min(int_y, src_rows - 1);
  int int_x = element_x * col_scale;
  int_x = min(int_x, src_cols - 1);

  Tn* input  = (Tn*)(src + int_y * src_stride);
  Tn* output = (Tn*)(dst + element_y * dst_stride);
  output[element_x] = input[int_x];
}

template <typename T>
__global__
void resizeAreaC1Kernel0(const T* src, int src_rows, int src_cols, int channels,
                         int src_stride, T* dst, int dst_rows, int dst_cols,
                         int dst_stride, int col_scale, int row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  int area = (x_end - x_start) * (y_end - y_start);

  float sum = 0.f;
  T* input;
  for (int i = y_start; i < y_end; ++i) {
    input = (T*)(src + i * src_stride);
    for (int j = x_start; j < x_end; ++j) {
      sum += input[j];
    }
  }
  sum /= area;

  T* output = (T*)(dst + element_y * dst_stride);
  if (sizeof(T) == 1) {
    output[element_x] = saturateCast(sum);
  }
  else {
    output[element_x] = sum;
  }
}

template <typename T, typename Tn>
__global__
void resizeAreaC3Kernel0(const T* src, int src_rows, int src_cols,
                         int channels, int src_stride, T* dst, int dst_rows,
                         int dst_cols, int dst_stride, int col_scale,
                         int row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  int area = (x_end - x_start) * (y_end - y_start);

  float3 sum = make_float3(0.f, 0.f, 0.f);
  Tn* input;
  for (int i = y_start; i < y_end; ++i) {
    input = (Tn*)(src + i * src_stride);
    for (int j = x_start; j < x_end; ++j) {
      sum += input[j];
    }
  }
  sum /= area;

  Tn* output = (Tn*)(dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tn, float3>(sum);
}

template <typename T, typename Tn>
__global__
void resizeAreaC4Kernel0(const T* src, int src_rows, int src_cols,
                         int channels, int src_stride, T* dst, int dst_rows,
                         int dst_cols, int dst_stride, int col_scale,
                         int row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  int area = (x_end - x_start) * (y_end - y_start);

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  Tn* input;
  for (int i = y_start; i < y_end; ++i) {
    input = (Tn*)(src + i * src_stride);
    for (int j = x_start; j < x_end; ++j) {
      sum += input[j];
    }
  }
  sum /= area;

  Tn* output = (Tn*)(dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tn, float4>(sum);
}

template <typename T>
__global__
void resizeAreaC1Kernel1(const T* src, int src_rows, int src_cols, int channels,
                         int src_stride, T* dst, int dst_rows, int dst_cols,
                         int dst_stride, float col_scale, float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_y0 = element_y * row_scale;
  float float_y1 = float_y0 + row_scale;
  int int_y0 = ceilf(float_y0);
  int int_y1 = floorf(float_y1);

  float float_x0 = element_x * col_scale;
  float float_x1 = float_x0 + col_scale;
  int int_x0 = ceilf(float_x0);
  int int_x1 = floorf(float_x1);

  T* input;
  float sum = 0.f;
  float area = fminf(col_scale, src_cols - float_x0) *
               fminf(row_scale, src_rows - float_y0);

  if (int_y0 - float_y0 > 1e-3) {
    input = (T*)(src + (int_y0 - 1) * src_stride);
    if (int_x0 - float_x0 > 1e-3) {
      sum = sum + input[int_x0 - 1] * (int_y0 - float_y0) * (int_x0 - float_x0);
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      sum = sum + input[dx] * (int_y0 - float_y0);
    }

    if (float_x1 - int_x1 > 1e-3) {
      sum = sum + input[int_x1] * (int_y0 - float_y0) * (float_x1 - int_x1);
    }
  }

  input = (T*)(src + int_y0 * src_stride);
  for (int dy = int_y0; dy < int_y1; ++dy) {
    if (int_x0 - float_x0 > 1e-3) {
      sum = sum + input[int_x0 - 1] * (int_x0 - float_x0);
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      sum = sum + input[dx];
    }

    if (float_x1 - int_x1 > 1e-3) {
      sum = sum + input[int_x1] * (float_x1 - int_x1);
    }
    input += src_stride;
  }

  if (float_y1 - int_y1 > 1e-3) {
    if (int_x0 - float_x0 > 1e-3) {
      sum = sum + input[int_x0 - 1] * (float_y1 - int_y1) * (int_x0 - float_x0);
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      sum = sum + input[dx] * (float_y1 - int_y1);
    }

    if (float_x1 - int_x1 > 1e-3) {
      sum = sum + input[int_x1] * (float_y1 - int_y1) * (float_x1 - int_x1);
    }
  }
  sum = sum / area;

  T* output = (T*)(dst + element_y * dst_stride);
  if (sizeof(T) == 1) {
    output[element_x] = saturateCast(sum);
  }
  else {
    output[element_x] = sum;
  }
}

template <typename T, typename Tn>
__global__
void resizeAreaC3Kernel1(const T* src, int src_rows, int src_cols,
                         int channels, int src_stride, T* dst, int dst_rows,
                         int dst_cols, int dst_stride, float col_scale,
                         float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_y0 = element_y * row_scale;
  float float_y1 = float_y0 + row_scale;
  int int_y0 = ceilf(float_y0);
  int int_y1 = floorf(float_y1);

  float float_x0 = element_x * col_scale;
  float float_x1 = float_x0 + col_scale;
  int int_x0 = ceilf(float_x0);
  int int_x1 = floorf(float_x1);

  Tn* input;
  float3 value;
  float3 sum = make_float3(0.f, 0.f, 0.f);
  float area = fminf(col_scale, src_cols - float_x0) *
               fminf(row_scale, src_rows - float_y0);

  if (int_y0 - float_y0 > 1e-3) {
    input = (Tn*)(src + (int_y0 - 1) * src_stride);
    if (int_x0 - float_x0 > 1e-3) {
      value = (int_y0 - float_y0) * (int_x0 - float_x0) * input[int_x0 - 1];
      sum += value;
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      value = (int_y0 - float_y0) * input[dx];
      sum += value;
    }

    if (float_x1 - int_x1 > 1e-3) {
      value = (int_y0 - float_y0) * (float_x1 - int_x1) * input[int_x1];
      sum += value;
    }
  }

  input = (Tn*)(src + int_y0 * src_stride);
  for (int dy = int_y0; dy < int_y1; ++dy) {
    if (int_x0 - float_x0 > 1e-3) {
      value = (int_x0 - float_x0) * input[int_x0 - 1];
      sum += value;
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      sum += input[dx];
    }

    if (float_x1 - int_x1 > 1e-3) {
      value = (float_x1 - int_x1) * input[int_x1];
      sum += value;
    }
    input = (Tn*)((T*)input + src_stride);
  }

  if (float_y1 - int_y1 > 1e-3) {
    if (int_x0 - float_x0 > 1e-3) {
      value = (float_y1 - int_y1) * (int_x0 - float_x0) * input[int_x0 - 1];
      sum += value;
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      value = (float_y1 - int_y1) * input[dx];
      sum += value;
    }

    if (float_x1 - int_x1 > 1e-3) {
      value = (float_y1 - int_y1) * (float_x1 - int_x1) * input[int_x1];
      sum += value;
    }
  }
  sum /= area;

  Tn* output = (Tn*)(dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tn, float3>(sum);
}

template <typename T, typename Tn>
__global__
void resizeAreaC4Kernel1(const T* src, int src_rows, int src_cols,
                         int channels, int src_stride, T* dst, int dst_rows,
                         int dst_cols, int dst_stride, float col_scale,
                         float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_y0 = element_y * row_scale;
  float float_y1 = float_y0 + row_scale;
  int int_y0 = ceilf(float_y0);
  int int_y1 = floorf(float_y1);

  float float_x0 = element_x * col_scale;
  float float_x1 = float_x0 + col_scale;
  int int_x0 = ceilf(float_x0);
  int int_x1 = floorf(float_x1);

  Tn* input;
  float4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float area = fminf(col_scale, src_cols - float_x0) *
               fminf(row_scale, src_rows - float_y0);

  if (int_y0 - float_y0 > 1e-3) {
    input = (Tn*)(src + (int_y0 - 1) * src_stride);
    if (int_x0 - float_x0 > 1e-3) {
      value = (int_y0 - float_y0) * (int_x0 - float_x0) * input[int_x0 - 1];
      sum += value;
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      value = (int_y0 - float_y0) * input[dx];
      sum += value;
    }

    if (float_x1 - int_x1 > 1e-3) {
      value = (int_y0 - float_y0) * (float_x1 - int_x1) * input[int_x1];
      sum += value;
    }
  }

  input = (Tn*)(src + int_y0 * src_stride);
  for (int dy = int_y0; dy < int_y1; ++dy) {
    if (int_x0 - float_x0 > 1e-3) {
      value = (int_x0 - float_x0) * input[int_x0 - 1];
      sum += value;
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      sum += input[dx];
    }

    if (float_x1 - int_x1 > 1e-3) {
      value = (float_x1 - int_x1) * input[int_x1];
      sum += value;
    }
    input = (Tn*)((T*)input + src_stride);
  }

  if (float_y1 - int_y1 > 1e-3) {
    if (int_x0 - float_x0 > 1e-3) {
      value = (float_y1 - int_y1) * (int_x0 - float_x0) * input[int_x0 - 1];
      sum += value;
    }

    for (int dx = int_x0; dx < int_x1; ++dx) {
      value = (float_y1 - int_y1) * input[dx];
      sum += value;
    }

    if (float_x1 - int_x1 > 1e-3) {
      value = (float_y1 - int_y1) * (float_x1 - int_x1) * input[int_x1];
      sum += value;
    }
  }
  sum /= area;

  Tn* output = (Tn*)(dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tn, float4>(sum);
}

__global__
void resizeAreaTextureKernel(uchar* dst, int dst_rows, int dst_cols,
                             int channels, int dst_stride, float col_scale,
                             float row_scale, float inv_col_scale,
                             float inv_row_scale) {
  cudaTextureObject_t uchar_c1_tex = 0;
  cudaTextureObject_t uchar_c4_tex = 0;
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_x = element_x * col_scale;
  float float_y = element_y * row_scale;
  int int_x = floor(float_x);
  int int_y = floor(float_y);
  float_x = element_x + 1 - (int_x + 1) * inv_col_scale;
  float_y = element_y + 1 - (int_y + 1) * inv_row_scale;
  float_x = float_x <= 0 ? 0.f : float_x - floor(float_x);
  float_y = float_y <= 0 ? 0.f : float_y - floor(float_y);
  float_x += (int_x + 0.5f);
  float_y += (int_y + 0.5f);

  if (channels == 1) {
    float value = tex2D<float>(uchar_c1_tex, float_x, float_y);
    value *= 255.0f;

    uchar* output = (uchar*)(dst + element_y * dst_stride);
    output[element_x] = saturateCast(value);
  }
  else {  // channels == 4
    float4 value = tex2D<float4>(uchar_c4_tex, float_x, float_y);
    value.x *= 255.0f;
    value.y *= 255.0f;
    value.z *= 255.0f;
    value.w *= 255.0f;

    uchar4* output = (uchar4*)(dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<uchar4, float4>(value);
  }
}

__global__
void resizeAreaTextureKernel(float* dst, int dst_rows, int dst_cols,
                             int channels, int dst_stride, float col_scale,
                             float row_scale, float inv_col_scale,
                             float inv_row_scale) {
  cudaTextureObject_t float_c1_tex = 0;
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_x = element_x * col_scale;
  float float_y = element_y * row_scale;
  int int_x = floor(float_x);
  int int_y = floor(float_y);
  float_x = element_x + 1 - (int_x + 1) * inv_col_scale;
  float_y = element_y + 1 - (int_y + 1) * inv_row_scale;
  float_x = float_x <= 0 ? 0.f : float_x - floor(float_x);
  float_y = float_y <= 0 ? 0.f : float_y - floor(float_y);
  float_x += (int_x + 0.5f);
  float_y += (int_y + 0.5f);

  if (channels == 1) {
    float value = tex2D<float>(float_c1_tex, float_x, float_y);

    float* output = (float*)(dst + element_y * dst_stride);
    output[element_x] = value;
  }
}

__global__
void resizeAreaKernel2(const uchar* src, int src_rows, int src_cols,
                       int channels, int src_stride, uchar* dst, int dst_rows,
                       int dst_cols, int dst_stride, float col_scale,
                       float row_scale, float inv_col_scale,
                       float inv_row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int int_y0 = floor(element_y * row_scale);
  int int_x0 = floor(element_x * col_scale);
  float float_y = element_y + 1 - (int_y0 + 1) * inv_row_scale;
  float float_x = element_x + 1 - (int_x0 + 1) * inv_col_scale;
  float_y = float_y <= 0 ? 0.f : float_y - floor(float_y);
  float_x = float_x <= 0 ? 0.f : float_x - floor(float_x);
  if (int_y0 < 0) {
    int_y0 = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0 = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0 = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0 = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0, src_rows);
  int buf_y[2];
  float_y = float_y * INTER_RESIZE_COEF_SCALE;
  buf_y[0] = rint(INTER_RESIZE_COEF_SCALE - float_y);
  buf_y[1] = rint(float_y);

  int int_x1 = INCREASE(int_x0, src_cols);
  int buf_x[2];
  float_x = float_x * INTER_RESIZE_COEF_SCALE;
  buf_x[0] = rint(INTER_RESIZE_COEF_SCALE - rint(float_x));
  buf_x[1] = rint(float_x);

  if (channels == 1) {
    int src_index0 = int_y0 * src_stride + int_x0;
    int src_index1 = int_y0 * src_stride + int_x1;
    int src_index2 = int_y1 * src_stride + int_x0;
    int src_index3 = int_y1 * src_stride + int_x1;

    int sum = 0;
    sum = buf_y[0] * buf_x[0] * src[src_index0] +
          buf_y[0] * buf_x[1] * src[src_index1] +
          buf_y[1] * buf_x[0] * src[src_index2] +
          buf_y[1] * buf_x[1] * src[src_index3];

    uchar* output = (uchar*)(dst + element_y * dst_stride);
    output[element_x] = (sum + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  }
  else if (channels == 3) {
    uchar3* input0 = (uchar3*)((uchar*)src + int_y0 * src_stride);
    uchar3* input1 = (uchar3*)((uchar*)src + int_y1 * src_stride);

    uchar3 values[2][2];
    values[0][0] = input0[int_x0];
    values[0][1] = input0[int_x1];
    values[1][0] = input1[int_x0];
    values[1][1] = input1[int_x1];

    uchar3* output = (uchar3*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = bilinearSample(values, buf_x[0], buf_x[1],
                                       buf_y[0], buf_y[1]);
  }
  else {
    uchar4* input0 = (uchar4*)((uchar*)src + int_y0 * src_stride);
    uchar4* input1 = (uchar4*)((uchar*)src + int_y1 * src_stride);

    uchar4 values[2][2];
    values[0][0] = input0[int_x0];
    values[0][1] = input0[int_x1];
    values[1][0] = input1[int_x0];
    values[1][1] = input1[int_x1];

    uchar4* output = (uchar4*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = bilinearSample(values, buf_x[0], buf_x[1],
                                       buf_y[0], buf_y[1]);
  }
}

__global__
void resizeAreaKernel2(const float* src, int src_rows, int src_cols,
                       int channels, int src_stride, float* dst, int dst_rows,
                       int dst_cols, int dst_stride, double col_scale,
                       float row_scale, float inv_col_scale,
                       float inv_row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int int_y0 = floor(element_y * row_scale);
  int int_x0 = floor(element_x * col_scale);
  float float_y = element_y + 1 - (int_y0 + 1) * inv_row_scale;
  float float_x = element_x + 1 - (int_x0 + 1) * inv_col_scale;
  float_y = float_y <= 0 ? 0.f : float_y - floor(float_y);
  float_x = float_x <= 0 ? 0.f : float_x - floor(float_x);
  if (int_y0 < 0) {
    int_y0 = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0 = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0 = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0 = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0,src_rows);
  float buf_y[2];
  buf_y[0] = 1.f - float_y;
  buf_y[1] = 1.f - buf_y[0];

  int int_x1 = INCREASE(int_x0,src_cols);
  float buf_x[2];
  buf_x[0] = 1.f - float_x;
  buf_x[1] = 1.f - buf_x[0];

  if (channels == 1) {
    int index = int_y0 * src_stride;
    float src0 = src[index + int_x0];
    float src1 = src[index + int_x1];
    float value0 = buf_y[0] * buf_x[0] * src0;
    float value1 = buf_y[0] * buf_x[1] * src1;
    float sum = 0.f;
    sum += value0 + value1;

    index = int_y1 * src_stride;
    src0 = src[index + int_x0];
    src1 = src[index + int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0 + value1;

    float* output = (float*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
  else if (channels == 3) {
    int index = int_y0 * src_stride;
    float3 src0 = ((float3*)(src + index))[int_x0];
    float3 src1 = ((float3*)(src + index))[int_x1];
    float3 value0 = buf_y[0] * buf_x[0] * src0;
    float3 value1 = buf_y[0] * buf_x[1] * src1;
    float3 sum = make_float3(0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = ((float3*)(src + index))[int_x0];
    src1 = ((float3*)(src + index))[int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    float3* output = (float3*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
  else {
    int index = int_y0 * src_stride;
    float4 src0 = ((float4*)(src + index))[int_x0];
    float4 src1 = ((float4*)(src + index))[int_x1];
    float4 value0 = buf_y[0] * buf_x[0] * src0;
    float4 value1 = buf_y[0] * buf_x[1] * src1;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = ((float4*)(src + index))[int_x0];
    src1 = ((float4*)(src + index))[int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    float4* output = (float4*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
}

RetCode resize(const uchar* src, int src_rows, int src_cols, int channels,
               int src_stride, uchar* dst, int dst_rows, int dst_cols,
               int dst_stride, InterpolationType interpolation,
               cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT ||
             interpolation == INTERPOLATION_AREA);

  cudaError_t code;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, src_rows * src_stride * sizeof(uchar),
                        cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int block_x = 32;
  int block_y = 16;
  if (interpolation == INTERPOLATION_NEAREST_POINT) {
    block_y = 4;
  }
  dim3 block(block_x, block_y);
  dim3 grid;
  grid.x = (dst_cols + block_x -1) / block_x;
  grid.y = (dst_rows + block_y - 1) / block_y;

  float col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;
  float inv_col_scale = 1.f / col_scale;
  float inv_row_scale = 1.f / row_scale;

  size_t texture_alignment = 32;
  size_t src_pitch = src_stride * sizeof(uchar);

  if (interpolation == INTERPOLATION_LINEAR) {
    if (channels == 1 && (src_pitch & (texture_alignment - 1)) == 0) {
      cudaResourceDesc resDesc;
      cudaTextureDesc texDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = (void*)src;
      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar>();
      resDesc.res.pitch2D.width = src_cols;
      resDesc.res.pitch2D.height = src_rows;
      resDesc.res.pitch2D.pitchInBytes = src_stride;

      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeNormalizedFloat;
      texDesc.normalizedCoords = false;

      code = cudaCreateTextureObject(&uchar_c1_tex, &resDesc, &texDesc, nullptr);

      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
        return RC_DEVICE_RUNTIME_ERROR;
      }

      resizeLinearTextureKernel<<<grid, block, 0, stream>>>(dst, dst_rows,
          dst_cols, channels, dst_stride, col_scale, row_scale);

      cudaDestroyTextureObject(uchar_c1_tex);
    }
    else if (channels == 4 && (src_pitch & (texture_alignment - 1)) == 0) {
      cudaResourceDesc resDesc;
      cudaTextureDesc texDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = (void*)src;
      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
      resDesc.res.pitch2D.width = src_cols;
      resDesc.res.pitch2D.height = src_rows;
      resDesc.res.pitch2D.pitchInBytes = src_stride;

      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeNormalizedFloat;
      texDesc.normalizedCoords = false;

      code = cudaCreateTextureObject(&uchar_c4_tex, &resDesc, &texDesc, nullptr);

      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
        return RC_DEVICE_RUNTIME_ERROR;
      }

      resizeLinearTextureKernel<<<grid, block, 0, stream>>>(dst, dst_rows,
          dst_cols, channels, dst_stride, col_scale, row_scale);
    }
    else {
      resizeLinearKernel<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
          channels, src_stride, dst, dst_rows, dst_cols, dst_stride, col_scale,
          row_scale);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      resizeNearestPointKernel<uchar, uchar><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
          dst_stride, col_scale, row_scale);
    }
    else if (channels == 3) {
      resizeNearestPointKernel<uchar, uchar3><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
          dst_stride, col_scale, row_scale);
    }
    else {
      resizeNearestPointKernel<uchar, uchar4><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
          dst_stride, col_scale, row_scale);
    }
  }
  else if (interpolation == INTERPOLATION_AREA) {
    if (src_cols > dst_cols && src_rows > dst_rows) {
      if (src_cols % dst_cols == 0 && src_rows % dst_rows == 0) {
        if (channels == 1) {
          resizeAreaC1Kernel0<uchar><<<grid, block, 0, stream>>>(src, src_rows,
              src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else if (channels == 3) {
          resizeAreaC3Kernel0<uchar, uchar3><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else {
          resizeAreaC4Kernel0<uchar, uchar4><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
      }
      else {
        if (channels == 1) {
          resizeAreaC1Kernel1<uchar><<<grid, block, 0, stream>>>(src, src_rows,
              src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else if (channels == 3) {
          resizeAreaC3Kernel1<uchar, uchar3><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else {
          resizeAreaC4Kernel1<uchar, uchar4><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
      }
    }
    else {
      if (channels == 1 && (src_pitch & (texture_alignment - 1)) == 0) {
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = (void*)src;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar>();
        resDesc.res.pitch2D.width = src_cols;
        resDesc.res.pitch2D.height = src_rows;
        resDesc.res.pitch2D.pitchInBytes = src_stride;

        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = false;

        code = cudaCreateTextureObject(&uchar_c1_tex, &resDesc, &texDesc, nullptr);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
          return RC_DEVICE_RUNTIME_ERROR;
        }

        resizeAreaTextureKernel<<<grid, block, 0, stream>>>(dst, dst_rows,
            dst_cols, channels, dst_stride, col_scale, row_scale, inv_col_scale,
            inv_row_scale);
      }
      else if (channels == 4 && (src_pitch & (texture_alignment - 1)) == 0) {
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = (void*)src;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
        resDesc.res.pitch2D.width = src_cols;
        resDesc.res.pitch2D.height = src_rows;
        resDesc.res.pitch2D.pitchInBytes = src_stride;

        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = false;

        code = cudaCreateTextureObject(&uchar_c4_tex, &resDesc, &texDesc, nullptr);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
          return RC_DEVICE_RUNTIME_ERROR;
        }

        resizeAreaTextureKernel<<<grid, block, 0, stream>>>(dst, dst_rows,
            dst_cols, channels, dst_stride, col_scale, row_scale, inv_col_scale,
            inv_row_scale);
      }
      else {
        resizeAreaKernel2<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
            channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
            col_scale, row_scale, inv_col_scale, inv_row_scale);
      }
    }
  }
  else {
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode resize(const float* src, int src_rows, int src_cols, int channels,
               int src_stride, float* dst, int dst_rows, int dst_cols,
               int dst_stride, InterpolationType interpolation,
               cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT ||
             interpolation == INTERPOLATION_AREA);

  cudaError_t code;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, src_rows * src_stride * sizeof(float),
                        cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int block_x = 32;
  int block_y = 16;
  if (interpolation == INTERPOLATION_LINEAR ||
    interpolation == INTERPOLATION_NEAREST_POINT) {
    block_y = 4;
  }
  dim3 block(block_x, block_y);
  dim3 grid;
  grid.x = (dst_cols + block_x - 1) / block_x;
  grid.y = (dst_rows + block_y - 1) / block_y;

  double col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;
  float inv_col_scale = 1.f / col_scale;
  float inv_row_scale = 1.f / row_scale;

  size_t texture_alignment = 32;
  size_t src_pitch = src_stride * sizeof(float);

  if (interpolation == INTERPOLATION_LINEAR) {
    if (channels == 1 && (src_pitch & (texture_alignment - 1)) == 0) {
      cudaResourceDesc resDesc;
      cudaTextureDesc texDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = (void*)src;
      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
      resDesc.res.pitch2D.width = src_cols;
      resDesc.res.pitch2D.height = src_rows;
      resDesc.res.pitch2D.pitchInBytes = src_stride * sizeof(float);

      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = false;

      code = cudaCreateTextureObject(&float_c1_tex, &resDesc, &texDesc, nullptr);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
        return RC_DEVICE_RUNTIME_ERROR;
      }

      resizeLinearTextureKernel<<<grid, block, 0, stream>>>(dst, dst_rows,
          dst_cols, channels, dst_stride, col_scale, row_scale);
    }
    else {
      resizeLinearKernel<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
          channels, src_stride, dst, dst_rows, dst_cols, dst_stride, col_scale,
          row_scale);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      resizeNearestPointKernel<float, float><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
          dst_stride, col_scale, row_scale);
    }
    else if (channels == 3) {
      resizeNearestPointKernel<float, float3><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
          dst_stride, col_scale, row_scale);
    }
    else {
      resizeNearestPointKernel<float, float4><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
          dst_stride, col_scale, row_scale);
    }
  }
  else if (interpolation == INTERPOLATION_AREA) {
    if (src_cols > dst_cols && src_rows > dst_rows) {
      if (src_cols % dst_cols == 0 && src_rows % dst_rows == 0) {
        if (channels == 1) {
          resizeAreaC1Kernel0<float><<<grid, block, 0, stream>>>(src, src_rows,
              src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else if (channels == 3) {
          resizeAreaC3Kernel0<float, float3><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else {
          resizeAreaC4Kernel0<float, float4><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
      }
      else {
        if (channels == 1) {
          resizeAreaC1Kernel1<float><<<grid, block, 0, stream>>>(src, src_rows,
              src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else if (channels == 3) {
          resizeAreaC3Kernel1<float, float3><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
        else {
          resizeAreaC4Kernel1<float, float4><<<grid, block, 0, stream>>>(src,
              src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
              dst_stride, col_scale, row_scale);
        }
      }
    }
    else {
      if (channels == 1 && (src_pitch & (texture_alignment - 1)) == 0) {
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = (void*)src;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        resDesc.res.pitch2D.width = src_cols;
        resDesc.res.pitch2D.height = src_rows;
        resDesc.res.pitch2D.pitchInBytes = src_stride * sizeof(float);

        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;

        code = cudaCreateTextureObject(&float_c1_tex, &resDesc, &texDesc, nullptr);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
          return RC_DEVICE_RUNTIME_ERROR;
        }

        resizeAreaTextureKernel<<<grid, block, 0, stream>>>(dst, dst_rows,
            dst_cols, channels, dst_stride, col_scale, row_scale, inv_col_scale,
            inv_row_scale);
      }
      else {
        resizeAreaKernel2<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
            channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
            col_scale, row_scale, inv_col_scale, inv_row_scale);
      }
    }
  }
  else {
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Resize<uchar, 1>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const uchar* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         uchar* outData,
                         InterpolationType interpolation) {
  RetCode code = resize(inData, inHeight, inWidth, 1, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, interpolation,
                        stream);

  return code;
}

template <>
RetCode Resize<uchar, 3>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const uchar* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         uchar* outData,
                         InterpolationType interpolation) {
  RetCode code = resize(inData, inHeight, inWidth, 3, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, interpolation,
                        stream);

  return code;
}

template <>
RetCode Resize<uchar, 4>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const uchar* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         uchar* outData,
                         InterpolationType interpolation) {
  RetCode code = resize(inData, inHeight, inWidth, 4, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, interpolation,
                        stream);

  return code;
}

template <>
RetCode Resize<float, 1>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const float* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         float* outData,
                         InterpolationType interpolation) {
  RetCode code = resize(inData, inHeight, inWidth, 1, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, interpolation,
                        stream);

  return code;
}

template <>
RetCode Resize<float, 3>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const float* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         float* outData,
                         InterpolationType interpolation) {
  RetCode code = resize(inData, inHeight, inWidth, 3, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, interpolation,
                        stream);

  return code;
}

template <>
RetCode Resize<float, 4>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const float* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         float* outData,
                         InterpolationType interpolation) {
  RetCode code = resize(inData, inHeight, inWidth, 4, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, interpolation,
                        stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

