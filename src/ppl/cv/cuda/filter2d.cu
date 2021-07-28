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

#include "filter2d.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns1(const uchar* src, const int* kernel, uchar* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  int value0 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      value0 += input[index1] * kernel[index2++];
    }
  }

  if (scale != 1) {
    value0 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);

  uchar* output = dst + element_y * dst_stride;
  output[element_x] = uvalue0;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns3(const uchar* src, const int* kernel, uchar* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  int value0 = 0, value1 = 0, value2 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);

  uchar* output = dst + element_y * dst_stride;
  int index_x = element_x * 3;
  output[index_x]     = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns4(const uchar* src, const int* kernel, uchar* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  int value0 = 0, value1 = 0, value2 = 0, value3 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);
  uchar uvalue3 = saturate_cast(value3);

  uchar* output = dst + element_y * dst_stride;
  int index_x = element_x << 2;
  output[index_x]     = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
  output[index_x + 3] = uvalue3;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns1(const uchar* src, const float* kernel, uchar* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  float value0 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      value0 += input[index1] * kernel[index2++];
    }
  }

  if (scale != 1) {
    value0 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);

  uchar* output = dst + element_y * dst_stride;
  output[element_x] = uvalue0;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns3(const uchar* src, const float* kernel, uchar* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);

  uchar* output = dst + element_y * dst_stride;
  int index_x = element_x * 3;
  output[index_x]     = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns4(const uchar* src, const float* kernel, uchar* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);
  uchar uvalue3 = saturate_cast(value3);

  uchar* output = dst + element_y * dst_stride;
  int index_x = element_x << 2;
  output[index_x]     = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
  output[index_x + 3] = uvalue3;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns1(const uchar* src, const int* kernel, short* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  int value0 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      value0 += input[index1] * kernel[index2++];
    }
  }

  if (scale != 1) {
    value0 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
  }

  short svalue0 = saturate_cast_i2s(value0);

  short* output = (short*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = svalue0;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns3(const uchar* src, const int* kernel, short* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  int value0 = 0, value1 = 0, value2 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  short svalue0 = saturate_cast_i2s(value0);
  short svalue1 = saturate_cast_i2s(value1);
  short svalue2 = saturate_cast_i2s(value2);

  short* output = (short*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x]     = svalue0;
  output[index_x + 1] = svalue1;
  output[index_x + 2] = svalue2;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns4(const uchar* src, const int* kernel, short* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  uchar* input;
  int index1, index2 = 0;
  int value0 = 0, value1 = 0, value2 = 0, value3 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  short svalue0 = saturate_cast_i2s(value0);
  short svalue1 = saturate_cast_i2s(value1);
  short svalue2 = saturate_cast_i2s(value2);
  short svalue3 = saturate_cast_i2s(value3);

  short* output = (short*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x]     = svalue0;
  output[index_x + 1] = svalue1;
  output[index_x + 2] = svalue2;
  output[index_x + 3] = svalue3;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns1(const float* src, const int* kernel, float* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  float* input;
  int index1, index2 = 0;
  float value0 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      value0 += input[index1] * kernel[index2++];
    }
  }

  if (scale != 1) {
    value0 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = value0;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns3(const float* src, const int* kernel, float* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  float* input;
  int index1, index2 = 0;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x]     = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns4(const float* src, const int* kernel, float* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  float* input;
  int index1, index2 = 0;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x]     = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
  output[index_x + 3] = value3;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns1(const float* src, const float* kernel, float* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  float* input;
  int index1, index2 = 0;
  float value0 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      value0 += input[index1] * kernel[index2++];
    }
  }

  if (scale != 1) {
    value0 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = value0;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns3(const float* src, const float* kernel, float* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  float* input;
  int index1, index2 = 0;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x]     = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
}

/**
 * Border type: BORDER_TYPE_REPLICATE.
 */
__global__
void filter2D_k1_cns4(const float* src, const float* kernel, float* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  float* input;
  int index1, index2 = 0;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0;
    if (i >= rows) index1 = rows - 1;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0;
      if (j >= cols) index1 = cols - 1;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x]     = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
  output[index_x + 3] = value3;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns1(const uchar* src, const int* kernel, uchar* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  int value0 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      value0 += input[index1] * kernel[index2++];
    }
  }

  value0 = value0 * scale + delta;
  uchar uvalue0 = saturate_cast(value0);

  uchar* output = dst + element_y * dst_stride;
  output[element_x] = uvalue0;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns3(const uchar* src, const int* kernel, uchar* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  int value0 = 0, value1 = 0, value2 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);

  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x] = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns4(const uchar* src, const int* kernel, uchar* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  int value0 = 0, value1 = 0, value2 = 0, value3 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);
  uchar uvalue3 = saturate_cast(value3);

  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x] = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
  output[index_x + 3] = uvalue3;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns1(const uchar* src, const float* kernel, uchar* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  float value0 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      value0 += input[index1] * kernel[index2++];
    }
  }

  value0 = value0 * scale + delta;
  uchar uvalue0 = saturate_cast(value0);

  uchar* output = dst + element_y * dst_stride;
  output[element_x] = uvalue0;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns3(const uchar* src, const float* kernel, uchar* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);

  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x] = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns4(const uchar* src, const float* kernel, uchar* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  uchar uvalue0 = saturate_cast(value0);
  uchar uvalue1 = saturate_cast(value1);
  uchar uvalue2 = saturate_cast(value2);
  uchar uvalue3 = saturate_cast(value3);

  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x] = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
  output[index_x + 3] = uvalue3;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns1(const uchar* src, const int* kernel, short* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  int value0 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      value0 += input[index1] * kernel[index2++];
    }
  }

  value0 = value0 * scale + delta;
  short uvalue0 = saturate_cast_i2s(value0);

  short* output = (short*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = uvalue0;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns3(const uchar* src, const int* kernel, short* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  int value0 = 0, value1 = 0, value2 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  short uvalue0 = saturate_cast_i2s(value0);
  short uvalue1 = saturate_cast_i2s(value1);
  short uvalue2 = saturate_cast_i2s(value2);

  short* output = (short*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x] = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns4(const uchar* src, const int* kernel, short* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  uchar* input;
  int value0 = 0, value1 = 0, value2 = 0, value3 = 0;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (uchar*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  short uvalue0 = saturate_cast_i2s(value0);
  short uvalue1 = saturate_cast_i2s(value1);
  short uvalue2 = saturate_cast_i2s(value2);
  short uvalue3 = saturate_cast_i2s(value3);

  short* output = (short*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x] = uvalue0;
  output[index_x + 1] = uvalue1;
  output[index_x + 2] = uvalue2;
  output[index_x + 3] = uvalue3;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns1(const float* src, const int* kernel, float* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  float* input;
  float value0 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      value0 += input[index1] * kernel[index2++];
    }
  }
  value0 = value0 * scale + delta;

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = value0;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns3(const float* src, const int* kernel, float* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  float* input;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x] = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns4(const float* src, const int* kernel, float* dst, int rows,
                      int cols, int src_stride, int dst_stride, int channels,
                      int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  float* input;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x] = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
  output[index_x + 3] = value3;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns1(const float* src, const float* kernel, float* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  float* input;
  float value0 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      value0 += input[index1] * kernel[index2++];
    }
  }
  value0 = value0 * scale + delta;

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = value0;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns3(const float* src, const float* kernel, float* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  float* input;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 * 3;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x * 3;
  output[index_x] = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
}

/**
 * Border type: BORDER_TYPE_DEFAULT(BORDER_TYPE_REFLECT_101).
 */
__global__
void filter2D_k2_cns4(const float* src, const float* kernel, float* dst,
                      int rows, int cols, int src_stride, int dst_stride,
                      int channels, int radius, double scale, double delta)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int index1, index2 = 0;
  float* input;
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
  for (int i = origin_y; i <= top_y; i++) {
    index1 = i;
    if (i < 0)     index1 = 0 - i;
    if (i >= rows) index1 = (rows << 1) - i - 2;
    input = (float*)((uchar*)src + index1 * src_stride);
    for (int j = origin_x; j <= top_x; j++) {
      index1 = j;
      if (j < 0)     index1 = 0 - j;
      if (j >= cols) index1 = (cols << 1) - j - 2;
      index1 = index1 << 2;
      value0 += input[index1] * kernel[index2];
      value1 += input[index1 + 1] * kernel[index2];
      value2 += input[index1 + 2] * kernel[index2];
      value3 += input[index1 + 3] * kernel[index2];
      index2++;
    }
  }

  if (scale != 1) {
    value0 *= scale;
    value1 *= scale;
    value2 *= scale;
    value3 *= scale;
  }

  if (delta != 0) {
    value0 += delta;
    value1 += delta;
    value2 += delta;
    value3 += delta;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  int index_x = element_x << 2;
  output[index_x] = value0;
  output[index_x + 1] = value1;
  output[index_x + 2] = value2;
  output[index_x + 3] = value3;
}

RetCode filter2D(const uchar* src, int rows, int cols, int channels,
                 int src_stride, uchar* dst, int dst_stride,
                 const float* kernel, int ksize, BorderType border_type,
                 cudaStream_t stream)
{
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(kernel != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_TYPE_DEFAULT ||
             border_type == BORDER_TYPE_REPLICATE);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  unsigned int radius = ksize >> 1;
  double scale = 1;
  double delta = 0;

  if (border_type == BORDER_TYPE_DEFAULT) {
    if (channels == 1) {
      filter2D_k2_cns1<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else if (channels == 3) {
      filter2D_k2_cns3<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else {
      filter2D_k2_cns4<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    if (channels == 1) {
      filter2D_k1_cns1<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else if (channels == 3) {
      filter2D_k1_cns3<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else {
      filter2D_k1_cns4<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
  }
  else {
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode filter2D(const float* src, int rows, int cols, int channels,
                 int src_stride, float* dst, int dst_stride,
                 const float* kernel, int ksize, BorderType border_type,
                 cudaStream_t stream)
{
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(kernel != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_TYPE_DEFAULT ||
             border_type == BORDER_TYPE_REPLICATE);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  unsigned int radius = ksize >> 1;
  double scale = 1;
  double delta = 0;

  if (border_type == BORDER_TYPE_DEFAULT) {
    if (channels == 1) {
      filter2D_k2_cns1<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else if (channels == 3) {
      filter2D_k2_cns3<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else {
      filter2D_k2_cns4<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    if (channels == 1) {
      filter2D_k1_cns1<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else if (channels == 3) {
      filter2D_k1_cns3<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
    else {
      filter2D_k1_cns4<<<grid, block, 0, stream>>>(src, kernel, dst, rows, cols,
                                                   src_stride, dst_stride,
                                                   channels, radius, scale,
                                                   delta);
    }
  }
  else {
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/*********************** uchar -> uchar ***********************/

template <>
RetCode Filter2D<uchar, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 1, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<uchar, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 3, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<uchar, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 4, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, border_type,
                          stream);

  return code;
}

/*********************** uchar -> short***********************/


/*********************** float -> float ***********************/

template <>
RetCode Filter2D<float, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 1, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<float, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 3, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<float, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 4, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, border_type,
                          stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
