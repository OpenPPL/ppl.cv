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

#include "kerneltypes.h"

#define BOXFILTERF32U8C1KERNEL(interpolate)                                  \
__kernel                                                                     \
void boxfilterF32U8##interpolate##C1Kernel(                                  \
    global const float* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global uchar* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x * 2, index_y = element_y * 2;                      \
  src = (global const float*)((global uchar*)src + src_offset);              \
  dst = (global uchar*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const float*)((global uchar*)src + index_y * src_stride);    \
  int remain_cols = cols - index_x, remain_rows = rows - index_y;            \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float2 input_value[2];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0) * 2);                             \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0) * 2);                    \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  global const float* src_temp;                                              \
  for (int i = 0; i < min(remain_rows, 2); i++) {                            \
    ((float2*)input_value + i)[0] = (float2)(0);                             \
    src_temp = src;                                                          \
    if (isnt_border_block) {                                                 \
      src_temp += bottom;                                                    \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float2*)input_value + i)[0] +=                                     \
            convert_float2(vload2(0, src_temp));                             \
        src_temp += 1;                                                       \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      float2 value;                                                          \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        value.x = convert_float(src_temp[data_index]);                       \
        data_index = interpolate(cols, radius, j + 1);                       \
        value.y = convert_float(src_temp[data_index]);                       \
        ;                                                                    \
        ((float2*)input_value + i)[0] += value;                              \
      }                                                                      \
    }                                                                        \
    src = (global const float*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 2); i++) {                          \
      ((float2*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 2) {                                                    \
    if (remain_cols >= 2) {                                                  \
      uchar2 output_value[2];                                                \
      output_value[0] =                                                      \
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));  \
      output_value[1] =                                                      \
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));  \
      for (int k = 0; k < 2; k++) {                                          \
        vstore2(output_value[k], element_y, dst);                            \
        dst = (global uchar*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
    else if (remain_cols == 1) {                                             \
      uchar2 output_value[1];                                                \
      output_value[0] =                                                      \
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));  \
      for (int k = 0; k < 1; k++) {                                          \
        vstore2(output_value[k], element_y, dst);                            \
        dst = (global uchar*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else if (remain_rows == 1) {                                               \
    if (remain_cols >= 2) {                                                  \
      uchar output_value[2];                                                 \
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));        \
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));        \
      for (int k = 0; k < 2; k++) {                                          \
        int offset = element_y * 2;                                          \
        dst[offset] = output_value[k];                                       \
        dst = (global uchar*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
    else if (remain_cols == 1) {                                             \
      uchar output_value[1];                                                 \
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));        \
      for (int k = 0; k < 1; k++) {                                          \
        int offset = element_y * 2;                                          \
        dst[offset] = output_value[k];                                       \
        dst = (global uchar*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERU8F32C1KERNEL(interpolate)                                  \
__kernel                                                                     \
void boxfilterU8F32##interpolate##C1Kernel(                                  \
    global const uchar* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global float* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x * 2, index_y = element_y * 2;                      \
  src = (global const uchar*)((global uchar*)src + src_offset);              \
  dst = (global float*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);    \
  int remain_cols = cols - index_x, remain_rows = rows - index_y;            \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float2 input_value[2];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0) * 2);                             \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0) * 2);                    \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  global const uchar* src_temp;                                              \
  for (int i = 0; i < min(remain_rows, 2); i++) {                            \
    ((float2*)input_value + i)[0] = (float2)(0);                             \
    src_temp = src;                                                          \
    if (isnt_border_block) {                                                 \
      src_temp += bottom;                                                    \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float2*)input_value + i)[0] +=                                     \
            convert_float2(vload2(0, src_temp));                             \
        src_temp += 1;                                                       \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      float2 value;                                                          \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        value.x = convert_float(src_temp[data_index]);                       \
        data_index = interpolate(cols, radius, j + 1);                       \
        value.y = convert_float(src_temp[data_index]);                       \
        ;                                                                    \
        ((float2*)input_value + i)[0] += value;                              \
      }                                                                      \
    }                                                                        \
    src = (global const uchar*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 2); i++) {                          \
      ((float2*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 2) {                                                    \
    if (remain_cols >= 2) {                                                  \
      float2 output_value[2];                                                \
      output_value[0] =                                                      \
          convert_float2((float2)(input_value[0].x, input_value[1].x));      \
      output_value[1] =                                                      \
          convert_float2((float2)(input_value[0].y, input_value[1].y));      \
      for (int k = 0; k < 2; k++) {                                          \
        vstore2(output_value[k], element_y, dst);                            \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
    else if (remain_cols == 1) {                                             \
      float2 output_value[1];                                                \
      output_value[0] =                                                      \
          convert_float2((float2)(input_value[0].x, input_value[1].x));      \
      for (int k = 0; k < 1; k++) {                                          \
        vstore2(output_value[k], element_y, dst);                            \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else if (remain_rows == 1) {                                               \
    if (remain_cols >= 2) {                                                  \
      float output_value[2];                                                 \
      output_value[0] = convert_float((float)(input_value[0].x));            \
      output_value[1] = convert_float((float)(input_value[0].y));            \
      for (int k = 0; k < 2; k++) {                                          \
        int offset = element_y * 2;                                          \
        dst[offset] = output_value[k];                                       \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
    else if (remain_cols == 1) {                                             \
      float output_value[1];                                                 \
      output_value[0] = convert_float((float)(input_value[0].x));            \
      for (int k = 0; k < 1; k++) {                                          \
        int offset = element_y * 2;                                          \
        dst[offset] = output_value[k];                                       \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERF32F32C1KERNEL(interpolate)                                   \
__kernel                                                                     \
void boxfilterF32F32##interpolate##C1Kernel(                                 \
    global const float* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global float* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x * 2, index_y = element_y * 2;                      \
  src = (global const float*)((global uchar*)src + src_offset);              \
  dst = (global float*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const float*)((global uchar*)src + index_y * src_stride);    \
  int remain_cols = cols - index_x, remain_rows = rows - index_y;            \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float2 input_value[2];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0) * 2);                             \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0) * 2);                    \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  global const float* src_temp;                                              \
  for (int i = 0; i < min(remain_rows, 2); i++) {                            \
    ((float2*)input_value + i)[0] = (float2)(0);                             \
    src_temp = src;                                                          \
    if (isnt_border_block) {                                                 \
      src_temp += bottom;                                                    \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float2*)input_value + i)[0] +=                                     \
            convert_float2(vload2(0, src_temp));                             \
        src_temp += 1;                                                       \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      float2 value;                                                          \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        value.x = convert_float(src_temp[data_index]);                       \
        data_index = interpolate(cols, radius, j + 1);                       \
        value.y = convert_float(src_temp[data_index]);                       \
        ;                                                                    \
        ((float2*)input_value + i)[0] += value;                              \
      }                                                                      \
    }                                                                        \
    src = (global const float*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 2); i++) {                          \
      ((float2*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 2) {                                                    \
    if (remain_cols >= 2) {                                                  \
      float2 output_value[2];                                                \
      output_value[0] =                                                      \
          convert_float2((float2)(input_value[0].x, input_value[1].x));      \
      output_value[1] =                                                      \
          convert_float2((float2)(input_value[0].y, input_value[1].y));      \
      for (int k = 0; k < 2; k++) {                                          \
        vstore2(output_value[k], element_y, dst);                            \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
    else if (remain_cols == 1) {                                             \
      float2 output_value[1];                                                \
      output_value[0] =                                                      \
          convert_float2((float2)(input_value[0].x, input_value[1].x));      \
      for (int k = 0; k < 1; k++) {                                          \
        vstore2(output_value[k], element_y, dst);                            \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else if (remain_rows == 1) {                                               \
    if (remain_cols >= 2) {                                                  \
      float output_value[2];                                                 \
      output_value[0] = convert_float((float)(input_value[0].x));            \
      output_value[1] = convert_float((float)(input_value[0].y));            \
      for (int k = 0; k < 2; k++) {                                          \
        int offset = element_y * 2;                                          \
        dst[offset] = output_value[k];                                       \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
    else if (remain_cols == 1) {                                             \
      float output_value[1];                                                 \
      output_value[0] = convert_float((float)(input_value[0].x));            \
      for (int k = 0; k < 1; k++) {                                          \
        int offset = element_y * 2;                                          \
        dst[offset] = output_value[k];                                       \
        dst = (global float*)((global uchar*)dst + dst_stride);              \
      }                                                                      \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERU8F32C3KERNEL(interpolate)                                  \
__kernel                                                                     \
void boxfilterU8F32##interpolate##C3Kernel(                                  \
    global const uchar* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global float* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x, index_y = element_y * 1;                          \
  src = (global const uchar*)((global uchar*)src + src_offset);              \
  dst = (global float*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);    \
  int remain_rows = rows - index_y;                                          \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float3 input_value[1];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  for (int i = 0; i < min(remain_rows, 1); i++) {                            \
    ((float3*)input_value + i)[0] = (float3)(0);                             \
    if (isnt_border_block) {                                                 \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float3*)input_value + i)[0] += convert_float3(vload3(j, src));     \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        ((float3*)input_value + i)[0] +=                                     \
            convert_float3(vload3(data_index, src));                         \
      }                                                                      \
    }                                                                        \
    src = (global const uchar*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 1); i++) {                          \
      ((float3*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 1) {                                                    \
    for (int i = 0; i < 1; i++) {                                            \
      vstore3(convert_float3(input_value[i]), index_y + i, dst);             \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERF32U8C3KERNEL(interpolate)                                  \
__kernel                                                                     \
void boxfilterF32U8##interpolate##C3Kernel(                                  \
    global const float* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global uchar* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x, index_y = element_y * 1;                          \
  src = (global const float*)((global uchar*)src + src_offset);              \
  dst = (global uchar*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const float*)((global uchar*)src + index_y * src_stride);    \
  int remain_rows = rows - index_y;                                          \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float3 input_value[1];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  for (int i = 0; i < min(remain_rows, 1); i++) {                            \
    ((float3*)input_value + i)[0] = (float3)(0);                             \
    if (isnt_border_block) {                                                 \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float3*)input_value + i)[0] += convert_float3(vload3(j, src));     \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        ((float3*)input_value + i)[0] +=                                     \
            convert_float3(vload3(data_index, src));                         \
      }                                                                      \
    }                                                                        \
    src = (global const float*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 1); i++) {                          \
      ((float3*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 1) {                                                    \
    for (int i = 0; i < 1; i++) {                                            \
      vstore3(convert_uchar3_sat(input_value[i]), index_y + i, dst);         \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERF32F32C3KERNEL(interpolate)                                 \
__kernel                                                                     \
void boxfilterF32F32##interpolate##C3Kernel(                                 \
    global const float* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global float* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x, index_y = element_y * 1;                          \
  src = (global const float*)((global uchar*)src + src_offset);              \
  dst = (global float*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const float*)((global uchar*)src + index_y * src_stride);    \
  int remain_rows = rows - index_y;                                          \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float3 input_value[1];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  for (int i = 0; i < min(remain_rows, 1); i++) {                            \
    ((float3*)input_value + i)[0] = (float3)(0);                             \
    if (isnt_border_block) {                                                 \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float3*)input_value + i)[0] += convert_float3(vload3(j, src));     \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        ((float3*)input_value + i)[0] +=                                     \
            convert_float3(vload3(data_index, src));                         \
      }                                                                      \
    }                                                                        \
    src = (global const float*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 1); i++) {                          \
      ((float3*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 1) {                                                    \
    for (int i = 0; i < 1; i++) {                                            \
      vstore3(convert_float3(input_value[i]), index_y + i, dst);             \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERU8F32C4KERNEL(interpolate)                                  \
__kernel                                                                     \
void boxfilterU8F32##interpolate##C4Kernel(                                  \
    global const uchar* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global float* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x, index_y = element_y * 1;                          \
  src = (global const uchar*)((global uchar*)src + src_offset);              \
  dst = (global float*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);    \
  int remain_rows = rows - index_y;                                          \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float4 input_value[1];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  for (int i = 0; i < min(remain_rows, 1); i++) {                            \
    ((float4*)input_value + i)[0] = (float4)(0);                             \
    if (isnt_border_block) {                                                 \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float4*)input_value + i)[0] += convert_float4(vload4(j, src));     \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        ((float4*)input_value + i)[0] +=                                     \
            convert_float4(vload4(data_index, src));                         \
      }                                                                      \
    }                                                                        \
    src = (global const uchar*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 1); i++) {                          \
      ((float4*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 1) {                                                    \
    for (int i = 0; i < 1; i++) {                                            \
      vstore4(convert_float4(input_value[i]), index_y + i, dst);             \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERF32U8C4KERNEL(interpolate)                                  \
__kernel                                                                     \
void boxfilterF32U8##interpolate##C4Kernel(                                  \
    global const float* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global uchar* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x, index_y = element_y * 1;                          \
  src = (global const float*)((global uchar*)src + src_offset);              \
  dst = (global uchar*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const float*)((global uchar*)src + index_y * src_stride);    \
  int remain_rows = rows - index_y;                                          \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float4 input_value[1];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  for (int i = 0; i < min(remain_rows, 1); i++) {                            \
    ((float4*)input_value + i)[0] = (float4)(0);                             \
    if (isnt_border_block) {                                                 \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float4*)input_value + i)[0] += convert_float4(vload4(j, src));     \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        ((float4*)input_value + i)[0] +=                                     \
            convert_float4(vload4(data_index, src));                         \
      }                                                                      \
    }                                                                        \
    src = (global const float*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 1); i++) {                          \
      ((float4*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 1) {                                                    \
    for (int i = 0; i < 1; i++) {                                            \
      vstore4(convert_uchar4_sat(input_value[i]), index_y + i, dst);         \
    }                                                                        \
  }                                                                          \
}

#define BOXFILTERF32F32C4KERNEL(interpolate)                                 \
__kernel                                                                     \
void boxfilterF32F32##interpolate##C4Kernel(                                 \
    global const float* src, int src_offset, int rows, int cols, int radius, \
    int src_stride, global float* dst, int dst_stride, int is_symmetric,     \
    int normalize, float weight, int dst_offset) {                           \
  int element_x = get_global_id(0);                                          \
  int element_y = get_global_id(1);                                          \
  int group_x = get_group_id(0);                                             \
  int group_y = get_group_id(1);                                             \
  int index_x = element_x, index_y = element_y * 1;                          \
  src = (global const float*)((global uchar*)src + src_offset);              \
  dst = (global float*)((global uchar*)dst + dst_offset);                    \
  if (index_x >= cols || index_y >= rows) {                                  \
    return;                                                                  \
  }                                                                          \
  src = (global const float*)((global uchar*)src + index_y * src_stride);    \
  int remain_rows = rows - index_y;                                          \
  int bottom = index_x - radius;                                             \
  int top = index_x + radius;                                                \
  int data_index;                                                            \
  if (!is_symmetric) {                                                       \
    top -= 1;                                                                \
  }                                                                          \
  float4 input_value[1];                                                     \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (group_x <= data_index)                                                 \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (group_x >= data_index)                                                 \
    isnt_border_block = false;                                               \
  for (int i = 0; i < min(remain_rows, 1); i++) {                            \
    ((float4*)input_value + i)[0] = (float4)(0);                             \
    if (isnt_border_block) {                                                 \
      for (int j = bottom; j <= top; j++) {                                  \
        ((float4*)input_value + i)[0] += convert_float4(vload4(j, src));     \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      for (int j = bottom; j <= top; j++) {                                  \
        data_index = interpolate(cols, radius, j);                           \
        ((float4*)input_value + i)[0] +=                                     \
            convert_float4(vload4(data_index, src));                         \
      }                                                                      \
    }                                                                        \
    src = (global const float*)((global uchar*)src + src_stride);            \
  }                                                                          \
  if (normalize) {                                                           \
    for (int i = 0; i < min(remain_rows, 1); i++) {                          \
      ((float4*)input_value + i)[0] *= weight;                               \
    }                                                                        \
  }                                                                          \
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);          \
  if (remain_rows >= 1) {                                                    \
    for (int i = 0; i < 1; i++) {                                            \
      vstore4(convert_float4(input_value[i]), index_y + i, dst);             \
    }                                                                        \
  }                                                                          \
}


#if defined(BOXFILTER_interpolateReplicateBorderU8C1) || defined(ALL_KERNELS)
BOXFILTERF32U8C1KERNEL(interpolateReplicateBorder)
BOXFILTERU8F32C1KERNEL(interpolateReplicateBorder)
#endif
#if defined(BOXFILTER_interpolateReplicateBorderF32C1) || defined(ALL_KERNELS)
BOXFILTERF32F32C1KERNEL(interpolateReplicateBorder)
#endif
#if defined(BOXFILTER_interpolateReplicateBorderU8C3) || defined(ALL_KERNELS)
BOXFILTERU8F32C3KERNEL(interpolateReplicateBorder)
BOXFILTERF32U8C3KERNEL(interpolateReplicateBorder)
#endif
#if defined(BOXFILTER_interpolateReplicateBorderF32C3) || defined(ALL_KERNELS)
BOXFILTERF32F32C3KERNEL(interpolateReplicateBorder)
#endif
#if defined(BOXFILTER_interpolateReplicateBorderU8C4) || defined(ALL_KERNELS)
BOXFILTERU8F32C4KERNEL(interpolateReplicateBorder)
BOXFILTERF32U8C4KERNEL(interpolateReplicateBorder)
#endif
#if defined(BOXFILTER_interpolateReplicateBorderF32C4) || defined(ALL_KERNELS)
BOXFILTERF32F32C4KERNEL(interpolateReplicateBorder)
#endif

#if defined(BOXFILTER_interpolateReflectBorderU8C1) || defined(ALL_KERNELS)
BOXFILTERF32U8C1KERNEL(interpolateReflectBorder)
BOXFILTERU8F32C1KERNEL(interpolateReflectBorder)
#endif
#if defined(BOXFILTER_interpolateReflectBorderF32C1) || defined(ALL_KERNELS)
BOXFILTERF32F32C1KERNEL(interpolateReflectBorder)
#endif
#if defined(BOXFILTER_interpolateReflectBorderU8C3) || defined(ALL_KERNELS)
BOXFILTERU8F32C3KERNEL(interpolateReflectBorder)
BOXFILTERF32U8C3KERNEL(interpolateReflectBorder)
#endif
#if defined(BOXFILTER_interpolateReflectBorderF32C3) || defined(ALL_KERNELS)
BOXFILTERF32F32C3KERNEL(interpolateReflectBorder)
#endif
#if defined(BOXFILTER_interpolateReflectBorderU8C4) || defined(ALL_KERNELS)
BOXFILTERU8F32C4KERNEL(interpolateReflectBorder)
BOXFILTERF32U8C4KERNEL(interpolateReflectBorder)
#endif
#if defined(BOXFILTER_interpolateReflectBorderF32C4) || defined(ALL_KERNELS)
BOXFILTERF32F32C4KERNEL(interpolateReflectBorder)
#endif

#if defined(BOXFILTER_interpolateReflect101BorderU8C1) || defined(ALL_KERNELS)
BOXFILTERF32U8C1KERNEL(interpolateReflect101Border)
BOXFILTERU8F32C1KERNEL(interpolateReflect101Border)
#endif
#if defined(BOXFILTER_interpolateReflect101BorderF32C1) || defined(ALL_KERNELS)
BOXFILTERF32F32C1KERNEL(interpolateReflect101Border)
#endif
#if defined(BOXFILTER_interpolateReflect101BorderU8C3) || defined(ALL_KERNELS)
BOXFILTERU8F32C3KERNEL(interpolateReflect101Border)
BOXFILTERF32U8C3KERNEL(interpolateReflect101Border)
#endif
#if defined(BOXFILTER_interpolateReflect101BorderF32C3) || defined(ALL_KERNELS)
BOXFILTERF32F32C3KERNEL(interpolateReflect101Border)
#endif
#if defined(BOXFILTER_interpolateReflect101BorderU8C4) || defined(ALL_KERNELS)
BOXFILTERU8F32C4KERNEL(interpolateReflect101Border)
BOXFILTERF32U8C4KERNEL(interpolateReflect101Border)
#endif
#if defined(BOXFILTER_interpolateReflect101BorderF32C4) || defined(ALL_KERNELS)
BOXFILTERF32F32C4KERNEL(interpolateReflect101Border)
#endif