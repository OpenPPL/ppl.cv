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

#define FILTER2DU8C1KERNEL(interpolate)                                      \
__kernel                                                                     \
void filter2DU8C1##interpolate##Kernel(                                      \
    const global uchar* src, int rows, int cols, int src_stride,             \
    const global float* src_kernel, int radius, global uchar* dst,           \
    int dst_stride, float delta) {                                           \
  int element_x, element_y;                                                  \
  element_x = get_global_id(0) * 4;                                          \
  element_y = get_global_id(1);                                              \
  if (element_x >= cols || element_y >= rows)                                \
    return;                                                                  \
  int origin_x = element_x - radius;                                         \
  int origin_y = element_y - radius;                                         \
  int top_x = element_x + radius;                                            \
  int top_y = element_y + radius;                                            \
  int data_index, kernel_index = 0;                                          \
  const global uchar* input;                                                 \
  uchar4 value;                                                              \
  float4 sum = (float4)(0.0f);                                               \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0) * 4);                             \
  if (get_group_id(0) <= data_index)                                         \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0) * 4);                    \
  if (get_group_id(0) >= data_index)                                         \
    isnt_border_block = false;                                               \
  if (isnt_border_block) {                                                   \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global uchar*)((uchar*)src + data_index * src_stride);  \
      input = input + origin_x;                                              \
      for (int j = origin_x; j <= top_x; j++) {                              \
        value = vload4(0, input);                                            \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
        input++;                                                             \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else {                                                                     \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global uchar*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        data_index = interpolate(cols, radius, j);                           \
        value.x = input[data_index];                                         \
        data_index = interpolate(cols, radius, j + 1);                       \
        value.y = input[data_index];                                         \
        data_index = interpolate(cols, radius, j + 2);                       \
        value.z = input[data_index];                                         \
        data_index = interpolate(cols, radius, j + 3);                       \
        value.w = input[data_index];                                         \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (delta != 0.f) {                                                        \
    sum += (float4)(delta);                                                  \
  }                                                                          \
  dst = (global uchar*)((uchar*)dst + element_y * dst_stride);               \
  if (element_x < cols - 3) {                                                \
    dst = dst + element_x;                                                   \
    vstore4(convert_uchar4_sat(sum), 0, dst);                                \
  }                                                                          \
  else {                                                                     \
    dst[element_x] = convert_uchar_sat(sum.x);                               \
    if (element_x < cols - 1) {                                              \
      dst[element_x + 1] = convert_uchar_sat(sum.y);                         \
    }                                                                        \
    if (element_x < cols - 2) {                                              \
      dst[element_x + 2] = convert_uchar_sat(sum.z);                         \
    }                                                                        \
  }                                                                          \
}

#define FILTER2DF32C1KERNEL(interpolate)                                     \
__kernel                                                                     \
void filter2DF32C1##interpolate##Kernel(                                     \
    const global float* src, int rows, int cols, int src_stride,             \
    const global float* src_kernel, int radius, global float* dst,           \
    int dst_stride, float delta) {                                           \
  int element_x, element_y;                                                  \
  element_x = get_global_id(0) * 4;                                          \
  element_y = get_global_id(1);                                              \
  if (element_x >= cols || element_y >= rows)                                \
    return;                                                                  \
  int origin_x = element_x - radius;                                         \
  int origin_y = element_y - radius;                                         \
  int top_x = element_x + radius;                                            \
  int top_y = element_y + radius;                                            \
  int data_index, kernel_index = 0;                                          \
  const global float* input;                                                 \
  float4 value;                                                              \
  float4 sum = (float4)(0.0f);                                               \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0) * 4);                             \
  if (get_group_id(0) <= data_index)                                         \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0) * 4);                    \
  if (get_group_id(0) >= data_index)                                         \
    isnt_border_block = false;                                               \
  if (isnt_border_block) {                                                   \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global float*)((uchar*)src + data_index * src_stride);  \
      input = input + origin_x;                                              \
      for (int j = origin_x; j <= top_x; j++) {                              \
        value = vload4(0, input);                                            \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
        input++;                                                             \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else {                                                                     \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global float*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        data_index = interpolate(cols, radius, j);                           \
        value.x = input[data_index];                                         \
        data_index = interpolate(cols, radius, j + 1);                       \
        value.y = input[data_index];                                         \
        data_index = interpolate(cols, radius, j + 2);                       \
        value.z = input[data_index];                                         \
        data_index = interpolate(cols, radius, j + 3);                       \
        value.w = input[data_index];                                         \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (delta != 0.f) {                                                        \
    sum += (float4)(delta);                                                  \
  }                                                                          \
  dst = (global float*)((uchar*)dst + element_y * dst_stride);               \
  if (element_x < cols - 3) {                                                \
    dst = dst + element_x;                                                   \
    vstore4(convert_float4(sum), 0, dst);                                    \
  }                                                                          \
  else {                                                                     \
    dst[element_x] = convert_float(sum.x);                                   \
    if (element_x < cols - 1) {                                              \
      dst[element_x + 1] = convert_float(sum.y);                             \
    }                                                                        \
    if (element_x < cols - 2) {                                              \
      dst[element_x + 2] = convert_float(sum.z);                             \
    }                                                                        \
  }                                                                          \
}

#define FILTER2DU8C3KERNEL(interpolate)                                      \
__kernel                                                                     \
void filter2DU8C3##interpolate##Kernel(                                      \
    const global uchar* src, int rows, int cols, int src_stride,             \
    const global float* src_kernel, int radius, global uchar* dst,           \
    int dst_stride, float delta) {                                           \
  int element_x, element_y;                                                  \
  element_x = get_global_id(0);                                              \
  element_y = get_global_id(1);                                              \
  if (element_x >= cols || element_y >= rows) {                              \
    return;                                                                  \
  }                                                                          \
  int origin_x = element_x - radius;                                         \
  int origin_y = element_y - radius;                                         \
  int top_x = element_x + radius;                                            \
  int top_y = element_y + radius;                                            \
  int data_index, kernel_index = 0;                                          \
  const global uchar* input;                                                 \
  uchar3 value;                                                              \
  float3 output = (float3)(0.0f);                                            \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (get_group_id(0) <= data_index)                                         \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (get_group_id(0) >= data_index)                                         \
    isnt_border_block = false;                                               \
  float3 sum = (float3)(0);                                                  \
  if (isnt_border_block) {                                                   \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global uchar*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        value = vload3(j, input);                                            \
        sum = sum + convert_float3(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else {                                                                     \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global uchar*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        data_index = interpolate(cols, radius, j);                           \
        value = vload3(data_index, input);                                   \
        sum = sum + convert_float3(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (delta != 0.f) {                                                        \
    sum += (float3)delta;                                                    \
  }                                                                          \
  dst = (global uchar*)((uchar*)dst + element_y * dst_stride);               \
  vstore3(convert_uchar3_sat(sum), element_x, dst);                          \
}

#define FILTER2DF32C3KERNEL(interpolate)                                     \
__kernel                                                                     \
void filter2DF32C3##interpolate##Kernel(                                     \
    const global float* src, int rows, int cols, int src_stride,             \
    const global float* src_kernel, int radius, global float* dst,           \
    int dst_stride, float delta) {                                           \
  int element_x, element_y;                                                  \
  element_x = get_global_id(0);                                              \
  element_y = get_global_id(1);                                              \
  if (element_x >= cols || element_y >= rows) {                              \
    return;                                                                  \
  }                                                                          \
  int origin_x = element_x - radius;                                         \
  int origin_y = element_y - radius;                                         \
  int top_x = element_x + radius;                                            \
  int top_y = element_y + radius;                                            \
  int data_index, kernel_index = 0;                                          \
  const global float* input;                                                 \
  float3 value;                                                              \
  float3 output = (float3)(0.0f);                                            \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (get_group_id(0) <= data_index)                                         \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (get_group_id(0) >= data_index)                                         \
    isnt_border_block = false;                                               \
  float3 sum = (float3)(0);                                                  \
  if (isnt_border_block) {                                                   \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global float*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        value = vload3(j, input);                                            \
        sum = sum + convert_float3(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else {                                                                     \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global float*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        data_index = interpolate(cols, radius, j);                           \
        value = vload3(data_index, input);                                   \
        sum = sum + convert_float3(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (delta != 0.f) {                                                        \
    sum += (float3)delta;                                                    \
  }                                                                          \
  dst = (global float*)((uchar*)dst + element_y * dst_stride);               \
  vstore3(convert_float3(sum), element_x, dst);                              \
}

#define FILTER2DU8C4KERNEL(interpolate)                                      \
__kernel                                                                     \
void filter2DU8C4##interpolate##Kernel(                                      \
    const global uchar* src, int rows, int cols, int src_stride,             \
    const global float* src_kernel, int radius, global uchar* dst,           \
    int dst_stride, float delta) {                                           \
  int element_x, element_y;                                                  \
  element_x = get_global_id(0);                                              \
  element_y = get_global_id(1);                                              \
  if (element_x >= cols || element_y >= rows) {                              \
    return;                                                                  \
  }                                                                          \
  int origin_x = element_x - radius;                                         \
  int origin_y = element_y - radius;                                         \
  int top_x = element_x + radius;                                            \
  int top_y = element_y + radius;                                            \
  int data_index, kernel_index = 0;                                          \
  const global uchar* input;                                                 \
  uchar4 value;                                                              \
  float4 output = (float4)(0.0f);                                            \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (get_group_id(0) <= data_index)                                         \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (get_group_id(0) >= data_index)                                         \
    isnt_border_block = false;                                               \
  float4 sum = (float4)(0);                                                  \
  if (isnt_border_block) {                                                   \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global uchar*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        value = vload4(j, input);                                            \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else {                                                                     \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global uchar*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        data_index = interpolate(cols, radius, j);                           \
        value = vload4(data_index, input);                                   \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (delta != 0.f) {                                                        \
    sum += (float4)delta;                                                    \
  }                                                                          \
  dst = (global uchar*)((uchar*)dst + element_y * dst_stride);               \
  vstore4(convert_uchar4_sat(sum), element_x, dst);                          \
}

#define FILTER2DF32C4KERNEL(interpolate)                                     \
__kernel                                                                     \
void filter2DF32C4##interpolate##Kernel(                                     \
    const global float* src, int rows, int cols, int src_stride,             \
    const global float* src_kernel, int radius, global float* dst,           \
    int dst_stride, float delta) {                                           \
  int element_x, element_y;                                                  \
  element_x = get_global_id(0);                                              \
  element_y = get_global_id(1);                                              \
  if (element_x >= cols || element_y >= rows) {                              \
    return;                                                                  \
  }                                                                          \
  int origin_x = element_x - radius;                                         \
  int origin_y = element_y - radius;                                         \
  int top_x = element_x + radius;                                            \
  int top_y = element_y + radius;                                            \
  int data_index, kernel_index = 0;                                          \
  const global float* input;                                                 \
  float4 value;                                                              \
  float4 output = (float4)(0.0f);                                            \
  bool isnt_border_block = true;                                             \
  data_index = radius / (get_local_size(0));                                 \
  if (get_group_id(0) <= data_index)                                         \
    isnt_border_block = false;                                               \
  data_index = (cols - radius) / (get_local_size(0));                        \
  if (get_group_id(0) >= data_index)                                         \
    isnt_border_block = false;                                               \
  float4 sum = (float4)(0);                                                  \
  if (isnt_border_block) {                                                   \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global float*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        value = vload4(j, input);                                            \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  else {                                                                     \
    for (int i = origin_y; i <= top_y; i++) {                                \
      data_index = interpolate(rows, radius, i);                             \
      input = (const global float*)((uchar*)src + data_index * src_stride);  \
      for (int j = origin_x; j <= top_x; j++) {                              \
        data_index = interpolate(cols, radius, j);                           \
        value = vload4(data_index, input);                                   \
        sum = sum + convert_float4(value) * src_kernel[kernel_index];        \
        kernel_index++;                                                      \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (delta != 0.f) {                                                        \
    sum += (float4)delta;                                                    \
  }                                                                          \
  dst = (global float*)((uchar*)dst + element_y * dst_stride);               \
  vstore4(convert_float4(sum), element_x, dst);                              \
}


#if defined(FILTER2D_interpolateReplicateBorderC1U8)
FILTER2DU8C1KERNEL(interpolateReplicateBorder)
#endif
#if defined(FILTER2D_interpolateReplicateBorderC1F32)
FILTER2DF32C1KERNEL(interpolateReplicateBorder)
#endif
#if defined(FILTER2D_interpolateReplicateBorderC3U8)
FILTER2DU8C3KERNEL(interpolateReplicateBorder)
#endif
#if defined(FILTER2D_interpolateReplicateBorderC3F32)
FILTER2DF32C3KERNEL(interpolateReplicateBorder)
#endif
#if defined(FILTER2D_interpolateReplicateBorderC4U8)
FILTER2DU8C4KERNEL(interpolateReplicateBorder)
#endif
#if defined(FILTER2D_interpolateReplicateBorderC4F32)
FILTER2DF32C4KERNEL(interpolateReplicateBorder)
#endif



#if defined(FILTER2D_interpolateReflectBorderC1U8)
FILTER2DU8C1KERNEL(interpolateReflectBorder)
#endif
#if defined(FILTER2D_interpolateReflectBorderC1F32)
FILTER2DF32C1KERNEL(interpolateReflectBorder)
#endif
#if defined(FILTER2D_interpolateReflectBorderC3U8)
FILTER2DU8C3KERNEL(interpolateReflectBorder)
#endif
#if defined(FILTER2D_interpolateReflectBorderC3F32)
FILTER2DF32C3KERNEL(interpolateReflectBorder)
#endif
#if defined(FILTER2D_interpolateReflectBorderC4U8)
FILTER2DU8C4KERNEL(interpolateReflectBorder)
#endif
#if defined(FILTER2D_interpolateReflectBorderC4F32)
FILTER2DF32C4KERNEL(interpolateReflectBorder)
#endif



#if defined(FILTER2D_interpolateReflect101BorderC1U8)
FILTER2DU8C1KERNEL(interpolateReflect101Border)
#endif
#if defined(FILTER2D_interpolateReflect101BorderC1F32)
FILTER2DF32C1KERNEL(interpolateReflect101Border)
#endif
#if defined(FILTER2D_interpolateReflect101BorderC3U8)
FILTER2DU8C3KERNEL(interpolateReflect101Border)
#endif
#if defined(FILTER2D_interpolateReflect101BorderC3F32)
FILTER2DF32C3KERNEL(interpolateReflect101Border)
#endif
#if defined(FILTER2D_interpolateReflect101BorderC4U8)
FILTER2DU8C4KERNEL(interpolateReflect101Border)
#endif
#if defined(FILTER2D_interpolateReflect101BorderC4F32)
FILTER2DF32C4KERNEL(interpolateReflect101Border)
#endif