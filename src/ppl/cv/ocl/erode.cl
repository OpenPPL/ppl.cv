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

/********************** 2d filter for all masked kernels *********************/

#if defined(ERODE_FULLLY_MASKED_2D_U8C1) || defined(ALL_KERNELS)
__kernel
void erode2DU8Kernel0(global const uchar* src, int rows, int cols,
                      int src_stride, int radius_x, int radius_y,
                      global uchar* dst, int dst_stride,
                      enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x0 = index_x - radius_x;
  int top_x0    = index_x + radius_x;
  if (bottom_x0 < 0) {
    bottom_x0 = 0;
    constant_border0 = true;
  }
  if (top_x0 >= cols) {
    top_x0 = cols - 1;
    constant_border0 = true;
  }
  int bottom_x1 = index_x + 1 - radius_x;
  int top_x1    = index_x + 1 + radius_x;
  if (bottom_x1 < 0) {
    bottom_x1 = 0;
    constant_border1 = true;
  }
  if (top_x1 >= cols) {
    top_x1 = cols - 1;
    constant_border1 = true;
  }
  int bottom_x2 = index_x + 2 - radius_x;
  int top_x2    = index_x + 2 + radius_x;
  if (bottom_x2 < 0) {
    bottom_x2 = 0;
    constant_border2 = true;
  }
  if (top_x2 >= cols) {
    top_x2 = cols - 1;
    constant_border2 = true;
  }
  int bottom_x3 = index_x + 3 - radius_x;
  int top_x3    = index_x + 3 + radius_x;
  if (bottom_x3 < 0) {
    bottom_x3 = 0;
    constant_border3 = true;
  }
  if (top_x3 >= cols) {
    top_x3 = cols - 1;
    constant_border3 = true;
  }
  int bottom_y = element_y - radius_y;
  int top_y    = element_y + radius_y;
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border0 = true;
    constant_border1 = true;
    constant_border2 = true;
    constant_border3 = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border0 = true;
    constant_border1 = true;
    constant_border2 = true;
    constant_border3 = true;
  }

  uchar4 target;
  uchar4 result = (uchar4)(255, 255, 255, 255);
  global uchar* data = src + bottom_y * src_stride;
  int index;
  for (int i = bottom_y; i <= top_y; i++) {
    index = bottom_x0;
    while (index <= top_x3) {
      target = vload4(0, data + index);
      if (index >= bottom_x0 && index <= top_x0) {
        result.x = min(result.x, target.x);
      }
      if (index >= bottom_x1 && index <= top_x1) {
        result.y = min(result.y, target.x);
      }
      if (index >= bottom_x2 && index <= top_x2) {
        result.z = min(result.z, target.x);
      }
      if (index >= bottom_x3 && index <= top_x3) {
        result.w = min(result.w, target.x);
      }
      index++;

      if (index >= bottom_x0 && index <= top_x0) {
        result.x = min(result.x, target.y);
      }
      if (index >= bottom_x1 && index <= top_x1) {
        result.y = min(result.y, target.y);
      }
      if (index >= bottom_x2 && index <= top_x2) {
        result.z = min(result.z, target.y);
      }
      if (index >= bottom_x3 && index <= top_x3) {
        result.w = min(result.w, target.y);
      }
      index++;

      if (index >= bottom_x0 && index <= top_x0) {
        result.x = min(result.x, target.z);
      }
      if (index >= bottom_x1 && index <= top_x1) {
        result.y = min(result.y, target.z);
      }
      if (index >= bottom_x2 && index <= top_x2) {
        result.z = min(result.z, target.z);
      }
      if (index >= bottom_x3 && index <= top_x3) {
        result.w = min(result.w, target.z);
      }
      index++;

      if (index >= bottom_x0 && index <= top_x0) {
        result.x = min(result.x, target.w);
      }
      if (index >= bottom_x1 && index <= top_x1) {
        result.y = min(result.y, target.w);
      }
      if (index >= bottom_x2 && index <= top_x2) {
        result.z = min(result.z, target.w);
      }
      if (index >= bottom_x3 && index <= top_x3) {
        result.w = min(result.w, target.w);
      }
      index++;
    }
    data +=src_stride;
  }

  if (border_type == BORDER_CONSTANT) {
    if (constant_border0) {
      result.x = min(result.x, border_value);
    }

    if (constant_border1) {
      result.y = min(result.y, border_value);
    }

    if (constant_border2) {
      result.z = min(result.z, border_value);
    }

    if (constant_border3) {
      result.w = min(result.w, border_value);
    }
  }

  data = dst + element_y * dst_stride;
  if (index_x < cols - 3) {
    vstore4(result, element_x, data);
  }
  else {
    data[index_x] = result.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = result.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = result.z;
    }
  }
}
#endif

#if defined(ERODE_FULLLY_MASKED_2D_U8C3) || defined(ALL_KERNELS)
__kernel
void erode2DU8Kernel1(global const uchar* src, int rows, int cols,
                      int src_stride, int radius_x, int radius_y,
                      global uchar* dst, int dst_stride,
                      enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;
  if (bottom_x < 0) {
    bottom_x = 0;
    constant_border = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  uchar3 target;
  uchar3 result = (uchar3)(255, 255, 255);
  global uchar* data = src + bottom_y * src_stride;
  for (int i = bottom_y; i <= top_y; i++) {
    for (int j = bottom_x; j <= top_x; j++) {
      target = vload3(j, data);
      result = min(result, target);
    }
    data +=src_stride;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    uchar3 borders = (uchar3)(border_value, border_value, border_value);
    result = min(result, borders);
  }

  data = dst + element_y * dst_stride;
  vstore3(result, element_x, data);
}
#endif

#if defined(ERODE_FULLLY_MASKED_2D_U8C4) || defined(ALL_KERNELS)
__kernel
void erode2DU8Kernel2(global const uchar* src, int rows, int cols,
                      int src_stride, int radius_x, int radius_y,
                      global uchar* dst, int dst_stride,
                      enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;
  if (bottom_x < 0) {
    bottom_x = 0;
    constant_border = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  uchar4 target;
  uchar4 result = (uchar4)(255, 255, 255, 255);
  global uchar* data = src + bottom_y * src_stride;
  for (int i = bottom_y; i <= top_y; i++) {
    for (int j = bottom_x; j <= top_x; j++) {
      target = vload4(j, data);
      result = min(result, target);
    }
    data +=src_stride;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    uchar4 borders = (uchar4)(border_value, border_value, border_value,
                              border_value);
    result = min(result, borders);
  }

  data = dst + element_y * dst_stride;
  vstore4(result, element_x, data);
}
#endif

#if defined(ERODE_FULLLY_MASKED_2D_F32C1) || defined(ALL_KERNELS)
__kernel
void erode2DF32Kernel0(global const float* src, int rows, int cols,
                       int src_stride, int radius_x, int radius_y,
                       global float* dst, int dst_stride,
                       enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;
  if (bottom_x < 0) {
    bottom_x = 0;
    constant_border = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  float target;
  float result = FLT_MAX;
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y * src_stride);
  for (int i = bottom_y; i <= top_y; i++) {
    for (int j = bottom_x; j <= top_x; j++) {
      target = data[j];
      result = min(result, target);
    }
    data = (global float*)((global uchar*)data + src_stride);
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    result = min(result, border_value);
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  data[element_x] = result;
}
#endif

#if defined(ERODE_FULLLY_MASKED_2D_F32C3) || defined(ALL_KERNELS)
__kernel
void erode2DF32Kernel1(global const float* src, int rows, int cols,
                       int src_stride, int radius_x, int radius_y,
                       global float* dst, int dst_stride,
                       enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;
  if (bottom_x < 0) {
    bottom_x = 0;
    constant_border = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  float3 target;
  float3 result = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y * src_stride);
  for (int i = bottom_y; i <= top_y; i++) {
    for (int j = bottom_x; j <= top_x; j++) {
      target = vload3(j, data);
      result = min(result, target);
    }
    data = (global float*)((global uchar*)data + src_stride);
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    float3 borders = (float3)(border_value, border_value, border_value);
    result = min(result, borders);
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore3(result, element_x, data);
}
#endif

#if defined(ERODE_FULLLY_MASKED_2D_F32C4) || defined(ALL_KERNELS)
__kernel
void erode2DF32Kernel2(global const float* src, int rows, int cols,
                       int src_stride, int radius_x, int radius_y,
                       global float* dst, int dst_stride,
                       enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;
  if (bottom_x < 0) {
    bottom_x = 0;
    constant_border = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  float4 target;
  float4 result = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y * src_stride);
  for (int i = bottom_y; i <= top_y; i++) {
    for (int j = bottom_x; j <= top_x; j++) {
      target = vload4(j, data);
      result = min(result, target);
    }
    data = (global float*)((global uchar*)data + src_stride);
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    float4 borders = (float4)(border_value, border_value, border_value,
                              border_value);
    result = min(result, borders);
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore4(result, element_x, data);
}
#endif

/********* separate 2d filter for all masked kernels with 4 vectors **********/

#if defined(ERODE_FULLLY_MASKED_SEP2D_U8_C1) || defined(ALL_KERNELS)
__kernel
void erodeRowU8Kernel0(global const uchar* src, int rows, int cols,
                       int src_stride, int radius_x, global uchar* dst,
                       int dst_stride, enum BorderType border_type,
                       uchar border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x0 = index_x - radius_x;
  int top_x0    = index_x + radius_x;
  if (bottom_x0 < 0) {
    bottom_x0 = 0;
    constant_border0 = true;
  }
  if (top_x0 >= cols) {
    top_x0 = cols - 1;
    constant_border0 = true;
  }
  int bottom_x1 = index_x + 1 - radius_x;
  int top_x1    = index_x + 1 + radius_x;
  if (bottom_x1 < 0) {
    bottom_x1 = 0;
    constant_border1 = true;
  }
  if (top_x1 >= cols) {
    top_x1 = cols - 1;
    constant_border1 = true;
  }
  int bottom_x2 = index_x + 2 - radius_x;
  int top_x2    = index_x + 2 + radius_x;
  if (bottom_x2 < 0) {
    bottom_x2 = 0;
    constant_border2 = true;
  }
  if (top_x2 >= cols) {
    top_x2 = cols - 1;
    constant_border2 = true;
  }
  int bottom_x3 = index_x + 3 - radius_x;
  int top_x3    = index_x + 3 + radius_x;
  if (bottom_x3 < 0) {
    bottom_x3 = 0;
    constant_border3 = true;
  }
  if (top_x3 >= cols) {
    top_x3 = cols - 1;
    constant_border3 = true;
  }

  int index = bottom_x0;
  uchar4 target;
  uchar4 result = (uchar4)(255, 255, 255, 255);
  global uchar* data = src + element_y * src_stride;
  while (index <= top_x3) {
    target = vload4(0, data + index);
    if (index >= bottom_x0 && index <= top_x0) {
      result.x = min(result.x, target.x);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result.y = min(result.y, target.x);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result.z = min(result.z, target.x);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result.w = min(result.w, target.x);
    }
    index++;

    if (index >= bottom_x0 && index <= top_x0) {
      result.x = min(result.x, target.y);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result.y = min(result.y, target.y);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result.z = min(result.z, target.y);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result.w = min(result.w, target.y);
    }
    index++;

    if (index >= bottom_x0 && index <= top_x0) {
      result.x = min(result.x, target.z);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result.y = min(result.y, target.z);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result.z = min(result.z, target.z);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result.w = min(result.w, target.z);
    }
    index++;

    if (index >= bottom_x0 && index <= top_x0) {
      result.x = min(result.x, target.w);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result.y = min(result.y, target.w);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result.z = min(result.z, target.w);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result.w = min(result.w, target.w);
    }
    index++;
  }

  if (border_type == BORDER_CONSTANT) {
    if (constant_border0) {
      result.x = min(result.x, border_value);
    }

    if (constant_border1) {
      result.y = min(result.y, border_value);
    }

    if (constant_border2) {
      result.z = min(result.z, border_value);
    }

    if (constant_border3) {
      result.w = min(result.w, border_value);
    }
  }

  data = dst + element_y * dst_stride;
  if (index_x < cols - 3) {
    vstore4(result, element_x, data);
  }
  else {
    data[index_x] = result.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = result.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = result.z;
    }
  }
}

__kernel
void erodeColU8Kernel0(global const uchar* src, int rows, int cols,
                       int src_stride, int radius_y, global uchar* dst,
                       int dst_stride, enum BorderType border_type,
                       uchar border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border = false;
  int bottom_y = element_y - radius_y;
  int top_y    = element_y + radius_y;
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  uchar4 target;
  uchar4 result = (uchar4)(255, 255, 255, 255);
  global uchar* data = src + bottom_y * src_stride;
  for (int i = bottom_y; i <= top_y; i++) {
    target = vload4(element_x, data);
    result = min(result, target);
    data +=src_stride;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    uchar4 borders = (uchar4)(border_value, border_value, border_value,
                              border_value);
    result = min(result, borders);
  }

  data = dst + element_y * dst_stride;
  if (index_x < cols - 3) {
    vstore4(result, element_x, data);
  }
  else {
    data[index_x] = result.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = result.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = result.z;
    }
  }
}
#endif

#if defined(ERODE_FULLLY_MASKED_SEP2D_U8_C3) || defined(ALL_KERNELS)
__kernel
void erodeRowU8Kernel1(global const uchar* src, int rows, int cols,
                       int src_stride, int radius_x, global uchar* dst,
                       int dst_stride, enum BorderType border_type,
                       uchar border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x0 = index_x - radius_x;
  int top_x0    = index_x + radius_x;
  if (bottom_x0 < 0) {
    bottom_x0 = 0;
    constant_border0 = true;
  }
  if (top_x0 >= cols) {
    top_x0 = cols - 1;
    constant_border0 = true;
  }
  int bottom_x1 = index_x + 1 - radius_x;
  int top_x1    = index_x + 1 + radius_x;
  if (bottom_x1 < 0) {
    bottom_x1 = 0;
    constant_border1 = true;
  }
  if (top_x1 >= cols) {
    top_x1 = cols - 1;
    constant_border1 = true;
  }
  int bottom_x2 = index_x + 2 - radius_x;
  int top_x2    = index_x + 2 + radius_x;
  if (bottom_x2 < 0) {
    bottom_x2 = 0;
    constant_border2 = true;
  }
  if (top_x2 >= cols) {
    top_x2 = cols - 1;
    constant_border2 = true;
  }
  int bottom_x3 = index_x + 3 - radius_x;
  int top_x3    = index_x + 3 + radius_x;
  if (bottom_x3 < 0) {
    bottom_x3 = 0;
    constant_border3 = true;
  }
  if (top_x3 >= cols) {
    top_x3 = cols - 1;
    constant_border3 = true;
  }

  int index = bottom_x0;
  uchar3 target;
  uchar3 result0 = (uchar3)(255, 255, 255);
  uchar3 result1 = (uchar3)(255, 255, 255);
  uchar3 result2 = (uchar3)(255, 255, 255);
  uchar3 result3 = (uchar3)(255, 255, 255);
  global uchar* data = src + element_y * src_stride;
  while (index <= top_x3) {
    target = vload3(index, data);
    if (index >= bottom_x0 && index <= top_x0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result3 = min(result3, target);
    }
    index++;
  }

  if (border_type == BORDER_CONSTANT) {
    uchar3 borders = (uchar3)(border_value, border_value, border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  data = dst + element_y * dst_stride;
  if (index_x < cols - 3) {
    vstore3(result0, index_x, data);
    vstore3(result1, index_x + 1, data);
    vstore3(result2, index_x + 2, data);
    vstore3(result3, index_x + 3, data);
  }
  else {
    vstore3(result0, index_x, data);
    if (index_x < cols - 1) {
      vstore3(result1, index_x + 1, data);
    }
    if ((index_x < cols - 2)) {
      vstore3(result2, index_x + 2, data);
    }
  }
}

__kernel
void erodeColU8Kernel1(global const uchar* src, int rows, int cols,
                       int src_stride, int radius_y, global uchar* dst,
                       int dst_stride, enum BorderType border_type,
                       uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_y = element_y << 2;
  if (index_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_y0 = index_y - radius_y;
  int top_y0    = index_y + radius_y;
  if (bottom_y0 < 0) {
    bottom_y0 = 0;
    constant_border0 = true;
  }
  if (top_y0 >= rows) {
    top_y0 = rows - 1;
    constant_border0 = true;
  }
  int bottom_y1 = index_y + 1 - radius_y;
  int top_y1    = index_y + 1 + radius_y;
  if (bottom_y1 < 0) {
    bottom_y1 = 0;
    constant_border1 = true;
  }
  if (top_y1 >= rows) {
    top_y1 = rows - 1;
    constant_border1 = true;
  }
  int bottom_y2 = index_y + 2 - radius_y;
  int top_y2    = index_y + 2 + radius_y;
  if (bottom_y2 < 0) {
    bottom_y2 = 0;
    constant_border2 = true;
  }
  if (top_y2 >= rows) {
    top_y2 = rows - 1;
    constant_border2 = true;
  }
  int bottom_y3 = index_y + 3 - radius_y;
  int top_y3    = index_y + 3 + radius_y;
  if (bottom_y3 < 0) {
    bottom_y3 = 0;
    constant_border3 = true;
  }
  if (top_y3 >= rows) {
    top_y3 = rows - 1;
    constant_border3 = true;
  }

  int index = bottom_y0;
  uchar3 target;
  uchar3 result0 = (uchar3)(255, 255, 255);
  uchar3 result1 = (uchar3)(255, 255, 255);
  uchar3 result2 = (uchar3)(255, 255, 255);
  uchar3 result3 = (uchar3)(255, 255, 255);
  global uchar* data = src + bottom_y0 * src_stride;
  while (index <= top_y3) {
    target = vload3(element_x, data);
    if (index >= bottom_y0 && index <= top_y0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_y1 && index <= top_y1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_y2 && index <= top_y2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_y3 && index <= top_y3) {
      result3 = min(result3, target);
    }
    index++;
    data +=src_stride;
  }

  if (border_type == BORDER_CONSTANT) {
    uchar3 borders = (uchar3)(border_value, border_value, border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  if (index_y < rows - 3) {
    data = dst + index_y * dst_stride;
    vstore3(result0, element_x, data);
    data +=dst_stride;
    vstore3(result1, element_x, data);
    data +=dst_stride;
    vstore3(result2, element_x, data);
    data +=dst_stride;
    vstore3(result3, element_x, data);
  }
  else {
    data = dst + index_y * dst_stride;
    vstore3(result0, element_x, data);
    if (index_y < rows - 1) {
      data +=dst_stride;
      vstore3(result1, element_x, data);
    }
    if ((index_y < rows - 2)) {
      data +=dst_stride;
      vstore3(result2, element_x, data);
    }
  }
}
#endif

#if defined(ERODE_FULLLY_MASKED_SEP2D_U8_C4) || defined(ALL_KERNELS)
__kernel
void erodeRowU8Kernel2(global const uchar* src, int rows, int cols,
                       int src_stride, int radius_x, global uchar* dst,
                       int dst_stride, enum BorderType border_type,
                       uchar border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x0 = index_x - radius_x;
  int top_x0    = index_x + radius_x;
  if (bottom_x0 < 0) {
    bottom_x0 = 0;
    constant_border0 = true;
  }
  if (top_x0 >= cols) {
    top_x0 = cols - 1;
    constant_border0 = true;
  }
  int bottom_x1 = index_x + 1 - radius_x;
  int top_x1    = index_x + 1 + radius_x;
  if (bottom_x1 < 0) {
    bottom_x1 = 0;
    constant_border1 = true;
  }
  if (top_x1 >= cols) {
    top_x1 = cols - 1;
    constant_border1 = true;
  }
  int bottom_x2 = index_x + 2 - radius_x;
  int top_x2    = index_x + 2 + radius_x;
  if (bottom_x2 < 0) {
    bottom_x2 = 0;
    constant_border2 = true;
  }
  if (top_x2 >= cols) {
    top_x2 = cols - 1;
    constant_border2 = true;
  }
  int bottom_x3 = index_x + 3 - radius_x;
  int top_x3    = index_x + 3 + radius_x;
  if (bottom_x3 < 0) {
    bottom_x3 = 0;
    constant_border3 = true;
  }
  if (top_x3 >= cols) {
    top_x3 = cols - 1;
    constant_border3 = true;
  }

  int index = bottom_x0;
  uchar4 target;
  uchar4 result0 = (uchar4)(255, 255, 255, 255);
  uchar4 result1 = (uchar4)(255, 255, 255, 255);
  uchar4 result2 = (uchar4)(255, 255, 255, 255);
  uchar4 result3 = (uchar4)(255, 255, 255, 255);
  global uchar* data = src + element_y * src_stride;
  while (index <= top_x3) {
    target = vload4(index, data);
    if (index >= bottom_x0 && index <= top_x0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result3 = min(result3, target);
    }
    index++;
  }

  if (border_type == BORDER_CONSTANT) {
    uchar4 borders = (uchar4)(border_value, border_value, border_value,
                              border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  data = dst + element_y * dst_stride;
  if (index_x < cols - 3) {
    vstore4(result0, index_x, data);
    vstore4(result1, index_x + 1, data);
    vstore4(result2, index_x + 2, data);
    vstore4(result3, index_x + 3, data);
  }
  else {
    vstore4(result0, index_x, data);
    if (index_x < cols - 1) {
      vstore4(result1, index_x + 1, data);
    }
    if ((index_x < cols - 2)) {
      vstore4(result2, index_x + 2, data);
    }
  }
}

__kernel
void erodeColU8Kernel2(global const uchar* src, int rows, int cols,
                       int src_stride, int radius_y, global uchar* dst,
                       int dst_stride, enum BorderType border_type,
                       uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_y = element_y << 2;
  if (index_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_y0 = index_y - radius_y;
  int top_y0    = index_y + radius_y;
  if (bottom_y0 < 0) {
    bottom_y0 = 0;
    constant_border0 = true;
  }
  if (top_y0 >= rows) {
    top_y0 = rows - 1;
    constant_border0 = true;
  }
  int bottom_y1 = index_y + 1 - radius_y;
  int top_y1    = index_y + 1 + radius_y;
  if (bottom_y1 < 0) {
    bottom_y1 = 0;
    constant_border1 = true;
  }
  if (top_y1 >= rows) {
    top_y1 = rows - 1;
    constant_border1 = true;
  }
  int bottom_y2 = index_y + 2 - radius_y;
  int top_y2    = index_y + 2 + radius_y;
  if (bottom_y2 < 0) {
    bottom_y2 = 0;
    constant_border2 = true;
  }
  if (top_y2 >= rows) {
    top_y2 = rows - 1;
    constant_border2 = true;
  }
  int bottom_y3 = index_y + 3 - radius_y;
  int top_y3    = index_y + 3 + radius_y;
  if (bottom_y3 < 0) {
    bottom_y3 = 0;
    constant_border3 = true;
  }
  if (top_y3 >= rows) {
    top_y3 = rows - 1;
    constant_border3 = true;
  }

  int index = bottom_y0;
  uchar4 target;
  uchar4 result0 = (uchar4)(255, 255, 255, 255);
  uchar4 result1 = (uchar4)(255, 255, 255, 255);
  uchar4 result2 = (uchar4)(255, 255, 255, 255);
  uchar4 result3 = (uchar4)(255, 255, 255, 255);
  global uchar* data = src + bottom_y0 * src_stride;
  while (index <= top_y3) {
    target = vload4(element_x, data);
    if (index >= bottom_y0 && index <= top_y0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_y1 && index <= top_y1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_y2 && index <= top_y2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_y3 && index <= top_y3) {
      result3 = min(result3, target);
    }
    index++;
    data +=src_stride;
  }

  if (border_type == BORDER_CONSTANT) {
    uchar4 borders = (uchar4)(border_value, border_value, border_value,
                              border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  if (index_y < rows - 3) {
    data = dst + index_y * dst_stride;
    vstore4(result0, element_x, data);
    data +=dst_stride;
    vstore4(result1, element_x, data);
    data +=dst_stride;
    vstore4(result2, element_x, data);
    data +=dst_stride;
    vstore4(result3, element_x, data);
  }
  else {
    data = dst + index_y * dst_stride;
    vstore4(result0, element_x, data);
    if (index_y < rows - 1) {
      data +=dst_stride;
      vstore4(result1, element_x, data);
    }
    if ((index_y < rows - 2)) {
      data +=dst_stride;
      vstore4(result2, element_x, data);
    }
  }
}
#endif

#if defined(ERODE_FULLLY_MASKED_SEP2D_F32_C1) || defined(ALL_KERNELS)
__kernel
void erodeRowF32Kernel0(global const float* src, int rows, int cols,
                        int src_stride, int radius_x, global float* dst,
                        int dst_stride, enum BorderType border_type,
                        float border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x0 = index_x - radius_x;
  int top_x0    = index_x + radius_x;
  if (bottom_x0 < 0) {
    bottom_x0 = 0;
    constant_border0 = true;
  }
  if (top_x0 >= cols) {
    top_x0 = cols - 1;
    constant_border0 = true;
  }
  int bottom_x1 = index_x + 1 - radius_x;
  int top_x1    = index_x + 1 + radius_x;
  if (bottom_x1 < 0) {
    bottom_x1 = 0;
    constant_border1 = true;
  }
  if (top_x1 >= cols) {
    top_x1 = cols - 1;
    constant_border1 = true;
  }
  int bottom_x2 = index_x + 2 - radius_x;
  int top_x2    = index_x + 2 + radius_x;
  if (bottom_x2 < 0) {
    bottom_x2 = 0;
    constant_border2 = true;
  }
  if (top_x2 >= cols) {
    top_x2 = cols - 1;
    constant_border2 = true;
  }
  int bottom_x3 = index_x + 3 - radius_x;
  int top_x3    = index_x + 3 + radius_x;
  if (bottom_x3 < 0) {
    bottom_x3 = 0;
    constant_border3 = true;
  }
  if (top_x3 >= cols) {
    top_x3 = cols - 1;
    constant_border3 = true;
  }

  int index = bottom_x0;
  float target;
  float result0 = FLT_MAX;
  float result1 = FLT_MAX;
  float result2 = FLT_MAX;
  float result3 = FLT_MAX;
  global float* data = (global float*)((global uchar*)src +
                                       element_y * src_stride);
  while (index <= top_x3) {
    target = data[index];
    if (index >= bottom_x0 && index <= top_x0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result3 = min(result3, target);
    }
    index++;
  }

  if (border_type == BORDER_CONSTANT) {
    if (constant_border0) {
      result0 = min(result0, border_value);
    }
    if (constant_border1) {
      result1 = min(result1, border_value);
    }
    if (constant_border2) {
      result2 = min(result2, border_value);
    }
    if (constant_border3) {
      result3 = min(result3, border_value);
    }
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  if (index_x < cols - 3) {
    data[index_x] = result0;
    data[index_x + 1] = result1;
    data[index_x + 2] = result2;
    data[index_x + 3] = result3;
  }
  else {
    data[index_x] = result0;
    if (index_x < cols - 1) {
      data[index_x + 1] = result1;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = result2;
    }
  }
}

__kernel
void erodeColF32Kernel0(global const float* src, int rows, int cols,
                        int src_stride, int radius_y, global float* dst,
                        int dst_stride, enum BorderType border_type,
                        float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_y = element_y << 2;
  if (index_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_y0 = index_y - radius_y;
  int top_y0    = index_y + radius_y;
  if (bottom_y0 < 0) {
    bottom_y0 = 0;
    constant_border0 = true;
  }
  if (top_y0 >= rows) {
    top_y0 = rows - 1;
    constant_border0 = true;
  }
  int bottom_y1 = index_y + 1 - radius_y;
  int top_y1    = index_y + 1 + radius_y;
  if (bottom_y1 < 0) {
    bottom_y1 = 0;
    constant_border1 = true;
  }
  if (top_y1 >= rows) {
    top_y1 = rows - 1;
    constant_border1 = true;
  }
  int bottom_y2 = index_y + 2 - radius_y;
  int top_y2    = index_y + 2 + radius_y;
  if (bottom_y2 < 0) {
    bottom_y2 = 0;
    constant_border2 = true;
  }
  if (top_y2 >= rows) {
    top_y2 = rows - 1;
    constant_border2 = true;
  }
  int bottom_y3 = index_y + 3 - radius_y;
  int top_y3    = index_y + 3 + radius_y;
  if (bottom_y3 < 0) {
    bottom_y3 = 0;
    constant_border3 = true;
  }
  if (top_y3 >= rows) {
    top_y3 = rows - 1;
    constant_border3 = true;
  }

  int index = bottom_y0;
  float target;
  float result0 = FLT_MAX;
  float result1 = FLT_MAX;
  float result2 = FLT_MAX;
  float result3 = FLT_MAX;
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y0 * src_stride);
  while (index <= top_y3) {
    target = data[element_x];
    if (index >= bottom_y0 && index <= top_y0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_y1 && index <= top_y1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_y2 && index <= top_y2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_y3 && index <= top_y3) {
      result3 = min(result3, target);
    }
    index++;
    data = (global float*)((global uchar*)data + src_stride);
  }

  if (border_type == BORDER_CONSTANT) {
    if (constant_border0) {
      result0 = min(result0, border_value);
    }
    if (constant_border1) {
      result1 = min(result1, border_value);
    }
    if (constant_border2) {
      result2 = min(result2, border_value);
    }
    if (constant_border3) {
      result3 = min(result3, border_value);
    }
  }

  if (index_y < rows - 3) {
    data = (global float*)((global uchar*)dst + index_y * dst_stride);
    data[element_x] = result0;
    data = (global float*)((global uchar*)data + dst_stride);
    data[element_x] = result1;
    data = (global float*)((global uchar*)data + dst_stride);
    data[element_x] = result2;
    data = (global float*)((global uchar*)data + dst_stride);
    data[element_x] = result3;
  }
  else {
    data = (global float*)((global uchar*)dst + index_y * dst_stride);
    data[element_x] = result0;
    if (index_y < rows - 1) {
      data = (global float*)((global uchar*)data + dst_stride);
      data[element_x] = result1;
    }
    if ((index_y < rows - 2)) {
      data = (global float*)((global uchar*)data + dst_stride);
      data[element_x] = result2;
    }
  }
}
#endif

#if defined(ERODE_FULLLY_MASKED_SEP2D_F32_C3) || defined(ALL_KERNELS)
__kernel
void erodeRowF32Kernel1(global const float* src, int rows, int cols,
                        int src_stride, int radius_x, global float* dst,
                        int dst_stride, enum BorderType border_type,
                        float border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x0 = index_x - radius_x;
  int top_x0    = index_x + radius_x;
  if (bottom_x0 < 0) {
    bottom_x0 = 0;
    constant_border0 = true;
  }
  if (top_x0 >= cols) {
    top_x0 = cols - 1;
    constant_border0 = true;
  }
  int bottom_x1 = index_x + 1 - radius_x;
  int top_x1    = index_x + 1 + radius_x;
  if (bottom_x1 < 0) {
    bottom_x1 = 0;
    constant_border1 = true;
  }
  if (top_x1 >= cols) {
    top_x1 = cols - 1;
    constant_border1 = true;
  }
  int bottom_x2 = index_x + 2 - radius_x;
  int top_x2    = index_x + 2 + radius_x;
  if (bottom_x2 < 0) {
    bottom_x2 = 0;
    constant_border2 = true;
  }
  if (top_x2 >= cols) {
    top_x2 = cols - 1;
    constant_border2 = true;
  }
  int bottom_x3 = index_x + 3 - radius_x;
  int top_x3    = index_x + 3 + radius_x;
  if (bottom_x3 < 0) {
    bottom_x3 = 0;
    constant_border3 = true;
  }
  if (top_x3 >= cols) {
    top_x3 = cols - 1;
    constant_border3 = true;
  }

  int index = bottom_x0;
  float3 target;
  float3 result0 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 result1 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 result2 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 result3 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  global float* data = (global float*)((global uchar*)src +
                                       element_y * src_stride);
  while (index <= top_x3) {
    target = vload3(index, data);
    if (index >= bottom_x0 && index <= top_x0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result3 = min(result3, target);
    }
    index++;
  }

  if (border_type == BORDER_CONSTANT) {
    float3 borders = (float3)(border_value, border_value, border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  if (index_x < cols - 3) {
    vstore3(result0, index_x, data);
    vstore3(result1, index_x + 1, data);
    vstore3(result2, index_x + 2, data);
    vstore3(result3, index_x + 3, data);
  }
  else {
    vstore3(result0, index_x, data);
    if (index_x < cols - 1) {
      vstore3(result1, index_x + 1, data);
    }
    if ((index_x < cols - 2)) {
      vstore3(result2, index_x + 2, data);
    }
  }
}

__kernel
void erodeColF32Kernel1(global const float* src, int rows, int cols,
                        int src_stride, int radius_y,
                        global float* dst, int dst_stride,
                        enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_y = element_y << 2;
  if (index_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_y0 = index_y - radius_y;
  int top_y0    = index_y + radius_y;
  if (bottom_y0 < 0) {
    bottom_y0 = 0;
    constant_border0 = true;
  }
  if (top_y0 >= rows) {
    top_y0 = rows - 1;
    constant_border0 = true;
  }
  int bottom_y1 = index_y + 1 - radius_y;
  int top_y1    = index_y + 1 + radius_y;
  if (bottom_y1 < 0) {
    bottom_y1 = 0;
    constant_border1 = true;
  }
  if (top_y1 >= rows) {
    top_y1 = rows - 1;
    constant_border1 = true;
  }
  int bottom_y2 = index_y + 2 - radius_y;
  int top_y2    = index_y + 2 + radius_y;
  if (bottom_y2 < 0) {
    bottom_y2 = 0;
    constant_border2 = true;
  }
  if (top_y2 >= rows) {
    top_y2 = rows - 1;
    constant_border2 = true;
  }
  int bottom_y3 = index_y + 3 - radius_y;
  int top_y3    = index_y + 3 + radius_y;
  if (bottom_y3 < 0) {
    bottom_y3 = 0;
    constant_border3 = true;
  }
  if (top_y3 >= rows) {
    top_y3 = rows - 1;
    constant_border3 = true;
  }

  global float* data = (global float*)((global uchar*)src +
                                       bottom_y0 * src_stride);
  float3 result0 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 result1 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 result2 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 result3 = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);

  float3 target;
  int index = bottom_y0;
  while (index <= top_y3) {
    target = vload3(element_x, data);
    if (index >= bottom_y0 && index <= top_y0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_y1 && index <= top_y1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_y2 && index <= top_y2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_y3 && index <= top_y3) {
      result3 = min(result3, target);
    }
    index++;
    data = (global float*)((global uchar*)data + src_stride);
  }

  if (border_type == BORDER_CONSTANT) {
    float3 borders = (float3)(border_value, border_value, border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  if (index_y < rows - 3) {
    data = (global float*)((global uchar*)dst + index_y * dst_stride);
    vstore3(result0, element_x, data);
    data = (global float*)((global uchar*)data + dst_stride);
    vstore3(result1, element_x, data);
    data = (global float*)((global uchar*)data + dst_stride);
    vstore3(result2, element_x, data);
    data = (global float*)((global uchar*)data + dst_stride);
    vstore3(result3, element_x, data);
  }
  else {
    data = (global float*)((global uchar*)dst + index_y * dst_stride);
    vstore3(result0, element_x, data);
    if (index_y < rows - 1) {
      data = (global float*)((global uchar*)data + dst_stride);
      vstore3(result1, element_x, data);
    }
    if ((index_y < rows - 2)) {
      data = (global float*)((global uchar*)data + dst_stride);
      vstore3(result2, element_x, data);
    }
  }
}
#endif

#if defined(ERODE_FULLLY_MASKED_SEP2D_F32_C4) || defined(ALL_KERNELS)
__kernel
void erodeRowF32Kernel2(global const float* src, int rows, int cols,
                        int src_stride, int radius_x, global float* dst,
                        int dst_stride, enum BorderType border_type,
                        float border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x0 = index_x - radius_x;
  int top_x0    = index_x + radius_x;
  if (bottom_x0 < 0) {
    bottom_x0 = 0;
    constant_border0 = true;
  }
  if (top_x0 >= cols) {
    top_x0 = cols - 1;
    constant_border0 = true;
  }
  int bottom_x1 = index_x + 1 - radius_x;
  int top_x1    = index_x + 1 + radius_x;
  if (bottom_x1 < 0) {
    bottom_x1 = 0;
    constant_border1 = true;
  }
  if (top_x1 >= cols) {
    top_x1 = cols - 1;
    constant_border1 = true;
  }
  int bottom_x2 = index_x + 2 - radius_x;
  int top_x2    = index_x + 2 + radius_x;
  if (bottom_x2 < 0) {
    bottom_x2 = 0;
    constant_border2 = true;
  }
  if (top_x2 >= cols) {
    top_x2 = cols - 1;
    constant_border2 = true;
  }
  int bottom_x3 = index_x + 3 - radius_x;
  int top_x3    = index_x + 3 + radius_x;
  if (bottom_x3 < 0) {
    bottom_x3 = 0;
    constant_border3 = true;
  }
  if (top_x3 >= cols) {
    top_x3 = cols - 1;
    constant_border3 = true;
  }

  int index = bottom_x0;
  float4 target;
  float4 result0 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  float4 result1 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  float4 result2 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  float4 result3 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  global float* data = (global float*)((global uchar*)src +
                                       element_y * src_stride);
  while (index <= top_x3) {
    target = vload4(index, data);
    if (index >= bottom_x0 && index <= top_x0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_x1 && index <= top_x1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_x2 && index <= top_x2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_x3 && index <= top_x3) {
      result3 = min(result3, target);
    }
    index++;
  }

  if (border_type == BORDER_CONSTANT) {
    float4 borders = (float4)(border_value, border_value, border_value,
                              border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  if (index_x < cols - 3) {
    vstore4(result0, index_x, data);
    vstore4(result1, index_x + 1, data);
    vstore4(result2, index_x + 2, data);
    vstore4(result3, index_x + 3, data);
  }
  else {
    vstore4(result0, index_x, data);
    if (index_x < cols - 1) {
      vstore4(result1, index_x + 1, data);
    }
    if ((index_x < cols - 2)) {
      vstore4(result2, index_x + 2, data);
    }
  }
}

__kernel
void erodeColF32Kernel2(global const float* src, int rows, int cols,
                        int src_stride, int radius_y, global float* dst,
                        int dst_stride, enum BorderType border_type,
                        float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_y = element_y << 2;
  if (index_y >= rows || element_x >= cols) {
    return;
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_y0 = index_y - radius_y;
  int top_y0    = index_y + radius_y;
  if (bottom_y0 < 0) {
    bottom_y0 = 0;
    constant_border0 = true;
  }
  if (top_y0 >= rows) {
    top_y0 = rows - 1;
    constant_border0 = true;
  }
  int bottom_y1 = index_y + 1 - radius_y;
  int top_y1    = index_y + 1 + radius_y;
  if (bottom_y1 < 0) {
    bottom_y1 = 0;
    constant_border1 = true;
  }
  if (top_y1 >= rows) {
    top_y1 = rows - 1;
    constant_border1 = true;
  }
  int bottom_y2 = index_y + 2 - radius_y;
  int top_y2    = index_y + 2 + radius_y;
  if (bottom_y2 < 0) {
    bottom_y2 = 0;
    constant_border2 = true;
  }
  if (top_y2 >= rows) {
    top_y2 = rows - 1;
    constant_border2 = true;
  }
  int bottom_y3 = index_y + 3 - radius_y;
  int top_y3    = index_y + 3 + radius_y;
  if (bottom_y3 < 0) {
    bottom_y3 = 0;
    constant_border3 = true;
  }
  if (top_y3 >= rows) {
    top_y3 = rows - 1;
    constant_border3 = true;
  }

  int index = bottom_y0;
  float4 target;
  float4 result0 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  float4 result1 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  float4 result2 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  float4 result3 = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y0 * src_stride);
  while (index <= top_y3) {
    target = vload4(element_x, data);
    if (index >= bottom_y0 && index <= top_y0) {
      result0 = min(result0, target);
    }
    if (index >= bottom_y1 && index <= top_y1) {
      result1 = min(result1, target);
    }
    if (index >= bottom_y2 && index <= top_y2) {
      result2 = min(result2, target);
    }
    if (index >= bottom_y3 && index <= top_y3) {
      result3 = min(result3, target);
    }
    index++;
    data = (global float*)((global uchar*)data + src_stride);
  }

  if (border_type == BORDER_CONSTANT) {
    float4 borders = (float4)(border_value, border_value, border_value,
                              border_value);
    if (constant_border0) {
      result0 = min(result0, borders);
    }
    if (constant_border1) {
      result1 = min(result1, borders);
    }
    if (constant_border2) {
      result2 = min(result2, borders);
    }
    if (constant_border3) {
      result3 = min(result3, borders);
    }
  }

  if (index_y < rows - 3) {
    data = (global float*)((global uchar*)dst + index_y * dst_stride);
    vstore4(result0, element_x, data);
    data = (global float*)((global uchar*)data + dst_stride);
    vstore4(result1, element_x, data);
    data = (global float*)((global uchar*)data + dst_stride);
    vstore4(result2, element_x, data);
    data = (global float*)((global uchar*)data + dst_stride);
    vstore4(result3, element_x, data);
  }
  else {
    data = (global float*)((global uchar*)dst + index_y * dst_stride);
    vstore4(result0, element_x, data);
    if (index_y < rows - 1) {
      data = (global float*)((global uchar*)data + dst_stride);
      vstore4(result1, element_x, data);
    }
    if ((index_y < rows - 2)) {
      data = (global float*)((global uchar*)data + dst_stride);
      vstore4(result2, element_x, data);
    }
  }
}
#endif

/******************* 2d filter for partial masked kernels *******************/

#if defined(ERODE_PARTIALLY_MASKED_2D_U8C1) || defined(ALL_KERNELS)
__kernel
void erode2DU8Kernel3(global const uchar* src, int rows, int cols,
                      int src_stride, global const uchar* mask, int radius_x,
                      int radius_y, int kernel_x, int kernel_y,
                      global uchar* dst, int dst_stride,
                      enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;

  bool constant_border = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (mask[mask_index + j - bottom_x]) {
            constant_border = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (mask[mask_index + j - bottom_x]) {
              constant_border = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_x = 0, kernel_bottom_y = 0;
  if (bottom_x < 0) {
    bottom_x = 0;
    kernel_bottom_x = radius_x - element_x;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    kernel_bottom_y = radius_y - element_y;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  int index;
  int kernel_bottom_start = kernel_bottom_y * kernel_x + kernel_bottom_x;
  uchar target;
  uchar result = 255;
  global uchar* data = src + bottom_y * src_stride;
  for (int i = bottom_y; i <= top_y; i++) {
    index = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (mask[index]) {
        target = data[j];
        result = min(result, target);
      }
      index++;
    }
    data +=src_stride;
    kernel_bottom_start += kernel_x;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    result = min(result, border_value);
  }

  data = dst + element_y * dst_stride;
  data[element_x] = result;
}
#endif

#if defined(ERODE_PARTIALLY_MASKED_2D_U8C3) || defined(ALL_KERNELS)
__kernel
void erode2DU8Kernel4(global const uchar* src, int rows, int cols,
                      int src_stride, global const uchar* mask, int radius_x,
                      int radius_y, int kernel_x, int kernel_y,
                      global uchar* dst, int dst_stride,
                      enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;

  bool constant_border = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (mask[mask_index + j - bottom_x]) {
            constant_border = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (mask[mask_index + j - bottom_x]) {
              constant_border = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_x = 0, kernel_bottom_y = 0;
  if (bottom_x < 0) {
    bottom_x = 0;
    kernel_bottom_x = radius_x - element_x;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    kernel_bottom_y = radius_y - element_y;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  int index;
  int kernel_bottom_start = kernel_bottom_y * kernel_x + kernel_bottom_x;
  uchar3 target;
  uchar3 result = (uchar3)(255, 255, 255);
  global uchar* data = src + bottom_y * src_stride;
  for (int i = bottom_y; i <= top_y; i++) {
    index = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (mask[index]) {
        target = vload3(j, data);
        result = min(result, target);
      }
      index++;
    }
    data +=src_stride;
    kernel_bottom_start += kernel_x;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    uchar3 borders = (uchar3)(border_value, border_value, border_value);
    result = min(result, borders);
  }

  data = dst + element_y * dst_stride;
  vstore3(result, element_x, data);
}
#endif

#if defined(ERODE_PARTIALLY_MASKED_2D_U8C4) || defined(ALL_KERNELS)
__kernel
void erode2DU8Kernel5(global const uchar* src, int rows, int cols,
                      int src_stride, global const uchar* mask, int radius_x,
                      int radius_y, int kernel_x, int kernel_y,
                      global uchar* dst, int dst_stride,
                      enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;

  bool constant_border = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (mask[mask_index + j - bottom_x]) {
            constant_border = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (mask[mask_index + j - bottom_x]) {
              constant_border = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_x = 0, kernel_bottom_y = 0;
  if (bottom_x < 0) {
    bottom_x = 0;
    kernel_bottom_x = radius_x - element_x;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    kernel_bottom_y = radius_y - element_y;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  int index;
  int kernel_bottom_start = kernel_bottom_y * kernel_x + kernel_bottom_x;
  uchar4 target;
  uchar4 result = (uchar4)(255, 255, 255, 255);
  global uchar* data = src + bottom_y * src_stride;
  for (int i = bottom_y; i <= top_y; i++) {
    index = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (mask[index]) {
        target = vload4(j, data);
        result = min(result, target);
      }
      index++;
    }
    data +=src_stride;
    kernel_bottom_start += kernel_x;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    uchar4 borders = (uchar4)(border_value, border_value, border_value,
                              border_value);
    result = min(result, borders);
  }

  data = dst + element_y * dst_stride;
  vstore4(result, element_x, data);
}
#endif

#if defined(ERODE_PARTIALLY_MASKED_2D_F32C1) || defined(ALL_KERNELS)
__kernel
void erode2DF32Kernel3(global const float* src, int rows, int cols,
                       int src_stride, global const uchar* mask, int radius_x,
                       int radius_y, int kernel_x, int kernel_y,
                       global float* dst, int dst_stride,
                       enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;

  bool constant_border = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (mask[mask_index + j - bottom_x]) {
            constant_border = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (mask[mask_index + j - bottom_x]) {
              constant_border = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_x = 0, kernel_bottom_y = 0;
  if (bottom_x < 0) {
    bottom_x = 0;
    kernel_bottom_x = radius_x - element_x;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    kernel_bottom_y = radius_y - element_y;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  int index;
  int kernel_bottom_start = kernel_bottom_y * kernel_x + kernel_bottom_x;
  float target;
  float result = FLT_MAX;
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y * src_stride);
  for (int i = bottom_y; i <= top_y; i++) {
    index = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (mask[index]) {
        target = data[j];
        result = min(result, target);
      }
      index++;
    }
    data = (global float*)((global uchar*)data + src_stride);
    kernel_bottom_start += kernel_x;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    result = min(result, border_value);
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  data[element_x] = result;
}
#endif

#if defined(ERODE_PARTIALLY_MASKED_2D_F32C3) || defined(ALL_KERNELS)
__kernel
void erode2DF32Kernel4(global const float* src, int rows, int cols,
                       int src_stride, global const uchar* mask, int radius_x,
                       int radius_y, int kernel_x, int kernel_y,
                       global float* dst, int dst_stride,
                       enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;

  bool constant_border = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (mask[mask_index + j - bottom_x]) {
            constant_border = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (mask[mask_index + j - bottom_x]) {
              constant_border = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_x = 0, kernel_bottom_y = 0;
  if (bottom_x < 0) {
    bottom_x = 0;
    kernel_bottom_x = radius_x - element_x;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    kernel_bottom_y = radius_y - element_y;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  int index;
  int kernel_bottom_start = kernel_bottom_y * kernel_x + kernel_bottom_x;
  float3 target;
  float3 result = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y * src_stride);
  for (int i = bottom_y; i <= top_y; i++) {
    index = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (mask[index]) {
        target = vload3(j, data);
        result = min(result, target);
      }
      index++;
    }
    data = (global float*)((global uchar*)data + src_stride);
    kernel_bottom_start += kernel_x;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    float3 borders = (float3)(border_value, border_value, border_value);
    result = min(result, borders);
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore3(result, element_x, data);
}
#endif

#if defined(ERODE_PARTIALLY_MASKED_2D_F32C4) || defined(ALL_KERNELS)
__kernel
void erode2DF32Kernel5(global const float* src, int rows, int cols,
                       int src_stride, global const uchar* mask, int radius_x,
                       int radius_y, int kernel_x, int kernel_y,
                       global float* dst, int dst_stride,
                       enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;

  bool constant_border = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (mask[mask_index + j - bottom_x]) {
            constant_border = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (mask[mask_index + j - bottom_x]) {
              constant_border = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_x = 0, kernel_bottom_y = 0;
  if (bottom_x < 0) {
    bottom_x = 0;
    kernel_bottom_x = radius_x - element_x;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    kernel_bottom_y = radius_y - element_y;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  int index;
  int kernel_bottom_start = kernel_bottom_y * kernel_x + kernel_bottom_x;
  float4 target;
  float4 result = (float4)(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  global float* data = (global float*)((global uchar*)src +
                                       bottom_y * src_stride);
  for (int i = bottom_y; i <= top_y; i++) {
    index = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (mask[index]) {
        target = vload4(j, data);
        result = min(result, target);
      }
      index++;
    }
    data = (global float*)((global uchar*)data + src_stride);
    kernel_bottom_start += kernel_x;
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    float4 borders = (float4)(border_value, border_value, border_value,
                              border_value);
    result = min(result, borders);
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore4(result, element_x, data);
}
#endif
