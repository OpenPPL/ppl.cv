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

#if defined(COPYMAKEBORDER_U8C1) || defined(ALL_KERNELS)
__kernel
void copyMakeBorderU8Kernel0(global const uchar* src, int rows, int cols,
                             int src_stride, global uchar* dst, int dst_stride,
                             int top, int left, enum BorderType border_type,
                             uchar border_value) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= (rows + (top << 1)) || index_x >= (cols + (left << 1))) {
    return;
  }

  int src_x0 = index_x - left;
  int src_x1 = src_x0 + 1;
  int src_x2 = src_x0 + 2;
  int src_x3 = src_x0 + 3;
  int src_y = element_y - top;
  if (border_type == BORDER_CONSTANT) {
    src_x0 = interpolateConstantBorder(cols, left, src_x0);
    src_x1 = interpolateConstantBorder(cols, left, src_x1);
    src_x2 = interpolateConstantBorder(cols, left, src_x2);
    src_x3 = interpolateConstantBorder(cols, left, src_x3);
    src_y = interpolateConstantBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x0 = interpolateReplicateBorder(cols, left, src_x0);
    src_x1 = interpolateReplicateBorder(cols, left, src_x1);
    src_x2 = interpolateReplicateBorder(cols, left, src_x2);
    src_x3 = interpolateReplicateBorder(cols, left, src_x3);
    src_y = interpolateReplicateBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT) {
    src_x0 = interpolateReflectBorder(cols, left, src_x0);
    src_x1 = interpolateReflectBorder(cols, left, src_x1);
    src_x2 = interpolateReflectBorder(cols, left, src_x2);
    src_x3 = interpolateReflectBorder(cols, left, src_x3);
    src_y = interpolateReflectBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_WRAP) {
    src_x0 = interpolateWarpBorder(cols, left, src_x0);
    src_x1 = interpolateWarpBorder(cols, left, src_x1);
    src_x2 = interpolateWarpBorder(cols, left, src_x2);
    src_x3 = interpolateWarpBorder(cols, left, src_x3);
    src_y = interpolateWarpBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT_101) {
    src_x0 = interpolateReflect101Border(cols, left, src_x0);
    src_x1 = interpolateReflect101Border(cols, left, src_x1);
    src_x2 = interpolateReflect101Border(cols, left, src_x2);
    src_x3 = interpolateReflect101Border(cols, left, src_x3);
    src_y = interpolateReflect101Border(rows, top, src_y);
  }
  else {
    src_x0 = INT_MAX;
    src_x1 = INT_MAX;
    src_x2 = INT_MAX;
    src_x3 = INT_MAX;
    src_y = INT_MAX;
  }

  uchar4 value;
  global uchar* data = src + src_y * src_stride;
  if (border_type != BORDER_CONSTANT) {
    if (src_x0 + 3 == src_x3) {
      value = vload4(0, data + src_x0);
    }
    else {
      value.x = data[src_x0];
      value.y = data[src_x1];
      value.z = data[src_x2];
      value.w = data[src_x3];
    }
  }
  else {
    if (src_y != -1) {
      if (src_x0 != -1 && src_x3 != -1) {
        value = vload4(0, data + src_x0);
      }
      else {
        if (src_x0 != -1) {
          value.x = data[src_x0];
        }
        else {
          value.x = border_value;
        }
        if (src_x1 != -1) {
          value.y = data[src_x1];
        }
        else {
          value.y = border_value;
        }
        if (src_x2 != -1) {
          value.z = data[src_x2];
        }
        else {
          value.z = border_value;
        }
        if (src_x3 != -1) {
          value.w = data[src_x3];
        }
        else {
          value.w = border_value;
        }
      }
    }
    else {
      value.x = border_value;
      value.y = border_value;
      value.z = border_value;
      value.w = border_value;
    }
  }

  data = dst + element_y * dst_stride;
  cols += (left << 1);
  if (index_x < cols - 3) {
    vstore4(value, element_x, data);
  }
  else {
    data[index_x] = value.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = value.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = value.z;
    }
  }
}
#endif

#if defined(COPYMAKEBORDER_U8C3) || defined(ALL_KERNELS)
uchar3 makeUchar3(uchar value) {
  return (uchar3)(value, value, value);
}

__kernel
void copyMakeBorderU8Kernel1(global const uchar* src, int rows, int cols,
                             int src_stride, global uchar* dst, int dst_stride,
                             int top, int left, enum BorderType border_type,
                             uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= (rows + (top << 1)) || element_x >= (cols + (left << 1))) {
    return;
  }

  int src_x = element_x - left;
  int src_y = element_y - top;
  if (border_type == BORDER_CONSTANT) {
    src_x = interpolateConstantBorder(cols, left, src_x);
    src_y = interpolateConstantBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = interpolateReplicateBorder(cols, left, src_x);
    src_y = interpolateReplicateBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT) {
    src_x = interpolateReflectBorder(cols, left, src_x);
    src_y = interpolateReflectBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_WRAP) {
    src_x = interpolateWarpBorder(cols, left, src_x);
    src_y = interpolateWarpBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT_101) {
    src_x = interpolateReflect101Border(cols, left, src_x);
    src_y = interpolateReflect101Border(rows, top, src_y);
  }
  else {
    src_x = INT_MAX;
    src_y = INT_MAX;
  }

  uchar3 value;
  global uchar* data = src + src_y * src_stride;
  if (border_type != BORDER_CONSTANT) {
    value = vload3(src_x, data);
  }
  else {
    if (src_x != -1 && src_y != -1) {
      value = vload3(src_x, data);
    }
    else {
      value = makeUchar3(border_value);
    }
  }

  data = dst + element_y * dst_stride;
  vstore3(value, element_x, data);
}
#endif

#if defined(COPYMAKEBORDER_U8C4) || defined(ALL_KERNELS)
uchar4 makeUchar4(uchar value) {
  return (uchar4)(value, value, value, value);
}

__kernel
void copyMakeBorderU8Kernel2(global const uchar* src, int rows, int cols,
                             int src_stride, global uchar* dst, int dst_stride,
                             int top, int left, enum BorderType border_type,
                             uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= (rows + (top << 1)) || element_x >= (cols + (left << 1))) {
    return;
  }

  int src_x = element_x - left;
  int src_y = element_y - top;
  if (border_type == BORDER_CONSTANT) {
    src_x = interpolateConstantBorder(cols, left, src_x);
    src_y = interpolateConstantBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = interpolateReplicateBorder(cols, left, src_x);
    src_y = interpolateReplicateBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT) {
    src_x = interpolateReflectBorder(cols, left, src_x);
    src_y = interpolateReflectBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_WRAP) {
    src_x = interpolateWarpBorder(cols, left, src_x);
    src_y = interpolateWarpBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT_101) {
    src_x = interpolateReflect101Border(cols, left, src_x);
    src_y = interpolateReflect101Border(rows, top, src_y);
  }
  else {
    src_x = INT_MAX;
    src_y = INT_MAX;
  }

  uchar4 value;
  global uchar* data = src + src_y * src_stride;
  if (border_type != BORDER_CONSTANT) {
    value = vload4(src_x, data);
  }
  else {
    if (src_x != -1 && src_y != -1) {
      value = vload4(src_x, data);
    }
    else {
      value = makeUchar4(border_value);
    }
  }

  data = dst + element_y * dst_stride;
  vstore4(value, element_x, data);
}
#endif

#if defined(COPYMAKEBORDER_F32C1) || defined(ALL_KERNELS)
__kernel
void copyMakeBorderF32Kernel0(global const float* src, int rows, int cols,
                              int src_stride, global float* dst, int dst_stride,
                              int top, int left, enum BorderType border_type,
                              float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= (rows + (top << 1)) || element_x >= (cols + (left << 1))) {
    return;
  }

  int src_x = element_x - left;
  int src_y = element_y - top;
  if (border_type == BORDER_CONSTANT) {
    src_x = interpolateConstantBorder(cols, left, src_x);
    src_y = interpolateConstantBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = interpolateReplicateBorder(cols, left, src_x);
    src_y = interpolateReplicateBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT) {
    src_x = interpolateReflectBorder(cols, left, src_x);
    src_y = interpolateReflectBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_WRAP) {
    src_x = interpolateWarpBorder(cols, left, src_x);
    src_y = interpolateWarpBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT_101) {
    src_x = interpolateReflect101Border(cols, left, src_x);
    src_y = interpolateReflect101Border(rows, top, src_y);
  }
  else {
    src_x = INT_MAX;
    src_y = INT_MAX;
  }

  float value;
  global float* data = (global float*)((global uchar*)src + src_y * src_stride);
  if (border_type != BORDER_CONSTANT) {
    value = data[src_x];
  }
  else {
    if (src_x != -1 && src_y != -1) {
      value = data[src_x];
    }
    else {
      value = border_value;
    }
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  data[element_x] = value;
}
#endif

#if defined(COPYMAKEBORDER_F32C3) || defined(ALL_KERNELS)
float3 makeFloat3(float value) {
  return (float3)(value, value, value);
}

__kernel
void copyMakeBorderF32Kernel1(global const float* src, int rows, int cols,
                              int src_stride, global float* dst, int dst_stride,
                              int top, int left, enum BorderType border_type,
                              float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= (rows + (top << 1)) || element_x >= (cols + (left << 1))) {
    return;
  }

  int src_x = element_x - left;
  int src_y = element_y - top;
  if (border_type == BORDER_CONSTANT) {
    src_x = interpolateConstantBorder(cols, left, src_x);
    src_y = interpolateConstantBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = interpolateReplicateBorder(cols, left, src_x);
    src_y = interpolateReplicateBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT) {
    src_x = interpolateReflectBorder(cols, left, src_x);
    src_y = interpolateReflectBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_WRAP) {
    src_x = interpolateWarpBorder(cols, left, src_x);
    src_y = interpolateWarpBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT_101) {
    src_x = interpolateReflect101Border(cols, left, src_x);
    src_y = interpolateReflect101Border(rows, top, src_y);
  }
  else {
    src_x = INT_MAX;
    src_y = INT_MAX;
  }

  float3 value;
  global float* data = (global float*)((global uchar*)src + src_y * src_stride);
  if (border_type != BORDER_CONSTANT) {
    value = vload3(src_x, data);
  }
  else {
    if (src_x != -1 && src_y != -1) {
      value = vload3(src_x, data);
    }
    else {
      value = makeFloat3(border_value);
    }
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore3(value, element_x, data);
}
#endif

#if defined(COPYMAKEBORDER_F32C4) || defined(ALL_KERNELS)
float4 makeFloat4(float value) {
  return (float4)(value, value, value, value);
}

__kernel
void copyMakeBorderF32Kernel2(global const float* src, int rows, int cols,
                              int src_stride, global float* dst, int dst_stride,
                              int top, int left, enum BorderType border_type,
                              float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= (rows + (top << 1)) || element_x >= (cols + (left << 1))) {
    return;
  }

  int src_x = element_x - left;
  int src_y = element_y - top;
  if (border_type == BORDER_CONSTANT) {
    src_x = interpolateConstantBorder(cols, left, src_x);
    src_y = interpolateConstantBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = interpolateReplicateBorder(cols, left, src_x);
    src_y = interpolateReplicateBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT) {
    src_x = interpolateReflectBorder(cols, left, src_x);
    src_y = interpolateReflectBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_WRAP) {
    src_x = interpolateWarpBorder(cols, left, src_x);
    src_y = interpolateWarpBorder(rows, top, src_y);
  }
  else if (border_type == BORDER_REFLECT_101) {
    src_x = interpolateReflect101Border(cols, left, src_x);
    src_y = interpolateReflect101Border(rows, top, src_y);
  }
  else {
    src_x = INT_MAX;
    src_y = INT_MAX;
  }

  float4 value;
  global float* data = (global float*)((global uchar*)src + src_y * src_stride);
  if (border_type != BORDER_CONSTANT) {
    value = vload4(src_x, data);
  }
  else {
    if (src_x != -1 && src_y != -1) {
      value = vload4(src_x, data);
    }
    else {
      value = makeFloat4(border_value);
    }
  }

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore4(value, element_x, data);
}
#endif