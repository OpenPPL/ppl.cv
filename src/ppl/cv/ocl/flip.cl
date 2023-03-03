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

#if defined(FLIP_U8C1) || defined(ALL_KERNELS)
__kernel
void flipU8Kernel0(global const uchar* src, int rows, int cols, int src_stride,
                   global uchar* dst, int dst_stride, int flip_code) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  int element_y = get_global_id(1);
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  int x0, x1, x2, x3, y;
  global uchar* data;
  uchar4 value;
  if (flip_code == 0) {
    y = rows - element_y - 1;

    data = src + y * src_stride;
    value = vload4(element_x, data);
  }
  else if (flip_code > 0) {
    x0 = cols - index_x - 1;
    x1 = x0 - 1;
    x2 = x0 - 2;
    x3 = x0 - 3;

    data = src + element_y * src_stride;
    if (x3 >= 0) {
      data += x3;
      uchar4 temp_value = vload4(0, data);
      value.x = temp_value.w;
      value.y = temp_value.z;
      value.z = temp_value.y;
      value.w = temp_value.x;
    }
    else {
      value.x = data[x0];
      if (x1 >= 0) {
        value.y = data[x1];
      }
      if (x2 >= 0) {
        value.z = data[x2];
      }
      if (x3 >= 0) {
        value.w = data[x3];
      }
    }
  }
  else {
    x0 = cols - index_x - 1;
    x1 = x0 - 1;
    x2 = x0 - 2;
    x3 = x0 - 3;
    y = rows - element_y - 1;

    data = src + y * src_stride;
    if (x3 >= 0) {
      data += x3;
      uchar4 temp_value = vload4(0, data);
      value.x = temp_value.w;
      value.y = temp_value.z;
      value.z = temp_value.y;
      value.w = temp_value.x;
    }
    else {
      value.x = data[x0];
      if (x1 >= 0) {
        value.y = data[x1];
      }
      if (x2 >= 0) {
        value.z = data[x2];
      }
      if (x3 >= 0) {
        value.w = data[x3];
      }
    }
  }

  data = dst + element_y * dst_stride;
  if (index_x < dst_stride - 3) {
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

#if defined(FLIP_U8C3) || defined(ALL_KERNELS)
__kernel
void flipU8Kernel1(global const uchar* src, int rows, int cols, int src_stride,
                   global uchar* dst, int dst_stride, int flip_code) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int x, y;
  if (flip_code == 0) {
    x = element_x;
    y = rows - element_y - 1;
  }
  else if (flip_code > 0) {
    x = cols - element_x - 1;
    y = element_y;
  }
  else {
    x = cols - element_x - 1;
    y = rows - element_y - 1;
  }
  global uchar* data = src + y * src_stride;
  uchar3 value = vload3(x, data);

  data = dst + element_y * dst_stride;
  vstore3(value, element_x, data);
}
#endif

#if defined(FLIP_U8C4) || defined(ALL_KERNELS)
__kernel
void flipU8Kernel2(global const uchar* src, int rows, int cols, int src_stride,
                   global uchar* dst, int dst_stride, int flip_code) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int x, y;
  if (flip_code == 0) {
    x = element_x;
    y = rows - element_y - 1;
  }
  else if (flip_code > 0) {
    x = cols - element_x - 1;
    y = element_y;
  }
  else {
    x = cols - element_x - 1;
    y = rows - element_y - 1;
  }
  global uchar* data = src + y * src_stride;
  uchar4 value = vload4(x, data);

  data = dst + element_y * dst_stride;
  vstore4(value, element_x, data);
}
#endif

#if defined(FLIP_F32C1) || defined(ALL_KERNELS)
__kernel
void flipF32Kernel0(global const float* src, int rows, int cols, int src_stride,
                    global float* dst, int dst_stride, int flip_code) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int x, y;
  if (flip_code == 0) {
    x = element_x;
    y = rows - element_y - 1;
  }
  else if (flip_code > 0) {
    x = cols - element_x - 1;
    y = element_y;
  }
  else {
    x = cols - element_x - 1;
    y = rows - element_y - 1;
  }
  global float* data = (global float*)((global uchar*)src + y * src_stride);
  float value = data[x];

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  data[element_x] = value;
}
#endif

#if defined(FLIP_F32C3) || defined(ALL_KERNELS)
__kernel
void flipF32Kernel1(global const float* src, int rows, int cols, int src_stride,
                    global float* dst, int dst_stride, int flip_code) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int x, y;
  if (flip_code == 0) {
    x = element_x;
    y = rows - element_y - 1;
  }
  else if (flip_code > 0) {
    x = cols - element_x - 1;
    y = element_y;
  }
  else {
    x = cols - element_x - 1;
    y = rows - element_y - 1;
  }
  global float* data = (global float*)((global uchar*)src + y * src_stride);
  float3 value = vload3(x, data);

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore3(value, element_x, data);
}
#endif

#if defined(FLIP_F32C4) || defined(ALL_KERNELS)
__kernel
void flipF32Kernel2(global const float* src, int rows, int cols, int src_stride,
                    global float* dst, int dst_stride, int flip_code) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int x, y;
  if (flip_code == 0) {
    x = element_x;
    y = rows - element_y - 1;
  }
  else if (flip_code > 0) {
    x = cols - element_x - 1;
    y = element_y;
  }
  else {
    x = cols - element_x - 1;
    y = rows - element_y - 1;
  }
  global float* data = (global float*)((global uchar*)src + y * src_stride);
  float4 value = vload4(x, data);

  data = (global float*)((global uchar*)dst + element_y * dst_stride);
  vstore4(value, element_x, data);
}
#endif
