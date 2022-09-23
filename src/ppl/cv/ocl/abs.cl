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

#include "utility/utility.hpp"

inline schar abs_device(const schar& src) {
  if (src == -128) {
    return 127;
  }
  else {
    return abs((int)src);
  }
}

inline float abs_device(const float& src) {
  if (src >= 0) {
    return src;
  }
  else {
    return (0 - src);
  }
  // return fabs(src);
}

__kernel
void absKernel2(global const schar* src, int rows, int cols, int src_stride,
                global schar* dst, int dst_stride) {
//   int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
//   int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const char* input = (char*)(src + element_y * src_stride);
  char4 input_value = vload4(element_x, input);

  char4 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);
  output_value.z = abs_device(input_value.z);
  output_value.w = abs_device(input_value.w);

  char* output = (char*)(dst + element_y * dst_stride);
  vstore4(output_value, element_x, output);
}

__kernel
void absKernel0(global const schar* src, int rows, int cols, int src_stride,
                global schar* dst, int dst_stride) {
//   int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
//   int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const char4* input = (char4*)(src + element_y * src_stride);
  char4 input_value = input[element_x];

  char4 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);
  output_value.z = abs_device(input_value.z);
  output_value.w = abs_device(input_value.w);

  char4* output = (char4*)(dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__kernel
void absKernel10(global const schar* src, int cols, global schar* dst) {
  // int element_x = (blockIdx.x << 8) + threadIdx.x;
  // int index_x = element_x << 2;
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  const char4* input = (char4*)src;
  char4 input_value, output_value;
  input_value = input[element_x];

  if (index_x < cols - 3) {
    output_value.x = abs_device(input_value.x);
    output_value.y = abs_device(input_value.y);
    output_value.z = abs_device(input_value.z);
    output_value.w = abs_device(input_value.w);

    char4* output = (char4*)dst;
    output[element_x] = output_value;
  }
  else {
    output_value.x = abs_device(input_value.x);
    if (index_x < cols - 1) {
      output_value.y = abs_device(input_value.y);
    }
    if ((index_x < cols - 2)) {
      output_value.z = abs_device(input_value.z);
    }

    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = output_value.z;
    }
  }
}

__kernel
void absKernel11(global const schar* src, int rows, int cols, int src_stride,
                 global schar* dst, int dst_stride) {
  // int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  // int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  // int index_x = element_x << 2;
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x   = get_group_id(0);
  int group_num = get_num_groups(0);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const schar* input = src + element_y * src_stride;
  schar* output = dst + element_y * dst_stride;

  schar input_value0, input_value1, input_value2, input_value3;
  schar output_value0, output_value1, output_value2, output_value3;

  if (group_x < group_num - 1) {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];
    input_value2 = input[index_x + 2];
    input_value3 = input[index_x + 3];

    output_value0 = abs_device(input_value0);
    output_value1 = abs_device(input_value1);
    output_value2 = abs_device(input_value2);
    output_value3 = abs_device(input_value3);

    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
    output[index_x + 2] = output_value2;
    output[index_x + 3] = output_value3;
  }
  else {
    input_value0 = input[index_x];
    if (index_x < cols - 1) {
      input_value1 = input[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value2 = input[index_x + 2];
    }

    output_value0 = abs_device(input_value0);
    if (index_x < cols - 1) {
      output_value1 = abs_device(input_value1);
    }
    if ((index_x < cols - 2)) {
      output_value2 = abs_device(input_value2);
    }

    output[index_x] = output_value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = output_value1;
    }
    if ((index_x < cols - 2)) {
      output[index_x + 2] = output_value2;
    }
  }
}

__kernel
void absKernel0(global const float* src, int rows, int cols, int src_stride,
                global float* dst, int dst_stride) {
  // int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  // int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input = (float2*)((uchar*)src + element_y * src_stride);
  float2 input_value = input[element_x];

  float2 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__kernel
void absKernel2f(global const float* src, int rows, int cols, int src_stride,
                global float* dst, int dst_stride) {
//   int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
//   int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float* input = (float*)(src + element_y * src_stride);
  float4 input_value = vload4(element_x, input);

  float4 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);
  output_value.z = abs_device(input_value.z);
  output_value.w = abs_device(input_value.w);

  float* output = (float*)(dst + element_y * dst_stride);
  vstore4(output_value, element_x, output);
}
