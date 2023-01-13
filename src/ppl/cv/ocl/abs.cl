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

#if defined(U8) || defined(SPIR)
inline char abs_device0(char src) {
  if (src == -128) {
    return 127;
  }
  else {
    return abs((int)src);
  }
}
#endif

#if defined(F32) || defined(SPIR)
inline float abs_device1(float src) {
  if (src >= 0.f) {
    return src;
  }
  else {
    return (0.f - src);
  }

}
#endif

#if defined(U8ALIGNED) || defined(SPIR)
__kernel
void absU8Kernel0(global const char* src, int rows, int cols, int src_stride,
                  global char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global const char* input = src + element_y * src_stride;
  char4 input_value = vload4(element_x, input);

  char4 output_value;
  output_value.x = abs_device0(input_value.x);
  output_value.y = abs_device0(input_value.y);
  output_value.z = abs_device0(input_value.z);
  output_value.w = abs_device0(input_value.w);

  global char* output = dst + element_y * dst_stride;
  vstore4(output_value, element_x, output);
}
#endif

#if defined(U81D) || defined(SPIR)
__kernel
void absU8Kernel1(global const char* src, int cols, global char* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  if (index_x < cols - 3) {
    char4 input_value = vload4(element_x, src);

    char4 output_value;
    output_value.x = abs_device0(input_value.x);
    output_value.y = abs_device0(input_value.y);
    output_value.z = abs_device0(input_value.z);
    output_value.w = abs_device0(input_value.w);

    vstore4(output_value, element_x, dst);
  }
  else {
    global const char4* input = src;
    char4 input_value, output_value;
    input_value = input[element_x];

    output_value.x = abs_device0(input_value.x);
    if (index_x < cols - 1) {
      output_value.y = abs_device0(input_value.y);
    }
    if ((index_x < cols - 2)) {
      output_value.z = abs_device0(input_value.z);
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
#endif

#if defined(U8UNALIGNED) || defined(SPIR)
__kernel
void absU8Kernel2(global const char* src, int rows, int cols, int src_stride,
                  global char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  if (index_x < cols - 3) {
    global const char* input = src + element_y * src_stride;
    char4 input_value = vload4(element_x, input);

    char4 output_value;
    output_value.x = abs_device0(input_value.x);
    output_value.y = abs_device0(input_value.y);
    output_value.z = abs_device0(input_value.z);
    output_value.w = abs_device0(input_value.w);

    global char* output = dst + element_y * dst_stride;
    vstore4(output_value, element_x, output);
  }
  else {
    global const char* input = src + element_y * src_stride;
    global char* output = dst + element_y * dst_stride;

    char input_value0, input_value1, input_value2;
    char output_value0, output_value1, output_value2;

    input_value0 = input[index_x];
    if (index_x < cols - 1) {
      input_value1 = input[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value2 = input[index_x + 2];
    }

    output_value0 = abs_device0(input_value0);
    if (index_x < cols - 1) {
      output_value1 = abs_device0(input_value1);
    }
    if ((index_x < cols - 2)) {
      output_value2 = abs_device0(input_value2);
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
#endif

#if defined(F32ALIGNED) || defined(SPIR)
__kernel
void absF32Kernel0(global const float* src, int rows, int cols, int src_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global const float* input = (global float*)((global uchar*)src +
                               element_y * src_stride);
  float2 input_value = vload2(element_x, input);

  float2 output_value;
  output_value.x = abs_device1(input_value.x);
  output_value.y = abs_device1(input_value.y);

  global float* output = (global float*)((global uchar*)dst +
                          element_y * dst_stride);
  vstore2(output_value, element_x, output);
}
#endif

#if defined(F32UNALIGNED) || defined(SPIR)
__kernel
void absF32Kernel1(global const float* src, int rows, int cols, int src_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  if (index_x < cols - 1) {
    global const float* input = (global float*)((global uchar*)src +
                                 element_y * src_stride);
    float2 input_value = vload2(element_x, input);

    float2 output_value;
    output_value.x = abs_device1(input_value.x);
    output_value.y = abs_device1(input_value.y);

    global float* output = (global float*)((global uchar*)dst +
                            element_y * dst_stride);
    vstore2(output_value, element_x, output);
  }
  else {
    global float* input = (global float*)((global uchar*)src +
                           element_y * src_stride);
    global float* output = (global float*)((global uchar*)dst +
                            element_y * dst_stride);

    float input_value = input[index_x];
    float output_value = abs_device1(input_value);

    output[index_x] = output_value;
  }
}
#endif
