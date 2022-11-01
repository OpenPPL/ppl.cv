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

#if defined(U8)
inline signed char abs_device0(signed char src) {
  if (src == -128) {
    return 127;
  }
  else {
    return abs((int)src);
  }
}

#elif defined(F32)
inline float abs_device1(float src) {
  if (src >= 0.f) {
    return src;
  }
  else {
    return (0.f - src);
  }

}
#endif

#if defined(U8ALIGNED)
__kernel void absU8Kernel0(global const signed char* src, int rows, int cols,
                  int src_stride, global signed char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global const char4* input = (global char4*)(src + element_y * src_stride);
  char4 input_value = input[element_x];

  char4 output_value;
  output_value.x = abs_device0(input_value.x);
  output_value.y = abs_device0(input_value.y);
  output_value.z = abs_device0(input_value.z);
  output_value.w = abs_device0(input_value.w);

  global char4* output = (global char4*)(dst + element_y * dst_stride);
  output[element_x] = output_value;
}

#elif defined(U81D)
__kernel void absU8Kernel1(global const signed char* src, int cols,
                           global signed char* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  global const char4* input = (global char4*)src;
  char4 input_value, output_value;
  input_value = input[element_x];

  if (index_x < cols - 3) {
    output_value.x = abs_device0(input_value.x);
    output_value.y = abs_device0(input_value.y);
    output_value.z = abs_device0(input_value.z);
    output_value.w = abs_device0(input_value.w);

    global char4* output = (global char4*)dst;
    output[element_x] = output_value;
  }
  else {
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

#elif defined(U8UNALIGNED)
__kernel void absU8Kernel2(global const signed char* src, int rows, int cols,
                  int src_stride, global signed char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global const signed char* input = (global char*)(src + element_y * src_stride);
  global signed char* output = (global char*)(dst + element_y * dst_stride);

  signed char input_value0, input_value1, input_value2, input_value3;
  signed char output_value0, output_value1, output_value2, output_value3;

  if (index_x < cols - 3) {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];
    input_value2 = input[index_x + 2];
    input_value3 = input[index_x + 3];

    output_value0 = abs_device0(input_value0);
    output_value1 = abs_device0(input_value1);
    output_value2 = abs_device0(input_value2);
    output_value3 = abs_device0(input_value3);

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

#elif defined(U8KERNEL3)
__kernel void absU8Kernel3(global const signed char* src, int rows, int cols,
                  int src_stride, global signed char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global const char* input = (global char*)(src + element_y * src_stride);
  char4 input_value = vload4(element_x, input);

  char4 output_value;
  output_value.x = abs_device0(input_value.x);
  output_value.y = abs_device0(input_value.y);
  output_value.z = abs_device0(input_value.z);
  output_value.w = abs_device0(input_value.w);

  global char* output = (global char*)(dst + element_y * dst_stride);
  vstore4(output_value, element_x, output);
}

#elif defined(F32ALIGNED)
__kernel void absF32Kernel0(global const float* src, int rows, int cols,
                            int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global const float2* input = (global float2*)((global uchar*)src + element_y * src_stride);
  float2 input_value = input[element_x];

  float2 output_value;
  output_value.x = abs_device1(input_value.x);
  output_value.y = abs_device1(input_value.y);

  global float2* output = (global float2*)((global uchar*)dst + element_y * dst_stride);
  output[element_x] = output_value;
}

#elif defined(F32UNALIGNED)
__kernel void absF32Kernel1(global const float* src, int rows, int cols, int src_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global float* input  = (global float*)((global uchar*)src + element_y * src_stride);
  global float* output = (global float*)((global uchar*)dst + element_y * dst_stride);
  float input_value0, input_value1;
  float output_value0, output_value1;

  if (index_x < cols - 1) {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];
    output_value0 = abs_device1(input_value0);
    output_value1 = abs_device1(input_value1);

    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
  }
  else {
    input_value0 = input[index_x];
    if (index_x < cols - 1) {
      input_value1 = input[index_x + 1];
    }

    output_value0 = abs_device1(input_value0);
    if (index_x < cols - 1) {
      output_value1 = abs_device1(input_value1);
    }

    output[index_x] = output_value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = output_value1;
    }
  }
}

#elif defined(F32KERNEL2)
__kernel void absF32Kernel2(global const float* src, int rows, int cols, int src_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global const float* input = (global float*)(src + element_y * src_stride);
  float4 input_value = vload4(element_x, input);

  float4 output_value;
  output_value.x = abs_device1(input_value.x);
  output_value.y = abs_device1(input_value.y);
  output_value.z = abs_device1(input_value.z);
  output_value.w = abs_device1(input_value.w);

  global float* output = (global float*)(dst + element_y * dst_stride);
  vstore4(output_value, element_x, output);
}
#endif