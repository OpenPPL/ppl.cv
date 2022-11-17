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

#if defined(ADD_U81D) || defined(SPIR)
__kernel
void addU8Kernel0(global const uchar* src0, int cols, global const uchar* src1,
                  global uchar* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  uchar4 input_value0 = vload4(element_x, src0);
  uchar4 input_value1 = vload4(element_x, src1);
  uint4 sum = convert_uint4(input_value0) + convert_uint4(input_value1);
  uchar4 output_value = convert_uchar4_sat(sum);

  if (index_x < cols - 3) {
    vstore4(output_value, element_x, dst);
  }
  else {
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

#if defined(ADD_U82D) || defined(SPIR)
__kernel
void addU8Kernel1(global const uchar* src0, int rows, int cols, int src0_stride,
                  global const uchar* src1, int src1_stride, global uchar* dst,
                  int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global const uchar* input0 = src0 + element_y * src0_stride;
  global const uchar* input1 = src1 + element_y * src1_stride;
  global uchar* output = dst + element_y * dst_stride;

  uchar4 input_value0 = vload4(element_x, input0);
  uchar4 input_value1 = vload4(element_x, input1);
  uint4 sum = convert_uint4(input_value0) + convert_uint4(input_value1);
  uchar4 output_value = convert_uchar4_sat(sum);

  if (index_x < cols - 3) {
    vstore4(output_value, element_x, output);
  }
  else {
    output[index_x] = output_value.x;
    if (index_x < cols - 1) {
      output[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      output[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(ADD_F32ALIGNED) || defined(SPIR)
__kernel
void addF32Kernel0(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global const float* input0 = (global float*)((global uchar*)src0 +
                                element_y * src0_stride);
  global const float* input1 = (global float*)((global uchar*)src1 +
                                element_y * src1_stride);
  float2 input_value0 = vload2(element_x, input0);
  float2 input_value1 = vload2(element_x, input1);
  float2 output_value = input_value0 + input_value1;

  global float* output = (global float*)((global uchar*)dst +
                          element_y * dst_stride);
  vstore2(output_value, element_x, output);
}
#endif

#if defined(ADD_F32UNALIGNED) || defined(SPIR)
__kernel
void addF32Kernel1(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global const float* input0 = (global float*)((global uchar*)src0 +
                                element_y * src0_stride);
  global const float* input1 = (global float*)((global uchar*)src1 +
                                element_y * src1_stride);
  global float* output = (global float*)((global uchar*)dst +
                          element_y * dst_stride);

  float2 input_value0 = vload2(element_x, input0);
  float2 input_value1 = vload2(element_x, input1);
  float2 output_value = input_value0 + input_value1;

  if (index_x < cols - 1) {
    vstore2(output_value, element_x, output);
  }
  else {
    output[index_x] = output_value.x;
  }
}
#endif
