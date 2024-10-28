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

#if defined(MERGE3_U81D) || defined(ALL_KERNELS)
__kernel
void merge3U8Kernel0(global const uchar* src0, global const uchar* src1,
                     global const uchar* src2, int cols, global uchar* dst) {
  int element_x = get_global_id(0);
  if (element_x >= cols) {
    return;
  }
  uchar input_value0 = src0[element_x];
  uchar input_value1 = src1[element_x];
  uchar input_value2 = src2[element_x];
  uchar3 output_value = (uchar3)(input_value0, input_value1, input_value2);
  vstore3(output_value, element_x, dst);
}
#endif

#if defined(MERGE3_U82D) || defined(ALL_KERNELS)
__kernel
void merge3U8Kernel1(global const uchar* src0, global const uchar* src1,
                     global const uchar* src2, int rows, int cols,
                     int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int offset = element_y * src_stride;
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  src0 = (global const uchar*)((uchar*)src0 + offset);
  src1 = (global const uchar*)((uchar*)src1 + offset);
  src2 = (global const uchar*)((uchar*)src2 + offset);
  dst = (global uchar*)((uchar*)dst + element_y * dst_stride);
  uchar input_value0 = src0[element_x];
  uchar input_value1 = src1[element_x];
  uchar input_value2 = src2[element_x];
  uchar3 output_value = (uchar3)(input_value0, input_value1, input_value2);
  vstore3(output_value, element_x, dst);
}
#endif

#if defined(MERGE4_U81D) || defined(ALL_KERNELS)
__kernel
void merge4U8Kernel0(global const uchar* src0, global const uchar* src1,
                     global const uchar* src2, global const uchar* src3,
                     int cols, global uchar* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x;
  if (index_x >= cols) {
    return;
  }
  uchar input_value0 = src0[index_x];
  uchar input_value1 = src1[index_x];
  uchar input_value2 = src2[index_x];
  uchar input_value3 = src3[index_x];
  uchar4 output_value =
      (uchar4)(input_value0, input_value1, input_value2, input_value3);
  vstore4(output_value, element_x, dst);
}
#endif

#if defined(MERGE4_U82D) || defined(ALL_KERNELS)
__kernel
void merge4U8Kernel1(global const uchar* src0, global const uchar* src1,
                     global const uchar* src2, global const uchar* src3,
                     int rows, int cols, int src_stride, global uchar* dst,
                     int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int offset = element_y * src_stride;
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  src0 = (global const uchar*)((uchar*)src0 + offset);
  src1 = (global const uchar*)((uchar*)src1 + offset);
  src2 = (global const uchar*)((uchar*)src2 + offset);
  src3 = (global const uchar*)((uchar*)src3 + offset);
  dst = (global uchar*)((uchar*)dst + element_y * dst_stride);
  uchar input_value0 = src0[element_x];
  uchar input_value1 = src1[element_x];
  uchar input_value2 = src2[element_x];
  uchar input_value3 = src3[element_x];
  uchar4 output_value =
      (uchar4)(input_value0, input_value1, input_value2, input_value3);
  vstore4(output_value, element_x, dst);
}
#endif

#if defined(MERGE3_F321D) || defined(ALL_KERNELS)
__kernel
void merge3F32Kernel0(global const float* src0, global const float* src1,
                      global const float* src2, int cols, global float* dst) {
  int element_x = get_global_id(0);
  if (element_x >= cols) {
    return;
  }
  float input_value0 = src0[element_x];
  float input_value1 = src1[element_x];
  float input_value2 = src2[element_x];
  float3 output_value = (float3)(input_value0, input_value1, input_value2);
  vstore3(output_value, element_x, dst);
}
#endif

#if defined(MERGE3_F322D) || defined(ALL_KERNELS)
__kernel
void merge3F32Kernel1(global const float* src0, global const float* src1,
                      global const float* src2, int rows, int cols,
                      int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int offset = element_y * src_stride;
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  src0 = (global const float*)((uchar*)src0 + offset);
  src1 = (global const float*)((uchar*)src1 + offset);
  src2 = (global const float*)((uchar*)src2 + offset);
  dst = (global float*)((uchar*)dst + element_y * dst_stride);
  float input_value0 = src0[element_x];
  float input_value1 = src1[element_x];
  float input_value2 = src2[element_x];
  float3 output_value = (float3)(input_value0, input_value1, input_value2);
  vstore3(output_value, element_x, dst);
}
#endif

#if defined(MERGE4_F321D) || defined(ALL_KERNELS)
__kernel
void merge4F32Kernel0(global const float* src0, global const float* src1,
                      global const float* src2, global const float* src3,
                      int cols, global float* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x;
  if (index_x >= cols) {
    return;
  }
  float input_value0 = src0[index_x];
  float input_value1 = src1[index_x];
  float input_value2 = src2[index_x];
  float input_value3 = src3[index_x];
  float4 output_value =
      (float4)(input_value0, input_value1, input_value2, input_value3);
  vstore4(output_value, element_x, dst);
}
#endif

#if defined(MERGE4_F322D) || defined(ALL_KERNELS)
__kernel
void merge4F32Kernel1(global const float* src0, global const float* src1,
                      global const float* src2, global const float* src3,
                      int rows, int cols, int src_stride, global float* dst,
                      int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int offset = element_y * src_stride;
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  src0 = (global const float*)((uchar*)src0 + offset);
  src1 = (global const float*)((uchar*)src1 + offset);
  src2 = (global const float*)((uchar*)src2 + offset);
  src3 = (global const float*)((uchar*)src3 + offset);
  dst = (global float*)((uchar*)dst + element_y * dst_stride);
  float input_value0 = src0[element_x];
  float input_value1 = src1[element_x];
  float input_value2 = src2[element_x];
  float input_value3 = src3[element_x];
  float4 output_value =
      (float4)(input_value0, input_value1, input_value2, input_value3);
  vstore4(output_value, element_x, dst);
}
#endif
