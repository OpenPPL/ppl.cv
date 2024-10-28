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

#if defined(SPLIT3_U81D) || defined(ALL_KERNELS)
__kernel
void split3U8Kernel0(global const uchar* src, int cols, global uchar* dst0,
                     global uchar* dst1, global uchar* dst2) {
  int element_x = get_global_id(0);
  if (element_x >= cols) {
    return;
  }
  uchar3 input_value;
  input_value = vload3(element_x, src);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
}
#endif

#if defined(SPLIT3_U82D) || defined(ALL_KERNELS)
__kernel
void split3U8Kernel1(global const uchar* src, int rows, int cols,
                     int src_stride, global uchar* dst0, global uchar* dst1,
                     global uchar* dst2, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  uchar3 input_value;
  input_value =
      vload3(element_x, (global uchar*)((uchar*)src + element_y * src_stride));
  int offset = element_y * dst_stride;
  dst0 = (global uchar*)((uchar*)dst0 + offset);
  dst1 = (global uchar*)((uchar*)dst1 + offset);
  dst2 = (global uchar*)((uchar*)dst2 + offset);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
}
#endif

#if defined(SPLIT4_U81D) || defined(ALL_KERNELS)
__kernel
void split4U8Kernel0(global const uchar* src, int cols, global uchar* dst0,
                     global uchar* dst1, global uchar* dst2,
                     global uchar* dst3) {
  int element_x = get_global_id(0);
  if (element_x >= cols) {
    return;
  }
  uchar4 input_value;
  input_value = vload4(element_x, src);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
  dst3[element_x] = input_value.w;
}
#endif

#if defined(SPLIT4_U82D) || defined(ALL_KERNELS)
__kernel
void split4U8Kernel1(global const uchar* src, int rows, int cols,
                     int src_stride, global uchar* dst0, global uchar* dst1,
                     global uchar* dst2, global uchar* dst3, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  uchar4 input_value;
  input_value =
      vload4(element_x, (global uchar*)((uchar*)src + element_y * src_stride));
  int offset = element_y * dst_stride;
  dst0 = (global uchar*)((uchar*)dst0 + offset);
  dst1 = (global uchar*)((uchar*)dst1 + offset);
  dst2 = (global uchar*)((uchar*)dst2 + offset);
  dst3 = (global uchar*)((uchar*)dst3 + offset);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
  dst3[element_x] = input_value.w;
}
#endif

#if defined(SPLIT3_F321D) || defined(ALL_KERNELS)
__kernel
void split3F32Kernel0(global const float* src, int cols, global float* dst0,
                      global float* dst1, global float* dst2) {
  int element_x = get_global_id(0);
  if (element_x >= cols) {
    return;
  }
  float3 input_value;
  input_value = vload3(element_x, src);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
}
#endif

#if defined(SPLIT3_F322D) || defined(ALL_KERNELS)
__kernel
void split3F32Kernel1(global const float* src, int rows, int cols,
                      int src_stride, global float* dst0, global float* dst1,
                      global float* dst2, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  float3 input_value;
  input_value =
      vload3(element_x, (global float*)((uchar*)src + element_y * src_stride));
  int offset = element_y * dst_stride;
  dst0 = (global float*)((uchar*)dst0 + offset);
  dst1 = (global float*)((uchar*)dst1 + offset);
  dst2 = (global float*)((uchar*)dst2 + offset);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
}
#endif

#if defined(SPLIT4_F321D) || defined(ALL_KERNELS)
__kernel
void split4F32Kernel0(global const float* src, int cols, global float* dst0,
                      global float* dst1, global float* dst2,
                      global float* dst3) {
  int element_x = get_global_id(0);
  if (element_x >= cols) {
    return;
  }
  float4 input_value;
  input_value = vload4(element_x, src);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
  dst3[element_x] = input_value.w;
}
#endif

#if defined(SPLIT4_F322D) || defined(ALL_KERNELS)
__kernel
void split4F32Kernel1(global const float* src, int rows, int cols,
                      int src_stride, global float* dst0, global float* dst1,
                      global float* dst2, global float* dst3, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= cols || element_y >= rows) {
    return;
  }
  float4 input_value;
  input_value =
      vload4(element_x, (global float*)((uchar*)src + element_y * src_stride));
  int offset = element_y * dst_stride;
  dst0 = (global float*)((uchar*)dst0 + offset);
  dst1 = (global float*)((uchar*)dst1 + offset);
  dst2 = (global float*)((uchar*)dst2 + offset);
  dst3 = (global float*)((uchar*)dst3 + offset);
  dst0[element_x] = input_value.x;
  dst1[element_x] = input_value.y;
  dst2[element_x] = input_value.z;
  dst3[element_x] = input_value.w;
}
#endif