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

/******************************* convertto operation *******************************/

#if defined(CONVERTTO_U8_2_U81D) || defined(ALL_KERNELS)
__kernel
void converttoU8_2_U8Kernel0(global const uchar* src0, int cols,
                             global uchar* dst, float scale, float delta) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }
  uchar4 input_value = vload4(element_x, src0);
  uchar4 output_value =
      convert_uchar4_sat(convert_float4(input_value) * scale + delta);
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

#if defined(CONVERTTO_F32_2_U81D) || defined(ALL_KERNELS)
__kernel
void converttoF32_2_U8Kernel0(global const float* src0, int cols,
                              global uchar* dst, float scale, float delta) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }
  float4 input_value = vload4(element_x, src0);
  uchar4 output_value =
      convert_uchar4_sat(convert_float4(input_value) * scale + delta);
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

#if defined(CONVERTTO_U8_2_U82D) || defined(ALL_KERNELS)
__kernel
void converttoU8_2_U8Kernel1(global const uchar* src0, int rows, int cols,
                             int src0_stride, global uchar* dst, int dst_stride,
                             float scale, float delta) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }
  global uchar* data =
      (global uchar*)((global uchar*)src0 + mul24(element_y, src0_stride));
  uchar4 input_value = vload4(element_x, data);
  uchar4 output_value =
      convert_uchar4_sat(convert_float4(input_value) * scale + delta);
  dst = dst + mul24(element_y, dst_stride);
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

#if defined(CONVERTTO_F32_2_U82D) || defined(ALL_KERNELS)
__kernel
void converttoF32_2_U8Kernel1(global const float* src0, int rows, int cols,
                              int src0_stride, global uchar* dst,
                              int dst_stride, float scale, float delta) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }
  global float* data =
      (global float*)((global uchar*)src0 + mul24(element_y, src0_stride));
  float4 input_value = vload4(element_x, data);
  uchar4 output_value =
      convert_uchar4_sat(convert_float4(input_value) * scale + delta);
  dst = dst + mul24(element_y, dst_stride);
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

#if defined(CONVERTTO_U8_2_F32ALIGNED) || defined(ALL_KERNELS)
__kernel
void converttoU8_2_F32Kernel0(global const uchar* src0, int rows, int cols,
                              int src0_stride, global float* dst,
                              int dst_stride, float scale, float delta) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }
  global uchar* data =
      (global uchar*)((global uchar*)src0 + mul24(element_y, src0_stride));
  uchar2 input_value = vload2(element_x, data);
  float2 output_value = convert_float2(input_value) * scale + delta;
  dst = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, dst);
}
#endif

#if defined(CONVERTTO_F32_2_F32ALIGNED) || defined(ALL_KERNELS)
__kernel
void converttoF32_2_F32Kernel0(global const float* src0, int rows, int cols,
                               int src0_stride, global float* dst,
                               int dst_stride, float scale, float delta) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }
  global float* data =
      (global float*)((global uchar*)src0 + mul24(element_y, src0_stride));
  float2 input_value = vload2(element_x, data);
  float2 output_value = convert_float2(input_value) * scale + delta;
  dst = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, dst);
}
#endif

#if defined(CONVERTTO_U8_2_F32UNALIGNED) || defined(ALL_KERNELS)
__kernel
void converttoU8_2_F32Kernel1(global const uchar* src0, int rows, int cols,
                              int src0_stride, global float* dst,
                              int dst_stride, float scale, float delta) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || element_x >= cols) {
    return;
  }
  global uchar* data =
      (global uchar*)((global uchar*)src0 + mul24(element_y, src0_stride));
  uchar2 input_value = vload2(element_x, data);
  float2 output_value = convert_float2(input_value) * scale + delta;
  dst = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, dst);
  if (index_x < cols - 1) {
    vstore2(output_value, element_x, dst);
  }
  else {
    dst[index_x] = output_value.x;
  }
}
#endif

#if defined(CONVERTTO_F32_2_F32UNALIGNED) || defined(ALL_KERNELS)
__kernel
void converttoF32_2_F32Kernel1(global const float* src0, int rows, int cols,
                               int src0_stride, global float* dst,
                               int dst_stride, float scale, float delta) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || element_x >= cols) {
    return;
  }
  global float* data =
      (global float*)((global uchar*)src0 + mul24(element_y, src0_stride));
  float2 input_value = vload2(element_x, data);
  float2 output_value = convert_float2(input_value) * scale + delta;
  dst = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, dst);
  if (index_x < cols - 1) {
    vstore2(output_value, element_x, dst);
  }
  else {
    dst[index_x] = output_value.x;
  }
}
#endif