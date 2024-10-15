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

/******************************* crop operation *******************************/

#if defined(CROP_U8) || defined(ALL_KERNELS)
__kernel
void cropU8Kernel(global const uchar* src, int src_stride, const int top,
                  const int left, const float scale, global uchar* dst,
                  int dst_rows, int dst_cols, int dst_stride) {
  int element_x = get_global_id(0) << 2;
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }
  src = src + (top + element_y) * src_stride;
  int offset_element_x = left + element_x;
  uchar input_value0 = src[offset_element_x];
  uchar input_value1 = src[offset_element_x + 1];
  uchar input_value2 = src[offset_element_x + 2];
  uchar input_value3 = src[offset_element_x + 3];

  if (scale != 1) {
    input_value0 = convert_uchar_sat(input_value0 * scale);
    input_value1 = convert_uchar_sat(input_value1 * scale);
    input_value2 = convert_uchar_sat(input_value2 * scale);
    input_value3 = convert_uchar_sat(input_value3 * scale);
  }

  dst = dst + element_y * dst_stride;
  if (element_x < dst_cols - 3) {
    dst[element_x] = input_value0;
    dst[element_x + 1] = input_value1;
    dst[element_x + 2] = input_value2;
    dst[element_x + 3] = input_value3;
  }
  else {
    dst[element_x] = input_value0;
    if (element_x < dst_cols - 1) {
      dst[element_x + 1] = input_value1;
    }
    if (element_x < dst_cols - 2) {
      dst[element_x + 2] = input_value2;
    }
  }
}
#endif

#if defined(CROP_F32) || defined(ALL_KERNELS)
__kernel
void cropF32Kernel(global const float* src, int src_stride, const int top,
                   const int left, const float scale, global float* dst,
                   int dst_rows, int dst_cols, int dst_stride) {
  int element_x = get_global_id(0) << 1;
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }
  src = (global const float*)((uchar*)src + (top + element_y) * src_stride);

  int offset_element_x = left + element_x;
  float input_value0 = src[offset_element_x];
  float input_value1 = src[offset_element_x + 1];

  if (scale != 1) {
    input_value0 = input_value0 * scale;
    input_value1 = input_value1 * scale;
  }

  dst = (global float*)((uchar*)dst + element_y * dst_stride);
  dst[element_x] = input_value0;
  if (element_x < dst_cols - 1) {
    dst[element_x + 1] = input_value1;
  }
}
#endif