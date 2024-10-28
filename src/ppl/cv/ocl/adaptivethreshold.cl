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

#if defined(ADAPTIVETHRESHOLD_GAUSSIANBLUE) ||                                 \
    defined(ADAPTIVETHRESHOLD_MEAN) || defined(ALL_KERNELS)
__kernel
void getGaussianKernelAT(float sigma, int ksize, global float* coefficients,
                       int offset) {
  float value = sigma > 0 ? sigma : ((ksize - 1) * 0.5f - 1) * 0.3f + 0.8f;
  float scale_2x = -0.5f / (value * value);
  float sum = 0.f;
  coefficients = (global float*)((global uchar*)coefficients + offset);

  int i;
  float x;
  for (i = 0; i < ksize; i++) {
    x = i - (ksize - 1) * 0.5f;
    value = exp(scale_2x * x * x);
    coefficients[i] = value;
    sum += value;
  }

  sum = 1.f / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] *= sum;
  }
}
#endif

#if defined(ADAPTIVETHRESHOLD_MEAN) || defined(ALL_KERNELS)
__kernel
void adaptivethreshold_meanF32U8C1Kernel(
    global const uchar* src_input, int src_input_stride,
    global const float* src, int rows, int cols, int radius, int src_stride,
    global uchar* dst, int dst_stride, int threshold_type, float weight,
    int delta, uchar setted_value, enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x * 4, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }

  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int remain_cols = cols - index_x, remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int data_index;
  float4 input_value[4];
  bool isnt_border_block = true;
  data_index = radius >> 7;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 7;
  if (group_x >= data_index)
    isnt_border_block = false;
  global const float* src_temp;
  for (int i = 0; i < min(remain_rows, 4); i++) {
    ((float4*)input_value + i)[0] = (float4)(0);
    src_temp = src;
    if (isnt_border_block) {
      src_temp += bottom;
      for (int j = bottom; j <= top; j++) {
        ((float4*)input_value + i)[0] += convert_float4(vload4(0, src_temp));
        src_temp += 1;
      }
    }
    else {
      float4 value;
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReplicateBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] += value;
        }
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflectBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] += value;
        }
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflect101Border(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] += value;
        }
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  for (int i = 0; i < min(remain_rows, 4); i++) {
    ((float4*)input_value + i)[0] *= weight;
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);
  src_input = src_input + index_x * src_input_stride;
  if (remain_rows >= 4) {
    if (remain_cols >= 4) {
      uchar4 output_value[4];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      output_value[1] =
          convert_uchar4_sat((float4)(input_value[0].y, input_value[1].y,
                                      input_value[2].y, input_value[3].y));
      output_value[2] =
          convert_uchar4_sat((float4)(input_value[0].z, input_value[1].z,
                                      input_value[2].z, input_value[3].z));
      output_value[3] =
          convert_uchar4_sat((float4)(input_value[0].w, input_value[1].w,
                                      input_value[2].w, input_value[3].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar4 output_value[1];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar4 output_value[2];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      output_value[1] =
          convert_uchar4_sat((float4)(input_value[0].y, input_value[1].y,
                                      input_value[2].y, input_value[3].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar4 output_value[3];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      output_value[1] =
          convert_uchar4_sat((float4)(input_value[0].y, input_value[1].y,
                                      input_value[2].y, input_value[3].y));
      output_value[2] =
          convert_uchar4_sat((float4)(input_value[0].z, input_value[1].z,
                                      input_value[2].z, input_value[3].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
  else if (remain_rows == 1) {
    if (remain_cols >= 4) {
      uchar output_value[4];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      output_value[2] = convert_uchar_sat((float)(input_value[0].z));
      output_value[3] = convert_uchar_sat((float)(input_value[0].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar output_value[1];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar output_value[2];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar output_value[3];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      output_value[2] = convert_uchar_sat((float)(input_value[0].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
  else if (remain_rows == 2) {
    if (remain_cols >= 4) {
      uchar2 output_value[4];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));
      output_value[2] =
          convert_uchar2_sat((float2)(input_value[0].z, input_value[1].z));
      output_value[3] =
          convert_uchar2_sat((float2)(input_value[0].w, input_value[1].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar2 output_value[1];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar2 output_value[2];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar2 output_value[3];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));
      output_value[2] =
          convert_uchar2_sat((float2)(input_value[0].z, input_value[1].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
  else if (remain_rows == 3) {
    if (remain_cols >= 4) {
      uchar3 output_value[4];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      output_value[1] = convert_uchar3_sat(
          (float3)(input_value[0].y, input_value[1].y, input_value[2].y));
      output_value[2] = convert_uchar3_sat(
          (float3)(input_value[0].z, input_value[1].z, input_value[2].z));
      output_value[3] = convert_uchar3_sat(
          (float3)(input_value[0].w, input_value[1].w, input_value[2].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar3 output_value[1];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar3 output_value[2];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      output_value[1] = convert_uchar3_sat(
          (float3)(input_value[0].y, input_value[1].y, input_value[2].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar3 output_value[3];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      output_value[1] = convert_uchar3_sat(
          (float3)(input_value[0].y, input_value[1].y, input_value[2].y));
      output_value[2] = convert_uchar3_sat(
          (float3)(input_value[0].z, input_value[1].z, input_value[2].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
}

__kernel
void adaptivethreshold_meanU8F32C1Kernel(
    global const uchar* src_input, int src_input_stride,
    global const uchar* src, int rows, int cols, int radius, int src_stride,
    global float* dst, int dst_stride, int threshold_type, float weight,
    int delta, uchar setted_value, enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x * 2, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }

  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int remain_cols = cols - index_x, remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int data_index;
  float2 input_value[2];
  bool isnt_border_block = true;
  data_index = radius >> 6;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 6;
  if (group_x >= data_index)
    isnt_border_block = false;
  global const uchar* src_temp;
  for (int i = 0; i < min(remain_rows, 2); i++) {
    ((float2*)input_value + i)[0] = (float2)(0);
    src_temp = src;
    if (isnt_border_block) {
      src_temp += bottom;
      for (int j = bottom; j <= top; j++) {
        ((float2*)input_value + i)[0] += convert_float2(vload2(0, src_temp));
        src_temp += 1;
      }
    }
    else {
      float2 value;
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReplicateBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] += value;
        }
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflectBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] += value;
        }
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflect101Border(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] += value;
        }
      }
    }
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
  src_input = src_input + index_x * src_input_stride;
  if (remain_rows >= 2) {
    if (remain_cols >= 2) {
      float2 output_value[2];
      output_value[0] =
          convert_float2((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_float2((float2)(input_value[0].y, input_value[1].y));
      for (int k = 0; k < 2; k++) {
        vstore2(output_value[k], element_y, dst);
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      float2 output_value[1];
      output_value[0] =
          convert_float2((float2)(input_value[0].x, input_value[1].x));
      for (int k = 0; k < 1; k++) {
        vstore2(output_value[k], element_y, dst);
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
  }
  else if (remain_rows == 1) {
    if (remain_cols >= 2) {
      float output_value[2];
      output_value[0] = convert_float((float)(input_value[0].x));
      output_value[1] = convert_float((float)(input_value[0].y));
      for (int k = 0; k < 2; k++) {
        int offset = element_y * 2;
        dst[offset] = output_value[k];
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      float output_value[1];
      output_value[0] = convert_float((float)(input_value[0].x));
      for (int k = 0; k < 1; k++) {
        int offset = element_y * 2;
        dst[offset] = output_value[k];
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
  }
}
#endif

#if defined(ADAPTIVETHRESHOLD_GAUSSIANBLUE) || defined(ALL_KERNELS)
__kernel
void adaptivethreshold_gaussianblurF32U8C1Kernel(
    global const uchar* src_input, int src_input_stride,
    global const float* src, int rows, int cols,
    global const float* filter_kernel, int kernel_offset, int radius,
    int src_stride, global uchar* dst, int dst_stride, int threshold_type,
    int delta, uchar setted_value, enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x * 4, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }

  src = (global const float*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_cols = cols - index_x, remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float4 input_value[4];
  bool isnt_border_block = true;
  data_index = radius >> 7;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 7;
  if (group_x >= data_index)
    isnt_border_block = false;
  global const float* src_temp;
  for (int i = 0; i < min(remain_rows, 4); i++) {
    ((float4*)input_value + i)[0] = (float4)(0);
    src_temp = src;
    filter_kernel_index = 0;
    if (isnt_border_block) {
      src_temp += bottom;
      for (int j = bottom; j <= top; j++) {
        ((float4*)input_value + i)[0] += convert_float4(vload4(0, src_temp)) *
                                         filter_kernel[filter_kernel_index];
        src_temp += 1;
        filter_kernel_index++;
      }
    }
    else {
      float4 value;
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReplicateBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] +=
              value * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflectBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] +=
              value * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflect101Border(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] +=
              value * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);
  src_input = src_input + index_x * src_input_stride;
  if (remain_rows >= 4) {
    if (remain_cols >= 4) {
      uchar4 output_value[4];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      output_value[1] =
          convert_uchar4_sat((float4)(input_value[0].y, input_value[1].y,
                                      input_value[2].y, input_value[3].y));
      output_value[2] =
          convert_uchar4_sat((float4)(input_value[0].z, input_value[1].z,
                                      input_value[2].z, input_value[3].z));
      output_value[3] =
          convert_uchar4_sat((float4)(input_value[0].w, input_value[1].w,
                                      input_value[2].w, input_value[3].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar4 output_value[1];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar4 output_value[2];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      output_value[1] =
          convert_uchar4_sat((float4)(input_value[0].y, input_value[1].y,
                                      input_value[2].y, input_value[3].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar4 output_value[3];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      output_value[1] =
          convert_uchar4_sat((float4)(input_value[0].y, input_value[1].y,
                                      input_value[2].y, input_value[3].y));
      output_value[2] =
          convert_uchar4_sat((float4)(input_value[0].z, input_value[1].z,
                                      input_value[2].z, input_value[3].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(setted_value)
                                     : (int4)(0));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          output_value[k] =
              convert_uchar4_sat(convert_int4(vload4(element_y, src_input)) >
                                         (convert_int4(output_value[k]) - delta)
                                     ? (int4)(0)
                                     : (int4)(setted_value));
          vstore4(output_value[k], element_y, dst);
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
  else if (remain_rows == 1) {
    if (remain_cols >= 4) {
      uchar output_value[4];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      output_value[2] = convert_uchar_sat((float)(input_value[0].z));
      output_value[3] = convert_uchar_sat((float)(input_value[0].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar output_value[1];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar output_value[2];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar output_value[3];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      output_value[2] = convert_uchar_sat((float)(input_value[0].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k]) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
  else if (remain_rows == 2) {
    if (remain_cols >= 4) {
      uchar2 output_value[4];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));
      output_value[2] =
          convert_uchar2_sat((float2)(input_value[0].z, input_value[1].z));
      output_value[3] =
          convert_uchar2_sat((float2)(input_value[0].w, input_value[1].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar2 output_value[1];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar2 output_value[2];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar2 output_value[3];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));
      output_value[2] =
          convert_uchar2_sat((float2)(input_value[0].z, input_value[1].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
  else if (remain_rows == 3) {
    if (remain_cols >= 4) {
      uchar3 output_value[4];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      output_value[1] = convert_uchar3_sat(
          (float3)(input_value[0].y, input_value[1].y, input_value[2].y));
      output_value[2] = convert_uchar3_sat(
          (float3)(input_value[0].z, input_value[1].z, input_value[2].z));
      output_value[3] = convert_uchar3_sat(
          (float3)(input_value[0].w, input_value[1].w, input_value[2].w));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 4; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 1) {
      uchar3 output_value[1];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 2) {
      uchar3 output_value[2];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      output_value[1] = convert_uchar3_sat(
          (float3)(input_value[0].y, input_value[1].y, input_value[2].y));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
    else if (remain_cols == 3) {
      uchar3 output_value[3];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      output_value[1] = convert_uchar3_sat(
          (float3)(input_value[0].y, input_value[1].y, input_value[2].y));
      output_value[2] = convert_uchar3_sat(
          (float3)(input_value[0].z, input_value[1].z, input_value[2].z));
      if (threshold_type == THRESH_BINARY) {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? setted_value
                  : 0;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? setted_value
                  : 0;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
      else {
        for (int k = 0; k < 3; k++) {
          int offset = element_y * 4;
          dst[offset] =
              convert_int(src_input[offset]) >
                      convert_int(convert_uchar_sat(output_value[k].x) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 1] =
              convert_int(src_input[offset + 1]) >
                      convert_int(convert_uchar_sat(output_value[k].y) - delta)
                  ? 0
                  : setted_value;
          dst[offset + 2] =
              convert_int(src_input[offset + 2]) >
                      convert_int(convert_uchar_sat(output_value[k].z) - delta)
                  ? 0
                  : setted_value;
          src_input =
              (global uchar*)((global uchar*)src_input + src_input_stride);
          dst = (global uchar*)((global uchar*)dst + dst_stride);
        }
      }
    }
  }
}

__kernel
void adaptivethreshold_gaussianblurU8F32C1Kernel(
    global const uchar* src_input, int src_input_stride,
    global const uchar* src, int rows, int cols,
    global const float* filter_kernel, int kernel_offset, int radius,
    int src_stride, global float* dst, int dst_stride, int threshold_type,
    int delta, uchar setted_value, enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x * 2, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }

  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_cols = cols - index_x, remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float2 input_value[2];
  bool isnt_border_block = true;
  data_index = radius >> 6;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 6;
  if (group_x >= data_index)
    isnt_border_block = false;
  global const uchar* src_temp;
  for (int i = 0; i < min(remain_rows, 2); i++) {
    ((float2*)input_value + i)[0] = (float2)(0);
    src_temp = src;
    filter_kernel_index = 0;
    if (isnt_border_block) {
      src_temp += bottom;
      for (int j = bottom; j <= top; j++) {
        ((float2*)input_value + i)[0] += convert_float2(vload2(0, src_temp)) *
                                         filter_kernel[filter_kernel_index];
        src_temp += 1;
        filter_kernel_index++;
      }
    }
    else {
      float2 value;
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReplicateBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              value * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflectBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              value * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = bottom; j <= top; j++) {
          data_index = interpolateReflect101Border(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              value * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
      }
    }
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
  src_input = src_input + index_x * src_input_stride;
  if (remain_rows >= 2) {
    if (remain_cols >= 2) {
      float2 output_value[2];
      output_value[0] =
          convert_float2((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_float2((float2)(input_value[0].y, input_value[1].y));
      for (int k = 0; k < 2; k++) {
        vstore2(output_value[k], element_y, dst);
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      float2 output_value[1];
      output_value[0] =
          convert_float2((float2)(input_value[0].x, input_value[1].x));
      for (int k = 0; k < 1; k++) {
        vstore2(output_value[k], element_y, dst);
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
  }
  else if (remain_rows == 1) {
    if (remain_cols >= 2) {
      float output_value[2];
      output_value[0] = convert_float((float)(input_value[0].x));
      output_value[1] = convert_float((float)(input_value[0].y));
      for (int k = 0; k < 2; k++) {
        int offset = element_y * 2;
        dst[offset] = output_value[k];
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      float output_value[1];
      output_value[0] = convert_float((float)(input_value[0].x));
      for (int k = 0; k < 1; k++) {
        int offset = element_y * 2;
        dst[offset] = output_value[k];
        dst = (global float*)((global uchar*)dst + dst_stride);
      }
    }
  }
}
#endif