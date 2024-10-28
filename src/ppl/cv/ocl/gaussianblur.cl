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

#if defined(GAUSSIANBLUR_U8C1) || defined(GAUSSIANBLUR_F32C1) ||               \
    defined(GAUSSIANBLUR_U8C3) || defined(GAUSSIANBLUR_F32C3) ||               \
    defined(GAUSSIANBLUR_U8C4) || defined(GAUSSIANBLUR_F32C4) ||               \
    defined(ALL_KERNELS)
__kernel
void getGaussianKernel(float sigma, int ksize, global float* coefficients,
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

#if defined(GAUSSIANBLUR_U8C1) || defined(ALL_KERNELS)
__kernel
void gaussianblurF32U8C1Kernel(global const float* src, int src_offset,
                               int rows, int cols,
                               global const float* filter_kernel,
                               int kernel_offset, int radius, int src_stride,
                               global uchar* dst, int dst_stride,
                               int dst_offset,
                               enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x * 4, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((uchar*)src + src_offset);
  dst = (global uchar*)((uchar*)dst + dst_offset);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  src = (global const float*)((global uchar*)src + index_y * src_stride);
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
  int data_index0;
  global const float* src_temp;
  for (int i = 0; i < min(remain_rows, 4); i++) {
    ((float4*)input_value + i)[0] = (float4)(0);
    src_temp = src;
    filter_kernel_index = 0;
    if (isnt_border_block) {
      src_temp += index_x;
      for (int j = radius; j > 0; j--) {
        ((float4*)input_value + i)[0] +=
            (convert_float4(vload4(0, src_temp - j)) +
             convert_float4(vload4(0, src_temp + j))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float4*)input_value + i)[0] += convert_float4(vload4(0, src_temp)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      float4 value;
      int j;
      float4 value1;
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReplicateBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReplicateBorder(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 2);
          value1.z = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 3);
          value1.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReplicateBorder(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReplicateBorder(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        data_index = interpolateReplicateBorder(cols, radius, j + 2);
        value.z = convert_float(src_temp[data_index]);
        data_index = interpolateReplicateBorder(cols, radius, j + 3);
        value.w = convert_float(src_temp[data_index]);
        ((float4*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReflectBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReflectBorder(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 2);
          value1.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 3);
          value1.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReflectBorder(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReflectBorder(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        data_index = interpolateReflectBorder(cols, radius, j + 2);
        value.z = convert_float(src_temp[data_index]);
        data_index = interpolateReflectBorder(cols, radius, j + 3);
        value.w = convert_float(src_temp[data_index]);
        ((float4*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReflect101Border(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 2);
          value.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 3);
          value.w = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReflect101Border(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 2);
          value1.z = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 3);
          value1.w = convert_float(src_temp[data_index]);
          ((float4*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReflect101Border(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReflect101Border(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        data_index = interpolateReflect101Border(cols, radius, j + 2);
        value.z = convert_float(src_temp[data_index]);
        data_index = interpolateReflect101Border(cols, radius, j + 3);
        value.w = convert_float(src_temp[data_index]);
        ((float4*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);
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
      for (int k = 0; k < 4; k++) {
        vstore4(output_value[k], element_y, dst);
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      uchar4 output_value[1];
      output_value[0] =
          convert_uchar4_sat((float4)(input_value[0].x, input_value[1].x,
                                      input_value[2].x, input_value[3].x));
      for (int k = 0; k < 1; k++) {
        vstore4(output_value[k], element_y, dst);
        dst = (global uchar*)((global uchar*)dst + dst_stride);
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
      for (int k = 0; k < 2; k++) {
        vstore4(output_value[k], element_y, dst);
        dst = (global uchar*)((global uchar*)dst + dst_stride);
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
      for (int k = 0; k < 3; k++) {
        vstore4(output_value[k], element_y, dst);
        dst = (global uchar*)((global uchar*)dst + dst_stride);
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
      for (int k = 0; k < 4; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k];
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      uchar output_value[1];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      for (int k = 0; k < 1; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k];
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 2) {
      uchar output_value[2];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      for (int k = 0; k < 2; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k];
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 3) {
      uchar output_value[3];
      output_value[0] = convert_uchar_sat((float)(input_value[0].x));
      output_value[1] = convert_uchar_sat((float)(input_value[0].y));
      output_value[2] = convert_uchar_sat((float)(input_value[0].z));
      for (int k = 0; k < 3; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k];
        dst = (global uchar*)((global uchar*)dst + dst_stride);
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
      for (int k = 0; k < 4; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      uchar2 output_value[1];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      for (int k = 0; k < 1; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 2) {
      uchar2 output_value[2];
      output_value[0] =
          convert_uchar2_sat((float2)(input_value[0].x, input_value[1].x));
      output_value[1] =
          convert_uchar2_sat((float2)(input_value[0].y, input_value[1].y));
      for (int k = 0; k < 2; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
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
      for (int k = 0; k < 3; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
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
      for (int k = 0; k < 4; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst[offset + 2] = output_value[k].z;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 1) {
      uchar3 output_value[1];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      for (int k = 0; k < 1; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst[offset + 2] = output_value[k].z;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
    else if (remain_cols == 2) {
      uchar3 output_value[2];
      output_value[0] = convert_uchar3_sat(
          (float3)(input_value[0].x, input_value[1].x, input_value[2].x));
      output_value[1] = convert_uchar3_sat(
          (float3)(input_value[0].y, input_value[1].y, input_value[2].y));
      for (int k = 0; k < 2; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst[offset + 2] = output_value[k].z;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
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
      for (int k = 0; k < 3; k++) {
        int offset = element_y * 4;
        dst[offset] = output_value[k].x;
        dst[offset + 1] = output_value[k].y;
        dst[offset + 2] = output_value[k].z;
        dst = (global uchar*)((global uchar*)dst + dst_stride);
      }
    }
  }
}

__kernel
void gaussianblurU8F32C1Kernel(global const uchar* src, int src_offset,
                               int rows, int cols,
                               global const float* filter_kernel,
                               int kernel_offset, int radius, int src_stride,
                               global float* dst, int dst_stride,
                               int dst_offset,
                               enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x * 2, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((uchar*)src + src_offset);
  dst = (global float*)((uchar*)dst + dst_offset);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
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
  int data_index0;
  global const uchar* src_temp;
  for (int i = 0; i < min(remain_rows, 2); i++) {
    ((float2*)input_value + i)[0] = (float2)(0);
    src_temp = src;
    filter_kernel_index = 0;
    if (isnt_border_block) {
      src_temp += index_x;
      for (int j = radius; j > 0; j--) {
        ((float2*)input_value + i)[0] +=
            (convert_float2(vload2(0, src_temp - j)) +
             convert_float2(vload2(0, src_temp + j))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float2*)input_value + i)[0] += convert_float2(vload2(0, src_temp)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      float2 value;
      int j;
      float2 value1;
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReplicateBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReplicateBorder(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReplicateBorder(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReplicateBorder(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        ((float2*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReflectBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReflectBorder(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReflectBorder(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReflectBorder(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        ((float2*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReflect101Border(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReflect101Border(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReflect101Border(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReflect101Border(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        ((float2*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
    }
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
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

#if defined(GAUSSIANBLUR_F32C1) || defined(ALL_KERNELS)
__kernel
void gaussianblurF32F32C1Kernel(global const float* src, int src_offset,
                                int rows, int cols,
                                global const float* filter_kernel,
                                int kernel_offset, int radius, int src_stride,
                                global float* dst, int dst_stride,
                                int dst_offset,
                                enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x * 2, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((uchar*)src + src_offset);
  dst = (global float*)((uchar*)dst + dst_offset);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  src = (global const float*)((global uchar*)src + index_y * src_stride);
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
  int data_index0;
  global const float* src_temp;
  for (int i = 0; i < min(remain_rows, 2); i++) {
    ((float2*)input_value + i)[0] = (float2)(0);
    src_temp = src;
    filter_kernel_index = 0;
    if (isnt_border_block) {
      src_temp += index_x;
      for (int j = radius; j > 0; j--) {
        ((float2*)input_value + i)[0] +=
            (convert_float2(vload2(0, src_temp - j)) +
             convert_float2(vload2(0, src_temp + j))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float2*)input_value + i)[0] += convert_float2(vload2(0, src_temp)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      float2 value;
      int j;
      float2 value1;
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReplicateBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReplicateBorder(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReplicateBorder(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReplicateBorder(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReplicateBorder(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        ((float2*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReflectBorder(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReflectBorder(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflectBorder(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReflectBorder(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReflectBorder(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        ((float2*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j_radius = radius; j_radius > 0; j_radius--) {
          j = index_x - j_radius;
          data_index = interpolateReflect101Border(cols, radius, j);
          value.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value.y = convert_float(src_temp[data_index]);
          j = index_x + j_radius;
          data_index = interpolateReflect101Border(cols, radius, j);
          value1.x = convert_float(src_temp[data_index]);
          data_index = interpolateReflect101Border(cols, radius, j + 1);
          value1.y = convert_float(src_temp[data_index]);
          ((float2*)input_value + i)[0] +=
              (value + value1) * filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        j = index_x;
        data_index = interpolateReflect101Border(cols, radius, j);
        value.x = convert_float(src_temp[data_index]);
        data_index = interpolateReflect101Border(cols, radius, j + 1);
        value.y = convert_float(src_temp[data_index]);
        ((float2*)input_value + i)[0] +=
            (value)*filter_kernel[filter_kernel_index];
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
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

#if defined(GAUSSIANBLUR_U8C3) || defined(ALL_KERNELS)
__kernel
void gaussianblurU8F32C3Kernel(global const uchar* src, int src_offset,
                               int rows, int cols,
                               global const float* filter_kernel,
                               int kernel_offset, int radius, int src_stride,
                               global float* dst, int dst_stride,
                               int dst_offset,
                               enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x, index_y = element_y * 1;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((uchar*)src + src_offset);
  dst = (global float*)((uchar*)dst + dst_offset);
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float3 input_value[1];
  bool isnt_border_block = true;
  data_index = radius >> 5;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 5;
  if (group_x >= data_index)
    isnt_border_block = false;
  int data_index0;
  for (int i = 0; i < min(remain_rows, 1); i++) {
    filter_kernel_index = 0;
    ((float3*)input_value + i)[0] = (float3)(0);
    if (isnt_border_block) {
      for (int j = radius; j > 0; j--) {
        ((float3*)input_value + i)[0] +=
            (convert_float3(vload3(index_x - j, src)) +
             convert_float3(vload3(index_x + j, src))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float3*)input_value + i)[0] += convert_float3(vload3(index_x, src)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReplicateBorder(cols, radius, index_x - j);
          data_index0 = interpolateReplicateBorder(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReplicateBorder(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflectBorder(cols, radius, index_x - j);
          data_index0 = interpolateReflectBorder(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflectBorder(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflect101Border(cols, radius, index_x - j);
          data_index0 = interpolateReflect101Border(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflect101Border(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
    }
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
  if (remain_rows >= 1) {
    for (int i = 0; i < 1; i++) {
      vstore3(convert_float3(input_value[i]), index_y + i, dst);
    }
  }
}

__kernel
void gaussianblurF32U8C3Kernel(global const float* src, int src_offset,
                               int rows, int cols,
                               global const float* filter_kernel,
                               int kernel_offset, int radius, int src_stride,
                               global uchar* dst, int dst_stride,
                               int dst_offset,
                               enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((uchar*)src + src_offset);
  dst = (global uchar*)((uchar*)dst + dst_offset);
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float3 input_value[4];
  bool isnt_border_block = true;
  data_index = radius >> 5;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 5;
  if (group_x >= data_index)
    isnt_border_block = false;
  int data_index0;
  for (int i = 0; i < min(remain_rows, 4); i++) {
    filter_kernel_index = 0;
    ((float3*)input_value + i)[0] = (float3)(0);
    if (isnt_border_block) {
      for (int j = radius; j > 0; j--) {
        ((float3*)input_value + i)[0] +=
            (convert_float3(vload3(index_x - j, src)) +
             convert_float3(vload3(index_x + j, src))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float3*)input_value + i)[0] += convert_float3(vload3(index_x, src)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReplicateBorder(cols, radius, index_x - j);
          data_index0 = interpolateReplicateBorder(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReplicateBorder(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflectBorder(cols, radius, index_x - j);
          data_index0 = interpolateReflectBorder(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflectBorder(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflect101Border(cols, radius, index_x - j);
          data_index0 = interpolateReflect101Border(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflect101Border(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);
  if (remain_rows >= 4) {
    for (int i = 0; i < 4; i++) {
      vstore3(convert_uchar3_sat(input_value[i]), index_y + i, dst);
    }
  }
  else if (remain_rows == 1) {
    vstore3(convert_uchar3_sat(input_value[0]), index_y, dst);
  }
  else if (remain_rows == 2) {
    vstore3(convert_uchar3_sat(input_value[0]), index_y, dst);
    vstore3(convert_uchar3_sat(input_value[1]), index_y + 1, dst);
  }
  else if (remain_rows == 3) {
    vstore3(convert_uchar3_sat(input_value[0]), index_y, dst);
    vstore3(convert_uchar3_sat(input_value[1]), index_y + 1, dst);
    vstore3(convert_uchar3_sat(input_value[2]), index_y + 2, dst);
  }
}
#endif

#if defined(GAUSSIANBLUR_F32C3) || defined(ALL_KERNELS)
__kernel
void gaussianblurF32F32C3Kernel(global const float* src, int src_offset,
                                int rows, int cols,
                                global const float* filter_kernel,
                                int kernel_offset, int radius, int src_stride,
                                global float* dst, int dst_stride,
                                int dst_offset,
                                enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x, index_y = element_y * 1;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((uchar*)src + src_offset);
  dst = (global float*)((uchar*)dst + dst_offset);
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float3 input_value[1];
  bool isnt_border_block = true;
  data_index = radius >> 5;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 5;
  if (group_x >= data_index)
    isnt_border_block = false;
  int data_index0;
  for (int i = 0; i < min(remain_rows, 1); i++) {
    filter_kernel_index = 0;
    ((float3*)input_value + i)[0] = (float3)(0);
    if (isnt_border_block) {
      for (int j = radius; j > 0; j--) {
        ((float3*)input_value + i)[0] +=
            (convert_float3(vload3(index_x - j, src)) +
             convert_float3(vload3(index_x + j, src))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float3*)input_value + i)[0] += convert_float3(vload3(index_x, src)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReplicateBorder(cols, radius, index_x - j);
          data_index0 = interpolateReplicateBorder(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReplicateBorder(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflectBorder(cols, radius, index_x - j);
          data_index0 = interpolateReflectBorder(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflectBorder(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflect101Border(cols, radius, index_x - j);
          data_index0 = interpolateReflect101Border(cols, radius, index_x + j);
          ((float3*)input_value + i)[0] +=
              (convert_float3(vload3(data_index, src)) +
               convert_float3(vload3(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflect101Border(cols, radius, index_x);
        ((float3*)input_value + i)[0] +=
            convert_float3(vload3(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
  if (remain_rows >= 1) {
    for (int i = 0; i < 1; i++) {
      vstore3(convert_float3(input_value[i]), index_y + i, dst);
    }
  }
}
#endif

#if defined(GAUSSIANBLUR_U8C4) || defined(ALL_KERNELS)
__kernel
void gaussianblurU8F32C4Kernel(global const uchar* src, int src_offset,
                               int rows, int cols,
                               global const float* filter_kernel,
                               int kernel_offset, int radius, int src_stride,
                               global float* dst, int dst_stride,
                               int dst_offset,
                               enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x, index_y = element_y * 1;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((uchar*)src + src_offset);
  dst = (global float*)((uchar*)dst + dst_offset);
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float4 input_value[1];
  bool isnt_border_block = true;
  data_index = radius >> 5;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 5;
  if (group_x >= data_index)
    isnt_border_block = false;
  int data_index0;
  for (int i = 0; i < min(remain_rows, 1); i++) {
    filter_kernel_index = 0;
    ((float4*)input_value + i)[0] = (float4)(0);
    if (isnt_border_block) {
      for (int j = radius; j > 0; j--) {
        ((float4*)input_value + i)[0] +=
            (convert_float4(vload4(index_x - j, src)) +
             convert_float4(vload4(index_x + j, src))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float4*)input_value + i)[0] += convert_float4(vload4(index_x, src)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReplicateBorder(cols, radius, index_x - j);
          data_index0 = interpolateReplicateBorder(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReplicateBorder(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflectBorder(cols, radius, index_x - j);
          data_index0 = interpolateReflectBorder(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflectBorder(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflect101Border(cols, radius, index_x - j);
          data_index0 = interpolateReflect101Border(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflect101Border(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
    }
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
  if (remain_rows >= 1) {
    for (int i = 0; i < 1; i++) {
      vstore4(convert_float4(input_value[i]), index_y + i, dst);
    }
  }
}

__kernel
void gaussianblurF32U8C4Kernel(global const float* src, int src_offset,
                               int rows, int cols,
                               global const float* filter_kernel,
                               int kernel_offset, int radius, int src_stride,
                               global uchar* dst, int dst_stride,
                               int dst_offset,
                               enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((uchar*)src + src_offset);
  dst = (global uchar*)((uchar*)dst + dst_offset);
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float4 input_value[4];
  bool isnt_border_block = true;
  data_index = radius >> 5;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 5;
  if (group_x >= data_index)
    isnt_border_block = false;
  int data_index0;
  for (int i = 0; i < min(remain_rows, 4); i++) {
    filter_kernel_index = 0;
    ((float4*)input_value + i)[0] = (float4)(0);
    if (isnt_border_block) {
      for (int j = radius; j > 0; j--) {
        ((float4*)input_value + i)[0] +=
            (convert_float4(vload4(index_x - j, src)) +
             convert_float4(vload4(index_x + j, src))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float4*)input_value + i)[0] += convert_float4(vload4(index_x, src)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReplicateBorder(cols, radius, index_x - j);
          data_index0 = interpolateReplicateBorder(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReplicateBorder(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflectBorder(cols, radius, index_x - j);
          data_index0 = interpolateReflectBorder(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflectBorder(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflect101Border(cols, radius, index_x - j);
          data_index0 = interpolateReflect101Border(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflect101Border(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x);
  if (remain_rows >= 4) {
    for (int i = 0; i < 4; i++) {
      vstore4(convert_uchar4_sat(input_value[i]), index_y + i, dst);
    }
  }
  else if (remain_rows == 1) {
    vstore4(convert_uchar4_sat(input_value[0]), index_y, dst);
  }
  else if (remain_rows == 2) {
    vstore4(convert_uchar4_sat(input_value[0]), index_y, dst);
    vstore4(convert_uchar4_sat(input_value[1]), index_y + 1, dst);
  }
  else if (remain_rows == 3) {
    vstore4(convert_uchar4_sat(input_value[0]), index_y, dst);
    vstore4(convert_uchar4_sat(input_value[1]), index_y + 1, dst);
    vstore4(convert_uchar4_sat(input_value[2]), index_y + 2, dst);
  }
}
#endif

#if defined(GAUSSIANBLUR_F32C4) || defined(ALL_KERNELS)
__kernel
void gaussianblurF32F32C4Kernel(global const float* src, int src_offset,
                                int rows, int cols,
                                global const float* filter_kernel,
                                int kernel_offset, int radius, int src_stride,
                                global float* dst, int dst_stride,
                                int dst_offset,
                                enum BorderType interpolate_type) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  int index_x = element_x, index_y = element_y * 1;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((uchar*)src + src_offset);
  dst = (global float*)((uchar*)dst + dst_offset);
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  filter_kernel =
      (global const float*)((global uchar*)filter_kernel + kernel_offset);
  int remain_rows = rows - index_y;
  int bottom = index_x - radius;
  int top = index_x + radius;
  int filter_kernel_index;
  int data_index;
  float4 input_value[1];
  bool isnt_border_block = true;
  data_index = radius >> 5;
  if (group_x <= data_index)
    isnt_border_block = false;
  data_index = (cols - radius) >> 5;
  if (group_x >= data_index)
    isnt_border_block = false;
  int data_index0;
  for (int i = 0; i < min(remain_rows, 1); i++) {
    filter_kernel_index = 0;
    ((float4*)input_value + i)[0] = (float4)(0);
    if (isnt_border_block) {
      for (int j = radius; j > 0; j--) {
        ((float4*)input_value + i)[0] +=
            (convert_float4(vload4(index_x - j, src)) +
             convert_float4(vload4(index_x + j, src))) *
            filter_kernel[filter_kernel_index];
        filter_kernel_index++;
      }
      ((float4*)input_value + i)[0] += convert_float4(vload4(index_x, src)) *
                                       filter_kernel[filter_kernel_index];
    }
    else {
      if (interpolate_type == BORDER_REPLICATE) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReplicateBorder(cols, radius, index_x - j);
          data_index0 = interpolateReplicateBorder(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReplicateBorder(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflectBorder(cols, radius, index_x - j);
          data_index0 = interpolateReflectBorder(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflectBorder(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
      else if (interpolate_type == BORDER_REFLECT_101) {
        for (int j = radius; j > 0; j--) {
          data_index = interpolateReflect101Border(cols, radius, index_x - j);
          data_index0 = interpolateReflect101Border(cols, radius, index_x + j);
          ((float4*)input_value + i)[0] +=
              (convert_float4(vload4(data_index, src)) +
               convert_float4(vload4(data_index0, src))) *
              filter_kernel[filter_kernel_index];
          filter_kernel_index++;
        }
        data_index = interpolateReflect101Border(cols, radius, index_x);
        ((float4*)input_value + i)[0] +=
            convert_float4(vload4(data_index, src)) *
            filter_kernel[filter_kernel_index];
      }
    }
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x);
  if (remain_rows >= 1) {
    for (int i = 0; i < 1; i++) {
      vstore4(convert_float4(input_value[i]), index_y + i, dst);
    }
  }
}
#endif