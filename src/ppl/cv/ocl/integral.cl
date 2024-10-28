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

#if defined(INTEGRAL_U8) || defined(ALL_KERNELS)
__kernel 
void setZeroI32(global int* dst, int offset, int cols) {
  int index_x = get_global_id(0);
  index_x <<= 1;
  dst = (global int*)((global uchar*)dst + offset);
  if (index_x >= cols) {
    return;
  }
  if (cols - index_x >= 2) {
    vstore2((int2)(0.0f, 0.0f), get_global_id(0), dst);
  }
  else {
    dst[index_x] = 0;
  }
}
#endif

#if defined(INTEGRAL_F32) || defined(ALL_KERNELS)
__kernel 
void setZeroF32(global float* dst, int offset, int cols) {
  int index_x = get_global_id(0);
  index_x <<= 1;
  dst = (global float*)((global uchar*)dst + offset);
  if (index_x >= cols) {
    return;
  }
  if (cols - index_x >= 2) {
    vstore2((float2)(0.0f, 0.0f), get_global_id(0), dst);
  }
  else {
    dst[index_x] = 0;
  }
}
#endif

#if defined(INTEGRAL_U8) || defined(ALL_KERNELS)
__kernel 
void integralU8I32Kernel(global const uchar* src, int src_offset, int src_rows, int src_cols,
                         int src_stride, global int* dst, int dst_offset, int dst_rows,
                         int dst_cols, int dst_stride) {
  int element_y = get_global_id(1);
  int index_y = element_y * 2;
  if (index_y >= src_rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + src_offset);
  dst = (global int*)((global uchar*)dst + dst_offset);
  if (src_rows == dst_cols) {
    dst_offset = 0;
  }
  else {
    dst_offset = 1;
  }
  global const uchar* src_tmp;
  int remain_cols = src_cols, remain_rows = src_rows - index_y;
  int2 input_value[2];
  global int* dst_tmp = dst + dst_offset;
  dst_tmp =
      (global int*)((global uchar*)dst_tmp + dst_stride * (dst_offset));
  if (get_global_id(0) == 0) {
    for (int i = 0; i < min(remain_rows, 2); i++) {
      dst_tmp[0] = 0;
      dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
    }
  }
  for (int i = 0; i < 2; i++) {
    input_value[i] = (int2)0;
  }
  src = (global const uchar*)((global uchar*)src + src_stride * index_y);
  int input_sum[2] = {0};
  int x_offset = 2;
  dst_tmp = dst + dst_offset;
  dst_tmp =
      (global int*)((global uchar*)dst_tmp + dst_stride * (dst_offset));
  global int* dst_tmp_prev;
  while (remain_cols > 0) {
    for (int i = 0; i < 2; i++) {
      input_sum[i] = input_value[i].y;
    }
    {
      int i = 0;
      src_tmp = src + i;
      for (int j = 0; j < min(remain_rows, 2); j++) {
        ((int2*)input_value + j)[0] = convert_int2(vload2(0, src_tmp));
        src_tmp = (global const uchar*)((global uchar*)src_tmp + src_stride);
        ((int2*)input_value + j)[0].x += ((int*)input_sum + j)[0];
        ((int2*)input_value + j)[0].y += ((int2*)input_value + j)[0].x;
      }
    }
    dst_tmp_prev = dst_tmp;
    if (remain_rows >= 2) {
      if (remain_cols >= 2) {
        int2 output_value[2];
        output_value[0] = (int2)(input_value[0].x, input_value[1].x);
        output_value[1] = (int2)(input_value[0].y, input_value[1].y);
        for (int k = 0; k < 2; k++) {
          vstore2(output_value[k], element_y, dst_tmp);
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
      else if (remain_cols == 1) {
        int2 output_value[1];
        output_value[0] = (int2)(input_value[0].x, input_value[1].x);
        for (int k = 0; k < 1; k++) {
          vstore2(output_value[k], element_y, dst_tmp);
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
    }
    else if (remain_rows == 1) {
      if (remain_cols >= 2) {
        int output_value[2];
        output_value[0] = (int)(input_value[0].x);
        output_value[1] = (int)(input_value[0].y);
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 2;
          dst_tmp[offset] = output_value[k];
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
      else if (remain_cols == 1) {
        int output_value[1];
        output_value[0] = (int)(input_value[0].x);
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 2;
          dst_tmp[offset] = output_value[k];
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
    }
    dst_tmp = (global int*)((global uchar*)dst_tmp_prev + dst_stride * (x_offset));
    src += x_offset;
    remain_cols = remain_cols - x_offset;
  }
}
#endif

#if defined(INTEGRAL_U8) || defined(ALL_KERNELS)
__kernel 
void integralI32I32Kernel(global const int* src, int src_offset, int src_rows, int src_cols,
                          int src_stride, global int* dst, int dst_offset, int dst_rows,
                          int dst_cols, int dst_stride) {
  int element_y = get_global_id(1);
  
  int index_y = element_y * 2;
  if (index_y >= src_rows) {
    return;
  }
  src = (global const int*)((global uchar*)src + src_offset);
  dst = (global int*)((global uchar*)dst + dst_offset);
  if (src_rows == dst_cols) {
    dst_offset = 0;
  }
  else {
    dst_offset = 1;
  }
  global const int* src_tmp;
  int remain_cols = src_cols, remain_rows = src_rows - index_y;
  int2 input_value[2];
  global int* dst_tmp = dst + dst_offset;
  dst_tmp =
      (global int*)((global uchar*)dst_tmp + dst_stride * (dst_offset));
  __local int prev_sum[2];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < min(remain_rows, 2); i++) {
      dst_tmp[0] = 0;
      dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
    }
  }
  for (int i = 0; i < 2; i++) {
    input_value[i] = (int2)0;
  }
  src = (global const int*)((global uchar*)src + src_stride * index_y);
  int input_sum[2] = {0};
  int x_offset = 2;
  dst_tmp = dst + dst_offset;
  dst_tmp =
      (global int*)((global uchar*)dst_tmp + dst_stride * (dst_offset));
  global int* dst_tmp_prev;
  while (remain_cols > 0) {
    for (int i = 0; i < 2; i++) {
      input_sum[i] = input_value[i].y;
    }
    {
      int i = 0;
      src_tmp = src + i;
      for (int j = 0; j < min(remain_rows, 2); j++) {
        ((int2*)input_value + j)[0] = convert_int2(vload2(0, src_tmp));
        src_tmp = (global const int*)((global uchar*)src_tmp + src_stride);
        ((int2*)input_value + j)[0].x += ((int*)input_sum + j)[0];
        ((int2*)input_value + j)[0].y += ((int2*)input_value + j)[0].x;
      }
    }
    dst_tmp_prev = dst_tmp;
    if (remain_rows >= 2) {
      if (remain_cols >= 2) {
        int2 output_value[2];
        output_value[0] = (int2)(input_value[0].x, input_value[1].x);
        output_value[1] = (int2)(input_value[0].y, input_value[1].y);
        for (int k = 0; k < 2; k++) {
          vstore2(output_value[k], element_y, dst_tmp);
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
      else if (remain_cols == 1) {
        int2 output_value[1];
        output_value[0] = (int2)(input_value[0].x, input_value[1].x);
        for (int k = 0; k < 1; k++) {
          vstore2(output_value[k], element_y, dst_tmp);
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
    }
    else if (remain_rows == 1) {
      if (remain_cols >= 2) {
        int output_value[2];
        output_value[0] = (int)(input_value[0].x);
        output_value[1] = (int)(input_value[0].y);
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 2;
          dst_tmp[offset] = output_value[k];
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
      else if (remain_cols == 1) {
        int output_value[1];
        output_value[0] = (int)(input_value[0].x);
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 2;
          dst_tmp[offset] = output_value[k];
          dst_tmp = (global int*)((global uchar*)dst_tmp + dst_stride);
        }
      }
    }
    dst_tmp = (global int*)((global uchar*)dst_tmp_prev + dst_stride * (x_offset));
    src += x_offset;
    remain_cols = remain_cols - x_offset;
  }
}
#endif

#if defined(INTEGRAL_F32) || defined(ALL_KERNELS)
__kernel 
void integralF32F32Kernel(global const float* src, int src_offset, int src_rows, int src_cols,
                          int src_stride, global float* dst, int dst_offset, int dst_rows,
                          int dst_cols, int dst_stride) {
  int element_y = get_global_id(1);
  int index_y = element_y * 2;
  if (index_y >= src_rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + src_offset);
  dst = (global float*)((global uchar*)dst + dst_offset);
  if (src_rows == dst_cols) {
    dst_offset = 0;
  }
  else {
    dst_offset = 1;
  }
  global const float* src_tmp;
  int remain_cols = src_cols, remain_rows = src_rows - index_y;
  float2 input_value[2];
  global float* dst_tmp = dst + dst_offset;
  dst_tmp =
      (global float*)((global uchar*)dst_tmp + dst_stride * (dst_offset));
  if (get_global_id(0) == 0) {
    for (int i = 0; i < min(remain_rows, 2); i++) {
      dst_tmp[0] = 0;
      dst_tmp = (global float*)((global uchar*)dst_tmp + dst_stride);
    }
  }
  for (int i = 0; i < 2; i++) {
    input_value[i] = (float2)0;
  }
  src = (global const float*)((global uchar*)src + src_stride * index_y);
  float input_sum[2] = {0};
  int x_offset = 2;
  dst_tmp = dst + dst_offset;
  dst_tmp =
      (global float*)((global uchar*)dst_tmp + dst_stride * (dst_offset));
  global float* dst_tmp_prev;
  while (remain_cols > 0) {
    for (int i = 0; i < 2; i++) {
      input_sum[i] = input_value[i].y;
    }
    {
      int i = 0;
      src_tmp = src + i;
      for (int j = 0; j < min(remain_rows, 2); j++) {
        ((float2*)input_value + j)[0] = convert_float2(vload2(0, src_tmp));
        src_tmp = (global const float*)((global uchar*)src_tmp + src_stride);
        ((float2*)input_value + j)[0].x += ((float*)input_sum + j)[0];
        ((float2*)input_value + j)[0].y += ((float2*)input_value + j)[0].x;
      }
    }
    dst_tmp_prev = dst_tmp;
    if (remain_rows >= 2) {
      if (remain_cols >= 2) {
        float2 output_value[2];
        output_value[0] = (float2)(input_value[0].x, input_value[1].x);
        output_value[1] = (float2)(input_value[0].y, input_value[1].y);
        for (int k = 0; k < 2; k++) {
          vstore2(output_value[k], element_y, dst_tmp);
          dst_tmp = (global float*)((global uchar*)dst_tmp + dst_stride);
        }
      }
      else if (remain_cols == 1) {
        float2 output_value[1];
        output_value[0] = (float2)(input_value[0].x, input_value[1].x);
        for (int k = 0; k < 1; k++) {
          vstore2(output_value[k], element_y, dst_tmp);
          dst_tmp = (global float*)((global uchar*)dst_tmp + dst_stride);
        }
      }
    }
    else if (remain_rows == 1) {
      if (remain_cols >= 2) {
        float output_value[2];
        output_value[0] = (float)(input_value[0].x);
        output_value[1] = (float)(input_value[0].y);
        for (int k = 0; k < 2; k++) {
          int offset = element_y * 2;
          dst_tmp[offset] = output_value[k];
          dst_tmp = (global float*)((global uchar*)dst_tmp + dst_stride);
        }
      }
      else if (remain_cols == 1) {
        float output_value[1];
        output_value[0] = (float)(input_value[0].x);
        for (int k = 0; k < 1; k++) {
          int offset = element_y * 2;
          dst_tmp[offset] = output_value[k];
          dst_tmp = (global float*)((global uchar*)dst_tmp + dst_stride);
        }
      }
    }
    dst_tmp = (global float*)((global uchar*)dst_tmp_prev + dst_stride * (x_offset));
    src += x_offset;
    remain_cols = remain_cols - x_offset;
  }
}
#endif