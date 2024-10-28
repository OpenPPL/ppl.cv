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

#if defined(EQUALIZEHIST_ALIGNED) || defined(EQUALIZEHIST_UNALIGNED) ||        \
    defined(ALL_KERNELS)
static global int group_count = 0;
__kernel
void equalizeHistKernel(global int* hist, int hist_offset) {
  int element_x = get_global_id(0);
  hist = (global int*)((uchar*)hist + hist_offset);
  hist[element_x] = 0;
  group_count = 0;
}
#endif

#if defined(EQUALIZEHIST_ALIGNED) || defined(ALL_KERNELS)
__kernel
void equalizeHistKernel0(global const uchar* src, const int cols,
                         global int* hist, int hist_offset) {
  int element_x = get_global_id(0);
  int index_x;
  int local_x = get_local_id(0);
  int offset = get_num_groups(0) * 256;
  local int local_hist[256];
  local_hist[local_x] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  hist = (global int*)((uchar*)hist + hist_offset);

  uchar4 value;
  for (; (element_x << 2) < cols; element_x += offset) {
    index_x = element_x << 2;
    value = vload4(element_x, src);
    if (index_x < cols - 3) {
      atomic_add(&local_hist[value.x], 1);
      atomic_add(&local_hist[value.y], 1);
      atomic_add(&local_hist[value.z], 1);
      atomic_add(&local_hist[value.w], 1);
    }
    else {
      atomic_add(&local_hist[value.x], 1);
      if (index_x < cols - 1) {
        atomic_add(&local_hist[value.y], 1);
      }
      if (index_x < cols - 2) {
        atomic_add(&local_hist[value.z], 1);
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  int count = local_hist[local_x];
  if (count > 0) {
    atomic_add(&hist[local_x], count);
  }
  local_hist[local_x] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  __local bool is_last_group;
  if (get_local_id(0) == 0) {
    uint local_count = atomic_inc(&group_count);
    is_last_group = (local_count == (get_num_groups(0) - 1));
    if (is_last_group) {
      int i = 0;
      while (!hist[i])
        ++i;
      float scale = (256 - 1.f) / (cols - hist[i]);
      int sum = 0;
      for (local_hist[i++] = 0; i < 256; ++i) {
        sum += hist[i];
        local_hist[i] = convert_int(sum * scale + 0.5f);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (is_last_group) {
    hist[local_x] = local_hist[local_x];
  }
}
#endif

#if defined(EQUALIZEHIST_ALIGNED) || defined(ALL_KERNELS)
__kernel
void equalizeHistKernel00(global const uchar* src, const int cols,
                          global uchar* dst, global int* hist,
                          int hist_offset) {
  local uchar local_hist[256];
  int element_x = get_global_id(0);
  int index_x;
  int local_x = get_local_id(0);
  int offset = get_num_groups(0) * 256;
  local_hist[local_x] = convert_int(hist[local_x]);
  barrier(CLK_LOCAL_MEM_FENCE);
  hist = (global int*)((uchar*)hist + hist_offset);

  uchar4 output;
  uchar4 value;
  for (; (element_x << 2) < cols; element_x += offset) {
    index_x = element_x << 2;
    value = vload4(element_x, src);
    if (index_x < cols - 3) {
      output.x = local_hist[value.x];
      output.y = local_hist[value.y];
      output.z = local_hist[value.z];
      output.w = local_hist[value.w];
      vstore4(output, element_x, dst);
    }
    else {
      dst[index_x] = local_hist[value.x];
      if (index_x < cols - 1) {
        dst[index_x + 1] = local_hist[value.y];
      }
      if (index_x < cols - 2) {
        dst[index_x + 2] = local_hist[value.z];
      }
    }
  }
}
#endif

#if defined(EQUALIZEHIST_UNALIGNED) || defined(ALL_KERNELS)
__kernel
void equalizeHistKernel1(global const uchar* src, int src_stride,
                         const int rows, const int cols, global int* hist,
                         int hist_offset) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;

  int index_hist = get_local_id(0) + get_local_id(1) * get_local_size(0);
  int offset = get_global_size(1);
  local int local_hist[256];
  local_hist[index_hist] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  hist = (global int*)((uchar*)hist + hist_offset);

  uchar4 value;
  global const uchar* src_ptr;
  for (; element_y < rows; element_y += offset) {
    if (index_x < cols) {
      src_ptr = src + element_y * src_stride;
      value = vload4(element_x, src_ptr);
      if (index_x < cols - 3) {
        atomic_add(&local_hist[value.x], 1);
        atomic_add(&local_hist[value.y], 1);
        atomic_add(&local_hist[value.z], 1);
        atomic_add(&local_hist[value.w], 1);
      }
      else {
        atomic_add(&local_hist[value.x], 1);
        if (index_x < cols - 1) {
          atomic_add(&local_hist[value.y], 1);
        }
        if (index_x < cols - 2) {
          atomic_add(&local_hist[value.z], 1);
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  int count = local_hist[index_hist];
  if (count > 0) {
    atomic_add(&hist[index_hist], count);
  }
  local_hist[index_hist] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  __local bool is_last_group;
  int elements = cols * rows;
  if (get_local_id(0) == 0 && get_local_id(1) == 0) {
    uint local_count = atomic_inc(&group_count);
    is_last_group =
        (local_count == (get_num_groups(0) * get_num_groups(1) - 1));
    if (is_last_group) {
      int i = 0;
      while (!hist[i])
        ++i;
      float scale = (256 - 1.f) / (elements - hist[i]);
      int sum = 0;
      for (local_hist[i++] = 0; i < 256; ++i) {
        sum += hist[i];
        local_hist[i] = convert_int(sum * scale + 0.5f);
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (is_last_group) {
    hist[index_hist] = local_hist[index_hist];
  }
}
#endif

#if defined(EQUALIZEHIST_UNALIGNED) || defined(ALL_KERNELS)
__kernel
void equalizeHistKernel11(global const uchar* src, int src_stride,
                          const int rows, const int cols, global uchar* dst,
                          int dst_stride, global int* hist, int hist_offset) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;

  int index_hist = get_local_id(0) + get_local_id(1) * get_local_size(0);
  int offset = get_global_size(1);
  local int local_hist[256];
  local_hist[index_hist] = convert_int(hist[index_hist]);
  barrier(CLK_LOCAL_MEM_FENCE);
  hist = (global int*)((uchar*)hist + hist_offset);

  uchar4 output;
  uchar4 value;
  global const uchar* src_ptr;
  global uchar* dst_ptr;
  for (; element_y < rows; element_y += offset) {
    if (index_x < cols) {
      src_ptr = src + element_y * src_stride;
      dst_ptr = dst + element_y * dst_stride;
      value = vload4(element_x, src_ptr);
      if (index_x < cols - 3) {
        output.x = local_hist[value.x];
        output.y = local_hist[value.y];
        output.z = local_hist[value.z];
        output.w = local_hist[value.w];
        vstore4(output, element_x, dst_ptr);
      }
      else {
        dst_ptr[index_x] = local_hist[value.x];
        if (index_x < cols - 1) {
          dst_ptr[index_x + 1] = local_hist[value.y];
        }
        if (index_x < cols - 2) {
          dst_ptr[index_x + 2] = local_hist[value.z];
        }
      }
    }
  }
}
#endif