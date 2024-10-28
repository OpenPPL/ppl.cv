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

/******************************* calchist operation *******************************/

#if defined(CALCHIST_UNMAKED_ALIGHED) || defined(ALL_KERNELS)
__kernel
void unmaskCalchistKernel0(global const uchar* src, const int cols,
                           global int* hist) {
  int element_x = get_global_id(0);
  int index_x;
  int local_x = get_local_id(0);
  int offset = get_num_groups(0) * 256;
  local int local_hist[256];
  local_hist[local_x] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

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
}
#endif

#if defined(CALCHIST_UNMAKED_UNALIGHED) || defined(ALL_KERNELS)
__kernel
void unmaskCalchistKernel1(global const uchar* src, int src_stride,
                           const int rows, const int cols, global int* hist) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;

  int index_hist = get_local_id(0) + get_local_id(1) * get_local_size(0);
  int offset = get_global_size(1);
  local int local_hist[256];
  local_hist[index_hist] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

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
}
#endif

#if defined(CALCHIST_MAKED_ALIGHED) || defined(ALL_KERNELS)
__kernel
void maskCalchistKernel0(global const uchar* src, global const uchar* mask,
                         const int cols, global int* hist) {
  int element_x = get_global_id(0);
  int index_x;
  int local_x = get_local_id(0);
  int offset = get_num_groups(0) * 256;
  local int local_hist[256];
  local_hist[local_x] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  uchar4 value, value_mask;
  for (; (element_x << 2) < cols; element_x += offset) {
    index_x = element_x << 2;
    value = vload4(element_x, src);
    value_mask = vload4(element_x, mask);
    if (index_x < cols - 3) {
      if (value_mask.x) {
        atomic_add(&local_hist[value.x], 1);
      }
      if (value_mask.y) {
        atomic_add(&local_hist[value.y], 1);
      }
      if (value_mask.z) {
        atomic_add(&local_hist[value.z], 1);
      }
      if (value_mask.w) {
        atomic_add(&local_hist[value.w], 1);
      }
    }
    else {
      if (value_mask.x) {
        atomic_add(&local_hist[value.x], 1);
      }
      if (index_x < cols - 1) {
        if (value_mask.y) {
          atomic_add(&local_hist[value.y], 1);
        }
      }
      if (index_x < cols - 2) {
        if (value_mask.z) {
          atomic_add(&local_hist[value.z], 1);
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  int count = local_hist[local_x];
  if (count > 0) {
    atomic_add(&hist[local_x], count);
  }
}
#endif

#if defined(CALCHIST_MAKED_UNALIGHED) || defined(ALL_KERNELS)
__kernel
void maskCalchistKernel1(global const uchar* src, int src_stride,
                         global const uchar* mask, int mask_stride,
                         const int rows, const int cols, global int* hist) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;

  int index_hist = get_local_id(0) + get_local_id(1) * get_local_size(0);
  int offset = get_global_size(1);
  local int local_hist[256];
  local_hist[index_hist] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  uchar4 value, value_mask;
  global const uchar *src_ptr, *mask_ptr;
  for (; element_y < rows; element_y += offset) {
    if (index_x < cols) {
      src_ptr = src + element_y * src_stride;
      mask_ptr = mask + element_y * mask_stride;
      value = vload4(element_x, src_ptr);
      value_mask = vload4(element_x, mask_ptr);

      if (index_x < cols - 3) {
        if (value_mask.x) {
          atomic_add(&local_hist[value.x], 1);
        }
        if (value_mask.y) {
          atomic_add(&local_hist[value.y], 1);
        }
        if (value_mask.z) {
          atomic_add(&local_hist[value.z], 1);
        }
        if (value_mask.w) {
          atomic_add(&local_hist[value.w], 1);
        }
      }
      else {
        if (value_mask.x) {
          atomic_add(&local_hist[value.x], 1);
        }
        if (index_x < cols - 1) {
          if (value_mask.y) {
            atomic_add(&local_hist[value.y], 1);
          }
        }
        if (index_x < cols - 2) {
          if (value_mask.z) {
            atomic_add(&local_hist[value.z], 1);
          }
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  int count = local_hist[index_hist];
  if (count > 0) {
    atomic_add(&hist[index_hist], count);
  }
}
#endif
