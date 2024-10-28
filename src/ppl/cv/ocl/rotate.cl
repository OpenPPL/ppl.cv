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

#if defined(ROTATE90_U8C1) || defined(ALL_KERNELS)
__kernel
void rotateC190U8Kernel(global const uchar* src, int rows, int cols,
                        int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x * 4, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int cols_remained = cols - index_x, rows_remained = rows - index_y;
  uchar4 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x) +
        max(rows - index_y - 4, 0);
  if (cols_remained >= 4) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[3].x, input_value[2].x,
                              input_value[1].x, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[3].y, input_value[2].y,
                              input_value[1].y, input_value[0].y);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[3].z, input_value[2].z,
                              input_value[1].z, input_value[0].z);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[3].w, input_value[2].w,
                              input_value[1].w, input_value[0].w);
      vstore4(output_value, 0, dst);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].y);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].z);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].w);
      dst[0] = output_value;
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[1].x, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].y, input_value[0].y);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].z, input_value[0].z);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].w, input_value[0].w);
      vstore2(output_value, 0, dst);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[2].x, input_value[1].x, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[2].y, input_value[1].y, input_value[0].y);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[2].z, input_value[1].z, input_value[0].z);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[2].w, input_value[1].w, input_value[0].w);
      vstore3(output_value, 0, dst);
    }
  }
  else if (cols_remained == 1) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[3].x, input_value[2].x,
                              input_value[1].x, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[1].x, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[2].x, input_value[1].x, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 2) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[3].x, input_value[2].x,
                              input_value[1].x, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[3].y, input_value[2].y,
                              input_value[1].y, input_value[0].y);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].y);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[1].x, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].y, input_value[0].y);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[2].x, input_value[1].x, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[2].y, input_value[1].y, input_value[0].y);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 3) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[3].x, input_value[2].x,
                              input_value[1].x, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[3].y, input_value[2].y,
                              input_value[1].y, input_value[0].y);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[3].z, input_value[2].z,
                              input_value[1].z, input_value[0].z);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].y);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].z);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[1].x, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].y, input_value[0].y);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].z, input_value[0].z);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[2].x, input_value[1].x, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[2].y, input_value[1].y, input_value[0].y);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[2].z, input_value[1].z, input_value[0].z);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
}
#endif

#if defined(ROTATE180_U8C1) || defined(ALL_KERNELS)
__kernel
void rotateC1180U8Kernel(global const uchar* src, int rows, int cols,
                         int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x * 4, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int cols_remained = cols - index_x, rows_remained = rows - index_y;
  uchar4 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst +
                        dst_stride * max(rows - index_y - 4, 0)) +
        max(cols - index_x - 4, 0);
  if (cols_remained >= 4) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[3].w, input_value[3].z,
                              input_value[3].y, input_value[3].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[2].w, input_value[2].z,
                              input_value[2].y, input_value[2].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[1].w, input_value[1].z,
                              input_value[1].y, input_value[1].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].w, input_value[0].z,
                              input_value[0].y, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[0].w, input_value[0].z,
                              input_value[0].y, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[1].w, input_value[1].z,
                              input_value[1].y, input_value[1].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].w, input_value[0].z,
                              input_value[0].y, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[2].w, input_value[2].z,
                              input_value[2].y, input_value[2].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[1].w, input_value[1].z,
                              input_value[1].y, input_value[1].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].w, input_value[0].z,
                              input_value[0].y, input_value[0].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 1) {
    if (rows_remained >= 4) {
      uchar output_value;
      output_value = (uchar)(input_value[3].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[2].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[1].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar output_value;
      output_value = (uchar)(input_value[1].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar output_value;
      output_value = (uchar)(input_value[2].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[1].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 2) {
    if (rows_remained >= 4) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[3].y, input_value[3].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[2].y, input_value[2].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].y, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].y, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[0].y, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[1].y, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].y, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[2].y, input_value[2].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[1].y, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].y, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 3) {
    if (rows_remained >= 4) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[3].z, input_value[3].y, input_value[3].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[2].z, input_value[2].y, input_value[2].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[1].z, input_value[1].y, input_value[1].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].z, input_value[0].y, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[0].z, input_value[0].y, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[1].z, input_value[1].y, input_value[1].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].z, input_value[0].y, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[2].z, input_value[2].y, input_value[2].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[1].z, input_value[1].y, input_value[1].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].z, input_value[0].y, input_value[0].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
}
#endif

#if defined(ROTATE270_U8C1) || defined(ALL_KERNELS)
__kernel
void rotateC1270U8Kernel(global const uchar* src, int rows, int cols,
                         int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x * 4, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int cols_remained = cols - index_x, rows_remained = rows - index_y;
  uchar4 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst +
                        dst_stride * max(cols - index_x - 4, 0)) +
        index_y;
  if (cols_remained >= 4) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[0].w, input_value[1].w,
                              input_value[2].w, input_value[3].w);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].z, input_value[1].z,
                              input_value[2].z, input_value[3].z);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].y, input_value[1].y,
                              input_value[2].y, input_value[3].y);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].x, input_value[1].x,
                              input_value[2].x, input_value[3].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].w);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].z);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].y);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[0].w, input_value[1].w);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].z, input_value[1].z);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].y, input_value[1].y);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].x, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[0].w, input_value[1].w, input_value[2].w);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].z, input_value[1].z, input_value[2].z);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].y, input_value[1].y, input_value[2].y);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].x, input_value[1].x, input_value[2].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 1) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[0].x, input_value[1].x,
                              input_value[2].x, input_value[3].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[0].x, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[0].x, input_value[1].x, input_value[2].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 2) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[0].y, input_value[1].y,
                              input_value[2].y, input_value[3].y);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].x, input_value[1].x,
                              input_value[2].x, input_value[3].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].y);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[0].y, input_value[1].y);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].x, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[0].y, input_value[1].y, input_value[2].y);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].x, input_value[1].x, input_value[2].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 3) {
    if (rows_remained >= 4) {
      uchar4 output_value;
      output_value = (uchar4)(input_value[0].z, input_value[1].z,
                              input_value[2].z, input_value[3].z);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].y, input_value[1].y,
                              input_value[2].y, input_value[3].y);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar4)(input_value[0].x, input_value[1].x,
                              input_value[2].x, input_value[3].x);
      vstore4(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      uchar output_value;
      output_value = (uchar)(input_value[0].z);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].y);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar)(input_value[0].x);
      dst[0] = output_value;
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 2) {
      uchar2 output_value;
      output_value = (uchar2)(input_value[0].z, input_value[1].z);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].y, input_value[1].y);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value = (uchar2)(input_value[0].x, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 3) {
      uchar3 output_value;
      output_value =
          (uchar3)(input_value[0].z, input_value[1].z, input_value[2].z);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].y, input_value[1].y, input_value[2].y);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
      output_value =
          (uchar3)(input_value[0].x, input_value[1].x, input_value[2].x);
      vstore3(output_value, 0, dst);
      dst = (global uchar*)((global uchar*)dst + dst_stride);
    }
  }
}
#endif

#if defined(ROTATE90_F32C1) || defined(ALL_KERNELS)
__kernel
void rotateC190F32Kernel(global const float* src, int rows, int cols,
                         int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x * 2, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int cols_remained = cols - index_x, rows_remained = rows - index_y;
  float2 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload2(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x) +
        max(rows - index_y - 2, 0);
  if (cols_remained >= 2) {
    if (rows_remained >= 2) {
      float2 output_value;
      output_value = (float2)(input_value[1].x, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
      output_value = (float2)(input_value[1].y, input_value[0].y);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      float output_value;
      output_value = (float)(input_value[0].x);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
      output_value = (float)(input_value[0].y);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 1) {
    if (rows_remained >= 2) {
      float2 output_value;
      output_value = (float2)(input_value[1].x, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      float output_value;
      output_value = (float)(input_value[0].x);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
  }
}
#endif

#if defined(ROTATE180_F32C1) || defined(ALL_KERNELS)
__kernel
void rotateC1180F32Kernel(global const float* src, int rows, int cols,
                          int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x * 2, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int cols_remained = cols - index_x, rows_remained = rows - index_y;
  float2 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload2(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst +
                        dst_stride * max(rows - index_y - 2, 0)) +
        max(cols - index_x - 2, 0);
  if (cols_remained >= 2) {
    if (rows_remained >= 2) {
      float2 output_value;
      output_value = (float2)(input_value[1].y, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
      output_value = (float2)(input_value[0].y, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      float2 output_value;
      output_value = (float2)(input_value[0].y, input_value[0].x);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 1) {
    if (rows_remained >= 2) {
      float output_value;
      output_value = (float)(input_value[1].x);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
      output_value = (float)(input_value[0].x);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      float output_value;
      output_value = (float)(input_value[0].x);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
  }
}
#endif

#if defined(ROTATE270_F32C1) || defined(ALL_KERNELS)
__kernel
void rotateC1270F32Kernel(global const float* src, int rows, int cols,
                          int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x * 2, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int cols_remained = cols - index_x, rows_remained = rows - index_y;
  float2 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload2(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst +
                        dst_stride * max(cols - index_x - 2, 0)) +
        index_y;
  if (cols_remained >= 2) {
    if (rows_remained >= 2) {
      float2 output_value;
      output_value = (float2)(input_value[0].y, input_value[1].y);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
      output_value = (float2)(input_value[0].x, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      float output_value;
      output_value = (float)(input_value[0].y);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
      output_value = (float)(input_value[0].x);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
  }
  else if (cols_remained == 1) {
    if (rows_remained >= 2) {
      float2 output_value;
      output_value = (float2)(input_value[0].x, input_value[1].x);
      vstore2(output_value, 0, dst);
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
    else if (rows_remained == 1) {
      float output_value;
      output_value = (float)(input_value[0].x);
      dst[0] = output_value;
      dst = (global float*)((global uchar*)dst + dst_stride);
    }
  }
}
#endif

#if defined(ROTATE90_U8C3) || defined(ALL_KERNELS)
__kernel
void rotateC390U8Kernel(global const uchar* src, int rows, int cols,
                        int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  uchar3 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload3(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x) +
        max(rows - index_y - 4, 0) * 3;
  if (rows_remained >= 4) {
    vstore3(input_value[3], 0, dst);
    dst += 3;
    vstore3(input_value[2], 0, dst);
    dst += 3;
    vstore3(input_value[1], 0, dst);
    dst += 3;
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 2) {
    vstore3(input_value[1], 0, dst);
    dst += 3;
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 3) {
    vstore3(input_value[2], 0, dst);
    dst += 3;
    vstore3(input_value[1], 0, dst);
    dst += 3;
    vstore3(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE180_U8C3) || defined(ALL_KERNELS)
__kernel
void rotateC3180U8Kernel(global const uchar* src, int rows, int cols,
                         int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  uchar3 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload3(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst +
                        dst_stride * max(rows - index_y - 4, 0)) +
        max(cols - index_x - 1, 0) * 3;
  if (rows_remained >= 4) {
    vstore3(input_value[3], 0, dst);
    dst += dst_stride;
    vstore3(input_value[2], 0, dst);
    dst += dst_stride;
    vstore3(input_value[1], 0, dst);
    dst += dst_stride;
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 2) {
    vstore3(input_value[1], 0, dst);
    dst += dst_stride;
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 3) {
    vstore3(input_value[2], 0, dst);
    dst += dst_stride;
    vstore3(input_value[1], 0, dst);
    dst += dst_stride;
    vstore3(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE270_U8C3) || defined(ALL_KERNELS)
__kernel
void rotateC3270U8Kernel(global const uchar* src, int rows, int cols,
                         int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  uchar3 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload3(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst +
                        dst_stride * max(cols - index_x - 1, 0)) +
        index_y * 3;
  if (rows_remained >= 4) {
    vstore3(input_value[0], 0, dst);
    dst += 3;
    vstore3(input_value[1], 0, dst);
    dst += 3;
    vstore3(input_value[2], 0, dst);
    dst += 3;
    vstore3(input_value[3], 0, dst);
  }
  if (rows_remained == 1) {
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 2) {
    vstore3(input_value[0], 0, dst);
    dst += 3;
    vstore3(input_value[1], 0, dst);
  }
  if (rows_remained == 3) {
    vstore3(input_value[0], 0, dst);
    dst += 3;
    vstore3(input_value[1], 0, dst);
    dst += 3;
    vstore3(input_value[2], 0, dst);
  }
}
#endif

#if defined(ROTATE90_F32C3) || defined(ALL_KERNELS)
__kernel
void rotateC390F32Kernel(global const float* src, int rows, int cols,
                         int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  float3 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload3(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x) +
        max(rows - index_y - 2, 0) * 3;
  if (rows_remained >= 2) {
    vstore3(input_value[1], 0, dst);
    dst += 3;
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore3(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE180_F32C3) || defined(ALL_KERNELS)
__kernel
void rotateC3180F32Kernel(global const float* src, int rows, int cols,
                          int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  float3 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload3(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst +
                        dst_stride * max(rows - index_y - 2, 0)) +
        max(cols - index_x - 1, 0) * 3;
  if (rows_remained >= 2) {
    vstore3(input_value[1], 0, dst);
    dst = (global float*)((global uchar*)dst + dst_stride);
    vstore3(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore3(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE270_F32C3) || defined(ALL_KERNELS)
__kernel
void rotateC3270F32Kernel(global const float* src, int rows, int cols,
                          int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  float3 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload3(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst +
                        dst_stride * max(cols - index_x - 1, 0)) +
        index_y * 3;
  if (rows_remained >= 2) {
    vstore3(input_value[0], 0, dst);
    dst += 3;
    vstore3(input_value[1], 0, dst);
  }
  if (rows_remained == 1) {
    vstore3(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE90_U8C4) || defined(ALL_KERNELS)
__kernel
void rotateC490U8Kernel(global const uchar* src, int rows, int cols,
                        int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  uchar4 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst + dst_stride * index_x) +
        max(rows - index_y - 4, 0) * 4;
  if (rows_remained >= 4) {
    vstore4(input_value[3], 0, dst);
    dst += 4;
    vstore4(input_value[2], 0, dst);
    dst += 4;
    vstore4(input_value[1], 0, dst);
    dst += 4;
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 2) {
    vstore4(input_value[1], 0, dst);
    dst += 4;
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 3) {
    vstore4(input_value[2], 0, dst);
    dst += 4;
    vstore4(input_value[1], 0, dst);
    dst += 4;
    vstore4(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE180_U8C4) || defined(ALL_KERNELS)
__kernel
void rotateC4180U8Kernel(global const uchar* src, int rows, int cols,
                         int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  uchar4 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst +
                        dst_stride * max(rows - index_y - 4, 0)) +
        max(cols - index_x - 1, 0) * 4;
  if (rows_remained >= 4) {
    vstore4(input_value[3], 0, dst);
    dst += dst_stride;
    vstore4(input_value[2], 0, dst);
    dst += dst_stride;
    vstore4(input_value[1], 0, dst);
    dst += dst_stride;
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 2) {
    vstore4(input_value[1], 0, dst);
    dst += dst_stride;
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 3) {
    vstore4(input_value[2], 0, dst);
    dst += dst_stride;
    vstore4(input_value[1], 0, dst);
    dst += dst_stride;
    vstore4(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE270_U8C4) || defined(ALL_KERNELS)
__kernel
void rotateC4270U8Kernel(global const uchar* src, int rows, int cols,
                         int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 4;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const uchar*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  uchar4 input_value[4];
  for (int i = 0; i < min(rows_remained, 4); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const uchar*)((global uchar*)src + src_stride);
  }
  dst = (global uchar*)((global uchar*)dst +
                        dst_stride * max(cols - index_x - 1, 0)) +
        index_y * 4;
  if (rows_remained >= 4) {
    vstore4(input_value[0], 0, dst);
    dst += 4;
    vstore4(input_value[1], 0, dst);
    dst += 4;
    vstore4(input_value[2], 0, dst);
    dst += 4;
    vstore4(input_value[3], 0, dst);
  }
  if (rows_remained == 1) {
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 2) {
    vstore4(input_value[0], 0, dst);
    dst += 4;
    vstore4(input_value[1], 0, dst);
  }
  if (rows_remained == 3) {
    vstore4(input_value[0], 0, dst);
    dst += 4;
    vstore4(input_value[1], 0, dst);
    dst += 4;
    vstore4(input_value[2], 0, dst);
  }
}
#endif

#if defined(ROTATE180_F32C4) || defined(ALL_KERNELS)
__kernel
void rotateC4180F32Kernel(global const float* src, int rows, int cols,
                          int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  float4 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = (vload4(element_x, src));
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst +
                        dst_stride * max(rows - index_y - 2, 0)) +
        max(cols - index_x - 1, 0) * 4;
  if (rows_remained >= 2) {
    vstore4(input_value[1], 0, dst);
    dst = (global float*)((global uchar*)dst + dst_stride);
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore4(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE90_F32C4) || defined(ALL_KERNELS)
__kernel
void rotateC490F32Kernel(global const float* src, int rows, int cols,
                         int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  float4 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst + dst_stride * index_x) +
        max(rows - index_y - 2, 0) * 4;
  if (rows_remained >= 2) {
    vstore4(input_value[1], 0, dst);
    dst += 4;
    vstore4(input_value[0], 0, dst);
  }
  if (rows_remained == 1) {
    vstore4(input_value[0], 0, dst);
  }
}
#endif

#if defined(ROTATE270_F32C4) || defined(ALL_KERNELS)
__kernel
void rotateC4270F32Kernel(global const float* src, int rows, int cols,
                          int src_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x, index_y = element_y * 2;
  if (index_x >= cols || index_y >= rows) {
    return;
  }
  src = (global const float*)((global uchar*)src + index_y * src_stride);
  int rows_remained = rows - index_y;
  float4 input_value[2];
  for (int i = 0; i < min(rows_remained, 2); i++) {
    input_value[i] = vload4(element_x, src);
    src = (global const float*)((global uchar*)src + src_stride);
  }
  dst = (global float*)((global uchar*)dst +
                        dst_stride * max(cols - index_x - 1, 0)) +
        index_y * 4;
  if (rows_remained >= 2) {
    vstore4(input_value[0], 0, dst);
    dst += 4;
    vstore4(input_value[1], 0, dst);
  }
  if (rows_remained == 1) {
    vstore4(input_value[0], 0, dst);
  }
}
#endif