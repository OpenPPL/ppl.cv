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
 *
 * The device and kernel function definitions of morphology operations.
 */

#include "utility/utility.hpp"

namespace ppl {
namespace cv {
namespace cuda {

#define MAX_SIZE 1024

/**************************** row&column filters *****************************/

template <typename T0, typename T1, typename Morphology>
__DEVICE__
T0 morphRowDevice0(const T1* src, int stride, int element_y, int bottom_x,
                   int top_x, Morphology morphology_swap) {
  T0* input = (T0*)((uchar*)src + element_y * stride);
  T0 value = input[bottom_x];

  T0 target;
  for (int i = bottom_x + 1; i <= top_x; i++) {
    target = input[i];
    morphology_swap(value, target);
  }

  return value;
}

template <typename Morphology>
__DEVICE__
uchar4 morphRowU8C1LeftDevice0(const uchar* src, int stride, int element_y,
                               int bottom_x, int top_x,
                               Morphology morphology_swap) {
  uchar* input = (uchar*)((uchar*)src + element_y * stride);

  uchar value0, value1, value2, value3;
  uchar target0, target1, target2, target3;
  morphology_swap.initialize(value0, value1, value2, value3);

  for (int i = bottom_x; i <= top_x; i++) {
    target0 = input[i <= 0 ? 0 : i];
    target1 = input[i + 1 <= 0 ? 0 : i + 1];
    target2 = input[i + 2 <= 0 ? 0 : i + 2];
    target3 = input[i + 3 <= 0 ? 0 : i + 3];
    morphology_swap(value0, target0);
    morphology_swap(value1, target1);
    morphology_swap(value2, target2);
    morphology_swap(value3, target3);
  }

  return make_uchar4(value0, value1, value2, value3);
}

template <typename Morphology>
__DEVICE__
uchar4 morphRowU8C1MiddleDevice0(const uchar* src, int stride, int element_y,
                                 int bottom_x, int top_x,
                                 Morphology morphology_swap) {
  uchar* input = (uchar*)((uchar*)src + element_y * stride);

  uchar value0, value1, value2, value3;
  uchar target0, target1, target2, target3;
  morphology_swap.initialize(value0, value1, value2, value3);

  for (int i = bottom_x; i <= top_x; i++) {
    target0 = input[i];
    target1 = input[i + 1];
    target2 = input[i + 2];
    target3 = input[i + 3];
    morphology_swap(value0, target0);
    morphology_swap(value1, target1);
    morphology_swap(value2, target2);
    morphology_swap(value3, target3);
  }

  return make_uchar4(value0, value1, value2, value3);
}

template <typename T0, typename T1, typename Morphology>
__DEVICE__
T0 morphColDevice0(const T1* src, int stride, int element_x, int bottom_y,
                   int top_y, Morphology morphology_swap) {
  T0* input = (T0*)((uchar*)src + bottom_y * stride);
  T0 value = input[element_x];

  T0 target;
  for (int i = bottom_y + 1; i <= top_y; i++) {
    input = (T0*)((uchar*)input + stride);

    target = input[element_x];
    morphology_swap(value, target);
  }

  return value;
}

/**
 * 1 Morphology operation with a fully masked rectangle kernel.
 */
template <typename T0, typename T1, typename Morphology>
__global__
void morphRowKernel0(const T1* src, int rows, int cols, int src_stride,
                     int radius_x, T1* dst, int dst_stride,
                     Morphology morphology_swap) {
  int element_x, element_y;
  if (sizeof(T1) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int top_x    = element_x + radius_x;
  if (bottom_x < 0) {
    bottom_x = 0;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }

  T0 result;
  result = morphRowDevice0<T0, T1, Morphology>(src, src_stride, element_y,
               bottom_x, top_x, morphology_swap);

  T0* output = (T0*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = result;
}

/**
 * 1 Morphology operation with a fully masked rectangle kernel.
 */
template <typename Morphology>
__global__
void morphRowU8C1Kernel0(const uchar* src, int rows, int cols, int columns,
                         int src_stride, int left_threads, int aligned_columns,
                         int radius_x, uchar* dst, int dst_stride,
                         Morphology morphology_swap) {
  int thread_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int thread_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (thread_x >= columns || thread_y >= rows) {
    return;
  }

  int element_x;
  if (thread_x < aligned_columns) {
    element_x = thread_x << 2;
  }
  else {
    element_x = thread_x - aligned_columns + (aligned_columns << 2);
  }

  int bottom_x = element_x - radius_x;
  int top_x    = element_x + radius_x;

  if (thread_x < aligned_columns) {
    uchar4 result;
    if (thread_x < left_threads) {
      result = morphRowU8C1LeftDevice0<Morphology>(src, src_stride, thread_y,
                   bottom_x, top_x, morphology_swap);
    }
    else {
      result = morphRowU8C1MiddleDevice0<Morphology>(src, src_stride, thread_y,
                   bottom_x, top_x, morphology_swap);
    }

    uchar4* output = (uchar4*)((uchar*)dst + thread_y * dst_stride);
    output[thread_x] = result;
  }
  else {
    if (top_x >= cols) {
      top_x = cols - 1;
    }

    uchar result;
    result = morphRowDevice0<uchar, uchar, Morphology>(src, src_stride,
                 thread_y, bottom_x, top_x, morphology_swap);

    uchar* output = (uchar*)((uchar*)dst + thread_y * dst_stride);
    output[element_x] = result;
  }
}

template <typename T0, typename T1, typename Morphology>
__global__
void morphColKernel0(const T1* src, int rows, int cols, int src_stride,
                     int radius_x, int radius_y, T1* dst, int dst_stride,
                     BorderType border_type, const T1 border_value,
                     Morphology morphology_swap) {
  int element_x, element_y;
  if (sizeof(T1) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  bool constant_border = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;
  if (bottom_x < 0) {
    constant_border = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_x >= cols) {
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  T0 result;
  result = morphColDevice0<T0, T1, Morphology>(src, src_stride, element_x,
               bottom_y, top_y, morphology_swap);
  if (border_type == BORDER_CONSTANT && constant_border) {
    morphology_swap.checkConstantResult(result, border_value);
  }

  T0* output = (T0*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = result;
}

/********************************* 2D filter *********************************/

template <typename T0, typename T1, typename Morphology>
__DEVICE__
T0 morph2DDevice0(const T1* src, int stride, int bottom_x, int bottom_y,
                  int top_x, int top_y, Morphology morphology_swap) {
  T0* input = (T0*)((uchar*)src + bottom_y * stride);
  T0 value = input[bottom_x];

  T0 target;
  int index;
  for (int i = bottom_y; i <= top_y; i++) {
    index = bottom_x;
    for (int j = bottom_x; j <= top_x; j++) {
      target = input[index];
      morphology_swap(value, target);
      index++;
    }
    input = (T0*)((uchar*)input + stride);
  }

  return value;
}

template <typename Morphology>
__DEVICE__
uchar4 morphU8C1LeftDevice0(const uchar* src, int stride, int bottom_x,
                            int bottom_y, int top_x, int top_y,
                            Morphology morphology_swap) {
  uchar* input = (uchar*)((uchar*)src + bottom_y * stride);

  uchar value0, value1, value2, value3;
  uchar target0, target1, target2, target3;
  morphology_swap.initialize(value0, value1, value2, value3);

  int index;
  for (int i = bottom_y; i <= top_y; i++) {
    index = bottom_x;
    for (int j = bottom_x; j <= top_x; j++) {
      target0 = input[index <= 0 ? 0 : index];
      target1 = input[index + 1 <= 0 ? 0 : index + 1];
      target2 = input[index + 2 <= 0 ? 0 : index + 2];
      target3 = input[index + 3 <= 0 ? 0 : index + 3];
      morphology_swap(value0, target0);
      morphology_swap(value1, target1);
      morphology_swap(value2, target2);
      morphology_swap(value3, target3);
      index++;
    }
    input = (uchar*)((uchar*)input + stride);
  }

  return make_uchar4(value0, value1, value2, value3);
}

template <typename Morphology>
__DEVICE__
uchar4 morphU8C1MiddleDevice0(const uchar* src, int stride, int bottom_x,
                              int bottom_y, int top_x, int top_y,
                              Morphology morphology_swap) {
  uchar* input = (uchar*)((uchar*)src + bottom_y * stride);

  uchar value0, value1, value2, value3;
  uchar target0, target1, target2, target3;
  morphology_swap.initialize(value0, value1, value2, value3);

  int index;
  for (int i = bottom_y; i <= top_y; i++) {
    index = bottom_x;
    for (int j = bottom_x; j <= top_x; j++) {
      target0 = input[index];
      target1 = input[index + 1];
      target2 = input[index + 2];
      target3 = input[index + 3];
      morphology_swap(value0, target0);
      morphology_swap(value1, target1);
      morphology_swap(value2, target2);
      morphology_swap(value3, target3);
      index++;
    }
    input = (uchar*)((uchar*)input + stride);
  }

  return make_uchar4(value0, value1, value2, value3);
}

template <typename T0, typename T1, typename Morphology>
__DEVICE__
T0 morph2DDevice1(const T1* src, int stride, const uchar* kernel, int bottom_x,
                  int bottom_y, int top_x, int top_y, int kernel_bottom_x,
                  int kernel_bottom_y, int kernel_x,
                  Morphology morphology_swap) {
  T0* input = (T0*)((uchar*)src + bottom_y * stride);
  T0 value;
  morphology_swap.initialize(value);

  T0 target;
  int index1, index2;
  int kernel_bottom_start = kernel_bottom_y * kernel_x + kernel_bottom_x;
  for (int i = bottom_y; i <= top_y; i++) {
    index1 = bottom_x;
    index2 = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (kernel[index2]) {
        target = input[index1];
        morphology_swap(value, target);
      }
      index1++;
      index2++;
    }
    input = (T0*)((uchar*)input + stride);
    kernel_bottom_start += kernel_x;
  }

  return value;
}

template <typename Morphology>
__DEVICE__
uchar4 morphU8C1LeftDevice1(const uchar* src, int stride, const uchar* kernel,
                            int bottom_x, int bottom_y, int top_x, int top_y,
                            int kernel_bottom_y, int kernel_x,
                            Morphology morphology_swap) {
  uchar* input = (uchar*)((uchar*)src + bottom_y * stride);

  uchar value0, value1, value2, value3;
  uchar target0, target1, target2, target3;
  morphology_swap.initialize(value0, value1, value2, value3);

  int index1, index2;
  int kernel_bottom_start = kernel_bottom_y * kernel_x;
  for (int i = bottom_y; i <= top_y; i++) {
    index1 = bottom_x;
    index2 = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (kernel[index2]) {
        if (j >= 0) {
          target0 = input[index1];
          morphology_swap(value0, target0);
        }
        if (j + 1 >= 0) {
          target1 = input[index1 + 1];
          morphology_swap(value1, target1);
        }
        if (j + 2 >= 0) {
          target2 = input[index1 + 2];
          morphology_swap(value2, target2);
        }
        if (j + 3 >= 0) {
          target3 = input[index1 + 3];
          morphology_swap(value3, target3);
        }
      }
      index1++;
      index2++;
    }
    input = (uchar*)((uchar*)input + stride);
    kernel_bottom_start += kernel_x;
  }

  return make_uchar4(value0, value1, value2, value3);
}

template <typename Morphology>
__DEVICE__
uchar4 morphU8C1MiddleDevice1(const uchar* src, int stride, const uchar* kernel,
                              int bottom_x, int bottom_y, int top_x, int top_y,
                              int kernel_bottom_y, int kernel_x,
                              Morphology morphology_swap) {
  uchar* input = (uchar*)((uchar*)src + bottom_y * stride);

  uchar value0, value1, value2, value3;
  uchar target0, target1, target2, target3;
  morphology_swap.initialize(value0, value1, value2, value3);

  int index1, index2;
  int kernel_bottom_start = kernel_bottom_y * kernel_x;
  for (int i = bottom_y; i <= top_y; i++) {
    index1 = bottom_x;
    index2 = kernel_bottom_start;
    for (int j = bottom_x; j <= top_x; j++) {
      if (kernel[index2]) {
        target0 = input[index1];
        target1 = input[index1 + 1];
        target2 = input[index1 + 2];
        target3 = input[index1 + 3];
        morphology_swap(value0, target0);
        morphology_swap(value1, target1);
        morphology_swap(value2, target2);
        morphology_swap(value3, target3);
      }
      index1++;
      index2++;
    }
    input = (uchar*)((uchar*)input + stride);
    kernel_bottom_start += kernel_x;
  }

  return make_uchar4(value0, value1, value2, value3);
}

/**
 * 1 Morphology operation with a fully masked rectangle kernel.
 */
template <typename T0, typename T1, typename Morphology>
__global__
void morph2DKernel0(const T1* src, int rows, int cols, int src_stride,
                    int radius_x, int radius_y, T1* dst, int dst_stride,
                    BorderType border_type, const T1 border_value,
                    Morphology morphology_swap) {
  int element_x, element_y;
  if (sizeof(T1) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  bool constant_border = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;
  if (bottom_x < 0) {
    bottom_x = 0;
    constant_border = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border = true;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
    constant_border = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border = true;
  }

  T0 result;
  result = morph2DDevice0<T0, T1, Morphology>(src, src_stride, bottom_x,
               bottom_y, top_x, top_y, morphology_swap);
  if (border_type == BORDER_CONSTANT && constant_border) {
    morphology_swap.checkConstantResult(result, border_value);
  }

  T0* output = (T0*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = result;
}

/**
 * 1 Morphology operation with a fully masked rectangle kernel.
 */
template <typename Morphology>
__global__
void morph2DU8C1Kernel0(const uchar* src, int rows, int cols, int columns,
                        int src_stride, int left_threads, int aligned_columns,
                        int radius_x, int radius_y, uchar* dst, int dst_stride,
                        BorderType border_type, const uchar border_value,
                        Morphology morphology_swap) {
  int thread_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int thread_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (thread_x >= columns || thread_y >= rows) {
    return;
  }

  int element_x;
  if (thread_x < aligned_columns) {
    element_x = thread_x << 2;
  }
  else {
    element_x = thread_x - aligned_columns + (aligned_columns << 2);
  }

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  int bottom_x = element_x - radius_x;
  int bottom_y = thread_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = thread_y + radius_y;
  if (bottom_x + 1 < 0) {
    constant_border1 = true;
  }
  if (bottom_x + 2 < 0) {
    constant_border2 = true;
  }
  if (bottom_x + 3 < 0) {
    constant_border3 = true;
  }
  if (top_x + 1 >= cols) {
    constant_border1 = true;
  }
  if (top_x + 2 >= cols) {
    constant_border2 = true;
  }
  if (top_x + 3 >= cols) {
    constant_border3 = true;
  }

  if (bottom_x < 0) {
    constant_border0 = true;
  }
  if (top_x >= cols) {
    constant_border0 = true;
  }
  if (bottom_y < 0) {
    bottom_y = 0;
    constant_border0 = true;
    constant_border1 = true;
    constant_border2 = true;
    constant_border3 = true;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
    constant_border0 = true;
    constant_border1 = true;
    constant_border2 = true;
    constant_border3 = true;
  }

  if (thread_x < aligned_columns) {
    uchar4 result;
    if (thread_x < left_threads) {
      result = morphU8C1LeftDevice0<Morphology>(src, src_stride, bottom_x,
                   bottom_y, top_x, top_y, morphology_swap);
    }
    else {
      result = morphU8C1MiddleDevice0<Morphology>(src, src_stride, bottom_x,
                   bottom_y, top_x, top_y, morphology_swap);
    }

    if (border_type == BORDER_CONSTANT) {
      morphology_swap.checkU8C1ConstantResult(result, border_value,
          constant_border0, constant_border1, constant_border2,
          constant_border3);
    }

    uchar4* output = (uchar4*)((uchar*)dst + thread_y * dst_stride);
    output[thread_x] = result;
  }
  else {
    if (top_x >= cols) {
      top_x = cols - 1;
    }
    uchar result;
    result = morph2DDevice0<uchar, uchar, Morphology>(src, src_stride, bottom_x,
                 bottom_y, top_x, top_y, morphology_swap);
    if (border_type == BORDER_CONSTANT && constant_border0) {
      morphology_swap.checkConstantResult(result, border_value);
    }

    uchar* output = (uchar*)((uchar*)dst + thread_y * dst_stride);
    output[element_x] = result;
  }
}

/**
 * 1 Morphology operation with a partially masked rectangle kernel.
 */
template <typename T0, typename T1, typename Morphology>
__global__
void morph2DKernel1(const T1* src, int rows, int cols, int src_stride,
                    const uchar* kernel, int radius_x, int radius_y,
                    int kernel_x, int kernel_y, T1* dst, int dst_stride,
                    BorderType border_type, const T1 border_value,
                    Morphology morphology_swap) {
  __shared__ uchar mask[MAX_SIZE];

  if (kernel_y <= 32 && kernel_x <= 32) {
    int kernel_elements = kernel_y * kernel_x;
    int threads = blockDim.y * blockDim.x;
    int index = threadIdx.y * blockDim.x + threadIdx.x;
    while (index < kernel_elements) {
      mask[index] = kernel[index];
      index += threads;
    }
    __syncthreads();
  }

  int element_x, element_y;
  if (sizeof(T1) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = element_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = element_y + radius_y;

  bool constant_border = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (kernel[mask_index + j - bottom_x]) {
            constant_border = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (kernel[mask_index + j - bottom_x]) {
              constant_border = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_x = 0, kernel_bottom_y = 0;
  if (bottom_x < 0) {
    bottom_x  = 0;
    kernel_bottom_x = radius_x - element_x;
  }
  if (bottom_y < 0) {
    bottom_y  = 0;
    kernel_bottom_y = radius_y - element_y;
  }
  if (top_x >= cols) {
    top_x = cols - 1;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  T0 result;
  if (kernel_y <= 32 && kernel_x <= 32) {
    result = morph2DDevice1<T0, T1, Morphology>(src, src_stride, mask, bottom_x,
                 bottom_y, top_x, top_y, kernel_bottom_x, kernel_bottom_y,
                 kernel_x, morphology_swap);
  }
  else {
    result = morph2DDevice1<T0, T1, Morphology>(src, src_stride, kernel,
                 bottom_x, bottom_y, top_x, top_y, kernel_bottom_x,
                 kernel_bottom_y, kernel_x, morphology_swap);
  }

  if (border_type == BORDER_CONSTANT && constant_border) {
    morphology_swap.checkConstantResult(result, border_value);
  }

  T0* output = (T0*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = result;
}

/**
 * 1 Morphology operation with a partially masked rectangle kernel.
 */
template <typename Morphology>
__global__
void morph2DU8C1Kernel1(const uchar* src, int rows, int cols, int columns,
                        int src_stride, const uchar* kernel, int left_threads,
                        int aligned_columns, int radius_x, int radius_y,
                        int kernel_x, int kernel_y, uchar* dst, int dst_stride,
                        BorderType border_type, const uchar border_value,
                        Morphology morphology_swap) {
  __shared__ uchar mask[MAX_SIZE];

  if (kernel_y <= 32 && kernel_x <= 32) {
    int kernel_elements = kernel_y * kernel_x;
    int threads = blockDim.y * blockDim.x;
    int index = threadIdx.y * blockDim.x + threadIdx.x;
    while (index < kernel_elements) {
      mask[index] = kernel[index];
      index += threads;
    }
    __syncthreads();
  }

  int thread_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int thread_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (thread_x >= columns || thread_y >= rows) {
    return;
  }

  int element_x;
  if (thread_x < aligned_columns) {
    element_x = thread_x << 2;
  }
  else {
    element_x = thread_x - aligned_columns + (aligned_columns << 2);
  }

  int bottom_x = element_x - radius_x;
  int bottom_y = thread_y - radius_y;
  int top_x    = element_x + radius_x;
  int top_y    = thread_y + radius_y;

  bool constant_border0 = false;
  bool constant_border1 = false;
  bool constant_border2 = false;
  bool constant_border3 = false;
  if (border_type == BORDER_CONSTANT) {
    int mask_index = 0;
    for (int i = bottom_y; i <= top_y; i++) {
      if (i < 0 || i >= rows) {
        for (int j = bottom_x; j <= top_x; j++) {
          if (kernel[mask_index + j - bottom_x]) {
            constant_border0 = true;
            constant_border1 = true;
            constant_border2 = true;
            constant_border3 = true;
          }
        }
      }
      else {
        for (int j = bottom_x; j <= top_x; j++) {
          if (j < 0 || j >= cols) {
            if (kernel[mask_index + j - bottom_x]) {
              constant_border0 = true;
            }
          }
          if (j + 1 < 0) {
            if (kernel[mask_index + j - bottom_x + 1]) {
              constant_border1 = true;
            }
          }
          if (j + 2 < 0) {
            if (kernel[mask_index + j - bottom_x + 2]) {
              constant_border2 = true;
            }
          }
          if (j + 3 < 0) {
            if (kernel[mask_index + j - bottom_x + 3]) {
              constant_border3 = true;
            }
          }
        }
      }
      mask_index += kernel_y;
    }
  }

  int kernel_bottom_y = 0;
  if (bottom_y < 0) {
    bottom_y = 0;
    kernel_bottom_y = radius_y - thread_y;
  }
  if (top_y >= rows) {
    top_y = rows - 1;
  }

  if (thread_x < aligned_columns) {
    uchar4 result;
    if (thread_x < left_threads) {
      if (kernel_y <= 32 && kernel_x <= 32) {
        result = morphU8C1LeftDevice1<Morphology>(src, src_stride, mask,
                     bottom_x, bottom_y, top_x, top_y, kernel_bottom_y,
                     kernel_x, morphology_swap);
      }
      else {
        result = morphU8C1LeftDevice1<Morphology>(src, src_stride, kernel,
                     bottom_x, bottom_y, top_x, top_y, kernel_bottom_y,
                     kernel_x, morphology_swap);
      }
    }
    else {
      if (kernel_y <= 32 && kernel_x <= 32) {
        result = morphU8C1MiddleDevice1<Morphology>(src, src_stride, mask,
                     bottom_x, bottom_y, top_x, top_y, kernel_bottom_y,
                     kernel_x, morphology_swap);
      }
      else {
        result = morphU8C1MiddleDevice1<Morphology>(src, src_stride, kernel,
                     bottom_x, bottom_y, top_x, top_y, kernel_bottom_y,
                     kernel_x, morphology_swap);
      }
    }

    if (border_type == BORDER_CONSTANT) {
      morphology_swap.checkU8C1ConstantResult(result, border_value,
          constant_border0, constant_border1, constant_border2,
          constant_border3);
    }

    uchar4* output = (uchar4*)((uchar*)dst + thread_y * dst_stride);
    output[thread_x] = result;
  }
  else {
    if (top_x >= cols) {
      top_x = cols - 1;
    }
    int kernel_bottom_x = 0;

    uchar result;
    if (kernel_y <= 32 && kernel_x <= 32) {
      result = morph2DDevice1<uchar, uchar, Morphology>(src, src_stride, mask,
                   bottom_x, bottom_y, top_x, top_y, kernel_bottom_x,
                   kernel_bottom_y, kernel_x, morphology_swap);
    }
    else {
      result = morph2DDevice1<uchar, uchar, Morphology>(src, src_stride, kernel,
                   bottom_x, bottom_y, top_x, top_y, kernel_bottom_x,
                   kernel_bottom_y, kernel_x, morphology_swap);
    }

    if (border_type == BORDER_CONSTANT && constant_border0) {
      morphology_swap.checkConstantResult(result, border_value);
    }

    uchar* output = (uchar*)((uchar*)dst + thread_y * dst_stride);
    output[element_x] = result;
  }
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
