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

#include "ppl/cv/cuda/distancetransform.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define MAX_THREADS 1024
#define INF 2147483647


__global__
void countColKernel(const uchar* src, int rows, int cols, int src_stride,
                    int* buffer, int buf_stride) {
	extern __shared__ uchar data[];

  int element_x = blockIdx.y;
  int element_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int index = threadIdx.x;
  for (; index < rows; index += blockDim.x) {
    uchar* input = (uchar*)((uchar*)src + index * src_stride);
    data[index] = input[element_x];
  }
  __syncthreads();

	int value = data[element_y];
	int distance = 1;
  if (value != 0) {
    value = cols * cols + rows * rows;
    for (index = 1; index < rows - element_y; index++) {
      if (data[element_y + index] == 0) {
        value = min(value, distance);
      }
      distance += index * 2 + 1;
      if (distance > value) break;
    }

    distance = 1;
    for (index = 1; index <= element_y; index++) {
      if (data[element_y - index] == 0) {
        value = min(value, distance);
      }
      distance += index * 2 + 1;
      if (distance > value) break;
    }
  }

  int* output = (int*)((uchar*)buffer + element_y * buf_stride);
  output[element_x] = value;
}

__global__
void countRowKernel(int* buffer, int rows, int cols, int buf_stride, float* dst,
                    int dst_stride) {
	extern __shared__ int counts[];

  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int* input = (int*)((uchar*)buffer + element_y * buf_stride);
  int index = threadIdx.x;
  for (; index < cols; index += blockDim.x) {
    counts[index] = input[index];
  }
  __syncthreads();

 	int value = counts[element_x];
 	int distance = 1;
  if (value != 0) {
    for (index = 1; index < cols - element_x; index++) {
      value = min(value, counts[element_x + index] + distance);
      distance += index * 2 + 1;
      if (distance > value) break;
    }
    distance = 1;
    for (index = 1; index <= element_x; index++) {
      value = min(value, counts[element_x - index] + distance);
      distance += index * 2 + 1;
      if (distance > value) break;
    }
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = sqrtf(value);
}

__global__
void initializeData(const uchar* src, int rows, int cols, int src_stride,
                    float* dst, int dst_stride) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  uchar input_value0, input_value1;
  const uchar* input = src + element_y * src_stride;

  float output_value0 = 0.f, output_value1 = 0.f;
  float* output = (float*)((uchar*)dst + element_y * dst_stride);

  if (index_x < cols - 1) {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];

    if (input_value0) output_value0 = INF;
    if (input_value1) output_value1 = INF;

    output[index_x] = output_value0;
    output[index_x + 1] = output_value1;
  }
  else {
    input_value0 = input[index_x];
    if (input_value0) output_value0 = INF;
    output[index_x] = output_value0;
  }
}

__DEVICE__
float getData(float* src, int rows, int cols, int src_stride, int y, int x) {
  if (y < 0) {
    y = 0;
  }
  if (y >= rows) {
    y = rows - 1;
  }

  if (x < 0) {
    x = 0;
  }
  if (x >= cols) {
    x = cols - 1;
  }
  float* input = (float*)((uchar*)src + y * src_stride);

  return input[x];
}

__global__
void calculateDistance3x3(float* dst, int rows, int cols, int dst_stride,
                          float a, float b, int* done) {
  __shared__ int found;

  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  bool main_thread = (threadIdx.x + threadIdx.y == 0);
  if (main_thread) {
    found = 1;
  }
  __syncthreads();

  if (element_x < cols && element_y < rows) {
    float data = output[element_x];
    if (data > 0 || main_thread) {
      float old_data = data;
      float new_data = data;
      while (found > 0) {
        if (main_thread) {
          found = 0;
        }
        __syncthreads();

        old_data = new_data < data ? new_data : data;
        new_data = INF;

        data = getData(dst, rows, cols, dst_stride, element_y - 1,
                       element_x - 1);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 1, element_x);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 1,
                       element_x + 1);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y, element_x - 1);
        data += a;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y, element_x + 1);
        data += a;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1,
                       element_x - 1);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1, element_x);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1,
                       element_x + 1);
        data += b;
        if (new_data > data) new_data = data;

        if (new_data < old_data) {
          output[element_x] = new_data;
          atomicExch(&found, 1);
        }
        __syncthreads();

        if (main_thread && found > 0) {
          *done = 0;
        }
      }
    }
  }
}

__global__
void calculateDistance5x5(float* dst, int rows, int cols, int dst_stride,
                          float a, float b, float c, int* done) {
  __shared__ int found;
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  bool main_thread = (threadIdx.x + threadIdx.y == 0);
  if (main_thread) {
    found = 1;
  }
  __syncthreads();

  if (element_x < cols && element_y < rows) {
    float data = output[element_x];
    if (data > 0 || main_thread) {
      float old_data = data;
      float new_data = data;
      while (found > 0) {
        if (main_thread) {
          found = 0;
        }
        __syncthreads();

        old_data = new_data < data ? new_data : data;
        new_data = INF;

        data = getData(dst, rows, cols, dst_stride, element_y - 2,
                       element_x - 1);
        data += c;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 2,
                       element_x + 1);
        data += c;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 1,
                       element_x - 2);
        data += c;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 1,
                       element_x - 1);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 1, element_x);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 1,
                       element_x + 1);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y - 1,
                       element_x + 2);
        data += c;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y, element_x - 1);
        data += a;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y, element_x + 1);
        data += a;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1,
                       element_x - 2);
        data += c;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1,
                       element_x - 1);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1, element_x);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1,
                       element_x + 1);
        data += b;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 1,
                       element_x + 2);
        data += c;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 2,
                       element_x - 1);
        data += c;
        if (new_data > data) new_data = data;

        data = getData(dst, rows, cols, dst_stride, element_y + 2,
                       element_x + 1);
        data += c;
        if (new_data > data) new_data = data;

        if (new_data < old_data) {
          output[element_x] = new_data;
          atomicExch(&found, 1);
        }
        __syncthreads();

        if (main_thread && found > 0) {
          *done = 0;
        }
      }
    }
  }
}

RetCode distanceTransform(const uchar* src, int rows, int cols, int src_stride,
                          float* dst, int dst_stride, DistTypes distance_type,
                          DistanceTransformMasks mask_size,
                          cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(distance_type == DIST_C || distance_type == DIST_L1 ||
             distance_type == DIST_L2);
  PPL_ASSERT(mask_size == DIST_MASK_3 || mask_size == DIST_MASK_5 ||
             mask_size == DIST_MASK_PRECISE);

  dim3 block, grid, grid1;
  if (mask_size == DIST_MASK_PRECISE) {
    block.x = MAX_THREADS;
    if ((size_t)rows < block.x) block.x = rows;
    grid.x = (rows + block.x - 1) / block.x;
    grid.y = cols;
    countColKernel<<<grid, block, rows * sizeof(uchar), stream>>>(src, rows,
        cols, src_stride, (int*)dst, dst_stride);

    block.x = MAX_THREADS;
    if ((size_t)cols < block.x) block.x = cols;
    grid.x = (cols + block.x - 1) / block.x;
    grid.y = rows;
    countRowKernel<<<grid, block, cols * sizeof(int), stream>>>((int*)dst,
        rows, cols, dst_stride, dst, dst_stride);
  }
  else {
    float a, b, c;
    if (distance_type == DIST_C) {
      a = b = 1.f;
    }
    else if (distance_type == DIST_L1) {
      a = 1.f;
      b = 2.f;
    }
    else {
      a = 0.955f;
      b = 1.3693f;
    }

    if (mask_size == DIST_MASK_5) {
      a = 1.f;
      b = 1.4f;
      c = 2.1969f;
    }

    block.x = kDimX0;
    block.y = kDimY0;
    grid.x  = divideUp(cols, kDimX0, kShiftX0);
    grid.y  = divideUp(rows, kDimY0, kShiftY0);
    grid1.x = divideUp(divideUp(cols, 2, 1), kDimX0, kShiftX0);
    grid1.y = grid.y;

    initializeData<<<grid1, block, 0, stream>>>(src, rows, cols, src_stride,
                                                dst, dst_stride);
    int* gpu_done;
    cudaMalloc((void**)&gpu_done, sizeof(int));
    int host_done;

    if (mask_size == DIST_MASK_3) {
      while (true) {
        cudaMemset(gpu_done, 1, sizeof(int));
        calculateDistance3x3<<<grid, block, sizeof(int), stream>>>(dst, rows,
            cols, dst_stride, a, b, gpu_done);
        cudaMemcpy(&host_done, gpu_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (host_done > 0) break;
      }
    }
    else {  // mask_size == DIST_MASK_5
      while (true) {
        cudaMemset(gpu_done, 1, sizeof(int));
        calculateDistance5x5<<<grid, block, sizeof(int), stream>>>(dst, rows,
            cols, dst_stride, a, b, c, gpu_done);
        cudaMemcpy(&host_done, gpu_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (host_done > 0) break;
      }
    }

    cudaFree(gpu_done);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode DistanceTransform<float>(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const uchar* inData,
                                 int outWidthStride,
                                 float* outData,
                                 DistTypes distanceType,
                                 DistanceTransformMasks maskSize) {
  outWidthStride *= sizeof(float);
  RetCode code = distanceTransform(inData, height, width, inWidthStride,
                                   outData, outWidthStride, distanceType,
                                   maskSize, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
