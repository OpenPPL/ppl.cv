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

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define MAX_THREADS 1024

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

RetCode distanceTransform(const uchar* src, int rows, int cols, int src_stride,
                          float* dst, int dst_stride, DistTypes distance_type,
                          DistanceTransformMasks mask_size,
                          cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(distance_type == DIST_L2);
  PPL_ASSERT(mask_size == DIST_MASK_PRECISE);

  dim3 block, grid;
  block.x = MAX_THREADS;
  if ((size_t)rows < block.x) block.x = rows;
  grid.x = (rows + block.x - 1) / block.x;
  grid.y = cols;
  countColKernel<<<grid, block, rows * sizeof(uchar), stream>>>(src, rows, cols,
      src_stride, (int*)dst, dst_stride);

  block.x = MAX_THREADS;
  if ((size_t)cols < block.x) block.x = cols;
  grid.x = (cols + block.x - 1) / block.x;
  grid.y = rows;
  countRowKernel<<<grid, block, cols * sizeof(int), stream>>>((int*)dst,
      rows, cols, dst_stride, dst, dst_stride);

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
