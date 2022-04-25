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

#include "ppl/cv/cuda/rotate.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define LENGTH 16
#define SHIFT  4
#define LENGTH_X 128
#define LENGTH_Y 4
#define SHIFT_X 7
#define SHIFT_Y 2

template <typename T>
__global__
void rotate90Kernel(const T* src, int src_rows, int src_cols, int src_stride,
                    T* dst, int dst_stride) {
  __shared__ T data[LENGTH][LENGTH + 1];
  int element_x = (blockIdx.x << SHIFT) + threadIdx.x;
  int element_y = (blockIdx.y << SHIFT) + threadIdx.y;
  int empty_x = (gridDim.y << SHIFT) - src_rows;

  if (blockIdx.x < gridDim.x - 1 && blockIdx.y < gridDim.y - 1) {
    T* input = (T*)((uchar*)src + element_y * src_stride);
    T value = input[element_x];
    data[threadIdx.x][LENGTH - threadIdx.y - 1] = value;
    __syncthreads();

    int dst_x = ((gridDim.y - blockIdx.y - 1) << SHIFT) + threadIdx.x;
    int dst_y = (blockIdx.x << SHIFT) + threadIdx.y;
    T* output = (T*)((uchar*)dst + dst_y * dst_stride);
    output[dst_x - empty_x] = data[threadIdx.y][threadIdx.x];
  }
  else {
    if (element_x < src_cols && element_y < src_rows) {
      T* input = (T*)((uchar*)src + element_y * src_stride);
      T value = input[element_x];
      data[threadIdx.x][LENGTH - threadIdx.y - 1] = value;
    }
    __syncthreads();

    int threadIdx_x = blockIdx.y == gridDim.y - 1 ?
                        src_rows - (blockIdx.y << SHIFT) : LENGTH;
    int threadIdx_y = blockIdx.x == gridDim.x - 1 ?
                        src_cols - (blockIdx.x << SHIFT) : LENGTH;
    if (threadIdx.x < threadIdx_x && threadIdx.y < threadIdx_y) {
      int dst_x = ((gridDim.y - blockIdx.y - 1) << SHIFT) + threadIdx.x;
      if (blockIdx.y < gridDim.y - 1) {
        dst_x -= empty_x;
      }
      int dst_y = (blockIdx.x << SHIFT) + threadIdx.y;
      T* output = (T*)((uchar*)dst + dst_y * dst_stride);
      if (blockIdx.y < gridDim.y - 1) {
        output[dst_x] = data[threadIdx.y][threadIdx.x];
      }
      else {
        output[dst_x] = data[threadIdx.y][threadIdx.x + empty_x];
      }
    }
  }
}

__global__
void rotate180UcharC1Kernel(const uchar* src, int src_rows, int src_cols,
                            int src_stride, uchar* dst, int dst_stride) {
  int element_x = (blockIdx.x << SHIFT_X) + threadIdx.x;
  int element_y = (blockIdx.y << SHIFT_Y) + threadIdx.y;

  if (element_x < src_cols && element_y < src_rows) {
    uchar* input = (uchar*)((uchar*)src + element_y * src_stride);
    uchar value = input[element_x];

    uchar* output = (uchar*)((uchar*)dst +
                             (src_rows - 1 - element_y) * dst_stride);
    output[src_cols - 1 - element_x] = value;
  }
}

template <typename T>
__global__
void rotate180Kernel(const T* src, int src_rows, int src_cols, int src_stride,
                     T* dst, int dst_stride) {
  __shared__ T data[LENGTH_Y][LENGTH_X];
  int element_x = (blockIdx.x << SHIFT_X) + threadIdx.x;
  int element_y = (blockIdx.y << SHIFT_Y) + threadIdx.y;
  int empty_x = (gridDim.x << SHIFT_X) - src_cols;
  int empty_y = (gridDim.y << SHIFT_Y) - src_rows;

  if (blockIdx.x < gridDim.x - 1 && blockIdx.y < gridDim.y - 1) {
    T* input = (T*)((uchar*)src + element_y * src_stride);
    T value = input[element_x];
    data[LENGTH_Y - threadIdx.y - 1][LENGTH_X - threadIdx.x - 1] = value;
    __syncthreads();

    int dst_x = ((gridDim.x - blockIdx.x - 1) << SHIFT_X) + threadIdx.x;
    int dst_y = ((gridDim.y - blockIdx.y - 1) << SHIFT_Y) + threadIdx.y;
    T* output = (T*)((uchar*)dst + (dst_y - empty_y) * dst_stride);
    output[dst_x - empty_x] = data[threadIdx.y][threadIdx.x];
  }
  else {
    if (element_x < src_cols && element_y < src_rows) {
      T* input = (T*)((uchar*)src + element_y * src_stride);
      T value = input[element_x];
      data[LENGTH_Y - threadIdx.y - 1][LENGTH_X - threadIdx.x - 1] = value;
    }
    __syncthreads();

    int threadIdx_x = blockIdx.x == gridDim.x - 1 ?
                        src_cols - (blockIdx.x << SHIFT_X) : LENGTH_X;
    int threadIdx_y = blockIdx.y == gridDim.y - 1 ?
                        src_rows - (blockIdx.y << SHIFT_Y) : LENGTH_Y;
    if (threadIdx.x < threadIdx_x && threadIdx.y < threadIdx_y) {
      int dst_x = ((gridDim.x - blockIdx.x - 1) << SHIFT_X) + threadIdx.x;
      int dst_y = ((gridDim.y - blockIdx.y - 1) << SHIFT_Y) + threadIdx.y;
      if (blockIdx.x < gridDim.x - 1) {
        dst_x -= empty_x;
      }
      if (blockIdx.y < gridDim.y - 1) {
        dst_y -= empty_y;
      }

      T* output = (T*)((uchar*)dst + dst_y * dst_stride);
      if (blockIdx.y < gridDim.y - 1) {
        output[dst_x] = data[threadIdx.y][threadIdx.x + empty_x];
      }
      else if (blockIdx.x < gridDim.x - 1) {
        output[dst_x] = data[threadIdx.y + empty_y][threadIdx.x];
      }
      else {
        output[dst_x] = data[threadIdx.y + empty_y][threadIdx.x + empty_x];
      }
    }
  }
}

template <typename T>
__global__
void rotate270Kernel(const T* src, int src_rows, int src_cols, int src_stride,
                     T* dst, int dst_stride) {
  __shared__ T data[LENGTH][LENGTH + 1];
  int element_x = (blockIdx.x << SHIFT) + threadIdx.x;
  int element_y = (blockIdx.y << SHIFT) + threadIdx.y;
  int empty_y = (gridDim.x << SHIFT) - src_cols;

  if (blockIdx.x < gridDim.x - 1 && blockIdx.y < gridDim.y - 1) {
    T* input = (T*)((uchar*)src + element_y * src_stride);
    T value = input[element_x];
    data[LENGTH - threadIdx.x - 1][threadIdx.y] = value;
    __syncthreads();

    int dst_x = (blockIdx.y << SHIFT) + threadIdx.x;
    int dst_y = ((gridDim.x - blockIdx.x - 1) << SHIFT) + threadIdx.y;
    T* output = (T*)((uchar*)dst + (dst_y - empty_y) * dst_stride);
    output[dst_x] = data[threadIdx.y][threadIdx.x];
  }
  else {
    if (element_x < src_cols && element_y < src_rows) {
      T* input = (T*)((uchar*)src + element_y * src_stride);
      T value = input[element_x];
      data[LENGTH - threadIdx.x - 1][threadIdx.y] = value;
    }
    __syncthreads();

    int threadIdx_x = blockIdx.y == gridDim.y - 1 ?
                        src_rows - (blockIdx.y << SHIFT) : LENGTH;
    int threadIdx_y = blockIdx.x == gridDim.x - 1 ?
                        src_cols - (blockIdx.x << SHIFT) : LENGTH;
    if (threadIdx.x < threadIdx_x && threadIdx.y < threadIdx_y) {
      int dst_x = (blockIdx.y << SHIFT) + threadIdx.x;
      int dst_y = ((gridDim.x - blockIdx.x - 1) << SHIFT) + threadIdx.y;
      if (blockIdx.x < gridDim.x - 1) {
        dst_y -= empty_y;
      }
      T* output = (T*)((uchar*)dst + dst_y * dst_stride);
      if (blockIdx.x < gridDim.x - 1) {
        output[dst_x] = data[threadIdx.y][threadIdx.x];
      }
      else {
        output[dst_x] = data[threadIdx.y + empty_y][threadIdx.x];
      }
    }
  }
}

RetCode rotate(const uchar* src, int src_rows, int src_cols, int channels,
               int src_stride, uchar* dst, int dst_rows, int dst_cols,
               int dst_stride, int degree, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);

  dim3 block0, grid0;
  block0.x = LENGTH;
  block0.y = LENGTH;
  grid0.x  = divideUp(src_cols, LENGTH, SHIFT);
  grid0.y  = divideUp(src_rows, LENGTH, SHIFT);

  dim3 block1, grid1;
  block1.x = LENGTH_X;
  block1.y = LENGTH_Y;
  grid1.x  = divideUp(src_cols, LENGTH_X, SHIFT_X);
  grid1.y  = divideUp(src_rows, LENGTH_Y, SHIFT_Y);

  if (degree == 90) {
    if (channels == 1) {
      rotate90Kernel<<<grid0, block0, 0, stream>>>(src, src_rows, src_cols,
          src_stride, (uchar*)dst, dst_stride);
    }
    else if (channels == 3) {
      rotate90Kernel<<<grid0, block0, 0, stream>>>((uchar3*)src, src_rows,
          src_cols, src_stride, (uchar3*)dst, dst_stride);
    }
    else {  // channels == 4
      rotate90Kernel<<<grid0, block0, 0, stream>>>((uchar4*)src, src_rows,
          src_cols, src_stride, (uchar4*)dst, dst_stride);
    }
  } else if (degree == 180) {
    if (channels == 1) {
      rotate180UcharC1Kernel<<<grid1, block1, 0, stream>>>(src, src_rows,
          src_cols, src_stride, dst, dst_stride);
    }
    else if (channels == 3) {
      rotate180Kernel<<<grid1, block1, 0, stream>>>((uchar3*)src, src_rows,
          src_cols, src_stride, (uchar3*)dst, dst_stride);
    }
    else {  // channels == 4
      rotate180Kernel<<<grid1, block1, 0, stream>>>((uchar4*)src, src_rows,
          src_cols, src_stride, (uchar4*)dst, dst_stride);
    }
  }
  else {  // degree == 270
    if (channels == 1) {
      rotate270Kernel<<<grid0, block0, 0, stream>>>(src, src_rows, src_cols,
          src_stride, (uchar*)dst, dst_stride);
    }
    else if (channels == 3) {
      rotate270Kernel<<<grid0, block0, 0, stream>>>((uchar3*)src, src_rows,
          src_cols, src_stride, (uchar3*)dst, dst_stride);
    }
    else {  // channels == 4
      rotate270Kernel<<<grid0, block0, 0, stream>>>((uchar4*)src, src_rows,
          src_cols, src_stride, (uchar4*)dst, dst_stride);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode rotate(const float* src, int src_rows, int src_cols, int channels,
               int src_stride, float* dst, int dst_rows, int dst_cols,
               int dst_stride, int degree, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(float));
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);

  dim3 block0, grid0;
  block0.x = LENGTH;
  block0.y = LENGTH;
  grid0.x  = divideUp(src_cols, LENGTH, SHIFT);
  grid0.y  = divideUp(src_rows, LENGTH, SHIFT);

  dim3 block1, grid1;
  block1.x = LENGTH_X;
  block1.y = LENGTH_Y;
  grid1.x  = divideUp(src_cols, LENGTH_X, SHIFT_X);
  grid1.y  = divideUp(src_rows, LENGTH_Y, SHIFT_Y);

  if (degree == 90) {
    if (channels == 1) {
      rotate90Kernel<<<grid0, block0, 0, stream>>>(src, src_rows, src_cols,
          src_stride, (float*)dst, dst_stride);
    }
    else if (channels == 3) {
      rotate90Kernel<<<grid0, block0, 0, stream>>>((float3*)src, src_rows,
          src_cols, src_stride, (float3*)dst, dst_stride);
    }
    else {  // channels == 4
      rotate90Kernel<<<grid0, block0, 0, stream>>>((float4*)src, src_rows,
          src_cols, src_stride, (float4*)dst, dst_stride);
    }
  } else if (degree == 180) {
    if (channels == 1) {
      rotate180Kernel<<<grid1, block1, 0, stream>>>(src, src_rows, src_cols,
          src_stride, (float*)dst, dst_stride);
    }
    else if (channels == 3) {
      rotate180Kernel<<<grid1, block1, 0, stream>>>((float3*)src, src_rows,
          src_cols, src_stride, (float3*)dst, dst_stride);
    }
    else {  // channels == 4
      rotate180Kernel<<<grid1, block1, 0, stream>>>((float4*)src, src_rows,
          src_cols, src_stride, (float4*)dst, dst_stride);
    }
  }
  else {  // degree == 270
    if (channels == 1) {
      rotate270Kernel<<<grid0, block0, 0, stream>>>(src, src_rows, src_cols,
          src_stride, (float*)dst, dst_stride);
    }
    else if (channels == 3) {
      rotate270Kernel<<<grid0, block0, 0, stream>>>((float3*)src, src_rows,
          src_cols, src_stride, (float3*)dst, dst_stride);
    }
    else {  // channels == 4
      rotate270Kernel<<<grid0, block0, 0, stream>>>((float4*)src, src_rows,
          src_cols, src_stride, (float4*)dst, dst_stride);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Rotate<uchar, 1>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const uchar* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         uchar* outData,
                         int degree) {
  RetCode code = rotate(inData, inHeight, inWidth, 1, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, degree, stream);

  return code;
}

template <>
RetCode Rotate<uchar, 3>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const uchar* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         uchar* outData,
                         int degree) {
  RetCode code = rotate(inData, inHeight, inWidth, 3, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, degree, stream);

  return code;
}

template <>
RetCode Rotate<uchar, 4>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const uchar* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         uchar* outData,
                         int degree) {
  RetCode code = rotate(inData, inHeight, inWidth, 4, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, degree, stream);

  return code;
}

template <>
RetCode Rotate<float, 1>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const float* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         float* outData,
                         int degree) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = rotate(inData, inHeight, inWidth, 1, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, degree, stream);

  return code;
}

template <>
RetCode Rotate<float, 3>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const float* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         float* outData,
                         int degree) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = rotate(inData, inHeight, inWidth, 3, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, degree, stream);

  return code;
}

template <>
RetCode Rotate<float, 4>(cudaStream_t stream,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const float* inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         float* outData,
                         int degree) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = rotate(inData, inHeight, inWidth, 4, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, degree, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
