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

#include "gaussianblur.h"

#include <cmath>

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define SMALL_SIZE 7

RetCode sepfilter2D(const uchar* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, uchar* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream);

RetCode sepfilter2D(const float* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, float* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream);

void getGaussianKernel(float* coefficients, float sigma, int ksize) {
  const float small_kernels[][SMALL_SIZE] = {
    {1.f},
    {0.25f, 0.5f, 0.25f},
    {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
    {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
  };
  bool fix = (ksize % 2 == 1) && (ksize <= SMALL_SIZE) && (sigma <= 0);
  const float* fixed_kernel = fix ? small_kernels[(ksize >> 1)] : 0;
  double sigma_x = sigma > 0 ? sigma : ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8;
  double scale_2x = -0.5 / (sigma_x * sigma_x);
  double sum = 0;

  int i;
  for (i = 0; i < ksize; i++) {
    double x = i - (ksize - 1) * 0.5;
    double value = fixed_kernel ? (double)fixed_kernel[i] :
                   std::exp(scale_2x * x * x);
    coefficients[i] = (float)value;
    sum += coefficients[i];
  }

  sum = 1.0 / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] = float(coefficients[i] * sum);
  }
}

RetCode gaussianblur(const uchar* src, int rows, int cols, int channels,
                     int src_stride, int ksize, float sigma, uchar* dst,
                     int dst_stride, BorderType border_type,
                     cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  cudaError_t code;
  if (ksize == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, rows * src_stride,
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int kernel_size = ksize * sizeof(float);
  float* kernel = (float*)malloc(kernel_size);
  float* gpu_kernel;
  code = cudaMalloc((void**)&gpu_kernel, kernel_size);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  getGaussianKernel(kernel, sigma, ksize);
  code = cudaMemcpy(gpu_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }

  RetCode return_code = RC_SUCCESS;
  return_code = sepfilter2D(src, rows, cols, channels, src_stride, gpu_kernel,
                            gpu_kernel, ksize, dst, dst_stride, 0, border_type,
                            stream);

  free(kernel);
  cudaFree(gpu_kernel);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return return_code;
}

RetCode gaussianblur(const float* src, int rows, int cols, int channels,
                     int src_stride, int ksize, float sigma, float* dst,
                     int dst_stride, BorderType border_type,
                     cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  cudaError_t code;
  if (ksize == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, rows * src_stride,
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int kernel_size = ksize * sizeof(float);
  float* kernel = (float*)malloc(kernel_size);
  float* gpu_kernel;
  code = cudaMalloc((void**)&gpu_kernel, kernel_size);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  getGaussianKernel(kernel, sigma, ksize);
  code = cudaMemcpy(gpu_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }

  RetCode return_code = RC_SUCCESS;
  return_code = sepfilter2D(src, rows, cols, channels, src_stride, gpu_kernel,
                            gpu_kernel, ksize, dst, dst_stride, 0, border_type,
                            stream);

  free(kernel);
  cudaFree(gpu_kernel);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return return_code;
}

template <>
RetCode GaussianBlur<uchar, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               uchar* outData,
                               BorderType border_type) {
  RetCode code = gaussianblur(inData, height, width, 1, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<uchar, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               uchar* outData,
                               BorderType border_type) {
  RetCode code = gaussianblur(inData, height, width, 3, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<uchar, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               uchar* outData,
                               BorderType border_type) {
  RetCode code = gaussianblur(inData, height, width, 4, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<float, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               float* outData,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = gaussianblur(inData, height, width, 1, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<float, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               float* outData,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = gaussianblur(inData, height, width, 3, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<float, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               float* outData,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = gaussianblur(inData, height, width, 4, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
