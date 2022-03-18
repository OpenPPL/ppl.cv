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

#include "ppl/cv/cuda/convertto.h"

#include <cmath>
#include <cfloat>

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename Tsrc, typename Tdst>
__global__
void convertToKernel0(const Tsrc* src, int rows, int cols, int src_stride,
                      Tdst* dst, int dst_stride, float alpha, float beta) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const Tsrc* input = (Tsrc*)((uchar*)src + element_y * src_stride);
  float value0, value1, value2, value3;
  if (element_x < cols - 3) {
    value0 = input[element_x];
    value1 = input[element_x + 1];
    value2 = input[element_x + 2];
    value3 = input[element_x + 3];
  }
  else {
    value0 = input[element_x];
    if (element_x < cols - 1) {
      value1 = input[element_x + 1];
    }
    if (element_x < cols - 2) {
      value2 = input[element_x + 2];
    }
  }

  value0 *= alpha;
  value1 *= alpha;
  value2 *= alpha;
  value3 *= alpha;

  value0 += beta;
  value1 += beta;
  value2 += beta;
  value3 += beta;

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  if (sizeof(Tdst) == 1) {
    if (element_x < cols - 3) {
      output[element_x] = saturateCast(value0);
      output[element_x + 1] = saturateCast(value1);
      output[element_x + 2] = saturateCast(value2);
      output[element_x + 3] = saturateCast(value3);
    }
    else {
      output[element_x] = saturateCast(value0);
      if (element_x < cols - 1) {
        output[element_x + 1] = saturateCast(value1);
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = saturateCast(value2);
      }
    }
  }
  else {
    if (element_x < cols - 3) {
      output[element_x] = value0;
      output[element_x + 1] = value1;
      output[element_x + 2] = value2;
      output[element_x + 3] = value3;
    }
    else {
      output[element_x] = value0;
      if (element_x < cols - 1) {
        output[element_x + 1] = value1;
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = value2;
      }
    }
  }
}

template <typename Tsrc, typename Tdst>
__global__
void convertToKernel1(const Tsrc* src, int rows, int cols, int src_stride,
                      Tdst* dst, int dst_stride, float alpha, float beta) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const Tsrc* input = (Tsrc*)((uchar*)src + element_y * src_stride);
  float value0, value1;
  if (element_x < cols - 1) {
    value0 = input[element_x];
    value1 = input[element_x + 1];
  }
  else {
    value0 = input[element_x];
  }

  value0 *= alpha;
  value1 *= alpha;

  value0 += beta;
  value1 += beta;

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  if (sizeof(Tdst) == 1) {
    if (element_x < cols - 1) {
      output[element_x] = saturateCast(value0);
      output[element_x + 1] = saturateCast(value1);
    }
    else {
      output[element_x] = saturateCast(value0);
    }
  }
  else {
    if (element_x < cols - 1) {
      output[element_x] = value0;
      output[element_x + 1] = value1;
    }
    else {
      output[element_x] = value0;
    }
  }
}

RetCode convertTo(const uchar* src, int rows, int cols, int channels,
                  int src_stride, uchar* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  cudaError_t code;
  if (std::fabs(alpha - 1.f) < FLT_EPSILON && std::fabs(beta) < FLT_EPSILON) {
    if (src == dst) {
      return RC_SUCCESS;
    }

    if (src_stride == dst_stride) {
      code = cudaMemcpy(dst, src, src_stride * rows, cudaMemcpyDeviceToDevice);
    }
    else {
      code = cudaMemcpy2D(dst, dst_stride, src, src_stride,
                          cols * sizeof(uchar), rows, cudaMemcpyDeviceToDevice);
    }
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }

    return RC_SUCCESS;
  }

  int columns = cols * channels;
  cols = divideUp(columns, 4, 2);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  convertToKernel0<uchar, uchar><<<grid, block, 0, stream>>>(src, rows, columns,
      src_stride, dst, dst_stride, alpha, beta);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode convertTo(const uchar* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  cols = divideUp(columns, 2, 1);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  convertToKernel1<uchar, float><<<grid, block, 0, stream>>>(src, rows, columns,
      src_stride, dst, dst_stride, alpha, beta);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode convertTo(const float* src, int rows, int cols, int channels,
                  int src_stride, uchar* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  int columns = cols * channels;
  cols = divideUp(columns, 4, 2);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  convertToKernel0<float, uchar><<<grid, block, 0, stream>>>(src, rows, columns,
      src_stride, dst, dst_stride, alpha, beta);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode convertTo(const float* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  cudaError_t code;
  if (std::fabs(alpha - 1.f) < FLT_EPSILON && std::fabs(beta) < FLT_EPSILON) {
    if (src == dst) {
      return RC_SUCCESS;
    }

    if (src_stride == dst_stride) {
      code = cudaMemcpy(dst, src, src_stride * rows, cudaMemcpyDeviceToDevice);
    }
    else {
      code = cudaMemcpy2D(dst, dst_stride, src, src_stride,
                          cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
    }
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }

    return RC_SUCCESS;
  }

  int columns = cols * channels;
  cols = divideUp(columns, 2, 1);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  convertToKernel1<float, float><<<grid, block, 0, stream>>>(src, rows, columns,
      src_stride, dst, dst_stride, alpha, beta);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode ConvertTo<uchar, uchar, 1>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   float alpha,
                                   float beta) {
  RetCode code = convertTo(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<uchar, uchar, 3>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   float alpha,
                                   float beta) {
  RetCode code = convertTo(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<uchar, uchar, 4>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   float alpha,
                                   float beta) {
  RetCode code = convertTo(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<uchar, float, 1>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   float* outData,
                                   float alpha,
                                   float beta) {
  outWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<uchar, float, 3>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   float* outData,
                                   float alpha,
                                   float beta) {
  outWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<uchar, float, 4>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   float* outData,
                                   float alpha,
                                   float beta) {
  outWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<float, uchar, 1>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   float alpha,
                                   float beta) {
  inWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<float, uchar, 3>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   float alpha,
                                   float beta) {
  inWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<float, uchar, 4>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   float alpha,
                                   float beta) {
  inWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<float, float, 1>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   float* outData,
                                   float alpha,
                                   float beta) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 1, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<float, float, 3>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   float* outData,
                                   float alpha,
                                   float beta) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 3, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

template <>
RetCode ConvertTo<float, float, 4>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   float* outData,
                                   float alpha,
                                   float beta) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = convertTo(inData, height, width, 4, inWidthStride, outData,
                           outWidthStride, alpha, beta, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
