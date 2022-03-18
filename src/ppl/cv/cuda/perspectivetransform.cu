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

#include "ppl/cv/cuda/perspectivetransform.h"

#include <cfloat>

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__global__
void perspectiveTransformKernel(const float* src, int rows, int cols,
         int src_channels, int src_stride, float coeff0, float coeff1,
         float coeff2, float coeff3, float coeff4, float coeff5, float coeff6,
         float coeff7, float coeff8, float coeff9, float coeff10, float coeff11,
         float coeff12, float coeff13, float coeff14, float coeff15, float* dst,
         int dst_channels, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float* input = (float*)((uchar*)src + element_y * src_stride);
  int index_x = element_x * src_channels;

  float input_value0, input_value1, input_value2;
  float output_value0 = 0.f, output_value1 = 0.f, output_value2 = 0.f, weight;

  if (src_channels == 2) {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];
  }
  else {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];
    input_value2 = input[index_x + 2];
  }

  if (src_channels == 2 && dst_channels == 2) {
    weight = input_value0 * coeff6 + input_value1 * coeff7 + coeff8;
    if (fabsf(weight) > FLT_EPSILON) {
      weight = 1.f / weight;
      output_value0 = (input_value0 * coeff0 + input_value1 * coeff1 +
                       coeff2) * weight;
      output_value1 = (input_value0 * coeff3 + input_value1 * coeff4 +
                       coeff5) * weight;
    }
  }
  else if (src_channels == 3 && dst_channels == 3) {
    weight = input_value0 * coeff12 + input_value1 * coeff13 +
             input_value2 * coeff14 + coeff15;
    if (fabsf(weight) > FLT_EPSILON) {
      weight = 1.f / weight;
      output_value0 = (input_value0 * coeff0 + input_value1 * coeff1 +
                       input_value2 * coeff2 + coeff3) * weight;
      output_value1 = (input_value0 * coeff4 + input_value1 * coeff5 +
                       input_value2 * coeff6 + coeff7) * weight;
      output_value2 = (input_value0 * coeff8 + input_value1 * coeff9 +
                       input_value2 * coeff10 + coeff11) * weight;
    }
  }
  else if (src_channels == 3 && dst_channels == 2) {
    weight = input_value0 * coeff8 + input_value1 * coeff9 +
             input_value2 * coeff10 + coeff11;
    if (fabsf(weight) > FLT_EPSILON) {
      weight = 1.f / weight;
      output_value0 = (input_value0 * coeff0 + input_value1 * coeff1 +
                       input_value2 * coeff2 + coeff3) * weight;
      output_value1 = (input_value0 * coeff4 + input_value1 * coeff5 +
                       input_value2 * coeff6 + coeff7) * weight;
    }
  }
  else {  // src_channels == 2 && dst_channels == 3
    weight = input_value0 * coeff9 + input_value1 * coeff10 + coeff11;
    if (fabsf(weight) > FLT_EPSILON) {
      // weight = 1.f / weight;
      output_value0 = (input_value0 * coeff0 + input_value1 * coeff1 +
                       coeff2) * weight;
      output_value1 = (input_value0 * coeff3 + input_value1 * coeff4 +
                       coeff5) * weight;
      output_value2 = (input_value0 * coeff6 + input_value1 * coeff7 +
                       coeff8) * weight;
    }
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  index_x = element_x * dst_channels;
  if (dst_channels == 2) {
    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
  }
  else {
    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
    output[index_x + 2] = output_value2;
  }
}

RetCode perspectiveTransform(const float* src, int rows, int cols,
                             int src_channels, int src_stride,
                             const float* trans_matrix, float* dst,
                             int dst_channels, int dst_stride,
                             cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(trans_matrix != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 0 && cols > 0);
  PPL_ASSERT(src_channels == 2 || src_channels == 3);
  PPL_ASSERT(dst_channels == 2 || dst_channels == 3);
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if (src_channels == 2 && dst_channels == 2) {
    perspectiveTransformKernel<<<grid, block, 0, stream>>>(src, rows, cols,
        src_channels, src_stride, trans_matrix[0], trans_matrix[1],
        trans_matrix[2], trans_matrix[3], trans_matrix[4], trans_matrix[5],
        trans_matrix[6], trans_matrix[7], trans_matrix[8], trans_matrix[9],
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, dst, dst_channels, dst_stride);
  }
  else {
    perspectiveTransformKernel<<<grid, block, 0, stream>>>(src, rows, cols,
        src_channels, src_stride, trans_matrix[0], trans_matrix[1],
        trans_matrix[2], trans_matrix[3], trans_matrix[4], trans_matrix[5],
        trans_matrix[6], trans_matrix[7], trans_matrix[8], trans_matrix[9],
        trans_matrix[10], trans_matrix[11], trans_matrix[12], trans_matrix[13],
        trans_matrix[14], trans_matrix[15], dst, dst_channels, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode PerspectiveTransform<float, 2, 2>(cudaStream_t stream,
                                          int height,
                                          int width,
                                          int inWidthStride,
                                          const float* inData,
                                          int outWidthStride,
                                          float* outData,
                                          const float* transData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = perspectiveTransform(inData, height, width, 2, inWidthStride,
                                      transData, outData, 2, outWidthStride,
                                      stream);

  return code;
}

template <>
RetCode PerspectiveTransform<float, 2, 3>(cudaStream_t stream,
                                          int height,
                                          int width,
                                          int inWidthStride,
                                          const float* inData,
                                          int outWidthStride,
                                          float* outData,
                                          const float* transData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = perspectiveTransform(inData, height, width, 2, inWidthStride,
                                      transData, outData, 3, outWidthStride,
                                      stream);

  return code;
}

template <>
RetCode PerspectiveTransform<float, 3, 2>(cudaStream_t stream,
                                          int height,
                                          int width,
                                          int inWidthStride,
                                          const float* inData,
                                          int outWidthStride,
                                          float* outData,
                                          const float* transData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = perspectiveTransform(inData, height, width, 3, inWidthStride,
                                      transData, outData, 2, outWidthStride,
                                      stream);

  return code;
}

template <>
RetCode PerspectiveTransform<float, 3, 3>(cudaStream_t stream,
                                          int height,
                                          int width,
                                          int inWidthStride,
                                          const float* inData,
                                          int outWidthStride,
                                          float* outData,
                                          const float* transData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = perspectiveTransform(inData, height, width, 3, inWidthStride,
                                      transData, outData, 3, outWidthStride,
                                      stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
