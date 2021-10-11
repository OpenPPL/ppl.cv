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

#include "ppl/cv/cuda/zeros.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

RetCode zeros(uchar* dst, int rows, int cols, int channels, int dst_stride,
              cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  cudaError_t code;
  code = cudaMemset2DAsync(dst, dst_stride, 0, cols * channels * sizeof(uchar),
                           rows, stream);

  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode zeros(float* dst, int rows, int cols, int channels, int dst_stride,
              cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  cudaError_t code;
  code = cudaMemset2DAsync(dst, dst_stride, 0, cols * channels * sizeof(float),
                           rows, stream);

  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Zeros<uchar, 1>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        uchar* outData) {
  RetCode code = zeros(outData, height, width, 1, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<uchar, 3>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        uchar* outData) {
  RetCode code = zeros(outData, height, width, 3, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<uchar, 4>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        uchar* outData) {
  RetCode code = zeros(outData, height, width, 4, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<float, 1>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = zeros(outData, height, width, 1, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<float, 3>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = zeros(outData, height, width, 3, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<float, 4>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = zeros(outData, height, width, 4, outWidthStride, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
