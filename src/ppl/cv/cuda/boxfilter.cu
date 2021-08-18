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

#include "boxfilter.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {



RetCode boxFilter(const uchar* src, int rows, int cols, int channels,
                  int src_stride, int ksize_x, int ksize_y, bool normalize,
                  uchar* dst, int dst_stride, BorderType border_type,
                  cudaStream_t stream)
{
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize_x > 0);
  PPL_ASSERT(ksize_y > 0);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  cudaError_t code;
  if (ksize_x == 1 && ksize_y == 1 && src_stride == dst_stride) {
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


  RetCode return_code = RC_SUCCESS;
  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return return_code;
}

RetCode boxFilter(const float* src, int rows, int cols, int channels,
                  int src_stride, int ksize_x, int ksize_y, bool normalize,
                  float* dst, int dst_stride, BorderType border_type,
                  cudaStream_t stream)
{
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize_x > 0);
  PPL_ASSERT(ksize_y > 0);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  cudaError_t code;
  if (ksize_x == 1 && ksize_y == 1 && src_stride == dst_stride) {
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


  RetCode return_code = RC_SUCCESS;
  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return return_code;
}

template <>
RetCode BoxFilter<uchar, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            uchar* outData,
                            BorderType border_type) {
  RetCode code = boxFilter(inData, height, width, 1, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<uchar, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            uchar* outData,
                            BorderType border_type) {
  RetCode code = boxFilter(inData, height, width, 3, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<uchar, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const uchar* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            uchar* outData,
                            BorderType border_type) {
  RetCode code = boxFilter(inData, height, width, 4, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<float, 1>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            float* outData,
                            BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = boxFilter(inData, height, width, 1, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<float, 3>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            float* outData,
                            BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = boxFilter(inData, height, width, 3, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

template <>
RetCode BoxFilter<float, 4>(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const float* inData,
                            int ksize_x,
                            int ksize_y,
                            bool normalize,
                            int outWidthStride,
                            float* outData,
                            BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = boxFilter(inData, height, width, 4, inWidthStride, ksize_x,
                           ksize_y, normalize, outData, outWidthStride,
                           border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
