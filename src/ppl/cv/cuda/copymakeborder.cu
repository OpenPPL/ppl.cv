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

#include "ppl/cv/cuda/copymakeborder.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__DEVICE__
int borderInterpolate0(int index, int range, BorderType border_type) {
  if (border_type == BORDER_DEFAULT || border_type == BORDER_REFLECT_101) {
    if (index < 0) {
      return 0 - index;
    }
    else if (index < range) {
      return index;
    }
    else {
      return (range << 1) - index - 2;
    }
  }
  else if (border_type == BORDER_CONSTANT) {
    if (index < 0) {
      return -1;
    }
    else if (index < range) {
      return index;
    }
    else {
      return -1;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    if (index < 0) {
      return 0;
    }
    else if (index < range) {
      return index;
    }
    else {
      return range - 1;
    }
  }
  else if (border_type == BORDER_REFLECT) {
    if (index < 0) {
      return -1 - index;
    }
    else if (index < range) {
      return index;
    }
    else {
      return (range << 1) - index - 1;
    }
  }
  else if (border_type == BORDER_WRAP) {
    if (index < 0) {
      return index + range;
    }
    else if (index < range) {
      return index;
    }
    else {
      return index - range;
    }
  }
  else {
    return -2;
  }
}

__DEVICE__
int borderInterpolate1(int index, int range, BorderType border_type) {
  if (border_type == BORDER_DEFAULT || border_type == BORDER_REFLECT_101) {
    if (index >= 0 && index < range) {
      return index;
    }
    else {
      if (range == 1) {
        index = 0;
      }
      else {
        do {
          if (index < 0)
            index = 0 - index;
          else
            index = (range << 1) - index - 2;
        } while (index >= range || index < 0);
      }

      return index;
    }
  }
  else if (border_type == BORDER_CONSTANT) {
    if (index < 0) {
      return -1;
    }
    else if (index < range) {
      return index;
    }
    else {
      return -1;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    if (index < 0) {
      return 0;
    }
    else if (index < range) {
      return index;
    }
    else {
      return range - 1;
    }
  }
  else if (border_type == BORDER_REFLECT) {
    if (index >= 0 && index < range) {
      return index;
    }
    else {
      if (range == 1) {
        index = 0;
      }
      else {
        do {
          if (index < 0)
            index = -1 - index;
          else
            index = (range << 1) - index - 1;
        } while (index >= range || index < 0);
      }

      return index;
    }
  }
  else if (border_type == BORDER_WRAP) {
    if (index >= 0 && index < range) {
      return index;
    }
    else {
      if (range == 1) {
        index = 0;
      }
      else {
        do {
          if (index < 0)
            index += range;
          else
            index -= range;
        } while (index >= range || index < 0);
      }

      return index;
    }
  }
  else {
    return -2;
  }
}

template <typename T0, typename T1>
__DEVICE__
T0 makeValuen(T1 value);

template <>
__DEVICE__
uchar makeValuen<uchar, uchar>(uchar value) {
  return value;
}

template <>
__DEVICE__
uchar3 makeValuen<uchar3, uchar>(uchar value) {
  return make_uchar3(value, value, value);
}

template <>
__DEVICE__
uchar4 makeValuen<uchar4, uchar>(uchar value) {
  return make_uchar4(value, value, value, value);
}

template <>
__DEVICE__
float makeValuen<float, float>(float value) {
  return value;
}

template <>
__DEVICE__
float3 makeValuen<float3, float>(float value) {
  return make_float3(value, value, value);
}

template <>
__DEVICE__
float4 makeValuen<float4, float>(float value) {
  return make_float4(value, value, value, value);
}

template <typename T0, typename T1>
__global__
void copyMakeBorderKernel(const T0* src, int rows, int cols, int src_stride,
                          T0* dst, int dst_stride, int top, int bottom,
                          int left, int right, BorderType border_type,
                          T1 border_value, bool small_border) {
  int element_x, element_y;
  if (sizeof(T1) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= (rows + top + bottom) ||
      element_x >= (cols + left + right)) {
    return;
  }

  int src_x = element_x - left;
  int src_y = element_y - top;
  if (small_border == true) {
    src_x = borderInterpolate0(src_x, cols, border_type);
    src_y = borderInterpolate0(src_y, rows, border_type);
  }
  else {
    src_x = borderInterpolate1(src_x, cols, border_type);
    src_y = borderInterpolate1(src_y, rows, border_type);
  }

  T0 value;
  T0 *input, *output;
  if (border_type != BORDER_CONSTANT) {
    input = (T0*)((uchar*)src + src_y * src_stride);
    value = input[src_x];
  }
  else {
    if (src_x != -1 && src_y != -1) {
      input = (T0*)((uchar*)src + src_y * src_stride);
      value = input[src_x];
    }
    else {
      value = makeValuen<T0, T1>(border_value);
    }
  }

  output = (T0*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = value;
}

RetCode copyMakeBorder(const uchar* src, int rows, int cols, int channels,
                       int src_stride, uchar* dst, int dst_stride, int top,
                       int bottom, int left, int right, BorderType border_type,
                       uchar border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 0 && cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(top >= 0);
  PPL_ASSERT(bottom >= 0);
  PPL_ASSERT(left >= 0);
  PPL_ASSERT(right >= 0);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_WRAP ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (top == 0 && bottom == 0 && left == 0 && right == 0 &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, rows * src_stride, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp((cols + left + right), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp((rows + top + bottom), kBlockDimY0, kBlockShiftY0);

  bool small_border = false;
  if (rows > top && rows > bottom && cols > left && cols > right) {
    small_border = true;
  }

  if (channels == 1) {
    copyMakeBorderKernel<uchar, uchar><<<grid, block, 0, stream>>>(src, rows,
        cols, src_stride, dst, dst_stride, top, bottom, left, right,
        border_type, border_value, small_border);
  }
  else if (channels == 3) {
    copyMakeBorderKernel<uchar3, uchar><<<grid, block, 0, stream>>>(
        (uchar3*)src, rows, cols, src_stride, (uchar3*)dst, dst_stride, top,
        bottom, left, right, border_type, border_value, small_border);
  }
  else {  // channels == 4
    copyMakeBorderKernel<uchar4, uchar><<<grid, block, 0, stream>>>(
        (uchar4*)src, rows, cols, src_stride, (uchar4*)dst, dst_stride, top,
        bottom, left, right, border_type, border_value, small_border);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode copyMakeBorder(const float* src, int rows, int cols, int channels,
                       int src_stride, float* dst, int dst_stride, int top,
                       int bottom, int left, int right, BorderType border_type,
                       float border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 0 && cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(top >= 0);
  PPL_ASSERT(bottom >= 0);
  PPL_ASSERT(left >= 0);
  PPL_ASSERT(right >= 0);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_WRAP ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (top == 0 && bottom == 0 && left == 0 && right == 0 &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, rows * src_stride, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp((cols + left + right), kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp((rows + top + bottom), kBlockDimY1, kBlockShiftY1);

  bool small_border = false;
  if (rows > top && rows > bottom && cols > left && cols > right) {
    small_border = true;
  }

  if (channels == 1) {
    copyMakeBorderKernel<float, float><<<grid, block, 0, stream>>>(src, rows,
        cols, src_stride, dst, dst_stride, top, bottom, left, right,
        border_type, border_value, small_border);
  }
  else if (channels == 3) {
    copyMakeBorderKernel<float3, float><<<grid, block, 0, stream>>>(
        (float3*)src, rows, cols, src_stride, (float3*)dst, dst_stride, top,
        bottom, left, right, border_type, border_value, small_border);
  }
  else {  // channels == 4
    copyMakeBorderKernel<float4, float><<<grid, block, 0, stream>>>(
        (float4*)src, rows, cols, src_stride, (float4*)dst, dst_stride, top,
        bottom, left, right, border_type, border_value, small_border);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode CopyMakeBorder<uchar, 1>(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const uchar* inData,
                                 int outWidthStride,
                                 uchar* outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 uchar border_value) {
  RetCode code = copyMakeBorder(inData, height, width, 1, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, stream);

  return code;
}

template <>
RetCode CopyMakeBorder<uchar, 3>(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const uchar* inData,
                                 int outWidthStride,
                                 uchar* outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 uchar border_value) {
  RetCode code = copyMakeBorder(inData, height, width, 3, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, stream);

  return code;
}

template <>
RetCode CopyMakeBorder<uchar, 4>(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const uchar* inData,
                                 int outWidthStride,
                                 uchar* outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 uchar border_value) {
  RetCode code = copyMakeBorder(inData, height, width, 4, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, stream);

  return code;
}

template <>
RetCode CopyMakeBorder<float, 1>(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const float* inData,
                                 int outWidthStride,
                                 float* outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = copyMakeBorder(inData, height, width, 1, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, stream);

  return code;
}

template <>
RetCode CopyMakeBorder<float, 3>(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const float* inData,
                                 int outWidthStride,
                                 float* outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = copyMakeBorder(inData, height, width, 3, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, stream);

  return code;
}

template <>
RetCode CopyMakeBorder<float, 4>(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const float* inData,
                                 int outWidthStride,
                                 float* outData,
                                 int top,
                                 int bottom,
                                 int left,
                                 int right,
                                 BorderType border_type,
                                 float border_value) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = copyMakeBorder(inData, height, width, 4, inWidthStride,
                                outData, outWidthStride, top, bottom, left,
                                right, border_type, border_value, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
