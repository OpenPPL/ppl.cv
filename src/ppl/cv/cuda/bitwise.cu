/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for anditional information regarding copyright ownership. The ASF
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

#include "ppl/cv/cuda/bitwise.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__global__
void unmaskedBitAndKernel0(const uchar* src0, int rows, int cols, int channels,
                           int src0_stride, const uchar* src1, int src1_stride,
                           uchar* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const uint* input0 = (uint*)(src0 + element_y * src0_stride);
  const uint* input1 = (uint*)(src1 + element_y * src1_stride);
  uint input_value0 = input0[element_x];
  uint input_value1 = input1[element_x];

  uint output_value = input_value0 & input_value1;

  uint* output = (uint*)(dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__global__
void unmaskedBitAndKernel1(const uchar* src0, int rows, int cols, int channels,
                           int src0_stride, const uchar* src1, int src1_stride,
                           uchar* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const uchar* input0 = src0 + element_y * src0_stride;
  const uchar* input1 = src1 + element_y * src1_stride;
  uchar* output = dst + element_y * dst_stride;

  uchar input_value00, input_value01, input_value02, input_value03;
  uchar input_value10, input_value11, input_value12, input_value13;
  uchar output_value0, output_value1, output_value2, output_value3;

  if (index_x < cols - 3) {
    input_value00 = input0[index_x];
    input_value01 = input0[index_x + 1];
    input_value02 = input0[index_x + 2];
    input_value03 = input0[index_x + 3];

    input_value10 = input1[index_x];
    input_value11 = input1[index_x + 1];
    input_value12 = input1[index_x + 2];
    input_value13 = input1[index_x + 3];

    output_value0 = input_value00 & input_value10;
    output_value1 = input_value01 & input_value11;
    output_value2 = input_value02 & input_value12;
    output_value3 = input_value03 & input_value13;

    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
    output[index_x + 2] = output_value2;
    output[index_x + 3] = output_value3;
  }
  else {
    input_value00 = input0[index_x];
    if (index_x < cols - 1) {
      input_value01 = input0[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value02 = input0[index_x + 2];
    }

    input_value10 = input1[index_x];
    if (index_x < cols - 1) {
      input_value11 = input1[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value12 = input1[index_x + 2];
    }

    output_value0 = input_value00 & input_value10;
    if (index_x < cols - 1) {
      output_value1 = input_value01 & input_value11;
    }
    if ((index_x < cols - 2)) {
      output_value2 = input_value02 & input_value12;
    }

    output[index_x] = output_value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = output_value1;
    }
    if ((index_x < cols - 2)) {
      output[index_x + 2] = output_value2;
    }
  }
}

__global__
void maskedBitAndKernel(const uchar* src0, int rows, int cols, int channels,
                        int src0_stride, const uchar* src1, int src1_stride,
                        uchar* mask, int mask_stride, uchar* dst,
                        int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const uchar* input0 = src0 + element_y * src0_stride;
  const uchar* input1 = src1 + element_y * src1_stride;
  uchar* output = dst + element_y * dst_stride;

  uchar input_value00, input_value01, input_value02, input_value03;
  uchar input_value10, input_value11, input_value12, input_value13;
  uchar output_value0 = 0, output_value1 = 0;
  uchar output_value2 = 0, output_value3 = 0;

  if (index_x < cols - 3) {
    input_value00 = input0[index_x];
    input_value01 = input0[index_x + 1];
    input_value02 = input0[index_x + 2];
    input_value03 = input0[index_x + 3];

    input_value10 = input1[index_x];
    input_value11 = input1[index_x + 1];
    input_value12 = input1[index_x + 2];
    input_value13 = input1[index_x + 3];

    uchar* mask_start;
    uchar mvalue0, mvalue1, mvalue2, mvalue3;
    mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
    if (channels == 1) {
      mvalue0 = mask_start[index_x];
      mvalue1 = mask_start[index_x + 1];
      mvalue2 = mask_start[index_x + 2];
      mvalue3 = mask_start[index_x + 3];
    }
    else if (channels == 3) {
      mvalue0 = mask_start[index_x / channels];
      mvalue1 = mask_start[(index_x + 1) / channels];
      mvalue2 = mask_start[(index_x + 2) / channels];
      mvalue3 = mask_start[(index_x + 3) / channels];
    }
    else {  // channels == 4
      mvalue0 = mask_start[index_x >> 2];
      mvalue1 = mask_start[(index_x + 1) >> 2];
      mvalue2 = mask_start[(index_x + 2) >> 2];
      mvalue3 = mask_start[(index_x + 3) >> 2];
    }
    if (mvalue0 > 0) {
      output_value0 = input_value00 & input_value10;
    }
    if (mvalue1 > 0) {
      output_value1 = input_value01 & input_value11;
    }
    if (mvalue2 > 0) {
      output_value2 = input_value02 & input_value12;
    }
    if (mvalue3 > 0) {
      output_value3 = input_value03 & input_value13;
    }

    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
    output[index_x + 2] = output_value2;
    output[index_x + 3] = output_value3;
  }
  else {
    input_value00 = input0[index_x];
    if (index_x < cols - 1) {
      input_value01 = input0[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value02 = input0[index_x + 2];
    }

    input_value10 = input1[index_x];
    if (index_x < cols - 1) {
      input_value11 = input1[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value12 = input1[index_x + 2];
    }

    uchar* mask_start;
    uchar mvalue0, mvalue1, mvalue2;
    mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
    if (channels == 1) {
      mvalue0 = mask_start[index_x];
      mvalue1 = mask_start[index_x + 1];
      mvalue2 = mask_start[index_x + 2];
    }
    else if (channels == 3) {
      mvalue0 = mask_start[index_x / channels];
      mvalue1 = mask_start[(index_x + 1) / channels];
      mvalue2 = mask_start[(index_x + 2) / channels];
    }
    else {  // channels == 4
      mvalue0 = mask_start[index_x >> 2];
      mvalue1 = mask_start[(index_x + 1) >> 2];
      mvalue2 = mask_start[(index_x + 2) >> 2];
    }
    if (mvalue0 > 0) {
      output_value0 = input_value00 & input_value10;
    }
    if (mvalue1 > 0 && index_x < cols - 1) {
      output_value1 = input_value01 & input_value11;
    }
    if (mvalue2 > 0 && index_x < cols - 2) {
      output_value2 = input_value02 & input_value12;
    }

    output[index_x] = output_value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = output_value1;
    }
    if ((index_x < cols - 2)) {
      output[index_x + 2] = output_value2;
    }
  }
}

RetCode bitwiseAnd(const uchar* src0, int rows, int cols, int channels,
                   int src0_stride, const uchar* src1, int src1_stride,
                   uchar* mask, int mask_stride, uchar* dst, int dst_stride,
                   cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(uchar));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  int columns = cols * channels;
  cols = divideUp(columns, 4, 2);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if (mask == nullptr) {
    if ((src0_stride & 3) == 0 && (src1_stride & 3) == 0 &&
        (dst_stride & 3) == 0) {
      unmaskedBitAndKernel0<<<grid, block, 0, stream>>>(src0, rows, cols,
          channels, src0_stride, src1, src1_stride, dst, dst_stride);
    }
    else {
      unmaskedBitAndKernel1<<<grid, block, 0, stream>>>(src0, rows, columns,
          channels, src0_stride, src1, src1_stride, dst, dst_stride);
    }
  }
  else {
    maskedBitAndKernel<<<grid, block, 0, stream>>>(src0, rows, columns,
        channels, src0_stride, src1, src1_stride, mask, mask_stride, dst,
        dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode BitwiseAnd<uchar, 1>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride0,
                             const uchar* inData0,
                             int inWidthStride1,
                             const uchar* inData1,
                             int outWidthStride,
                             uchar* outData,
                             int maskWidthStride,
                             uchar* mask) {
  RetCode code = bitwiseAnd(inData0, height, width, 1, inWidthStride0, inData1,
                            inWidthStride1, mask, maskWidthStride, outData,
                            outWidthStride, stream);

  return code;
}

template <>
RetCode BitwiseAnd<uchar, 3>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride0,
                             const uchar* inData0,
                             int inWidthStride1,
                             const uchar* inData1,
                             int outWidthStride,
                             uchar* outData,
                             int maskWidthStride,
                             uchar* mask) {
  RetCode code = bitwiseAnd(inData0, height, width, 3, inWidthStride0, inData1,
                            inWidthStride1, mask, maskWidthStride, outData,
                            outWidthStride, stream);

  return code;
}

template <>
RetCode BitwiseAnd<uchar, 4>(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride0,
                             const uchar* inData0,
                             int inWidthStride1,
                             const uchar* inData1,
                             int outWidthStride,
                             uchar* outData,
                             int maskWidthStride,
                             uchar* mask) {
  RetCode code = bitwiseAnd(inData0, height, width, 4, inWidthStride0, inData1,
                            inWidthStride1, mask, maskWidthStride, outData,
                            outWidthStride, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
