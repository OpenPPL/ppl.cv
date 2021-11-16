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

#include "ppl/cv/cuda/equalizehist.h"
#include "ppl/cv/cuda/calchist.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__global__
void equalizehistKernel(const uchar* src, int rows, int cols, int src_stride,
                        int* lut, uchar* dst, int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  uchar* input = (uchar*)((uchar*)src + element_y * src_stride);
  uchar value0, value1, value2, value3;
  value0 = input[element_x];
  value1 = input[element_x + 1];
  value2 = input[element_x + 2];
  value3 = input[element_x + 3];

  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 3) {
    output[element_x] = lut[value0];
    output[element_x + 1] = lut[value1];
    output[element_x + 2] = lut[value2];
    output[element_x + 3] = lut[value3];
  }
  else {
    output[element_x] = lut[value0];
    if (element_x < cols - 1) {
      output[element_x + 1] = lut[value1];
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = lut[value2];
    }
  }
}

RetCode EqualizeHist(cudaStream_t stream, int rows, int cols, int src_stride,
                     const uchar* src, int dst_stride, uchar* dst) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  const int hist_size = 256;
  int* histogram;
  cudaMalloc((void**)&histogram, sizeof(int) * hist_size);
  CalcHist<uchar>(stream, rows, cols, src_stride, src, histogram);

  int hist[hist_size] = {0,};
  int lut[hist_size];
  cudaMemcpyAsync(hist, histogram, sizeof(int) * hist_size,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int i = 0;
  while(!hist[i]) ++i;
  int total = rows * cols;
  float scale = (hist_size - 1.f) / (total - hist[i]);

  int sum = 0;
  for (lut[i++] = 0; i < hist_size; ++i) {
    sum += hist[i];
    lut[i]= round(sum * scale);
  }

  cudaMemcpy(histogram, lut, sizeof(int) * hist_size, cudaMemcpyHostToDevice);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  equalizehistKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
                                                 histogram, dst, dst_stride);
  cudaFree(histogram);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

}  // cuda
}  // cv
}  // ppl
