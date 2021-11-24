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

#include "ppl/cv/cuda/guidedfilter.h"

#include <cfloat>

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/**************************** function declaration **************************/

RetCode convertTo(const uchar* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream);

RetCode convertTo(const float* src, int rows, int cols, int channels,
                  int src_stride, uchar* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream);

/******************************** add() *********************************/

__global__
void addKernel1(const float* src0, const float* src1, float* dst, int rows,
                int cols, int src0_stride, int src1_stride, int dst_stride,
                int src0_pitch, int src1_pitch, int dst_pitch) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_pitch + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_pitch + offset);
  float* output  = (float*)((uchar*)dst + dst_pitch + element_y * dst_stride);

  float input_value00, input_value01;
  float input_value10, input_value11;
  float output_value0, output_value1;

  input_value00 = input0[element_x];
  input_value01 = input0[element_x + 1];

  input_value10 = input1[element_x];
  input_value11 = input1[element_x + 1];

  output_value0 = input_value00 + input_value10;
  output_value1 = input_value01 + input_value11;

  output[element_x]     = output_value0;
  output[element_x + 1] = output_value1;
}

__global__
void addKernel2(const float* src0, const float* src1, float* dst, int rows,
                int cols, int src0_stride, int src1_stride, int dst_stride,
                int src0_pitch, int src1_pitch, int dst_pitch) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_pitch + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_pitch + offset);
  float* output  = (float*)((uchar*)dst + dst_pitch + element_y * dst_stride);

  if (blockIdx.x < gridDim.x - 1) {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    output_value0 = input_value00 + input_value10;
    output_value1 = input_value01 + input_value11;

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    output_value0 = input_value00 + input_value10;
    if (element_x != cols - 1) {
      output_value1 = input_value01 + input_value11;
    }

    output[element_x] = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode add(const float* src0, const float* src1, float* dst, int rows,
        int cols, int dst_stride, int src0_pitch, int src1_pitch,
        int dst_pitch, int channels, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  int padded_stride = roundUp(cols, 2, 1) * channels * sizeof(float);
  if (dst_stride >= padded_stride) {
    addKernel1<<<grid, block, 0, stream>>>(src0, src1, dst, rows, columns,
                                            dst_stride, dst_stride, dst_stride,
                                            src0_pitch, src1_pitch, dst_pitch);
  }
  else {
    addKernel2<<<grid, block, 0, stream>>>(src0, src1, dst, rows, columns,
                                            dst_stride, dst_stride, dst_stride,
                                            src0_pitch, src1_pitch, dst_pitch);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** add_scalar() ******************************/

__global__
void add_scalarKernel(float* dst, int rows, int columns, int stride,
                      int dst_pitch, float value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= columns) {
    return;
  }

  int offset = element_y * stride;
  float* output = (float*)((uchar*)dst + dst_pitch + offset);
  float output0 = output[element_x];
  output0 += value;

  output[element_x] = output0;
}

RetCode add_scalar(float* dst, int rows, int cols, int stride, int dst_pitch,
                   int channels, float value, cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(stride > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(columns, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  add_scalarKernel<<<grid, block, 0, stream>>>(dst, rows, columns, stride,
                                                dst_pitch, value);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** multiply() ******************************/

__global__
void multiplyKernel1(const float* src0, int src0_pitch, const float* src1,
                     int src1_pitch, float* dst, int dst_pitch,
                     int rows, int cols, int src0_stride, int src1_stride,
                     int dst_stride, double scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_pitch + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_pitch + offset);
  float* output  = (float*)((uchar*)dst + dst_pitch + element_y * dst_stride);

  float input_value00, input_value01;
  float input_value10, input_value11;
  float output_value0, output_value1;

  input_value00 = input0[element_x];
  input_value01 = input0[element_x + 1];

  input_value10 = input1[element_x];
  input_value11 = input1[element_x + 1];

  if (scale == 1) {
    output_value0 = input_value00 * input_value10;
    output_value1 = input_value01 * input_value11;
  }
  else {
    output_value0 = input_value00 * input_value10 * scale;
    output_value1 = input_value01 * input_value11 * scale;
  }

  output[element_x]     = output_value0;
  output[element_x + 1] = output_value1;
}

__global__
void multiplyKernel2(const float* src0, int src0_pitch, const float* src1,
                     int src1_pitch, float* dst, int dst_pitch,
                     int rows, int cols, int src0_stride, int src1_stride,
                     int dst_stride, double scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_pitch + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_pitch + offset);
  float* output  = (float*)((uchar*)dst + dst_pitch + element_y * dst_stride);

  if (blockIdx.x < gridDim.x - 1) {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    if (scale == 1) {
      output_value0 = input_value00 * input_value10;
      output_value1 = input_value01 * input_value11;
    }
    else {
      output_value0 = input_value00 * input_value10 * scale;
      output_value1 = input_value01 * input_value11 * scale;
    }

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    if (scale == 1) {
      output_value0 = input_value00 * input_value10;
      if (element_x != cols - 1) {
        output_value1 = input_value01 * input_value11;
      }
    }
    else {
      output_value0 = input_value00 * input_value10 * scale;
      if (element_x != cols - 1) {
        output_value1 = input_value01 * input_value11 * scale;
      }
    }

    output[element_x]     = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode multiply(cudaStream_t stream, int rows, int cols, int channels,
                 int src0_stride, int src0_pitch, const float* src0,
                 int src1_stride, int src1_pitch, const float* src1,
                 int dst_stride, int dst_pitch, float* dst, double scale) {
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  int padded_stride = roundUp(cols, 2, 1) * channels * sizeof(float);
  if (dst_stride >= padded_stride) {
    multiplyKernel1<<<grid, block, 0, stream>>>(src0, src0_pitch, src1,
                                                 src1_pitch, dst, dst_pitch,
                                                 rows, columns,
                                                 src0_stride, src1_stride,
                                                 dst_stride, scale);
  }
  else {
    multiplyKernel2<<<grid, block, 0, stream>>>(src0, src0_pitch, src1,
                                                 src1_pitch, dst, dst_pitch,
                                                 rows, columns,
                                                 src0_stride, src1_stride,
                                                 dst_stride, scale);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** divide() ******************************/

__global__
void divideKernel1(const float* src0, int src0_pitch, const float* src1,
                   int src1_pitch, float* dst, int dst_pitch,
                   int rows, int cols, int src0_stride, int src1_stride,
                   int dst_stride, double scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_pitch + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_pitch + offset);
  float* output  = (float*)((uchar*)dst + dst_pitch + element_y * dst_stride);

  float input_value00, input_value01;
  float input_value10, input_value11;
  float output_value0, output_value1;

  input_value00 = input0[element_x];
  input_value01 = input0[element_x + 1];

  input_value10 = input1[element_x];
  input_value11 = input1[element_x + 1];

  if (scale == 1) {
    output_value0 = input_value10 == 0 ? 0 : input_value00 / input_value10;
    output_value1 = input_value11 == 0 ? 0 : input_value01 / input_value11;
  }
  else {
    output_value0 = input_value10 == 0 ? 0 :
                      scale * input_value00 / input_value10;
    output_value1 = input_value11 == 0 ? 0 :
                      scale * input_value01 / input_value11;
  }

  output[element_x]     = output_value0;
  output[element_x + 1] = output_value1;
}

__global__
void divideKernel2(const float* src0, int src0_pitch, const float* src1,
                   int src1_pitch, float* dst, int dst_pitch,
                   int rows, int cols, int src0_stride, int src1_stride,
                   int dst_stride, double scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_pitch + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_pitch + offset);
  float* output  = (float*)((uchar*)dst + dst_pitch + element_y * dst_stride);

  if (blockIdx.x < gridDim.x - 1) {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    if (scale == 1) {
      output_value0 = input_value10 == 0 ? 0 : input_value00 / input_value10;
      output_value1 = input_value11 == 0 ? 0 : input_value01 / input_value11;
    }
    else {
      output_value0 = input_value10 == 0 ? 0 :
                        scale * input_value00 / input_value10;
      output_value1 = input_value11 == 0 ? 0 :
                        scale * input_value01 / input_value11;
    }

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    if (scale == 1) {
      output_value0 = input_value10 == 0 ? 0 : input_value00 / input_value10;
      if (element_x != cols - 1) {
        output_value1 = input_value11 == 0 ? 0 : input_value01 / input_value11;
      }
    }
    else {
      output_value0 = input_value10 == 0 ? 0 :
                        scale * input_value00 / input_value10;
      if (element_x != cols - 1) {
        output_value1 = input_value11 == 0 ? 0 :
                        scale * input_value01 / input_value11;
      }
    }

    output[element_x] = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode divide(cudaStream_t stream, int rows, int cols, int channels,
               int src0_stride, int src0_pitch, const float* src0,
               int src1_stride, int src1_pitch, const float* src1,
               int dst_stride, int dst_pitch, float* dst, double scale) {
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  int padded_stride = roundUp(cols, 2, 1) * channels * sizeof(float);
  if (dst_stride >= padded_stride) {
    divideKernel1<<<grid, block, 0, stream>>>(src0, src0_pitch, src1,
                                               src1_pitch, dst, dst_pitch,
                                               rows, columns,
                                               src0_stride, src1_stride,
                                               dst_stride, scale);
  }
  else {
    divideKernel2<<<grid, block, 0, stream>>>(src0, src0_pitch, src1,
                                               src1_pitch, dst, dst_pitch,
                                               rows, columns,
                                               src0_stride, src1_stride,
                                               dst_stride, scale);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** subtract() ******************************/

__global__
void subtractKernel(const float* src1, const float* src2, float* dst,
                    int rows, int columns, int stride, int src1_pitch,
                    int src2_pitch, int dst_pitch) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= columns) {
    return;
  }

  int offset = element_y * stride;
  float* input1 = (float*)((uchar*)src1 + src1_pitch + offset);
  float value10 = input1[element_x];

  float* input2 = (float*)((uchar*)src2 + src2_pitch + offset);
  float value20 = input2[element_x];

  float output0 = value10 - value20;

  float* output = (float*)((uchar*)dst + dst_pitch + offset);
  output[element_x] = output0;
}

RetCode subtract(const float* src1, const float* src2, float* dst, int rows,
                 int cols, int stride, int src1_pitch, int src2_pitch,
                 int dst_pitch, int channels, cudaStream_t stream) {
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(columns, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  subtractKernel<<<grid, block, 0, stream>>>(src1, src2, dst, rows, columns,
                                              stride, src1_pitch, src2_pitch,
                                              dst_pitch);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** boxfilter() ******************************/

template<int BSY,int BSX,int nc>
__global__
void boxFilterKernel21(int height , int width, int inWidthStride, int in_pitch,
                       const float* in_buf, int radiusX, int radiusY,
                       bool normal, int outWidthStride, int out_pitch,
                       float* out_buf) {
  float* src = (float*)((uchar*)in_buf + in_pitch);
  float *dst = (float*)((uchar*)out_buf + out_pitch);
  float sum;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int j = bx * blockDim.x + tx;
  int i = by * blockDim.y + ty;
  float weight = 1.f/((2*radiusY+1)*(2*radiusX+1));
  extern __shared__ float pixels[];
  int stride = BSX + 2 * radiusX;
  if (bx>0 && by >0 && by<gridDim.y-2 && bx<gridDim.x-2) {
      int Idx = i* inWidthStride + j * nc ;
      int idxDst = i* outWidthStride + j * nc;
      //center
      for(int k=0;k<nc;k++) {
        pixels[((ty+radiusY)*stride+(tx+radiusX))*nc+k] = src[Idx+k];
      }

      //left & right
      if(tx<radiusX){
          int idx2 = i*inWidthStride + (j - radiusX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY)*stride+tx)*nc+k] = src[idx2+k];
          }

          idx2 = i*inWidthStride + (j + BSX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY)*stride+(tx+BSX+radiusX))*nc+k] = src[idx2+k];
          }
      }

      //upper & down
      if(ty<radiusY){
          int idx2 = (i-radiusY)*inWidthStride + j * nc;
          for(int k=0;k<nc;k++){
              pixels[(ty*stride+(radiusX+tx))*nc+k] = src[idx2+k];
          }

          idx2 = (i+BSY)*inWidthStride + j * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+BSY+radiusY)*stride+(radiusX+tx))*nc+k] = src[idx2+k];
          }
      }
      //four corners
      if(ty<radiusY && tx<radiusX){
          int idx2 = (i-radiusY)*inWidthStride + (j-radiusX) * nc;
          for(int k=0;k<nc;k++){
              pixels[(ty*stride+(tx))*nc+k] = src[idx2+k];
          }

          idx2 = (i-radiusY)*inWidthStride + (j+BSX) * nc;
          for(int k=0;k<nc;k++){
              pixels[(ty*stride+(tx+BSX+radiusX))*nc+k] = src[idx2+k];
          }

          idx2 = (i+BSY)*inWidthStride + (j-radiusX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY+BSY)*stride+tx)*nc+k] = src[idx2+k];
          }

          idx2 = (i+BSY)*inWidthStride + (j+BSX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY+BSY)*stride+(tx+radiusX+BSX))*nc+k] = src[idx2+k];
          }

      }
      __syncthreads();

      for(int k=0;k<nc;k++){
          sum = 0;
          for(int l=0;l<(2*radiusY+1);l++)
              for(int m=0;m<(2*radiusX+1);m++){
              int ki = l-radiusY;
              int kj = m-radiusX;
              int x,y;
              y = ki + ty + radiusY;
              x = kj + tx + radiusX;
              sum += pixels[(y*stride+x)*nc+k];
          }
          if(normal)
              dst[idxDst+k] = weight * sum;
          else dst[idxDst+k] = sum;
      }//k
  }
  else {
    //处理边界
    int flag = i<height && j<width;
    bool flag1 = (by==0 && flag);//上边
    bool flag2 = (bx==0 && by>0 && flag );//左侧
    bool flag3 = (bx>=gridDim.x-2 && by>0 && flag); //右侧
    bool flag4 = (by>=gridDim.y-2 && bx>0 && flag);//下侧
    int index;
    if(flag1 || flag2 ||flag3 ||flag4) {
      int idxDst = i* outWidthStride + j*nc;
      for(int k=0; k<nc; k++) {
        sum = 0;
        for(int l=0;l<(2*radiusY+1);l++) {
          for(int m=0;m<(2*radiusX+1);m++) {
            int ki = l-radiusY;
            int kj = m-radiusX;
            int x,y;
            index = ki + i;
            if(index<0) {
                y = -index;
            }else if(index > height -1) {
                y = (height -1)-(index-(height-1));
            }else {
                y = ki + i;
            }
            index = kj + j;
            if (index < 0) {
                x = -index;
            } else if(index > width -1) {
                x = (width -1) - (index-(width-1));
            } else {
                x = kj + j;
            }
            sum += src[y*inWidthStride+ x * nc + k];
          }// l&m
        }

        if (normal) {
            dst[idxDst+k] = weight * sum;
        }
        else {
            dst[idxDst+k] = sum;
        }
      }//k
    }//if
  }
}

template<int BSY,int BSX,int nc>
__global__
void boxFilterKernel22(int height , int width, int inWidthStride, int in_pitch,
                       const float* in_buf, int radiusX, int radiusY,
                       bool normal, int outWidthStride, int out_pitch,
                       float * out_buf) {
  float* src = (float*)((uchar*)in_buf + in_pitch);
  float *dst = (float*)((uchar*)out_buf + out_pitch);
  float sum;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int j = bx * blockDim.x + tx;
  int i = by * blockDim.y + ty;
  float weight = 1.f/((2*radiusY+1)*(2*radiusX+1));
  extern __shared__ float pixels[];
  int stride = BSX + 2 * radiusX;
  if(bx>0 && by >0 && by<gridDim.y-2 && bx<gridDim.x-2) {
      int Idx = i* inWidthStride + j * nc ;
      int idxDst = i* outWidthStride + j * nc;
      //center
      for(int k=0;k<nc;k++) {
        pixels[((ty+radiusY)*stride+(tx+radiusX))*nc+k] = src[Idx+k];
      }

      //left & right
      if(tx<radiusX){
          int idx2 = i*inWidthStride + (j - radiusX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY)*stride+tx)*nc+k] = src[idx2+k];
          }

          idx2 = i*inWidthStride + (j + BSX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY)*stride+(tx+BSX+radiusX))*nc+k] = src[idx2+k];
          }
      }

      //upper & down
      if(ty<radiusY){
          int idx2 = (i-radiusY)*inWidthStride + j * nc;
          for(int k=0;k<nc;k++){
              pixels[(ty*stride+(radiusX+tx))*nc+k] = src[idx2+k];
          }

          idx2 = (i+BSY)*inWidthStride + j * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+BSY+radiusY)*stride+(radiusX+tx))*nc+k] = src[idx2+k];
          }
      }
      //four corners
      if(ty<radiusY && tx<radiusX){
          int idx2 = (i-radiusY)*inWidthStride + (j-radiusX) * nc;
          for(int k=0;k<nc;k++){
              pixels[(ty*stride+(tx))*nc+k] = src[idx2+k];
          }

          idx2 = (i-radiusY)*inWidthStride + (j+BSX) * nc;
          for(int k=0;k<nc;k++){
              pixels[(ty*stride+(tx+BSX+radiusX))*nc+k] = src[idx2+k];
          }

          idx2 = (i+BSY)*inWidthStride + (j-radiusX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY+BSY)*stride+tx)*nc+k] = src[idx2+k];
          }

          idx2 = (i+BSY)*inWidthStride + (j+BSX) * nc;
          for(int k=0;k<nc;k++){
              pixels[((ty+radiusY+BSY)*stride+(tx+radiusX+BSX))*nc+k] = src[idx2+k];
          }

      }
      __syncthreads();

      for(int k=0;k<nc;k++){
          sum = 0;
          for(int l=0;l<(2*radiusY+1);l++)
              for(int m=0;m<(2*radiusX+1);m++){
              int ki = l-radiusY;
              int kj = m-radiusX;
              int x,y;
              y = ki + ty + radiusY;
              x = kj + tx + radiusX;
              sum += pixels[(y*stride+x)*nc+k];
          }
          if(normal)
              dst[idxDst+k] = weight * sum;
          else dst[idxDst+k] = sum;
      }//k
  }
  else {
    //处理边界
    int flag = i<height && j<width;
    bool flag1 = (by==0 && flag);//上边
    bool flag2 = (bx==0 && by>0 && flag );//左侧
    bool flag3 = (bx>=gridDim.x-2 && by>0 && flag); //右侧
    bool flag4 = (by>=gridDim.y-2 && bx>0 && flag);//下侧
    int index;
    if(flag1 || flag2 ||flag3 ||flag4) {
      int idxDst = i* outWidthStride + j*nc;
      for(int k=0; k<nc; k++) {
        sum = 0;
        for(int l=0;l<(2*radiusY+1);l++) {
          for(int m=0;m<(2*radiusX+1);m++) {
            int ki = l-radiusY;
            int kj = m-radiusX;
            int x,y;
            index = ki + i;
            if(index<0) {
                y = -index - 1;
            }else if(index > height -1) {
                y = height-(index-(height-1));
            }else {
                y = ki + i;
            }
            index = kj + j;
            if (index < 0) {
                x = -index - 1;
            } else if(index > width -1) {
                x = width - (index-(width-1));
            } else {
                x = kj + j;
            }
            sum += src[y*inWidthStride+ x * nc + k];
          }// l&m
        }

        if (normal) {
            dst[idxDst+k] = weight * sum;
        }
        else {
            dst[idxDst+k] = sum;
        }
      }//k
    }//if
  }
}

void BoxFilter2(cudaStream_t stream,
                int height,
                int width,
                int inWidthStride,
                int in_pitch,
                const float* inData,
                int kernelx_len,
                int kernely_len,
                bool normalize,
                int outWidthStride,
                int out_pitch,
                float* outData,
                BorderType border_type) {
  const int BX = 32;
  const int BY = 32;
  const int radiusX = kernelx_len>>1;
  const int radiusY = kernely_len>>1;
  dim3 block(BX,BY);
  dim3 grid;
  grid.x = (width + BX-1)/BX;
  grid.y = (height + BY-1)/BY;
  int smem = (BY + 2 * radiusY) * (BX + 2 * radiusX) * 1 * sizeof(float);

  if (border_type == ppl::cv::BORDER_TYPE_REFLECT_101) {
    boxFilterKernel21<BY, BX, 1><<<grid, block, smem, stream>>>
      (height, width, inWidthStride, in_pitch, inData, radiusX, radiusY, normalize,
       outWidthStride, out_pitch, outData);
  } else if(border_type == ppl::cv::BORDER_TYPE_REFLECT) {
    boxFilterKernel22<BY, BX, 1><<<grid, block, smem, stream>>>
      (height, width, inWidthStride, in_pitch, inData, radiusX, radiusY, normalize,
       outWidthStride, out_pitch, outData);
  }
}

/***************************** SplitXChannels() ******************************/

__global__
void split3ChannelsKernel(const float* src, float* dst, int dst0_array_pitch,
                          int dst1_array_pitch, int dst2_array_pitch,
                          int rows, int cols, int src_stride, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int input_x = element_x * 3;
  float* input = (float*)((uchar*)src + element_y * src_stride);
  float value0, value1, value2;
  value0 = input[input_x];
  value1 = input[input_x + 1];
  value2 = input[input_x + 2];

  int offset = element_y * dst_stride;
  float* output0 = (float*)((uchar*)dst + dst0_array_pitch + offset);
  float* output1 = (float*)((uchar*)dst + dst1_array_pitch + offset);
  float* output2 = (float*)((uchar*)dst + dst2_array_pitch + offset);
  output0[element_x] = value0;
  output1[element_x] = value1;
  output2[element_x] = value2;
}

RetCode split3Channels(const float* src, float* dst, int dst0_array_pitch,
                    int dst1_array_pitch, int dst2_array_pitch,
                    int rows, int cols, int src_stride, int dst_stride,
                    cudaStream_t stream) {
  PPL_ASSERT(src  != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 1 && cols > 1);
  PPL_ASSERT(src_stride >= cols * 3 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split3ChannelsKernel<<<grid, block, 0, stream>>>(src, dst, dst0_array_pitch,
      dst1_array_pitch, dst2_array_pitch, rows, cols, src_stride, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

__global__
void split4ChannelsKernel(const float* src, float* dst, int dst0_array_pitch,
                          int dst1_array_pitch, int dst2_array_pitch,
                          int dst3_array_pitch, int rows, int cols,
                          int src_stride, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int input_x = element_x << 2;
  float* input = (float*)((uchar*)src + element_y * src_stride);
  float value0, value1, value2, value3;
  value0 = input[input_x];
  value1 = input[input_x + 1];
  value2 = input[input_x + 2];
  value3 = input[input_x + 3];

  int offset = element_y * dst_stride;
  float* output0 = (float*)((uchar*)dst + dst0_array_pitch + offset);
  float* output1 = (float*)((uchar*)dst + dst1_array_pitch + offset);
  float* output2 = (float*)((uchar*)dst + dst2_array_pitch + offset);
  float* output3 = (float*)((uchar*)dst + dst3_array_pitch + offset);
  output0[element_x] = value0;
  output1[element_x] = value1;
  output2[element_x] = value2;
  output3[element_x] = value3;
}

RetCode split4Channels(const float* src, float* dst, int dst0_array_pitch,
                       int dst1_array_pitch, int dst2_array_pitch,
                       int dst3_array_pitch, int rows, int cols, int src_stride,
                       int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src  != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 1 && cols > 1);
  PPL_ASSERT(src_stride >= cols * 4 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split4ChannelsKernel<<<grid, block, 0, stream>>>(src, dst, dst0_array_pitch,
    dst1_array_pitch, dst2_array_pitch, dst3_array_pitch, rows, cols,
    src_stride, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** MergeXChannels() ******************************/

__global__
void merge3ChannelsKernel(const float* src, int src0_array_pitch,
                          int src1_array_pitch, int src2_array_pitch, int rows,
                          int cols, int src_stride, float* dst,
                          int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src_stride;
  float* input0 = (float*)((uchar*)src + src0_array_pitch + offset);
  float* input1 = (float*)((uchar*)src + src1_array_pitch + offset);
  float* input2 = (float*)((uchar*)src + src2_array_pitch + offset);
  float value0  = input0[element_x];
  float value1  = input1[element_x];
  float value2  = input2[element_x];

  element_x = element_x * 3;
  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
}

RetCode merge3Channels(const float* src, int src0_array_pitch,
                       int src1_array_pitch, int src2_array_pitch,
                       int rows, int cols, int src_stride, float* dst,
                       int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 1 && cols > 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 3 * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge3ChannelsKernel<<<grid, block, 0, stream>>>(src, src0_array_pitch,
    src1_array_pitch, src2_array_pitch, rows, cols, src_stride, dst,
    dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

__global__
void merge4ChannelsKernel(const float* src, int src0_array_pitch,
                          int src1_array_pitch, int src2_array_pitch,
                          int src3_array_pitch, int rows, int cols,
                          int src_stride, float* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src_stride;
  float* input0 = (float*)((uchar*)src + src0_array_pitch + offset);
  float* input1 = (float*)((uchar*)src + src1_array_pitch + offset);
  float* input2 = (float*)((uchar*)src + src2_array_pitch + offset);
  float* input3 = (float*)((uchar*)src + src3_array_pitch + offset);

  float value0  = input0[element_x];
  float value1  = input1[element_x];
  float value2  = input2[element_x];
  float value3  = input3[element_x];

  element_x = element_x << 2;
  dst_stride >>= 2;
  float* output = dst + element_y * dst_stride;
  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
  output[element_x + 3] = value3;
}

RetCode merge4Channels(const float* src, int src0_array_pitch,
                       int src1_array_pitch, int src2_array_pitch,
                       int src3_array_pitch, int rows, int cols, int src_stride,
                       float* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst  != nullptr);
  PPL_ASSERT(rows > 1 && cols > 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 4 * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge4ChannelsKernel<<<grid, block, 0, stream>>>(src, src0_array_pitch,
    src1_array_pitch, src2_array_pitch, src3_array_pitch, rows, cols,
    src_stride, dst, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** guidedFilter() ******************************/

/*
 * guide image: 1 channel.
 * input image: 1 channel.
 * output image: 1 channel.
 */
void guidedFilter_1to1(const float* guide, int guide_stride, const float* src,
                       int src_rows, int src_cols, int src_stride, float* dst,
                       int dst_stride, int radius, double eps,
                       BorderType border_type) {
  float* buffer;
  size_t pitch;
  cudaMallocPitch(&buffer, &pitch, src_cols * sizeof(float), src_rows * 8);

  int array_pitch = pitch * src_rows;
  float* II = buffer;
  float* IP = buffer;
  float* meanI = buffer;
  float* meanP = buffer;
  float* meanII = buffer;
  float* meanIP = buffer;
  float* varI = buffer;
  float* covIP = buffer;
  int IP_array_pitch     = array_pitch;
  int meanI_array_pitch  = array_pitch * 2;
  int meanP_array_pitch  = array_pitch * 3;
  int meanII_array_pitch = array_pitch * 4;
  int meanIP_array_pitch = array_pitch * 5;
  int varI_array_pitch   = array_pitch * 6;
  int covIP_array_pitch  = array_pitch * 7;

  multiply(0, src_rows, src_cols, 1, guide_stride, 0, guide, guide_stride, 0,
           guide, pitch, 0, II, 1.0);
  multiply(0, src_rows, src_cols, 1, guide_stride, 0, guide, src_stride, 0, src,
           pitch, IP_array_pitch, IP, 1.0);

  int side_length = (radius << 1) + 1;
  int src_width = src_stride / sizeof(float);
  int buffer_width = pitch / sizeof(float);
  BoxFilter2(0, src_rows, src_cols, src_width, 0, guide, side_length,
             side_length, true, buffer_width, meanI_array_pitch, meanI,
             border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, 0, src, side_length, side_length,
             true, buffer_width, meanP_array_pitch, meanP, border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, 0, II, side_length, side_length,
             true, buffer_width, meanII_array_pitch, meanII, border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, IP_array_pitch, IP, side_length,
             side_length, true, buffer_width, meanIP_array_pitch, meanIP,
             border_type);

  float* meanII_mul = II;
  float* meanIP_mul = IP;
  multiply(0, src_rows, src_cols, 1, pitch, meanI_array_pitch, meanI, pitch,
           meanI_array_pitch, meanI, pitch, 0, meanII_mul, 1.0);
  multiply(0, src_rows, src_cols, 1, pitch, meanI_array_pitch, meanI, pitch,
           meanP_array_pitch, meanP, pitch, IP_array_pitch, meanIP_mul, 1.0);
  subtract(meanII, meanII_mul, varI, src_rows, src_cols, pitch,
           meanII_array_pitch, 0, varI_array_pitch, 1, 0);
  subtract(meanIP, meanIP_mul, covIP, src_rows, src_cols, pitch,
           meanIP_array_pitch, IP_array_pitch, covIP_array_pitch, 1, 0);

  float* a = meanII;
  float* b = meanIP;
  float* aMeanI = covIP;
  add_scalar(varI, src_rows, src_cols, pitch, varI_array_pitch, 1, eps, 0);
  divide(0, src_rows, src_cols, 1, pitch, covIP_array_pitch, covIP, pitch,
         varI_array_pitch, varI, pitch, meanII_array_pitch, a, 1);
  multiply(0, src_rows, src_cols, 1, pitch, meanII_array_pitch, a, pitch,
           meanI_array_pitch, meanI, pitch, covIP_array_pitch, aMeanI, 1.0);
  subtract(meanP, aMeanI, b, src_rows, src_cols, pitch, meanP_array_pitch,
           covIP_array_pitch, meanIP_array_pitch, 1, 0);

  float* meanA = II;
  float* meanB = IP;
  BoxFilter2(0, src_rows, src_cols, src_width, meanII_array_pitch, a,
             side_length, side_length, true, buffer_width, 0, meanA,
             border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, meanIP_array_pitch, b,
             side_length, side_length, true, buffer_width, IP_array_pitch,
             meanB, border_type);

  float* meanAI = meanI;
  multiply(0, src_rows, src_cols, 1, pitch, 0, meanA, guide_stride, 0, guide,
           pitch, meanI_array_pitch, meanAI, 1.0);
  add(meanAI, meanB, dst, src_rows, src_cols, dst_stride, meanI_array_pitch,
      IP_array_pitch, 0, 1, 0);

  cudaFree(buffer);
}

void guidedFilter_1to1(const float* guide, int guide_stride, const float* src,
                       int src_rows, int src_cols, int src_stride,
                       int src_array_pitch, float* dst, int dst_stride,
                       int dst_array_pitch, int radius, double eps,
                       BorderType border_type) {
  float* buffer;
  size_t pitch;
  cudaMallocPitch(&buffer, &pitch, src_cols * sizeof(float), src_rows * 8);

  int array_pitch = pitch * src_rows;
  float* II = buffer;
  float* IP = buffer;
  float* meanI = buffer;
  float* meanP = buffer;
  float* meanII = buffer;
  float* meanIP = buffer;
  float* varI = buffer;
  float* covIP = buffer;
  int IP_array_pitch     = array_pitch;
  int meanI_array_pitch  = array_pitch * 2;
  int meanP_array_pitch  = array_pitch * 3;
  int meanII_array_pitch = array_pitch * 4;
  int meanIP_array_pitch = array_pitch * 5;
  int varI_array_pitch   = array_pitch * 6;
  int covIP_array_pitch  = array_pitch * 7;

  multiply(0, src_rows, src_cols, 1, guide_stride, 0, guide, guide_stride, 0,
           guide, pitch, 0, II, 1.0);
  multiply(0, src_rows, src_cols, 1, guide_stride, 0, guide, src_stride,
           src_array_pitch, src,
           pitch, IP_array_pitch, IP, 1.0);

  int side_length = (radius << 1) + 1;
  int src_width = src_stride / sizeof(float);
  int buffer_width = pitch / sizeof(float);
  BoxFilter2(0, src_rows, src_cols, src_width, 0, guide, side_length,
             side_length, true, buffer_width, meanI_array_pitch, meanI,
             border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, src_array_pitch, src, side_length,
             side_length, true, buffer_width, meanP_array_pitch, meanP,
             border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, 0, II, side_length, side_length,
             true, buffer_width, meanII_array_pitch, meanII, border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, IP_array_pitch, IP, side_length,
             side_length, true, buffer_width, meanIP_array_pitch, meanIP,
             border_type);

  float* meanII_mul = II;
  float* meanIP_mul = IP;
  multiply(0, src_rows, src_cols, 1, pitch, meanI_array_pitch, meanI, pitch,
           meanI_array_pitch, meanI, pitch, 0, meanII_mul, 1.0);
  multiply(0, src_rows, src_cols, 1, pitch, meanI_array_pitch, meanI, pitch,
           meanP_array_pitch, meanP, pitch, IP_array_pitch, meanIP_mul, 1.0);
  subtract(meanII, meanII_mul, varI, src_rows, src_cols, pitch,
           meanII_array_pitch, 0, varI_array_pitch, 1, 0);
  subtract(meanIP, meanIP_mul, covIP, src_rows, src_cols, pitch,
           meanIP_array_pitch, IP_array_pitch, covIP_array_pitch, 1, 0);

  float* a = meanII;
  float* b = meanIP;
  float* aMeanI = covIP;
  add_scalar(varI, src_rows, src_cols, pitch, varI_array_pitch, 1, eps, 0);
  divide(0, src_rows, src_cols, 1, pitch, covIP_array_pitch, covIP, pitch,
         varI_array_pitch, varI, pitch, meanII_array_pitch, a, 1);
  multiply(0, src_rows, src_cols, 1, pitch, meanII_array_pitch, a, pitch,
           meanI_array_pitch, meanI, pitch, covIP_array_pitch, aMeanI, 1.0);
  subtract(meanP, aMeanI, b, src_rows, src_cols, pitch, meanP_array_pitch,
           covIP_array_pitch, meanIP_array_pitch, 1, 0);

  float* meanA = II;
  float* meanB = IP;
  BoxFilter2(0, src_rows, src_cols, src_width, meanII_array_pitch, a,
             side_length, side_length, true, buffer_width, 0, meanA,
             border_type);
  BoxFilter2(0, src_rows, src_cols, src_width, meanIP_array_pitch, b,
             side_length, side_length, true, buffer_width, IP_array_pitch,
             meanB, border_type);

  float* meanAI = meanI;
  multiply(0, src_rows, src_cols, 1, pitch, 0, meanA, guide_stride, 0, guide,
           pitch, meanI_array_pitch, meanAI, 1.0);
  add(meanAI, meanB, dst, src_rows, src_cols, dst_stride, meanI_array_pitch,
      IP_array_pitch, dst_array_pitch, 1, 0);

  cudaFree(buffer);
}

void filtering(const float* guide, int guide_stride, int guide_channels,
               const float* src, int src_stride, int src_channels,
               float* dst, int dst_stride, int rows, int cols, int radius,
               float eps, BorderType border_type, cudaStream_t stream) {
  if (guide_channels == 1) {
    if (src_channels == 1) {
      guidedFilter_1to1(guide, guide_stride, src, rows, cols, src_stride,
                        dst, dst_stride, radius, eps, border_type);
    }
    else if (src_channels == 3) { // src_channels == 3
      float* buffer;
      size_t pitch;
      cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 6);

      int array_pitch = pitch * rows;
      float* src0 = buffer;
      float* src1 = buffer;
      float* src2 = buffer;
      float* dst0 = buffer;
      float* dst1 = buffer;
      float* dst2 = buffer;
      size_t src0_array_pitch = 0;
      size_t src1_array_pitch = array_pitch;
      size_t src2_array_pitch = array_pitch * 2;
      size_t dst0_array_pitch = array_pitch * 3;
      size_t dst1_array_pitch = array_pitch * 4;
      size_t dst2_array_pitch = array_pitch * 5;

      split3Channels(src, buffer, 0, src1_array_pitch, src2_array_pitch,
                     rows, cols, src_stride, pitch, stream);
      guidedFilter_1to1(guide, guide_stride, src0, rows, cols,
                        pitch, src0_array_pitch, dst0, pitch,
                        dst0_array_pitch, radius, eps, border_type);
      guidedFilter_1to1(guide, guide_stride, src1, rows, cols,
                        pitch, src1_array_pitch, dst1, pitch,
                        dst1_array_pitch, radius, eps, border_type);
      guidedFilter_1to1(guide, guide_stride, src2, rows, cols,
                        pitch, src2_array_pitch, dst2, pitch,
                        dst2_array_pitch, radius, eps, border_type);
      merge3Channels(buffer, dst0_array_pitch, dst1_array_pitch,
                     dst2_array_pitch, rows, cols, pitch, dst,
                     dst_stride, stream);

      cudaFree(buffer);
    }
    else { // src_channels == 4
      float* buffer;
      size_t pitch;
      cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 8);

      int array_pitch = pitch * rows;
      float* src0 = buffer;
      float* src1 = buffer;
      float* src2 = buffer;
      float* src3 = buffer;
      float* dst0 = buffer;
      float* dst1 = buffer;
      float* dst2 = buffer;
      float* dst3 = buffer;
      size_t src0_array_pitch = 0;
      size_t src1_array_pitch = array_pitch;
      size_t src2_array_pitch = array_pitch * 2;
      size_t src3_array_pitch = array_pitch * 3;
      size_t dst0_array_pitch = array_pitch * 4;
      size_t dst1_array_pitch = array_pitch * 5;
      size_t dst2_array_pitch = array_pitch * 6;
      size_t dst3_array_pitch = array_pitch * 7;

      split4Channels(src, buffer, 0, src1_array_pitch, src2_array_pitch,
                     src3_array_pitch, rows, cols, src_stride, pitch,
                     stream);
      guidedFilter_1to1(guide, guide_stride, src0, rows, cols,
                        pitch, src0_array_pitch, dst0, pitch,
                        dst0_array_pitch, radius, eps, border_type);
      guidedFilter_1to1(guide, guide_stride, src1, rows, cols,
                        pitch, src1_array_pitch, dst1, pitch,
                        dst1_array_pitch, radius, eps, border_type);
      guidedFilter_1to1(guide, guide_stride, src2, rows, cols,
                        pitch, src2_array_pitch, dst2, pitch,
                        dst2_array_pitch, radius, eps, border_type);
      guidedFilter_1to1(guide, guide_stride, src3, rows, cols,
                        pitch, src3_array_pitch, dst3, pitch,
                        dst3_array_pitch, radius, eps, border_type);
      merge4Channels(buffer, dst0_array_pitch, dst1_array_pitch,
                     dst2_array_pitch, dst3_array_pitch, rows, cols,
                     pitch, dst, dst_stride, stream);

      cudaFree(buffer);
    }
  }
  else {  // guide_channels == 3
    if (src_channels == 1) {
    }
    else if (src_channels == 3) { // src_channels == 3
    }
    else { // src_channels == 4
    }
  }
}

RetCode guidedFilter(const uchar* guide, int guide_stride, int guide_channels,
                     const uchar* src, int src_stride, int src_channels,
                     uchar* dst, int dst_stride, int rows, int cols, int radius,
                     float eps, BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(guide != nullptr);
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 0 && cols > 0);
  PPL_ASSERT(guide_stride > 0 && src_stride > 0 && dst_stride > 0);
  PPL_ASSERT(guide_channels == 1 || guide_channels == 3);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_TYPE_REFLECT_101 ||
         border_type == BORDER_TYPE_REFLECT);

  float* fguide;
  float* fsrc;
  float* fdst;
  size_t fguide_stride, fsrc_stride, fdst_stride;
  cudaMallocPitch(&fguide, &fguide_stride,
                  cols * guide_channels * sizeof(float), rows);
  cudaMallocPitch(&fsrc, &fsrc_stride, cols * src_channels * sizeof(float),
                  rows);
  cudaMallocPitch(&fdst, &fdst_stride, cols * src_channels * sizeof(float),
                  rows);

  convertTo(guide, rows, cols, guide_channels, guide_stride, fguide,
            fguide_stride, 1, 0.0, stream);
  convertTo(src, rows, cols, src_channels, src_stride, fsrc, fsrc_stride,
            1, 0.0, stream);

  filtering(fguide, fguide_stride, guide_channels, fsrc, fsrc_stride,
            src_channels, fdst, fdst_stride, rows, cols, radius, eps,
            border_type, stream);

  convertTo(fdst, rows, cols, src_channels, fdst_stride, dst, dst_stride,
            1, 0.0, stream);

  cudaFree(fguide);
  cudaFree(fsrc);
  cudaFree(fdst);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode guidedFilter(const float* guide, int guide_stride, int guide_channels,
                     const float* src, int src_stride, int src_channels,
                     float* dst, int dst_stride, int rows, int cols, int radius,
                     float eps, BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(guide != nullptr);
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows > 0 && cols > 0);
  PPL_ASSERT(guide_stride > 0 && src_stride > 0 && dst_stride > 0);
  PPL_ASSERT(guide_channels == 1 || guide_channels == 3);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_TYPE_REFLECT_101 ||
         border_type == BORDER_TYPE_REFLECT);

  filtering(guide, guide_stride, guide_channels, src, src_stride, src_channels,
            dst, dst_stride, rows, cols, radius, eps, border_type, stream);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode GuidedFilter<uchar, 1, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(guideData, guideWidthStride, 1, inData,
                              inWidthStride, 1, outData, outWidthStride, height,
                              width, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 3, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(guideData, guideWidthStride, 1, inData,
                              inWidthStride, 3, outData, outWidthStride, height,
                              width, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 4, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(guideData, guideWidthStride, 1, inData,
                              inWidthStride, 4, outData, outWidthStride, height,
                              width, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 1, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(guideData, guideWidthStride, 1, inData,
                              inWidthStride, 1, outData, outWidthStride, height,
                              width, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 3, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(guideData, guideWidthStride, 1, inData,
                              inWidthStride, 3, outData, outWidthStride, height,
                              width, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 4, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(guideData, guideWidthStride, 1, inData,
                              inWidthStride, 4, outData, outWidthStride, height,
                              width, radius, eps, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
