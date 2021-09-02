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

#include "ppl/cv/cuda/sobel.h"

#include <vector>
// #include <iostream> // debug

// #include "sobel_coeffs.hpp"
#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define __UNIFIED__ __device__ __managed__
#define MAX_KSIZE 32

__UNIFIED__ static float kernel_xs[MAX_KSIZE];
__UNIFIED__ static float kernel_ys[MAX_KSIZE];

RetCode sepfilter2D(const uchar* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, uchar* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream);

RetCode sepfilter2D(const uchar* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, short* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream);

RetCode sepfilter2D(const float* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, float* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream);

void getScharrKernels(float* kernel_x, float* kernel_y, int dx, int dy,
                      float scale, bool normalize) {
  if (dx < 0 || dy < 0 || dx + dy != 1) {
    return;
  }

  int ksize = 3;
  for (int k = 0; k < 2; k++) {
    float* kernel = k == 0 ? kernel_x : kernel_y;
    int order = k == 0 ? dx : dy;

    if (order == 0) {
      kernel[0] = 3, kernel[1] = 10, kernel[2] = 3;
    }
    if (order == 1) {
      kernel[0] = -1, kernel[1] = 0, kernel[2] = 1;
    }

    double scale0 = !normalize || order == 1 ? 1. : 1. / 32;
    if (k == 1) {
      scale0 *= scale;
    }
    for (int i = 0; i < ksize; i++) {
      kernel[i] *= scale0;
    }
  }
}

void getSobelKernels(float* kernel_x, float* kernel_y, int dx, int dy,
                     int ksize, float scale, bool normalize) {
  if (ksize > 31 || (ksize & 1) == 0) {
    return;
  }

  if (ksize <= 0) {
    getScharrKernels(kernel_x, kernel_y, dx, dy, scale, normalize);
    return;
  }

  int i, j, ksize_x = ksize, ksize_y = ksize;
  if (ksize_x == 1 && dx > 0) {
    ksize_x = 3;
  }
  if (ksize_y == 1 && dy > 0) {
    ksize_y = 3;
  }

  for (int k = 0; k < 2; k++) {
    std::vector<int> kerI(std::max(ksize_x, ksize_y) + 1);
    float* kernel = k == 0 ? kernel_x : kernel_y;
    int order = k == 0 ? dx : dy;
    int size = k == 0 ? ksize_x : ksize_y;

    if (size <= order) return;

    if (size == 1) {
      // kerI[0] = 1;
      kerI[0] = 0, kerI[1] = 1, kerI[2] = 0;
    }
    else if (size == 3) {
      if (order == 0) {
        kerI[0] = 1, kerI[1] = 2, kerI[2] = 1;
      }
      else if (order == 1) {
        kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;
      }
      else {
        kerI[0] = 1, kerI[1] = -2, kerI[2] = 1;
      }
    }
    else {
      int old_value, new_value;
      kerI[0] = 1;
      for (i = 0; i < size; i++) {
        kerI[i + 1] = 0;
      }

      for (i = 0; i < size - order - 1; i++) {
        old_value = kerI[0];
        for (j = 1; j <= size; j++) {
          new_value = kerI[j] + kerI[j - 1];
          kerI[j - 1] = old_value;
          old_value = new_value;
        }
      }

      for (i = 0; i < order; i++) {
        old_value = -kerI[0];
        for (j = 1; j <= size; j++) {
          new_value = kerI[j - 1] - kerI[j];
          kerI[j - 1] = old_value;
          old_value = new_value;
        }
      }
    }

    double scale0 = !normalize ? 1. : 1. / (1 << (size - order - 1));
    if (k == 1) {
      scale0 *= scale;
    }
    for (i = 0; i < std::max(ksize_x, ksize_y); i++) {
      kernel[i] = kerI[i] * scale0;
    }
  }

  // for (i = 0; i < 5; i++) {
  //   std::cout << "kernels[" << i << "]: " << kernel_x[i] << ", " << kernel_y[i] << std::endl;
  // }
}

RetCode sobel(const uchar* src, int rows, int cols, int channels,
              int src_stride, int dx, int dy, int ksize, uchar* dst,
              int dst_stride, float scale, float delta, BorderType border_type,
              cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dx == 0 || dx == 1 || dx == 2 || dx == 3);
  PPL_ASSERT(dy == 0 || dy == 1 || dy == 2 || dy == 3);
  PPL_ASSERT(!(dx > 0 && dy > 0));
  PPL_ASSERT(ksize == -1 || ksize == 1 || ksize == 3 || ksize == 5 ||
             ksize == 7);
  PPL_ASSERT(!(ksize == -1 && (dx > 1 || dy > 1)));
  PPL_ASSERT(!(ksize == 1 && (dx > 2 || dy > 2)));
  PPL_ASSERT(!(ksize == 3 && (dx > 2 || dy > 2)));
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  getSobelKernels(kernel_xs, kernel_ys, dx, dy, ksize, scale, false);

  if (ksize == -1 || ksize == 1) {
    ksize = 3;
  }
  RetCode code = sepfilter2D(src, rows, cols, channels, src_stride, kernel_xs,
                             kernel_ys, ksize, dst, dst_stride, delta,
                             border_type, stream);

  return code;
}

RetCode sobel(const uchar* src, int rows, int cols, int channels,
              int src_stride, int dx, int dy, int ksize, short* dst,
              int dst_stride, float scale, float delta, BorderType border_type,
              cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(short));
  PPL_ASSERT(dx == 0 || dx == 1 || dx == 2 || dx == 3);
  PPL_ASSERT(dy == 0 || dy == 1 || dy == 2 || dy == 3);
  PPL_ASSERT(!(dx > 0 && dy > 0));
  PPL_ASSERT(ksize == -1 || ksize == 1 || ksize == 3 || ksize == 5 ||
             ksize == 7);
  PPL_ASSERT(!(ksize == -1 && (dx > 1 || dy > 1)));
  PPL_ASSERT(!(ksize == 1 && (dx > 2 || dy > 2)));
  PPL_ASSERT(!(ksize == 3 && (dx > 2 || dy > 2)));
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  getSobelKernels(kernel_xs, kernel_ys, dx, dy, ksize, scale, false);

  if (ksize == -1 || ksize == 1) {
    ksize = 3;
  }
  RetCode code = sepfilter2D(src, rows, cols, channels, src_stride, kernel_xs,
                             kernel_ys, ksize, dst, dst_stride, delta,
                             border_type, stream);

  return code;
}

RetCode sobel(const float* src, int rows, int cols, int channels,
              int src_stride, int dx, int dy, int ksize, float* dst,
              int dst_stride, float scale, float delta, BorderType border_type,
              cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dx == 0 || dx == 1 || dx == 2 || dx == 3);
  PPL_ASSERT(dy == 0 || dy == 1 || dy == 2 || dy == 3);
  PPL_ASSERT(!(dx > 0 && dy > 0));
  PPL_ASSERT(ksize == -1 || ksize == 1 || ksize == 3 || ksize == 5 ||
             ksize == 7);
  PPL_ASSERT(!(ksize == -1 && (dx > 1 || dy > 1)));
  PPL_ASSERT(!(ksize == 1 && (dx > 2 || dy > 2)));
  PPL_ASSERT(!(ksize == 3 && (dx > 2 || dy > 2)));
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  getSobelKernels(kernel_xs, kernel_ys, dx, dy, ksize, scale, false);

  if (ksize == -1 || ksize == 1) {
    ksize = 3;
  }
  RetCode code = sepfilter2D(src, rows, cols, channels, src_stride, kernel_xs,
                             kernel_ys, ksize, dst, dst_stride, delta,
                             border_type, stream);

  return code;
}

template <>
RetCode Sobel<uchar, uchar, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               uchar* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  RetCode code = sobel(inData, height, width, 1, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, uchar, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               uchar* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  RetCode code = sobel(inData, height, width, 3, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, uchar, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               uchar* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  RetCode code = sobel(inData, height, width, 4, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, short, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               short* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = sobel(inData, height, width, 1, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, short, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               short* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = sobel(inData, height, width, 3, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, short, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               short* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = sobel(inData, height, width, 4, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<float, float, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int outWidthStride,
                               float* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sobel(inData, height, width, 1, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<float, float, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int outWidthStride,
                               float* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sobel(inData, height, width, 3, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<float, float, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int outWidthStride,
                               float* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sobel(inData, height, width, 4, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
