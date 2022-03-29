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

#ifndef _ST_HPC_PPL_CV_CUDA_DILATE_H_
#define _ST_HPC_PPL_CV_CUDA_DILATE_H_

#include <cfloat>

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Dilates an image by using a specific structuring element.
 * @tparam T The data type of input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input&output image, 1, 3 and 4
 *         are supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param kernelx_len    the length of mask, x direction.
 * @param kernely_len    the length of mask, y direction.
 * @param kernel         the mask used for dilation.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @param border_type    ways to deal with border. BORDER_CONSTANT,
 *                       BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP
 *                       and BORDER_REFLECT_101 are supported now.
 * @param border_value   value for BORDER_CONSTANT.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 The destination matrix has the same data type, size, stride, and
 *         channels as the source matrix.
 *       3 kernel must be a single channel matrix and stored in host memory as
 *         an uchar 1D array.
 *       4 The anchor is at the kernel center.
 *       5 When kernelx_len and kernely_len is bigger than 7, some
 *         implementations of this function need a memory buffer to store the
 *         intermediate result, which is not less than
 *         ppl::cv::cuda::ceil2DVolume(width * channels * sizeof(T),
 *         height). When CUDA Memory Pool is used, the capacity of CUDA Memory
 *         Pool must be not less than the size of the memory buffer.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/dilate.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/dilate.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Dilate<float, 3>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, 3, 3, nullptr, output_pitch / sizeof(float),
 *                    gpu_output, ppl::cv::BORDER_REPLICATE);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Dilate(cudaStream_t stream,
                            int height,
                            int width,
                            int inWidthStride,
                            const T* inData,
                            int kernelx_len,
                            int kernely_len,
                            const uchar* kernel,
                            int outWidthStride,
                            T* outData,
                            BorderType border_type = BORDER_CONSTANT,
                            const T border_value = -FLT_MAX);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_DILATE_H_
