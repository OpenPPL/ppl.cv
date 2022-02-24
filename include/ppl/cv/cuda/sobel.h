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

#ifndef _ST_HPC_PPL_CV_CUDA_SOBEL_H_
#define _ST_HPC_PPL_CV_CUDA_SOBEL_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Calculates the first, second and third image derivatives using an
 *        extended Sobel operator.
 * @tparam Tsrc The data type of input image, currently only uint8_t(uchar) and
 *         float are supported.
 * @tparam Tdst The data type of output image, currently uint8_t(uchar), short
 *         and float are supported.
 * @tparam channels The number of channels of input&output image, 1, 3 and 4
 *         are supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @param dx             order of the derivative x.
 * @param dy             order of the derivative y.
 * @param ksize          the length of laplace kernel in X&Y direction.
 *                       -1(Scharr), 1, 3, 5and 7 are supported.
 * @param scale          optional scale factor for the computed derivative
 *                       values; by default, no scaling is applied.
 * @param delta          optional value added to the filtered pixels.
 * @param border_type    ways to deal with border. BORDER_REPLICATE,
 *                       BORDER_REFLECT, BORDER_REFLECT_101 and BORDER_DEFAULT 
 *                       are supported now.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 The output image has the same size and channels as the input image.
 *       3 When input is uchar, output should be 16 bits signed integer. There
 *         will be an overflow in the uchar input&output.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(Tsrc)<th>Data type(Tdst)<th>channels
 * <tr><td>uint8_t(uchar)<td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>uint8_t(uchar)<td>4
 * <tr><td>uint8_t(uchar)<td>short<td>1
 * <tr><td>uint8_t(uchar)<td>short<td>3
 * <tr><td>uint8_t(uchar)<td>short<td>4
 * <tr><td>float<td>float<td>1
 * <tr><td>float<td>float<td>3
 * <tr><td>float<td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/sobel.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/sobel.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *   int ksize = 3;
 *   int dx = 2;
 *   int dy = 0;
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
 *   Sobel<float, float, 3>(stream, height, width,
 *                          input_pitch / sizeof(float), gpu_input,
 *                          output_pitch / sizeof(float), gpu_output, dx, dy,
 *                          ksize, 1.f, 0.f, ppl::cv::BORDER_DEFAULT);
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
template <typename Tsrc, typename Tdst, int channels>
ppl::common::RetCode Sobel(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const Tsrc* inData,
                           int outWidthStride,
                           Tdst* outData,
                           int dx,
                           int dy,
                           int ksize = 3,
                           float scale = 1.f,
                           float delta = 0.f,
                           BorderType border_type = BORDER_DEFAULT);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_SOBEL_H_

