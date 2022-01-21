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

#ifndef _ST_HPC_PPL_CV_CUDA_BITWISE_H_
#define _ST_HPC_PPL_CV_CUDA_BITWISE_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief computes bitwise conjunction of the two matrices.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) is supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream          cuda stream object.
 * @param height          input&output image's height.
 * @param width           input&output image's width.
 * @param inWidthStride0  first input image's width stride, it is `width *
 *                        channels` for cudaMalloc() allocated data, `pitch /
 *                        sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData0         first input image data.
 * @param inWidthStride1  second input image's width stride, similar to
 *                        inWidthStride0.
 * @param inData1         second input image data.
 * @param outWidthStride  the width stride of output image, similar to
 *                        inWidthStride0.
 * @param outData         output image data.
 * @param maskWidthStride the width stride of mask, similar to inWidthStride0.
 * @param mask            optional operation mask, 8-bit single channel array,
 *                        that specifies elements of the output array to be
 *                        changed.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/bitwise.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/bitwise.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BitwiseAnd<uchar, 3>(stream, height, width,
 *                        input_pitch / sizeof(uchar), gpu_input,
 *                        input_pitch / sizeof(uchar), gpu_input,
 *                        output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode BitwiseAnd(cudaStream_t stream,
                                int height,
                                int width,
                                int inWidthStride0,
                                const T* inData0,
                                int inWidthStride1,
                                const T* inData1,
                                int outWidthStride,
                                T* outData,
                                int maskWidthStride = 0,
                                unsigned char* mask = nullptr);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_BITWISE_H_
