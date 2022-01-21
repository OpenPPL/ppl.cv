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

#ifndef _ST_HPC_PPL_CV_CUDA_PERSPECTIVETRANSFORM_H_
#define _ST_HPC_PPL_CV_CUDA_PERSPECTIVETRANSFORM_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Performs the perspective matrix transformation of vectors.
 * @tparam T The data type of input and output image, currently only
 *         float is supported.
 * @tparam srcCns The number of channels of input image, 2 or 3 supported.
 * @tparam dstCns The number of channels of output image, 2 or 3 supported.
 * @param stream           cuda stream object.
 * @param height           input&output image's height.
 * @param width            input&output image's width.
 * @param inWidthStride    input image's width stride, it is `width * channels`
 *                         for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                         for 2D cudaMallocPitch() allocated data.
 * @param inData           input image data.
 * @param outWidthStride   the width stride of output image, similar to
 *                         inWidthStride.
 * @param outData          output image data.
 * @param transData        transformation matrix.
 * @return The execution status, succeeds or fails with an error code.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>srcCns<th>dstCns
 * <tr><td>float<td>2<td>2
 * <tr><td>float<td>2<td>3
 * <tr><td>float<td>3<td>2
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/perspectivetransform.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/perspectivetransform.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int src_width  = 320;
 *   int src_height = 240;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *   float trans_matrix[16];
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   PerspectiveTransform<float, 3, 3>(stream, src_height, src_width,
 *       input_pitch / sizeof(float), gpu_input, output_pitch / sizeof(float),
 *       gpu_output, trans_matrix);
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
template <typename T, int srcCns, int dstCns>
ppl::common::RetCode PerspectiveTransform(cudaStream_t stream,
                                          int height,
                                          int width,
                                          int inWidthStride,
                                          const T* inData,
                                          int outWidthStride,
                                          T* outData,
                                          const float* transData);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_PERSPECTIVETRANSFORM_H_
