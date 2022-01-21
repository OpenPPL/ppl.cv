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

#ifndef _ST_HPC_PPL_CV_CUDA_CALCHIST_H_
#define _ST_HPC_PPL_CV_CUDA_CALCHIST_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Calculates a histogram of an image.
 * @tparam T The data type of input and output image, currently only
 *         uint8_t(uchar) is supported.
 * @param stream          cuda stream object.
 * @param height          input image's height.
 * @param width           input image's width.
 * @param inWidthStride   input image's width stride, it is `width` for
 *                        cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                        for 2D cudaMallocPitch() allocated data.
 * @param inData          input image data.
 * @param outHist         output histogram data.
 * @param maskWidthStride the width stride of mask, similar to inWidthStride.
 * @param mask            optional operation mask; it must have the same size as
 *                        inData, and is uchar and single channel type.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 The size of the output histogram is 256.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/calchist.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/calchist.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int channels = 1;
 *
 *   uchar* gpu_input;
 *   int* gpu_output;
 *   size_t input_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(uchar), height);
 *   cudaMalloc(&gpu_output, 256 * sizeof(int));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   CalcHist<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, gpu_output);
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
template <typename T>
ppl::common::RetCode CalcHist(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int* outHist,
                              int maskWidthStride = 0,
                              const unsigned char* mask = nullptr);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_CALCHIST_H_
