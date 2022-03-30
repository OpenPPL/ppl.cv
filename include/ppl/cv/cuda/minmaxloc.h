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

#ifndef _ST_HPC_PPL_CV_CUDA_MINMAXLOC_H_
#define _ST_HPC_PPL_CV_CUDA_MINMAXLOC_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Finds the global minimum and maximum element values and their
 *        positions in a 2D array.
 * @tparam T The data type of input array, currently only uint8_t(uchar) and
 *         float are supported.
 * @param stream           cuda stream object.
 * @param height           input array's height.
 * @param width            input array's width.
 * @param inWidthStride    input array's width stride, it is `width * channels`
 *                         for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                         for 2D cudaMallocPitch() allocated data.
 * @param inData           input array data.
 * @param minVal           pointer to the returned minimum value.
 * @param maxVal           pointer to the returned maximum value.
 * @param minIdxX          pointer to the returned minimum column index.
 * @param minIdxY          pointer to the returned minimum row index.
 * @param maxIdxX          pointer to the returned maximum column index.
 * @param maxIdxY          pointer to the returned maximum row index.
 * @param maskWidthStride  the width stride of mask array, similar to
 *                         inWidthStride.
 * @param mask             specified array region.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 inData and mask have the same height, width and channels.
 *       3 The function only works with single-channel arrays.
 *       4 This function needs a memory buffer to store the intermediate
 *         result, which is not less than 6144 bytes. When CUDA Memory Pool
 *         is used, the capacity of CUDA Memory Pool must be not less than
 *         the size of the memory buffer.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>float<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/minmaxloc.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/minmaxloc.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int channels = 1;
 *
 *   float* gpu_input;
 *   float* gpu_mask;
 *   size_t input_pitch, mask_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_mask, &mask_pitch,
 *                   width * channels * sizeof(uchar), height);
 *
 *   float min_value, max_value;
 *   int min_index_x, min_index_y, max_index_x, max_index_y;
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   MinMaxLoc<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, &min_value, &max_value, &min_index_x,
 *                    &min_index_y, &max_index_x, &max_index_y,
 *                    mask_pitch / sizeof(float), gpu_mask);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_mask);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode MinMaxLoc(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               T* minVal,
                               T* maxVal,
                               int* minIdxX,
                               int* minIdxY,
                               int* maxIdxX,
                               int* maxIdxY,
                               int maskWidthStride = 0,
                               const unsigned char* mask = nullptr);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_MINMAXLOC_H_
