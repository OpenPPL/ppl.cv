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

#ifndef _ST_HPC_PPL_CV_OCL_DILATE_H_
#define _ST_HPC_PPL_CV_OCL_DILATE_H_

#include <cfloat>

#include "CL/cl.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Dilates an image by using a specific structuring element.
 * @tparam T The data type of input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input&output image, 1, 3 and 4
 *         are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less than
 *                       `width * channels`.
 * @param inData         input image data.
 * @param kernelx_len    the length of mask, x direction.
 * @param kernely_len    the length of mask, y direction.
 * @param kernel         the mask used for dilation.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @param border_type    ways to deal with border. BORDER_CONSTANT,
 *                       BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP
 *                       and BORDER_REFLECT_101 are supported now.
 * @param border_value   value for BORDER_CONSTANT.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, rows of input&output aligned with 64 bits are
 *         recommended.
 *       2 The destination matrix has the same data type, size, stride, and
 *         channels as the source matrix.
 *       3 kernel must be a single channel matrix and stored in host memory as
 *         an uchar 1D array.
 *       4 The anchor is at the kernel center.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/dilate.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/dilate.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(float);
 *   float* input = (float*)malloc(data_size);
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   Dilate<float, 3>(queue, height, width, width * channels, gpu_input, 3, 3,
 *                    nullptr, width * channels, gpu_output,
 *                    ppl::cv::BORDER_REPLICATE);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Dilate(cl_command_queue queue,
                            int height,
                            int width,
                            int inWidthStride,
                            const cl_mem inData,
                            int kernelx_len,
                            int kernely_len,
                            const uchar* kernel,
                            int outWidthStride,
                            cl_mem outData,
                            BorderType border_type = BORDER_CONSTANT,
                            const T border_value = -FLT_MAX);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_DILATE_H_
