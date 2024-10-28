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

#ifndef _ST_HPC_PPL_CV_OCL_INTEGRAL_H_
#define _ST_HPC_PPL_CV_OCL_INTEGRAL_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Calculates the integral of an image.
 * @tparam Tsrc The data type of input image, currently only uint8_t(uchar) and
 *         float are supported.
 * @tparam Tdst The data type of output image, currently only int and float are
 *         supported.
 * @tparam channels The number of channels of input image, only 1 is supported.
 * @param queue            opencl command queue.
 * @param inHeight         input image's height.
 * @param inWidth          input image's width.
 * @param inWidthStride    input image's width stride, which is not less than
 *                         `width * channels * sizeof(Tsrc)`.
 * @param inData           input image data, it should be a buffer object.
 * @param outHeight        output image's height.
 * @param outWidth         output image's width.
 * @param outWidthStride   input image's width stride, which is not less than
 *                         `width * channels * sizeof(Tdst)`.
 * @param outData          output image data, it should be a buffer object.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 There are 2 implementation, in version 1 the input&output have the
 *         same size; in version 2 outHeight = inHeight + 1 && outWidth =
 *         inWidth + 1. Version 2 is compatible with integral() in OpenCV 4.1.
 *         outHeight&outWidth decides which implementation will be run.
 *       2 For best performance, this function
 *         need a memory pool to store the intermediate result, which is not
 *         less than ppl::cv::ocl::ceil2DVolume(width * channels * sizeof(float),
 *         height).
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(Tsrc)<th>Data type(Tdst)<th>channels
 * <tr><td>uint8_t(uchar)<td>int<td>1
 * <tr><td>float<td>float<td>1
 * </table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/integral.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/integral.h"
 * #include "ppl/common/oclcommon.h"
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int channels = 1;
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
 *   Integral<float, float>(queue, height, width,
 *       width, gpu_input, height + 1, width + 1,
 *       width + 1, gpu_output);
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
template <typename Tsrc, typename Tdst>
ppl::common::RetCode Integral(cl_command_queue queue,
                                     int inHeight,
                                     int inWidth,
                                     int inWidthStride,
                                     const cl_mem inData,
                                     int outHeight,
                                     int outWidth,
                                     int outWidthStride,
                                     cl_mem outData);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_INTEGRAL_H_
