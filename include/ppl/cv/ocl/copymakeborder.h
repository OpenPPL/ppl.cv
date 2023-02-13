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

#ifndef _ST_HPC_PPL_CV_OCL_COPY_MAKE_BORDER_H_
#define _ST_HPC_PPL_CV_OCL_COPY_MAKE_BORDER_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Forms a border around an image.
 * @tparam T The data type of input and output image, uint8_t(uchar) and float
 *         are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param queue          opencl command queue.
 * @param height         input and output image's height.
 * @param width          input and output image's width.
 * @param inWidthStride  input image's width stride, which is not less than
 *                       `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @param top            number of top pixels.
 * @param bottom         number of bottom pixels.
 * @param left           number of left pixels.
 * @param right          number of right pixels.
 * @param border_type    ways to deal with border. BORDER_CONSTANT,
 *                       BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP,
 *                       BORDER_REFLECT_101 and BORDER_DEFAULT are supported
 *                       now.
 * @param border_value   value for BORDER_CONSTANT.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>Header files <td>#include "ppl/cv/ocl/copymakeborder.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/copymakeborder.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *   int border   = 16;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * channels * sizeof(float);
 *   int dst_size = (height + border * 2) * (width + border * 2) * channels *
 *                   sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   CopyMakeBorder<float, 3>(queue, height, width, width * channels, gpu_input,
 *                            (width + border * 2) * channels, gpu_output,
 *                            border, border, border, border,
 *                            ppl::cv::BORDER_DEFAULT);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
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
ppl::common::RetCode CopyMakeBorder(cl_command_queue queue,
                                    int height,
                                    int width,
                                    int inWidthStride,
                                    const cl_mem inData,
                                    int outWidthStride,
                                    cl_mem outData,
                                    int top,
                                    int bottom,
                                    int left,
                                    int right,
                                    BorderType border_type,
                                    T border_value = 0);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_COPY_MAKE_BORDER_H_
