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

#ifndef _ST_HPC_PPL_CV_OCL_SEPFILTER2D_H_
#define _ST_HPC_PPL_CV_OCL_SEPFILTER2D_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/ocl/kerneltypes.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Convolves an image with separable linear filters.
 * @tparam Tsrc The data type of input image, currently only uint8_t(uchar) and
 *         float are supported.
 * @tparam Tdst The data type of output image, currently uint8_t(uchar), short
 *         and float are supported.
 * @tparam channels The number of channels of input&output image, 1, 3 and 4
 *         are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less than
 *                       `width * channels * sizeof(T)`.
 * @param inData         input image data, it should be a buffer object.
 * @param ksize          the length of kernel in X&Y direction.
 * @param kernelX        coefficients for filtering each row.
 * @param kernelY        coefficients for filtering each column.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data, it should be a buffer object.
 * @param delta          optional value added to the filtered pixels.
 * @param border_type    ways to deal with border. BORDER_REPLICATE,
 *                       BORDER_REFLECT, BORDER_REFLECT_101 and BORDER_DEFAULT
 *                       are supported now.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, this function
 *         need a memory pool to store the intermediate result, which is not
 *         less than ppl::cv::ocl::ceil2DVolume(width * channels * sizeof(float),
 *         height).
 *       2 The anchor is at the kernel center.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/sepfilter2d.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/sepfilter2d.h"
 * #include "ppl/cv/ocl/use_memory_pool.h"
 * 
 * using namespace ppl::cv::ocl;
 * using namespace ppl::common::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *   int ksize = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   size_t size_width = width * channels * sizeof(float);
 *   size_t ceiled_volume = ppl::cv::ocl::ceil2DVolume(size_width, height);
 *   ppl::cv::ocl::activateGpuMemoryPool(ceiled_volume);
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(float);
 *   int kernel_size = ksize * (int)sizeof(float);
 *   float* input = (float*)malloc(data_size);
 *   float* kernel = (float*)malloc(kernel_size);
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_input_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY, kernel_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input_kernel, CL_FALSE, 0,
 *                                     kernel_size, kernel, 0, NULL, NULL);
 *
 *   SepFilter2D<float, float, 3>(
 *       queue, height, width, width * channels, gpu_input, ksize,
 *       gpu_input_kernel, gpu_input_kernel, width * channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *   ppl::cv::ocl::shutDownGpuMemoryPool();
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename Tsrc, typename Tdst, int channels>
ppl::common::RetCode SepFilter2D(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride,
                                 const cl_mem inData,
                                 int ksize,
                                 const cl_mem kernelX,
                                 const cl_mem kernelY,
                                 int outWidthStride,
                                 cl_mem outData,
                                 float delta = 0.f,
                                 BorderType border_type = BORDER_DEFAULT);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_SEPFILTER2D_H_
