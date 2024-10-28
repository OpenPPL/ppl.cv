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

#ifndef _ST_HPC_PPL_CV_OCL_WARPPERSPECTIVE_H_
#define _ST_HPC_PPL_CV_OCL_WARPPERSPECTIVE_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Applies a perspective transformation to an image.
 * @tparam T The data type of input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param queue            opencl command queue.
 * @param inHeight         input image's height.
 * @param inWidth          input image's width.
 * @param inWidthStride    input image's width stride, which is not less than
 *                         `width * channels`.
 * @param inData           input image data, it should be a buffer object.
 * @param outHeight        output image's height.
 * @param outWidth         output image's width.
 * @param outWidthStride   the width stride of output image, similar to
 *                         inWidthStride.
 * @param outData          output image data, it should be a buffer object.
 * @param affineMatrix     3 x 3 transformation matrix.
 * @param interpolation    Interpolation method. INTERPOLATION_LINEAR and
 *                         INTERPOLATION_NEAREST_POINT are supported.
 * @param border_type      ways to deal with border. BORDER_CONSTANT/
 *                         BORDER_REPLICATE/BORDER_TRANSPARENT are supported.
 * @param borderValue      value used in case of a constant border; by default,
 *                         it is 0.
 * @return The execution status, succeeds or fails with an error code.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
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
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/warpperspective.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/warpperspective.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int src_width  = 320;
 *   int src_height = 240;
 *   int dst_width  = 640;
 *   int dst_height = 480;
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
 *   float affine_matrix[6] = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f};
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   WarpPerspective<float, 3>(queue, dst_height, src_width,
 *       src_width * channels, gpu_input, dst_height, dst_width,
 *       dst_width * channels, gpu_output, affine_matrix, ppl::cv::INTERPOLATION_LINEAR);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   free(affine_matrix);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode WarpPerspective(cl_command_queue queue,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const cl_mem inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  cl_mem outData,
                                  const float *perspectiveMatrix,
                                  InterpolationType interpolation,
                                  BorderType border_type = BORDER_CONSTANT,
                                  T borderValue = 0);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_WARPPERSPECTIVE_H_
