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

#ifndef _ST_HPC_PPL_CV_OCL_ADAPTIVETHRESHOLD_H_
#define _ST_HPC_PPL_CV_OCL_ADAPTIVETHRESHOLD_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/ocl/kerneltypes.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Applies an adaptive threshold to an image.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less than
 *                       `width * channels`.
 * @param inData         input image data, it should be a buffer object.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data, similar to inData.
 * @param maxValue       Non-zero value assigned to the pixels for which the
 *                       condition is satisfied.
 * @param adaptiveMethod adaptive thresholding algorithm to use,
 *                       ADAPTIVE_THRESH_MEAN_C and ADAPTIVE_THRESH_GAUSSIAN_C
 *                       are supported.
 * @param thresholdType  thresholding type, THRESH_BINARY and THRESH_BINARY_INV
 *                       are supported.
 * @param blockSize      size of a pixel neighborhood that is used to calculate
 *                       a threshold value for the pixel. It must be odd and
 *                       greater than 1.
 * @param delta          constant subtracted from the mean or weighted mean.
 * @param border_type    ways to deal with border. BORDER_REPLICATE,
 *                       BORDER_REFLECT, BORDER_REFLECT_101 and BORDER_DEFAULT
 *                       are supported now.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, this
 *         function need a memory buffer to store the intermediate result, which
 *         is not less than ppl::cv::ocl::ceil2DVolume(width * sizeof(uchar),
 *         height).
 *       2 The output image has the same size and channels as the input image.
 *       3 Only the uchar and single-channels data is supported.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/adaptivethreshold.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/adaptivethreshold.h"
 * #include "ppl/cv/ocl/use_memory_pool.h"
 * #include "ppl/common/ocl/pplopencl.h"
 * #include "ppl/common/oclcommon.h"
 * #include "kerneltypes.h"
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 1;
 *   int adaptive_method = ADAPTIVE_THRESH_MEAN_C;
 *   int threshold_type = THRESH_BINARY;
 *   int ksize = 3;
 *   float max_value = 25.f;
 *   float delta = 5.f;
 *   BorderType border_type = BORDER_REPLICATE;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   size_t size_width = width * channels * sizeof(uchar);
 *   size_t ceiled_volume = ceil2DVolume(size_width, height);
 *   activateGpuMemoryPool(ceiled_volume);
 * 
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(uchar);
 *   uchar* input = (uchar*)malloc(data_size);
 *   uchar* output = (uchar*)malloc(data_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *   ppl::cv::ocl::AdaptiveThreshold(
 *       queue, height, width, width * channels, gpu_input, width * channels, gpu_output,
 *       max_value, adaptive_method, threshold_type, ksize, delta, border_type);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   shutDownGpuMemoryPool();
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */

ppl::common::RetCode AdaptiveThreshold(cl_command_queue queue,
                                       int height,
                                       int width,
                                       int inWidthStride,
                                       const cl_mem inData,
                                       int outWidthStride,
                                       cl_mem outData,
                                       float maxValue,
                                       int adaptiveMethod,
                                       int thresholdType,
                                       int blockSize,
                                       float delta,
                                       BorderType border_type = BORDER_DEFAULT);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_ADAPTIVETHRESHOLD_H_
