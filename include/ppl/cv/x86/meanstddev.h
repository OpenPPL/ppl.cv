// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_HPC_PPL_CV_X86_MEANSTDDEV_H_
#define __ST_HPC_PPL_CV_X86_MEANSTDDEV_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Calculates a mean and standard deviation of array elements.
 * @tparam T The data type of input image, currently only \a uint8_t and \a float are supported.
 * @tparam c The number of channels of input image, 1, 3 and 4 are supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param srcStride         input image's width stride, usually it equals to `width * channels`
 * @param ptrSrc            input image data
 * @param mean              output parameter: calculated mean value of each channel
 * @param stddev            output parameter: calculateded standard deviation of each channel
 * @param maskStride        input mask's width stride,  usually it equals to `width`
 * @param mask              input mask data
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
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/meanstddev.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/meanstddev.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t C = 3;
 *     const int32_t ksize = 3;
 *     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
 *     double mean[3];
 *     double stddev[3];
 *     ppl::cv::x86::MeanStdDev<float, 3>(H, W, W * C, dev_iImage, mean, stddev);
 *
 *     free(dev_iImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/

template <typename T, int32_t c>
::ppl::common::RetCode MeanStdDev(
    int32_t height,
    int32_t width,
    int32_t srcStride,
    const T* ptrSrc,
    float* mean,
    float* stddev,
    int32_t maskStride     = 0,
    const uint8_t* ptrMask = nullptr);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_MEANSTDDEV_H_
