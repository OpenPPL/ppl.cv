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

#ifndef __ST_HPC_PPL_CV_X86_FILTER2D_H_
#define __ST_HPC_PPL_CV_X86_FILTER2D_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Image convolution operation, filter2D.Process the boundary according to the refect101 method.
* @tparam T The data type of input image and output image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param kernel_len        the length of kernel
* @param kernel            Store specific data for the kernel
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param border_type       ways to deal with border. Only BORDER_REFLECT_101 or BORDER_DEFAULT are supported now.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
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
* <tr><td>x86 platforms supported<td> x86 x86
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/filter2d.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/filter2d.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     const float kernel_len = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_matrix = (float*)malloc(kernel_len * kernel_len * sizeof(float));
*
*     ppl::cv::x86::Filter2D<float, 3>(H, W, W * C, dev_iImage, kernel_len, dev_matrix, W * C, dev_oImage, ppl::cv::BORDER_DEFAULT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     free(dev_matrix);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t nc>
::ppl::common::RetCode Filter2D(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t kernel_len,
    const float* kernel,
    int32_t outWidthStride,
    T* outData,
    BorderType border_type);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_FILTER2D_H_
