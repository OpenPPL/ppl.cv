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

#ifndef __ST_HPC_PPL_CV_X86_PYRUP_H_
#define __ST_HPC_PPL_CV_X86_PYRUP_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Upsamples an image and then blurs it.
* @tparam T The data type of input and output image, currently only \a uint8_t(uchar) and \a float are supported.
* @tparam nc The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride     input image's width stride, usually it equals to `width * nc`
* @param inData            input image data
* @param outWidthStride    output image's width stride, usually it equals to `outWidth * nc`
* @param outData           output image data
* @param border_type       ways to deal with border. Only BORDER_REFLECT_101 or BORDER_DEFAULT are supported now.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @warning outHeight == height * 2 or height * 2 + 1
* @warning outWidth == width * 2 or width * 2 + 1
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* <tr><td>uint8_t(uchar)<td>1
* <tr><td>uint8_t(uchar)<td>3
* <tr><td>uint8_t(uchar)<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All 
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/pyrUp.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/pyrUp.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 320;
*     const int32_t H = 240;
*     const int32_t OW = W * 2;
*     const int32_t OH = H * 2;
*     const int32_t C = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(OW * OH * C * sizeof(float));
*
*     ppl::cv::x86::PyrUp<float, 3>(H, W, W * C, dev_iImage, OW * C, dev_oImage, ppl::cv::BORDER_DEFAULT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t nc>
::ppl::common::RetCode PyrUp(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData,
    BorderType border_type);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_PYRUP_H_
