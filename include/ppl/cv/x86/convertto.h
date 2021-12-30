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

#ifndef __ST_HPC_PPL_CV_X86_CONVERTTO_H_
#define __ST_HPC_PPL_CV_X86_CONVERTTO_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Calculates the per-element sum of two arrays
* @tparam Tsrc The data type of input image, currently only \a uint8 and \a float are supported.
* @tparam nc The number of channels of input image and output image, 1, 3 and 4 are supported.
* @tparam Tdst The data type of output image, currently only \a float and \a uint8 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param scale             coeffcient to mutiply
* @param outWidthStride    output image's width stride, usually it equals to `width * channels`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>1<td>uint8_t
* <tr><td>float<td>3<td>uint8_t
* <tr><td>float<td>4<td>uint8_t
* <tr><td>uint8_t<td>1<td>float
* <tr><td>uint8_t<td>3<td>float
* <tr><td>uint8_t<td>4<td>float
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/convertto.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/convertto.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(uint8_t));
*
*     ppl::cv::x86::ConvertTo<float, 3, uint8_t>(H, W, W * C, dev_iImage, 1.0f, W * C, dev_oImage);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename TSrc, int32_t channels, typename TDst>
::ppl::common::RetCode ConvertTo(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const TSrc* inData,
    float scale,
    int32_t outWidthStride,
    TDst* outData);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_CONVERTTO_H_
