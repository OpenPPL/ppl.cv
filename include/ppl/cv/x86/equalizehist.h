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

#ifndef __ST_HPC_PPL_CV_X86_EQUALIZEHIST_H_
#define __ST_HPC_PPL_CV_X86_EQUALIZEHIST_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Equalizes the histogram of a grayscale image.
* @param inHeight          inData's height, the same height as outData
* @param inWidth           inData's width,  the same width as outData
* @param inWidthStride     inData's width stride, usually it equals to `width`
* @param inData            input data
* @param outWidthStride    outData's width stride, usually it equals to `width`
* @param outData           output data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type is supported.
* <table>
* <tr><td>uint8_t
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/equalizehist.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/equalizehist.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t));
*     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t));
*
*     int32_t iStride = width;
*     int32_t oStride = width;
*     ppl::cv::x86::EqualizeHist(H, W, iStride, dev_iIamge, oStride, dev_oImage)
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

::ppl::common::RetCode EqualizeHist(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_EQUALIZEHIST_H_
