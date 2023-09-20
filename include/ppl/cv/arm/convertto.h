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

#ifndef __ST_HPC_PPL_CV_AARCH64_CONVERTTO_H_
#define __ST_HPC_PPL_CV_AARCH64_CONVERTTO_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace arm {

/**
* @brief Converts an image to another data type with optional scale and delta.
* @tparam Tsrc The data type of input image, currently only \a uint8 and \a float are supported.
* @tparam Tdst The data type of output image, currently only \a float and \a uint8 are supported.
* @tparam nc The number of channels of input&output image, 1, 3 and 4 are supported.
* @param height            input&output image's height
* @param width             input&output image's width
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outWidthStride    output image's width stride, usually it equals to `width * channels`
* @param scale             optional scale factor.
* @param delta             optional delta added to the scaled values.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>uint8_t<td>1
* <tr><td>float<td>uint8_t<td>3
* <tr><td>float<td>uint8_t<td>4
* <tr><td>uint8_t<td>float<td>1
* <tr><td>uint8_t<td>float<td>3
* <tr><td>uint8_t<td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>arm platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/convertto.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/convertto.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(uint8_t));
*
*     ppl::cv::arm::ConvertTo<float, 3, uint8_t>(H, W, W * C, dev_iImage, W * C, dev_oImage, 1.0f, 0,0f);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename TSrc, typename TDst, int32_t channels>
::ppl::common::RetCode ConvertTo(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const TSrc* inData,
    int32_t outWidthStride,
    TDst* outData,
    float scale=1.0f,
    float delta=0.0f);

}
}
} // namespace ppl::cv::arm
#endif //! __ST_HPC_PPL_CV_AARCH64_CONVERTTO_H_
