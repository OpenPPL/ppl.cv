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

#ifndef __ST_HPC_PPL_CV_X86_MEAN_H_
#define __ST_HPC_PPL_CV_X86_MEAN_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Calculates a mean of array elements.
* @tparam T The data type of input image, currently \a uint8_t and \a float are supported.
* @tparam c The number of channels of input data, 1, 3 and 4 are supported.
* @param height            height of image
* @param width             widht of image 
* @param inWidthStride     width stride of input image, usually equals to width * channel
* @param inData            input image data
* @param outMeanData       output mean data, its size depends on channel wise, 1 or channel
* @param inMaskStride      width stride of mask, usually equals to width
* @param inMask            mask to determine whether to include this pixel for computation, if NULL, all pixels will be included
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
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/mean.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/mean.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t height = 640;
*     const int32_t width = 480;
*     const int32_t nc = 3;
*     float* inData     = (float*)malloc(width * height * nc * sizeof(float));
*     float* outMeanData= (float*)malloc(nc * sizeof(float));
*
*     ppl::cv::x86::Mean<float, 3>(height, width, width * nc, inData, outMeanData);
*     
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t c>
::ppl::common::RetCode Mean(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    float* outMeanData,
    int32_t inMaskStride = 0,
    uint8_t* inMask = NULL);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_MEAN_H_
