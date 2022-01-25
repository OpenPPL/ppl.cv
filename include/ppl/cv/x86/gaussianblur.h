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

#ifndef __ST_HPC_PPL_CV_X86_GAUSSIANBLUR_H_
#define __ST_HPC_PPL_CV_X86_GAUSSIANBLUR_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Denoise or obscure an image with gaussian alogrithm.
 * @tparam T The data type of input image, currently only \a uint8_t(uchar) and \a float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param kernel_len        the length of mask, only odd num is supported.
 * @param sigma             standard deviation
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @param border_type       ways to deal with border. For float, only BORDER_REFLECT_101 or BORDER_DEFAULT are supported now.
 *                          For uchar, all BORDER_TYPE are supported.
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
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/gaussianblur.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/gaussianblur.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t C = 3;
 *     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
 *
 *     ppl::cv::x86::GaussianBlur<float, 3>(H, W, W * 3, dev_iImage, 5, 1.2f, W * 3, dev_oImage, ppl::cv::BORDER_DEFAULT);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T, int32_t numChannels>
::ppl::common::RetCode GaussianBlur(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T *inData,
    int32_t kernel_len,
    float sigma,
    int32_t outWidthStride,
    T *outData,
    BorderType border_type = ppl::cv::BORDER_DEFAULT);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_GAUSSIANBLUR_H_
