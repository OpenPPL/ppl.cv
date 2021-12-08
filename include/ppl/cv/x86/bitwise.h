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

#ifndef __ST_HPC_PPL_CV_X86_BITWISE_H_
#define __ST_HPC_PPL_CV_X86_BITWISE_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Computes bitwise conjunction of the two arrays (out = in1 & in2).
 * @tparam T The data type of input image, currently only \a uint8_t is supported.
 * @tparam c The number of channels of input image, 1, 3 and 4 are supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride0    first input image's width stride, usually it equals to `width * channels`
 * @param inData0           input image data0
 * @param inWidthStride1    second input image's width stride, usually it equals to `width * channels`
 * @param inData1           input image data1
 * @param outWidthStride    output image's width stride, usually it equals to `width * channels`
 * @param outData           output image data
 * @param inMaskWidthStride [optional] input mask's width stride,  usually it equals to `width`
 * @param inMask            [optional] input mask data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uint8_t)<td>1
 * <tr><td>uint8_t(uint8_t)<td>3
 * <tr><td>uint8_t(uint8_t)<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/bitwise.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/bitwise.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t C = 3;
 *     int32_t Stride = W * C;
 *     uint8_t* dev_iImage1 = (uint8_t*)malloc(W * H * C * sizeof(float));
 *     uint8_t* dev_iImage2 = (uint8_t*)malloc(W * H * C * sizeof(float));
 *     uint8_t* dev_oImage  = (uint8_t*)malloc(W * H * C * sizeof(float));
 *     ppl::cv::x86::BitwiseAnd<uint8_t, C>(H, W, Stride, dev_iImage1, Stride, dev_iImage2, Stride, dev_oImage);
 *
 *     free(dev_iImage1);
 *     free(dev_iImage2);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T, int32_t c>
::ppl::common::RetCode BitwiseAnd(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T* inData0,
    int32_t inWidthStride1,
    const T* inData1,
    int32_t outWidthStride,
    T* outData,
    int32_t inMaskWidthStride = 0,
    const uint8_t* inMask     = nullptr);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_BITWISE_H_