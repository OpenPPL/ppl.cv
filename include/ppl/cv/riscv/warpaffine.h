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

#ifndef __ST_HPC_PPL_CV_RISCV_WARPAFFINE_H_
#define __ST_HPC_PPL_CV_RISCV_WARPAFFINE_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace riscv {

/**
* @brief Affine transformation with nearest neighbor interpolation method
* @tparam T The data type of input image and output image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image and output image, 1, 4 are supported.
* @param height            input image's height
* @param width             input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param affineMatrix      the mask of warpaffine
* @param border_type       support ppl::cv::BORDER_CONSTANT/ppl::cv::BORDER_REPLICATE/ppl::cv::BORDER_TRANSPARENT
* @param border_value      border value for BORDER_CONSTANT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>uint8_t(uint8_t)<td>1
* <tr><td>uint8_t(uint8_t)<td>3
* <tr><td>uint8_t(uint8_t)<td>4
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>riscv platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/riscv/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/riscv/warpaffine.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t inWidth = 640;
*     const int32_t inHeight = 480;
*     const int32_t outWidth = 320;
*     const int32_t outHeight = 240;
*     const int32_t C = 1;
*     float* dev_iImage = (float*)malloc(inWidth * inHeight * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(outWidth * outHeight * C * sizeof(float));
*     float* affineMatrix = (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::riscv::WarpAffineNearestPoint<float, 4>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, affineMatrix, ppl::cv::BORDER_CONSTANT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int32_t numChannels>
::ppl::common::RetCode WarpAffineNearestPoint(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData,
    const double* affineMatrix,
    BorderType border_type = BORDER_CONSTANT,
    T border_value = 0);

/**
* @brief Affine transformation with linear interpolation method
* @tparam T The data type of input image and output image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image and output image, 1, 4 are supported.
* @param height            input image's height
* @param width             input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param affineMatrix      the mask of warpaffine
* @param border_type       support ppl::cv::BORDER_CONSTANT/ppl::cv::BORDER_REPLICATE/ppl::cv::BORDER_TRANSPARENT
* @param border_value      border value for BORDER_CONSTANT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>uint8_t(uint8_t)<td>1
* <tr><td>uint8_t(uint8_t)<td>3
* <tr><td>uint8_t(uint8_t)<td>4
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>riscv platforms supported<td> all
* <tr><td>Header files<td> #include &lt;ppl/cv/riscv/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/riscv/warpaffine.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t inWidth = 640;
*     const int32_t inHeight = 480;
*     const int32_t outWidth = 320;
*     const int32_t outHeight = 240;
*     const int32_t C = 1;
*     float* dev_iImage = (float*)malloc(inWidth * inHeight * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(outWidth * outHeight * C * sizeof(float));
*     float* affineMatrix = (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::riscv::WarpAffineLinear<float, 4>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, affineMatrix, ppl::cv::BORDER_CONSTANT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t numChannels>
::ppl::common::RetCode WarpAffineLinear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData,
    const double* affineMatrix,
    BorderType border_type = BORDER_CONSTANT,
    T border_value = 0);

}
}
} // namespace ppl::cv::riscv
#endif //! __ST_HPC_PPL3_CV_RISCV_WARPAFFINE_H_
