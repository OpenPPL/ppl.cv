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

#ifndef __ST_HPC_PPL_CV_X86_ADAPTIVETHRESHOLD_H_
#define __ST_HPC_PPL_CV_X86_ADAPTIVETHRESHOLD_H_

#include "ppl/common/retcode.h"
#include <ppl/cv/types.h>

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Applies an adaptive threshold to an image.
 * @param inHeight       input image's height.
 * @param inWidth        input image's width need to be processed.
 * @param inWidthStride  input image's width stride
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image
 * @param outData        output image data.
 * @param maxValue       value assigned to the pixels for which the condition is satisfied.
 * @param adaptiveMethod adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C and ADAPTIVE_THRESH_GAUSSIAN_C are supported.
 * @param thresholdType  thresholding type, THRESH_BINARY and THRESH_BINARY_INV are supported.
 * @param blockSize      size of a pixel neighborhood that is used to calculate a threshold value for the pixel, greater than 1 and up to 15.
 * @param delta          constant subtracted from the mean or weighted mean.
 * @param border_type    ways to deal with border, only BORDER_REPLICATE is supported now.
 * @warning There is a very small probability that some pixels will be calculated incorrectly.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/arm/adaptiveThreshold.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/adaptiveThreshold.h>
 * using namespace ppl::cv::x86;
 *
 * int32_t main(int32_t argc, char** argv) {
 *   int32_t width    = 640;
 *   int32_t height   = 480;
 *
 *   uint8_t* devInputImage;
 *   uint8_t* devOutputImage;
 *   uint8_t* image = (uint8_t*)malloc(width * height * sizeof(uint8_t));
 *   AdaptiveThreshold(height, width, width, devInputImage, width,
 *                     devOutputImage, 155, ADAPTIVE_THRESH_MEAN_C,
 *                     THRESH_BINARY, 5, 0.5);
 *   free(image);
 *   return 0;
 * }
 * @endcode
 */
::ppl::common::RetCode AdaptiveThreshold(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    double maxValue,
    int32_t adaptiveMethod,
    int32_t thresholdType,
    int32_t blockSize,
    double delta,
    BorderType border_type = BORDER_REPLICATE);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_ADAPTIVETHRESHOLD_H_
