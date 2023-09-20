/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef __ST_HPC_PPL_CV_AARCH64_ROTATE_H_
#define __ST_HPC_PPL_CV_AARCH64_ROTATE_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace arm {

/**
 * @brief  Rotates a 2D array in multiples of 90 degrees.
 * @tparam T The data type of input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param inHeight         input image's height.
 * @param inWidth          input image's width.
 * @param inWidthStride    input image's width stride, usually it equals to `width * channels`
 * @param inData           input image data.
 * @param outHeight        output image's height.
 * @param outWidth         output image's width.
 * @param outWidthStride   the width stride of output image, usually it equals to `width * channels`
 * @param outData          output image data.
 * @param degree           rotation angle, 90, 180 and 270 are supported.
 * @return The execution status, succeeds or fails with an error code.
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
 * <tr><td>arm platforms supported <td> All
 * <tr><td>Header files  <td> #include "ppl/cv/arm/rotate.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/arm/rotate.h"
 *
 * int main(int argc, char** argv) {
 *     const int inHeight = 640;
 *     const int inWidth = 480;
 *     const int C = 3;
 *     const int outHeight = 480;
 *     const int outWidth = 640;
 *     const int degree = 90;
 *     float* dev_iImage = (float*)malloc(inWidth * inHeight * C * sizeof(float));
 *     float* dev_oImage = (float*)malloc(outWidth * outHeight * C * sizeof(float));
 *
 *     ppl::cv::arm::Rotate<float, 3>(inHeight, inWidth, inWidth * C, dev_iImage,
 *             outHeight, outWidth, outWidth * C, dev_oImage, degree);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Rotate(int inHeight,
                            int inWidth,
                            int inWidthStride,
                            const T* inData,
                            int outHeight,
                            int outWidth,
                            int outWidthStride,
                            T* outData,
                            int degree);

}  // namespace arm
}  // namespace cv
}  // namespace ppl

#endif  // __ST_HPC_PPL_CV_AARCH64_ROTATE_H_
