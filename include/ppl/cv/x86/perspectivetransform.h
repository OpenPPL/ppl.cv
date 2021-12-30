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

#ifndef __ST_HPC_PPL_CV_X86_PERSPECTIVETRANSFORM_H_
#define __ST_HPC_PPL_CV_X86_PERSPECTIVETRANSFORM_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Performs the perspective matrix transformation of vectors.
 * @tparam T The data type of input and output image, currently only \a float is supported.
 * @tparam ncSrc The number of channels of input image, 2 or 3 supported.
 * @tparam ncDst The number of channels of output image, 2 or 3 supported.
 * @param height             input and output image's height
 * @param width              input and output image's width need to be processed
 * @param inWidthStride      input image's width stride, usually it equals to `width * ncSrc`
 * @param inData             input image data
 * @param outWidthStride     output image's width stride, usually it equals to `width * ncDst`
 * @param outData            output image data
 * @param mData              transformation matrix
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark Input two-channel or three-channel floating-point array; each element is a 2D/3D vector to be transformed.The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>float<td>2<td>2
 * <tr><td>float<td>2<td>3
 * <tr><td>float<td>3<td>2
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/perspectivetransform.h.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/perspectivetransform.h>
 * int32_t main(int32_t argc, char** argv) {
 *     int32_t W = 640;
 *     int32_t H = 480;
 *     int32_t ncSrc = 3;
 *     int32_t ncDst = 2;
 *     int32_t mHeight = ncDst + 1;
 *     int32_t mWidth  = ncSrc + 1;
 *     float* inImage = (float*)malloc(W * H * ncSrc * sizeof(float));
 *     float* outImage = (float*)malloc(W * H * ncDst * sizeof(float));
 *     float* mData = (float*)malloc(mWidth * mHeight * sizeof(double));
 *     ppl::cv::x86::PerspectiveTransform<float, ncSrc, ncDst>(H, W, W * ncSrc, inImage, W * ncDst, outImage, mData);
 *
 *     free(inImage);
 *     free(outImage);
 *     free(mData);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode PerspectiveTransform(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData,
    const float* mData);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_PERSPECTIVETRANSFORM_H_
