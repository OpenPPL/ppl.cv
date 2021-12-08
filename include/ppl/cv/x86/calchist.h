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

#ifndef __ST_HPC_PPL_CV_X86_CALCHIST_H_
#define __ST_HPC_PPL_CV_X86_CALCHIST_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief  calculate the input image's histogram
 * @tparam T The data type of input image and output image, currently only \a uint8_t is supported.
 * @param heigth            input/output image's heigth
 * @param width             input/output image's width
 * @param inWidthStride     input image's stride
 * @param inData            input image data
 * @param outWidthStride    output histogram's stride
 * @param outHist           output histogram data
 * @param maskWidthStride   input mask's stride, should be 0 if mask is null
 * @param mask              define the pixels involved, may be null if all pixels are involved
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type is supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uchar(uint_8)
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
 * #include <ppl/cv/types.h>
 * #include <cuda_runtime.h>
 * #include <memory>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     uchar* dev_iImage = (uchar*)malloc(W * H * sizeof(uchar));
 *     uchar* dev_oImage = (uchar*)malloc(W * H * sizeof(uchar));
 * 
 *     int32_t iStride = width;
 *     int32_t oStride = width;
 *     ppl::cv::x86::CalcHist<uchar>(H, W, iStride, dev_iIamge, oStride, dev_oImage)
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T>
::ppl::common::RetCode CalcHist(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t* outHist,
    int32_t maskWidthStride = 0,
    const unsigned char* mask = nullptr);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_CALCHIST_H_
