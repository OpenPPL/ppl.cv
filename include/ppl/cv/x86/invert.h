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

#ifndef __ST_HPC_PPL_CV_X86_INVERT_H_
#define __ST_HPC_PPL_CV_X86_INVERT_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Inverse (or pseudo-inverse) of a matrix
 * @tparam T The data type of input data, currently only \a float and \a double are supported.
 * @param height height of matrix
 * @param widht  width of matrix
 * @param inWidthStride input matrix's width stride, usually it equals to `width`
 * @param src   input matrix data
 * @param outWidthStride output matrix's width stride, usually it equals to `width`
 * @param dst   output matrix data
 * @param method inverse method, see enum InvertMethod. Only DECOMP_CHOLESKY is supported now.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>float <td>1
 * <tr><td>double<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/invert.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/invert.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t width = 64;
 *     const int32_t height= 64;
 *     const sstep = width;
 *     const dstep = width;
 *     float* dev_iImage = (float*)malloc(width * height * sizeof(float));
 *     float* dev_oImage = (float*)malloc(width * height * sizeof(float));
 *
 *     ppl::cv::x86::Invert<float>(height, width, sstep, dev_iImage, dstep, dev_oImage, ppl::cv::DECOMP_CHOLESKY);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T>
::ppl::common::RetCode Invert(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T *src,
    int32_t outWidthStride,
    T *dst,
    InvertMethod decompType);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_INVERT_H_
