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

#ifndef __ST_HPC_PPL_CV_RISCV_ERODE_H_
#define __ST_HPC_PPL_CV_RISCV_ERODE_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace riscv {

/**
 * @brief Denoise or obscure an image with dialte alogrithm
 * @tparam T The data type of input and output image, currently only \a uint8_t and \a float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param kernelx_len       the length of mask , x direction.
 * @param kernely_len       the length of mask , y direction.
 * @param kernel            the data of the kernel mask.
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @param border_type       ways to deal with border. Only BORDER_CONSTANT is supported now.
 * @param border_value      filling border_value for BORDER_CONSTANT
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
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
 * <tr><td>riscv platforms supported<td> &gt; riscvv7 riscvv8
 * <tr><td>Header files<td> #include &lt;ppl/cv/riscv/dilate.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/riscv/dilate.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t C = 3;
 *     const int32_t kernelx_len = 3;
 *     const int32_t kernely_len = 3;
 *     (T*)dev_iImage = (T*)malloc(W * H * C * sizeof(T));
 *     (T*)dev_oImage = (T*)malloc(W * H * C * sizeof(T));
 *     (unsigned char*)kernel = (unsigned char*)malloc(kernel_len * kernel_len * sizeof(unsigned char));
 *
 *     ppl::cv::riscv::Erode<float, 3>(H, W, W * 3, dev_iImage, kernelx_len, kernely_len, kernel, W * 3, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     free(kernel);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T, int32_t channels>
::ppl::common::RetCode Erode(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const unsigned char* kernel,
    int32_t outWidthStride,
    T* outData,
    BorderType border_type = BORDER_CONSTANT,
    T border_value = 0);

}
}
} // namespace ppl::cv::riscv

#endif //!__ST_HPC_PPL_CV_RISCV_ERODE_H_
