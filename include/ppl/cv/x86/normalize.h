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

#ifndef __ST_HPC_PPL_CV_X86_NORMALIZE_H_
#define __ST_HPC_PPL_CV_X86_NORMALIZE_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Normalizes the norm or value range of an array.
 * @param inHeight       input image's height.
 * @param inWidth        input image's width need to be processed.
 * @param inWidthStride  input image's width stride, usually it equals "width * channels".
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @param alpha          norm value to normalize to or the lower range boundary in case of the range normalization.
 * @param beta           upper range boundary in case of the range normalization; it is not used for the norm normalization.
 * @param normType       normalization type,NORM_L1, NORM_L2, NORM_INF, NORM_MINMAX are supported.
 * @param maskSteps      [optional] steps of mask,in Bytes, usually it equals to `width * sizeof(uchar)`
 * @param mask           optional operation mask; it must have the same size as inData and CV_8UC1 type.
 * @warning All parameters must be valid, or undefined behaviour may occur.
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
 * <tr><td>Header files<td> #include &lt;ppl/cv/arm/normalize.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/normalize.h>
 * using namespace ppl::cv::x86;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *
 *   uchar* devInputImage;
 *   uchar* devOutputImage;
 *   uchar* image = (uchar*)malloc(width * height * sizeof(uchar));
 *   Normalize<uchar, 3>(height, width, width * nc, devInputImage, width * nc,
 *                     devOutputImage, 1, 0, NORM_L2, 0, NULL);
 *   free(image);
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int nc>
::ppl::common::RetCode Normalize(int height,
                                 int width,
                                 int inWidthStride,
                                 const T *inData,
                                 int outWidthStride,
                                 float *outData,
                                 double alpha                = 1,
                                 double beta                 = 0,
                                 ppl::cv::NormTypes normType = NORM_L2,
                                 int maskSteps               = 0,
                                 const uchar *mask           = NULL);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL3_CV_X86_NORMALIZE_H_
