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

#ifndef PPL_CV_X86_INTERNAL_AVX_H_
#define PPL_CV_X86_INTERNAL_AVX_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

template <int32_t dcn, int32_t bIdx>
::ppl::common::RetCode YUV420ptoRGB_avx(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData);

template <typename Tsrc, int32_t ncSrc, typename Tdst, int32_t ncDst>
::ppl::common::RetCode BGR2GRAYImage_avx(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData);

template <typename Tsrc, int32_t ncSrc, typename Tdst, int32_t ncDst>
::ppl::common::RetCode RGB2GRAYImage_avx(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData);

}
}
} // namespace ppl::cv::x86
#endif //! PPL_CV_X86_INTERNAL_AVX_H_
