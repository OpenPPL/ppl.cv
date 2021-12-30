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

#include "ppl/cv/x86/calchist.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/log.h"
#include <string.h>

namespace ppl {
namespace cv {
namespace x86 {

template <>
::ppl::common::RetCode CalcHist<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t* outHist,
    int32_t maskWidthStride,
    const unsigned char* mask)
{
    if (nullptr == inData || nullptr == outHist) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }

    memset(outHist, 0, sizeof(int32_t) * 256);

    if (mask) {
        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++w) {
                int32_t inIdx      = h * inWidthStride + w;
                int32_t inIdx_mask = h * maskWidthStride + w;
                if (mask[inIdx_mask]) {
                    int32_t value = inData[inIdx];
                    outHist[value]++;
                }
            }
        }
    } else {
        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++w) {
                int32_t inIdx = h * inWidthStride + w;
                int32_t value = inData[inIdx];
                outHist[value]++;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
