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

#include "ppl/cv/x86/equalizehist.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/log.h"
#include <string.h>
#include <cmath>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

::ppl::common::RetCode EqualizeHist(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inWidth <= 0 || inHeight <= 0 || inWidthStride < inWidth || outWidthStride < inWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int32_t hist_sz = 256;
    int32_t hist[hist_sz] = {
        0,
    };
    int32_t lut[hist_sz];

    for (int32_t h = 0; h < inHeight; ++h)
        for (int32_t w = 0; w < inWidth; ++w) {
            int32_t inIdx = h * inWidthStride + w;
            int32_t value = inData[inIdx];
            hist[value]++;
        }

    int32_t i = 0;
    while (!hist[i])
        ++i;

    int32_t total = inHeight * inWidth;
    float scale   = (hist_sz - 1.f) / (total - hist[i]);

    int32_t sum = 0;
    for (lut[i++] = 0; i < hist_sz; ++i) {
        sum += hist[i];
        lut[i] = std::round(sum * scale);
    }

    for (int32_t h = 0; h < inHeight; ++h) {
        for (int32_t w = 0; w < inWidth; ++w) {
            int32_t inIdx   = h * inWidthStride + w;
            int32_t outIdx  = h * outWidthStride + w;
            int32_t value   = inData[inIdx];
            outData[outIdx] = lut[value];
        }
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
