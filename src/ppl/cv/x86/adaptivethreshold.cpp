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

#include "ppl/cv/x86/adaptivethreshold.h"
#include "ppl/cv/types.h"
#include <assert.h>
#include <cmath>
#include "ppl/cv/x86/setvalue.h"
#include "ppl/cv/x86/boxfilter.h"
#include "ppl/cv/x86/gaussianblur.h"

namespace ppl {
namespace cv {
namespace x86 {

::ppl::common::RetCode AdaptiveThreshold(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    double max_value,
    int32_t adaptive_method,
    int32_t threshold_type,
    int32_t ksize,
    double delta,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride < width || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    uint8_t setted_value = 0;
    if (max_value < 0) {
        Zeros<uint8_t, 1>(height, width, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else if (max_value < 255.f) {
        setted_value = (uint8_t)max_value;
    } else {
        setted_value = 255;
    }
    uint8_t* mean = outData;
    if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_MEAN_C) {
        BoxFilter<uint8_t, 1>(height, width, inWidthStride, inData, ksize, ksize, true, outWidthStride, mean, ppl::cv::BORDER_REPLICATE);
    } else if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C) {
        GaussianBlur<uint8_t, 1>(height, width, inWidthStride, inData, ksize, 0, outWidthStride, mean, ppl::cv::BORDER_REPLICATE);
    }
    int32_t idelta = threshold_type == ppl::cv::CV_THRESH_BINARY ? std::ceil(delta) : std::floor(delta);
    uint8_t tab[768];
    int32_t i, j;
    if (threshold_type == ppl::cv::CV_THRESH_BINARY) {
        for (i = 0; i < 768; i++)
            tab[i] = (uint8_t)(i - 255 > -idelta ? setted_value : 0);
    } else if (threshold_type == ppl::cv::CV_THRESH_BINARY_INV) {
        for (i = 0; i < 768; i++)
            tab[i] = (uint8_t)(i - 255 <= -idelta ? setted_value : 0);
    }
    for (i = 0; i < height; i++) {
        const uint8_t* sdata = inData + i * inWidthStride;
        const uint8_t* mdata = mean + i * outWidthStride;
        uint8_t* ddata       = outData + i * outWidthStride;
        for (j = 0; j < width; j++)
            ddata[j] = tab[sdata[j] - mdata[j] + 255];
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86