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

#include "ppl/cv/types.h"
#include <arm_neon.h>
#include <string.h>
#include <limits>
#include <float.h>
#include "ppl/cv/arm/medianblur.h"
#include "ppl/cv/arm/copymakeborder.h"
#include <time.h>
#include <stdlib.h>

namespace ppl {
namespace cv {
namespace arm {

template <typename T>
static T findKth(T* a, int32_t n, int32_t k)
{
    T x = a[0];
    int32_t i = 0, j = n - 1, pos = 0;
    while (i < j) {
        while (i < j && a[j] >= x) {
            --j;
        }
        if (i < j) {
            a[pos] = a[j];
            pos = j;
        }
        while (i < j && a[i] <= x) {
            ++i;
        }
        if (i < j) {
            a[pos] = a[i];
            pos = i;
        }
    }
    a[pos] = x;
    if (pos == k - 1) {
        return a[pos];
    }
    if (pos < k - 1) {
        return findKth(a + pos + 1, n - pos - 1, k - pos - 1);
    } else
        return findKth(a, pos + 1, k);
}

template <typename T, int32_t channels>
::ppl::common::RetCode MedianBlur(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData,
    int32_t ksize,
    BorderType border_type)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_REFLECT_101 && border_type != BORDER_REFLECT && border_type != BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int32_t cn = channels;
    int32_t radius_x = ksize / 2;
    int32_t radius_y = ksize / 2;

    T* buffer = (T*)malloc((height + 2 * radius_y) * (width + 2 * radius_x) * sizeof(T) * cn);
    T* temp = (T*)malloc(ksize * ksize * sizeof(T));

    CopyMakeBorder<T, cn>(height, width, inWidthStride, inData, height + 2 * radius_y, width + 2 * radius_x, (width + 2 * radius_x) * cn, buffer, border_type);

    int32_t area = ksize * ksize;
    int32_t midIndex = (area >> 1) + 1;
    for (int32_t i = 0; i < height; ++i) {
        for (int32_t j = 0; j < width; ++j) {
            for (int32_t c = 0; c < cn; ++c) {
                for (int32_t ky = 0; ky < ksize; ++ky) {
                    for (int32_t kx = 0; kx < ksize; ++kx) {
                        temp[ky * ksize + kx] = buffer[(i + ky) * (width + 2 * radius_x) * cn + (j + kx) * cn + c];
                    }
                }
                outData[i * outWidthStride + j * cn + c] = findKth(temp, area, midIndex);
            }
        }
    }
    free(buffer);
    free(temp);
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode MedianBlur<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    int32_t ksize,
    BorderType border_type);

template ::ppl::common::RetCode MedianBlur<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    int32_t ksize,
    BorderType border_type);

template ::ppl::common::RetCode MedianBlur<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    int32_t ksize,
    BorderType border_type);

}
}
} // namespace ppl::cv::arm
