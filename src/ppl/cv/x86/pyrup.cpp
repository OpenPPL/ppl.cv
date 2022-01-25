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

#include "ppl/cv/x86/pyrup.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <string.h>
#include <cmath>

namespace ppl {
namespace cv {
namespace x86 {

void pyrup_kernel_f32(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    const int32_t PU_SZ = 3;
    int32_t bufstep     = ((outWidth + 1) * channels + 16 - 1) / 16 * 16;
    float *buf          = (float *)malloc(bufstep * PU_SZ * sizeof(float));
    memset(buf, 0, bufstep * PU_SZ * sizeof(float));
    int32_t *dtab = (int32_t *)malloc(width * channels * sizeof(int32_t));

    float *rows[3];

    int32_t k, x, sy0 = -PU_SZ / 2, sy = sy0;
    float r = 1.0f / 64;

    width *= channels;
    outWidth *= channels;

    for (x = 0; x < width; ++x) {
        dtab[x] = (x / channels) * 2 * channels + x % channels;
    }

    for (int32_t y = 0; y < height; ++y) {
        float *dst0 = outData + (y * 2) * outWidthStride;

        int32_t oy1 = y * 2 + 1;
        if (oy1 >= outHeight) {
            oy1 = outHeight - 1;
        }
        float *dst1 = outData + oy1 * outWidthStride;
        float *row0, *row1, *row2;

        for (; sy <= y + 1; ++sy) {
            float *row  = buf + ((sy - sy0) % PU_SZ) * bufstep;
            int32_t _sy = borderInterpolate(sy * 2, height * 2) / 2;

            const float *src = inData + _sy * inWidthStride;

            if (width == channels) {
                for (x = 0; x < channels; ++x) {
                    row[x] = row[x + channels] = src[x] * 8;
                }
                continue;
            }

            for (x = 0; x < channels; ++x) {
                int32_t dx         = dtab[x];
                float t0           = src[x] * 6 + src[x + channels] * 2;
                float t1           = (src[x] + src[x + channels]) * 4;
                row[dx]            = t0;
                row[dx + channels] = t1;
                dx                 = dtab[width - channels + x];
                int32_t sx         = width - channels + x;
                t0                 = src[sx - channels] + src[sx] * 7;
                t1                 = src[sx] * 8;
                row[dx]            = t0;
                row[dx + channels] = t1;

                if (outWidth > width * 2) {
                    row[outWidth - channels + x] = row[dx + channels];
                }
            }

            for (x = channels; x < width - channels; ++x) {
                int32_t dx         = dtab[x];
                float t0           = src[x - channels] + src[x] * 6 + src[x + channels];
                float t1           = (src[x] + src[x + channels]) * 4;
                row[dx]            = t0;
                row[dx + channels] = t1;
            }
        }

        for (k = 0; k < PU_SZ; ++k) {
            rows[k] = buf + ((y - PU_SZ / 2 + k - sy0) % PU_SZ) * bufstep;
        }
        row0 = rows[0];
        row1 = rows[1];
        row2 = rows[2];

        x = 0;
        for (; x < outWidth; ++x) {
            float t1 = (row1[x] + row2[x]) * 4;
            float t0 = (row0[x] + row1[x] * 6 + row2[x]);
            dst1[x]  = t1 * r;
            dst0[x]  = t0 * r;
        }
    }

    if (outHeight > height * 2) {
        float *dst0 = outData + (height * 2 - 2) * outWidthStride;
        float *dst2 = outData + (height * 2) * outWidthStride;

        for (x = 0; x < outWidth; ++x) {
            dst2[x] = dst0[x];
        }
    }

    free(dtab);
    free(buf);
}

void pyrup_kernel_u8(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *outData)
{
    const int32_t PU_SZ = 3;
    int32_t bufstep     = ((outWidth + 1) * channels + 16 - 1) / 16 * 16;
    int32_t *buf        = (int32_t *)malloc(bufstep * PU_SZ * sizeof(int32_t));
    memset(buf, 0, bufstep * PU_SZ * sizeof(int32_t));
    int32_t *dtab = (int32_t *)malloc(width * channels * sizeof(int32_t));

    int32_t *rows[3];

    int32_t k, x, sy0 = -PU_SZ / 2, sy = sy0;

    width *= channels;
    outWidth *= channels;

    for (x = 0; x < width; ++x) {
        dtab[x] = (x / channels) * 2 * channels + x % channels;
    }

    for (int32_t y = 0; y < height; ++y) {
        uint8_t *dst0 = outData + (y * 2) * outWidthStride;

        int32_t oy1 = y * 2 + 1;
        if (oy1 >= outHeight) {
            oy1 = outHeight - 1;
        }
        uint8_t *dst1 = outData + oy1 * outWidthStride;
        int32_t *row0, *row1, *row2;

        for (; sy <= y + 1; ++sy) {
            int32_t *row = buf + ((sy - sy0) % PU_SZ) * bufstep;
            int32_t _sy  = borderInterpolate(sy * 2, height * 2) / 2;

            const uint8_t *src = inData + _sy * inWidthStride;

            if (width == channels) {
                for (x = 0; x < channels; ++x) {
                    row[x] = row[x + channels] = src[x] * 8;
                }
                continue;
            }

            for (x = 0; x < channels; ++x) {
                int32_t dx         = dtab[x];
                int32_t t0         = src[x] * 6 + src[x + channels] * 2;
                int32_t t1         = (src[x] + src[x + channels]) * 4;
                row[dx]            = t0;
                row[dx + channels] = t1;
                dx                 = dtab[width - channels + x];
                int32_t sx         = width - channels + x;
                t0                 = src[sx - channels] + src[sx] * 7;
                t1                 = src[sx] * 8;
                row[dx]            = t0;
                row[dx + channels] = t1;

                if (outWidth > width * 2) {
                    row[outWidth - channels + x] = row[dx + channels];
                }
            }

            for (x = channels; x < width - channels; ++x) {
                int32_t dx         = dtab[x];
                int32_t t0         = src[x - channels] + src[x] * 6 + src[x + channels];
                int32_t t1         = (src[x] + src[x + channels]) * 4;
                row[dx]            = t0;
                row[dx + channels] = t1;
            }
        }

        for (k = 0; k < PU_SZ; ++k) {
            rows[k] = buf + ((y - PU_SZ / 2 + k - sy0) % PU_SZ) * bufstep;
        }
        row0 = rows[0];
        row1 = rows[1];
        row2 = rows[2];

        x = 0;
        for (; x < outWidth; ++x) {
            int32_t t1 = (row1[x] + row2[x]) * 4;
            int32_t t0 = (row0[x] + row1[x] * 6 + row2[x]);
            t1 += 1 << 5;
            t0 += 1 << 5;

            dst1[x] = t1 >> 6;
            dst0[x] = t0 >> 6;
        }
    }

    if (outHeight > height * 2) {
        uint8_t *dst0 = outData + (height * 2 - 2) * outWidthStride;
        uint8_t *dst2 = outData + (height * 2) * outWidthStride;

        for (x = 0; x < outWidth; ++x) {
            dst2[x] = dst0[x];
        }
    }

    free(dtab);
    free(buf);
}

template <>
::ppl::common::RetCode PyrUp<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = height * 2;
    int32_t outWidth  = width * 2;
    pyrup_kernel_f32(height, width, 1, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrUp<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = height * 2;
    int32_t outWidth  = width * 2;
    pyrup_kernel_f32(height, width, 3, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrUp<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = height * 2;
    int32_t outWidth  = width * 2;
    pyrup_kernel_f32(height, width, 4, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrUp<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = height * 2;
    int32_t outWidth  = width * 2;
    pyrup_kernel_u8(height, width, 1, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrUp<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = height * 2;
    int32_t outWidth  = width * 2;
    pyrup_kernel_u8(height, width, 3, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrUp<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = height * 2;
    int32_t outWidth  = width * 2;
    pyrup_kernel_u8(height, width, 4, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
