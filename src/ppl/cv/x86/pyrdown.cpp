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

#include "ppl/cv/x86/pyrdown.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <string.h>
#include <cmath>

namespace ppl {
namespace cv {
namespace x86 {

void pyrdown_kernel_f32(
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
    const int32_t PD_SZ = 5;
    int32_t bufstep     = (outWidth * channels + 16 - 1) / 16 * 16;
    float *buf          = (float *)malloc(bufstep * PD_SZ * sizeof(float));
    memset(buf, 0, bufstep * PD_SZ * sizeof(float));
    int32_t tabL[4 * (PD_SZ + 2)], tabR[4 * (PD_SZ + 2)];
    int32_t *tabM = (int32_t *)malloc(outWidth * channels * sizeof(int32_t));

    float *rows[PD_SZ];

    int32_t k, x, sy0 = -PD_SZ / 2, sy = sy0;
    int32_t width0 = outWidth;
    if ((width - PD_SZ / 2 - 1) / 2 + 1 < width0) {
        width0 = (width - PD_SZ / 2 - 1) / 2 + 1;
    }

    float r = 1.0f / 256.0f;

    for (x = 0; x <= PD_SZ + 1; ++x) {
        int32_t sx0 = borderInterpolate(x - PD_SZ / 2, width) * channels;
        int32_t sx1 = borderInterpolate(x + width0 * 2 - PD_SZ / 2, width) * channels;
        for (k = 0; k < channels; ++k) {
            tabL[x * channels + k] = sx0 + k;
            tabR[x * channels + k] = sx1 + k;
        }
    }

    width *= channels;
    outWidth *= channels;
    width0 *= channels;

    for (x = 0; x < outWidth; ++x) {
        tabM[x] = (x / channels) * 2 * channels + x % channels;
    }

    for (int32_t y = 0; y < outHeight; ++y) {
        float *dst = outData + y * outWidthStride;

        for (; sy <= y * 2 + 2; ++sy) {
            float *row         = buf + ((sy - sy0) % PD_SZ) * bufstep;
            int32_t _sy        = borderInterpolate(sy, height);
            const float *src   = inData + _sy * inWidthStride;
            int32_t limit      = channels;
            const int32_t *tab = tabL;

            for (x = 0;;) {
                for (; x < limit; ++x) {
                    row[x] = src[tab[x + channels * 2]] * 6 + (src[tab[x + channels]] + src[tab[x + channels * 3]]) * 4 +
                             src[tab[x]] + src[tab[x + channels * 4]];
                }
                if (x == outWidth) {
                    break;
                }

                if (channels == 1) {
                    for (; x < width0; ++x) {
                        row[x] = src[x * 2] * 6 + (src[x * 2 - 1] + src[x * 2 + 1]) * 4 +
                                 src[x * 2 - 2] + src[x * 2 + 2];
                    }
                } else if (channels == 2) {
                    for (; x < width0; x += 2) {
                        const float *s = src + x * 2;
                        float t0       = s[0] * 6 + (s[-2] + s[2]) * 4 + s[-4] + s[4];
                        float t1       = s[1] * 6 + (s[-1] + s[3]) * 4 + s[-3] + s[5];
                        row[x]         = t0;
                        row[x + 1]     = t1;
                    }
                } else if (channels == 3) {
                    for (; x < width0; x += 3) {
                        const float *s = src + x * 2;
                        float t0       = s[0] * 6 + (s[-3] + s[3]) * 4 + s[-6] + s[6];
                        float t1       = s[1] * 6 + (s[-2] + s[4]) * 4 + s[-5] + s[7];
                        float t2       = s[2] * 6 + (s[-1] + s[5]) * 4 + s[-4] + s[8];
                        row[x]         = t0;
                        row[x + 1]     = t1;
                        row[x + 2]     = t2;
                    }
                } else if (channels == 4) {
                    for (; x < width0; x += 4) {
                        const float *s = src + x * 2;
                        float t0       = s[0] * 6 + (s[-4] + s[4]) * 4 + s[-8] + s[8];
                        float t1       = s[1] * 6 + (s[-3] + s[5]) * 4 + s[-7] + s[9];
                        float t2       = s[2] * 6 + (s[-2] + s[6]) * 4 + s[-6] + s[10];
                        float t3       = s[3] * 6 + (s[-1] + s[7]) * 4 + s[-5] + s[11];
                        row[x]         = t0;
                        row[x + 1]     = t1;
                        row[x + 2]     = t2;
                        row[x + 3]     = t3;
                    }
                } else {
                    for (; x < width0; ++x) {
                        int32_t sx = tabM[x];
                        row[x]     = src[sx] * 6 + (src[sx - channels] + src[sx + channels]) * 4 +
                                 src[sx - channels * 2] + src[sx + channels * 2];
                    }
                }

                limit = outWidth;
                tab   = tabR - x;
            }
        }

        for (k = 0; k < PD_SZ; ++k) {
            rows[k] = buf + ((y * 2 - PD_SZ / 2 + k - sy0) % PD_SZ) * bufstep;
        }
        x = 0;
        for (; x < outWidth; ++x) {
            dst[x] = (rows[2][x] * 6 + (rows[1][x] + rows[3][x]) * 4 + rows[0][x] + rows[4][x]) * r;
        }
    }
    free(tabM);
    ppl::common::AlignedFree(buf);
}

void pyrdown_kernel_u8(
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
    const int32_t PD_SZ = 5;
    int32_t bufstep     = (outWidth * channels + 16 - 1) / 16 * 16;
    int32_t *buf        = (int32_t *)malloc(bufstep * PD_SZ * sizeof(int32_t));
    memset(buf, 0, bufstep * PD_SZ * sizeof(int32_t));
    int32_t tabL[4 * (PD_SZ + 2)], tabR[4 * (PD_SZ + 2)];
    int32_t *tabM = (int32_t *)malloc(outWidth * channels * sizeof(int32_t));

    int32_t *rows[PD_SZ];

    int32_t k, x, sy0 = -PD_SZ / 2, sy = sy0;
    int32_t width0 = outWidth;
    if ((width - PD_SZ / 2 - 1) / 2 + 1 < width0) {
        width0 = (width - PD_SZ / 2 - 1) / 2 + 1;
    }

    for (x = 0; x <= PD_SZ + 1; ++x) {
        int32_t sx0 = borderInterpolate(x - PD_SZ / 2, width) * channels;
        int32_t sx1 = borderInterpolate(x + width0 * 2 - PD_SZ / 2, width) * channels;
        for (k = 0; k < channels; ++k) {
            tabL[x * channels + k] = sx0 + k;
            tabR[x * channels + k] = sx1 + k;
        }
    }

    width *= channels;
    outWidth *= channels;
    width0 *= channels;

    for (x = 0; x < outWidth; ++x) {
        tabM[x] = (x / channels) * 2 * channels + x % channels;
    }

    for (int32_t y = 0; y < outHeight; ++y) {
        uint8_t *dst = outData + y * outWidthStride;

        for (; sy <= y * 2 + 2; ++sy) {
            int32_t *row       = buf + ((sy - sy0) % PD_SZ) * bufstep;
            int32_t _sy        = borderInterpolate(sy, height);
            const uint8_t *src = inData + _sy * inWidthStride;
            int32_t limit      = channels;
            const int32_t *tab = tabL;

            for (x = 0;;) {
                for (; x < limit; ++x) {
                    row[x] = src[tab[x + channels * 2]] * 6 + (src[tab[x + channels]] + src[tab[x + channels * 3]]) * 4 +
                             src[tab[x]] + src[tab[x + channels * 4]];
                }
                if (x == outWidth) {
                    break;
                }

                if (channels == 1) {
                    for (; x < width0; ++x) {
                        row[x] = src[x * 2] * 6 + (src[x * 2 - 1] + src[x * 2 + 1]) * 4 +
                                 src[x * 2 - 2] + src[x * 2 + 2];
                    }
                } else if (channels == 2) {
                    for (; x < width0; x += 2) {
                        const uint8_t *s = src + x * 2;
                        int32_t t0       = s[0] * 6 + (s[-2] + s[2]) * 4 + s[-4] + s[4];
                        int32_t t1       = s[1] * 6 + (s[-1] + s[3]) * 4 + s[-3] + s[5];
                        row[x]           = t0;
                        row[x + 1]       = t1;
                    }
                } else if (channels == 3) {
                    for (; x < width0; x += 3) {
                        const uint8_t *s = src + x * 2;
                        int32_t t0       = s[0] * 6 + (s[-3] + s[3]) * 4 + s[-6] + s[6];
                        int32_t t1       = s[1] * 6 + (s[-2] + s[4]) * 4 + s[-5] + s[7];
                        int32_t t2       = s[2] * 6 + (s[-1] + s[5]) * 4 + s[-4] + s[8];
                        row[x]           = t0;
                        row[x + 1]       = t1;
                        row[x + 2]       = t2;
                    }
                } else if (channels == 4) {
                    for (; x < width0; x += 4) {
                        const uint8_t *s = src + x * 2;
                        int32_t t0       = s[0] * 6 + (s[-4] + s[4]) * 4 + s[-8] + s[8];
                        int32_t t1       = s[1] * 6 + (s[-3] + s[5]) * 4 + s[-7] + s[9];
                        int32_t t2       = s[2] * 6 + (s[-2] + s[6]) * 4 + s[-6] + s[10];
                        int32_t t3       = s[3] * 6 + (s[-1] + s[7]) * 4 + s[-5] + s[11];
                        row[x]           = t0;
                        row[x + 1]       = t1;
                        row[x + 2]       = t2;
                        row[x + 3]       = t3;
                    }
                } else {
                    for (; x < width0; ++x) {
                        int32_t sx = tabM[x];
                        row[x]     = src[sx] * 6 + (src[sx - channels] + src[sx + channels]) * 4 +
                                 src[sx - channels * 2] + src[sx + channels * 2];
                    }
                }

                limit = outWidth;
                tab   = tabR - x;
            }
        }

        for (k = 0; k < PD_SZ; ++k) {
            rows[k] = buf + ((y * 2 - PD_SZ / 2 + k - sy0) % PD_SZ) * bufstep;
        }
        x = 0;
        for (; x < outWidth; ++x) {
            int32_t temp = (rows[2][x] * 6 + (rows[1][x] + rows[3][x]) * 4 + rows[0][x] + rows[4][x]);
            temp += (1 << 7);
            dst[x] = temp >> 8;
        }
    }
    free(tabM);
    ppl::common::AlignedFree(buf);
}

template <>
::ppl::common::RetCode PyrDown<float, 1>(
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
    if (height <= 0 || width <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = (height + 1) / 2;
    int32_t outWidth  = (width + 1) / 2;
    pyrdown_kernel_f32(height, width, 1, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrDown<float, 3>(
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
    if (height <= 0 || width <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = (height + 1) / 2;
    int32_t outWidth  = (width + 1) / 2;
    pyrdown_kernel_f32(height, width, 3, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrDown<float, 4>(
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
    if (height <= 0 || width <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = (height + 1) / 2;
    int32_t outWidth  = (width + 1) / 2;
    pyrdown_kernel_f32(height, width, 4, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrDown<uint8_t, 1>(
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
    if (height <= 0 || width <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = (height + 1) / 2;
    int32_t outWidth  = (width + 1) / 2;
    pyrdown_kernel_u8(height, width, 1, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrDown<uint8_t, 3>(
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
    if (height <= 0 || width <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = (height + 1) / 2;
    int32_t outWidth  = (width + 1) / 2;
    pyrdown_kernel_u8(height, width, 3, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode PyrDown<uint8_t, 4>(
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
    if (height <= 0 || width <= 0 || inWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t outHeight = (height + 1) / 2;
    int32_t outWidth  = (width + 1) / 2;
    pyrdown_kernel_u8(height, width, 4, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
