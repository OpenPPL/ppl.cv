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

#include "ppl/cv/x86/sobel.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "intrinutils.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>
#include <assert.h>
#include <immintrin.h>

namespace ppl {
namespace cv {
namespace x86 {

void getScharrKernels_f32(
    float *kx,
    float *ky,
    int32_t &ksizeX,
    int32_t &ksizeY,
    int32_t dx,
    int32_t dy,
    double scale)
{
    if (dx == 0) {
        kx[0] = 3 * scale;
        kx[1] = 10 * scale;
        kx[2] = 3 * scale;
    } else if (dx == 1) {
        kx[0] = -scale;
        kx[1] = 0;
        kx[2] = scale;
    }

    if (dy == 0) {
        ky[0] = 3 * scale;
        ky[1] = 10 * scale;
        ky[2] = 3 * scale;
    } else if (dy == 1) {
        ky[0] = -scale;
        ky[1] = 0;
        ky[2] = scale;
    }

    ksizeX = 3;
    ksizeY = 3;
}

void getSobelKernels_f32(
    float *kx,
    float *ky,
    int32_t &ksizeX,
    int32_t &ksizeY,
    int32_t dx,
    int32_t dy,
    double scale,
    int32_t ksize)
{
    if (ksize == -1) {
        getScharrKernels_f32(kx, ky, ksizeX, ksizeY, dx, dy, scale);
        return;
    }

    int32_t i, j;

    ksizeX = ksize;
    ksizeY = ksize;
    if (ksizeX == 1 && dx > 0) {
        ksizeX = 3;
    }
    if (ksizeY == 1 && dy > 0) {
        ksizeY = 3;
    }

    float kerI[33];

    for (int32_t k = 0; k < 2; ++k) {
        float *kernel = k == 0 ? kx : ky;
        int32_t order = k == 0 ? dx : dy;
        int32_t ksize = k == 0 ? ksizeX : ksizeY;

        if (ksize == 1) {
            kerI[0] = 1;
        } else if (ksize == 3) {
            if (order == 0) {
                kerI[0] = 1;
                kerI[1] = 2;
                kerI[2] = 1;
            } else if (order == 1) {
                kerI[0] = -1;
                kerI[1] = 0;
                kerI[2] = 1;
            } else {
                kerI[0] = 1;
                kerI[1] = -2;
                kerI[2] = 1;
            }
        } else {
            int32_t oldval, newval;
            kerI[0] = 1;
            for (i = 0; i < ksize; ++i) {
                kerI[i + 1] = 0;
            }
            for (i = 0; i < ksize - order - 1; ++i) {
                oldval = kerI[0];
                for (j = 1; j <= ksize; ++j) {
                    newval      = kerI[j] + kerI[j - 1];
                    kerI[j - 1] = oldval;
                    oldval      = newval;
                }
            }

            for (i = 0; i < order; ++i) {
                oldval = -kerI[0];
                for (j = 1; j <= ksize; ++j) {
                    newval      = kerI[j - 1] - kerI[j];
                    kerI[j - 1] = oldval;
                    oldval      = newval;
                }
            }
        }

        for (i = 0; i < ksize; ++i) {
            kernel[i] = kerI[i] * scale;
        }
    }
}

void sobel_kernel_reflect101_f32(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride,
    const float *inData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    int32_t outWidthStride,
    float *outData)
{
    float kx[33];
    float ky[33];
    int32_t ksizeX = 0;
    int32_t ksizeY = 0;

    getSobelKernels_f32(kx, ky, ksizeX, ksizeY, dx, dy, scale, ksize);

    int32_t x_r = ksizeX / 2;
    int32_t y_r = ksizeY / 2;

    float sum[4] = {};

    for (int32_t i = 0; i < height; ++i) {
        for (int32_t j = 0; j < width; ++j) {
            for (int32_t c = 0; c < channels; ++c) {
                sum[c] = delta;
            }

            for (int32_t ii = -y_r; ii <= y_r; ++ii) {
                for (int32_t jj = -x_r; jj <= x_r; ++jj) {
                    int32_t y = i + ii;
                    int32_t x = j + jj;
                    if (y < 0) {
                        y = -y;
                    }
                    if (x < 0) {
                        x = -x;
                    }
                    if (y >= height) {
                        y = (height - 2 - (y - height));
                    }
                    if (x >= width) {
                        x = (width - 2 - (x - width));
                    }

                    for (int32_t c = 0; c < channels; ++c) {
                        sum[c] += ky[ii + y_r] * kx[jj + x_r] * inData[y * inWidthStride + x * channels + c];
                    }
                }
            }

            for (int32_t c = 0; c < channels; ++c) {
                outData[i * outWidthStride + j * channels + c] = sum[c];
            }
        }
    }
}

void getScharrKernels_u8(
    int16_t *kx,
    int16_t *ky,
    int32_t &ksizeX,
    int32_t &ksizeY,
    int32_t dx,
    int32_t dy,
    double scale)
{
    if (dx == 0) {
        kx[0] = 3 * scale;
        kx[1] = 10 * scale;
        kx[2] = 3 * scale;
    } else if (dx == 1) {
        kx[0] = -scale;
        kx[1] = 0;
        kx[2] = scale;
    }

    if (dy == 0) {
        ky[0] = 3 * scale;
        ky[1] = 10 * scale;
        ky[2] = 3 * scale;
    } else if (dy == 1) {
        ky[0] = -scale;
        ky[1] = 0;
        ky[2] = scale;
    }

    ksizeX = 3;
    ksizeY = 3;
}

void getSobelKernels_u8(
    int16_t *kx,
    int16_t *ky,
    int32_t &ksizeX,
    int32_t &ksizeY,
    int32_t dx,
    int32_t dy,
    double scale,
    int32_t ksize)
{
    if (ksize == -1) {
        getScharrKernels_u8(kx, ky, ksizeX, ksizeY, dx, dy, scale);
        return;
    }

    int32_t i, j;

    ksizeX = ksize;
    ksizeY = ksize;
    if (ksizeX == 1 && dx > 0) {
        ksizeX = 3;
    }
    if (ksizeY == 1 && dy > 0) {
        ksizeY = 3;
    }

    int16_t kerI[33];

    for (int32_t k = 0; k < 2; ++k) {
        int16_t *kernel = k == 0 ? kx : ky;
        int32_t order   = k == 0 ? dx : dy;
        int32_t ksize   = k == 0 ? ksizeX : ksizeY;

        if (ksize == 1) {
            kerI[0] = 1;
        } else if (ksize == 3) {
            if (order == 0) {
                kerI[0] = 1;
                kerI[1] = 2;
                kerI[2] = 1;
            } else if (order == 1) {
                kerI[0] = -1;
                kerI[1] = 0;
                kerI[2] = 1;
            } else {
                kerI[0] = 1;
                kerI[1] = -2;
                kerI[2] = 1;
            }
        } else {
            int32_t oldval, newval;
            kerI[0] = 1;
            for (i = 0; i < ksize; ++i) {
                kerI[i + 1] = 0;
            }
            for (i = 0; i < ksize - order - 1; ++i) {
                oldval = kerI[0];
                for (j = 1; j <= ksize; ++j) {
                    newval      = kerI[j] + kerI[j - 1];
                    kerI[j - 1] = oldval;
                    oldval      = newval;
                }
            }

            for (i = 0; i < order; ++i) {
                oldval = -kerI[0];
                for (j = 1; j <= ksize; ++j) {
                    newval      = kerI[j - 1] - kerI[j];
                    kerI[j - 1] = oldval;
                    oldval      = newval;
                }
            }
        }

        for (i = 0; i < ksize; ++i) {
            kernel[i] = kerI[i] * scale;
        }
    }
}

void sobel_kernel_reflect101_u8(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    int32_t outWidthStride,
    int16_t *outData)
{
    int16_t kx[33];
    int16_t ky[33];
    int32_t ksizeX = 0;
    int32_t ksizeY = 0;

    getSobelKernels_u8(kx, ky, ksizeX, ksizeY, dx, dy, scale, ksize);

    int32_t x_r = ksizeX / 2;
    int32_t y_r = ksizeY / 2;

    int16_t sum[4] = {};

    for (int32_t i = 0; i < height; ++i) {
        for (int32_t j = 0; j < width; ++j) {
            for (int32_t c = 0; c < channels; ++c) {
                sum[c] = delta;
            }

            for (int32_t ii = -y_r; ii <= y_r; ++ii) {
                for (int32_t jj = -x_r; jj <= x_r; ++jj) {
                    int32_t y = i + ii;
                    int32_t x = j + jj;
                    if (y < 0) {
                        y = -y;
                    }
                    if (x < 0) {
                        x = -x;
                    }
                    if (y >= height) {
                        y = (height - 2 - (y - height));
                    }
                    if (x >= width) {
                        x = (width - 2 - (x - width));
                    }

                    for (int32_t c = 0; c < channels; ++c) {
                        sum[c] += ky[ii + y_r] * kx[jj + x_r] * inData[y * inWidthStride + x * channels + c];
                    }
                }
            }

            for (int32_t c = 0; c < channels; ++c) {
                outData[i * outWidthStride + j * channels + c] = sum[c];
            }
        }
    }
}

template <>
::ppl::common::RetCode Sobel<float, float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    assert(border_type == ppl::cv::BORDER_REFLECT_101);
    sobel_kernel_reflect101_f32(height, width, 1, inWidthStride, inData, dx, dy, ksize, scale, delta, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Sobel<float, float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    assert(border_type == ppl::cv::BORDER_REFLECT_101);
    sobel_kernel_reflect101_f32(height, width, 3, inWidthStride, inData, dx, dy, ksize, scale, delta, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Sobel<float, float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    assert(border_type == ppl::cv::BORDER_REFLECT_101);
    sobel_kernel_reflect101_f32(height, width, 4, inWidthStride, inData, dx, dy, ksize, scale, delta, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Sobel<uint8_t, int16_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    int16_t *outData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    assert(border_type == ppl::cv::BORDER_REFLECT_101);
    sobel_kernel_reflect101_u8(height, width, 1, inWidthStride, inData, dx, dy, ksize, scale, delta, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Sobel<uint8_t, int16_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    int16_t *outData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    assert(border_type == ppl::cv::BORDER_REFLECT_101);
    sobel_kernel_reflect101_u8(height, width, 3, inWidthStride, inData, dx, dy, ksize, scale, delta, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Sobel<uint8_t, int16_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    int16_t *outData,
    int32_t dx,
    int32_t dy,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    assert(border_type == ppl::cv::BORDER_REFLECT_101);
    sobel_kernel_reflect101_u8(height, width, 4, inWidthStride, inData, dx, dy, ksize, scale, delta, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
