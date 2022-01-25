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

#include "ppl/cv/x86/laplacian.h"
#include "ppl/cv/types.h"
#include <string.h>
#include <cmath>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

inline static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = (data < 0) ? 0 : val;
    return val;
}

void x86laplacian_c_1(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    const uint8_t* inData,
    uint8_t* outData,
    double scale,
    double delta,
    int32_t cn)
{
    for (int32_t i = 0; i < width * cn; ++i) {
        int16_t temp = -4 * inData[i] + inData[i + inWidthStride] * 2;
        if (i < cn) {
            temp += inData[i + cn] * 2;
        } else if (i >= (width - 1) * cn) {
            temp += inData[i - cn] * 2;
        } else {
            temp += inData[i - cn] + inData[i + cn];
        }
        outData[i] = sat_cast(scale * temp + delta);
    }

    for (int32_t i = 1; i < height - 1; ++i) {
        int32_t j;
        for (j = 0; j < cn; ++j) {
            int16_t temp = -4 * inData[i * inWidthStride + j] +
                           2 * inData[i * inWidthStride + j + cn] +
                           inData[(i - 1) * inWidthStride + j] +
                           inData[(i + 1) * inWidthStride + j];
            outData[i * outWidthStride + j] = sat_cast(scale * temp + delta);
        }
        for (j = cn; j < (width - 1) * cn; ++j) {
            int16_t temp = -4 * inData[i * inWidthStride + j] +
                           inData[i * inWidthStride + j - cn] +
                           inData[i * inWidthStride + j + cn] +
                           inData[(i - 1) * inWidthStride + j] +
                           inData[(i + 1) * inWidthStride + j];
            outData[i * outWidthStride + j] = sat_cast(scale * temp + delta);
        }
        for (j = (width - 1) * cn; j < width * cn; ++j) {
            int16_t temp = -4 * inData[i * inWidthStride + j] +
                           2 * inData[i * inWidthStride + j - cn] +
                           inData[(i - 1) * inWidthStride + j] +
                           inData[(i + 1) * inWidthStride + j];
            outData[i * outWidthStride + j] = sat_cast(scale * temp + delta);
        }
    }

    int32_t bottom_offset = (height - 1) * inWidthStride;
    for (int32_t i = 0; i < width * cn; ++i) {
        int16_t temp = -4 * inData[bottom_offset + i] + inData[bottom_offset + i - inWidthStride] * 2;
        if (i < cn) {
            temp += inData[bottom_offset + i + cn] * 2;
        } else if (i >= (width - 1) * cn) {
            temp += inData[bottom_offset + i - cn] * 2;
        } else {
            temp += inData[bottom_offset + i - cn] + inData[bottom_offset + i + cn];
        }
        outData[(height - 1) * outWidthStride + i] = sat_cast(scale * temp + delta);
    }
}

void x86laplacian_c_3(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    const uint8_t* inData,
    uint8_t* outData,
    double scale,
    double delta,
    int32_t cn)
{
    for (int32_t i = 0; i < width * cn; ++i) {
        int16_t temp = -8 * inData[i];
        if (i < cn) {
            temp += inData[i + inWidthStride + cn] * 8;
        } else if (i >= (width - 1) * cn) {
            temp += inData[i + inWidthStride - cn] * 8;
        } else {
            temp += inData[i + inWidthStride - cn] * 4 +
                    inData[i + inWidthStride + cn] * 4;
        }
        outData[i] = sat_cast(scale * temp + delta);
    }

    for (int32_t i = 1; i < height - 1; ++i) {
        int32_t j;
        for (j = 0; j < cn; ++j) {
            int16_t temp = -8 * inData[i * inWidthStride + j] +
                           4 * inData[(i - 1) * inWidthStride + j + cn] +
                           4 * inData[(i + 1) * inWidthStride + j + cn];
            outData[i * outWidthStride + j] = sat_cast(scale * temp + delta);
        }
        for (j = cn; j < (width - 1) * cn; ++j) {
            int16_t temp = -8 * inData[i * inWidthStride + j] +
                           2 * inData[(i - 1) * inWidthStride + j - cn] +
                           2 * inData[(i - 1) * inWidthStride + j + cn] +
                           2 * inData[(i + 1) * inWidthStride + j - cn] +
                           2 * inData[(i + 1) * inWidthStride + j + cn];
            outData[i * outWidthStride + j] = sat_cast(scale * temp + delta);
        }
        for (j = (width - 1) * cn; j < width * cn; ++j) {
            int16_t temp = -8 * inData[i * inWidthStride + j] +
                           4 * inData[(i - 1) * inWidthStride + j - cn] +
                           4 * inData[(i + 1) * inWidthStride + j - cn];
            outData[i * outWidthStride + j] = sat_cast(scale * temp + delta);
        }
    }

    int32_t bottom_offset = (height - 1) * inWidthStride;
    for (int32_t i = 0; i < width * cn; ++i) {
        int16_t temp = -8 * inData[bottom_offset + i];
        if (i < cn) {
            temp += inData[bottom_offset + i - inWidthStride + cn] * 8;
        } else if (i >= (width - 1) * cn) {
            temp += inData[bottom_offset + i - inWidthStride - cn] * 8;
        } else {
            temp += inData[bottom_offset + i - inWidthStride - cn] * 4 +
                    inData[bottom_offset + i - inWidthStride + cn] * 4;
        }
        outData[(height - 1) * outWidthStride + i] = sat_cast(scale * temp + delta);
    }
}

void x86laplacian_f_1(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    const float* inData,
    float* outData,
    double scale,
    double delta,
    int32_t cn)
{
    for (int32_t i = 0; i < width * cn; ++i) {
        float temp = -4 * inData[i] + inData[i + inWidthStride] * 2;
        if (i < cn) {
            temp += inData[i + cn] * 2;
        } else if (i >= (width - 1) * cn) {
            temp += inData[i - cn] * 2;
        } else {
            temp += inData[i - cn] + inData[i + cn];
        }
        outData[i] = scale * temp + delta;
    }

    for (int32_t i = 1; i < height - 1; ++i) {
        int32_t j;
        for (j = 0; j < cn; ++j) {
            float temp = -4 * inData[i * inWidthStride + j] +
                         2 * inData[i * inWidthStride + j + cn] +
                         inData[(i - 1) * inWidthStride + j] +
                         inData[(i + 1) * inWidthStride + j];
            outData[i * outWidthStride + j] = scale * temp + delta;
        }
        for (j = cn; j < (width - 1) * cn; ++j) {
            float temp = -4 * inData[i * inWidthStride + j] +
                         inData[i * inWidthStride + j - cn] +
                         inData[i * inWidthStride + j + cn] +
                         inData[(i - 1) * inWidthStride + j] +
                         inData[(i + 1) * inWidthStride + j];
            outData[i * outWidthStride + j] = scale * temp + delta;
        }
        for (j = (width - 1) * cn; j < width * cn; ++j) {
            float temp = -4 * inData[i * inWidthStride + j] +
                         2 * inData[i * inWidthStride + j - cn] +
                         inData[(i - 1) * inWidthStride + j] +
                         inData[(i + 1) * inWidthStride + j];
            outData[i * outWidthStride + j] = scale * temp + delta;
        }
    }

    int32_t bottom_offset = (height - 1) * inWidthStride;
    for (int32_t i = 0; i < width * cn; ++i) {
        float temp = -4 * inData[bottom_offset + i] + inData[bottom_offset + i - inWidthStride] * 2;
        if (i < cn) {
            temp += inData[bottom_offset + i + cn] * 2;
        } else if (i >= (width - 1) * cn) {
            temp += inData[bottom_offset + i - cn] * 2;
        } else {
            temp += inData[bottom_offset + i - cn] + inData[bottom_offset + i + cn];
        }
        outData[(height - 1) * outWidthStride + i] = scale * temp + delta;
    }
}

void x86laplacian_f_3(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    int32_t outWidthStride,
    const float* inData,
    float* outData,
    double scale,
    double delta,
    int32_t cn)
{
    for (int32_t i = 0; i < width * cn; ++i) {
        float temp = -8 * inData[i];
        if (i < cn) {
            temp += inData[i + inWidthStride + cn] * 8;
        } else if (i >= (width - 1) * cn) {
            temp += inData[i + inWidthStride - cn] * 8;
        } else {
            temp += inData[i + inWidthStride - cn] * 4 +
                    inData[i + inWidthStride + cn] * 4;
        }
        outData[i] = scale * temp + delta;
    }

    for (int32_t i = 1; i < height - 1; ++i) {
        int32_t j;
        for (j = 0; j < cn; ++j) {
            float temp = -8 * inData[i * inWidthStride + j] +
                         4 * inData[(i - 1) * inWidthStride + j + cn] +
                         4 * inData[(i + 1) * inWidthStride + j + cn];
            outData[i * outWidthStride + j] = scale * temp + delta;
        }
        for (j = cn; j < (width - 1) * cn; ++j) {
            float temp = -8 * inData[i * inWidthStride + j] +
                         2 * inData[(i - 1) * inWidthStride + j - cn] +
                         2 * inData[(i - 1) * inWidthStride + j + cn] +
                         2 * inData[(i + 1) * inWidthStride + j - cn] +
                         2 * inData[(i + 1) * inWidthStride + j + cn];
            outData[i * outWidthStride + j] = scale * temp + delta;
        }
        for (j = (width - 1) * cn; j < width * cn; ++j) {
            float temp = -8 * inData[i * inWidthStride + j] +
                         4 * inData[(i - 1) * inWidthStride + j - cn] +
                         4 * inData[(i + 1) * inWidthStride + j - cn];
            outData[i * outWidthStride + j] = scale * temp + delta;
        }
    }

    int32_t bottom_offset = (height - 1) * inWidthStride;
    for (int32_t i = 0; i < width * cn; ++i) {
        float temp = -8 * inData[bottom_offset + i];
        if (i < cn) {
            temp += inData[bottom_offset + i - inWidthStride + cn] * 8;
        } else if (i >= (width - 1) * cn) {
            temp += inData[bottom_offset + i - inWidthStride - cn] * 8;
        } else {
            temp += inData[bottom_offset + i - inWidthStride - cn] * 4 +
                    inData[bottom_offset + i - inWidthStride + cn] * 4;
        }
        outData[(height - 1) * outWidthStride + i] = scale * temp + delta;
    }
}

template <>
::ppl::common::RetCode Laplacian<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 1 || width <= 1 || (ksize != 1 && ksize != 3) || border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    } 
    if (ksize == 1) {
        x86laplacian_c_1(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 1);
    } else if (ksize == 3) {
        x86laplacian_c_3(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 1);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Laplacian<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 1 || width <= 1 || (ksize != 1 && ksize != 3) || border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    } 
    if (ksize == 1) {
        x86laplacian_c_1(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 3);
    } else if (ksize == 3) {
        x86laplacian_c_3(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 3);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Laplacian<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 1 || width <= 1 || (ksize != 1 && ksize != 3) || border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    } 
    if (ksize == 1) {
        x86laplacian_c_1(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 4);
    } else if (ksize == 3) {
        x86laplacian_c_3(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 4);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Laplacian<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 1 || width <= 1 || (ksize != 1 && ksize != 3) || border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    } 
    if (ksize == 1) {
        x86laplacian_f_1(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 1);
    } else if (ksize == 3) {
        x86laplacian_f_3(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 1);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Laplacian<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 1 || width <= 1 || (ksize != 1 && ksize != 3) || border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    } 
    if (ksize == 1) {
        x86laplacian_f_1(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 3);
    } else if (ksize == 3) {
        x86laplacian_f_3(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 3);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Laplacian<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData,
    int32_t ksize,
    double scale,
    double delta,
    BorderType border_type)
{
    if (inData == nullptr || outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 1 || width <= 1 || (ksize != 1 && ksize != 3) || border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    } 
    if (ksize == 1) {
        x86laplacian_f_1(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 4);
    } else if (ksize == 3) {
        x86laplacian_f_3(height, width, inWidthStride, outWidthStride, inData, outData, scale, delta, 4);
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
