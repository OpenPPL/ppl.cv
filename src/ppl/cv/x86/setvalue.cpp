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

#include "ppl/cv/x86/setvalue.h"
#include "ppl/cv/types.h"
#include <string.h>
#include <cmath>
#include <type_traits>
#include <limits.h>

namespace ppl {
namespace cv {
namespace x86 {

template <typename T, int32_t numOutChannels, int32_t numMaskChannels>
::ppl::common::RetCode SetTo(
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T *outData,
    const T value,
    int32_t maskWidthStride,
    const uint8_t *mask) {
    if (outHeight <= 0 || outWidth <= 0 || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (outData == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (numMaskChannels != 1 && numMaskChannels != numOutChannels) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (mask == nullptr && maskWidthStride != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (mask == nullptr) {
        for (int32_t i = 0; i < outHeight; ++i) {
            for (int32_t j = 0; j < (outWidth * numOutChannels); ++j) {
                outData[i * outWidthStride + j] = value;
            }
        }
    } else {
        if (numMaskChannels == 1) {
            for (int32_t i = 0; i < outHeight; ++i) {
                for (int32_t j = 0; j < outWidth; ++j) {
                    bool maskOn = mask[i * maskWidthStride + j] != 0;
                    if (maskOn) {
                        for (int32_t k = 0; k < numOutChannels; ++k) {
                            outData[i * outWidthStride + j * numOutChannels + k] = value;
                        }
                    }
                }
            }
        } else {
            for (int32_t i = 0; i < outHeight; ++i) {
                for (int32_t j = 0; j < (outWidth * numOutChannels); ++j) {
                    bool maskOn = mask[i * maskWidthStride + j] != 0;
                    if (maskOn) {
                        outData[i * outWidthStride + j] = value;
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode SetTo<uint8_t, 1, 1>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *outData, const uint8_t value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<uint8_t, 3, 1>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *outData, const uint8_t value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<uint8_t, 3, 3>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *outData, const uint8_t value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<uint8_t, 4, 1>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *outData, const uint8_t value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<uint8_t, 4, 4>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *outData, const uint8_t value, int32_t maskWidthStride, const uint8_t *mask);

template ::ppl::common::RetCode SetTo<float, 1, 1>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *outData, const float value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<float, 3, 1>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *outData, const float value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<float, 3, 3>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *outData, const float value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<float, 4, 1>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *outData, const float value, int32_t maskWidthStride, const uint8_t *mask);
template ::ppl::common::RetCode SetTo<float, 4, 4>(int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *outData, const float value, int32_t maskWidthStride, const uint8_t *mask);

template <typename T, int32_t numChannels>
::ppl::common::RetCode Ones(
    int32_t height,
    int32_t width,
    int32_t stride,
    T *out) {
    if (height <= 0 || width <= 0 || stride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (out == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (std::is_same<T, uint8_t>::value) {
        if (numChannels == 1) {
            for (int32_t i = 0; i < height; ++i) {
                memset(out + i * stride, 1, sizeof(T) * width * numChannels);
            }
        } else {
            memset(out, 0, sizeof(T) * height * width * numChannels);
            for (int32_t i = 0; i < height * width * numChannels; i += numChannels) {
                out[i] = 1;
            }
        }
    } else {
        if (numChannels == 1) {
            SetTo<T, numChannels>(height, width, stride, out, 1, 0, nullptr);
        } else {
            memset(out, 0, sizeof(T) * height * width * numChannels);
            for (int32_t i = 0; i < height * width * numChannels; i += numChannels) {
                out[i] = 1;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Ones<uint8_t, 1>(int32_t height, int32_t width, int32_t stride, uint8_t *out);
template ::ppl::common::RetCode Ones<uint8_t, 3>(int32_t height, int32_t width, int32_t stride, uint8_t *out);
template ::ppl::common::RetCode Ones<uint8_t, 4>(int32_t height, int32_t width, int32_t stride, uint8_t *out);

template ::ppl::common::RetCode Ones<float, 1>(int32_t height, int32_t width, int32_t stride, float *out);
template ::ppl::common::RetCode Ones<float, 3>(int32_t height, int32_t width, int32_t stride, float *out);
template ::ppl::common::RetCode Ones<float, 4>(int32_t height, int32_t width, int32_t stride, float *out);

template <typename T, int32_t numChannels>
::ppl::common::RetCode Zeros(
    int32_t height,
    int32_t width,
    int32_t stride,
    T *out) {
    if (height <= 0 || width <= 0 || stride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (out == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    for (int32_t i = 0; i < height; ++i) {
        memset(out + i * stride, 0, sizeof(T) * width * numChannels);
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Zeros<uint8_t, 1>(int32_t height, int32_t width, int32_t stride, uint8_t *out);
template ::ppl::common::RetCode Zeros<uint8_t, 3>(int32_t height, int32_t width, int32_t stride, uint8_t *out);
template ::ppl::common::RetCode Zeros<uint8_t, 4>(int32_t height, int32_t width, int32_t stride, uint8_t *out);

template ::ppl::common::RetCode Zeros<float, 1>(int32_t height, int32_t width, int32_t stride, float *out);
template ::ppl::common::RetCode Zeros<float, 3>(int32_t height, int32_t width, int32_t stride, float *out);
template ::ppl::common::RetCode Zeros<float, 4>(int32_t height, int32_t width, int32_t stride, float *out);

}
}
} // namespace ppl::cv::x86
