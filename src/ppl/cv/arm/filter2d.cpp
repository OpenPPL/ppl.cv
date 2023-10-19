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

#include "ppl/cv/arm/filter2d.h"
#include "ppl/cv/arm/copymakeborder.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <limits.h>
#include <algorithm>
#include <vector>

#include "filter_engine.hpp"

namespace ppl {
namespace cv {
namespace arm {

static uint8_t saturate_cast_f32_u8(float val)
{
    if (val > 255) {
        return 255;
    } else if (val < 0) {
        return 0;
    }

    uint8_t ival = static_cast<uint8_t>(lrintf(val));
    return ival;
}

template <typename ST, typename DT, typename KT>
struct NonSeperableFilter2D {
    const KT *kernel;
    KT delta;
    int kernel_len;
    int kernel_elem_cnt;
    std::vector<ST *> srcKptr;

    NonSeperableFilter2D(const KT *_kernel, int _kernel_len, float _delta) : kernel(_kernel)
    {
        kernel_len = _kernel_len;
        kernel_elem_cnt = kernel_len * kernel_len;
        delta = static_cast<KT>(_delta);
        srcKptr.resize(kernel_len * kernel_len);
    }

    void operator()(ST **src, DT *dst, int dststep, int count, int width, int cn)
    {
        width *= cn;

        ST **skp = srcKptr.data();
        const KT *kval = kernel;

        for (; count > 0; count--, dst += dststep, src++) {
            for (int n = 0; n < kernel_len; n++) {
                for (int m = 0; m < kernel_len; m++) {
                    skp[n * kernel_len + m] = src[n] + m * cn;
                }
            }

            int i = 0;
            for (; i < width; i++) {
                KT s0 = delta;
                for (int k = 0; k < kernel_elem_cnt; k++) {
                    s0 += kval[k] * skp[k][i];
                }
                dst[i] = static_cast<DT>(s0);
            }
        }
    }
};

template <>
struct NonSeperableFilter2D<float, float, float> {
    const float *kernel;
    float delta;
    int kernel_len;
    int kernel_elem_cnt;
    std::vector<float *> srcKptr;

    NonSeperableFilter2D(const float *_kernel, int _kernel_len, float _delta) : kernel(_kernel)
    {
        kernel_len = _kernel_len;
        kernel_elem_cnt = kernel_len * kernel_len;
        delta = static_cast<float>(_delta);
        srcKptr.resize(kernel_len * kernel_len);
    }

    void operator()(float **src, float *dst, int dststep, int count, int width, int cn)
    {
        width *= cn;

        float **skp = srcKptr.data();
        const float *kval = kernel;

        for (; count > 0; count--, dst += dststep, src++) {
            for (int n = 0; n < kernel_len; n++) {
                for (int m = 0; m < kernel_len; m++) {
                    skp[n * kernel_len + m] = src[n] + m * cn;
                }
            }

            int i = 0;
            for (; i <= width - 16; i += 16) {
                float32x4_t vKval = vdupq_n_f32(kval[0]);

                float32x4_t vIn0 = vld1q_f32(skp[0] + i + 0);
                float32x4_t vIn1 = vld1q_f32(skp[0] + i + 4);
                float32x4_t vIn2 = vld1q_f32(skp[0] + i + 8);
                float32x4_t vIn3 = vld1q_f32(skp[0] + i + 12);

                float32x4_t vRes0 = vdupq_n_f32(delta);
                float32x4_t vRes1 = vdupq_n_f32(delta);
                float32x4_t vRes2 = vdupq_n_f32(delta);
                float32x4_t vRes3 = vdupq_n_f32(delta);

                vRes0 = vfmaq_f32(vRes0, vIn0, vKval);
                vRes1 = vfmaq_f32(vRes1, vIn1, vKval);
                vRes2 = vfmaq_f32(vRes2, vIn2, vKval);
                vRes3 = vfmaq_f32(vRes3, vIn3, vKval);

                for (int k = 1; k < kernel_elem_cnt; k++) {
                    vKval = vdupq_n_f32(kval[k]);
                    vIn0 = vld1q_f32(skp[k] + i + 0);
                    vIn1 = vld1q_f32(skp[k] + i + 4);
                    vIn2 = vld1q_f32(skp[k] + i + 8);
                    vIn3 = vld1q_f32(skp[k] + i + 12);

                    vRes0 = vfmaq_f32(vRes0, vIn0, vKval);
                    vRes1 = vfmaq_f32(vRes1, vIn1, vKval);
                    vRes2 = vfmaq_f32(vRes2, vIn2, vKval);
                    vRes3 = vfmaq_f32(vRes3, vIn3, vKval);
                }

                vst1q_f32(dst + i + 0, vRes0);
                vst1q_f32(dst + i + 4, vRes1);
                vst1q_f32(dst + i + 8, vRes2);
                vst1q_f32(dst + i + 12, vRes3);
            }

            for (; i < width; i++) {
                float s0 = delta;
                for (int k = 0; k < kernel_elem_cnt; k++) {
                    s0 += kval[k] * skp[k][i];
                }
                dst[i] = static_cast<float>(s0);
            }
        }
    }
};

template <>
struct NonSeperableFilter2D<uint8_t, uint8_t, float> {
    const float *kernel;
    float delta;
    int kernel_len;
    int kernel_elem_cnt;
    std::vector<uint8_t *> srcKptr;

    NonSeperableFilter2D(const float *_kernel, int _kernel_len, float _delta) : kernel(_kernel)
    {
        kernel_len = _kernel_len;
        kernel_elem_cnt = kernel_len * kernel_len;
        delta = static_cast<float>(_delta);
        srcKptr.resize(kernel_len * kernel_len);
    }

    void operator()(uint8_t **src, uint8_t *dst, int dststep, int count, int width, int cn)
    {
        width *= cn;

        uint8_t **skp = srcKptr.data();
        const float *kval = kernel;

        for (; count > 0; count--, dst += dststep, src++) {
            for (int n = 0; n < kernel_len; n++) {
                for (int m = 0; m < kernel_len; m++) {
                    skp[n * kernel_len + m] = src[n] + m * cn;
                }
            }

            int i = 0;
            for (; i <= width - 16; i += 16) {
                float32x4_t vKval = vdupq_n_f32(kval[0]);

                uint8x16_t vInU8_0 = vld1q_u8(skp[0] + i + 0);
                uint16x8_t vInU16_0 = vmovl_u8(vget_low_u8(vInU8_0));
                uint16x8_t vInU16_1 = vmovl_high_u8(vInU8_0);
                uint32x4_t vInU32_0 = vmovl_u16(vget_low_u16(vInU16_0));
                uint32x4_t vInU32_1 = vmovl_high_u16(vInU16_0);
                uint32x4_t vInU32_2 = vmovl_u16(vget_low_u16(vInU16_1));
                uint32x4_t vInU32_3 = vmovl_high_u16(vInU16_1);
                float32x4_t vIn0 = vcvtq_f32_u32(vInU32_0);
                float32x4_t vIn1 = vcvtq_f32_u32(vInU32_1);
                float32x4_t vIn2 = vcvtq_f32_u32(vInU32_2);
                float32x4_t vIn3 = vcvtq_f32_u32(vInU32_3);

                float32x4_t vRes0 = vdupq_n_f32(delta);
                float32x4_t vRes1 = vdupq_n_f32(delta);
                float32x4_t vRes2 = vdupq_n_f32(delta);
                float32x4_t vRes3 = vdupq_n_f32(delta);

                vRes0 = vfmaq_f32(vRes0, vIn0, vKval);
                vRes1 = vfmaq_f32(vRes1, vIn1, vKval);
                vRes2 = vfmaq_f32(vRes2, vIn2, vKval);
                vRes3 = vfmaq_f32(vRes3, vIn3, vKval);

                for (int k = 1; k < kernel_elem_cnt; k++) {
                    vKval = vdupq_n_f32(kval[k]);
                    vInU8_0 = vld1q_u8(skp[k] + i + 0);
                    vInU16_0 = vmovl_u8(vget_low_u8(vInU8_0));
                    vInU16_1 = vmovl_high_u8(vInU8_0);
                    vInU32_0 = vmovl_u16(vget_low_u16(vInU16_0));
                    vInU32_1 = vmovl_high_u16(vInU16_0);
                    vInU32_2 = vmovl_u16(vget_low_u16(vInU16_1));
                    vInU32_3 = vmovl_high_u16(vInU16_1);
                    vIn0 = vcvtq_f32_u32(vInU32_0);
                    vIn1 = vcvtq_f32_u32(vInU32_1);
                    vIn2 = vcvtq_f32_u32(vInU32_2);
                    vIn3 = vcvtq_f32_u32(vInU32_3);

                    vRes0 = vfmaq_f32(vRes0, vIn0, vKval);
                    vRes1 = vfmaq_f32(vRes1, vIn1, vKval);
                    vRes2 = vfmaq_f32(vRes2, vIn2, vKval);
                    vRes3 = vfmaq_f32(vRes3, vIn3, vKval);
                }

                int32x4_t vResI32_0 = vcvtnq_s32_f32(vRes0);
                int32x4_t vResI32_1 = vcvtnq_s32_f32(vRes1);
                int32x4_t vResI32_2 = vcvtnq_s32_f32(vRes2);
                int32x4_t vResI32_3 = vcvtnq_s32_f32(vRes3);

                uint16x4_t vUhData00 = vqmovun_s32(vResI32_0);
                uint16x8_t vUhData0 = vqmovun_high_s32(vUhData00, vResI32_1);
                uint16x4_t vUhData10 = vqmovun_s32(vResI32_2);
                uint16x8_t vUhData1 = vqmovun_high_s32(vUhData10, vResI32_3);

                uint8x16_t vRes = vqmovn_high_u16(vqmovn_u16(vUhData0), vUhData1);

                vst1q_u8(dst + i + 0, vRes);
            }

            for (; i < width; i++) {
                float s0 = delta;
                for (int k = 0; k < kernel_elem_cnt; k++) {
                    s0 += kval[k] * skp[k][i];
                }
                dst[i] = saturate_cast_f32_u8(s0);
            }
        }
    }
};

template <>
::ppl::common::RetCode Filter2D<float, 1>(int32_t height,
                                          int32_t width,
                                          int32_t inWidthStride,
                                          const float *inData,
                                          int32_t kernel_len,
                                          const float *filter,
                                          int32_t outWidthStride,
                                          float *outData,
                                          float delta,
                                          BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int channels = 1;

    NonSeperableFilter2D<float, float, float> nonSeperableFilter2d(filter, kernel_len, delta);
    FilterEngine<float, float, NonSeperableFilter2D<float, float, float>> engine(
        height, width, channels, kernel_len, border_type, 0, nonSeperableFilter2d);

    engine.process(inData, inWidthStride, outData, outWidthStride);

    return ppl::common::RC_SUCCESS;
}
template <>
::ppl::common::RetCode Filter2D<float, 3>(int32_t height,
                                          int32_t width,
                                          int32_t inWidthStride,
                                          const float *inData,
                                          int32_t kernel_len,
                                          const float *filter,
                                          int32_t outWidthStride,
                                          float *outData,
                                          float delta,
                                          BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int channels = 3;

    NonSeperableFilter2D<float, float, float> nonSeperableFilter2d(filter, kernel_len, delta);
    FilterEngine<float, float, NonSeperableFilter2D<float, float, float>> engine(
        height, width, channels, kernel_len, border_type, 0, nonSeperableFilter2d);

    engine.process(inData, inWidthStride, outData, outWidthStride);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<float, 4>(int32_t height,
                                          int32_t width,
                                          int32_t inWidthStride,
                                          const float *inData,
                                          int32_t kernel_len,
                                          const float *filter,
                                          int32_t outWidthStride,
                                          float *outData,
                                          float delta,
                                          BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int channels = 4;

    NonSeperableFilter2D<float, float, float> nonSeperableFilter2d(filter, kernel_len, delta);
    FilterEngine<float, float, NonSeperableFilter2D<float, float, float>> engine(
        height, width, channels, kernel_len, border_type, 0, nonSeperableFilter2d);

    engine.process(inData, inWidthStride, outData, outWidthStride);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<uint8_t, 1>(int32_t height,
                                            int32_t width,
                                            int32_t inWidthStride,
                                            const uint8_t *inData,
                                            int32_t kernel_len,
                                            const float *filter,
                                            int32_t outWidthStride,
                                            uint8_t *outData,
                                            float delta,
                                            BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int channels = 1;

    NonSeperableFilter2D<uint8_t, uint8_t, float> nonSeperableFilter2d(filter, kernel_len, delta);
    FilterEngine<uint8_t, uint8_t, NonSeperableFilter2D<uint8_t, uint8_t, float>> engine(
        height, width, channels, kernel_len, border_type, 0, nonSeperableFilter2d);

    engine.process(inData, inWidthStride, outData, outWidthStride);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<uint8_t, 3>(int32_t height,
                                            int32_t width,
                                            int32_t inWidthStride,
                                            const uint8_t *inData,
                                            int32_t kernel_len,
                                            const float *filter,
                                            int32_t outWidthStride,
                                            uint8_t *outData,
                                            float delta,
                                            BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int channels = 3;

    NonSeperableFilter2D<uint8_t, uint8_t, float> nonSeperableFilter2d(filter, kernel_len, delta);
    FilterEngine<uint8_t, uint8_t, NonSeperableFilter2D<uint8_t, uint8_t, float>> engine(
        height, width, channels, kernel_len, border_type, 0, nonSeperableFilter2d);

    engine.process(inData, inWidthStride, outData, outWidthStride);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Filter2D<uint8_t, 4>(int32_t height,
                                            int32_t width,
                                            int32_t inWidthStride,
                                            const uint8_t *inData,
                                            int32_t kernel_len,
                                            const float *filter,
                                            int32_t outWidthStride,
                                            uint8_t *outData,
                                            float delta,
                                            BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101 && border_type != ppl::cv::BORDER_REFLECT &&
        border_type != ppl::cv::BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int channels = 4;

    NonSeperableFilter2D<uint8_t, uint8_t, float> nonSeperableFilter2d(filter, kernel_len, delta);
    FilterEngine<uint8_t, uint8_t, NonSeperableFilter2D<uint8_t, uint8_t, float>> engine(
        height, width, channels, kernel_len, border_type, 0, nonSeperableFilter2d);

    engine.process(inData, inWidthStride, outData, outWidthStride);

    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
