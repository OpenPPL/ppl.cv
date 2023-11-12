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

#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/arm/typetraits.hpp"
#include "ppl/cv/types.h"
#include <algorithm>
#include <complex>
#include <string.h>
#include <cfloat>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

namespace {

template <int lane>
void rgb_to_hsv_f32_clac_h(float32x4_t& vMax, float32x4_t& b, float32x4_t& g, float32x4_t& vb_h, float32x4_t& vg_h, float32x4_t& vr_h, float32x4_t& dst_vec)
{
    if (vgetq_lane_f32(vMax, lane) == vgetq_lane_f32(b, lane)) {
        dst_vec = vsetq_lane_f32(vgetq_lane_f32(vb_h, lane), dst_vec, lane);
    } else if (vgetq_lane_f32(vMax, lane) == vgetq_lane_f32(g, lane)) {
        dst_vec = vsetq_lane_f32(vgetq_lane_f32(vg_h, lane), dst_vec, lane);
    } else {
        dst_vec = vsetq_lane_f32(vgetq_lane_f32(vr_h, lane), dst_vec, lane);
    }
}

enum HSV_CONVERT_RGB_TYPE {
    RGB = 0,
    RGBA,
    BGR,
    BGRA
};
enum HsvShifts {
    kHSVSift = 12,
};

const int32_t c_HsvDivTable[256] = {0, 1044480, 522240, 348160, 261120, 208896, 174080, 149211, 130560, 116053, 104448, 94953, 87040, 80345, 74606, 69632, 65280, 61440, 58027, 54973, 52224, 49737, 47476, 45412, 43520, 41779, 40172, 38684, 37303, 36017, 34816, 33693, 32640, 31651, 30720, 29842, 29013, 28229, 27486, 26782, 26112, 25475, 24869, 24290, 23738, 23211, 22706, 22223, 21760, 21316, 20890, 20480, 20086, 19707, 19342, 18991, 18651, 18324, 18008, 17703, 17408, 17123, 16846, 16579, 16320, 16069, 15825, 15589, 15360, 15137, 14921, 14711, 14507, 14308, 14115, 13926, 13743, 13565, 13391, 13221, 13056, 12895, 12738, 12584, 12434, 12288, 12145, 12006, 11869, 11736, 11605, 11478, 11353, 11231, 11111, 10995, 10880, 10768, 10658, 10550, 10445, 10341, 10240, 10141, 10043, 9947, 9854, 9761, 9671, 9582, 9495, 9410, 9326, 9243, 9162, 9082, 9004, 8927, 8852, 8777, 8704, 8632, 8561, 8492, 8423, 8356, 8290, 8224, 8160, 8097, 8034, 7973, 7913, 7853, 7795, 7737, 7680, 7624, 7569, 7514, 7461, 7408, 7355, 7304, 7253, 7203, 7154, 7105, 7057, 7010, 6963, 6917, 6872, 6827, 6782, 6739, 6695, 6653, 6611, 6569, 6528, 6487, 6447, 6408, 6369, 6330, 6292, 6254, 6217, 6180, 6144, 6108, 6073, 6037, 6003, 5968, 5935, 5901, 5868, 5835, 5803, 5771, 5739, 5708, 5677, 5646, 5615, 5585, 5556, 5526, 5497, 5468, 5440, 5412, 5384, 5356, 5329, 5302, 5275, 5249, 5222, 5196, 5171, 5145, 5120, 5095, 5070, 5046, 5022, 4998, 4974, 4950, 4927, 4904, 4881, 4858, 4836, 4813, 4791, 4769, 4748, 4726, 4705, 4684, 4663, 4642, 4622, 4601, 4581, 4561, 4541, 4522, 4502, 4483, 4464, 4445, 4426, 4407, 4389, 4370, 4352, 4334, 4316, 4298, 4281, 4263, 4246, 4229, 4212, 4195, 4178, 4161, 4145, 4128, 4112, 4096};
const int32_t c_HsvDivTable180[256] = {0, 122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192, 7680, 7228, 6827, 6467, 6144, 5851, 5585, 5343, 5120, 4915, 4726, 4551, 4389, 4237, 4096, 3964, 3840, 3724, 3614, 3511, 3413, 3321, 3234, 3151, 3072, 2997, 2926, 2858, 2793, 2731, 2671, 2614, 2560, 2508, 2458, 2409, 2363, 2318, 2276, 2234, 2194, 2156, 2119, 2083, 2048, 2014, 1982, 1950, 1920, 1890, 1862, 1834, 1807, 1781, 1755, 1731, 1707, 1683, 1661, 1638, 1617, 1596, 1575, 1555, 1536, 1517, 1499, 1480, 1463, 1446, 1429, 1412, 1396, 1381, 1365, 1350, 1336, 1321, 1307, 1293, 1280, 1267, 1254, 1241, 1229, 1217, 1205, 1193, 1182, 1170, 1159, 1148, 1138, 1127, 1117, 1107, 1097, 1087, 1078, 1069, 1059, 1050, 1041, 1033, 1024, 1016, 1007, 999, 991, 983, 975, 968, 960, 953, 945, 938, 931, 924, 917, 910, 904, 897, 890, 884, 878, 871, 865, 859, 853, 847, 842, 836, 830, 825, 819, 814, 808, 803, 798, 793, 788, 783, 778, 773, 768, 763, 759, 754, 749, 745, 740, 736, 731, 727, 723, 719, 714, 710, 706, 702, 698, 694, 690, 686, 683, 679, 675, 671, 668, 664, 661, 657, 654, 650, 647, 643, 640, 637, 633, 630, 627, 624, 621, 617, 614, 611, 608, 605, 602, 599, 597, 594, 591, 588, 585, 582, 580, 577, 574, 572, 569, 566, 564, 561, 559, 556, 554, 551, 549, 546, 544, 541, 539, 537, 534, 532, 530, 527, 525, 523, 521, 518, 516, 514, 512, 510, 508, 506, 504, 502, 500, 497, 495, 493, 492, 490, 488, 486, 484, 482};
const int32_t c_HsvSectorData[6][3] = {{1, 3, 0}, {1, 0, 2}, {3, 0, 1}, {0, 2, 1}, {0, 1, 3}, {2, 1, 0}};
} // namespace

template <HSV_CONVERT_RGB_TYPE srcColorType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode rgb_to_hsv_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const uint8_t* srcPtr = src;
    uint8_t* dstPtr = dst;
    typedef typename DT<ncSrc, uint8_t>::vec_DT srcType;
    typedef typename DT<ncDst, uint8_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst;
        uint32_t i = 0;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src = vldx_u8_f32<ncSrc, uint8_t, srcType>(srcPtr + ncSrc * i);
            uint16x8_t r, g, b;
            if (RGB == srcColorType || RGBA == srcColorType) {
                r = vmovl_u8(v_src.val[0]);
                g = vmovl_u8(v_src.val[1]);
                b = vmovl_u8(v_src.val[2]);
            } else {
                b = vmovl_u8(v_src.val[0]);
                g = vmovl_u8(v_src.val[1]);
                r = vmovl_u8(v_src.val[2]);
            }

            uint16x8_t vMax = vmaxq_u16(vmaxq_u16(r, g), b);
            uint16x8_t vMin = vminq_u16(vminq_u16(r, g), b);
            uint16x8_t vDiff = vsubq_u16(vMax, vMin);
            int16x8_t vr = vandq_s16(vreinterpretq_s16_u16(vceqq_u16(vMax, r)), vdupq_n_s16(-1));
            int16x8_t vg = vandq_s16(vreinterpretq_s16_u16(vceqq_u16(vMax, g)), vdupq_n_s16(-1));

            int32x4_t c_HsvDiv_vec_low{c_HsvDivTable[vgetq_lane_u16(vMax, 0)], c_HsvDivTable[vgetq_lane_u16(vMax, 1)], c_HsvDivTable[vgetq_lane_u16(vMax, 2)], c_HsvDivTable[vgetq_lane_u16(vMax, 3)]};
            int32x4_t c_HsvDiv_vec_high{c_HsvDivTable[vgetq_lane_u16(vMax, 4)], c_HsvDivTable[vgetq_lane_u16(vMax, 5)], c_HsvDivTable[vgetq_lane_u16(vMax, 6)], c_HsvDivTable[vgetq_lane_u16(vMax, 7)]};
            int32x4_t hsv_offset = vshlq_n_s32(vdupq_n_s32(1), kHSVSift - 1);
            int32x4_t s_low = vshrq_n_s32(vaddq_s32(vmulq_s32(vmovl_s16(vreinterpret_s16_u16(vget_low_u16(vDiff))), c_HsvDiv_vec_low), hsv_offset), kHSVSift);
            int32x4_t s_high = vshrq_n_s32(vaddq_s32(vmulq_s32(vmovl_s16(vreinterpret_s16_u16(vget_high_u16(vDiff))), c_HsvDiv_vec_high), hsv_offset), kHSVSift);
            uint8x8_t s = vqmovun_s16(vcombine_s16(vmovn_s32(s_low), vmovn_s32(s_high)));

            int16x8_t h_temp = vaddq_s16(vandq_s16(vreinterpretq_s16_u16(vsubq_u16(g, b)), vr), vandq_s16(vmvnq_s16(vr), vaddq_s16(vandq_s16(vg, vaddq_s16(vreinterpretq_s16_u16(vsubq_u16(b, r)), vreinterpretq_s16_u16(vmulq_n_u16(vDiff, 2)))), vandq_s16(vmvnq_s16(vg), vaddq_s16(vreinterpretq_s16_u16(vsubq_u16(r, g)), vreinterpretq_s16_u16(vmulq_n_u16(vDiff, 4)))))));
            int32x4_t c_HsvDiv180_vec_low{c_HsvDivTable180[vgetq_lane_u16(vDiff, 0)], c_HsvDivTable180[vgetq_lane_u16(vDiff, 1)], c_HsvDivTable180[vgetq_lane_u16(vDiff, 2)], c_HsvDivTable180[vgetq_lane_u16(vDiff, 3)]};
            int32x4_t c_HsvDiv180_vec_high{c_HsvDivTable180[vgetq_lane_u16(vDiff, 4)], c_HsvDivTable180[vgetq_lane_u16(vDiff, 5)], c_HsvDivTable180[vgetq_lane_u16(vDiff, 6)], c_HsvDivTable180[vgetq_lane_u16(vDiff, 7)]};
            int32x4_t h_low = vshrq_n_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_low_s16(h_temp)), c_HsvDiv180_vec_low), hsv_offset), kHSVSift);
            h_low = vaddq_s32(h_low, vandq_s32(vreinterpretq_s32_u32(vcltq_s32(h_low, vdupq_n_s32(0))), vdupq_n_s32(180)));
            int32x4_t h_high = vshrq_n_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_high_s16(h_temp)), c_HsvDiv180_vec_high), hsv_offset), kHSVSift);
            h_high = vaddq_s32(h_high, vandq_s32(vreinterpretq_s32_u32(vcltq_s32(h_high, vdupq_n_s32(0))), vdupq_n_s32(180)));
            uint8x8_t h = vqmovun_s16(vcombine_s16(vmovn_s32(h_low), vmovn_s32(h_high)));

            v_dst.val[0] = h;
            v_dst.val[1] = s;
            v_dst.val[2] = vqmovn_u16(vMax);

            vstx_u8_f32<ncDst, uint8_t, dstType>(dstPtr + ncDst * i, v_dst);
        }

        for (; i < width; i++) {
            uint8_t r = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], b = srcPtr[ncSrc * i + 2];

            uint8_t v = std::max(std::max(r, g), b);
            uint8_t vDiff = v - std::min(std::min(r, g), b);
            int32_t vr = (v == r) ? (-1) : 0;
            int32_t vg = (v == g) ? (-1) : 0;

            int32_t s = (vDiff * c_HsvDivTable[v] + (1 << (kHSVSift - 1))) >> kHSVSift;
            int32_t h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * vDiff)) + ((~vg) & (r - g + 4 * vDiff))));
            h = (h * c_HsvDivTable180[vDiff] + (1 << (kHSVSift - 1))) >> kHSVSift;
            h += (h < 0) * 180;

            dstPtr[ncDst * i] = static_cast<uint8_t>(h);
            dstPtr[ncDst * i + 1] = s;
            dstPtr[ncDst * i + 2] = v;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <HSV_CONVERT_RGB_TYPE srcColorType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode rgb_to_hsv_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float32_t* src,
    const int32_t dstStride,
    float32_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const float32_t* srcPtr = src;
    float32_t* dstPtr = dst;
    typedef typename DT<ncSrc, float32_t>::vec_DT srcType;
    typedef typename DT<ncDst, float32_t>::vec_DT dstType;
    for (int32_t k = 0; k < height; k++) {
        dstType v_dst;
        uint32_t i;
        for (i = 0; i <= width - 4; i += 4) {
            srcType v_src = vldx_u8_f32<ncSrc, float32_t, srcType>(srcPtr);
            float32x4_t r, g, b;
            if (RGB == srcColorType || RGBA == srcColorType) {
                r = v_src.val[0];
                g = v_src.val[1];
                b = v_src.val[2];
            } else {
                // BGR BGRA
                b = v_src.val[0];
                g = v_src.val[1];
                r = v_src.val[2];
            }
            float32x4_t vMax = vmaxq_f32(vmaxq_f32(r, g), b);
            float32x4_t vMin = vminq_f32(vminq_f32(r, g), b);
            float32x4_t vDiff = vsubq_f32(vMax, vMin);
            v_dst.val[2] = vMax; // v
            v_dst.val[1] = vdivq_f32(vDiff, vaddq_f32(vMax, vdupq_n_f32(FLT_EPSILON)));
            vDiff = vdivq_f32(vdupq_n_f32(60.0f), vaddq_f32(vDiff, vdupq_n_f32(FLT_EPSILON))); // s
            // fast_op is (a-b)*c
            auto fast_op = [](float32x4_t& a_vec, float32x4_t& b_vec, float32x4_t& c_vec) {
                return vmulq_f32(vsubq_f32(a_vec, b_vec), c_vec);
            };

            float32x4_t vb_h = vaddq_f32(fast_op(r, g, vDiff), vdupq_n_f32(240.0f));
            float32x4_t vg_h = vaddq_f32(fast_op(b, r, vDiff), vdupq_n_f32(120.0f));
            float32x4_t vr_h = fast_op(g, b, vDiff);

            // h
            rgb_to_hsv_f32_clac_h<0>(vMax, b, g, vb_h, vg_h, vr_h, v_dst.val[0]);
            rgb_to_hsv_f32_clac_h<1>(vMax, b, g, vb_h, vg_h, vr_h, v_dst.val[0]);
            rgb_to_hsv_f32_clac_h<2>(vMax, b, g, vb_h, vg_h, vr_h, v_dst.val[0]);
            rgb_to_hsv_f32_clac_h<3>(vMax, b, g, vb_h, vg_h, vr_h, v_dst.val[0]);
            v_dst.val[0] = vaddq_f32(v_dst.val[0], vcvtq_f32_u32(vandq_u32(vcltq_f32(v_dst.val[0], vdupq_n_f32(0)), vdupq_n_u32(360))));
            vst3q_f32(dstPtr, v_dst);
            srcPtr += 4 * ncSrc;
            dstPtr += 4 * ncDst;
        }
        for (; i < width; i++) {
            float32_t r, g, b;
            if (RGB == srcColorType || RGBA == srcColorType) {
                r = srcPtr[0];
                g = srcPtr[1];
                b = srcPtr[2];
            } else {
                // BGR BGRA
                b = srcPtr[0];
                g = srcPtr[1];
                r = srcPtr[2];
            }

            float32_t v = std::max(std::max(r, g), b);
            float32_t diff = v - std::min(std::min(r, g), b);
            float32_t s = diff / (float)(v + FLT_EPSILON);

            diff = (float)(60.0f / (diff + FLT_EPSILON));
            float32_t h;
            if (v == b) {
                h = (r - g) * diff + 240.0f;
            } else if (v == g) {
                h = (b - r) * diff + 120.0f;
            } else {
                h = (g - b) * diff;
            }
            if (h < 0.0f) {
                h += 360.0f;
            }
            dstPtr[0] = h;
            dstPtr[1] = s;
            dstPtr[2] = v;
            dstPtr += ncSrc * 1;
            srcPtr += ncDst * 1;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <HSV_CONVERT_RGB_TYPE srcColorType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode hsv_to_rgb_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const float hscale = 6.f / 180.f;
    const uint8_t* srcPtr = src;
    uint8_t* dstPtr = dst;
    for (int32_t i = 0; i < height; i++) {
        int32_t j = 0;
        
        for (; j < width; j++) {
            float32_t h = static_cast<float32_t>(srcPtr[0]);
            float32_t s = static_cast<float32_t>(srcPtr[1]);
            float32_t v = static_cast<float32_t>(srcPtr[2]);
            s *= (1.f / 255.f);
            v *= (1.f / 255.f);
            float32_t b, g, r;
            b = g = r = v;
            if (s != 0) {
                h *= hscale;

                if (h < 0) {
                    do {
                        h += 6;
                    } while (h < 0);
                } else if (h >= 6) {
                    do {
                        h -= 6;
                    } while (h >= 6);
                }
            }

            int32_t sector = static_cast<int32_t>(h);
            h -= sector;

            if ((unsigned int)sector >= 6u) {
                sector = 0;
                h = 0.f;
            }

            float tab[4];
            tab[0] = v;
            tab[1] = v * (1.f - s);
            tab[2] = v * (1.f - s * h);
            tab[3] = v * (1.f - s * (1.f - h));

            b = tab[c_HsvSectorData[sector][0]];
            g = tab[c_HsvSectorData[sector][1]];
            r = tab[c_HsvSectorData[sector][2]];
            if (RGB == srcColorType || RGBA == srcColorType) {
                dstPtr[0] = (uint8_t)(r * 255.f);
                dstPtr[1] = (uint8_t)(g * 255.f);
                dstPtr[2] = (uint8_t)(b * 255.f);
                if (RGBA == srcColorType) {
                    dstPtr[4] = 255;
                }
            } else {
                // BGR BGRA
                dstPtr[2] = (uint8_t)(r * 255.f);
                dstPtr[1] = (uint8_t)(g * 255.f);
                dstPtr[0] = (uint8_t)(b * 255.f);
                if (BGRA == srcColorType) {
                    dstPtr[4] = 255;
                }
            }
            dstPtr += ncDst;
            srcPtr += ncSrc;
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <HSV_CONVERT_RGB_TYPE srcColorType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode hsv_to_rgb_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float32_t* src,
    const int32_t dstStride,
    float32_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const float32_t* srcPtr = src;
    float32_t* dstPtr = dst;
    typedef typename DT<ncSrc, float32_t>::vec_DT srcType;
    typedef typename DT<ncDst, float32_t>::vec_DT dstType;
    for (int32_t k = 0; k < height; k++) {
        dstType v_dst;
        int32_t i{0};
        
        for (; i < width; i++) {
            float32_t h = srcPtr[0], s = srcPtr[1], v = srcPtr[2];
            float32_t _1_60 = 1.f / 60.f;
            float32_t diff0 = s * v;
            float32_t min0 = v - diff0;
            float32_t h0 = h * _1_60;

            // V = B
            float32_t b0 = v;
            float32_t tmp0 = diff0 * (h0 - 4);
            bool mask0 = h0 < 4;
            float32_t r0 = mask0 ? min0 : (min0 + tmp0);
            float32_t g0 = mask0 ? (min0 - tmp0) : min0;

            // V = G
            tmp0 = diff0 * (h0 - 2);
            mask0 = h0 < 2;
            bool mask1 = h0 < 3;
            g0 = mask1 ? v : g0;
            mask1 = ~mask0 & mask1;
            b0 = mask0 ? min0 : b0;
            b0 = mask1 ? (min0 + tmp0) : b0;
            r0 = mask0 ? (min0 - tmp0) : r0;
            r0 = mask1 ? min0 : r0;

            // V = R
            mask0 = h0 < 1;
            tmp0 = diff0 * h0;
            r0 = mask0 ? v : r0;
            b0 = mask0 ? min0 : b0;
            g0 = mask0 ? (min0 + tmp0) : g0;

            mask0 = h0 >= 5;
            tmp0 = diff0 * (h0 - 6);
            r0 = mask0 ? v : r0;
            g0 = mask0 ? min0 : g0;
            b0 = mask0 ? (min0 - tmp0) : b0;

            if (RGB == srcColorType || RGBA == srcColorType) {
                dstPtr[0] = r0;
                dstPtr[1] = g0;
                dstPtr[2] = b0;
                if (RGBA == srcColorType) {
                    dstPtr[4] = 1.0;
                }
            } else {
                // BGR BGRA
                dstPtr[2] = r0;
                dstPtr[1] = g0;
                dstPtr[0] = b0;
                if (BGRA == srcColorType) {
                    dstPtr[4] = 1.0;
                }
            }

            srcPtr += ncSrc * 1;
            dstPtr += ncDst * 1;
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2HSV<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_hsv_u8<RGB, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode RGBA2HSV<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_hsv_u8<RGBA, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGR2HSV<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_hsv_u8<BGR, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGRA2HSV<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_hsv_u8<BGRA, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return hsv_to_rgb_u8<RGB, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return hsv_to_rgb_u8<RGBA, 3, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return hsv_to_rgb_u8<BGR, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return hsv_to_rgb_u8<BGRA, 3, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode RGB2HSV<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return rgb_to_hsv_f32<RGB, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode RGBA2HSV<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return rgb_to_hsv_f32<RGBA, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGR2HSV<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return rgb_to_hsv_f32<BGR, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGRA2HSV<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return rgb_to_hsv_f32<BGRA, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2RGB<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return hsv_to_rgb_f32<RGB, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2RGBA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return hsv_to_rgb_f32<RGBA, 3, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2BGR<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return hsv_to_rgb_f32<BGR, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode HSV2BGRA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return hsv_to_rgb_f32<BGRA, 3, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm