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
#include <arm_neon.h>
#include <math.h>

namespace ppl {
namespace cv {
namespace arm {

namespace {
enum COLOR_LAB_RGB_TYPE {
    RGB = 0,
    RGBA,
    BGR,
    BGRA
};

enum LABShifts {
    kGammaTabSize = 1024,
    kLabShift = 12,
    kGammaShift = 3,
    kLabShift2 = (kLabShift + kGammaShift),
};

const ushort c_sRGBGammaTab_b[] = {0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 28, 29, 31, 33, 34, 36, 38, 40, 41, 43, 45, 47, 49, 51, 54, 56, 58, 60, 63, 65, 68, 70, 73, 75, 78, 81, 83, 86, 89, 92, 95, 98, 101, 105, 108, 111, 115, 118, 121, 125, 129, 132, 136, 140, 144, 147, 151, 155, 160, 164, 168, 172, 176, 181, 185, 190, 194, 199, 204, 209, 213, 218, 223, 228, 233, 239, 244, 249, 255, 260, 265, 271, 277, 282, 288, 294, 300, 306, 312, 318, 324, 331, 337, 343, 350, 356, 363, 370, 376, 383, 390, 397, 404, 411, 418, 426, 433, 440, 448, 455, 463, 471, 478, 486, 494, 502, 510, 518, 527, 535, 543, 552, 560, 569, 578, 586, 595, 604, 613, 622, 631, 641, 650, 659, 669, 678, 688, 698, 707, 717, 727, 737, 747, 757, 768, 778, 788, 799, 809, 820, 831, 842, 852, 863, 875, 886, 897, 908, 920, 931, 943, 954, 966, 978, 990, 1002, 1014, 1026, 1038, 1050, 1063, 1075, 1088, 1101, 1113, 1126, 1139, 1152, 1165, 1178, 1192, 1205, 1218, 1232, 1245, 1259, 1273, 1287, 1301, 1315, 1329, 1343, 1357, 1372, 1386, 1401, 1415, 1430, 1445, 1460, 1475, 1490, 1505, 1521, 1536, 1551, 1567, 1583, 1598, 1614, 1630, 1646, 1662, 1678, 1695, 1711, 1728, 1744, 1761, 1778, 1794, 1811, 1828, 1846, 1863, 1880, 1897, 1915, 1933, 1950, 1968, 1986, 2004, 2022, 2040};

template <typename T>
inline const uint8_t sat_cast_u8(T data)
{
    return data > 255 ? 255 : (data < 0 ? 0 : data);
}

} // namespace

template <COLOR_LAB_RGB_TYPE rgbType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode rgb_to_lab_u8(
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

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;

    const int Lscale = (116 * 255 + 50) / 100;
    const int Lshift = -((16 * 255 * (1 << kLabShift2) + 50) / 100);
    for (int32_t k = 0; k < height; k++) {
        int32_t i = 0;
        // todo: 向量化计算
        for (; i < width; i++) {
            int32_t R, G, B;
            if (COLOR_LAB_RGB_TYPE::RGB == rgbType || COLOR_LAB_RGB_TYPE::RGBA == rgbType) {
                R = static_cast<int32_t>(srcPtr[0]);
                G = static_cast<int32_t>(srcPtr[1]);
                B = static_cast<int32_t>(srcPtr[2]);
            } else {
                B = static_cast<int32_t>(srcPtr[0]);
                G = static_cast<int32_t>(srcPtr[1]);
                R = static_cast<int32_t>(srcPtr[2]);
            }

            // process R G B ---> lab
            B = c_sRGBGammaTab_b[B];
            G = c_sRGBGammaTab_b[G];
            R = c_sRGBGammaTab_b[R];

            int divideUp1_X = (B * 778 + G * 1541 + R * 1777 + (1 << (kLabShift - 1))) >> kLabShift;
            int divideUp1_Y = (B * 296 + G * 2929 + R * 871 + (1 << (kLabShift - 1))) >> kLabShift;
            int divideUp1_Z = (B * 3575 + G * 448 + R * 73 + (1 << (kLabShift - 1))) >> kLabShift;

            float32_t tmp_x = divideUp1_X * (1.f / (255.f * (1 << kGammaShift)));
            int fX = (1 << kLabShift2) * (tmp_x < 0.008856f ? tmp_x * 7.787f + 0.13793103448275862f : std::cbrtf(tmp_x));
            float32_t tmp_y = divideUp1_Y * (1.f / (255.f * (1 << kGammaShift)));
            int fY = (1 << kLabShift2) * (tmp_y < 0.008856f ? tmp_y * 7.787f + 0.13793103448275862f : std::cbrtf(tmp_y));
            float32_t tmp_z = divideUp1_Z * (1.f / (255.f * (1 << kGammaShift)));
            int fZ = (1 << kLabShift2) * (tmp_z < 0.008856f ? tmp_z * 7.787f + 0.13793103448275862f : std::cbrtf(tmp_z));

            int L = (Lscale * fY + Lshift + (1 << (kLabShift2 - 1))) >> kLabShift2;
            int a = (500 * (fX - fY) + 128 * (1 << kLabShift2) + (1 << (kLabShift2 - 1))) >> kLabShift2;
            int b = (200 * (fY - fZ) + 128 * (1 << kLabShift2) + (1 << (kLabShift2 - 1))) >> kLabShift2;

            // write lab to dst
            dstPtr[0] = sat_cast_u8(L);
            dstPtr[1] = sat_cast_u8(a);
            dstPtr[2] = sat_cast_u8(b);
            dstPtr += ncDst * 1;
            srcPtr += ncSrc * 1;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <COLOR_LAB_RGB_TYPE rgbType, int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode lab_to_rgb_u8(
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

    float32_t _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float32_t lThresh = 7.9996248f; // 0.008856f * 903.3f;
    float32_t fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

    const uint8_t* srcPtr = src;
    uint8_t* dstPtr = dst;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++) {
        int32_t i = 0;
        // todo: 向量化计算
        for (; i < width; i++) {
            float32_t L, a, b;
            L = static_cast<float32_t>(srcPtr[0]) * 0.392156863f; // (100.f / 255.f);
            a = static_cast<float32_t>(srcPtr[1]) - 128;
            b = static_cast<float32_t>(srcPtr[2]) - 128;

            float32_t Y, fy;

            if (L <= lThresh) {
                Y = L / 903.3f;
                fy = 7.787f * Y + _16_116;
            } else {
                fy = (L + 16.0f) / 116.0f;
                Y = fy * fy * fy;
            }

            float32_t X = a / 500.0f + fy;
            float32_t Z = fy - b / 200.0f;

            if (X <= fThresh) {
                X = (X - _16_116) / 7.787f;
            } else {
                X = X * X * X;
            }

            if (Z <= fThresh) {
                Z = (Z - _16_116) / 7.787f;
            } else {
                Z = Z * Z * Z;
            }

            float32_t R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;
            float32_t G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
            float32_t B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;

            R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;
            G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
            B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;

            R = R * 255.f;
            G = G * 255.f;
            B = B * 255.f;

            // write RGB to dst
            if (COLOR_LAB_RGB_TYPE::RGB == rgbType) {
                dstPtr[0] = sat_cast_u8(R);
                dstPtr[1] = sat_cast_u8(G);
                dstPtr[2] = sat_cast_u8(B);
            } else if (COLOR_LAB_RGB_TYPE::RGBA == rgbType) {
                dstPtr[0] = sat_cast_u8(R);
                dstPtr[1] = sat_cast_u8(G);
                dstPtr[2] = sat_cast_u8(B);
                dstPtr[3] = 255;
            } else if (COLOR_LAB_RGB_TYPE::BGR == rgbType) {
                dstPtr[0] = sat_cast_u8(B);
                dstPtr[1] = sat_cast_u8(G);
                dstPtr[2] = sat_cast_u8(R);
            } else if (COLOR_LAB_RGB_TYPE::BGRA == rgbType) {
                dstPtr[0] = sat_cast_u8(B);
                dstPtr[1] = sat_cast_u8(G);
                dstPtr[2] = sat_cast_u8(R);
                dstPtr[3] = 255;
            }
            dstPtr += ncDst * 1;
            srcPtr += ncSrc * 1;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2LAB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_lab_u8<RGB, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode RGBA2LAB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_lab_u8<RGBA, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode BGR2LAB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_lab_u8<BGR, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode BGRA2LAB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return rgb_to_lab_u8<BGRA, 4, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
// fzx
template <>
::ppl::common::RetCode LAB2RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return lab_to_rgb_u8<RGB, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode LAB2RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return lab_to_rgb_u8<RGBA, 3, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode LAB2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return lab_to_rgb_u8<BGR, 3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode LAB2BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return lab_to_rgb_u8<BGRA, 3, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm