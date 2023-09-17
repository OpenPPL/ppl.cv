// // to you under the Apache License, Version 2.0 (the
// // "License"); you may not use this file except in compliance
// // with the License.  You may obtain a copy of the License at
// //
// //   http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing,
// // software distributed under the License is distributed on an
// // "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// // KIND, either express or implied.  See the License for the
// // specific language governing permissions and limitations
// // under the License.

// #include "ppl/cv/arm/cvtcolor.h"
// #include "ppl/cv/arm/typetraits.hpp"
// #include "ppl/cv/types.h"
// #include <algorithm>
// #include <complex>
// #include <string.h>
// #include <arm_neon.h>

// namespace ppl {
// namespace cv {
// namespace arm {

// enum COLOR_LAB_RGB_TYPE{
//     RGB = 0,
//     RGBA,
//     BGR,
//     BGRA
// }

// template <COLOR_LAB_RGB_TYPE rgbType,int32_t ncSrc, int32_t ncDst>
// ::ppl::common::RetCode rgb_to_lab_u8(
//     const int32_t height,
//     const int32_t width,
//     const int32_t srcStride,
//     const uint8_t* src,
//     const int32_t dstStride,
//     uint8_t* dst)
// {
//     if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
//         return ppl::common::RC_INVALID_VALUE;
//     }
//     for(){
//         for(){
//             uint8_t r = ;
//             uint8_t g = ;
//             uint8_t b = ; 

//             //process r g b ---> lab

//             //write lab to dst

//         }

//     }



// }

// template <COLOR_LAB_RGB_TYPE rgbType,int32_t ncSrc, int32_t ncDst>
// ::ppl::common::RetCode lab_to_rgb_u8(
//     const int32_t height,
//     const int32_t width,
//     const int32_t srcStride,
//     const uint8_t* src,
//     const int32_t dstStride,
//     uint8_t* dst)
// {
//     if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
//         return ppl::common::RC_INVALID_VALUE;
//     }

// }

// // template <COLOR_LAB_RGB_TYPE rgbType,int32_t ncSrc, int32_t ncDst>
// // ::ppl::common::RetCode rgb_to_lab_u8(
// //     const int32_t height,
// //     const int32_t width,
// //     const int32_t srcStride,
// //     const uint8_t* src,
// //     const int32_t dstStride,
// //     uint8_t* dst)
// // {


// // }



// template <>
// ::ppl::common::RetCode RGB2LAB<uint8_t>(
//     int32_t height,
//     int32_t width,
//     int32_t inWidthStride,
//     const uint8_t* inData,
//     int32_t outWidthStride,
//     uint8_t* outData)
// {
//     return rgb_to_lab_u8<3, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
// }


// }
// }
// } // namespace ppl::cv::arm