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
#include "color_yuv_simd.hpp"
#include <algorithm>
#include <complex>
#include <string.h>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

template <YUV_TYPE yuvType>
void nv_to_i420_uchar_video_range(
    int32_t h,
    int32_t w,
    int32_t yStride,
    const uint8_t* y_ptr,
    int32_t uStride,
    const uint8_t* u_ptr,
    int32_t vStride,
    const uint8_t* v_ptr,
    int32_t i420Stride,
    uint8_t* i420_ptr)
{
    // int32_t remain = w >= 15 ? w - 15 : 0;
    const uint8_t* yptr = y_ptr;
    uint8_t* dst_y_ptr = i420_ptr;
    uint8_t* dst_u_ptr = dst_y_ptr + w * h;
    uint8_t* dst_v_ptr = dst_y_ptr + w * h + w * h / 4;
    int32_t uv_w = w / 2;

    for (int32_t i = 0; i < h; i++) {
        // process y
        memcpy(dst_y_ptr, yptr, w);
        yptr += w;
        dst_y_ptr += w;
        // process u v
        const uint8_t* uvptr = (nullptr == u_ptr) ? v_ptr : u_ptr;
        int j = 0;
        for (; j < (uv_w - 16); j += 16) {
            typedef typename DT<2, uint8_t>::vec_DT srcType;
            // nv12 uvuv; nv21 vuvu
            if (YUV_NV12 == yuvType) {
                srcType uv_src = vldx_u8_f32<2, uint8_t, srcType>(uvptr);
                vstx_u8_f32<1, uint8_t, uint8x8_t>(dst_u_ptr, uv_src.val[0]);
                vstx_u8_f32<1, uint8_t, uint8x8_t>(dst_v_ptr, uv_src.val[1]);
            } else {
                srcType uv_src = vldx_u8_f32<2, uint8_t, srcType>(uvptr);
                vstx_u8_f32<1, uint8_t, uint8x8_t>(dst_u_ptr, uv_src.val[1]);
                vstx_u8_f32<1, uint8_t, uint8x8_t>(dst_v_ptr, uv_src.val[0]);
            }
            uvptr += 16;
            dst_u_ptr += 8;
            dst_v_ptr += 8;
        }
        for (; j < uv_w - 2; j += 2) {
            uint8_t u, v;
            if (YUV_NV12 == yuvType) {
                u = uvptr[0];
                v = uvptr[1];
            } else {
                u = uvptr[1];
                v = uvptr[0];
            }
            dst_u_ptr[0] = u;
            dst_v_ptr[0] = v;
            uvptr += 2;
            dst_u_ptr += 1;
            dst_v_ptr += 1;
        }
    }
}

template <YUV_TYPE yuvType>
void i420_to_nv_uchar_video_range(
    int32_t h,
    int32_t w,
    int32_t yStride,
    const uint8_t* y_ptr,
    int32_t uStride,
    const uint8_t* u_ptr,
    int32_t vStride,
    const uint8_t* v_ptr,
    int32_t nv_Stride,
    uint8_t* nv_ptr)
{
    // int32_t remain = w >= 15 ? w - 15 : 0;
    const uint8_t* yptr = y_ptr;
    const uint8_t* uptr = u_ptr;
    const uint8_t* vptr = v_ptr;
    uint8_t* dst_y_ptr = nv_ptr;
    uint8_t* dst_uv_ptr = nv_ptr + w * h;
    int32_t uv_w = w / 4;

    for (int32_t i = 0; i < h; i ++) {
        // process y
        memcpy(dst_y_ptr, yptr, w);
        dst_y_ptr += w;
        yptr += w;
        // process u v
        int j = 0;
        for (; j < uv_w - 8; j += 8) {
            uint8x8_t u_src = vld1_u8(uptr);
            uint8x8_t v_src = vld1_u8(vptr);
            // nv12 uvuv; nv21 vuvu
            if (YUV_NV12 == yuvType) {
                uint8x8x2_t uv;
                uv.val[0] = u_src;
                uv.val[1] = v_src;
                vst2_u8(dst_uv_ptr, uv);
            } else {
                uint8x8x2_t vu;
                vu.val[0] = v_src;
                vu.val[1] = u_src;
                vst2_u8(dst_uv_ptr, vu);
            }
            dst_uv_ptr += 16;
            uptr += 8;
            vptr += 8;
        }
        for (; j < uv_w; j ++) {
            uint8_t u = uptr[0], v = vptr[0];
            if (YUV_NV12 == yuvType) {
                dst_uv_ptr[0] = u;
                dst_uv_ptr[1] = v;
            } else {
                dst_uv_ptr[0] = v;
                dst_uv_ptr[1] = u;
            }
            dst_uv_ptr += 2;
            uptr += 1;
            vptr += 1;
        }
    }
}

template <>
::ppl::common::RetCode NV212I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t vStride = inWidthStride;
    const uint8_t* v_ptr = inData + inWidthStride * height;
    int32_t uStride = 0;
    const uint8_t* u_ptr = nullptr;
    int32_t i420Stride = outWidthStride;
    uint8_t* i420_ptr = outData;
    nv_to_i420_uchar_video_range<YUV_NV21>(height, width, inWidthStride, inData, uStride, u_ptr, vStride, v_ptr, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t vStride = 0;
    const uint8_t* v_ptr = nullptr;
    int32_t uStride = inWidthStride;
    const uint8_t* u_ptr = inData + inWidthStride * height;
    int32_t i420Stride = outWidthStride;
    uint8_t* i420_ptr = outData;
    nv_to_i420_uchar_video_range<YUV_NV12>(height, width, inWidthStride, inData, uStride, u_ptr, vStride, v_ptr, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode I4202NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t uStride = inWidthStride >> 1;
    const uint8_t* u_ptr = inData + inWidthStride * height;
    int32_t vStride = inWidthStride >> 1;
    const uint8_t* v_ptr = inData + inWidthStride * height + inWidthStride * height / 4;
    int32_t nvStride = outWidthStride;
    uint8_t* nv_ptr = outData;
    i420_to_nv_uchar_video_range<YUV_NV12>(height, width, inWidthStride, inData, uStride, u_ptr, vStride, v_ptr, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode I4202NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (!inData || !outData || height == 0 || width == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int32_t yStride = inWidthStride;
    const uint8_t* y_ptr = inData;
    int32_t uStride = inWidthStride >> 1;
    const uint8_t* u_ptr = inData + inWidthStride * height;
    int32_t vStride = inWidthStride >> 1;
    const uint8_t* v_ptr = inData + inWidthStride * height + inWidthStride * height / 4;
    int32_t nvStride = outWidthStride;
    uint8_t* nv_ptr = outData;
    i420_to_nv_uchar_video_range<YUV_NV21>(height, width, inWidthStride, inData, uStride, u_ptr, vStride, v_ptr, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm