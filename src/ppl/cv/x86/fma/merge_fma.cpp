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

#include <vector>
#include <stdint.h>
#include <immintrin.h>

#define YMM_FP32_LANE_NUM  8
#define YMM_UINT8_LANE_NUM 32
namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

template <typename T, int nc>
void mergeSOA2AOS(
    int height,
    int width,
    int inWidthStride,
    const T **in,
    int outWidthStride,
    T *out);

template <>
void mergeSOA2AOS<float, 4>(
    int height,
    int width,
    int inWidthStride,
    const float **in,
    int outWidthStride,
    float *out)
{
    for (int h = 0; h < height; ++h) {
        int in_height_offset      = h * inWidthStride;
        int out_height_offset     = h * outWidthStride;
        float *base_out_ptr       = out + out_height_offset;
        const float *base_in0_ptr = in[0] + in_height_offset;
        const float *base_in1_ptr = in[1] + in_height_offset;
        const float *base_in2_ptr = in[2] + in_height_offset;
        const float *base_in3_ptr = in[3] + in_height_offset;
        for (int w = 0; w < width / YMM_FP32_LANE_NUM * YMM_FP32_LANE_NUM; w += YMM_FP32_LANE_NUM) {
            __m256 r_vec      = _mm256_loadu_ps(base_in0_ptr + w);
            __m256 g_vec      = _mm256_loadu_ps(base_in1_ptr + w);
            __m256 b_vec      = _mm256_loadu_ps(base_in2_ptr + w);
            __m256 a_vec      = _mm256_loadu_ps(base_in3_ptr + w);
            __m256 rg0145_vec = _mm256_unpacklo_ps(r_vec, g_vec);
            __m256 rg2367_vec = _mm256_unpackhi_ps(r_vec, g_vec);
            __m256 ba0145_vec = _mm256_unpacklo_ps(b_vec, a_vec);
            __m256 ba2367_vec = _mm256_unpackhi_ps(b_vec, a_vec);
            __m256 rgba04_vec = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(rg0145_vec), _mm256_castps_pd(ba0145_vec)));
            __m256 rgba15_vec = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(rg0145_vec), _mm256_castps_pd(ba0145_vec)));
            __m256 rgba26_vec = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(rg2367_vec), _mm256_castps_pd(ba2367_vec)));
            __m256 rgba37_vec = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(rg2367_vec), _mm256_castps_pd(ba2367_vec)));
            __m256 rgba01_vec = _mm256_permute2f128_ps(rgba04_vec, rgba15_vec, 0 + (2 << 4));
            __m256 rgba23_vec = _mm256_permute2f128_ps(rgba26_vec, rgba37_vec, 0 + (2 << 4));
            __m256 rgba45_vec = _mm256_permute2f128_ps(rgba04_vec, rgba15_vec, 1 + (3 << 4));
            __m256 rgba67_vec = _mm256_permute2f128_ps(rgba26_vec, rgba37_vec, 1 + (3 << 4));
            float *cur_out    = base_out_ptr + 4 * w;
            _mm256_storeu_ps(cur_out, rgba01_vec);
            _mm256_storeu_ps(cur_out + 8, rgba23_vec);
            _mm256_storeu_ps(cur_out + 16, rgba45_vec);
            _mm256_storeu_ps(cur_out + 24, rgba67_vec);
        }
        for (int w = width / YMM_FP32_LANE_NUM * YMM_FP32_LANE_NUM; w < width; ++w) {
            float *cur_out = base_out_ptr + 4 * w;
            float r        = base_in0_ptr[w];
            float g        = base_in1_ptr[w];
            float b        = base_in2_ptr[w];
            float a        = base_in3_ptr[w];
            cur_out[0]     = r;
            cur_out[1]     = g;
            cur_out[2]     = b;
            cur_out[3]     = a;
        }
    }
}

template <>
void mergeSOA2AOS<float, 3>(
    int height,
    int width,
    int inWidthStride,
    const float **in,
    int outWidthStride,
    float *out)
{
    __m256i r_idx = _mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0);
    __m256i g_idx = _mm256_set_epi32(2, 7, 4, 1, 6, 3, 0, 5);
    __m256i b_idx = _mm256_set_epi32(7, 4, 1, 6, 3, 0, 5, 2);
    for (int h = 0; h < height; ++h) {
        int in_height_offset      = h * inWidthStride;
        int out_height_offset     = h * outWidthStride;
        float *base_out_ptr       = out + out_height_offset;
        const float *base_in0_ptr = in[0] + in_height_offset;
        const float *base_in1_ptr = in[1] + in_height_offset;
        const float *base_in2_ptr = in[2] + in_height_offset;
        for (int w = 0; w < width / YMM_FP32_LANE_NUM * YMM_FP32_LANE_NUM; w += YMM_FP32_LANE_NUM) {
            __m256 r_vec    = _mm256_loadu_ps(base_in0_ptr + w);
            __m256 g_vec    = _mm256_loadu_ps(base_in1_ptr + w);
            __m256 b_vec    = _mm256_loadu_ps(base_in2_ptr + w);
            r_vec           = _mm256_permutevar8x32_ps(r_vec, r_idx);
            g_vec           = _mm256_permutevar8x32_ps(g_vec, g_idx);
            b_vec           = _mm256_permutevar8x32_ps(b_vec, b_idx);
            __m256 out0_vec = _mm256_blend_ps(r_vec, g_vec, 0x92);
            out0_vec        = _mm256_blend_ps(out0_vec, b_vec, 0x24);
            __m256 out1_vec = _mm256_blend_ps(b_vec, r_vec, 0x92);
            out1_vec        = _mm256_blend_ps(out1_vec, g_vec, 0x24);
            __m256 out2_vec = _mm256_blend_ps(g_vec, b_vec, 0x92);
            out2_vec        = _mm256_blend_ps(out2_vec, r_vec, 0x24);
            float *cur_out  = base_out_ptr + 3 * w;
            _mm256_storeu_ps(cur_out, out0_vec);
            _mm256_storeu_ps(cur_out + 8, out1_vec);
            _mm256_storeu_ps(cur_out + 16, out2_vec);
        }
        for (int w = width / YMM_FP32_LANE_NUM * YMM_FP32_LANE_NUM; w < width; ++w) {
            float *cur_out = base_out_ptr + 3 * w;
            float r        = base_in0_ptr[w];
            float g        = base_in1_ptr[w];
            float b        = base_in2_ptr[w];
            cur_out[0]     = r;
            cur_out[1]     = g;
            cur_out[2]     = b;
        }
    }
}

template <>
void mergeSOA2AOS<uint8_t, 4>(
    int height,
    int width,
    int inWidthStride,
    const uint8_t **in,
    int outWidthStride,
    uint8_t *out)
{
    for (int h = 0; h < height; ++h) {
        int in_height_offset        = h * inWidthStride;
        int out_height_offset       = h * outWidthStride;
        uint8_t *base_out_ptr       = out + out_height_offset;
        const uint8_t *base_in0_ptr = in[0] + in_height_offset;
        const uint8_t *base_in1_ptr = in[1] + in_height_offset;
        const uint8_t *base_in2_ptr = in[2] + in_height_offset;
        const uint8_t *base_in3_ptr = in[3] + in_height_offset;
        for (int w = 0; w < width / YMM_UINT8_LANE_NUM * YMM_UINT8_LANE_NUM; w += YMM_UINT8_LANE_NUM) {
            __m256i r_vec    = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base_in0_ptr + w));
            __m256i g_vec    = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base_in1_ptr + w));
            __m256i b_vec    = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base_in2_ptr + w));
            __m256i a_vec    = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base_in3_ptr + w));
            __m256i tmp0_vec = _mm256_unpacklo_epi8(r_vec, g_vec);
            __m256i tmp1_vec = _mm256_unpackhi_epi8(r_vec, g_vec);
            __m256i tmp2_vec = _mm256_unpacklo_epi8(b_vec, a_vec);
            __m256i tmp3_vec = _mm256_unpackhi_epi8(b_vec, a_vec);
            __m256i tmp4_vec = _mm256_unpacklo_epi16(tmp0_vec, tmp2_vec);
            __m256i tmp5_vec = _mm256_unpackhi_epi16(tmp0_vec, tmp2_vec);
            __m256i tmp6_vec = _mm256_unpacklo_epi16(tmp1_vec, tmp3_vec);
            __m256i tmp7_vec = _mm256_unpackhi_epi16(tmp1_vec, tmp3_vec);
            __m256i out0_vec = _mm256_permute2x128_si256(tmp4_vec, tmp5_vec, 0 + (2 << 4));
            __m256i out1_vec = _mm256_permute2x128_si256(tmp6_vec, tmp7_vec, 0 + (2 << 4));
            __m256i out2_vec = _mm256_permute2x128_si256(tmp4_vec, tmp5_vec, 1 + (3 << 4));
            __m256i out3_vec = _mm256_permute2x128_si256(tmp6_vec, tmp7_vec, 1 + (3 << 4));
            uint8_t *cur_out = base_out_ptr + 4 * w;
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(cur_out), out0_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(cur_out + 32), out1_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(cur_out + 64), out2_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(cur_out + 96), out3_vec);
        }
        for (int w = width / YMM_UINT8_LANE_NUM * YMM_UINT8_LANE_NUM; w < width; ++w) {
            uint8_t *cur_out = base_out_ptr + 4 * w;
            uint8_t r        = base_in0_ptr[w];
            uint8_t g        = base_in1_ptr[w];
            uint8_t b        = base_in2_ptr[w];
            uint8_t a        = base_in3_ptr[w];
            cur_out[0]       = r;
            cur_out[1]       = g;
            cur_out[2]       = b;
            cur_out[3]       = a;
        }
    }
}

template <>
void mergeSOA2AOS<uint8_t, 3>(
    int height,
    int width,
    int inWidthStride,
    const uint8_t **in,
    int outWidthStride,
    uint8_t *out)
{
    uint8_t r_mask[16];
    uint8_t g_mask[16];
    uint8_t b_mask[16];
    for (int i = 0; i < 16; ++i) {
        r_mask[(3 * i) % 16]     = i;
        g_mask[(3 * i + 1) % 16] = i;
        b_mask[(3 * i + 2) % 16] = i;
    }
    __m256i r_mask_vec = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<__m128i *>(r_mask)));
    __m256i g_mask_vec = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<__m128i *>(g_mask)));
    __m256i b_mask_vec = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<__m128i *>(b_mask)));

    int8_t out_mask0[16]  = {0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0};
    int8_t out_mask1[16]  = {0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0};
    __m256i out_mask0_vec = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<__m128i *>(out_mask0)));
    __m256i out_mask1_vec = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<__m128i *>(out_mask1)));

    for (int h = 0; h < height; ++h) {
        int in_height_offset        = h * inWidthStride;
        int out_height_offset       = h * outWidthStride;
        uint8_t *base_out_ptr       = out + out_height_offset;
        const uint8_t *base_in0_ptr = in[0] + in_height_offset;
        const uint8_t *base_in1_ptr = in[1] + in_height_offset;
        const uint8_t *base_in2_ptr = in[2] + in_height_offset;
        for (int w = 0; w < width / YMM_UINT8_LANE_NUM * YMM_UINT8_LANE_NUM; w += YMM_UINT8_LANE_NUM) {
            __m256i r_vec          = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base_in0_ptr + w));
            __m256i g_vec          = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base_in1_ptr + w));
            __m256i b_vec          = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base_in2_ptr + w));
            __m256i shuffled_r_vec = _mm256_shuffle_epi8(r_vec, r_mask_vec);
            __m256i shuffled_g_vec = _mm256_shuffle_epi8(g_vec, g_mask_vec);
            __m256i shuffled_b_vec = _mm256_shuffle_epi8(b_vec, b_mask_vec);
            __m256i out0_vec       = _mm256_blendv_epi8(shuffled_r_vec, shuffled_g_vec, out_mask0_vec);
            out0_vec               = _mm256_blendv_epi8(out0_vec, shuffled_b_vec, out_mask1_vec);
            __m256i out1_vec       = _mm256_blendv_epi8(shuffled_g_vec, shuffled_b_vec, out_mask0_vec);
            out1_vec               = _mm256_blendv_epi8(out1_vec, shuffled_r_vec, out_mask1_vec);
            __m256i out2_vec       = _mm256_blendv_epi8(shuffled_b_vec, shuffled_r_vec, out_mask0_vec);
            out2_vec               = _mm256_blendv_epi8(out2_vec, shuffled_g_vec, out_mask1_vec);
            uint8_t *cur_out       = base_out_ptr + 3 * w;
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(cur_out), _mm256_permute2x128_si256(out0_vec, out1_vec, 0 + (2 << 4)));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(cur_out + 32), _mm256_permute2x128_si256(out2_vec, out0_vec, 0 + (3 << 4)));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(cur_out + 64), _mm256_permute2x128_si256(out1_vec, out2_vec, 1 + (3 << 4)));
        }
        for (int w = width / YMM_UINT8_LANE_NUM * YMM_UINT8_LANE_NUM; w < width; ++w) {
            uint8_t *cur_out = base_out_ptr + 3 * w;
            uint8_t r        = base_in0_ptr[w];
            uint8_t g        = base_in1_ptr[w];
            uint8_t b        = base_in2_ptr[w];
            cur_out[0]       = r;
            cur_out[1]       = g;
            cur_out[2]       = b;
        }
    }
}
}
}
}
} // namespace ppl::cv::x86::fma
