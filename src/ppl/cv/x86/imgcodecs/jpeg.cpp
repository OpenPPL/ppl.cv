// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "jpeg.h"
#include "codecs.h"

#include <memory.h>
#include <immintrin.h>
#include <thread>
#include <vector>

#include "ppl/cv/x86/intrinutils.hpp"
#include "ppl/common/log.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace x86 {

#define FLOAT2FLOAT(x) ((int32_t) (((x) * 4096 + 0.5)))
#define FSH(x) ((x) * 4096)
#define DIVIDE4(x) ((uint8_t) ((x) >> 2))
#define DIVIDE16(x) ((uint8_t) ((x) >> 4))
#define FLOAT2FIXED(x) (((int32_t) ((x) * 4096.0f + 0.5f)) << 8)

#define NULL_MARKER 0xff
#define DRI_RESTART(marker) ((marker) >= 0xD0 && (marker) <= 0xD7)
#define ROTATE_BITS(x, y) (((x) << (y)) | ((x) >> ((BUFFER_BITS - (y)))))

// derived from jidctint -- DCT_ISLOW
#define IDCT_1D(s0, s1, s2, s3, s4, s5, s6, s7)                                \
   int32_t t0, t1, t2, t3, p1, p2, p3, p4, p5, x0, x1, x2, x3;                 \
   p2 = s2;                                                                    \
   p3 = s6;                                                                    \
   p1 = (p2 + p3) * FLOAT2FLOAT( 0.5411961f);                                  \
   t2 = p1 + p3 * FLOAT2FLOAT(-1.847759065f);                                  \
   t3 = p1 + p2 * FLOAT2FLOAT( 0.765366865f);                                  \
   p2 = s0;                                                                    \
   p3 = s4;                                                                    \
   t0 = FSH(p2 + p3);                                                          \
   t1 = FSH(p2 - p3);                                                          \
   x0 = t0 + t3;                                                               \
   x3 = t0 - t3;                                                               \
   x1 = t1 + t2;                                                               \
   x2 = t1 - t2;                                                               \
   t0 = s7;                                                                    \
   t1 = s5;                                                                    \
   t2 = s3;                                                                    \
   t3 = s1;                                                                    \
   p3 = t0 + t2;                                                               \
   p4 = t1 + t3;                                                               \
   p1 = t0 + t3;                                                               \
   p2 = t1 + t2;                                                               \
   p5 = (p3 + p4) * FLOAT2FLOAT( 1.175875602f);                                \
   t0 = t0 * FLOAT2FLOAT( 0.298631336f);                                       \
   t1 = t1 * FLOAT2FLOAT( 2.053119869f);                                       \
   t2 = t2 * FLOAT2FLOAT( 3.072711026f);                                       \
   t3 = t3 * FLOAT2FLOAT( 1.501321110f);                                       \
   p1 = p5 + p1 * FLOAT2FLOAT(-0.899976223f);                                  \
   p2 = p5 + p2 * FLOAT2FLOAT(-2.562915447f);                                  \
   p3 = p3 * FLOAT2FLOAT(-1.961570560f);                                       \
   p4 = p4 * FLOAT2FLOAT(-0.390180644f);                                       \
   t3 += p1 + p4;                                                              \
   t2 += p2 + p3;                                                              \
   t1 += p2 + p4;                                                              \
   t0 += p1 + p3;

static const uint8_t dezigzag_indices[64 + 15] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63
};

static const uint32_t bit_mask[17] = {0, 1, 3, 7, 15, 31, 63, 127, 255, 511,
                                      1023, 2047, 4095, 8191, 16383, 32767,
                                      65535};
static const uint64_t bit_mask1[17] = {0, 1, 3, 7, 15, 31, 63, 127, 255, 511,
                                       1023, 2047, 4095, 8191, 16383, 32767,
                                       65535};

inline static uint8_t clampInt8(int32_t value) {
   value = value > 255 ? 255 : (value < 0 ? 0 : value);

   return value;
}

static uint8_t *resampleRow1(uint8_t *out, uint8_t *in_near, uint8_t *in_far,
                             uint32_t width, uint32_t hs) {
   return in_near;
}

// generate two samples vertically for every one in input.
static uint8_t* resampleRowV2(uint8_t *out, uint8_t *in_near, uint8_t *in_far,
                              uint32_t width, uint32_t hs) {
    for (uint32_t i = 0; i < width; ++i) {
        out[i] = DIVIDE4(3 * in_near[i] + in_far[i] + 2);
    }

    return out;
}

// generate two samples horizontally for every one in input.
static uint8_t* resampleRowH2(uint8_t *out, uint8_t *in_near, uint8_t *in_far,
                              uint32_t width, uint32_t hs) {
    uint32_t i;
    uint8_t *input = in_near;

    // if only one sample, can't do any interpolation.
    if (width == 1) {
        out[0] = out[1] = input[0];
        return out;
    }

    out[0] = input[0];
    out[1] = DIVIDE4(input[0] * 3 + input[1] + 2);
    for (i = 1; i < width - 1; ++i) {
        int32_t n = 3 * input[i] + 2;
        out[i * 2 + 0] = DIVIDE4(n + input[i - 1]);
        out[i * 2 + 1] = DIVIDE4(n + input[i + 1]);
    }
    out[i * 2 + 0] = DIVIDE4(input[width - 2] * 3 + input[width - 1] + 2);
    out[i * 2 + 1] = input[width - 1];

    return out;
}

// generate 2x2 samples for every one in input.
static uint8_t* resampleRowHV2(uint8_t *out, uint8_t *in_near, uint8_t *in_far,
                               uint32_t width, uint32_t hs) {
    uint32_t i, t0, t1;
    if (width == 1) {
        out[0] = out[1] = DIVIDE4(3 * in_near[0] + in_far[0] + 2);
        return out;
    }

    t1 = 3 * in_near[0] + in_far[0];
    out[0] = DIVIDE4(t1 + 2);

    __m128i packed3 = _mm_set_epi16(3, 3, 3, 3, 3, 3, 3, 3);
    __m128i packed8 = _mm_set_epi16(8, 8, 8, 8, 8, 8, 8, 8);
    __m128i order   = _mm_set_epi8(15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9,
                                   1, 8, 0);
    __m128i value_near, value_far;
    __m128i value0, value1, value2, value3, value4, sum0, sum1;

    uint8_t* src0 = in_near;
    uint8_t* src1 = in_far;
    uint8_t* dst = out + 1;
    for (i = 0; i <= width - 16; i += 15) {
        value_near = _mm_loadu_si128((__m128i const*)src0);
        value_far  = _mm_loadu_si128((__m128i const*)src1);

        // process the first 8 elements.
        value0 = _mm_cvtepu8_epi16(value_near);
        value1 = _mm_mullo_epi16(value0, packed3);
        value2 = _mm_cvtepu8_epi16(value_far);
        sum0 = _mm_add_epi16(value1, value2);

        value3 = _mm_srli_si128(value_near, 1);
        value4 = _mm_srli_si128(value_far, 1);
        value0 = _mm_cvtepu8_epi16(value3);
        value1 = _mm_mullo_epi16(value0, packed3);
        value2 = _mm_cvtepu8_epi16(value4);
        sum1 = _mm_add_epi16(value1, value2);

        value0 = _mm_mullo_epi16(sum0, packed3);
        value1 = _mm_add_epi16(sum1, packed8);
        value2 = _mm_add_epi16(value0, value1);
        value3 = _mm_srli_epi16(value2, 4);
        value0 = _mm_mullo_epi16(sum1, packed3);
        value1 = _mm_add_epi16(sum0, packed8);
        value2 = _mm_add_epi16(value0, value1);
        value4 = _mm_srli_epi16(value2, 4);

        value0 = _mm_packus_epi16(value3, value4);
        value1 = _mm_shuffle_epi8(value0, order);
        _mm_storeu_si128((__m128i*)dst, value1);
        dst += 16;

        // process the last 7 elements.
        value_near = _mm_srli_si128(value_near, 8);
        value_far  = _mm_srli_si128(value_far, 8);
        value0 = _mm_cvtepu8_epi16(value_near);
        value1 = _mm_mullo_epi16(value0, packed3);
        value2 = _mm_cvtepu8_epi16(value_far);
        sum0 = _mm_add_epi16(value1, value2);

        value3 = _mm_srli_si128(value_near, 1);
        value4 = _mm_srli_si128(value_far, 1);
        value0 = _mm_cvtepu8_epi16(value3);
        value1 = _mm_mullo_epi16(value0, packed3);
        value2 = _mm_cvtepu8_epi16(value4);
        sum1 = _mm_add_epi16(value1, value2);

        value0 = _mm_mullo_epi16(sum0, packed3);
        value1 = _mm_add_epi16(sum1, packed8);
        value2 = _mm_add_epi16(value0, value1);
        value3 = _mm_srli_epi16(value2, 4);
        value0 = _mm_mullo_epi16(sum1, packed3);
        value1 = _mm_add_epi16(sum0, packed8);
        value2 = _mm_add_epi16(value0, value1);
        value4 = _mm_srli_epi16(value2, 4);

        value0 = _mm_packus_epi16(value3, value4);
        value1 = _mm_shuffle_epi8(value0, order);
        _mm_storeu_si128((__m128i*)dst, value1);

        src0 += 15;
        src1 += 15;
        dst += 14;
    }

    t1 = 3 * in_near[i - 1] + in_far[i - 1];
    for (; i < width; ++i) {
        t0 = t1;
        t1 = 3 * in_near[i] + in_far[i];
        out[i * 2 - 1] = DIVIDE16(3 * t0 + t1 + 8);
        out[i * 2]     = DIVIDE16(3 * t1 + t0 + 8);
    }
    out[width * 2 - 1] = DIVIDE4(t1 + 2);

    return out;
}

// resample with nearest-neighbor.
static uint8_t *resampleRowGeneric(uint8_t *out, uint8_t *in_near,
                                   uint8_t *in_far, uint32_t width,
                                   uint32_t hs) {
    for (uint32_t i = 0; i < width; ++i) {
        for (uint32_t j = 0; j < hs; ++j) {
            out[i * hs + j] = in_near[i];
        }
    }

    return out;
}

// fast 0..255 * 0..255 => 0..255 rounded multiplication
static uint8_t blinn8x8(uint8_t x, uint8_t y) {
   uint32_t value = x * y + 128;
   return (uint8_t) ((value + (value >> 8)) >> 8);
}

static uint8_t computeY(int32_t r, int32_t g, int32_t b) {
    return (uint8_t)((r * 77 + g * 150 + 29 * b) >> 8);
}

static void RGB2YSSE(uint8_t* input0, uint8_t* input1, uint8_t* input2,
                     uint8_t* output, uint32_t width) {
    __m128i rs, gs, bs, rs0, rs1, gs0, gs1, bs0, bs1;
    __m128i sum0, sum1, value0, value1, packed_output;
    __m128i packed77  = _mm_set_epi16(77, 77, 77, 77, 77, 77, 77, 77);
    __m128i packed150 = _mm_set_epi16(150, 150, 150, 150, 150, 150, 150, 150);
    __m128i packed29  = _mm_set_epi16(29, 29, 29, 29, 29, 29, 29, 29);

    uint32_t index = 0;
    for (; index <= width - 32; index += 32) {
        rs = _mm_loadu_si128((__m128i const*)input0);
        gs = _mm_loadu_si128((__m128i const*)input1);
        bs = _mm_loadu_si128((__m128i const*)input2);
        rs0 = _mm_cvtepu8_epi16(rs);
        gs0 = _mm_cvtepu8_epi16(gs);
        bs0 = _mm_cvtepu8_epi16(bs);
        rs = _mm_srli_si128(rs, 8);
        gs = _mm_srli_si128(gs, 8);
        bs = _mm_srli_si128(bs, 8);
        rs1 = _mm_cvtepu8_epi16(rs);
        gs1 = _mm_cvtepu8_epi16(gs);
        bs1 = _mm_cvtepu8_epi16(bs);

        rs = _mm_mullo_epi16(rs0, packed77);
        gs = _mm_mullo_epi16(gs0, packed150);
        bs = _mm_mullo_epi16(bs0, packed29);
        sum0 = _mm_add_epi16(rs, gs);
        sum1 = _mm_add_epi16(sum0, bs);
        value0 = _mm_srli_epi16(sum1, 8);

        rs = _mm_mullo_epi16(rs1, packed77);
        gs = _mm_mullo_epi16(gs1, packed150);
        bs = _mm_mullo_epi16(bs1, packed29);
        sum0 = _mm_add_epi16(rs, gs);
        sum1 = _mm_add_epi16(sum0, bs);
        value1 = _mm_srli_epi16(sum1, 8);

        packed_output = _mm_packus_epi16(value0, value1);
        _mm_storeu_si128((__m128i*)output, packed_output);
        input0 += 32;
        input1 += 32;
        input2 += 32;
        output += 32;
    }

    for (; index < width; ++index) {
        output[index] = computeY(input0[index], input1[index], input2[index]);
    }
}

/* This is a reduced-precision calculation of YCbCr-to-BGR introduced
 * to make sure the code produces the same results in both SIMD and scalar.
 */
YCrCb2BGR::YCrCb2BGR(uint32_t width, uint32_t channels) : width_(width),
                     channels_(channels) {
    sign_flip_ = _mm_set1_epi8(-0x80);
    cr_const0_ = _mm_set1_epi16( (int16_t)(1.40200f * 4096.0f + 0.5f));
    cr_const1_ = _mm_set1_epi16(-(int16_t)(0.71414f * 4096.0f + 0.5f));
    cb_const0_ = _mm_set1_epi16(-(int16_t)(0.34414f * 4096.0f + 0.5f));
    cb_const1_ = _mm_set1_epi16( (int16_t)(1.77200f * 4096.0f + 0.5f));
    y_bias_    = _mm_set1_epi8((char)128);
}

YCrCb2BGR::~YCrCb2BGR() {
}

void YCrCb2BGR::process8Elements(uint8_t const *y, uint8_t const *pcb,
                                 uint8_t const *pcr, uint32_t index,
                                 __m128i &b16s, __m128i &g16s,
                                 __m128i &r16s) const {
    // load
    __m128i y_bytes = _mm_loadl_epi64((__m128i *)(y + index));
    __m128i cb_bytes = _mm_loadl_epi64((__m128i *)(pcb + index));
    __m128i cr_bytes = _mm_loadl_epi64((__m128i *)(pcr + index));
    __m128i cb_biased = _mm_xor_si128(cb_bytes, sign_flip_);  // -128
    __m128i cr_biased = _mm_xor_si128(cr_bytes, sign_flip_);  // -128

    // unpack to int16_t (and left-shift cr, cb by 8).
    __m128i yw  = _mm_unpacklo_epi8(y_bias_, y_bytes);
    __m128i cbw = _mm_unpacklo_epi8(_mm_setzero_si128(), cb_biased);
    __m128i crw = _mm_unpacklo_epi8(_mm_setzero_si128(), cr_biased);

    // color transform
    __m128i yws = _mm_srli_epi16(yw, 4);
    __m128i cb0 = _mm_mulhi_epi16(cb_const0_, cbw);
    __m128i cr0 = _mm_mulhi_epi16(cr_const0_, crw);
    __m128i cb1 = _mm_mulhi_epi16(cbw, cb_const1_);
    __m128i cr1 = _mm_mulhi_epi16(crw, cr_const1_);
    __m128i bws = _mm_add_epi16(yws, cb1);
    __m128i gwt = _mm_add_epi16(cb0, yws);
    __m128i rws = _mm_add_epi16(cr0, yws);
    __m128i gws = _mm_add_epi16(gwt, cr1);

    // descale
    b16s = _mm_srai_epi16(bws, 4);
    g16s = _mm_srai_epi16(gws, 4);
    r16s = _mm_srai_epi16(rws, 4);
}

void YCrCb2BGR::convertBGR(uint8_t const *y, uint8_t const *pcb,
                           uint8_t const *pcr, uint8_t *dst) {
    uint32_t i = 0;
    for (; i <= width_ - 32; i += 32, dst += channels_ * 32) {
        process8Elements(y, pcb, pcr, i, b16s0_, g16s0_, r16s0_);
        process8Elements(y, pcb, pcr, i + 8, b16s1_, g16s1_, r16s1_);
        __m128i b8s0 = _mm_packus_epi16(b16s0_, b16s1_);
        __m128i g8s0 = _mm_packus_epi16(g16s0_, g16s1_);
        __m128i r8s0 = _mm_packus_epi16(r16s0_, r16s1_);

        process8Elements(y, pcb, pcr, i + 16, b16s0_, g16s0_, r16s0_);
        process8Elements(y, pcb, pcr, i + 24, b16s1_, g16s1_, r16s1_);
        __m128i b8s1 = _mm_packus_epi16(b16s0_, b16s1_);
        __m128i g8s1 = _mm_packus_epi16(g16s0_, g16s1_);
        __m128i r8s1 = _mm_packus_epi16(r16s0_, r16s1_);

        _mm_interleave_epi8(b8s0, b8s1, g8s0, g8s1, r8s0, r8s1);
        _mm_storeu_si128((__m128i *)(dst), b8s0);
        _mm_storeu_si128((__m128i *)(dst + 16), b8s1);
        _mm_storeu_si128((__m128i *)(dst + 32), g8s0);
        _mm_storeu_si128((__m128i *)(dst + 48), g8s1);
        _mm_storeu_si128((__m128i *)(dst + 64), r8s0);
        _mm_storeu_si128((__m128i *)(dst + 80), r8s1);
    }

    for (; i < width_; ++i) {
        int32_t y_fixed = (y[i] << 20) + (1 << 19); // rounding
        int32_t b, g, r;
        int32_t cb = pcb[i] - 128;
        int32_t cr = pcr[i] - 128;
        b = y_fixed + cb * FLOAT2FIXED(1.77200f);
        g = y_fixed + (cr * -FLOAT2FIXED(0.71414f)) +
            ((cb * -FLOAT2FIXED(0.34414f)) & 0xffff0000);
        r = y_fixed + cr * FLOAT2FIXED(1.40200f);
        b >>= 20;
        g >>= 20;
        r >>= 20;
        b = clampInt8(b);
        g = clampInt8(g);
        r = clampInt8(r);
        dst[0] = (uint8_t)b;
        dst[1] = (uint8_t)g;
        dst[2] = (uint8_t)r;
        dst += channels_;
    }
}

JpegDecoder::JpegDecoder(BytesReader& file_data) {
    file_data_ = &file_data;

    jpeg_ = (JpegDecodeData*) malloc(sizeof(JpegDecodeData));
    if (jpeg_ == nullptr) {
       LOG(ERROR) << "No enough memory to initialize JpegDecoder.";
    }
    jpeg_->marker = NULL_MARKER;

    hardware_threads_ = std::thread::hardware_concurrency();
    hardware_threads_ = hardware_threads_ == 0 ? 1 : hardware_threads_;
    hardware_threads_ = hardware_threads_ > 4 ? 4 : hardware_threads_;
}

JpegDecoder::~JpegDecoder() {
   free(jpeg_);
}

bool JpegDecoder::parseAPP0(JpegDecodeData *jpeg) {
    uint32_t length = file_data_->getWordBigEndian();
    if (length < 16) {
        LOG(ERROR) << "Invalid length of JFIF APP0 segment: " << length
                   << ", correct value be not less than 16.";
        return false;
    }

    uint8_t tag[5] = {0x4A, 0x46, 0x49, 0x46, 0x0};  // {'J','F','I','F','\0'}
    uint32_t i = 0;
    uint8_t value;
    for (; i < 5; ++i) {
        value = file_data_->getByte();
        if (value != tag[i]) break;
    }

    if (i == 5) {
        jpeg->jfif = 1;
    }
    else {
        jpeg->jfif = 0;
    }
    file_data_->skipBytes(length - 7);

    return true;
}

bool JpegDecoder::parseAPP14(JpegDecodeData *jpeg) {
    uint32_t length = file_data_->getWordBigEndian();
    if (length < 8) {
        LOG(ERROR) << "Invalid length of Adobe APP14 segment: " << length
                   << ", correct value be not less than 8.";
        return false;
    }

    // {'A','d','o','b','e','\0'}
    uint8_t tag[6] = {0x41, 0x64, 0x6F, 0x62, 0x65, 0x0};
    int32_t i = 0;
    uint8_t value;
    for (; i < 6; ++i) {
        value = file_data_->getByte();
        if (value != tag[i]) break;
    }

    if (i == 6) {
        file_data_->skipBytes(5);
        jpeg->app14_color_transform = file_data_->getByte();
        length -= 6;
    }
    file_data_->skipBytes(length - 8);

    return true;
}

bool JpegDecoder::parseSOF(JpegDecodeData *jpeg) {
    uint32_t length = file_data_->getWordBigEndian();
    if (length < 11) {
        LOG(ERROR) << "Invalid SOF length: " << length
                   << ", correct value be not less than 11.";
        return false;
    }
    uint32_t value = file_data_->getByte();  // precision bit
    if (value != 8) {
        LOG(ERROR) << "Invalid pixel component precision: " << value
                   << ", correct value: 8 (bit).";
        return false;
    }

    height_ = file_data_->getWordBigEndian();
    width_  = file_data_->getWordBigEndian();
    if (height_ < 1 || width_ < 1) {
        LOG(ERROR) << "Invalid image height/width: " << height_ << ", "
                   << width_;
        return false;
    }

    jpeg->components = file_data_->getByte();  // Gray(1), YCbCr/YIQ(3), CMYK(4)
    channels_ = jpeg->components >= 3 ? 3 : 1;
    if (jpeg->components != 1 && jpeg->components != 3 &&
        jpeg->components != 4) {
        LOG(ERROR) << "Invalid component count: " << jpeg->components
                   << ", correct value: 0(Gray), 3(YCbCr), 4(CMYK).";
        return false;
    }
    if (height_ * width_ * jpeg->components > MAX_IMAGE_SIZE) {
        LOG(ERROR) << "the JPEG image is too large.";
        return false;
    }
    if (length != 8 + 3 * jpeg->components) {
        LOG(ERROR) << "Invalid SOF length: " << length << ", correct value: "
                   << 8 + 3 * jpeg->components;
        return false;
    }

    uint32_t i = 0;
    for (; i < jpeg->components; i++) {
        jpeg->img_comp[i].data = nullptr;
        jpeg->img_comp[i].line_buffer = nullptr;
    }

    jpeg->rgb = 0;
    uint32_t h_max = 1, v_max = 1;
    for (i = 0; i < jpeg->components; ++i) {
        const uint8_t rgb[3] = { 'R', 'G', 'B' };
        // component id: Y(1), Cb(2), Cr(3), I(4), Q(5)
        jpeg->img_comp[i].id = file_data_->getByte();
        if (jpeg->components == 3 && jpeg->img_comp[i].id == rgb[i]) {
            ++jpeg->rgb;
        }
        value = file_data_->getByte();
        jpeg->img_comp[i].hsampling = (value >> 4);  // horizonal sampling rate
        if (!jpeg->img_comp[i].hsampling || jpeg->img_comp[i].hsampling > 4) {
            LOG(ERROR) << "Invalid horizontal sampling rate: " << (value >> 4)
                       << " of component " << jpeg->img_comp[i].id
                       << ", valid value: 1-3.";
            return false;
        }
        jpeg->img_comp[i].vsampling = value & 15;   // vertical sampling rate
        if (!jpeg->img_comp[i].vsampling || jpeg->img_comp[i].vsampling > 4) {
            LOG(ERROR) << "Invalid vertical sampling rate: " << (value & 15)
                       << " of component " << jpeg->img_comp[i].id
                       << ", valid value: 1-3.";
            return false;
        }
        value = file_data_->getByte();
        jpeg->img_comp[i].quant_id = value;        // quantification table ID
        if (jpeg->img_comp[i].quant_id > 3) {
            LOG(ERROR) << "Invalid ID of quantification table: " << value
                       << " of component " << jpeg->img_comp[i].id
                       << ", valid value: 0-3.";
            return false;
        }

        if (jpeg->img_comp[i].hsampling > h_max) {
            h_max = jpeg->img_comp[i].hsampling;
        }
        if (jpeg->img_comp[i].vsampling > v_max) {
            v_max = jpeg->img_comp[i].vsampling;
        }
    }

    for (i = 0; i < jpeg->components; ++i) {
        if (h_max % jpeg->img_comp[i].hsampling != 0) {
            LOG(ERROR) << "Invalid horizontal component samples.";
            return false;
        }
        if (v_max % jpeg->img_comp[i].vsampling != 0) {
            LOG(ERROR) << "Invalid vertical component samples.";
            return false;
        }
    }

    // compute interleaved mcu info.
    jpeg->hsampling_max = h_max;
    jpeg->vsampling_max = v_max;
    jpeg->mcu_width  = h_max * 8;
    jpeg->mcu_height = v_max * 8;
    jpeg->mcus_x = (width_ + jpeg->mcu_width - 1) / jpeg->mcu_width;
    jpeg->mcus_y = (height_ + jpeg->mcu_height - 1) / jpeg->mcu_height;

    for (i = 0; i < jpeg->components; ++i) {
        // number of effective pixels(e.g. for non-interleaved MCU).
        jpeg->img_comp[i].x =
            (width_ * jpeg->img_comp[i].hsampling + h_max - 1) / h_max;
        jpeg->img_comp[i].y =
            (height_ * jpeg->img_comp[i].vsampling + v_max - 1) / v_max;
        jpeg->img_comp[i].w2 = jpeg->mcus_x * jpeg->img_comp[i].hsampling * 8;
        jpeg->img_comp[i].h2 = jpeg->mcus_y * jpeg->img_comp[i].vsampling * 8;
        jpeg->img_comp[i].data =
            (uint8_t*)malloc(jpeg->img_comp[i].w2 * jpeg->img_comp[i].h2 + 15);
        if (jpeg->img_comp[i].data == nullptr) {
            freeComponents(jpeg, i + 1);
            LOG(ERROR) << "failed to allocate data buffer for component "
                       << jpeg->img_comp[i].id;
            return false;
        }
        jpeg->img_comp[i].line_buffer = nullptr;
        jpeg->img_comp[i].coeff = nullptr;
        if (jpeg->progressive) {
            jpeg->img_comp[i].coeff_w = jpeg->img_comp[i].w2 / 8;
            size_t size = jpeg->img_comp[i].w2 * jpeg->img_comp[i].h2 *
                          sizeof(int16_t) + 15;
            jpeg->img_comp[i].coeff = (int16_t*)malloc(size);
            if (jpeg->img_comp[i].coeff == nullptr) {
                freeComponents(jpeg, i + 1);
                LOG(ERROR) << "failed to allocate coeff buffer for component "
                           << jpeg->img_comp[i].id;
                return false;
            }
            memset(jpeg->img_comp[i].coeff, 0, size);
        }
    }

    return true;
}

bool JpegDecoder::parseSOS(JpegDecodeData *jpeg) {
    uint32_t length = file_data_->getWordBigEndian();
    uint32_t components = file_data_->getByte();
    if (!(components == 1 || components == 3 || components == 4)) {
        LOG(ERROR) << "Invalid SOS component count: " << components
                   << ", valid value: Gray(1), YCbCr(3), CMYK(4).";
        return false;
    }
    jpeg->scan_n = components;  // Gray(1), YCbCr/YIQ(3), CMYK(4)
    if (length != 6 + 2 * components) {
        LOG(ERROR) << "Invalid SOS length: " << length << ", valid value: "
                   << 6 + 2 * components;
        return false;
    }

    uint32_t component_id, table_ids, index;
    for (uint32_t i = 0; i < components; i++) {
        component_id = file_data_->getByte();
        table_ids = file_data_->getByte();
        for (index = 0; index < jpeg->components; index++) {
            if (jpeg->img_comp[index].id == component_id) break;
        }
        if (index == jpeg->components) return false;
        jpeg->img_comp[index].dc_id = table_ids >> 4;
        if (jpeg->img_comp[index].dc_id > 3) {
            LOG(ERROR) << "Invalid table id of DC: "
                       << jpeg->img_comp[index].dc_id << ", valid value: 0-3.";
            return false;
        }
        jpeg->img_comp[index].ac_id = table_ids & 15;
        if (jpeg->img_comp[index].ac_id > 3) {
            LOG(ERROR) << "Invalid table id of AC: "
                       << jpeg->img_comp[index].ac_id << ", valid value: 0-3.";
            return false;
        }
        jpeg->order[i] = index;
    }

    jpeg->index_start = file_data_->getByte();  // 0x00?
    jpeg->index_end   = file_data_->getByte();  // 0x3F? should be 63, but might be 0
    uint32_t value = file_data_->getByte();     // 0x00?
    jpeg->succ_high = (value >> 4);
    jpeg->succ_low  = (value & 15);

    if (jpeg->progressive) {
        if (jpeg->index_start > 63 || jpeg->index_end > 63  ||
            jpeg->index_start > jpeg->index_end || jpeg->succ_high > 13 ||
            jpeg->succ_low > 13) {
            LOG(ERROR) << "Invalid index/succ_high/succ_low for progressive jpeg.";
            return false;
        }
    } else {
        if (jpeg->index_start != 0) {
            LOG(ERROR) << "Invalid index start for non-progressive jpeg.";
            return false;
        }
        if (jpeg->succ_high != 0 || jpeg->succ_low != 0) {
            LOG(ERROR) << "Invalid succ_high/succ_low for non-progressive jpeg.";
            return false;
        }
        jpeg->index_end = 63;
    }

    return true;
}

bool JpegDecoder::parseDQT(JpegDecodeData *jpeg) {
    uint32_t length = file_data_->getWordBigEndian();

    length -= 2;
    while (length > 0) {
        uint32_t value = file_data_->getByte();
        uint32_t precision = value >> 4;
        uint32_t table_id  = value & 15;
        if (precision != 0 && precision != 1) {
            LOG(ERROR) << "Invalid quantization table precision type: "
                       << precision
                       << ", correct value: 0(8 bits), 1(16 bits).";
            return false;
        }
        bool is_16bits = (precision == 1);
        if (table_id > 3) {
            LOG(ERROR) << "Invalid quantization table id: " << table_id
                       << ", correct value: 0~3.";
            return false;
        }

        uint16_t* table = jpeg->dequant[table_id];
        if (is_16bits) {
            for (uint32_t i = 0; i < 64; ++i) {
                table[dezigzag_indices[i]] = file_data_->getWordBigEndian();
            }
        }
        else {
            for (uint32_t i = 0; i < 64; ++i) {
                table[dezigzag_indices[i]] = file_data_->getByte();
            }
        }

        length -= precision ? 129 : 65;
    }

    return (length == 0);
}

bool JpegDecoder::buildHuffmanTable(HuffmanLookupTable *huffman_table,
                                    uint32_t *symbol_counts) {
    uint8_t bit_lengths[257];
    uint16_t codes[256];
    uint32_t bit_number, i, index = 0;
    // build code length list for each symbol(from JPEG spec).
    for (bit_number = 1; bit_number <= MAX_BITS; ++bit_number) {
        for (i = 0; i < symbol_counts[bit_number - 1]; ++i) {
            bit_lengths[index++] = (uint8_t)bit_number;
        }
    }
    bit_lengths[index] = 0;

    // compute actual binary codes(from jpeg spec).
    uint32_t code = 0;
    index = 0;
    for (bit_number = 1; bit_number <= MAX_BITS; ++bit_number) {
        // compute delta to add to code to compute symbol id.
        huffman_table->delta[bit_number] = index - code;
        if (bit_lengths[index] == bit_number) {
            while (bit_lengths[index] == bit_number) {
                codes[index++] = (uint16_t)(code++);
            }
            if (code >= ((uint32_t)1 << bit_number)) {
                LOG(ERROR) << "Wrong code during huffman tabale generation.";
                return false;
            }
        }
        // compute largest code + 1 for this size, preshifted as needed later.
        huffman_table->max_codes[bit_number] = code << (MAX_BITS - bit_number);
        code <<= 1;
    }
    huffman_table->max_codes[bit_number] = 0xFFFF;

    // build non-spec acceleration table; oxFFFF is flag for not-accelerated.
    // store indices of tuples of bit length - symbol.
    memset(huffman_table->lookups, 255,
           sizeof(uint16_t) * (1 << LOOKAHEAD_BITS));
    uint32_t size = index;
    for (index = 0; index < size; ++index) {
        uint32_t bit_length = bit_lengths[index];
        if (bit_length <= LOOKAHEAD_BITS) {
            code = codes[index] << (LOOKAHEAD_BITS - bit_length);
            uint32_t count = 1 << (LOOKAHEAD_BITS - bit_length);
            for (i = 0; i < count; ++i) {
                huffman_table->lookups[code++] = (bit_length << 8) |
                                                 huffman_table->symbols[index];
            }
        }
    }

    return true;
}

bool JpegDecoder::parseDHT(JpegDecodeData *jpeg) {
    uint32_t length = file_data_->getWordBigEndian();
    if (length <= 19) {
        LOG(ERROR) << "Invalid DHT length: " << length
                   << ", valid value should be not less than 19.";
        return false;
    }

    length -= 2;
    while (length > 0) {
        uint32_t value = file_data_->getByte();
        uint32_t type = value >> 4;      // 0(DC table), 1(AC table)
        uint32_t table_id = value & 15;  // 0~3
        if (type > 1) {
            LOG(ERROR) << "Invalid DHT type: " << type
                       << ", correct value: 0(DC table), 1(AC table).";
            return false;
        }
        if (table_id > 3) {
            LOG(ERROR) << "Invalid DHT table id: " << table_id
                       << ", correct value: 0-3.";
            return false;
        }

        uint32_t symbol_counts[16], count = 0;
        for (uint32_t i = 0; i < 16; ++i) {
            symbol_counts[i] = file_data_->getByte();
            count += symbol_counts[i];
        }

        uint8_t *symbol;
        if (type == 0) {
            symbol = jpeg->huff_dc[table_id].symbols;
            for (uint32_t i = 0; i < count; ++i) {
                symbol[i] = file_data_->getByte();
            }
            if (!buildHuffmanTable(jpeg->huff_dc + table_id, symbol_counts)) {
                return false;
            }
        }
        else {
            symbol = jpeg->huff_ac[table_id].symbols;
            for (uint32_t i = 0; i < count; ++i) {
                symbol[i] = file_data_->getByte();
            }
            if (!buildHuffmanTable(jpeg->huff_ac + table_id, symbol_counts)) {
                return false;
            }
        }

        length -= (17 + count);
    }

    return (length == 0);
}

bool JpegDecoder::parseCOM() {
    uint32_t length = file_data_->getWordBigEndian();
    if (length < 2) {
        LOG(ERROR) << "Invalid comment length: " << length
                   << ", valid value should be not less than 2.";
        return false;
    }

    file_data_->skipBytes(length - 2);

    return true;
}

bool JpegDecoder::parseDRI(JpegDecodeData *jpeg) {
    uint32_t length = file_data_->getWordBigEndian();
    if (length != 4) {
        LOG(ERROR) << "Invalid comment length: " << length
                   << ", valid value should be 4.";
        return false;
    }

    int32_t value = file_data_->getWordBigEndian();
    jpeg->restart_interval = value;

    return true;
}

bool JpegDecoder::parseDNL() {
    uint32_t length = file_data_->getWordBigEndian();
    if (length != 4) {
        LOG(ERROR) << "Invalid DNL length: " << length
                   << ", valid value should be 4.";
        return false;
    }

    uint32_t height = file_data_->getWordBigEndian();
    if (height != height_) {
        LOG(ERROR) << "Invalid DNL height: " << height
                   << ", valid value should be " << height_;
        return false;
    }

    return true;
}

bool JpegDecoder::processOtherSegments(int32_t marker) {
    uint32_t length = file_data_->getWordBigEndian();
    if (length < 2) {
        LOG(ERROR) << "Invalid segment length: " << length
                   << ", valid value should be not less than 2.";
        return false;
    }

    file_data_->skipBytes(length - 2);

    return true;
}

bool JpegDecoder::processSegments(JpegDecodeData *jpeg, uint8_t marker) {
    bool succeeded;
    switch (marker) {
        // case 0xE1~0xEF: optional segments, APP1 for exif, APP14 for adobe.
        case 0xE0:  // JFIF
            succeeded = parseAPP0(jpeg);
            break;
        case 0xE1:  // Exif, APP1
            LOG(ERROR) << "EXIF JPEG is not supported.";
            succeeded = false;
            break;
        case 0xEE:  // Adobe APP14
            succeeded = parseAPP14(jpeg);
            break;
        case 0xDB:  // define quantization table
            succeeded = parseDQT(jpeg);
            break;
        // case 0xC0~0xCF: optional segments.
        case 0xC0:  // start of frame0, baseline DCT-based JPEG
        case 0xC2:  // start of frame2, progressive DCT-based JPEG
            if (marker == 0xC2) jpeg_->progressive = true;
            succeeded = parseSOF(jpeg);
            break;
        case 0xC4:  // define huffman table
            succeeded = parseDHT(jpeg);
            break;
        case 0xDD:  // define restart interval
            succeeded = parseDRI(jpeg);
            break;
        case 0xFE:  // comment
            succeeded = parseCOM();
            break;
        case 0xDC:  // define number of lines
            succeeded = parseDNL();
            break;
        default:
            succeeded = processOtherSegments(marker);
    }

    return succeeded;
}

void JpegDecoder::freeComponents(JpegDecodeData *jpeg, uint32_t ncomp) {
    for (uint32_t i = 0; i < ncomp; ++i) {
        if (jpeg->img_comp[i].data) {
            free(jpeg->img_comp[i].data);
            jpeg->img_comp[i].data = nullptr;
        }
        if (jpeg->img_comp[i].coeff) {
            free(jpeg->img_comp[i].coeff);
            jpeg->img_comp[i].coeff = nullptr;
        }
        if (jpeg->img_comp[i].line_buffer) {
            free(jpeg->img_comp[i].line_buffer);
            jpeg->img_comp[i].line_buffer = nullptr;
        }
    }
}

/* after a restart interval, resetJpegDecoder the entropy decoder and
 * the dc prediction.
 */
void JpegDecoder::resetJpegDecoder(JpegDecodeData *jpeg) {
    jpeg->code_bits = 0;
    jpeg->code_buffer = 0;
    jpeg->nomore = 0;
    jpeg->img_comp[0].dc_pred = 0;
    jpeg->img_comp[1].dc_pred = 0;
    jpeg->img_comp[2].dc_pred = 0;
    jpeg->img_comp[3].dc_pred = 0;
    jpeg->marker = NULL_MARKER;
    jpeg->todo = jpeg->restart_interval ? jpeg->restart_interval : 0x7fffffff;
    jpeg->eob_run = 0;
}

/* If there's a pending marker from the entropy stream, return that
 * otherwise, fetch from the stream and get a marker. if there's no
 * marker, return 0xff, which is never a valid marker value.
 */
uint8_t JpegDecoder::getMarker(JpegDecodeData *jpeg) {
    uint8_t marker;
    if (jpeg->marker != NULL_MARKER) {
        marker = jpeg->marker;
        jpeg->marker = NULL_MARKER;
        return marker;
    }

    marker = file_data_->getByte();
    if (marker != 0xFF) {
        LOG(ERROR) << "invalid segment identifier.";
        return NULL_MARKER;
    }
    while (marker == 0xFF) {
        marker = file_data_->getByte();
    }

    return marker;
}

static bool global_prefix = false;
static __m128i swap_index = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8,
                                         0, 1, 2, 3, 4, 5, 6, 7);

inline void growBitBuffer(BytesReader* file_data, JpegDecodeData *jpeg) {
    uint64_t buffer;
    uint32_t valid_bytes = 0, invalid_bytes = 0;
    bool prefix_ff = global_prefix;

    uint8_t* current_data = file_data->getCurrentPosition();
    __m128i value0 = _mm_lddqu_si128((__m128i const*)current_data);
    __m128i value1 = _mm_shuffle_epi8(value0, swap_index);
    buffer = _mm_extract_epi64(value1, 0);

    if ((!prefix_ff) && (buffer & 0xFF00000000000000) != 0xFF00000000000000 &&
                        (buffer & 0xFF000000000000) != 0xFF000000000000 &&
                        (buffer & 0xFF0000000000) != 0xFF0000000000 &&
                        (buffer & 0xFF00000000) != 0xFF00000000 &&
                        (buffer & 0xFF000000) != 0xFF000000 &&
                        (buffer & 0xFF0000) != 0xFF0000 &&
                        (buffer & 0xFF00) != 0xFF00 &&
                        (buffer & 0xFF) != 0xFF) {
        valid_bytes = BUFFER_BYTES;

        jpeg->code_buffer |= buffer >> jpeg->code_bits;
        jpeg->code_bits += (valid_bytes) << 3;
        if (jpeg->code_bits > BUFFER_BITS) {
            invalid_bytes = ((jpeg->code_bits - BUFFER_BITS) + 7) >> 3;
            jpeg->code_bits -= (invalid_bytes << 3);
            valid_bytes -= invalid_bytes;
        }
        file_data->skipBytes(valid_bytes);
    }
    else {
        uint32_t index = 0, processed_bytes = 0, ff00_index = 0;
        uint8_t current_byte;
        do {
            if (jpeg->nomore) {
                break;
            }
            if (index == 0) {
                current_byte = (buffer & 0xFF00000000000000) >> 56;
                if (processed_bytes == BUFFER_BYTES - 1 &&
                    current_byte == 0xFF) {
                    break;
                }
                if (prefix_ff) {
                    if (current_byte == 0xFF) {
                        buffer = (buffer << 8);
                    }
                    else if (current_byte == 0) {
                        buffer |= 0xFF00000000000000;
                        prefix_ff = false;
                        index++;
                    }
                    else {
                        jpeg->marker = current_byte;
                        jpeg->nomore = 1;
                        prefix_ff = false;
                    }
                }
                else {
                    if (current_byte == 0xFF) {
                        buffer = (buffer << 8);
                        prefix_ff = true;
                    }
                    else {
                        index++;
                    }
                }
            }
            else if (index == 1) {
                current_byte = (buffer & 0xFF000000000000) >> 48;
                if (processed_bytes == BUFFER_BYTES - 1 &&
                    current_byte == 0xFF) {
                    break;
                }
                if (prefix_ff) {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFF00000000000000) |
                                 ((buffer & 0xFFFFFFFFFFFF) << 8);
                    }
                    else if (current_byte == 0) {
                        buffer |= 0xFF000000000000;
                        prefix_ff = false;
                        index++;
                    }
                    else {
                        jpeg->marker = current_byte;
                        jpeg->nomore = 1;
                        prefix_ff = false;
                    }
                }
                else {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFF00000000000000) |
                                 ((buffer & 0xFFFFFFFFFFFF) << 8);
                        prefix_ff = true;
                    }
                    else {
                        index++;
                    }
                }
            }
            else if (index == 2) {
                current_byte = (buffer & 0xFF0000000000) >> 40;
                if (processed_bytes == BUFFER_BYTES - 1 &&
                    current_byte == 0xFF) {
                    break;
                }
                if (prefix_ff) {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFF000000000000) |
                                 ((buffer & 0xFFFFFFFFFF) << 8);
                    }
                    else if (current_byte == 0) {
                        buffer |= 0xFF0000000000;
                        prefix_ff = false;
                        ff00_index = 2;
                        index++;
                    }
                    else {
                        jpeg->marker = current_byte;
                        jpeg->nomore = 1;
                        prefix_ff = false;
                    }
                }
                else {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFF000000000000) |
                                 ((buffer & 0xFFFFFFFFFF) << 8);
                        prefix_ff = true;
                    }
                    else {
                        index++;
                    }
                }
            }
            else if (index == 3) {
                current_byte = (buffer & 0xFF00000000) >> 32;
                if (processed_bytes == BUFFER_BYTES - 1 &&
                    current_byte == 0xFF) {
                    break;
                }
                if (prefix_ff) {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFF0000000000) |
                                 ((buffer & 0xFFFFFFFF) << 8);
                    }
                    else if (current_byte == 0) {
                        buffer |= 0xFF00000000;
                        prefix_ff = false;
                        ff00_index = 3;
                        index++;
                    }
                    else {
                        jpeg->marker = current_byte;
                        jpeg->nomore = 1;
                        prefix_ff = false;
                    }
                }
                else {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFF0000000000) |
                                 ((buffer & 0xFFFFFFFF) << 8);
                        prefix_ff = true;
                    }
                    else {
                        index++;
                    }
                }
            }
            else if (index == 4) {
                current_byte = (buffer & 0xFF000000) >> 24;
                if (processed_bytes == BUFFER_BYTES - 1 &&
                    current_byte == 0xFF) {
                    break;
                }
                if (prefix_ff) {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFFFF00000000) |
                                 ((buffer & 0xFFFFFF) << 8);
                    }
                    else if (current_byte == 0) {
                        buffer |= 0xFF000000;
                        prefix_ff = false;
                        ff00_index = 4;
                        index++;
                    }
                    else {
                        jpeg->marker = current_byte;
                        jpeg->nomore = 1;
                        prefix_ff = false;
                    }
                }
                else {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFFFF00000000) |
                                 ((buffer & 0xFFFFFF) << 8);
                        prefix_ff = true;
                    }
                    else {
                        index++;
                    }
                }
            }
            else if (index == 5) {
                current_byte = (buffer & 0xFF0000) >> 16;
                if (processed_bytes == BUFFER_BYTES - 1 &&
                    current_byte == 0xFF) {
                    break;
                }
                if (prefix_ff) {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFFFFFF000000) |
                                 ((buffer & 0xFFFF) << 8);
                    }
                    else if (current_byte == 0) {
                        buffer |= 0xFF0000;
                        prefix_ff = false;
                        ff00_index = 5;
                        index++;
                    }
                    else {
                        jpeg->marker = current_byte;
                        jpeg->nomore = 1;
                        prefix_ff = false;
                    }
                }
                else {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFFFFFF000000) |
                                 ((buffer & 0xFFFF) << 8);
                        prefix_ff = true;
                    }
                    else {
                        index++;
                    }
                }
            }
            else if (index == 6) {
                current_byte = (buffer & 0xFF00) >> 8;
                if (processed_bytes == BUFFER_BYTES - 1 &&
                    current_byte == 0xFF) {
                    break;
                }
                if (prefix_ff) {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFFFFFFFF0000) |
                                 ((buffer & 0xFF) << 8);
                    }
                    else if (current_byte == 0) {
                        buffer |= 0xFF00;
                        prefix_ff = false;
                        ff00_index = 6;
                        index++;
                    }
                    else {
                        jpeg->marker = current_byte;
                        jpeg->nomore = 1;
                        prefix_ff = false;
                    }
                }
                else {
                    if (current_byte == 0xFF) {
                        buffer = (buffer & 0xFFFFFFFFFFFF0000) |
                                 ((buffer & 0xFF) << 8);
                        prefix_ff = true;
                    }
                    else {
                        index++;
                    }
                }
            }
            else {  // index == 7
                current_byte = buffer & 0xFF;
                if (current_byte == 0xFF) {
                    break;
                }
                else {
                    if (prefix_ff) {
                        if (current_byte == 0) {
                            buffer |= 0xFF;
                            prefix_ff = false;
                            ff00_index = 7;
                            index++;
                        }
                        else {
                            jpeg->marker = current_byte;
                            jpeg->nomore = 1;
                            prefix_ff = false;
                        }
                    }
                    else {
                        index++;
                    }
                }
            }
            processed_bytes++;

            if (processed_bytes == BUFFER_BYTES && index == 0) {
                file_data->skipBytes(BUFFER_BYTES);
                uint8_t* current_data = file_data->getCurrentPosition();
                __m128i value0 = _mm_lddqu_si128((__m128i const*)current_data);
                __m128i value1 = _mm_shuffle_epi8(value0, swap_index);
                buffer = _mm_extract_epi64(value1, 0);

                processed_bytes = 0;
            }
        } while (processed_bytes < BUFFER_BYTES);

        global_prefix = prefix_ff;
        jpeg->code_buffer |= buffer >> jpeg->code_bits;
        jpeg->code_bits += (index) << 3;
        if (jpeg->code_bits > BUFFER_BITS) {
            invalid_bytes = ((jpeg->code_bits - BUFFER_BITS) + 7) >> 3;
            jpeg->code_bits -= (invalid_bytes << 3);
            valid_bytes = index - invalid_bytes;
            invalid_bytes = ff00_index >= valid_bytes ? invalid_bytes + 1 :
                            invalid_bytes;
        }
        processed_bytes -= invalid_bytes;
        file_data->skipBytes(processed_bytes);
    }
}

inline uint32_t getBits(JpegDecodeData *jpeg, BytesReader* file_data,
                        uint32_t bit_length) {
    if (jpeg->marker == NULL_MARKER && jpeg->code_bits < bit_length) {
        growBitBuffer(file_data, jpeg);
    }

    uint64_t value = ROTATE_BITS(jpeg->code_buffer, bit_length);
    jpeg->code_buffer = value & ~bit_mask1[bit_length];
    value &= bit_mask1[bit_length];
    jpeg->code_bits -= bit_length;

    return (uint32_t)value;
}

inline uint32_t getBit(JpegDecodeData *jpeg, BytesReader* file_data) {
    if (jpeg->code_bits < 1) {
        growBitBuffer(file_data, jpeg);
    }

    uint64_t value = jpeg->code_buffer;
    jpeg->code_buffer <<= 1;
    --jpeg->code_bits;
    uint32_t result = (value & ((uint64_t)1 << (BUFFER_BITS - 1))) >> 32;

    return result;
}

inline int32_t extendReceive(JpegDecodeData *jpeg, BytesReader* file_data,
                             uint32_t bit_length) {
    if (jpeg->marker == NULL_MARKER && jpeg->code_bits < bit_length) {
        growBitBuffer(file_data, jpeg);
    }

    // sign bit always in MSB; 0 if MSB clear (positive), 1 if MSB set (negative)
    uint32_t sign = jpeg->code_buffer >> (BUFFER_BITS - 1);
    uint64_t value = ROTATE_BITS(jpeg->code_buffer, bit_length);
    value &= ((1 << bit_length) - 1);
    value += (((-(1 << bit_length)) + 1) & (sign - 1));

    jpeg->code_buffer <<= bit_length;
    jpeg->code_bits -= bit_length;

    return (int32_t)value;
}

// decode a jpeg huffman value(bit length) from the bitstream
inline int32_t JpegDecoder::decodeHuffmanData(JpegDecodeData *jpeg,
                                            HuffmanLookupTable *huffman_table) {
    if (jpeg_->marker == NULL_MARKER && jpeg->code_bits < LOOKAHEAD_BITS) {
        growBitBuffer(file_data_, jpeg);
    }

    // look at the top LOOKAHEAD_BITS and fast indexed table to determine
    // bit length and symbol if the bits is <= LOOKAHEAD_BITS.
    uint16_t bits = (jpeg->code_buffer >> (BUFFER_BITS - LOOKAHEAD_BITS)) &
                    ((1 << LOOKAHEAD_BITS) - 1);
    int32_t value = huffman_table->lookups[bits];
    int32_t bit_length;
    if (value != 0xFFFF) {
        bit_length = (value >> 8) & 0xFF;
        jpeg->code_buffer <<= bit_length;
        jpeg->code_bits -= bit_length;
        return (value & 0xFF);
    }

    if (jpeg_->marker == NULL_MARKER && jpeg->code_bits < MAX_BITS) {
        growBitBuffer(file_data_, jpeg);
    }

    bits = jpeg->code_buffer >> (BUFFER_BITS - MAX_BITS);
    for (bit_length = LOOKAHEAD_BITS + 1; ; ++bit_length) {
        if (bits < huffman_table->max_codes[bit_length]) {
            break;
        }
    }
    if (bit_length >= 17) {  // error! code not found.
        jpeg->code_bits -= 16;
        return -1;
    }

    // convert the huffman code to the symbol id.
    int32_t index = ((jpeg->code_buffer >> (BUFFER_BITS - bit_length)) &
                    bit_mask[bit_length]) + huffman_table->delta[bit_length];
    jpeg->code_buffer <<= bit_length;
    jpeg->code_bits -= bit_length;

    return huffman_table->symbols[index];
}

// decode a 8x8 block from huffman encoding + zigzag ordering + dequantization.
bool JpegDecoder::decodeBlock(JpegDecodeData *jpeg, int16_t decoded_data[64],
                              HuffmanLookupTable *huffman_dc,
                              HuffmanLookupTable *huffman_ac,
                              uint32_t component_id, uint16_t *dequant_table) {
    // decode DC component.
    int32_t bit_length = decodeHuffmanData(jpeg, huffman_dc);
    if (bit_length < 0 || bit_length > 15) {
        LOG(ERROR) << "Invalid bit length of DC value from huffman decoding: "
                   << bit_length << ", valid value: 0-15.";
    }

    int32_t value = bit_length ?
                    extendReceive(jpeg, file_data_, bit_length) : 0;
    int32_t dc_value = jpeg->img_comp[component_id].dc_pred + value;
    jpeg->img_comp[component_id].dc_pred = dc_value;
    decoded_data[0] = (int16_t)(dc_value * dequant_table[0]);

    // decode AC components.
    uint32_t combined_value, zeroes, zig_index, ac_index = 1;
    do {
        // combined_value: number of zero + bit length of incoming code of the
        // jpeg fixed encoding table.
        combined_value = decodeHuffmanData(jpeg, huffman_ac);
        zeroes = combined_value >> 4;
        bit_length = combined_value & 15;
        if (bit_length == 0) {
            if (zeroes != 0xF0) {
                break;  // end of block
            }
            ac_index += 16;
        } else {
            ac_index += zeroes;
            zig_index = dezigzag_indices[ac_index++];
            value = extendReceive(jpeg, file_data_, bit_length) *
                    dequant_table[zig_index];
            decoded_data[zig_index] = (int16_t)value;
        }
    } while (ac_index < 64);

    return true;
}

void JpegDecoder::idctDecodeBlock(uint8_t *output, int32_t out_stride,
                                  int16_t data[64]) {
    int32_t i, val[64], *v = val;
    uint8_t *o;
    int16_t *d = data;
    int32_t scaled = 65536 + (128 << 17);

    // columns
    for (i = 0; i < 8; ++i, ++d, ++v) {
        // if all zeroes, shortcut -- this avoids dequantizing 0s and IDCTing.
        if (d[8] == 0 && d[16] == 0 && d[24] == 0 && d[32] == 0 &&
            d[40] == 0 && d[48] == 0 && d[56] == 0) {
            //    no shortcut                 0     seconds
            //    (1|2|3|4|5|6|7)==0          0     seconds
            //    all separate               -0.047 seconds
            //    1 && 2|3 && 4|5 && 6|7:    -0.047 seconds
            int32_t dcterm = d[0] * 4;
            v[0] = v[8] = v[16] = v[24] = v[32] = v[40] = v[48] = v[56] =
                   dcterm;
        } else {
            IDCT_1D(d[0], d[8], d[16], d[24], d[32], d[40], d[48], d[56])
            // constants scaled things up by 1<<12; let's bring them back
            // down, but keep 2 extra bits of precision.
            x0 += 512; x1 += 512; x2 += 512; x3 += 512;
            v[ 0] = (x0 + t3) >> 10;
            v[56] = (x0 - t3) >> 10;
            v[ 8] = (x1 + t2) >> 10;
            v[48] = (x1 - t2) >> 10;
            v[16] = (x2 + t1) >> 10;
            v[40] = (x2 - t1) >> 10;
            v[24] = (x3 + t0) >> 10;
            v[32] = (x3 - t0) >> 10;
        }
    }

    for (i = 0, v = val, o = output; i < 8; ++i, v += 8, o += out_stride) {
        // no fast case since the first 1D IDCT spread components out.
        IDCT_1D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
        // constants scaled things up by 1<<12, plus we had 1<<2 from first
        // loop, plus horizontal and vertical each scale by sqrt(8) so together
        // we've got an extra 1<<3, so 1<<17 total we need to remove.
        // so we want to round that, which means adding 0.5 * 1<<17,
        // aka 65536. Also, we'll end up with -128 to 127 that we want
        // to encode as 0..255 by adding 128, so we'll add that before the shift
        x0 += scaled;
        x1 += scaled;
        x2 += scaled;
        x3 += scaled;
        // tried computing the shifts into temps, or'ing the temps to see
        // if any were out of range, but that was slower.
        o[0] = clampInt8((x0 + t3) >> 17);
        o[7] = clampInt8((x0 - t3) >> 17);
        o[1] = clampInt8((x1 + t2) >> 17);
        o[6] = clampInt8((x1 - t2) >> 17);
        o[2] = clampInt8((x2 + t1) >> 17);
        o[5] = clampInt8((x2 - t1) >> 17);
        o[3] = clampInt8((x3 + t0) >> 17);
        o[4] = clampInt8((x3 - t0) >> 17);
    }
}

void JpegDecoder::idctprocess0(JpegDecodeData *jpeg, int16_t* buffer,
                               uint8_t* output, uint32_t height_begin,
                               uint32_t height_end, uint32_t width,
                               uint32_t width2) {
    buffer += height_begin * width * 64;
    for (uint32_t i = height_begin; i < height_end; i++) {
        uint8_t* result = output + width2 * i * 8;
        for (uint32_t j = 0; j < width; j++) {
            idctDecodeBlock(result, width2, buffer);
            buffer += 64;
            result += 8;
        }
    }
}

bool JpegDecoder::decodeProgressiveDCBlock(JpegDecodeData *jpeg,
                                           int16_t decoded_data[64],
                                           HuffmanLookupTable *huffman_dc,
                                           uint32_t component_id,
                                           uint32_t succ_value) {
    if (jpeg->index_end != 0) {
        LOG(ERROR) << "DC index_end should be 0.";
        return false;
    }

    if (jpeg->succ_high == 0) {  // first scan for DC coefficient.
        int32_t bit_length = decodeHuffmanData(jpeg, huffman_dc);
        if (bit_length < 0 || bit_length > 15) {
            LOG(ERROR) << "Invalid bit length of DC value from huffman "
                       << "decoding: " << bit_length << ", valid value: 0-15.";
        }
        int32_t value_diff = bit_length ?
                             extendReceive(jpeg, file_data_, bit_length) : 0;

        int32_t dc_value = jpeg->img_comp[component_id].dc_pred + value_diff;
        jpeg->img_comp[component_id].dc_pred = dc_value;
        decoded_data[0] = (int16_t)(dc_value * succ_value);
    } else {  // refinement scan for DC coefficient.
        if (getBit(jpeg, file_data_)) {
            decoded_data[0] += (int16_t)succ_value;
        }
    }

    return true;
}

bool JpegDecoder::decodeProgressiveACBlock(JpegDecodeData *jpeg,
                                           int16_t decoded_data[64],
                                           HuffmanLookupTable *huffman_ac) {
    if (jpeg->index_start == 0) {
        LOG(ERROR) << "Invalid index start: " << jpeg->index_start
                   << ", valid value should be 1-63.";
        return false;
    }

    uint32_t ac_index, combined_value, zeroes, bit_length;
    if (jpeg->succ_high == 0) {  // first scan for AC coefficients.
        int32_t shift = jpeg->succ_low;

        if (jpeg->eob_run) {
            --jpeg->eob_run;
            return true;
        }

        int32_t zig_index, value;
        ac_index = jpeg->index_start;
        do {
            combined_value = decodeHuffmanData(jpeg, huffman_ac);
            zeroes = (combined_value >> 4) & 15;
            bit_length = combined_value & 15;
            if (bit_length == 0) {
                if (zeroes < 15) {
                    jpeg->eob_run = (1 << zeroes);
                    if (zeroes) {
                        jpeg->eob_run += getBits(jpeg, file_data_, zeroes);
                    }
                    --jpeg->eob_run;
                    break;
                }
                ac_index += 16;
            } else {
                ac_index += zeroes;
                zig_index = dezigzag_indices[ac_index++];
                value = extendReceive(jpeg, file_data_, bit_length) *
                        (1 << shift);
                decoded_data[zig_index] = (int16_t)value;
            }
        } while (ac_index <= jpeg->index_end);
    } else {  // refinement scan for AC coefficients.
        int16_t bit = (int16_t)(1 << jpeg->succ_low);

        if (jpeg->eob_run) {
            --jpeg->eob_run;
            for (ac_index = jpeg->index_start; ac_index <= jpeg->index_end;
                 ++ac_index) {
                int16_t &data = decoded_data[dezigzag_indices[ac_index]];
                if (data != 0) {
                    if (getBit(jpeg, file_data_)) {
                        if ((data & bit) == 0) {
                            if (data > 0) {
                                data += bit;
                            } else {
                                data -= bit;
                            }
                        }
                    }
                }
            }
        } else {
            ac_index = jpeg->index_start;
            do {
                combined_value = decodeHuffmanData(jpeg, huffman_ac);
                zeroes = combined_value >> 4;
                bit_length = combined_value & 15;
                if (bit_length == 0) {
                    if (zeroes < 15) {
                        jpeg->eob_run = (1 << zeroes) - 1;
                        if (zeroes) {
                            jpeg->eob_run += getBits(jpeg, file_data_, zeroes);
                        }
                        zeroes = 64; // force end of block.
                    } else {
                        // zeroes =15 & bit_length=0 should write 16 0s, so we just do
                        // a run of 15 0s and then write bit_length (which is 0),
                        // so we don't have to do anything special here.
                    }
                } else {
                    if (bit_length != 1) {
                        LOG(ERROR) << "Invalid bit length, which should be 1.";
                        return false;
                    }
                    if (getBit(jpeg, file_data_)) {  // sign bit
                        bit_length = bit;
                    }
                    else {
                        bit_length = -bit;
                    }
                }

                // advance by zeroes
                while (ac_index <= jpeg->index_end) {
                    int16_t &data = decoded_data[dezigzag_indices[ac_index++]];
                    if (data != 0) {
                        if (getBit(jpeg, file_data_)) {
                            if ((data & bit) == 0) {
                                if (data > 0) {
                                    data += bit;
                                } else {
                                    data -= bit;
                                }
                            }
                        }
                    } else {
                        if (zeroes == 0) {
                            data = (int16_t) bit_length;
                            break;
                        }
                        --zeroes;
                    }
                }
            } while (ac_index <= jpeg->index_end);
        }
    }

    return true;
}

bool JpegDecoder::parseEntropyCodedData(JpegDecodeData *jpeg) {
    bool succeeded;
    resetJpegDecoder(jpeg);
    if (!jpeg->progressive) {  // baseline jpeg.
        HuffmanLookupTable *huffman_dc, *huffman_ac;
        uint16_t *dequant_table;
        if (jpeg->scan_n == 1) {  // 1 components, a data block in an mcu.
            uint32_t comp_id = jpeg->order[0];
            uint32_t height = (jpeg->img_comp[comp_id].y + 7) >> 3;
            uint32_t width  = (jpeg->img_comp[comp_id].x + 7) >> 3;
            size_t size = height * width * 64;
            int16_t* buffer = new int16_t[size];
            memset(buffer, 0, size * sizeof(int16_t));
            int16_t* data = buffer;
            huffman_dc = jpeg->huff_dc + jpeg->img_comp[comp_id].dc_id;
            huffman_ac = jpeg->huff_ac + jpeg->img_comp[comp_id].ac_id;
            dequant_table = jpeg->dequant[jpeg->img_comp[comp_id].quant_id];
            for (uint32_t i = 0; i < height; ++i) {
                for (uint32_t j = 0; j < width; ++j) {
                    succeeded = decodeBlock(jpeg, data, huffman_dc, huffman_ac,
                                            comp_id, dequant_table);
                    if (!succeeded) return false;
                    data += 64;

                    // every data block is an MCU, so countdown the restart
                    // interval.
                    if (--jpeg->todo <= 0) {
                        if (!DRI_RESTART(jpeg->marker)) return false;
                        resetJpegDecoder(jpeg);
                    }
                }
            }

            uint8_t* output = jpeg->img_comp[comp_id].data;
            uint32_t width2 = jpeg->img_comp[comp_id].w2;
            std::vector<std::thread> threads;
            uint32_t interval = (height + hardware_threads_ - 1) /
                                hardware_threads_;
            for (uint32_t heigh_begin = 0; heigh_begin < height;
                 heigh_begin += interval) {
                uint32_t heigh_end = heigh_begin + interval < height ?
                                     heigh_begin + interval : height;
                threads.push_back(std::thread(&JpegDecoder::idctprocess0, this,
                                  jpeg, buffer, output, heigh_begin, heigh_end,
                                  width, width2));
            }

            for (auto &worker: threads) {
                worker.join();
            }
            delete [] buffer;

            return true;
        } else {  // n components, interleaved data blocks in an mcu.
            uint32_t i, j, k, x, y;
            int16_t** buffer = new int16_t*[jpeg->scan_n];
            for (i = 0; i < jpeg->scan_n; i++) {
                size_t size = jpeg->img_comp[i].h2 * jpeg->img_comp[i].w2;
                buffer[i] = new int16_t[size];
                memset(buffer[i], 0, size * sizeof(int16_t));
            }

            int16_t* data_ptr;
            for (i = 0; i < jpeg->mcus_y; ++i) {
                for (j = 0; j < jpeg->mcus_x; ++j) {
                    // process scan_n components in order
                    for (k = 0; k < jpeg->scan_n; ++k) {
                        uint32_t comp_id = jpeg->order[k];
                        huffman_dc = jpeg->huff_dc +
                                     jpeg->img_comp[comp_id].dc_id;
                        huffman_ac = jpeg->huff_ac +
                                     jpeg->img_comp[comp_id].ac_id;
                        dequant_table =
                                jpeg->dequant[jpeg->img_comp[comp_id].quant_id];
                        int32_t mcu_width = jpeg->mcus_x *
                                            jpeg->img_comp[comp_id].hsampling;
                        int32_t y2 = i * jpeg->img_comp[comp_id].vsampling;
                        int32_t x2 = j * jpeg->img_comp[comp_id].hsampling;
                        for (y = 0; y < jpeg->img_comp[comp_id].vsampling; ++y) {
                            for (x = 0; x < jpeg->img_comp[comp_id].hsampling; ++x) {
                                data_ptr = buffer[k] + ((y2 + y) * mcu_width + x2 + x) * 64;
                                succeeded = decodeBlock(jpeg, data_ptr, huffman_dc,
                                            huffman_ac, comp_id, dequant_table);
                                if (!succeeded) return false;
                            }
                        }
                    }

                    // after all interleaved components, that's an interleaved MCU,
                    // so now count down the restart interval
                    if (--jpeg->todo <= 0) {
                        if (!DRI_RESTART(jpeg->marker)) return false;
                        resetJpegDecoder(jpeg);
                    }
                }
            }

            for (i = 0; i < jpeg->scan_n; i++) {
                uint32_t comp_id = jpeg->order[i];
                uint32_t height = jpeg->mcus_y * jpeg->img_comp[comp_id].vsampling;
                uint32_t width  = jpeg->mcus_x * jpeg->img_comp[comp_id].hsampling;
                uint8_t* output = jpeg->img_comp[comp_id].data;
                uint32_t width2 = jpeg->img_comp[comp_id].w2;
                std::vector<std::thread> threads;
                uint32_t interval = (height + hardware_threads_ - 1) /
                                     hardware_threads_;
                for (uint32_t heigh_begin = 0; heigh_begin < height;
                     heigh_begin += interval) {
                    uint32_t heigh_end = heigh_begin + interval < height ?
                                         heigh_begin + interval : height;
                    threads.push_back(std::thread(&JpegDecoder::idctprocess0,
                                      this, jpeg, buffer[i], output,
                                      heigh_begin, heigh_end, width, width2));
                }

                for (auto &worker: threads) {
                    worker.join();
                }
            }

            for (i = 0; i < jpeg->scan_n; i++) {
                delete [] buffer[i];
            }
            delete [] buffer;

            return true;
        }
    } else {  // progressive jpeg.
        uint32_t succ_value = 1 << jpeg->succ_low;
        if (jpeg->scan_n == 1) {  // 1 components, AC scan.
            uint32_t comp_id = jpeg->order[0];
            // number of blocks to do just depends on how many actual "pixels"
            // this component has, independent of interleaved MCU blocking.
            uint32_t height = (jpeg->img_comp[comp_id].y + 7) >> 3;
            uint32_t width  = (jpeg->img_comp[comp_id].x + 7) >> 3;
            HuffmanLookupTable *huffman_dc = nullptr;
            uint32_t ac_id = 0;
            if (jpeg->index_start == 0) {
                huffman_dc = &jpeg->huff_dc[jpeg->img_comp[comp_id].dc_id];
            } else {
                ac_id = jpeg->img_comp[comp_id].ac_id;
            }
            for (uint32_t i = 0; i < height; ++i) {
                int16_t *data = jpeg->img_comp[comp_id].coeff +
                                jpeg->img_comp[comp_id].coeff_w * i * 64;
                for (uint32_t j = 0; j < width; ++j) {
                    if (jpeg->index_start == 0) {
                        succeeded = decodeProgressiveDCBlock(jpeg, data,
                                        huffman_dc, comp_id, succ_value);
                        if (!succeeded) return false;
                    } else {
                        succeeded = decodeProgressiveACBlock(jpeg, data,
                                        &jpeg->huff_ac[ac_id]);
                        if (!succeeded) return false;
                    }
                    data += 64;

                    // every data block is an MCU, so countdown the restart
                    // interval.
                    if (--jpeg->todo <= 0) {
                        if (!DRI_RESTART(jpeg->marker)) return true;
                        resetJpegDecoder(jpeg);
                    }
                }
            }
            return true;
        } else {  // n components, DC scan.
            uint32_t i, j, k, x, y;
            for (i = 0; i < jpeg->mcus_y; ++i) {
                for (j = 0; j < jpeg->mcus_x; ++j) {
                    // scan an interleaved mcu, process components in order.
                    for (k = 0; k < jpeg->scan_n; ++k) {
                        uint32_t comp_id = jpeg->order[k];
                        HuffmanLookupTable *huffman_dc =
                            &jpeg->huff_dc[jpeg->img_comp[comp_id].dc_id];
                        for (y = 0; y < jpeg->img_comp[comp_id].vsampling; ++y) {
                            for (x = 0; x < jpeg->img_comp[comp_id].hsampling; ++x) {
                                int32_t y2 = (i * jpeg->img_comp[comp_id].vsampling + y);
                                int32_t x2 = (j * jpeg->img_comp[comp_id].hsampling + x);
                                int16_t *data = jpeg->img_comp[comp_id].coeff +
                                    (y2 * jpeg->img_comp[comp_id].coeff_w + x2) * 64;
                                succeeded = decodeProgressiveDCBlock(jpeg, data,
                                                huffman_dc, comp_id, succ_value);
                                if (!succeeded) return false;
                            }
                        }
                    }

                    // after all interleaved components, that's an interleaved MCU,
                    // so now count down the restart interval
                    if (--jpeg->todo <= 0) {
                        if (!DRI_RESTART(jpeg->marker)) return true;
                        resetJpegDecoder(jpeg);
                    }
                }
            }
            return true;
        }
    }
}

// data[i] *= dequant_table[i]
void JpegDecoder::dequantizeData(int16_t *data, uint16_t *dequant_table) {
    for (int32_t i = 0; i < 8; ++i) {
        __m128i input  = _mm_lddqu_si128((__m128i const*)data);
        __m128i coeffs = _mm_lddqu_si128((__m128i const*)dequant_table);
        __m128i output = _mm_mullo_epi16(input, coeffs);
        _mm_storeu_si128((__m128i*)data, output);
        data += 8;
        dequant_table += 8;
    }
}

void JpegDecoder::idctprocess1(JpegDecodeData *jpeg, uint32_t height_begin,
                               uint32_t height_end, uint32_t width,
                               uint32_t comp_id, uint32_t width2) {
    for (uint32_t i = height_begin; i < height_end; i++) {
        uint8_t* result = jpeg->img_comp[comp_id].data +
                          jpeg->img_comp[comp_id].w2 * i * 8;
        int16_t *data = jpeg->img_comp[comp_id].coeff +
                        jpeg->img_comp[comp_id].coeff_w * i * 64;

        for (uint32_t j = 0; j < width; j++) {
            dequantizeData(data,
                           jpeg->dequant[jpeg->img_comp[comp_id].quant_id]);
            idctDecodeBlock(result, width2, data);
            data += 64;
            result += 8;
        }
    }
}

void JpegDecoder::finishProgressiveJpeg(JpegDecodeData *jpeg) {
    for (uint32_t n = 0; n < jpeg->components; ++n) {
        uint32_t height = (jpeg->img_comp[n].y + 7) >> 3;
        uint32_t width  = (jpeg->img_comp[n].x + 7) >> 3;
        uint32_t width2 = jpeg->img_comp[n].w2;
        std::vector<std::thread> threads;
        int32_t interval = (height + hardware_threads_ - 1) / hardware_threads_;

        for (uint32_t height_begin = 0; height_begin < height;
             height_begin += interval) {
             uint32_t height_end = height_begin + interval < height ?
                                 height_begin + interval : height;
            threads.push_back(std::thread(&JpegDecoder::idctprocess1, this,
                              jpeg, height_begin, height_end, width, n,
                              width2));
        }

        for (auto &worker: threads) {
            worker.join();
        }
    }
}

bool JpegDecoder::convertColor(int32_t stride, uint8_t* image) {
    uint32_t target_comps, decode_n, is_rgb;
    // determine actual number of components to generate.
    // target_comps: target components, jpeg_->components: encoded components.
    target_comps = jpeg_->components >= 3 ? 3 : 1;

    is_rgb = jpeg_->components == 3 && (jpeg_->rgb == 3 ||
                (jpeg_->app14_color_transform == 0 && !jpeg_->jfif));

    if (jpeg_->components == 3 && target_comps < 3 && !is_rgb) {
        decode_n = 1;
    }
    else {
        decode_n = jpeg_->components;
    }

    // resample and color-convert
    uint32_t k, i;
    uint8_t *output[4] = { NULL, NULL, NULL, NULL };
    SampleData res_comp[4];

    for (k = 0; k < decode_n; ++k) {
        SampleData *sample = &res_comp[k];

        // allocate line buffer big enough for upsampling off the edges
        // with upsample factor of 4
        jpeg_->img_comp[k].line_buffer = (uint8_t *) malloc(width_ + 3);
        if (!jpeg_->img_comp[k].line_buffer) {
            freeComponents(jpeg_, jpeg_->components);
            LOG(ERROR) << "No enough memory to convert sample.";
            return false;
        }

        sample->hs      = jpeg_->hsampling_max / jpeg_->img_comp[k].hsampling;
        sample->vs      = jpeg_->vsampling_max / jpeg_->img_comp[k].vsampling;
        sample->ystep   = sample->vs >> 1;
        sample->w_lores = (width_ + sample->hs - 1) / sample->hs;
        sample->ypos    = 0;
        sample->line0   = sample->line1 = jpeg_->img_comp[k].data;

        if (sample->hs == 1 && sample->vs == 1) {
            sample->resample = resampleRow1;
        }
        else if (sample->hs == 1 && sample->vs == 2) {
            sample->resample = resampleRowV2;
        }
        else if (sample->hs == 2 && sample->vs == 1) {
            sample->resample = resampleRowH2;
        }
        else if (sample->hs == 2 && sample->vs == 2) {
            sample->resample = resampleRowHV2;
        }
        else {
            sample->resample = resampleRowGeneric;
        }
    }

    if (target_comps == 3 && jpeg_->components == 3 && !is_rgb) {
        ycrcb2bgr_ = new YCrCb2BGR(width_, channels_);
    }

    // now go ahead and resample
    for (uint32_t row = 0; row < height_; ++row) {
        uint8_t *output_row = image + stride * row;
        for (k = 0; k < decode_n; ++k) {
            SampleData *sample = &res_comp[k];   // optimize
            int32_t y_bot = sample->ystep >= (sample->vs >> 1);
            output[k] = sample->resample(jpeg_->img_comp[k].line_buffer,
                                     y_bot ? sample->line1 : sample->line0,
                                     y_bot ? sample->line0 : sample->line1,
                                     sample->w_lores, sample->hs);
            if (++sample->ystep >= sample->vs) {
                sample->ystep = 0;
                sample->line0 = sample->line1;
                if (++sample->ypos < jpeg_->img_comp[k].y) {
                    sample->line1 += jpeg_->img_comp[k].w2;
                }
            }
        }
        if (target_comps == 3) {
            uint8_t *y = output[0];
            if (jpeg_->components == 3) {
                if (is_rgb) {  // input is rgb
                    for (i = 0; i < width_; ++i) {
                        output_row[0] = y[i];
                        output_row[1] = output[1][i];
                        output_row[2] = output[2][i];
                        output_row += target_comps;
                    }
                } else {  // input is YCrCb
                    ycrcb2bgr_->convertBGR(y, output[1], output[2], output_row);
                }
            } else if (jpeg_->components == 4) {
                if (jpeg_->app14_color_transform == 0) {  // CMYK
                    for (i = 0; i < width_; ++i) {
                        uint8_t m = output[3][i];
                        output_row[0] = blinn8x8(output[0][i], m);
                        output_row[1] = blinn8x8(output[1][i], m);
                        output_row[2] = blinn8x8(output[2][i], m);
                        output_row += target_comps;
                    }
                } else if (jpeg_->app14_color_transform == 2) { // YCCK
                    ycrcb2bgr_->convertBGR(y, output[1], output[2], output_row);
                    for (i = 0; i < width_; ++i) {
                        uint8_t m = output[3][i];
                        output_row[0] = blinn8x8(255 - output_row[0], m);
                        output_row[1] = blinn8x8(255 - output_row[1], m);
                        output_row[2] = blinn8x8(255 - output_row[2], m);
                        output_row += target_comps;
                    }
                } else { // YCbCr + alpha?  Ignore the fourth channel for now
                    ycrcb2bgr_->convertBGR(y, output[1], output[2], output_row);
                }
            } else {
                for (i = 0; i < width_; ++i) {
                    output_row[0] = output_row[1] = output_row[2] = y[i];
                    output_row += target_comps;
                }
            }
        } else {  // target_comps == 1
            if (is_rgb) {
                if (target_comps == 1) {
                    RGB2YSSE(output[0], output[1], output[2], output_row,
                             width_);
                }
                else {
                    for (i = 0; i < width_; ++i, output_row += 2) {
                        output_row[0] = computeY(output[0][i], output[1][i],
                                                 output[2][i]);
                        output_row[1] = 255;
                    }
                }
            } else if (jpeg_->components == 4 &&
                       jpeg_->app14_color_transform == 0) {
                for (i = 0; i < width_; ++i) {
                    uint8_t m = output[3][i];
                    uint8_t r = blinn8x8(output[0][i], m);
                    uint8_t g = blinn8x8(output[1][i], m);
                    uint8_t b = blinn8x8(output[2][i], m);
                    output_row[0] = computeY(r, g, b);
                    output_row[1] = 255;
                    output_row += target_comps;
                }
            } else if (jpeg_->components == 4 &&
                       jpeg_->app14_color_transform == 2) {
                for (i = 0; i < width_; ++i) {
                    output_row[0] = blinn8x8(255 - output[0][i],
                                             output[3][i]);
                    output_row[1] = 255;
                    output_row += target_comps;
                }
            } else {
                uint8_t *y = output[0];
                if (target_comps == 1) {
                    memcpy(output_row, y, width_);
                }
                else {
                    for (i = 0; i < width_; ++i) {
                        *output_row++ = y[i];
                        *output_row++ = 255;
                    }
                }
            }
        }
    }

    freeComponents(jpeg_, jpeg_->components);
    if (target_comps == 3 && jpeg_->components == 3 && !is_rgb) {
        delete ycrcb2bgr_;
    }

    return true;
}

bool JpegDecoder::readHeader() {
    for (uint32_t i = 0; i < 4; i++) {
        jpeg_->img_comp[i].data  = nullptr;
        jpeg_->img_comp[i].coeff = nullptr;
    }
    jpeg_->restart_interval = 0;
    jpeg_->jfif = 0;
    // valid values are 0(Unknown, 3->RGB, 4->CMYK), 1(YCbCr), 2(YCCK)
    jpeg_->app14_color_transform = -1;
    jpeg_->progressive = false;

    bool succeeded;
    file_data_->skipBytes(2);
    jpeg_->marker = NULL_MARKER;
    uint8_t marker = getMarker(jpeg_);
    while (marker != 0xDA && marker != 0xD9) {  // Start of scan or end of image
        succeeded = processSegments(jpeg_, marker);
        if (!succeeded) {
            freeComponents(jpeg_, jpeg_->components);
            return false;
        }

        marker = getMarker(jpeg_);
        if (marker == 0xDA || marker == 0xD9) {
            jpeg_->marker = marker;
        }
    }

    if (marker == 0xD9) {
        LOG(ERROR) << "No image data is datected.";
        return false;
    }

    return true;
}

bool JpegDecoder::decodeData(uint32_t stride, uint8_t* image) {
    bool succeeded;
    uint8_t marker = getMarker(jpeg_);
    while (marker != 0xD9) {   // end of image
        if (marker == 0xDA) {  // start of scan
            succeeded = parseSOS(jpeg_);
            if (!succeeded) {
                freeComponents(jpeg_, jpeg_->components);
                LOG(ERROR) << "Failed to parse the start of scan segment.";
                return false;
            }

            succeeded = parseEntropyCodedData(jpeg_);
            if (!succeeded) {
                freeComponents(jpeg_, jpeg_->components);
                LOG(ERROR) << "Failed to decode the compressed data.";
                return false;
            }
        }
        else {
            succeeded = processSegments(jpeg_, marker);
            if (!succeeded) {
                freeComponents(jpeg_, jpeg_->components);
                return false;
            }
        }

        marker = getMarker(jpeg_);
    }

    if (jpeg_->progressive) {
        finishProgressiveJpeg(jpeg_);
    }

    succeeded = convertColor(stride, image);
    freeComponents(jpeg_, jpeg_->components);
    if (!succeeded) {
        LOG(ERROR) << "Failed to sample and convert YCrCb data to the target"
                   << " color format.";
        return false;
    }

    return true;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl
