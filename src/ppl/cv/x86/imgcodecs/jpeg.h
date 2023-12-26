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

#ifndef __ST_HPC_PPL_CV_X86_IMGCODECS_JPEG_H_
#define __ST_HPC_PPL_CV_X86_IMGCODECS_JPEG_H_

#include "imagecodecs.h"
#include "bytesreader.h"

#include <stdint.h>

#include <emmintrin.h>

namespace ppl {
namespace cv {
namespace x86 {

// huffman decoding acceleration
#define BUFFER_BYTES 8
#define BUFFER_BITS 64
#define MAX_BITS 16
#define LOOKAHEAD_BITS 9

typedef uint8_t *(*resampleRow)(uint8_t *out, uint8_t *in0, uint8_t *in1,
                                uint32_t width, uint32_t hs);

typedef struct {
    resampleRow resample;
    uint8_t *line0, *line1;
    uint32_t hs, vs;   // expansion factor in each axis
    uint32_t w_lores;  // horizontal pixels pre-expansion
    uint32_t ystep;    // how far through vertical expansion we are
    uint32_t ypos;     // which pre-expansion row we're on
} SampleData;

typedef struct {
    uint8_t  symbols[256];
    uint16_t max_codes[18];
    uint32_t delta[17];
    uint16_t lookups[1 << LOOKAHEAD_BITS];  // bit number of symbol + symbol
} HuffmanLookupTable;

typedef struct {
    HuffmanLookupTable huff_dc[4];   // 2 huffman dc tables for YCrCb
    HuffmanLookupTable huff_ac[4];   // 2 huffman ac tables for YCrCb
    uint16_t dequant[4][64];         // 2 quantization tables for YCrCb

    // interleaved MCUs
    uint32_t hsampling_max, vsampling_max;
    uint32_t mcu_width, mcu_height;
    uint32_t mcus_x, mcus_y;

    struct {
        uint32_t id;            // component id: Y(1), Cb(2), Cr(3), I(4), Q(5)
        uint32_t hsampling, vsampling; // horizontal/vertical sampling rate.
        uint32_t quant_id;      // quantification table id
        uint32_t dc_id, ac_id;  // table id of dc/ac
        int32_t dc_pred;

        uint32_t x, y, w2, h2, coeff_w;
        uint8_t *data;    // sequentially stored mcu data of YCrCb
        uint8_t *line_buffer;
        int16_t *coeff;   // progressive only
    } img_comp[4];

    uint64_t code_buffer;  // jpeg entropy-coded buffer
    uint32_t code_bits;    // number of valid bits
    uint8_t marker;        // marker seen while filling entropy buffer
    uint32_t nomore;       // flag if we saw a marker so must stop

    bool progressive;
    uint32_t index_start;
    uint32_t index_end;
    uint32_t succ_high;
    uint32_t succ_low;
    uint32_t eob_run;
    uint32_t jfif;
    int32_t app14_color_transform;  // Adobe APP14 tag
    int32_t rgb;

    uint32_t components;        // Gray(1), YCbCr/YIQ(3), CMYK(4)
    uint32_t scan_n, order[4];  // scan_n: number of components, order[]: component id
    int32_t restart_interval, todo;
} JpegDecodeData;

class YCrCb2BGR {
  public:
    YCrCb2BGR(uint32_t width, uint32_t channels);
    ~YCrCb2BGR();

    void convertBGR(uint8_t const *y, uint8_t const *pcb, uint8_t const *pcr,
                    uint8_t *dst);

  private:
    void process8Elements(uint8_t const *y, uint8_t const *pcb,
                          uint8_t const *pcr, uint32_t index, __m128i &b16s,
                          __m128i &g16s, __m128i &r16s) const;

  private:
    uint32_t width_, channels_;
    __m128i sign_flip_;
    __m128i cr_const0_;
    __m128i cr_const1_;
    __m128i cb_const0_;
    __m128i cb_const1_;
    __m128i y_bias_;
    __m128i b16s0_, b16s1_;
    __m128i g16s0_, g16s1_;
    __m128i r16s0_, r16s1_;
};

class JpegDecoder : public ImageDecoder {
  public:
    JpegDecoder(BytesReader& file_data);
    ~JpegDecoder();

    bool readHeader() override;
    bool decodeData(uint32_t stride, uint8_t* image) override;

  private:
    bool parseAPP0(JpegDecodeData *jpeg);
    bool parseAPP14(JpegDecodeData *jpeg);
    bool parseSOF(JpegDecodeData *jpeg);
    bool parseSOS(JpegDecodeData *jpeg);
    bool parseDQT(JpegDecodeData *jpeg);
    bool buildHuffmanTable(HuffmanLookupTable *huffman_table, uint32_t *count);
    bool parseDHT(JpegDecodeData *jpeg);
    bool parseCOM();
    bool parseDRI(JpegDecodeData *jpeg);
    bool parseDNL();
    bool processOtherSegments(int32_t marker);
    bool processSegments(JpegDecodeData *jpeg, uint8_t marker);

    void freeComponents(JpegDecodeData *jpeg, uint32_t ncomp);
    void resetJpegDecoder(JpegDecodeData *jpeg);
    uint8_t getMarker(JpegDecodeData *jpeg);
    int32_t decodeHuffmanData(JpegDecodeData *jpeg,
                              HuffmanLookupTable *huffman_table);
    bool decodeBlock(JpegDecodeData *jpeg, int16_t decoded_data[64],
                     HuffmanLookupTable *huffman_dc,
                     HuffmanLookupTable *huffman_ac,
                     uint32_t component_id, uint16_t *dequant_table);
    void idctDecodeBlock(uint8_t *output, int32_t out_stride, int16_t data[64]);
    void idctprocess0(JpegDecodeData *jpeg, int16_t* buffer, uint8_t* output,
                      uint32_t height_begin, uint32_t height_end,
                      uint32_t width, uint32_t width2);
    bool decodeProgressiveDCBlock(JpegDecodeData *jpeg,
                                  int16_t decoded_data[64],
                                  HuffmanLookupTable *huffman_dc,
                                  uint32_t component_id, uint32_t succ_value);
    bool decodeProgressiveACBlock(JpegDecodeData *jpeg,
                                  int16_t decoded_data[64],
                                  HuffmanLookupTable *huffman_ac);
    bool parseEntropyCodedData(JpegDecodeData *jpeg);
    void dequantizeData(int16_t *data, uint16_t *dequant_table);
    void idctprocess1(JpegDecodeData *jpeg, uint32_t height_begin,
                      uint32_t height_end, uint32_t width, uint32_t comp_id,
                      uint32_t width2);
    void finishProgressiveJpeg(JpegDecodeData *jpeg);
    bool convertColor(int32_t stride, uint8_t* image);

  private:
    BytesReader* file_data_;
    JpegDecodeData* jpeg_;
    YCrCb2BGR* ycrcb2bgr_;
    uint32_t hardware_threads_;
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMGCODECS_JPEG_H_
