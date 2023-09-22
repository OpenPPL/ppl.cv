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

#include <emmintrin.h>

#include <stdint.h>

namespace ppl {
namespace cv {
namespace x86 {

// huffman decoding acceleration
#define FAST_BITS 9  // larger handles more cases; smaller stomps less cache
#define BUFFER_BYTES 8
#define BUFFER_BITS 64
// #define BUFFER_BYTES 4
// #define BUFFER_BITS 32
#define SHIFT_BYTES 3
#define JPEG_SHIFT_SIZE 24
// #define BUFFER_BITS 64
// #define JPEG_SHIFT_SIZE 56

typedef uint8_t *(*resampleRow)(uint8_t *out, uint8_t *in0, uint8_t *in1,
                                int32_t width, int32_t hs);

typedef struct {
    resampleRow resample;
    uint8_t *line0, *line1;
    int32_t hs, vs;   // expansion factor in each axis
    int32_t w_lores;  // horizontal pixels pre-expansion
    int32_t ystep;    // how far through vertical expansion we are
    int32_t ypos;     // which pre-expansion row we're on
} SampleData;

typedef struct {
    uint8_t  fast_indices[1 << FAST_BITS];
    // weirdly, repacking this into AoS is a 10% speed loss, instead of a win
    uint8_t  bit_lengths[257];  // bit lengths of code, remove?
    uint16_t codes[256];        // bit code, remove?
    uint8_t  symbols[256];      // the stored symbol/code value
    uint32_t maxcode[18];
    int32_t  delta[17];         // old 'firstsymbol' - old 'firstcode'
} HuffmanLookupTable;

typedef struct {
    HuffmanLookupTable huff_dc[2];       // 2 huffman dc tables for YCrCb
    HuffmanLookupTable huff_ac[2];       // 2 huffman ac tables for YCrCb
    uint16_t dequant[2][64];             // 2 quantization tables for YCrCb
    int16_t fast_ac[2][1 << FAST_BITS];  // 1 fast ac tables for YCrCb

    // sizes for components, interleaved MCUs
    int32_t img_h_max, img_v_max;
    int32_t mcus_x, mcus_y;
    int32_t mcu_width, mcu_height;

    // definition of jpeg image component
    struct {
        int32_t id;   // color_id, channel id
        int32_t hsampling, vsampling; // horizontal/vertical sampling rate.
        int32_t quant_id;   // quantification table id
        int32_t huffman_dc_id, huffman_ac_id;  // table id of huffman dc/ ac
        int32_t dc_pred;

        int32_t x, y, w2, h2;
        uint8_t *data;  // sequentially stored mcu data of YCrCb
        void *raw_data, *raw_coeff;
        uint8_t *line_buffer;
        int16_t *coeff;   // progressive only
        int32_t coeff_w, coeff_h; // number of 8x8 coefficient blocks
    } img_comp[4];

    uint64_t code_buffer; // jpeg entropy-coded buffer
    // uint32_t code_buffer; // jpeg entropy-coded buffer
    int32_t code_bits;    // number of valid bits
    uint8_t marker;       // marker seen while filling entropy buffer
    int32_t nomore;       // flag if we saw a marker so must stop

    int32_t progressive;
    int32_t spec_start;
    int32_t spec_end;
    int32_t succ_high;
    int32_t succ_low;
    int32_t eob_run;
    int32_t jfif;
    int32_t app14_color_transform; // Adobe APP14 tag
    int32_t rgb;

    int32_t components;  // z->s->img_n
    int32_t scan_n, order[4];  // scan_n: number of components, order[]: component id
    int32_t restart_interval, todo;

    // kernels
    void (*idctBlockKernel)(uint8_t *out, int32_t out_stride, int16_t data[64]);
    void (*YCbCr2BGRKernel)(uint8_t *out, const uint8_t *y, const uint8_t *pcb,
                            const uint8_t *pcr, int32_t count, int32_t step);
    uint8_t *(*resampleRowHV2Kernel)(uint8_t *out, uint8_t *in_near,
                                     uint8_t *in_far, int32_t w, int32_t hs);
} JpegDecodeData;

class YCrCb2BGR_i {
  public:
    YCrCb2BGR_i(int32_t _width, int32_t _channels);
    ~YCrCb2BGR_i();

    // void convertBGR(uint8_t *dst, uint8_t const *y, uint8_t const *pcb,
    //                 uint8_t const *pcr, int32_t count, int32_t step);
    void convertBGR(uint8_t const *y, uint8_t const *pcb, uint8_t const *pcr,
                    uint8_t *dst);
  private:
    void process8Elements(uint8_t const *y, uint8_t const *pcb,
                          uint8_t const *pcr, uint32_t index, __m128i &b16s,
                          __m128i &g16s, __m128i &r16s) const;

  private:
    int32_t width, channels;
    __m128i signflip;
    __m128i cr_const0;
    __m128i cr_const1;
    __m128i cb_const0;
    __m128i cb_const1;
    __m128i y_bias;
    __m128i b16s0, b16s1;
    __m128i g16s0, g16s1;
    __m128i r16s0, r16s1;
};

class JpegDecoder : public ImageDecoder {
  public:
    JpegDecoder(BytesReader& file_data);
    ~JpegDecoder();

    bool readHeader() override;
    bool decodeData(uint32_t stride, uint8_t* image) override;

  private:
    bool buildHuffmanTable(HuffmanLookupTable *huffman_table, int32_t *count);
    void buildFastAC(int16_t *fast_ac, HuffmanLookupTable *huffman_data);
    bool parseAPP0(JpegDecodeData *jpeg);
    bool parseAPP14(JpegDecodeData *jpeg);
    bool parseSOF(JpegDecodeData *jpeg);
    bool parseSOS(JpegDecodeData *jpeg);
    bool parseDQT(JpegDecodeData *jpeg);
    bool parseDHT(JpegDecodeData *jpeg);
    bool parseCOM();
    bool parseDRI(JpegDecodeData *jpeg);
    bool parseDNL();
    bool processOtherSegments(int32_t marker);
    bool processSegments(JpegDecodeData *jpeg, uint8_t marker);

    void setJpegFunctions(JpegDecodeData *jpeg);
    void freeComponents(JpegDecodeData *jpeg, int32_t ncomp);
    void resetJpegDecoder(JpegDecodeData *jpeg);
    // void growBitBuffer(JpegDecodeData *jpeg);
    uint8_t getMarker(JpegDecodeData *jpeg);
    uint8_t getMarker1(JpegDecodeData *jpeg);
    // int32_t getBits(JpegDecodeData *jpeg, int32_t n);
    // int32_t getBit(JpegDecodeData *jpeg);
    // int32_t extendReceive(JpegDecodeData *jpeg, int32_t value_bit_length);
    int32_t decodeHuffmanData(JpegDecodeData *jpeg,
                              HuffmanLookupTable *huffman_table);
    bool decodeProgressiveDCBlock(JpegDecodeData *jpeg,
                                  int16_t decoded_data[64],
                                  HuffmanLookupTable *huffman_dc, int32_t b);
    bool decodeProgressiveACBlock(JpegDecodeData *jpeg,
                                  int16_t decoded_data[64],
                                  HuffmanLookupTable *huffman_ac,
                                  int16_t *fast_ac);
    bool decodeBlock(JpegDecodeData *jpeg, int16_t decoded_data[64],
                     HuffmanLookupTable *huffman_dc,
                     HuffmanLookupTable *huffman_ac, int16_t *fast_ac,
                     int32_t component_id, uint16_t *dequant_table);
    bool parseEntropyCodedData(JpegDecodeData *jpeg);
    bool sampleConvertColor(int32_t stride, uint8_t* image);
    void dequantizeData(int16_t *data, uint16_t *dequant_table);
    void finishProgressiveJpeg(JpegDecodeData *jpeg);

  private:
    BytesReader* file_data_;
    JpegDecodeData* jpeg_;
    YCrCb2BGR_i* ycrcb2bgr_;
    size_t shortcode;
    size_t longcode;
    size_t fastcode;
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMGCODECS_JPEG_H_

