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

#ifndef __ST_HPC_PPL_CV_X86_IMGCODECS_PNG_H_
#define __ST_HPC_PPL_CV_X86_IMGCODECS_PNG_H_

#include "imagecodecs.h"
#include "bytesreader.h"
#include "crc32.h"

#include <string>

namespace ppl {
namespace cv {
namespace x86 {

// huffman decoding acceleration
#define FAST_BITS 9  // larger handles more cases; smaller stomps less cache
// #define BIT_BUFFER_SIZE0 64
#define SHIFT_SIZE 56
// fast-way is faster to check than jpeg huffman, but slow way is slower
#define ZLIB_FAST_BITS  9 // accelerate all cases in default tables
#define ZLIB_FAST_MASK  ((1 << ZLIB_FAST_BITS) - 1)
#define SYMBOL_NUMBER 288 // number of symbols in literal/length alphabet

enum ColorTypes {
    GRAY = 0,
    TRUE_COLOR = 2,
    INDEXED_COLOR = 3,
    GRAY_WITH_ALPHA = 4,
    TRUE_COLOR_WITH_ALPHA = 6,
};

enum CompressionMethods {
    DEFLATE = 0,
};

enum FilterMethods {
    ADAPTIVE = 0,
};

enum InterlaceMethods {
    NO_INTRLACE = 0,
    ADAM7 = 1,
};

enum EncodingMethod {
    STORED_SECTION = 0,
    STATIC_HUFFMAN = 1,
    DYNAMIC_HUFFMAN = 2,
    RESERVED = 3,
};

struct DataChunk {
    uint32_t length;
    uint32_t type;
    // uint8_t type[4];
    uint32_t crc;
};

struct png_time {
   uint16_t year; /* full year, as in, 1995 */
   uint8_t month;   /* month of year, 1 - 12 */
   uint8_t day;     /* day of month, 1 - 31 */
   uint8_t hour;    /* hour of day, 0 - 23 */
   uint8_t minute;  /* minute of hour, 0 - 59 */
   uint8_t second;  /* second of minute, 0 - 60 (for leap seconds) */
};

// zlib-style huffman encoding
// (jpegs packs from left, zlib from right, so can't share code)
struct ZlibHuffman {
   uint16_t fast[1 << ZLIB_FAST_BITS];
   uint16_t firstcode[16];
   uint16_t firstsymbol[16];
   int32_t maxcode[17];
   uint8_t size[SYMBOL_NUMBER];
   uint16_t value[SYMBOL_NUMBER];
};

struct ZlibBuffer {
    uint32_t bit_number;
    uint64_t code_buffer;

    // deflate stream/zlib block info
    bool is_final_block;
    EncodingMethod encoding_method;
    uint32_t window_size;
    ZlibHuffman length_huffman, distance_huffman;
};

struct PngInfo {
    DataChunk current_chunk;
    uint8_t* palette;
    uint32_t palette_length;
    uint8_t alpha_values[6];
    uint8_t chunk_status;
    uint8_t* decompressed_image_start;
    uint8_t* decompressed_image_end;
    uint8_t* decompressed_image;
    uint8_t* defiltered_image;
    ZlibBuffer zlib_buffer;
    bool fixed_huffman_done;
    bool header_after_idat;
    uint32_t idat_count;
};

class PngDecoder : public ImageDecoder {
  public:
    PngDecoder(BytesReader& file_data);
    ~PngDecoder();

    bool readHeader() override;
    bool decodeData(uint32_t stride, uint8_t* image) override;

  private:
    void getChunkHeader(PngInfo& png_info);
    bool parseIHDR(PngInfo& png_info);
    bool parsetIME(PngInfo& png_info);
    bool parsezTXt(PngInfo& png_info);
    bool parsetEXt(PngInfo& png_info);
    bool parseiTXt(PngInfo& png_info);
    bool parsepHYs(PngInfo& png_info);
    bool parsesPLT(PngInfo& png_info);
    bool parseiCCP(PngInfo& png_info);
    bool parsesRGB(PngInfo& png_info);
    bool parsesBIT(PngInfo& png_info);
    bool parsegAMA(PngInfo& png_info);
    bool parsecHRM(PngInfo& png_info);
    bool parsePLTE(PngInfo& png_info);
    bool parsetRNS(PngInfo& png_info);
    bool parsehIST(PngInfo& png_info);
    bool parsebKGD(PngInfo& png_info);
    bool parseIDATs(PngInfo& png_info, uint8_t* image, uint32_t stride);
    bool parseIEND(PngInfo& png_info);
    bool parsetUnknownChunk(PngInfo& png_info);

    void releaseSource();
    std::string getChunkName(uint32_t chunk_type);
    bool setCrc32();
    bool isCrcCorrect();

    bool parseDeflateHeader();
    bool fillBits(ZlibBuffer *z);
    uint32_t getNumber(ZlibBuffer *z, int n);
    bool parseZlibUncompressedBlock(PngInfo& png_info);
    bool buildHuffmanCode(ZlibHuffman *z, const uint8_t *sizelist, int num);
    int32_t huffmanDecodeSlowly(ZlibBuffer *zlib_buffer,
                                ZlibHuffman *huffman_coding);
    int32_t huffmanDecode(ZlibBuffer *zlib_buffer,
                          ZlibHuffman *huffman_coding);
    bool computeDynamicHuffman(ZlibBuffer* a);
    bool decodeHuffmanData(ZlibBuffer* a);
    bool inflateImage(PngInfo& png_info, uint8_t* image, uint32_t stride);
    bool deFilterImage(PngInfo& png_info, uint8_t* image, uint32_t stride);
    bool deFilterImageTrueColor(PngInfo& png_info, uint8_t* image,
                                uint32_t stride);
    bool computeTransparency(PngInfo& png_info, uint8_t* image, uint32_t stride);
    bool expandPalette(PngInfo& png_info, uint8_t* image, uint32_t stride);

  private:
    BytesReader* file_data_;
    PngInfo png_info_;
    Crc32 crc32_;
    uint8_t bit_depth_;
    uint8_t color_type_;
    uint8_t compression_method_;
    uint8_t filter_method_;
    uint8_t interlace_method_;
    uint32_t encoded_channels_;  // ?
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMGCODECS_PNG_H_
