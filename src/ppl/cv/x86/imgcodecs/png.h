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
#define BIT_BUFFER_SIZE 32
#define SHIFT_SIZE 24
// #define BIT_BUFFER_SIZE 64
// #define SHIFT_SIZE 56

// fast-way is faster to check than jpeg huffman, but slow way is slower
#define STBI__ZFAST_BITS  9 // accelerate all cases in default tables
#define STBI__ZFAST_MASK  ((1 << STBI__ZFAST_BITS) - 1)
#define STBI__ZNSYMS 288 // number of symbols in literal/length alphabet

// zlib-style huffman encoding
// (jpegs packs from left, zlib from right, so can't share code)
typedef struct
{
   uint16_t fast[1 << STBI__ZFAST_BITS];
   uint16_t firstcode[16];
   int32_t maxcode[17];
   uint16_t firstsymbol[16];
   uint8_t  size[STBI__ZNSYMS];
   uint16_t value[STBI__ZNSYMS];
} stbi__zhuffman;

typedef struct
{
   uint8_t *zbuffer, *zbuffer_end;
   int32_t num_bits;
   uint32_t code_buffer;

   char *zout;
   char *zout_start;
   char *zout_end;
   int   z_expandable;

   stbi__zhuffman z_length, z_distance;
} stbi__zbuf;

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

enum FilterTypes {
    FILTER_NONE = 0,
    FILTER_SUB = 1,
    FILTER_UP = 2,
    FILTER_AVERAGE = 3,
    FILTER_PAETH = 4,
    // synthetic filters used for first scanline to avoid needing a dummy row of 0s
    // FILTER_avg_first,
    // FILTER_paeth_first
};

enum InterlaceMethods {
    NO_INTRLACE = 0,
    ADAM7 = 1,
};

struct PlteEntry {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct DataChunk {
    uint32_t length;    // horizontal pixels pre-expansion
    uint32_t type;
    // uint8_t type[4];    // how far through vertical expansion we are
    uint32_t crc;       // which pre-expansion row we're on
};

typedef struct png_time_struct
{
   uint16_t year; /* full year, as in, 1995 */
   uint8_t month;   /* month of year, 1 - 12 */
   uint8_t day;     /* day of month, 1 - 31 */
   uint8_t hour;    /* hour of day, 0 - 23 */
   uint8_t minute;  /* minute of hour, 0 - 59 */
   uint8_t second;  /* second of minute, 0 - 60 (for leap seconds) */
} png_time;

struct PngInfo {
    DataChunk current_chunk;
    // uint8_t bit_depth;
    PlteEntry* color_palette;
    uint32_t palette_length;
    uint8_t* alpha_palette;
    uint8_t alpha_values[6];
    uint8_t chunk_status;

};


class PngDecoder : public ImageDecoder {
  public:
    PngDecoder(BytesReader& file_data);
    ~PngDecoder();

    bool readHeader() override;
    bool decodeData(int32_t stride, uint8_t* image) override;

  private:
    void getChunkHeader(PngInfo& png_info);
    bool parseIHDR(PngInfo& png_info);
    bool parseIDAT(PngInfo& png_info);
    bool parsePLTE(PngInfo& png_info);
    bool parseIEND(PngInfo& png_info);
    bool parsetRNS(PngInfo& png_info);
    bool parsecHRM(PngInfo& png_info);
    bool parsegAMA(PngInfo& png_info);
    bool parseiCCP(PngInfo& png_info);
    bool parsesBIT(PngInfo& png_info);
    bool parsesRGB(PngInfo& png_info);
    bool parsetEXt(PngInfo& png_info);
    bool parsezTXt(PngInfo& png_info);
    bool parseiTXt(PngInfo& png_info);
    bool parsebKGD(PngInfo& png_info);
    bool parsehIST(PngInfo& png_info);
    bool parsepHYs(PngInfo& png_info);
    bool parsesPLT(PngInfo& png_info);
    bool parsetIME(PngInfo& png_info);
    bool parsetUnknownChunk(PngInfo& png_info);

    void releaseSource();
    std::string getChunkName(uint32_t chunk_type);
    bool setCrc32();
    bool isCrcCorrect();

  private:
    BytesReader* file_data_;
    Crc32 crc32_;
    PngInfo png_info_;
    uint8_t bit_depth_;
    uint8_t color_type_;
    uint8_t compression_method_;
    uint8_t filter_method_;
    uint8_t interlace_method_;
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMGCODECS_PNG_H_
