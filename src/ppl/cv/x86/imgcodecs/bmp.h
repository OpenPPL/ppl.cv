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

#ifndef __ST_HPC_PPL_CV_X86_IMGCODECS_BMP_H_
#define __ST_HPC_PPL_CV_X86_IMGCODECS_BMP_H_

#include "imagecodecs.h"
#include "bytesreader.h"

#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

struct PaletteEntry {
    uchar b, g, r, a;
};

enum BmpCompression {
    BMP_RGB = 0,
    BMP_RLE8 = 1,
    BMP_RLE4 = 2,
    BMP_BITFIELDS = 3,
};

enum Origin {
    ORIGIN_TL = 0,
    ORIGIN_BL = 1,
};

class BmpDecoder : public ImageDecoder {
  public:
    BmpDecoder(BytesReader& file_data);
    ~BmpDecoder();

    bool readHeader() override;
    bool decodeData(uint32_t stride, uint8_t* image) override;

  private:
    bool checkPaletteColor(PaletteEntry* palette, int bits_per_pixel);
    void BGR2Gray(const uchar* bgr, uchar* gray, int width);
    void BGRA2Gray(const uchar* bgra, uchar* gray, int width);
    void BGRA2BGR(const uchar* bgra, uchar* bgr, int width);
    void BGR5552Gray(const uchar* bgr555, uchar* gray, int width);
    void BGR5552BGR(const uchar* bgr555, uchar* bgr, int width);
    void BGR5652Gray(const uchar* bgr565, uchar* gray, int width);
    void BGR5652BGR(const uchar* bgr565, uchar* bgr, int width);
    void PaletteToGray(const PaletteEntry* color_palette, uchar* gray_palette,
                       int entries);
    void fillColorRow1(uchar* data, uchar* indices, int length,
                       PaletteEntry* palette);
    uchar* fillColorRow4(uchar* data, uchar* indices, int length,
                         PaletteEntry* palette);
    uchar* fillColorRow8(uchar* data, uchar* indices, int length,
                         PaletteEntry* palette);
    void fillGrayRow1(uchar* data, uchar* indices, int length, uchar* palette);
    uchar* fillGrayRow4(uchar* data, uchar* indices, int length,
                        uchar* palette);
    uchar* fillGrayRow8(uchar* data, uchar* indices, int length,
                        uchar* palette);
    uchar* fillUniColor(uchar* data, uchar*& line_end, int step, uint width3,
                        uint& y, uint height, int count3, PaletteEntry color);
    uchar* fillUniGray(uchar* data, uchar*& line_end, int step, uint width,
                       uint& y, uint height, int count, uchar color);
    void maskBGRA(uchar* dst, uchar* src, int number);

  private:
    BytesReader* file_data_;
    int data_offset_;
    int bits_per_pixel_;
    BmpCompression compression_type_;
    int used_colors_;
    PaletteEntry color_palette_[256];
    Origin origin_;
    uint rgba_mask_[4];
    int rgba_bit_offset_[4];
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMGCODECS_BMP_H_
