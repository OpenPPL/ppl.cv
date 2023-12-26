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

#include "bmp.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ppl/cv/types.h"
#include "ppl/common/log.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace x86 {

#define SCALE 14

const int cR = (int)(0.299f * (1 << SCALE) + 0.5f);
const int cG = (int)(0.587f * (1 << SCALE) + 0.5f);
const int cB = ((1 << SCALE) - cR - cG);

inline int descale(int x, int n) {
    return (((x) + (1 << ((n)-1))) >> (n));
}

inline void writePixel(uchar* pixel, PaletteEntry& color) {
    pixel[0] = color.b;
    pixel[1] = color.g;
    pixel[2] = color.r;
}

BmpDecoder::BmpDecoder(BytesReader& file_data) {
    file_data_ = &file_data;
    data_offset_ = -1;
    bits_per_pixel_ = 0;
    compression_type_ = BMP_RGB;
    used_colors_ = 0;
    origin_ = ORIGIN_TL;
}

BmpDecoder::~BmpDecoder() {
}

bool BmpDecoder::checkPaletteColor(PaletteEntry* palette, int bits_per_pixel) {
    int length = 1 << bits_per_pixel;

    for (int i = 0; i < length; i++) {
        if (palette[i].b != palette[i].g || palette[i].b != palette[i].r) {
            return true;
        }
    }

    return false;
}

void BmpDecoder::BGR2Gray(const uchar* bgr, uchar* gray, int width) {
    for (int i = 0; i < width; i++, bgr += 3) {
        int value = descale(bgr[0] * cB + bgr[1] * cG + bgr[2] * cR,
                            SCALE);
        gray[i] = (uchar)value;
    }
}

void BmpDecoder::BGRA2Gray(const uchar* bgra, uchar* gray, int width) {
    for (int i = 0; i < width; i++, bgra += 4) {
        int value = descale(bgra[0] * cB + bgra[1] * cG + bgra[2] * cR,
                            SCALE);
        gray[i] = (uchar)value;
    }
}

void BmpDecoder::BGRA2BGR(const uchar* bgra, uchar* bgr, int width) {
    for (int i = 0; i < width; i++, bgra += 4, bgr += 3) {
        uchar value0 = bgra[0];
        uchar value1 = bgra[1];
        uchar value2 = bgra[2];
        bgr[0] = value0;
        bgr[1] = value1;
        bgr[2] = value2;
    }
}

void BmpDecoder::BGR5552Gray(const uchar* bgr555, uchar* gray, int width) {
    for (int i = 0; i < width; i++) {
        int value = descale(((((ushort*)bgr555)[i] << 3) & 0xf8) * cB +
                            ((((ushort*)bgr555)[i] >> 2) & 0xf8) * cG +
                            ((((ushort*)bgr555)[i] >> 7) & 0xf8) * cR, SCALE);
        gray[i] = (uchar)value;
    }
}

void BmpDecoder::BGR5552BGR(const uchar* bgr555, uchar* bgr, int width) {
    for (int i = 0; i < width; i++, bgr += 3) {
        int value0 = (((ushort*)bgr555)[i] << 3) & 0xf8;
        int value1 = (((ushort*)bgr555)[i] >> 2) & 0xf8;
        int value2 = (((ushort*)bgr555)[i] >> 7) & 0xf8;
        bgr[0] = (uchar)value0;
        bgr[1] = (uchar)value1;
        bgr[2] = (uchar)value2;
    }
}

void BmpDecoder::BGR5652Gray(const uchar* bgr565, uchar* gray, int width) {
    for (int i = 0; i < width; i++) {
        int value = descale(((((ushort*)bgr565)[i] << 3) & 0xf8) * cB +
                            ((((ushort*)bgr565)[i] >> 3) & 0xfc) * cG +
                            ((((ushort*)bgr565)[i] >> 8) & 0xf8) * cR, SCALE);
        gray[i] = (uchar)value;
    }
}

void BmpDecoder::BGR5652BGR(const uchar* bgr565, uchar* bgr, int width) {
    for (int i = 0; i < width; i++, bgr += 3) {
        int value0 = (((ushort*)bgr565)[i] << 3) & 0xf8;
        int value1 = (((ushort*)bgr565)[i] >> 3) & 0xfc;
        int value2 = (((ushort*)bgr565)[i] >> 8) & 0xf8;
        bgr[0] = (uchar)value0;
        bgr[1] = (uchar)value1;
        bgr[2] = (uchar)value2;
    }
}

void BmpDecoder::PaletteToGray(const PaletteEntry* color_palette,
                               uchar* gray_palette, int entries) {
    for (int i = 0; i < entries; i++) {
        BGR2Gray((uchar*)(color_palette + i), gray_palette + i, 1);
    }
}

void BmpDecoder::fillColorRow1(uchar* data, uchar* indices, int length,
                               PaletteEntry* palette) {
    uchar* end = data + length * 3;
    const PaletteEntry p0 = palette[0], p1 = palette[1];

    while ((data += 24) < end) {
        int index = *indices++;
        *((PaletteEntry*)(data - 24)) = (index & 128) ? p1 : p0;
        *((PaletteEntry*)(data - 21)) = (index & 64) ? p1 : p0;
        *((PaletteEntry*)(data - 18)) = (index & 32) ? p1 : p0;
        *((PaletteEntry*)(data - 15)) = (index & 16) ? p1 : p0;
        *((PaletteEntry*)(data - 12)) = (index & 8) ? p1 : p0;
        *((PaletteEntry*)(data - 9)) = (index & 4) ? p1 : p0;
        *((PaletteEntry*)(data - 6)) = (index & 2) ? p1 : p0;
        *((PaletteEntry*)(data - 3)) = (index & 1) ? p1 : p0;
    }

    int index = indices[0];
    for (data -= 24; data < end; data += 3, index += index) {
        PaletteEntry color = (index & 128) ? p1 : p0;
        writePixel(data, color);
    }
}

uchar* BmpDecoder::fillColorRow4(uchar* data, uchar* indices, int length,
                                 PaletteEntry* palette) {
    uchar* end = data + length * 3;

    while ((data += 6) < end) {
        int index = *indices++;
        *((PaletteEntry*)(data - 6)) = palette[index >> 4];
        *((PaletteEntry*)(data - 3)) = palette[index & 15];
    }

    int index = indices[0];
    PaletteEntry color = palette[index >> 4];
    writePixel(data - 6, color);

    if (data == end) {
        color = palette[index & 15];
        writePixel(data - 3, color);
    }

    return end;
}

uchar* BmpDecoder::fillColorRow8(uchar* data, uchar* indices, int length,
                                 PaletteEntry* palette) {
    uchar* end = data + length * 3;

    while ((data += 3) < end) {
        *((PaletteEntry*)(data - 3)) = palette[*indices++];
    }

    PaletteEntry color = palette[indices[0]];
    writePixel(data - 3, color);

    return data;
}

void BmpDecoder::fillGrayRow1(uchar* data, uchar* indices, int length,
                              uchar* palette) {
    uchar* end = data + length;
    uchar p0 = palette[0], p1 = palette[1];

    while ((data += 8) < end) {
        int index = *indices++;
        data[-8] = (index & 128) ? p1 : p0;
        data[-7] = (index & 64) ? p1 : p0;
        data[-6] = (index & 32) ? p1 : p0;
        data[-5] = (index & 16) ? p1 : p0;
        data[-4] = (index & 8) ? p1 : p0;
        data[-3] = (index & 4) ? p1 : p0;
        data[-2] = (index & 2) ? p1 : p0;
        data[-1] = (index & 1) ? p1 : p0;
    }

    int index = indices[0];
    for (data -= 8; data < end; data++, index += index) {
        uchar value = (index & 128) ? p1 : p0;
        *data = value;
    }
}

uchar* BmpDecoder::fillGrayRow4(uchar* data, uchar* indices, int length,
                                uchar* palette) {
    uchar* end = data + length;

    while ((data += 2) < end) {
        int index = *indices++;
        data[-2] = palette[index >> 4];
        data[-1] = palette[index & 15];
    }

    int index = indices[0];
    uchar value = palette[index >> 4];
    data[-2] = value;

    if (data == end) {
        value = palette[index & 15];
        data[-1] = value;
    }

    return end;
}

uchar* BmpDecoder::fillGrayRow8(uchar* data, uchar* indices, int length,
                                uchar* palette) {
    for (int i = 0; i < length; i++) {
        data[i] = palette[indices[i]];
    }

    return data + length;
}

uchar* BmpDecoder::fillUniColor(uchar* data, uchar*& line_end, int step,
                                uint width3, uint& y, uint height, int count3,
                                PaletteEntry color) {
    do {
        uchar* end = data + count3;

        if (end > line_end) {
            end = line_end;
        }

        count3 -= (int)(end - data);

        for (; data < end; data += 3) {
            writePixel(data, color);
        }

        if (data >= line_end) {
            line_end += step;
            data = line_end - width3;
            if (++y >= height) {
                break;
            }
        }
    } while (count3 > 0);

    return data;
}

uchar* BmpDecoder::fillUniGray(uchar* data, uchar*& line_end, int step,
                               uint width, uint& y, uint height, int count,
                               uchar color) {
    do {
        uchar* end = data + count;

        if (end > line_end) {
            end = line_end;
        }

        count -= (int)(end - data);

        for (; data < end; data++) {
            *data = color;
        }

        if (data >= line_end) {
            line_end += step;
            data = line_end - width;
            if (++y >= height) break;
        }
    } while (count > 0);

    return data;
}

void BmpDecoder::maskBGRA(uchar* dst, uchar* src, int number) {
    for (int i = 0; i < number; i++, dst += 4, src += 4) {
        uint data = *((uint*)src);
        dst[0] = (uchar)((rgba_mask_[2] & data) >> rgba_bit_offset_[2]);
        dst[1] = (uchar)((rgba_mask_[1] & data) >> rgba_bit_offset_[1]);
        dst[2] = (uchar)((rgba_mask_[0] & data) >> rgba_bit_offset_[0]);
        if (rgba_bit_offset_[3] >= 0) {
            dst[3] = (uchar)((rgba_mask_[3] & data) >> rgba_bit_offset_[3]);
        }
        else {
            dst[3] = 255;
        }
    }
}

bool BmpDecoder::readHeader() {
    bool result = false;
    bool colorful = false;

    file_data_->skipBytes(10);
    data_offset_ = file_data_->getDWordLittleEndian();

    int size = file_data_->getDWordLittleEndian();
    assert(size > 0);

    int width = 0;
    int height = 0;
    if (size >= 36) {
        width  = file_data_->getDWordLittleEndian();
        height = file_data_->getDWordLittleEndian();
        bits_per_pixel_ = file_data_->getDWordLittleEndian() >> 16;
        int bmp_compression = file_data_->getDWordLittleEndian();
        assert(bmp_compression >= 0 && bmp_compression <= BMP_BITFIELDS);
        compression_type_ = (BmpCompression)bmp_compression;
        file_data_->skipBytes(12);
        used_colors_ = file_data_->getDWordLittleEndian();

        if (bits_per_pixel_ == 32 && compression_type_ == BMP_BITFIELDS &&
            size >= 56) {
            file_data_->skipBytes(4);  //important colors

            memset(rgba_mask_, 0, sizeof(rgba_mask_));
            memset(rgba_bit_offset_, -1, sizeof(rgba_bit_offset_));
            for (int index_rgba = 0; index_rgba < 4; ++index_rgba) {
                uint mask = file_data_->getDWordLittleEndian();
                rgba_mask_[index_rgba] = mask;
                if (mask != 0) {
                    int bit_count = 0;
                    while (!(mask & 1)) {
                        mask >>= 1;
                        ++bit_count;
                    }
                    rgba_bit_offset_[index_rgba] = bit_count;
                }
            }
            file_data_->skipBytes(size - 56);
        }
        else {
            file_data_->skipBytes(size - 36);
        }

        if (width > 0 && height != 0 && (((bits_per_pixel_ == 1 ||
            bits_per_pixel_ == 4 || bits_per_pixel_ == 8 ||
            bits_per_pixel_ == 24 || bits_per_pixel_ == 32) &&
            compression_type_ == BMP_RGB) || ((bits_per_pixel_ == 16 ||
            bits_per_pixel_ == 32) && (compression_type_ == BMP_RGB ||
            compression_type_ == BMP_BITFIELDS)) || (bits_per_pixel_ == 4 &&
            compression_type_ == BMP_RLE4) || (bits_per_pixel_ == 8 &&
            compression_type_ == BMP_RLE8))) {
            colorful = true;
            result = true;

            if (bits_per_pixel_ <= 8) {
                assert(used_colors_ >= 0 && used_colors_ <= 256);
                memset(color_palette_, 0, sizeof(color_palette_));
                int number = (used_colors_ == 0 ? (1 << bits_per_pixel_) :
                              used_colors_) * 4;
                file_data_->getBytes(color_palette_, number);
                colorful = checkPaletteColor(color_palette_, bits_per_pixel_);
            }
            else if (bits_per_pixel_ == 16 &&
                     compression_type_ == BMP_BITFIELDS) {
                int red_mask   = file_data_->getDWordLittleEndian();
                int green_mask = file_data_->getDWordLittleEndian();
                int blue_mask  = file_data_->getDWordLittleEndian();

                if (blue_mask == 0x1f && green_mask == 0x3e0 &&
                    red_mask == 0x7c00) {
                    bits_per_pixel_ = 15;
                }
                else if (blue_mask == 0x1f && green_mask == 0x7e0 &&
                         red_mask == 0xf800) {
                }
                else {
                    result = false;
                }
            }
            else if (bits_per_pixel_ == 32 &&
                     compression_type_ == BMP_BITFIELDS) {
            }
            else if (bits_per_pixel_ == 16 && compression_type_ == BMP_RGB) {
                bits_per_pixel_ = 15;
            }
        }
    }
    else if (size == 12) {
        width  = file_data_->getWordLittleEndian();
        height = file_data_->getWordLittleEndian();
        bits_per_pixel_ = file_data_->getDWordLittleEndian() >> 16;
        compression_type_ = BMP_RGB;

        if (width > 0 && height != 0 && (bits_per_pixel_ == 1 ||
            bits_per_pixel_ == 4 || bits_per_pixel_ == 8 ||
            bits_per_pixel_ == 24 || bits_per_pixel_ == 32)) {
            if (bits_per_pixel_ <= 8) {
                uchar buffer[256 * 3];
                used_colors_ = 1 << bits_per_pixel_;
                file_data_->getBytes(buffer, used_colors_ * 3);
                for (int index = 0; index < used_colors_; index++) {
                    color_palette_[index].b = buffer[3 * index + 0];
                    color_palette_[index].g = buffer[3 * index + 1];
                    color_palette_[index].r = buffer[3 * index + 2];
                }
            }
            result = true;
        }
    }

    channels_ = colorful ? (bits_per_pixel_ == 32 ? 4 : 3) : 1;
    origin_ = height > 0 ? ORIGIN_BL : ORIGIN_TL;
    width_  = abs(width);
    height_ = abs(height);

    if (!result) {
        data_offset_ = -1;
        width_  = 0;
        height_ = 0;
    }

    return result;
}

bool BmpDecoder::decodeData(uint32_t stride, uint8_t* image) {
    if ((uint64_t)height_ * width_ * channels_ > (1 << 30)) {
        LOG(ERROR) << "BMP reader implementation doesn't support "
                   << "large images >= 1Gb";
        return false;
    }

    if (data_offset_ < 0) {
        return false;
    }

    bool result = false;
    bool colorful = channels_ > 1;
    uchar gray_palette[256] = {0};
    int src_pitch = ((width_ * (bits_per_pixel_ != 15 ? bits_per_pixel_ : 16)
                     + 31) >> 5) << 2;
    int channels = colorful ? 3 : 1;
    uint row, width3 = width_ * channels;

    size_t src_size = src_pitch + 32;
    uchar* src = new uchar[src_size];

    if (!colorful) {
        if (bits_per_pixel_ <= 8) {
            PaletteToGray(color_palette_, gray_palette, 1 << bits_per_pixel_);
        }
    }

    file_data_->setPosition(data_offset_);
    int bmp_stride = 0;
    if (origin_ == ORIGIN_BL) {
        image += (height_ - 1) * (size_t)stride;
        bmp_stride = -stride;
    }

    switch (bits_per_pixel_) {
    case 1:  // 1 BPP
        for (row = 0; row < height_; row++, image += bmp_stride) {
            file_data_->getBytes(src, src_pitch);
            if (colorful) {
                fillColorRow1(image, src, width_, color_palette_);
            }
            else {
                fillGrayRow1(image, src, width_, gray_palette);
            }
        }
        result = true;
        break;

    case 4:  // 4 BPP
        if (compression_type_ == BMP_RGB) {
            for (row = 0; row < height_; row++, image += bmp_stride) {
                file_data_->getBytes(src, src_pitch);
                if (colorful) {
                    fillColorRow4(image, src, width_, color_palette_);
                }
                else {
                    fillGrayRow4(image, src, width_, gray_palette);
                }
            }
            result = true;
        }
        else if (compression_type_ == BMP_RLE4) {
            uchar* line_end = image + width3;
            row = 0;

            for(;;) {
                int code = file_data_->getWordLittleEndian();
                int length = code & 255;
                code >>= 8;
                if (length != 0) {  // encoded mode
                    PaletteEntry colors[2];
                    uchar gray_color[2];
                    int index = 0;

                    colors[0] = color_palette_[code >> 4];
                    colors[1] = color_palette_[code & 15];
                    gray_color[0] = gray_palette[code >> 4];
                    gray_color[1] = gray_palette[code & 15];

                    uchar* end = image + length * channels;
                    if (end > line_end) {
                        goto decode_rle4_bad;
                    }
                    do {
                        if (colorful) {
                            writePixel(image, colors[index]);
                        }
                        else {
                            *image = gray_color[index];
                        }
                        index ^= 1;
                    } while ((image += channels) < end);
                }
                else if (code > 2) {  // absolute mode
                    if (image + code * channels > line_end) {
                        goto decode_rle4_bad;
                    }
                    int size = (((code + 1)>>1) + 1) & (~1);
                    assert((size_t)size < src_size);
                    file_data_->getBytes(src, size);
                    if (colorful) {
                        image = fillColorRow4(image, src, code, color_palette_);
                    }
                    else {
                        image = fillGrayRow4(image, src, code, gray_palette);
                    }
                }
                else {
                    int x_shift3 = (int)(line_end - image);

                    if (code == 2) {
                        x_shift3 = file_data_->getByte() * channels;
                        file_data_->getByte();
                    }

                    if (colorful) {
                        image = fillUniColor(image, line_end, bmp_stride, width3,
                                             row, height_, x_shift3,
                                             color_palette_[0]);
                    }
                    else {
                        image = fillUniGray(image, line_end, bmp_stride, width3,
                                            row, height_, x_shift3,
                                            gray_palette[0]);
                    }

                    if (row >= height_) {
                        break;
                    }
                }
            }

            result = true;
decode_rle4_bad: ;
        }
        break;

    case 8:  // 8 BPP
        if (compression_type_ == BMP_RGB) {
            for (row = 0; row < height_; row++, image += bmp_stride) {
                file_data_->getBytes(src, src_pitch);
                if (colorful) {
                    fillColorRow8(image, src, width_, color_palette_);
                }
                else {
                    fillGrayRow8(image, src, width_, gray_palette);
                }
            }
            result = true;
        }
        else if (compression_type_ == BMP_RLE8) {  // rle8 compression
            uchar* line_end = image + width3;
            int line_end_flag = 0;
            row = 0;

            for (;;) {
                int code = file_data_->getWordLittleEndian();
                int length = code & 255;
                code >>= 8;
                if (length != 0) {  // encoded mode
                    int prev_row = row;
                    length *= channels;

                    if (image + length > line_end) {
                        goto decode_rle8_bad;
                    }

                    if (colorful) {
                        image = fillUniColor(image, line_end, bmp_stride, width3,
                                             row, height_, length,
                                             color_palette_[code]);
                    }
                    else {
                        image = fillUniGray(image, line_end, bmp_stride, width3,
                                            row, height_, length,
                                            gray_palette[code]);
                    }

                    line_end_flag = row - prev_row;

                    if (row >= height_) {
                        break;
                    }
                }
                else if (code > 2) {  // absolute mode
                    int prev_row = row;
                    int code3 = code * channels;

                    if (image + code3 > line_end) {
                        goto decode_rle8_bad;
                    }
                    int size = (code + 1) & (~1);
                    assert((size_t)size < src_size);
                    file_data_->getBytes(src, size);
                    if (colorful) {
                        image = fillColorRow8(image, src, code, color_palette_);
                    }
                    else {
                        image = fillGrayRow8(image, src, code, gray_palette);
                    }

                    line_end_flag = row - prev_row;
                }
                else {
                    int x_shift3 = (int)(line_end - image);
                    int y_shift = height_ - row;

                    if (code || !line_end_flag || x_shift3 < (int)width3) {
                        if (code == 2) {
                            x_shift3 = file_data_->getByte() * channels;
                            y_shift = file_data_->getByte();
                        }

                        x_shift3 += (y_shift * width3) & ((code == 0) - 1);

                        if (row >= height_) {
                            break;
                        }

                        if (colorful) {
                            image = fillUniColor(image, line_end, bmp_stride,
                                                 width3, row, height_, x_shift3,
                                                 color_palette_[0]);
                        }
                        else {
                            image = fillUniGray(image, line_end, bmp_stride, width3,
                                                row, height_, x_shift3,
                                                gray_palette[0]);
                        }

                        if (row >= height_) {
                            break;
                        }
                    }

                    line_end_flag = 0;
                    if (row >= height_) {
                        break;
                    }
                }
            }

            result = true;
decode_rle8_bad: ;
        }
        break;
    case 15:  // 15 BPP
        for (row = 0; row < height_; row++, image += bmp_stride) {
            file_data_->getBytes(src, src_pitch);
            if (!colorful) {
                BGR5552Gray(src, image, width_);
            }
            else {
                BGR5552BGR(src, image, width_);
            }
        }
        result = true;
        break;
    case 16:   // 16 BPP
        for (row = 0; row < height_; row++, image += bmp_stride) {
            file_data_->getBytes(src, src_pitch);
            if (!colorful) {
                BGR5652Gray(src, image, width_);
            }
            else {
                BGR5652BGR(src, image, width_);
            }
        }
        result = true;
        break;
    case 24:  // 24 BPP
        for (row = 0; row < height_; row++, image += bmp_stride) {
            file_data_->getBytes(src, src_pitch);
            if (!colorful) {
                BGR2Gray(src, image, width_);
            }
            else {
                memcpy(image, src, width_ * 3);
            }

        }
        result = true;
        break;
    case 32:  // 32 BPP
        for (row = 0; row < height_; row++, image += bmp_stride) {
            file_data_->getBytes(src, src_pitch);

            if (!colorful) {
                BGRA2Gray(src, image, width_);
            }
            else if (channels_ == 3) {
                BGRA2BGR(src, image, width_);
            }
            else if (channels_ == 4) {
                if (compression_type_ == BMP_BITFIELDS) {
                    bool has_bit_mask = (rgba_bit_offset_[0] >= 0) &&
                                        (rgba_bit_offset_[1] >= 0) &&
                                        (rgba_bit_offset_[2] >= 0);
                    if (has_bit_mask) {
                        maskBGRA(image, src, width_);
                    }
                    else {
                        memcpy(image, src, width_ * 4);
                    }
                }
                else {
                    memcpy(image, src, width_ * 4);
                }
            }
        }
        result = true;
        break;
    default:
        LOG(ERROR) << "Invalid/unsupported bit-per-pixel mode.";
    }

    delete [] src;

    return result;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl
