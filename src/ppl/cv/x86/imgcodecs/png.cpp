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

#include "png.h"
#include "codecs.h"

#include <string.h>

// #include "ppl/cv/x86/intrinutils.hpp"
// #include <immintrin.h>

#include "ppl/cv/types.h"
#include "ppl/common/log.h"

#include <iostream>  // debug

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace x86 {

#define MAKE_CHUNK_TYPE(s0, s1, s2, s3) (((uint32_t)(s0) << 24) | \
                                         ((uint32_t)(s1) << 16) | \
                                         ((uint32_t)(s2) << 8)  | \
                                         ((uint32_t)(s3)))
// CHUNK_XXXX
#define png_IHDR MAKE_CHUNK_TYPE( 73,  72,  68,  82)
#define png_PLTE MAKE_CHUNK_TYPE( 80,  76,  84,  69)
#define png_IDAT MAKE_CHUNK_TYPE( 73,  68,  65,  84)
#define png_IEND MAKE_CHUNK_TYPE( 73,  69,  78,  68)
#define png_tRNS MAKE_CHUNK_TYPE(116,  82,  78,  83)
#define png_cHRM MAKE_CHUNK_TYPE( 99,  72,  82,  77)
#define png_gAMA MAKE_CHUNK_TYPE(103,  65,  77,  65)
#define png_iCCP MAKE_CHUNK_TYPE(105,  67,  67,  80)
#define png_sBIT MAKE_CHUNK_TYPE(115,  66,  73,  84)
#define png_sRGB MAKE_CHUNK_TYPE(115,  82,  71,  66)
#define png_tEXt MAKE_CHUNK_TYPE(116,  69,  88, 116)
#define png_zTXt MAKE_CHUNK_TYPE(122,  84,  88, 116)
#define png_iTXt MAKE_CHUNK_TYPE(105,  84,  88, 116)
#define png_bKGD MAKE_CHUNK_TYPE( 98,  75,  71,  68)
#define png_hIST MAKE_CHUNK_TYPE(104,  73,  83,  84)
#define png_pHYs MAKE_CHUNK_TYPE(112,  72,  89, 115)
#define png_sPLT MAKE_CHUNK_TYPE(115,  80,  76,  84)
#define png_tIME MAKE_CHUNK_TYPE(116,  73,  77,  69)

#define HAVE_PLTE 0x01
#define HAVE_tRNS 0x02
#define HAVE_hIST 0x04
#define HAVE_bKGD 0x08
#define HAVE_IDAT 0x10
#define HAVE_iCCP 0x20
#define HAVE_sRGB 0x40

static const uint8_t stbi__depth_scale_table[9] = { 0, 0xff, 0x55, 0, 0x11, 0,0,0, 0x01 };

PngDecoder::PngDecoder(BytesReader& file_data) {
    file_data_ = &file_data;
    png_info_.palette_length = 0;
    png_info_.color_palette = nullptr;
    png_info_.alpha_palette = nullptr;

    // jpeg_ = (JpegDecodeData*) malloc(sizeof(JpegDecodeData));
    // if (jpeg_ == nullptr) {
    //    LOG(ERROR) << "No enough memory to initialize PngDecoder.";
    // }
}

PngDecoder::~PngDecoder() {
    if (png_info_.color_palette != nullptr) {
        delete [] png_info_.color_palette;
    }
    if (png_info_.alpha_palette != nullptr) {
        delete [] png_info_.alpha_palette;
    }
}

void PngDecoder::getChunkHeader(PngInfo& png_info) {
    png_info.current_chunk.length = file_data_->getDWordBigEndian1();
    png_info.current_chunk.type = file_data_->getDWordBigEndian1();
    // int readed = file_data_->getBytes(png_info.current_chunk.type, 4);
    // if (readed != 4) {
    //     LOG(ERROR) << "Invalid chunk type. It must be a sequence of 4 bytes.";
    //     return false;
    // }
}

bool PngDecoder::parseIHDR(PngInfo& png_info) {
    if (png_info.current_chunk.length != 13) {
        LOG(ERROR) << "The IHDR length: " << png_info.current_chunk.length
                   << ", correct bytes: 13.";
        return false;
    }

    width_ = file_data_->getDWordBigEndian1();
    height_ = file_data_->getDWordBigEndian1();
    if (height_ < 1 || width_ < 1) {
        LOG(ERROR) << "Invalid image height/width: " << height_ << ", " << width_;
        return false;
    }

    bit_depth_ = file_data_->getByte();
    if (bit_depth_ != 1 && bit_depth_ != 2 && bit_depth_ != 4 &&
        bit_depth_ != 8 && bit_depth_ != 16) {
        LOG(ERROR) << "Invalid bit depth: " << bit_depth_
                   << ", valid value: 1/2/4/8/16.";
        return false;
    }

    color_type_ = file_data_->getByte();
    if (color_type_ != 0 && color_type_ != 2 && color_type_ != 3 &&
        color_type_ != 4 && color_type_ != 6) {
        LOG(ERROR) << "Invalid color type: " << color_type_
                   << ", valid value: 0/2/3/4/6.";
        return false;
    }

    compression_method_ = file_data_->getByte();
    if (compression_method_ != 0) {
        LOG(ERROR) << "Invalid compression method: " << compression_method_
                   << ", only deflate(0) compression is supported.";
        return false;
    }

    filter_method_ = file_data_->getByte();
    if (filter_method_ != 0) {
        LOG(ERROR) << "Invalid filter method: " << filter_method_
                   << ", only adaptive filtering(0) is supported.";
        return false;
    }

    interlace_method_ = file_data_->getByte();
    if (interlace_method_ != 0 && interlace_method_ != 1) {
        LOG(ERROR) << "Invalid interlace method: " << interlace_method_
                   << ", only no interlace(0) and Adam7 interlace(1) are "
                   << "supported.";
        return false;
    }

    if (color_type_ == 0) {
        channels_ = 1;
    }
    else if (color_type_ == 2 || color_type_ == 3) {
        channels_ = 3;  // further check tRNS for alpha channel
    }
    else if (color_type_ == 4) {
        channels_ = 2;
    }
    else {
        channels_ = 4;
    }

    if (width_ * channels_ * height_ > MAX_IMAGE_SIZE) {
        LOG(ERROR) << "The target image: " << width_ << " * " << channels_
                   << " * " << height_ << ", it is too large to be decoded.";
        return false;
    }

    if (color_type_ == 2 && (bit_depth_ == 1 || bit_depth_ == 2 ||
                             bit_depth_ == 4)) {
        LOG(ERROR) << "Invalid bit depth for true color: " << bit_depth_
                   << ", valid value: 8/16.";
        return false;
    }
    if (color_type_ == 3 && bit_depth_ == 16) {
        LOG(ERROR) << "Invalid bit depth for indexed color: " << bit_depth_
                   << ", valid value: 1/2/4/8.";
        return false;
    }
    if ((color_type_ == 4 || color_type_ == 6) &&
        (bit_depth_ == 1 || bit_depth_ == 2 || bit_depth_ == 4)) {
        LOG(ERROR) << "Invalid bit depth for greyscale/true color with alpha: "
                   << bit_depth_ << ", valid value: 8/16.";
        return false;
    }

    std::cout << "IHDR chunk appear." << std::endl;
    std::cout << "width_: " << width_ << ", height_: " << height_
              << ", bit_depth_: " << (uint32_t)bit_depth_ << ", color_type_: " << (uint32_t)color_type_
              << ", compression_method_: " << (uint32_t)compression_method_ << ", filter_method_: "
              << (uint32_t)filter_method_ << ", interlace_method_: " << (uint32_t)interlace_method_
              << ", channels_: " << channels_ << std::endl;

    return true;
}

bool PngDecoder::parsePLTE(PngInfo& png_info) {
    if (color_type_ == 4 || color_type_ == 6) {
        LOG(ERROR) << "The PLTE chunk must not come with greyscale/greyscale "
                   << "with alpha.";
        return false;
    }

    if ((png_info.chunk_status & HAVE_tRNS) ||
        (png_info.chunk_status & HAVE_hIST) ||
        (png_info.chunk_status & HAVE_bKGD)) {
        LOG(ERROR) << "The PLTE chunk must come before the tRNS/hIST/bKGD chunk.";
        return false;
    }

    if (png_info.current_chunk.length > 256 * 3 ||
        png_info.current_chunk.length % 3 != 0) {
        LOG(ERROR) << "Invalid PLTE chunk length: "
                   << png_info.current_chunk.length
                   << ", it must be less than 768 and divisible by 3.";
        return false;
    }

    uint32_t length = png_info.current_chunk.length / 3;
    png_info.palette_length = length;
    png_info.color_palette = new PlteEntry[length];
    uint32_t readed = file_data_->getBytes((void*)png_info.color_palette,
                                           png_info.current_chunk.length);
    if (readed != png_info.current_chunk.length) {
        LOG(ERROR) << "The color palette readed bytes: " << readed
                   << ", correct bytes: " << png_info.current_chunk.length;
        return false;
    }
    png_info.chunk_status |= HAVE_PLTE;

    std::cout << "PLTE chunk appear, the color palette: " << std::endl;
    for (uint32_t index = 0; index < length; index++) {
        std::cout << "palette " << index << ": " << (uint32_t)png_info.color_palette[index].blue
                  << ", " << (uint32_t)png_info.color_palette[index].green
                  << ", " << (uint32_t)png_info.color_palette[index].red << std::endl;
    }

    return true;
}

bool PngDecoder::parseIDAT(PngInfo& png_info) {
    if (color_type_ == 3 && (!(png_info.chunk_status & HAVE_PLTE))) {
        LOG(ERROR) << "A color palette chunk is needed for indexed color, but "
                   << " no one appears befor an IDAT chunk.";
        return false;
    }

    file_data_->skip(png_info.current_chunk.length);
    std::cout << "IDAT size: " << png_info.current_chunk.length << std::endl; //

    png_info.chunk_status |= HAVE_IDAT;

    return true;
}

bool PngDecoder::parseIEND(PngInfo& png_info) {
    if (!(png_info.chunk_status & HAVE_IDAT)) {
        LOG(ERROR) << " No IDAT chunk appears in this file.";
        return false;
    }

    if (png_info.current_chunk.length != 0) {
        LOG(ERROR) << "The IEND length: " << png_info.current_chunk.length
                   << ", correct bytes: 0.";
        return false;
    }
    std::cout << "IEND chunk appear." << std::endl;

    return true;
}

bool PngDecoder::parsetRNS(PngInfo& png_info) {
    if (color_type_ == 4 || color_type_ == 6) {
        LOG(ERROR) << "The tRNS chunk must not come with greyscale/truecolor "
                   << "with alpha.";
        return false;
    }

    if (color_type_ == 3) {
        // if (png_info.color_palette == nullptr) {
        if (!(png_info.chunk_status & HAVE_PLTE)) {
            LOG(ERROR) << "The tRNS chunk must come after the PLTE chunk.";
            return false;
        }
        if (png_info.current_chunk.length > png_info.palette_length) {
            LOG(ERROR) << "The number of values in tRNS: "
                       << png_info.current_chunk.length
                       << ", palletter entries: " << png_info.palette_length
                       << ", the former must contain values which are not more "
                       << "than the latter.";
            return false;
        }

        png_info.alpha_palette = new uint8_t[png_info.palette_length];
        uint32_t readed = file_data_->getBytes((void*)png_info.alpha_palette,
                                            png_info.current_chunk.length);
        if (readed != png_info.current_chunk.length) {
            LOG(ERROR) << "The alpha palette readed bytes: " << readed
                       << ", correct bytes: " << png_info.current_chunk.length;
            return false;
        }
        if (png_info.current_chunk.length < png_info.palette_length) {
            memset(png_info.alpha_palette + png_info.current_chunk.length, 255,
                   png_info.palette_length - png_info.current_chunk.length);
        }
        channels_ = 4;
        std::cout << "tRNS chunk appear, the alpha palette: " << std::endl;
        for (uint32_t index = 0; index < png_info.palette_length; index++) {
            std::cout << "palette " << index << ": "
                      << (uint32_t)png_info.alpha_palette[index] << std::endl;
        }
    }
    else {
        if (color_type_ & 4) {
            LOG(ERROR) << "The tRNS chunk can not come with greyscale/"
                       << "truecolor with alpha.";
            return false;
        }

        if (png_info.current_chunk.length != (uint32_t)channels_ * 2) {
            LOG(ERROR) << "The alpha palette bytes: "
                       << png_info.current_chunk.length
                       << ", correct bytes: " << channels_ * 2;
            return false;
        }

        if (bit_depth_ == 16) {
            uint32_t readed = file_data_->getBytes((void*)png_info.alpha_values,
                                png_info.current_chunk.length);
            if (readed != png_info.current_chunk.length) {
                LOG(ERROR) << "The alpha palette readed bytes: " << readed
                           << ", correct bytes: "
                           << png_info.current_chunk.length;
                return false;
            }
            std::cout << "tRNS chunk appear, the alpha palette: " << std::endl;
            uint16_t* value = (uint16_t*)png_info.alpha_values;
            for (int index = 0; index < channels_; index++) {
                std::cout << "palette " << index << ": "
                          << (uint32_t)value[index] << std::endl;
            }
        }
        else {
            uint8_t* value = png_info.alpha_values;
            for (int i = 0; i < channels_; i++) {
                value[i] = (file_data_->getWordBigEndian() & 0xFF) *
                            stbi__depth_scale_table[bit_depth_];  // cast to 0-255
            }
            std::cout << "tRNS chunk appear, the alpha palette: " << std::endl;
            for (int index = 0; index < channels_; index++) {
                std::cout << "palette " << index << ": "
                          << (uint32_t)value[index] << std::endl;
            }
        }
        channels_++;
    }
    png_info.chunk_status |= HAVE_tRNS;

    std::cout << "tRNS chunk appear." << std::endl;
    std::cout << "in tRNS, channels_: " << channels_ << std::endl;

    return true;
}

bool PngDecoder::parsecHRM(PngInfo& png_info) {
    if (png_info.chunk_status & HAVE_PLTE) {
        LOG(ERROR) << "The cHRM chunk must come before the PLTE chunk.";
        return false;
    }

    if (png_info.current_chunk.length != 32) {
        LOG(ERROR) << "The cHRM length: " << png_info.current_chunk.length
                   << ", correct value: 32.";
        return false;
    }

    file_data_->skip(32);
    LOG(INFO) << "The cHRM chunk is skipped.";

    return true;
}

bool PngDecoder::parsegAMA(PngInfo& png_info) {
    if (png_info.chunk_status & HAVE_PLTE) {
        LOG(ERROR) << "The gAMA chunk must come before the PLTE chunk.";
        return false;
    }

    if (png_info.current_chunk.length != 4) {
        LOG(ERROR) << "The gAMA length: " << png_info.current_chunk.length
                   << ", correct value: 4.";
        return false;
    }

    file_data_->skip(4);
    LOG(INFO) << "The gAMA chunk is skipped.";

    return true;
}

bool PngDecoder::parseiCCP(PngInfo& png_info) {
    if (png_info.chunk_status & HAVE_PLTE) {
        LOG(ERROR) << "The iCCP chunk must come before the PLTE chunk.";
        return false;
    }

    if (png_info.chunk_status & HAVE_sRGB) {
        LOG(ERROR) << "The sRGB chunk has appeared, the iCCP chunk must not "
                   << "be present.";
        return false;
    }

    if (png_info.current_chunk.length < 4) {
        LOG(ERROR) << "The iCCP length: " << png_info.current_chunk.length
                   << ", it should not less than 4.";
        return false;
    }

    file_data_->skip(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_iCCP;
    LOG(INFO) << "The iCCP chunk is skipped.";

    return true;
}

bool PngDecoder::parsesBIT(PngInfo& png_info) {
    if (png_info.chunk_status & HAVE_PLTE) {
        LOG(ERROR) << "The sBIT chunk must come before the PLTE chunk.";
        return false;
    }

    if (color_type_ == 0) {
        if (png_info.current_chunk.length != 1) {
            LOG(ERROR) << "The sBIT length: " << png_info.current_chunk.length
                       << ", correct value: 1.";
            return false;
        }
    }
    else if (color_type_ == 2 || color_type_ == 3) {
        if (png_info.current_chunk.length != 3) {
            LOG(ERROR) << "The sBIT length: " << png_info.current_chunk.length
                       << ", correct value: 3.";
            return false;
        }
    }
    else if (color_type_ == 4) {
        if (png_info.current_chunk.length != 2) {
            LOG(ERROR) << "The sBIT length: " << png_info.current_chunk.length
                       << ", correct value: 2.";
            return false;
        }
    }
    else {
        if (png_info.current_chunk.length != 4) {
            LOG(ERROR) << "The sBIT length: " << png_info.current_chunk.length
                       << ", correct value: 4.";
            return false;
        }
    }

    file_data_->skip(png_info.current_chunk.length);
    LOG(INFO) << "The sBIT chunk is skipped.";

    return true;
}

bool PngDecoder::parsesRGB(PngInfo& png_info) {
    if (png_info.chunk_status & HAVE_PLTE) {
        LOG(ERROR) << "The sRGB chunk must come before the PLTE chunk.";
        return false;
    }

    if (png_info.chunk_status & HAVE_iCCP) {
        LOG(ERROR) << "The iCCP chunk has appeared, the sRGB chunk must not "
                   << "be present.";
        return false;
    }

    if (png_info.current_chunk.length != 1) {
        LOG(ERROR) << "The sRGB length: " << png_info.current_chunk.length
                   << ", correct value: 1.";
        return false;
    }

    file_data_->skip(1);
    png_info.chunk_status |= HAVE_sRGB;
    LOG(INFO) << "The sRGB chunk is skipped.";

    return true;
}

bool PngDecoder::parsetEXt(PngInfo& png_info) {
    if (png_info.current_chunk.length < 2) {
        LOG(ERROR) << "The tEXt length: " << png_info.current_chunk.length
                   << ", it should not less than 2.";
        return false;
    }

    file_data_->skip(png_info.current_chunk.length);
    LOG(INFO) << "A tEXt chunk is skipped.";

    return true;
}

bool PngDecoder::parsezTXt(PngInfo& png_info) {
    if (png_info.current_chunk.length < 3) {
        LOG(ERROR) << "The zTXt length: " << png_info.current_chunk.length
                   << ", it should not less than 3.";
        return false;
    }

    file_data_->skip(png_info.current_chunk.length);
    LOG(INFO) << "A zTXt chunk is skipped.";

    return true;
}

bool PngDecoder::parseiTXt(PngInfo& png_info) {
    if (png_info.current_chunk.length < 6) {
        LOG(ERROR) << "The iTXt length: " << png_info.current_chunk.length
                   << ", it should not less than 6.";
        return false;
    }

    file_data_->skip(png_info.current_chunk.length);
    LOG(INFO) << "A iTXt chunk is skipped.";

    return true;
}

bool PngDecoder::parsebKGD(PngInfo& png_info) {
    if (color_type_ == 0 || color_type_ == 4) {
        if (png_info.current_chunk.length != 2) {
            LOG(ERROR) << "The bKGD length: " << png_info.current_chunk.length
                       << ", correct value: 2.";
            return false;
        }
    }
    else if (color_type_ == 2 || color_type_ == 6) {
        if (png_info.current_chunk.length != 6) {
            LOG(ERROR) << "The bKGD length: " << png_info.current_chunk.length
                       << ", correct value: 6.";
            return false;
        }
    }
    else {
        if (png_info.current_chunk.length != 1) {  // indexed color
            LOG(ERROR) << "The bKGD length: " << png_info.current_chunk.length
                       << ", correct value: 1.";
            return false;
        }
    }

    file_data_->skip(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_bKGD;
    LOG(INFO) << "The bKGD chunk is skipped.";

    return true;
}

bool PngDecoder::parsehIST(PngInfo& png_info) {
    // if (png_info.color_palette == nullptr) {
    if (!(png_info.chunk_status & HAVE_PLTE)) {
        LOG(ERROR) << "The hIST chunk must come after the PLTE chunk.";
        return false;
    }

    if (png_info.current_chunk.length < 2) {
        LOG(ERROR) << "The hIST length: " << png_info.current_chunk.length
                   << ", it should not less than 2.";
        return false;
    }

    file_data_->skip(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_hIST;
    LOG(INFO) << "The hIST chunk is skipped.";

    return true;
}

bool PngDecoder::parsepHYs(PngInfo& png_info) {
    if (png_info.current_chunk.length != 9) {
        LOG(ERROR) << "The pHYs length: " << png_info.current_chunk.length
                   << ", correct value: 9.";
        return false;
    }

    file_data_->skip(9);
    LOG(INFO) << "The pHYs chunk is skipped.";

    return true;
}

bool PngDecoder::parsesPLT(PngInfo& png_info) {
    if (png_info.current_chunk.length < 9) {
        LOG(ERROR) << "The sPLT length: " << png_info.current_chunk.length
                   << ", it should not less than 9.";
        return false;
    }

    file_data_->skip(png_info.current_chunk.length);
    LOG(INFO) << "A sPLT chunk is skipped.";

    return true;
}

bool PngDecoder::parsetIME(PngInfo& png_info) {
    if (png_info.current_chunk.length != 7) {
        LOG(ERROR) << "The tIME length: " << png_info.current_chunk.length
                   << ", correct value: 7.";
        return false;
    }

    file_data_->skip(7);
    LOG(INFO) << "The tIME chunk is skipped.";

    return true;
}

bool PngDecoder::parsetUnknownChunk(PngInfo& png_info) {
    std::string chunk_name = getChunkName(png_info.current_chunk.type);
    if (!(png_info.current_chunk.type & (1 << 29))) {  // critical chunk, 5 + 24
        LOG(ERROR) << "Encountering an unknown critical chunk: " << chunk_name;
        return false;
    }
    else {  // ancillary chunk
        LOG(INFO) << "Encountering an unknown ancillary chunk: " << chunk_name;
        file_data_->skip(png_info.current_chunk.length);
        return true;
    }
}

void PngDecoder::releaseSource() {
    if (png_info_.color_palette != nullptr) {
        delete [] png_info_.color_palette;
    }
    if (png_info_.alpha_palette != nullptr) {
        delete [] png_info_.alpha_palette;
    }
}

std::string PngDecoder::getChunkName(uint32_t chunk_type) {
    std::string chunk_name;
    chunk_name.push_back((int8_t)(chunk_type >> 24));
    chunk_name.push_back((int8_t)((chunk_type >> 16) & 0xFF));
    chunk_name.push_back((int8_t)((chunk_type >> 8) & 0xFF));
    chunk_name.push_back((int8_t)(chunk_type & 0xFF));

    return chunk_name;
}

bool PngDecoder::readHeader() {
    // file_data_->skip(8);
    getChunkHeader(png_info_);
    if (png_info_.current_chunk.type != png_IHDR) {
        std::string chunk_name = getChunkName(png_info_.current_chunk.type);
        // std::string chunk_name = getChunkName(png_IHDR);
        for (auto c : chunk_name) {
            std::cout << (uint32_t)c << ", ";
        }
        std::cout << "string size: " << chunk_name.size() << std::endl;
        LOG(ERROR) << "encountering the chunk: "
                    << chunk_name
                    << ", expecting an IHDR chunk.";
        LOG(ERROR) << "The first chunk must be IHDR.";
        return false;
    }

    bool succeeded = parseIHDR(png_info_);
    if (!succeeded) return false;

    // png_info_.current_chunk.crc = file_data_->getDWordBigEndian1();
    // crc checking.
    file_data_->skip(4);

    getChunkHeader(png_info_);
    while (png_info_.current_chunk.type != png_IDAT &&
           png_info_.current_chunk.type != png_IEND) {
        switch (png_info_.current_chunk.type) {
            case png_PLTE:
                succeeded = parsePLTE(png_info_);
                if (!succeeded) return false;
                break;
            case png_tRNS:
                succeeded = parsetRNS(png_info_);
                if (!succeeded) return false;
                break;
            case png_cHRM:
                succeeded = parsecHRM(png_info_);
                if (!succeeded) return false;
                break;
            case png_gAMA:
                succeeded = parsegAMA(png_info_);
                if (!succeeded) return false;
                break;
            case png_iCCP:
                succeeded = parseiCCP(png_info_);
                if (!succeeded) return false;
                break;
            case png_sBIT:
                succeeded = parsesBIT(png_info_);
                if (!succeeded) return false;
                break;
            case png_sRGB:
                succeeded = parsesRGB(png_info_);
                if (!succeeded) return false;
                break;
            case png_tEXt:
                succeeded = parsetEXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_zTXt:
                succeeded = parsezTXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_iTXt:
                succeeded = parseiTXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_bKGD:
                succeeded = parsebKGD(png_info_);
                if (!succeeded) return false;
                break;
            case png_hIST:
                succeeded = parsehIST(png_info_);
                if (!succeeded) return false;
                break;
            case png_pHYs:
                succeeded = parsepHYs(png_info_);
                if (!succeeded) return false;
                break;
            case png_sPLT:
                succeeded = parsesPLT(png_info_);
                if (!succeeded) return false;
                break;
            case png_tIME:
                succeeded = parsetIME(png_info_);
                if (!succeeded) return false;
                break;
            default:
                succeeded = parsetUnknownChunk(png_info_);
                if (!succeeded) return false;
                break;
        }

        file_data_->skip(4);  // skip crc checking.
        getChunkHeader(png_info_);
    }

    if (png_info_.current_chunk.type == png_IEND) {
        LOG(ERROR) << "No image data chunk appears.";
        return false;
    }

    return true;
}

bool PngDecoder::decodeData(int32_t stride, uint8_t* image) {
    if (png_info_.current_chunk.type != png_IDAT) {
        LOG(ERROR) << "encountering the chunk: "
                   << getChunkName(png_info_.current_chunk.type)
                   << ", expecting the IDAT chunk.";
        return false;
    }

    bool succeeded;
    int count = 0;
    while (png_info_.current_chunk.type != png_IEND) {
        switch (png_info_.current_chunk.type) {
            case png_IDAT:
                succeeded = parseIDAT(png_info_);
                if (!succeeded) return false;
                std::cout << "processing IDAT chunk: " << count++ << std::endl;
                break;
            case png_tEXt:
                succeeded = parsetEXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_zTXt:
                succeeded = parsezTXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_iTXt:
                succeeded = parseiTXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_tIME:
                succeeded = parsetIME(png_info_);
                if (!succeeded) return false;
                break;
            default:
                succeeded = parsetUnknownChunk(png_info_);
                if (!succeeded) return false;
                break;
        }

        // file_data_->skip(png_info_.current_chunk.length);
        file_data_->skip(4);  // skip crc checking.
        getChunkHeader(png_info_);
    }

    succeeded = parseIEND(png_info_);
    if (!succeeded) {
        return false;
    }

    return true;
}



} //! namespace x86
} //! namespace cv
} //! namespace ppl