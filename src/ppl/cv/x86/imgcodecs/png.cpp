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

#include <limits.h>

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

static const uint8_t stbi__zdefault_length[STBI__ZNSYMS] = {
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8
};

static const uint8_t stbi__zdefault_distance[32] = {
   5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
};

static const int stbi__zlength_base[31] = {
   3,4,5,6,7,8,9,10,11,13,
   15,17,19,23,27,31,35,43,51,59,
   67,83,99,115,131,163,195,227,258,0,0 };

static const int stbi__zlength_extra[31]=
{ 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0 };

static const int stbi__zdist_base[32] = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,
257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,0,0};

static const int stbi__zdist_extra[32] =
{ 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

PngDecoder::PngDecoder(BytesReader& file_data) {
    file_data_ = &file_data;
    png_info_.palette_length = 0;
    png_info_.palette = nullptr;
    // png_info_.alpha_palette = nullptr;
    png_info_.chunk_status = 0;
    encoded_channels_ = 0;

    file_data_->setCrcChecking(&crc32_);
    png_info_.fixed_huffman_done = false;
    png_info_.header_after_idat = false;
    png_info_.idat_count = 0;
    // crc32_.turnOff();
    // jpeg_ = (JpegDecodeData*) malloc(sizeof(JpegDecodeData));
    // if (jpeg_ == nullptr) {
    //    LOG(ERROR) << "No enough memory to initialize PngDecoder.";
    // }
}

PngDecoder::~PngDecoder() {
    if (png_info_.palette != nullptr) {
        delete [] png_info_.palette;
    }

    file_data_->unsetCrcChecking();
    // if (png_info_.alpha_palette != nullptr) {
    //     delete [] png_info_.alpha_palette;
    // }
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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

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
    depth_ = bit_depth_ == 16 ? 16 : 8;

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

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    if (color_type_ == 0) {
        channels_ = 1;
        encoded_channels_ = 1;
    }
    else if (color_type_ == 2) {
        channels_ = 3;  // further check tRNS for alpha channel
        encoded_channels_ = 3;
    }
    else if (color_type_ == 3) {
        channels_ = 3;  // further check tRNS for alpha channel
        encoded_channels_ = 1;
    }
    else if (color_type_ == 4) {
        channels_ = 2;
        encoded_channels_ = 2;
    }
    else {
        channels_ = 4;
        encoded_channels_ = 4;
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

    if (bit_depth_ == 16) {
        LOG(ERROR) << "This implementation does not support bit depth 16.";
        return false;
    }

    std::cout << "IHDR chunk appear." << std::endl;
    std::cout << "width_: " << width_ << ", height_: " << height_
              << ", bit_depth_: " << (uint32_t)bit_depth_ << ", color_type_: " << (uint32_t)color_type_
              << ", compression_method_: " << (uint32_t)compression_method_ << ", filter_method_: "
              << (uint32_t)filter_method_ << ", interlace_method_: " << (uint32_t)interlace_method_
              << ", channels_: " << channels_ << ", encoded_channels_: "
              << encoded_channels_ << std::endl;

    return true;
}

bool PngDecoder::parsetIME(PngInfo& png_info) {
    if (png_info.current_chunk.length != 7) {
        LOG(ERROR) << "The tIME length: " << png_info.current_chunk.length
                   << ", correct value: 7.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(7);
    // LOG(INFO) << "The tIME chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsezTXt(PngInfo& png_info) {
    if (png_info.current_chunk.length < 3) {
        LOG(ERROR) << "The zTXt length: " << png_info.current_chunk.length
                   << ", it should not less than 3.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    // LOG(INFO) << "A zTXt chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsetEXt(PngInfo& png_info) {
    if (png_info.current_chunk.length < 2) {
        LOG(ERROR) << "The tEXt length: " << png_info.current_chunk.length
                   << ", it should not less than 2.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    // LOG(INFO) << "A tEXt chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parseiTXt(PngInfo& png_info) {
    if (png_info.current_chunk.length < 6) {
        LOG(ERROR) << "The iTXt length: " << png_info.current_chunk.length
                   << ", it should not less than 6.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    // LOG(INFO) << "A iTXt chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsepHYs(PngInfo& png_info) {
    if (png_info.current_chunk.length != 9) {
        LOG(ERROR) << "The pHYs length: " << png_info.current_chunk.length
                   << ", correct value: 9.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(9);
    // LOG(INFO) << "The pHYs chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsesPLT(PngInfo& png_info) {
    if (png_info.current_chunk.length < 9) {
        LOG(ERROR) << "The sPLT length: " << png_info.current_chunk.length
                   << ", it should not less than 9.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    // LOG(INFO) << "A sPLT chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_iCCP;
    // LOG(INFO) << "The iCCP chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(1);
    png_info.chunk_status |= HAVE_sRGB;
    // LOG(INFO) << "The sRGB chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    // LOG(INFO) << "The sBIT chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(4);
    // LOG(INFO) << "The gAMA chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(32);
    // LOG(INFO) << "The cHRM chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

// reorder the channels from RGB to BGR for truecolour.
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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    uint32_t length = png_info.current_chunk.length / 3;
    png_info.palette_length = length;
    png_info.palette = new uint8_t[1024];
    uint8_t value0, value1, value2;
    for (uint32_t index = 0; index < length * 4; index += 4) {
        value0 = file_data_->getByte();
        value1 = file_data_->getByte();
        value2 = file_data_->getByte();
        png_info.palette[index] = value2;
        png_info.palette[index + 1] = value1;
        png_info.palette[index + 2] = value0;
        png_info.palette[index + 3] = 255;
    }
    // uint32_t readed = file_data_->getBytes((void*)png_info.palette,
    //                                        png_info.current_chunk.length);
    // if (readed != png_info.current_chunk.length) {
    //     LOG(ERROR) << "The color palette readed bytes: " << readed
    //                << ", correct bytes: " << png_info.current_chunk.length;
    //     return false;
    // }
    png_info.chunk_status |= HAVE_PLTE;

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    // std::cout << "PLTE chunk appear, the color palette: " << std::endl;
    // for (uint32_t index = 0; index < length * 4; index += 4) {
    //     std::cout << "palette " << (index >> 2) << ": "
    //               << (uint32_t)png_info.palette[index]
    //               << ", " << (uint32_t)png_info.palette[index + 1]
    //               << ", " << (uint32_t)png_info.palette[index + 2] << std::endl;
    // }

    return true;
}

bool PngDecoder::parsetRNS(PngInfo& png_info) {
    if (color_type_ == 4 || color_type_ == 6) {
        LOG(ERROR) << "The tRNS chunk must not come with greyscale/truecolor "
                   << "with alpha.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    if (color_type_ == 3) {
        // if (png_info.palette == nullptr) {
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

        // png_info.alpha_palette = new uint8_t[png_info.palette_length];
        // uint32_t readed = file_data_->getBytes((void*)png_info.alpha_palette,
        //                                        png_info.current_chunk.length);
        // if (readed != png_info.current_chunk.length) {
        //     LOG(ERROR) << "The alpha palette readed bytes: " << readed
        //                << ", correct bytes: " << png_info.current_chunk.length;
        //     return false;
        // }
        // if (png_info.current_chunk.length < png_info.palette_length) {
        //     memset(png_info.alpha_palette + png_info.current_chunk.length, 255,
        //            png_info.palette_length - png_info.current_chunk.length);
        // }
        for (uint32_t index = 3; index < png_info.current_chunk.length * 4;
             index += 4) {
            png_info.palette[index] = file_data_->getByte();
        }
        channels_ = 4;
        // std::cout << "tRNS chunk appear, the alpha palette: " << std::endl;
        // for (uint32_t index = 3; index <= png_info.palette_length * 4; index+= 4) {
        //     std::cout << "palette " << index << ": "
        //               << (uint32_t)png_info.palette[index] << std::endl;
        // }
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
            // uint32_t readed = file_data_->getBytes((void*)png_info.alpha_values,
            //                     png_info.current_chunk.length);
            // if (readed != png_info.current_chunk.length) {
            //     LOG(ERROR) << "The alpha palette readed bytes: " << readed
            //                << ", correct bytes: "
            //                << png_info.current_chunk.length;
            //     return false;
            // }
            uint16_t* values = (uint16_t*)png_info.alpha_values;
            if (color_type_ == 0) {
                values[0] = file_data_->getWordBigEndian();
            }
            else {  // color_type_ == 2
                values[2] = file_data_->getWordBigEndian();
                values[1] = file_data_->getWordBigEndian();
                values[0] = file_data_->getWordBigEndian();
            }
            // std::cout << "tRNS chunk appear, the alpha palette: " << std::endl;
            // uint16_t* value = (uint16_t*)png_info.alpha_values;
            // for (uint32_t index = 0; index < channels_; index++) {
            //     std::cout << "palette " << index << ": "
            //               << (uint32_t)value[index] << std::endl;
            // }
        }
        else {
            uint8_t* values = png_info.alpha_values;
            // for (uint32_t i = 0; i < channels_; i++) {
            //     value[i] = (file_data_->getWordBigEndian() & 0xFF) *
            //                 stbi__depth_scale_table[bit_depth_];  // cast to 0-255
            // }
            if (color_type_ == 0) {
                values[0] = (file_data_->getWordBigEndian() & 0xFF) *
                             stbi__depth_scale_table[bit_depth_];
            }
            else {  // color_type_ == 2
                values[2] = (file_data_->getWordBigEndian() & 0xFF) *
                             stbi__depth_scale_table[bit_depth_];
                values[1] = (file_data_->getWordBigEndian() & 0xFF) *
                             stbi__depth_scale_table[bit_depth_];
                values[0] = (file_data_->getWordBigEndian() & 0xFF) *
                             stbi__depth_scale_table[bit_depth_];
            }
            // std::cout << "tRNS chunk appear, the alpha palette: " << std::endl;
            // for (uint32_t index = 0; index < channels_; index++) {
            //     std::cout << "palette " << index << ": "
            //               << (uint32_t)value[index] << std::endl;
            // }
        }
        channels_++;
    }
    png_info.chunk_status |= HAVE_tRNS;

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    // std::cout << "tRNS chunk appear." << std::endl;
    // std::cout << "in tRNS, channels_: " << channels_ << std::endl;

    return true;
}

bool PngDecoder::parsehIST(PngInfo& png_info) {
    // if (png_info.palette == nullptr) {
    if (!(png_info.chunk_status & HAVE_PLTE)) {
        LOG(ERROR) << "The hIST chunk must come after the PLTE chunk.";
        return false;
    }

    if (png_info.current_chunk.length < 2) {
        LOG(ERROR) << "The hIST length: " << png_info.current_chunk.length
                   << ", it should not less than 2.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_hIST;
    // LOG(INFO) << "The hIST chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skip(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_bKGD;
    // LOG(INFO) << "The bKGD chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

// This function processes one or more IDAT chunk.
bool PngDecoder::parseIDATs(PngInfo& png_info, uint8_t* image,
                            uint32_t stride) {
    if (color_type_ == 3 && (!(png_info.chunk_status & HAVE_PLTE))) {
        LOG(ERROR) << "A color palette chunk is needed for indexed color, but "
                   << " no one appears befor an IDAT chunk.";
        return false;
    }

    bool succeeded = setCrc32();  // processing crc of the first or current IDAT chunk.
    if (!succeeded) return false;

    if (!(png_info.chunk_status & HAVE_IDAT)) {  // first IDAT chunk
        succeeded = parseDeflateHeader();
        if (!succeeded) return false;
    }

    // std::cout << "IDAT size: " << png_info.current_chunk.length << std::endl; //
    succeeded = inflateImage(png_info, image, stride);
    if (!succeeded) return false;
    // skipping 32 bits of the optional ADLER32 checksum in zlib data stream.
    // std::cout << "IDAT leftover: " << png_info.current_chunk.length << std::endl;
    file_data_->skip(png_info.current_chunk.length);

    // process crc of the current of last IDAT chunk.
    if (png_info.current_chunk.type == png_IDAT) {
        png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
        succeeded = isCrcCorrect();
        if (!succeeded) return false;
    }

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
    // std::cout << "IEND chunk appear." << std::endl;

    // uint32_t crc2 = crc32(0, nullptr, 0);
    // std::cout << "$$$$IEND, chunk type crc2: " << crc2 << std::endl;
    // crc2 = crc32(crc2, ((uint8_t*)(&png_info.current_chunk.type)), 4);
    // std::cout << "$$$$IEND, chunk type crc2: " << crc2 << std::endl;

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsetUnknownChunk(PngInfo& png_info) {
    std::string chunk_name = getChunkName(png_info.current_chunk.type);
    if (!(png_info.current_chunk.type & (1 << 29))) {  // critical chunk, 5 + 24
        LOG(ERROR) << "Encountering an unknown critical chunk: " << chunk_name;
        return false;
    }
    else {  // ancillary chunk
        bool succeeded = setCrc32();
        if (!succeeded) return false;

        // LOG(INFO) << "Encountering an unknown ancillary chunk: " << chunk_name;
        file_data_->skip(png_info.current_chunk.length);

        png_info.current_chunk.crc = file_data_->getDWordBigEndian1();
        succeeded = isCrcCorrect();
        if (!succeeded) return false;

        return true;
    }
}

void PngDecoder::releaseSource() {
    if (png_info_.palette != nullptr) {
        delete [] png_info_.palette;
    }
    // if (png_info_.alpha_palette != nullptr) {
    //     delete [] png_info_.alpha_palette;
    // }
}

std::string PngDecoder::getChunkName(uint32_t chunk_type) {
    std::string chunk_name;
    chunk_name.push_back((int8_t)(chunk_type >> 24));
    chunk_name.push_back((int8_t)((chunk_type >> 16) & 0xFF));
    chunk_name.push_back((int8_t)((chunk_type >> 8) & 0xFF));
    chunk_name.push_back((int8_t)(chunk_type & 0xFF));

    return chunk_name;
}

bool PngDecoder::setCrc32() {
    if (!crc32_.isChecking()) {
        // LOG(INFO) << "Crc32 checksum verification is turned off.";
        return true;
    }

    if (png_info_.current_chunk.length != 0) {
        uint8_t* buffer = file_data_->getCurrentPosition();
        uint32_t buffer_size = file_data_->getValidSize();
        crc32_.setCrc(buffer, buffer_size, 0, png_info_.current_chunk.length);

        // std::cout << "####png chunk type crc: " << crc32_.getCrcValue() << std::endl;
        bool succeeded = crc32_.calculateCrc(png_info_.current_chunk.type);  // return false?
        // std::cout << "####png chunk type crc: " << crc32_.getCrcValue() << std::endl;
        if (!succeeded) return false;
        succeeded = crc32_.calculateCrc();
        if (!succeeded) return false;
    }
    else {
        crc32_.setCrc(nullptr, 0, 0, 0);
        bool succeeded = crc32_.calculateCrc(png_info_.current_chunk.type);  // return false?
        if (!succeeded) return false;
    }

    return true;
}

bool PngDecoder::isCrcCorrect() {
    if (!crc32_.isChecking()) return true;

    uint32_t calculated_crc = crc32_.getCrcValue();
    uint32_t transmitted_crc = png_info_.current_chunk.crc;
    // if (png_info_.current_chunk.length != 0) {
    // }

    // debug
    // std::string chunk_name = getChunkName(png_info_.current_chunk.type);
    // LOG(INFO) << "The crc of Chunk " << chunk_name
    //           << ", calculated value: " << calculated_crc
    //           << ", transmitted value: " << transmitted_crc;

    if (calculated_crc == transmitted_crc) {
        return true;
    }
    else {
        std::string chunk_name = getChunkName(png_info_.current_chunk.type);
        LOG(ERROR) << "The crc of Chunk " << chunk_name << " mismatchs, "
                   << "calculated value: " << calculated_crc
                   << ", transmitted value: " << transmitted_crc;
        return false;
    }
}

bool PngDecoder::parseDeflateHeader() {
    uint8_t cmf = file_data_->getByte();
    uint8_t flags = file_data_->getByte();
    uint32_t compression_method = cmf & 15;
    uint32_t compresson_info = (cmf >> 4) & 15;
    // std::cout << "CM: " << compression_method << std::endl;
    // std::cout << "CINFO: " << compresson_info << std::endl;
    if (compression_method != 8 ||
        (compression_method == 8 && compresson_info > 7)) {
        LOG(ERROR) << "Only deflate compressin method with a window size "
                   << "up to 32K is supported.";
        return false;
    }
    if (((cmf << 8) + flags) % 31 != 0) {
        LOG(ERROR) << "zlib flag check failed.";
        return false;
    }

    uint32_t fdict = (int)((flags >> 5) & 1);
    // std::cout << "FCHECK: " << (int)(flags & 31) << std::endl;
    // std::cout << "FDICT: " << fdict << std::endl;
    // std::cout << "FLEVEL: " << (int)((flags >> 6) & 3) << std::endl;
    // *zlib_header_length = 2;
    if (fdict) {
        // *zlib_header_length += 4;  // debug
        // file_data_->skip(4);      // debug
        LOG(ERROR) << "A preset dictonary is not allowed in the PNG specification.";
        return false;
    }
    png_info_.current_chunk.length -= 2;
    png_info_.zbuffer.window_size = 1 << (compresson_info + 8);
    // std::cout << "sliding window size: " << png_info_.zbuffer.window_size << std::endl;

    return true;
}

bool PngDecoder::fillBits(ZbufferInfo *z) {  // z -> zbuffer
    // if (png_info_.current_chunk.length > 8) {
    // }
    // else if (png_info_.current_chunk.length > 4) {
    // }
    // else {
    // }

    do {
        if (png_info_.current_chunk.length == 0) {
            png_info_.current_chunk.crc = file_data_->getDWordBigEndian1();
            bool succeeded = isCrcCorrect();
            if (!succeeded) return false;

            getChunkHeader(png_info_);
            if (png_info_.current_chunk.type == png_IDAT) {
                // std::cout << "coming in IDAT chunk: " << ++(png_info_.idat_count) << std::endl;
                succeeded = setCrc32();
                if (!succeeded) return false;
            }
            if (png_info_.current_chunk.type != png_IDAT) {
                png_info_.header_after_idat = true;
                std::string chunk_name = getChunkName(png_info_.current_chunk.type);
                // LOG(INFO) << "The reached Chunk: " << chunk_name
                //           << ", this is the last IDAT chunk.";
                return true;
            }
        }

        z->code_buffer |= (uint32_t)(file_data_->getByte() << z->num_bits);
        z->num_bits += 8;
        png_info_.current_chunk.length -= 1;
    } while (z->num_bits <= 24);

    return true;
}

uint32_t PngDecoder::zreceive(ZbufferInfo *z, int n) {
    uint32_t k;
    if (z->num_bits < n) {
        bool succeeded = fillBits(z);
        if (!succeeded) return UINT32_MAX;
    }
    k = z->code_buffer & ((1 << n) - 1);
    z->code_buffer >>= n;
    z->num_bits -= n;

    return k;
}

bool PngDecoder::parseZlibUncompressedBlock(PngInfo& png_info) {
    ZbufferInfo &zlib_buffer = png_info.zbuffer;
    // if (png_info.unprocessed_zbuffer_size == 0) {
        uint8_t header[4];
        uint32_t ignored_bits = zlib_buffer.num_bits & 7;
        if (ignored_bits) {
            zreceive(&zlib_buffer, ignored_bits);
        }
        uint32_t k = 0;
        while (zlib_buffer.num_bits > 0) {
            header[k++] = (uint8_t)(zlib_buffer.code_buffer & 255);
            zlib_buffer.code_buffer >>= 8;
            zlib_buffer.num_bits -= 8;
        }
        while (k < 4) {
            header[k++] = file_data_->getByte();
            png_info_.current_chunk.length -= 1;
        }
        uint32_t len  = header[1] * 256 + header[0];
        uint32_t nlen = header[3] * 256 + header[2];
        // std::cout << "uncompressed block size: " << len << std::endl;
        if (nlen != (len ^ 0xffff)) {
            LOG(ERROR) << "Non-compressed block in zlib is corrupt.";
            return false;
        }
        if (png_info.decompressed_image + len > png_info.decompressed_image_end) {
            LOG(ERROR) << "No space stores decompressed blocks in zlib.";
            return false;
        }

        do {
            if (len <= png_info.current_chunk.length) {
                file_data_->getBytes(png_info.decompressed_image, len);
                png_info.decompressed_image += len;
                // png_info.unprocessed_zbuffer_size = 0;  // ???
                png_info.current_chunk.length -= len;
                len = 0;
                // std::cout << "end of uncompressed block in zlib." << std::endl;
            }
            else {
                file_data_->getBytes(png_info.decompressed_image,
                                     png_info.current_chunk.length);
                png_info.decompressed_image += png_info.current_chunk.length;
                // png_info.unprocessed_zbuffer_size = len -
                //                                     png_info.current_chunk.length;
                // png_info.current_chunk.length = 0;
                len -= png_info.current_chunk.length;

                png_info_.current_chunk.crc = file_data_->getDWordBigEndian1();
                bool succeeded = isCrcCorrect();
                if (!succeeded) return false;

                getChunkHeader(png_info_);
                if (png_info_.current_chunk.type == png_IDAT) {
                    // std::cout << "coming in IDAT chunk: " << ++(png_info.idat_count) << std::endl;
                    succeeded = setCrc32();
                    if (!succeeded) return false;
                }
                if (png_info_.current_chunk.type != png_IDAT) {
                    png_info_.header_after_idat = true;
                    std::string chunk_name = getChunkName(png_info_.current_chunk.type);
                    // LOG(INFO) << "The reached Chunk: " << chunk_name
                    //           << ", this is the last IDAT chunk.";
                    return true;
                }
            }
        } while (len > 0);

    // }
    // else {
    //     if (png_info.unprocessed_zbuffer_size <= png_info.current_chunk.length) {
    //         file_data_->getBytes(png_info.decompressed_image,
    //                              png_info.unprocessed_zbuffer_size);
    //         png_info.decompressed_image += png_info.unprocessed_zbuffer_size;
    //         png_info.unprocessed_zbuffer_size = 0;  // ???
    //         png_info.current_chunk.length -= png_info.unprocessed_zbuffer_size;
    //     }
    //     else {
    //         file_data_->getBytes(png_info.decompressed_image,
    //                              png_info.current_chunk.length);
    //         png_info.decompressed_image += png_info.current_chunk.length;
    //         png_info.unprocessed_zbuffer_size -= png_info.current_chunk.length;
    //         png_info.current_chunk.length = 0;
    //     }
    // }

    return true;
}

static int stbi__bit_reverse(int v, int bits) {
    v = ((v & 0xAAAA) >>  1) | ((v & 0x5555) << 1);
    v = ((v & 0xCCCC) >>  2) | ((v & 0x3333) << 2);
    v = ((v & 0xF0F0) >>  4) | ((v & 0x0F0F) << 4);
    v = ((v & 0xFF00) >>  8) | ((v & 0x00FF) << 8);

    // to bit reverse n bits, reverse 16 and shift
    // e.g. 11 bits, bit reverse and shift away 5
    int value = v >> (16 - bits);

    return value;
}

bool PngDecoder::buildHuffmanCode(stbi__zhuffman *z, const uint8_t *sizelist,
                                  int num) {
    int i, k=0;
    int code, next_code[16], sizes[17];

    // DEFLATE spec for generating codes
    memset(sizes, 0, sizeof(sizes));
    memset(z->fast, 0, sizeof(z->fast));
    for (i=0; i < num; ++i) {
        ++sizes[sizelist[i]];
    }
    sizes[0] = 0;
    for (i=1; i < 16; ++i) {
        if (sizes[i] > (1 << i)) {
            LOG(ERROR) << "The size of code lengths in huffman is wrong.";
            return false;
        }
    }
    code = 0;
    for (i=1; i < 16; ++i) {
        next_code[i] = code;
        z->firstcode[i] = (uint16_t) code;
        z->firstsymbol[i] = (uint16_t) k;
        code = (code + sizes[i]);
        if (sizes[i]) {
            if (code-1 >= (1 << i)) {
                LOG(ERROR) << "The code lengths of huffman are wrong.";
                return false;
            }
        }
        z->maxcode[i] = code << (16-i); // preshift for inner loop
        code <<= 1;
        k += sizes[i];
    }
    z->maxcode[16] = 0x10000; // sentinel
    for (i=0; i < num; ++i) {
        int s = sizelist[i];
        if (s) {
            int c = next_code[s] - z->firstcode[s] + z->firstsymbol[s];
            uint16_t fastv = (uint16_t) ((s << 9) | i);
            z->size [c] = (uint8_t) s;
            z->value[c] = (uint16_t) i;
            if (s <= STBI__ZFAST_BITS) {
                int j = stbi__bit_reverse(next_code[s],s);
                while (j < (1 << STBI__ZFAST_BITS)) {
                    z->fast[j] = fastv;
                    j += (1 << s);
                }
            }
            ++next_code[s];
        }
    }
    return true;
}

int PngDecoder::stbi__zhuffman_decode_slowpath(ZbufferInfo *a, stbi__zhuffman *z) {
   int b,s,k;
   // not resolved by fast table, so compute it the slow way
   // use jpeg approach, which requires MSbits at top
   k = stbi__bit_reverse(a->code_buffer, 16);
   for (s=STBI__ZFAST_BITS+1; ; ++s)
      if (k < z->maxcode[s])
         break;
   if (s >= 16) return -1; // invalid code!
   // code size is s, so:
   b = (k >> (16-s)) - z->firstcode[s] + z->firstsymbol[s];
   if (b >= STBI__ZNSYMS) return -1; // some data was corrupt somewhere!
   if (z->size[b] != s) return -1;  // was originally an assert, but report failure instead.
   a->code_buffer >>= s;
   a->num_bits -= s;
   return z->value[b];
}

int PngDecoder::stbi__zhuffman_decode(ZbufferInfo *a, stbi__zhuffman *z) {
   int b,s;
   if (a->num_bits < 16) {
    //   if (stbi__zeof(a)) {
    //      return -1;   /* report error for unexpected end of data. */
    //   }
      fillBits(a);
   }
   b = z->fast[a->code_buffer & STBI__ZFAST_MASK];
   if (b) {
      s = b >> 9;
      a->code_buffer >>= s;
      a->num_bits -= s;
      return b & 511;
   }
   return stbi__zhuffman_decode_slowpath(a, z);
}

bool PngDecoder::computeDynamicHuffman(ZbufferInfo* a) {
   static const uint8_t length_dezigzag[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };
   stbi__zhuffman z_codelength;
   uint8_t lencodes[286+32+137];//padding for maximum single op
   uint8_t codelength_sizes[19];
   int i,n;

   int hlit  = zreceive(a, 5) + 257;
   int hdist = zreceive(a, 5) + 1;
   int hclen = zreceive(a, 4) + 4;
   int ntot  = hlit + hdist;

   memset(codelength_sizes, 0, sizeof(codelength_sizes));
   for (i=0; i < hclen; ++i) {
      int s = zreceive(a, 3);
      codelength_sizes[length_dezigzag[i]] = (uint8_t) s;
   }
   if (!buildHuffmanCode(&z_codelength, codelength_sizes, 19)) return false;

   n = 0;
   while (n < ntot) {
      int c = stbi__zhuffman_decode(a, &z_codelength);
      if (c < 0 || c >= 19) {
        LOG(ERROR) << "Invalid code length: " << c
                   << ", valid value: 0-19.";
        return false;
      }
      if (c < 16) {
         lencodes[n++] = (uint8_t) c;
      }
      else {
         uint8_t fill = 0;
         if (c == 16) {
            c = zreceive(a, 2) + 3;
            if (n == 0) {
                LOG(ERROR) << "Invalid code length " << c
                           << " for the first code.";
                return false;
            }
            fill = lencodes[n-1];
         } else if (c == 17) {
            c = zreceive(a, 3) + 3;
         } else if (c == 18) {
            c = zreceive(a, 7) + 11;
         } else {
            LOG(ERROR) << "Invalid code length: " << c
                       << ", valid value: 0-19.";
            return false;
         }
         if (ntot - n < c) {
            LOG(ERROR) << "Invalid repeating count: " << c
                       << ", valid value: " << ntot - n;
            return false;
         }
         memset(lencodes+n, fill, c);
         n += c;
      }
   }
   if (n != ntot) {
        LOG(ERROR) << "Invalid count of code length: " << n
                   << ", valid value: " << ntot;
        return false;
   }
   if (!buildHuffmanCode(&a->z_length, lencodes, hlit)) return false;
   if (!buildHuffmanCode(&a->z_distance, lencodes+hlit, hdist)) return false;

   return true;
}

bool PngDecoder::decodeHuffmanData(ZbufferInfo* a) {
   uint8_t *zout = png_info_.decompressed_image;
   for(;;) {
      int z = stbi__zhuffman_decode(a, &a->z_length);
      if (z < 256) {
         if (z < 0) {
            LOG(ERROR) << "Invalid decoded literal/length code: " << z
                       << ", valid value: 0-285.";
            return false;
         }
         *zout++ = (uint8_t) z;
      } else {
         uint8_t *p;
         int len, dist;
         if (z == 256) {
            // std::cout << "end of zlib block." << std::endl;
            png_info_.decompressed_image = zout;
            return true;
         }
         z -= 257;
         len = stbi__zlength_base[z];
         if (stbi__zlength_extra[z]) len += zreceive(a, stbi__zlength_extra[z]);
         z = stbi__zhuffman_decode(a, &a->z_distance);
         if (z < 0) {
            LOG(ERROR) << "Invalid decoded distance code: " << z
                       << ", valid value: 0-31.";
            return false;
         }
         dist = stbi__zdist_base[z];
         if (stbi__zdist_extra[z]) dist += zreceive(a, stbi__zdist_extra[z]);
         if (zout - png_info_.decompressed_image_start < dist) {
            LOG(ERROR) << "Invalid decoded distance: " << dist
                       << ", valid value: 0-" << zout - png_info_.decompressed_image_start;
            return false;
         }
         if (zout + len > png_info_.decompressed_image_end) {
            LOG(ERROR) << "Invalid decoded distance: " << dist
                       << ", valid value: 0-" << png_info_.decompressed_image_end - zout;
            return false;
         }
         if (dist > a->window_size) {
            LOG(ERROR) << "Invalid decoded distance: " << dist
                       << ", valid value: 0-" << a->window_size;
            return false;
         }
         p = (uint8_t *) (zout - dist);
         if (dist == 1) { // run of one byte; common in images.
            uint8_t v = *p;
            if (len) { do *zout++ = v; while (--len); }
         } else {
            if (len) { do *zout++ = *p++; while (--len); }
         }
      }
   }

   return true;
}

bool PngDecoder::inflateImage(PngInfo& png_info, uint8_t* image,
                              uint32_t stride) {
    if (png_info.current_chunk.length == 0) {
        LOG(ERROR) << "No data in the IDAT chunk is needed to be decompressed.";
        return false;
    }

    // uint32_t block_count = 0;
    bool succeeded;
    ZbufferInfo &zbuffer = png_info.zbuffer;
    do {
        zbuffer.is_final_block = zreceive(&zbuffer, 1);
        zbuffer.encoding_method = (EncodingMethod)zreceive(&zbuffer, 2);
        // std::cout << "zlib block " << block_count++ << ": " << std::endl;
        // std::cout << "zlib is last block: " << (int)zbuffer.is_final_block << std::endl;
        // std::cout << "zlib encoding method: " << (int)zbuffer.encoding_method << std::endl;

        switch (zbuffer.encoding_method) {
            case STORED_SECTION:
                succeeded = parseZlibUncompressedBlock(png_info);
                if (!succeeded) return false;
                break;
            case STATIC_HUFFMAN:
                if (!png_info.fixed_huffman_done) {
                    succeeded = buildHuffmanCode(&zbuffer.z_length,
                                 stbi__zdefault_length, STBI__ZNSYMS);
                    if (!succeeded) return false;
                    succeeded = buildHuffmanCode(&zbuffer.z_distance,
                                 stbi__zdefault_distance, 32);
                    if (!succeeded) return false;
                    png_info.fixed_huffman_done = true;
                }
                succeeded = decodeHuffmanData(&zbuffer);
                if (!succeeded) return false;
                break;
            case DYNAMIC_HUFFMAN:
                // if (png_info.unprocessed_zbuffer_size == 0) {
                    succeeded = computeDynamicHuffman(&zbuffer);
                    if (!succeeded) return false;
                // }
                succeeded = decodeHuffmanData(&zbuffer);
                if (!succeeded) return false;
                break;
            default:
                LOG(ERROR) << "Invalid zlib encoding method: "
                           << (uint32_t)zbuffer.encoding_method
                           << ", valid values: 0/1/2.";
                break;
        }
        // std::cout << "leftover of IDAT size: " << png_info.current_chunk.length << std::endl;

    } while (!zbuffer.is_final_block);

    return true;
}

#define STBI__BYTECAST(x)  ((uint8_t) ((x) & 255))  // truncate int to byte without warnings

enum FilterTypes {
    FILTER_NONE = 0,
    FILTER_SUB = 1,
    FILTER_UP = 2,
    FILTER_AVERAGE = 3,
    FILTER_PAETH = 4,
    // synthetic filters used for first scanline to avoid needing a dummy row of 0s
    FILTER_AVG_FIRST,
    FILTER_PAETH_FIRST,
};

static uint8_t first_row_filter[5] = {
    FILTER_NONE,
    FILTER_SUB,
    FILTER_NONE,
    FILTER_AVG_FIRST,
    FILTER_PAETH_FIRST,
};

static int filterPaeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = abs(p - a);
    int pb = abs(p - b);
    int pc = abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;

    return c;
}

// process an image with color type 0/3/4, which need not rgb-> bgr transformation.
bool PngDecoder::deFilterImage(PngInfo& png_info, uint8_t* image,
                               uint32_t stride) {
    std::cout << "########coming in deFilterImage#######" << std::endl;
    int bytes = (bit_depth_ == 16 ? 2 : 1);
    int pixel_bytes = bytes * encoded_channels_;
    if (bit_depth_ < 8) {
        pixel_bytes = 1;
    }
    int k;
    uint32_t scanline_bytes = ((width_ * encoded_channels_ * bit_depth_) >> 3) + 1;
    uint32_t stride_offset0 = 0;
    if (bit_depth_ < 8 || color_type_ == 3 ||
        (color_type_ == 0 && encoded_channels_ + 1 == channels_)) {
        stride_offset0 = stride - scanline_bytes;  // align with 4 bytes?
    }

    uint8_t *input = png_info.decompressed_image_start;
    uint8_t *current_row = image + stride_offset0;
    uint8_t *prior_row;
    uint8_t filter_type;
    int remainder_bytes = scanline_bytes - pixel_bytes - 1;
    for (uint32_t row = 0; row < height_; ++row) {
        prior_row = current_row - stride;
        filter_type = *input++;
        if (filter_type > 4) {
            LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                       << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                       << "Average(3)/Paeth(4)";
            return false;
        }
        // if first row, use special filter_type that doesn't sample previous row
        if (row == 0) filter_type = first_row_filter[filter_type];
        // std::cout << "row " << row << ": " << (int)filter_type << std::endl;

        // handle first pixel explicitly
        for (k = 0; k < pixel_bytes; ++k) {
            switch (filter_type) {
                case FILTER_NONE:
                case FILTER_SUB:
                case FILTER_AVG_FIRST:
                case FILTER_PAETH_FIRST:
                    current_row[k] = input[k];
                    break;
                case FILTER_UP:
                    current_row[k] = (uint32_t)input[k] + prior_row[k];
                    break;
                case FILTER_AVERAGE:
                    current_row[k] = (uint32_t)input[k] + (prior_row[k] >> 1);
                    break;
                case FILTER_PAETH:
                    current_row[k] = input[k] + filterPaeth(0, prior_row[k], 0);
                    break;
            }
            // std::cout << "input[" << k << "]: " << (int)input[k] << std::endl;
        }
        input += pixel_bytes;
        current_row += pixel_bytes;
        prior_row += pixel_bytes;

        // this is a little gross, so that we don't switch per-pixel or per-component
        #define STBI__CASE(f) \
            case f:     \
                for (k=0; k < remainder_bytes; ++k)
        switch (filter_type) {
            // "none" filter_type turns into a memcpy here; make that explicit.
            case FILTER_NONE:         memcpy(current_row, input, remainder_bytes);
            // std::cout << "row " << row << ", remainder_bytes: " << remainder_bytes << std::endl;
            break;
            STBI__CASE(FILTER_SUB)     { current_row[k] = STBI__BYTECAST(input[k] + current_row[k-pixel_bytes]); } break;
            STBI__CASE(FILTER_UP)      { current_row[k] = STBI__BYTECAST(input[k] + prior_row[k]); } break;
            STBI__CASE(FILTER_AVERAGE)     { current_row[k] = STBI__BYTECAST(input[k] + ((prior_row[k] + current_row[k-pixel_bytes])>>1)); } break;
            STBI__CASE(FILTER_PAETH)        { current_row[k] = STBI__BYTECAST(input[k] + filterPaeth(current_row[k-pixel_bytes],prior_row[k],prior_row[k-pixel_bytes])); } break;
            STBI__CASE(FILTER_AVG_FIRST)    { current_row[k] = STBI__BYTECAST(input[k] + (current_row[k-pixel_bytes] >> 1)); } break;
            STBI__CASE(FILTER_PAETH_FIRST)  { current_row[k] = STBI__BYTECAST(input[k] + filterPaeth(current_row[k-pixel_bytes],0,0)); } break;
        }
        #undef STBI__CASE
        //  for (int i = 0; i < remainder_bytes; i++) {
        //     std::cout << "input[" << i << "]: " << (int)input[i] << std::endl;
        //  }
        input += remainder_bytes;
        current_row += (stride - pixel_bytes);
    }

    // we make a separate pass to expand bits to pixels; for performance,
    // this could run two scanlines behind the above code, so it won't
    // intefere with filtering but will still be in the cache.
    uint32_t stride_offset1 = 0;
    if ((color_type_ == 0 && encoded_channels_ + 1 == channels_) ||
        color_type_ == 3) {
        stride_offset1 = stride - width_ * pixel_bytes;
    }
    if (bit_depth_ < 8) {
        for (uint32_t row = 0; row < height_; ++row) {
            input       = image + stride * row + stride_offset0;
            current_row = image + stride * row + stride_offset1;
            // unpack 1/2/4-bit into a 8-bit buffer. allows us to keep the common 8-bit path optimal at minimal cost for 1/2/4-bit
            uint8_t scale = (color_type_ == 0) ? stbi__depth_scale_table[bit_depth_] : 1; // scale grayscale values to 0..255 range

            if (bit_depth_ == 4) {
                for (k = width_; k >= 2; k -= 2, ++input) {
                    *current_row++ = scale * ((*input >> 4)       );
                    *current_row++ = scale * ((*input     ) & 0x0f);
                }
                if (k > 0) *current_row++ = scale * ((*input >> 4));
            } else if (bit_depth_ == 2) {
                for (k = width_; k >= 4; k -= 4, ++input) {
                    *current_row++ = scale * ((*input >> 6)       );
                    *current_row++ = scale * ((*input >> 4) & 0x03);
                    *current_row++ = scale * ((*input >> 2) & 0x03);
                    *current_row++ = scale * ((*input     ) & 0x03);
                }
                if (k > 0) *current_row++ = scale * ((*input >> 6)       );
                if (k > 1) *current_row++ = scale * ((*input >> 4) & 0x03);
                if (k > 2) *current_row++ = scale * ((*input >> 2) & 0x03);
            } else if (bit_depth_ == 1) {
                for (k = width_; k >= 8; k -= 8, ++input) {
                    *current_row++ = scale * ((*input >> 7)       );
                    *current_row++ = scale * ((*input >> 6) & 0x01);
                    *current_row++ = scale * ((*input >> 5) & 0x01);
                    *current_row++ = scale * ((*input >> 4) & 0x01);
                    *current_row++ = scale * ((*input >> 3) & 0x01);
                    *current_row++ = scale * ((*input >> 2) & 0x01);
                    *current_row++ = scale * ((*input >> 1) & 0x01);
                    *current_row++ = scale * ((*input     ) & 0x01);
                }
                if (k > 0) *current_row++ = scale * ((*input >> 7)       );
                if (k > 1) *current_row++ = scale * ((*input >> 6) & 0x01);
                if (k > 2) *current_row++ = scale * ((*input >> 5) & 0x01);
                if (k > 3) *current_row++ = scale * ((*input >> 4) & 0x01);
                if (k > 4) *current_row++ = scale * ((*input >> 3) & 0x01);
                if (k > 5) *current_row++ = scale * ((*input >> 2) & 0x01);
                if (k > 6) *current_row++ = scale * ((*input >> 1) & 0x01);
            }
        }
    } else if (bit_depth_ == 16) {
        // force the image data from big-endian to platform-native.
        // this is done in a separate pass due to the decoding relying
        // on the data being untouched, but could probably be done
        // per-line during decode if care is taken.
        uint8_t value0, value1;
        uint8_t *current_row = image + stride_offset0;
        uint16_t *current_row16;
        for (uint32_t row = 0; row < height_; row++) {
            current_row16 = (uint16_t*)current_row;
            for (uint32_t col = 0, col1 = 0; col < width_; col++, col1 += 2) {
                value0 = current_row[col1];
                value1 = current_row[col1 + 1];
                current_row16[col] = (value0 << 8) | value1;
            }
            current_row += stride;
        }
    }

    if (bit_depth_ < 8) {
        png_info.defiltered_image = image + stride_offset1;
    }
    else {
        png_info.defiltered_image = image + stride_offset0;
    }

    return true;
}

// reorder channels in a pixel form RGB to BGR for truecolor or truecolor with alpha.
bool PngDecoder::deFilterImageTrueColor(PngInfo& png_info, uint8_t* image,
                                        uint32_t stride) {
    std::cout << "######## coming in deFilterImageTrueColor #######" << std::endl;
    int bytes = (bit_depth_ == 16 ? 2 : 1);
    int pixel_bytes = bytes * encoded_channels_;
    uint32_t scanline_bytes = width_ * pixel_bytes + 1;
    uint32_t stride_offset = 0;  // align with 4 bytes?
    if (png_info.chunk_status & HAVE_tRNS) {
        stride_offset = stride - scanline_bytes;  // align with 4 bytes?
    }

    uint8_t *input = png_info.decompressed_image_start;
    uint8_t *current_row = image + stride_offset;
    uint8_t *prior_row;
    uint8_t filter_type;
    int remainder_bytes = scanline_bytes - pixel_bytes - 1;
    for (uint32_t row = 0; row < height_; ++row) {
        prior_row = current_row - stride;

        filter_type = *input++;
        if (filter_type > 4) {
            LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                       << ", correct filter type: None(0)/Sub(1)/Up(2)/Average(3)/"
                       << "Paeth(4)";
            return false;
        }
        // if first row, use special filter_type that doesn't sample previous row
        if (row == 0) filter_type = first_row_filter[filter_type];
        // std::cout << "row " << row << ": " << (int)filter_type << std::endl;
        // std::cout << "stride_offset: " << stride_offset << std::endl;
        // std::cout << "pixel_bytes: " << pixel_bytes << std::endl;
        // std::cout << "***pixel 0: ";
        // for (int i = 0; i < pixel_bytes; i++) {
        //     std::cout << (int)input[i] << ", ";
        // }
        // std::cout << std::endl;

        // handle first pixel explicitly
        uint8_t value0, value1, value2, value3, value4, value5, value6, value7;
        switch (filter_type) {
            case FILTER_NONE:
            case FILTER_SUB:
            case FILTER_AVG_FIRST:
            case FILTER_PAETH_FIRST:
                if (pixel_bytes == 3) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    current_row[0] = value2;
                    current_row[1] = value1;
                    current_row[2] = value0;
                }
                else if (pixel_bytes == 4) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    current_row[0] = value2;
                    current_row[1] = value1;
                    current_row[2] = value0;
                    current_row[3] = value3;
                }
                else if (pixel_bytes == 6) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    current_row[0] = value4;
                    current_row[1] = value5;
                    current_row[2] = value2;
                    current_row[3] = value3;
                    current_row[4] = value0;
                    current_row[5] = value1;
                }
                else {  // pixel_bytes == 8
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    value6 = input[6];
                    value7 = input[7];
                    current_row[0] = value4;
                    current_row[1] = value5;
                    current_row[2] = value2;
                    current_row[3] = value3;
                    current_row[4] = value0;
                    current_row[5] = value1;
                    current_row[6] = value6;
                    current_row[7] = value7;
                }
                break;
            case FILTER_UP:
                if (pixel_bytes == 3) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    current_row[0] = (uint32_t)value2 + prior_row[0];
                    current_row[1] = (uint32_t)value1 + prior_row[1];
                    current_row[2] = (uint32_t)value0 + prior_row[2];
                }
                else if (pixel_bytes == 4) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    current_row[0] = (uint32_t)value2 + prior_row[0];
                    current_row[1] = (uint32_t)value1 + prior_row[1];
                    current_row[2] = (uint32_t)value0 + prior_row[2];
                    current_row[3] = (uint32_t)value3 + prior_row[3];
                }
                else if (pixel_bytes == 6) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    current_row[0] = (uint32_t)value4 + prior_row[0];
                    current_row[1] = (uint32_t)value5 + prior_row[1];
                    current_row[2] = (uint32_t)value2 + prior_row[2];
                    current_row[3] = (uint32_t)value3 + prior_row[3];
                    current_row[4] = (uint32_t)value0 + prior_row[4];
                    current_row[5] = (uint32_t)value1 + prior_row[5];
                }
                else {  // pixel_bytes == 8
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    value6 = input[6];
                    value7 = input[7];
                    current_row[0] = (uint32_t)value4 + prior_row[0];
                    current_row[1] = (uint32_t)value5 + prior_row[1];
                    current_row[2] = (uint32_t)value2 + prior_row[2];
                    current_row[3] = (uint32_t)value3 + prior_row[3];
                    current_row[4] = (uint32_t)value0 + prior_row[4];
                    current_row[5] = (uint32_t)value1 + prior_row[5];
                    current_row[6] = (uint32_t)value6 + prior_row[6];
                    current_row[7] = (uint32_t)value7 + prior_row[7];
                }
                break;
            case FILTER_AVERAGE:
                if (pixel_bytes == 3) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    current_row[0] = (uint32_t)value2 + (prior_row[0] >> 1);
                    current_row[1] = (uint32_t)value1 + (prior_row[1] >> 1);
                    current_row[2] = (uint32_t)value0 + (prior_row[2] >> 1);
                }
                else if (pixel_bytes == 4) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    current_row[0] = (uint32_t)value2 + (prior_row[0] >> 1);
                    current_row[1] = (uint32_t)value1 + (prior_row[1] >> 1);
                    current_row[2] = (uint32_t)value0 + (prior_row[2] >> 1);
                    current_row[3] = (uint32_t)value3 + (prior_row[3] >> 1);
                }
                else if (pixel_bytes == 6) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    current_row[0] = (uint32_t)value4 + (prior_row[0] >> 1);
                    current_row[1] = (uint32_t)value5 + (prior_row[1] >> 1);
                    current_row[2] = (uint32_t)value2 + (prior_row[2] >> 1);
                    current_row[3] = (uint32_t)value3 + (prior_row[3] >> 1);
                    current_row[4] = (uint32_t)value0 + (prior_row[4] >> 1);
                    current_row[5] = (uint32_t)value1 + (prior_row[5] >> 1);
                }
                else {  // pixel_bytes == 8
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    value6 = input[6];
                    value7 = input[7];
                    current_row[0] = (uint32_t)value4 + (prior_row[0] >> 1);
                    current_row[1] = (uint32_t)value5 + (prior_row[1] >> 1);
                    current_row[2] = (uint32_t)value2 + (prior_row[2] >> 1);
                    current_row[3] = (uint32_t)value3 + (prior_row[3] >> 1);
                    current_row[4] = (uint32_t)value0 + (prior_row[4] >> 1);
                    current_row[5] = (uint32_t)value1 + (prior_row[5] >> 1);
                    current_row[6] = (uint32_t)value6 + (prior_row[6] >> 1);
                    current_row[7] = (uint32_t)value7 + (prior_row[7] >> 1);
                }
                break;
            case FILTER_PAETH:
                if (pixel_bytes == 3) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    current_row[0] = (uint32_t)value2 + filterPaeth(0, prior_row[0], 0);
                    current_row[1] = (uint32_t)value1 + filterPaeth(0, prior_row[1], 0);
                    current_row[2] = (uint32_t)value0 + filterPaeth(0, prior_row[2], 0);
                }
                else if (pixel_bytes == 4) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    current_row[0] = (uint32_t)value2 + filterPaeth(0, prior_row[0], 0);
                    current_row[1] = (uint32_t)value1 + filterPaeth(0, prior_row[1], 0);
                    current_row[2] = (uint32_t)value0 + filterPaeth(0, prior_row[2], 0);
                    current_row[3] = (uint32_t)value3 + filterPaeth(0, prior_row[3], 0);
                }
                else if (pixel_bytes == 6) {
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    current_row[0] = (uint32_t)value4 + filterPaeth(0, prior_row[0], 0);
                    current_row[1] = (uint32_t)value5 + filterPaeth(0, prior_row[1], 0);
                    current_row[2] = (uint32_t)value2 + filterPaeth(0, prior_row[2], 0);
                    current_row[3] = (uint32_t)value3 + filterPaeth(0, prior_row[3], 0);
                    current_row[4] = (uint32_t)value0 + filterPaeth(0, prior_row[4], 0);
                    current_row[5] = (uint32_t)value1 + filterPaeth(0, prior_row[5], 0);
                }
                else {  // pixel_bytes == 8
                    value0 = input[0];
                    value1 = input[1];
                    value2 = input[2];
                    value3 = input[3];
                    value4 = input[4];
                    value5 = input[5];
                    value6 = input[6];
                    value7 = input[7];
                    current_row[0] = (uint32_t)value4 + filterPaeth(0, prior_row[0], 0);
                    current_row[1] = (uint32_t)value5 + filterPaeth(0, prior_row[1], 0);
                    current_row[2] = (uint32_t)value2 + filterPaeth(0, prior_row[2], 0);
                    current_row[3] = (uint32_t)value3 + filterPaeth(0, prior_row[3], 0);
                    current_row[4] = (uint32_t)value0 + filterPaeth(0, prior_row[4], 0);
                    current_row[5] = (uint32_t)value1 + filterPaeth(0, prior_row[5], 0);
                    current_row[6] = (uint32_t)value6 + filterPaeth(0, prior_row[6], 0);
                    current_row[7] = (uint32_t)value7 + filterPaeth(0, prior_row[7], 0);
                }
                break;
        }
        //   std::cout << "input[" << k << "]: " << (int)input[k] << std::endl;
        input += pixel_bytes;
        current_row += pixel_bytes;
        prior_row += pixel_bytes;
        int32_t former_k;
        // std::cout << "***pixel 0: ";
        // for (int i = 0; i < pixel_bytes; i++) {
        //     std::cout << (int)input[i] << ", ";
        // }
        // std::cout << std::endl;

        for (int k = 0; k < remainder_bytes; k += pixel_bytes) {
            former_k = k - pixel_bytes;
            switch (filter_type) {
                case FILTER_NONE:
                    if (pixel_bytes == 3) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        current_row[k + 0] = value2;
                        current_row[k + 1] = value1;
                        current_row[k + 2] = value0;
                    }
                    else if (pixel_bytes == 4) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        current_row[k + 0] = value2;
                        current_row[k + 1] = value1;
                        current_row[k + 2] = value0;
                        current_row[k + 3] = value3;
                    }
                    else if (pixel_bytes == 6) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        current_row[k + 0] = value4;
                        current_row[k + 1] = value5;
                        current_row[k + 2] = value2;
                        current_row[k + 3] = value3;
                        current_row[k + 4] = value0;
                        current_row[k + 5] = value1;
                    }
                    else {  // pixel_bytes == 8
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        value6 = input[k + 6];
                        value7 = input[k + 7];
                        current_row[k + 0] = value4;
                        current_row[k + 1] = value5;
                        current_row[k + 2] = value2;
                        current_row[k + 3] = value3;
                        current_row[k + 4] = value0;
                        current_row[k + 5] = value1;
                        current_row[k + 6] = value6;
                        current_row[k + 7] = value7;
                    }
                    break;
                case FILTER_SUB:
                    if (pixel_bytes == 3) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        current_row[k + 0] = (uint32_t)value2 +
                                              current_row[former_k + 0];
                        current_row[k + 1] = (uint32_t)value1 +
                                              current_row[former_k + 1];
                        current_row[k + 2] = (uint32_t)value0 +
                                              current_row[former_k + 2];
                    }
                    else if (pixel_bytes == 4) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        current_row[k + 0] = (uint32_t)value2 +
                                              current_row[former_k + 0];
                        current_row[k + 1] = (uint32_t)value1 +
                                              current_row[former_k + 1];
                        current_row[k + 2] = (uint32_t)value0 +
                                              current_row[former_k + 2];
                        current_row[k + 3] = (uint32_t)value3 +
                                              current_row[former_k + 3];
                    }
                    else if (pixel_bytes == 6) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        current_row[k + 0] = (uint32_t)value4 +
                                              current_row[former_k + 0];
                        current_row[k + 1] = (uint32_t)value5 +
                                              current_row[former_k + 1];
                        current_row[k + 2] = (uint32_t)value2 +
                                              current_row[former_k + 2];
                        current_row[k + 3] = (uint32_t)value3 +
                                              current_row[former_k + 3];
                        current_row[k + 4] = (uint32_t)value0 +
                                              current_row[former_k + 4];
                        current_row[k + 5] = (uint32_t)value1 +
                                              current_row[former_k + 5];
                    }
                    else {  // pixel_bytes == 8
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        value6 = input[k + 6];
                        value7 = input[k + 7];
                        current_row[k + 0] = (uint32_t)value4 +
                                              current_row[former_k + 0];
                        current_row[k + 1] = (uint32_t)value5 +
                                              current_row[former_k + 1];
                        current_row[k + 2] = (uint32_t)value2 +
                                              current_row[former_k + 2];
                        current_row[k + 3] = (uint32_t)value3 +
                                              current_row[former_k + 3];
                        current_row[k + 4] = (uint32_t)value0 +
                                              current_row[former_k + 4];
                        current_row[k + 5] = (uint32_t)value1 +
                                              current_row[former_k + 5];
                        current_row[k + 6] = (uint32_t)value6 +
                                              current_row[former_k + 6];
                        current_row[k + 7] = (uint32_t)value7 +
                                              current_row[former_k + 7];
                    }
                    break;
                case FILTER_UP:
                    if (pixel_bytes == 3) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        current_row[k + 0] = (uint32_t)value2 + prior_row[k + 0];
                        current_row[k + 1] = (uint32_t)value1 + prior_row[k + 1];
                        current_row[k + 2] = (uint32_t)value0 + prior_row[k + 2];
                    }
                    else if (pixel_bytes == 4) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        current_row[k + 0] = (uint32_t)value2 + prior_row[k + 0];
                        current_row[k + 1] = (uint32_t)value1 + prior_row[k + 1];
                        current_row[k + 2] = (uint32_t)value0 + prior_row[k + 2];
                        current_row[k + 3] = (uint32_t)value3 + prior_row[k + 3];
                    }
                    else if (pixel_bytes == 6) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        current_row[k + 0] = (uint32_t)value4 + prior_row[k + 0];
                        current_row[k + 1] = (uint32_t)value5 + prior_row[k + 1];
                        current_row[k + 2] = (uint32_t)value2 + prior_row[k + 2];
                        current_row[k + 3] = (uint32_t)value3 + prior_row[k + 3];
                        current_row[k + 4] = (uint32_t)value0 + prior_row[k + 4];
                        current_row[k + 5] = (uint32_t)value1 + prior_row[k + 5];
                    }
                    else {  // pixel_bytes == 8
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        value6 = input[k + 6];
                        value7 = input[k + 7];
                        current_row[k + 0] = (uint32_t)value4 + prior_row[k + 0];
                        current_row[k + 1] = (uint32_t)value5 + prior_row[k + 1];
                        current_row[k + 2] = (uint32_t)value2 + prior_row[k + 2];
                        current_row[k + 3] = (uint32_t)value3 + prior_row[k + 3];
                        current_row[k + 4] = (uint32_t)value0 + prior_row[k + 4];
                        current_row[k + 5] = (uint32_t)value1 + prior_row[k + 5];
                        current_row[k + 6] = (uint32_t)value6 + prior_row[k + 6];
                        current_row[k + 7] = (uint32_t)value7 + prior_row[k + 7];
                    }
                    break;
                case FILTER_AVERAGE:
                    if (pixel_bytes == 3) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        current_row[k + 0] = (uint32_t)value2 + ((prior_row[k + 0] + current_row[former_k + 0]) >> 1);
                        current_row[k + 1] = (uint32_t)value1 + ((prior_row[k + 1] + current_row[former_k + 1]) >> 1);
                        current_row[k + 2] = (uint32_t)value0 + ((prior_row[k + 2] + current_row[former_k + 2]) >> 1);
                    }
                    else if (pixel_bytes == 4) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        current_row[k + 0] = (uint32_t)value2 + ((prior_row[k + 0] + current_row[former_k + 0]) >> 1);
                        current_row[k + 1] = (uint32_t)value1 + ((prior_row[k + 1] + current_row[former_k + 1]) >> 1);
                        current_row[k + 2] = (uint32_t)value0 + ((prior_row[k + 2] + current_row[former_k + 2]) >> 1);
                        current_row[k + 3] = (uint32_t)value3 + ((prior_row[k + 3] + current_row[former_k + 3]) >> 1);
                    }
                    else if (pixel_bytes == 6) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        current_row[k + 0] = (uint32_t)value4 + ((prior_row[k + 0] + current_row[former_k + 0]) >> 1);
                        current_row[k + 1] = (uint32_t)value5 + ((prior_row[k + 1] + current_row[former_k + 1]) >> 1);
                        current_row[k + 2] = (uint32_t)value2 + ((prior_row[k + 2] + current_row[former_k + 2]) >> 1);
                        current_row[k + 3] = (uint32_t)value3 + ((prior_row[k + 3] + current_row[former_k + 3]) >> 1);
                        current_row[k + 4] = (uint32_t)value0 + ((prior_row[k + 4] + current_row[former_k + 4]) >> 1);
                        current_row[k + 5] = (uint32_t)value1 + ((prior_row[k + 5] + current_row[former_k + 5]) >> 1);
                    }
                    else {  // pixel_bytes == 8
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        value6 = input[k + 6];
                        value7 = input[k + 7];
                        current_row[k + 0] = (uint32_t)value4 + ((prior_row[k + 0] + current_row[former_k + 0]) >> 1);
                        current_row[k + 1] = (uint32_t)value5 + ((prior_row[k + 1] + current_row[former_k + 1]) >> 1);
                        current_row[k + 2] = (uint32_t)value2 + ((prior_row[k + 2] + current_row[former_k + 2]) >> 1);
                        current_row[k + 3] = (uint32_t)value3 + ((prior_row[k + 3] + current_row[former_k + 3]) >> 1);
                        current_row[k + 4] = (uint32_t)value0 + ((prior_row[k + 4] + current_row[former_k + 4]) >> 1);
                        current_row[k + 5] = (uint32_t)value1 + ((prior_row[k + 5] + current_row[former_k + 5]) >> 1);
                        current_row[k + 6] = (uint32_t)value6 + ((prior_row[k + 6] + current_row[former_k + 6]) >> 1);
                        current_row[k + 7] = (uint32_t)value7 + ((prior_row[k + 7] + current_row[former_k + 7]) >> 1);
                    }
                    break;
                case FILTER_PAETH:
                    if (pixel_bytes == 3) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        current_row[k + 0] = (uint32_t)value2 + filterPaeth(current_row[former_k + 0], prior_row[k + 0], prior_row[former_k + 0]);
                        current_row[k + 1] = (uint32_t)value1 + filterPaeth(current_row[former_k + 1], prior_row[k + 1], prior_row[former_k + 1]);
                        current_row[k + 2] = (uint32_t)value0 + filterPaeth(current_row[former_k + 2], prior_row[k + 2], prior_row[former_k + 2]);
                    }
                    else if (pixel_bytes == 4) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        current_row[k + 0] = (uint32_t)value2 + filterPaeth(current_row[former_k + 0], prior_row[k + 0], prior_row[former_k + 0]);
                        current_row[k + 1] = (uint32_t)value1 + filterPaeth(current_row[former_k + 1], prior_row[k + 1], prior_row[former_k + 1]);
                        current_row[k + 2] = (uint32_t)value0 + filterPaeth(current_row[former_k + 2], prior_row[k + 2], prior_row[former_k + 2]);
                        current_row[k + 3] = (uint32_t)value3 + filterPaeth(current_row[former_k + 3], prior_row[k + 3], prior_row[former_k + 3]);
                    }
                    else if (pixel_bytes == 6) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        current_row[k + 0] = (uint32_t)value4 + filterPaeth(current_row[former_k + 0], prior_row[k + 0], prior_row[former_k + 0]);
                        current_row[k + 1] = (uint32_t)value5 + filterPaeth(current_row[former_k + 1], prior_row[k + 1], prior_row[former_k + 1]);
                        current_row[k + 2] = (uint32_t)value2 + filterPaeth(current_row[former_k + 2], prior_row[k + 2], prior_row[former_k + 2]);
                        current_row[k + 3] = (uint32_t)value3 + filterPaeth(current_row[former_k + 3], prior_row[k + 3], prior_row[former_k + 3]);
                        current_row[k + 4] = (uint32_t)value0 + filterPaeth(current_row[former_k + 4], prior_row[k + 4], prior_row[former_k + 4]);
                        current_row[k + 5] = (uint32_t)value1 + filterPaeth(current_row[former_k + 5], prior_row[k + 5], prior_row[former_k + 5]);
                    }
                    else {  // pixel_bytes == 8
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        value6 = input[k + 6];
                        value7 = input[k + 7];
                        current_row[k + 0] = (uint32_t)value4 + filterPaeth(current_row[former_k + 0], prior_row[k + 0], prior_row[former_k + 0]);
                        current_row[k + 1] = (uint32_t)value5 + filterPaeth(current_row[former_k + 1], prior_row[k + 1], prior_row[former_k + 1]);
                        current_row[k + 2] = (uint32_t)value2 + filterPaeth(current_row[former_k + 2], prior_row[k + 2], prior_row[former_k + 2]);
                        current_row[k + 3] = (uint32_t)value3 + filterPaeth(current_row[former_k + 3], prior_row[k + 3], prior_row[former_k + 3]);
                        current_row[k + 4] = (uint32_t)value0 + filterPaeth(current_row[former_k + 4], prior_row[k + 4], prior_row[former_k + 4]);
                        current_row[k + 5] = (uint32_t)value1 + filterPaeth(current_row[former_k + 5], prior_row[k + 5], prior_row[former_k + 5]);
                        current_row[k + 6] = (uint32_t)value6 + filterPaeth(current_row[former_k + 6], prior_row[k + 6], prior_row[former_k + 6]);
                        current_row[k + 7] = (uint32_t)value7 + filterPaeth(current_row[former_k + 7], prior_row[k + 7], prior_row[former_k + 7]);
                    }
                    break;
                case FILTER_AVG_FIRST:
                    if (pixel_bytes == 3) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        current_row[k + 0] = (uint32_t)value2 +
                                              (current_row[former_k + 0] >> 1);
                        current_row[k + 1] = (uint32_t)value1 +
                                              (current_row[former_k + 1] >> 1);
                        current_row[k + 2] = (uint32_t)value0 +
                                              (current_row[former_k + 2] >> 1);
                    }
                    else if (pixel_bytes == 4) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        current_row[k + 0] = (uint32_t)value2 +
                                              (current_row[former_k + 0] >> 1);
                        current_row[k + 1] = (uint32_t)value1 +
                                              (current_row[former_k + 1] >> 1);
                        current_row[k + 2] = (uint32_t)value0 +
                                              (current_row[former_k + 2] >> 1);
                        current_row[k + 3] = (uint32_t)value3 +
                                              (current_row[former_k + 3] >> 1);
                    }
                    else if (pixel_bytes == 6) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        current_row[k + 0] = (uint32_t)value4 +
                                              (current_row[former_k + 0] >> 1);
                        current_row[k + 1] = (uint32_t)value5 +
                                              (current_row[former_k + 1] >> 1);
                        current_row[k + 2] = (uint32_t)value2 +
                                              (current_row[former_k + 2] >> 1);
                        current_row[k + 3] = (uint32_t)value3 +
                                              (current_row[former_k + 3] >> 1);
                        current_row[k + 4] = (uint32_t)value0 +
                                              (current_row[former_k + 4] >> 1);
                        current_row[k + 5] = (uint32_t)value1 +
                                              (current_row[former_k + 5] >> 1);
                    }
                    else {  // pixel_bytes == 8
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        value6 = input[k + 6];
                        value7 = input[k + 7];
                        current_row[k + 0] = (uint32_t)value4 +
                                              (current_row[former_k + 0] >> 1);
                        current_row[k + 1] = (uint32_t)value5 +
                                              (current_row[former_k + 1] >> 1);
                        current_row[k + 2] = (uint32_t)value2 +
                                              (current_row[former_k + 2] >> 1);
                        current_row[k + 3] = (uint32_t)value3 +
                                              (current_row[former_k + 3] >> 1);
                        current_row[k + 4] = (uint32_t)value0 +
                                              (current_row[former_k + 4] >> 1);
                        current_row[k + 5] = (uint32_t)value1 +
                                              (current_row[former_k + 5] >> 1);
                        current_row[k + 6] = (uint32_t)value6 +
                                              (current_row[former_k + 6] >> 1);
                        current_row[k + 7] = (uint32_t)value7 +
                                              (current_row[former_k + 7] >> 1);
                    }
                    break;
                case FILTER_PAETH_FIRST:
                    if (pixel_bytes == 3) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        current_row[k + 0] = (uint32_t)value2 +
                            filterPaeth(current_row[former_k + 0], 0, 0);
                        current_row[k + 1] = (uint32_t)value1 +
                            filterPaeth(current_row[former_k + 1], 0, 0);
                        current_row[k + 2] = (uint32_t)value0 +
                            filterPaeth(current_row[former_k + 2], 0, 0);
                    }
                    else if (pixel_bytes == 4) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        current_row[k + 0] = (uint32_t)value2 +
                            filterPaeth(current_row[former_k + 0], 0, 0);
                        current_row[k + 1] = (uint32_t)value1 +
                            filterPaeth(current_row[former_k + 1], 0, 0);
                        current_row[k + 2] = (uint32_t)value0 +
                            filterPaeth(current_row[former_k + 2], 0, 0);
                        current_row[k + 3] = (uint32_t)value3 +
                            filterPaeth(current_row[former_k + 3], 0, 0);
                    }
                    else if (pixel_bytes == 6) {
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        current_row[k + 0] = (uint32_t)value4 +
                            filterPaeth(current_row[former_k + 0], 0, 0);
                        current_row[k + 1] = (uint32_t)value5 +
                            filterPaeth(current_row[former_k + 1], 0, 0);
                        current_row[k + 2] = (uint32_t)value2 +
                            filterPaeth(current_row[former_k + 2], 0, 0);
                        current_row[k + 3] = (uint32_t)value3 +
                            filterPaeth(current_row[former_k + 3], 0, 0);
                        current_row[k + 4] = (uint32_t)value0 +
                            filterPaeth(current_row[former_k + 4], 0, 0);
                        current_row[k + 5] = (uint32_t)value1 +
                            filterPaeth(current_row[former_k + 5], 0, 0);
                    }
                    else {  // pixel_bytes == 8
                        value0 = input[k + 0];
                        value1 = input[k + 1];
                        value2 = input[k + 2];
                        value3 = input[k + 3];
                        value4 = input[k + 4];
                        value5 = input[k + 5];
                        value6 = input[k + 6];
                        value7 = input[k + 7];
                        current_row[k + 0] = (uint32_t)value4 +
                            filterPaeth(current_row[former_k + 0], 0, 0);
                        current_row[k + 1] = (uint32_t)value5 +
                            filterPaeth(current_row[former_k + 1], 0, 0);
                        current_row[k + 2] = (uint32_t)value2 +
                            filterPaeth(current_row[former_k + 2], 0, 0);
                        current_row[k + 3] = (uint32_t)value3 +
                            filterPaeth(current_row[former_k + 3], 0, 0);
                        current_row[k + 4] = (uint32_t)value0 +
                            filterPaeth(current_row[former_k + 4], 0, 0);
                        current_row[k + 5] = (uint32_t)value1 +
                            filterPaeth(current_row[former_k + 5], 0, 0);
                        current_row[k + 6] = (uint32_t)value6 +
                            filterPaeth(current_row[former_k + 6], 0, 0);
                        current_row[k + 7] = (uint32_t)value7 +
                            filterPaeth(current_row[former_k + 7], 0, 0);
                    }
                    break;
            }
        }
        //  for (int i = 0; i < remainder_bytes; i++) {
        //     std::cout << "input[" << i << "]: " << (int)input[i] << std::endl;
        //  }
        input += remainder_bytes;
        current_row += (stride - pixel_bytes);
    }

    if (bit_depth_ == 16) {
        // force the image data from big-endian to platform-native.
        // this is done in a separate pass due to the decoding relying
        // on the data being untouched, but could probably be done
        // per-line during decode if care is taken.
        uint8_t value0, value1;
        current_row = image + stride_offset;
        uint16_t *current_row16;
        for (uint32_t row = 0; row < height_; row++) {
            current_row16 = (uint16_t*)current_row;
            for (uint32_t col = 0, col1 = 0; col < width_; col++, col1 += 2) {
                value0 = current_row[col1];
                value1 = current_row[col1 + 1];
                current_row16[col] = (value0 << 8) | value1;
            }
            current_row += stride;
        }
    }
    png_info.defiltered_image = image + stride_offset;

    return true;
}

bool PngDecoder::computeTransparency(PngInfo& png_info, uint8_t* image,
                                     uint32_t stride) {
    std::cout << "########coming in computeTransparency#######" << std::endl;
    if (color_type_ != 0 && color_type_ != 2) {
        LOG(ERROR) << "The expected color type for expanding alpha: "
                   << "greyscale(0)/true color(2).";
        return false;
    }

    if (bit_depth_ <= 8) {
        uint8_t* input_line = png_info.defiltered_image;
        uint8_t* output_line = image;
        uint8_t value0, value1, value2, value3;

        if (color_type_ == 0) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_; col0++) {
                     value0 = input_line[col0];
                     value1 = value0 == png_info.alpha_values[0] ? 0 : 255;
                     output_line[col1] = value0;
                     output_line[col1 + 1] = value1;
                     col1 += 2;
                }
                input_line  += stride;
                output_line += stride;
            }
        }
        else if (color_type_ == 2) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_ * 3;
                     col0 += 3) {
                     value0 = input_line[col0];
                     value1 = input_line[col0 + 1];
                     value2 = input_line[col0 + 2];
                     if (value0 == png_info.alpha_values[0] &&
                         value1 == png_info.alpha_values[1] &&
                         value2 == png_info.alpha_values[2]) {
                         value3 = 0;
                     }
                     else {
                         value3 = 255;
                     }
                     output_line[col1] = value0;
                     output_line[col1 + 1] = value1;
                     output_line[col1 + 2] = value2;
                     output_line[col1 + 3] = value3;
                     col1 += 4;
                }
                input_line  += stride;
                output_line += stride;
            }
        }
        else {
        }
    }
    else {  // bit_depth_ == 16
        uint16_t* input_line = (uint16_t*)(png_info.defiltered_image);
        uint16_t* output_line = (uint16_t*)image;
        uint16_t value0, value1, value2, value3;

        if (color_type_ == 0) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_; col0++) {
                     value0 = input_line[col0];
                     value1 = value0 == png_info.alpha_values[0] ? 0 : 65535;
                     output_line[col1] = value0;
                     output_line[col1 + 1] = value1;
                     col1 += 2;
                }
                input_line  += stride;
                output_line += stride;
            }
        }
        else if (color_type_ == 2) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_ * 3;
                     col0 += 3) {
                     value0 = input_line[col0];
                     value1 = input_line[col0 + 1];
                     value2 = input_line[col0 + 2];
                     if (value0 == png_info.alpha_values[0] &&
                         value1 == png_info.alpha_values[1] &&
                         value2 == png_info.alpha_values[2]) {
                         value3 = 0;
                     }
                     else {
                         value3 = 65535;
                     }
                     output_line[col1] = value0;
                     output_line[col1 + 1] = value1;
                     output_line[col1 + 2] = value2;
                     output_line[col1 + 3] = value3;
                     col1 += 4;
                }
                input_line  += stride;
                output_line += stride;
            }
        }
        else {
        }
    }

    return true;
}

bool PngDecoder::expandPalette(PngInfo& png_info, uint8_t* image,
                               uint32_t stride) {
    std::cout << "########coming in expandPalette#######" << std::endl;
    if (color_type_ != 3) {
        LOG(ERROR) << "The expected color type for expanding palette: "
                   << "indexed-colour(3).";
        return false;
    }

    uint8_t* input_line = png_info_.defiltered_image;
    uint8_t* palette = png_info.palette;
    uint8_t* output_line = image;
    uint32_t index;
    uint8_t value0, value1, value2, value3;

    if (channels_ == 3) {
        for (uint32_t row = 0; row < height_; row++) {
            for (uint32_t col0 = 0, col1 = 0; col0 < width_; col0++) {
                index = input_line[col0];
                index *= 4;
                value0 = palette[index];
                value1 = palette[index + 1];
                value2 = palette[index + 2];
                output_line[col1] = value0;
                output_line[col1 + 1] = value1;
                output_line[col1 + 2] = value2;
                col1 += 3;
            }
            input_line  += stride;
            output_line += stride;
        }
    }
    else if (channels_ == 4) {
        for (uint32_t row = 0; row < height_; row++) {
            for (uint32_t col0 = 0, col1 = 0; col0 < width_; col0++) {
                index = input_line[col0];
                index *= 4;
                value0 = palette[index];
                value1 = palette[index + 1];
                value2 = palette[index + 2];
                value3 = palette[index + 3];
                output_line[col1] = value0;
                output_line[col1 + 1] = value1;
                output_line[col1 + 2] = value2;
                output_line[col1 + 3] = value3;
                col1 += 4;
            }
            input_line  += stride;
            output_line += stride;
        }
    }
    else {
    }

    return true;
}

bool PngDecoder::readHeader() {
    // file_data_->skip(8);
    getChunkHeader(png_info_);
    if (png_info_.current_chunk.type != png_IHDR) {
        std::string chunk_name = getChunkName(png_info_.current_chunk.type);
        // std::string chunk_name = getChunkName(png_IHDR);
        // for (auto c : chunk_name) {
        //     std::cout << (uint32_t)c << ", ";
        // }
        // std::cout << "string size: " << chunk_name.size() << std::endl;
        LOG(ERROR) << "encountering the chunk: " << chunk_name
                   << ", expecting an IHDR chunk.";
        LOG(ERROR) << "The first chunk must be IHDR.";
        return false;
    }

    bool succeeded = parseIHDR(png_info_);
    if (!succeeded) return false;
    // return true;  // debug

    // png_info_.current_chunk.crc = file_data_->getDWordBigEndian1();
    // crc checking.
    // file_data_->skip(4);

    getChunkHeader(png_info_);
    while (png_info_.current_chunk.type != png_IDAT &&
           png_info_.current_chunk.type != png_IEND) {
        switch (png_info_.current_chunk.type) {
            case png_tIME:
                succeeded = parsetIME(png_info_);
                if (!succeeded) return false;
                break;
            case png_zTXt:
                succeeded = parsezTXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_tEXt:
                succeeded = parsetEXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_iTXt:
                succeeded = parseiTXt(png_info_);
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
            case png_iCCP:
                succeeded = parseiCCP(png_info_);
                if (!succeeded) return false;
                break;
            case png_sRGB:
                succeeded = parsesRGB(png_info_);
                if (!succeeded) return false;
                break;
            case png_sBIT:
                succeeded = parsesBIT(png_info_);
                if (!succeeded) return false;
                break;
            case png_gAMA:
                succeeded = parsegAMA(png_info_);
                if (!succeeded) return false;
                break;
            case png_cHRM:
                succeeded = parsecHRM(png_info_);
                if (!succeeded) return false;
                break;
            case png_PLTE:
                succeeded = parsePLTE(png_info_);
                if (!succeeded) return false;
                break;
            case png_tRNS:
                succeeded = parsetRNS(png_info_);
                if (!succeeded) return false;
                break;
            case png_hIST:
                succeeded = parsehIST(png_info_);
                if (!succeeded) return false;
                break;
            case png_bKGD:
                succeeded = parsebKGD(png_info_);
                if (!succeeded) return false;
                break;
            default:
                succeeded = parsetUnknownChunk(png_info_);
                if (!succeeded) return false;
                break;
        }

        // file_data_->skip(4);  // skip crc checking.
        getChunkHeader(png_info_);
    }

    if (png_info_.current_chunk.type == png_IEND) {
        LOG(ERROR) << "No image data chunk appears.";
        return false;
    }

    return true;
}

bool PngDecoder::decodeData(uint32_t stride, uint8_t* image) {
    // return true;  // debug
    if (png_info_.current_chunk.type != png_IDAT) {
        LOG(ERROR) << "encountering the chunk: "
                   << getChunkName(png_info_.current_chunk.type)
                   << ", expecting the IDAT chunk.";
        return false;
    }

    png_info_.decompressed_image_size = (((width_ * encoded_channels_ * bit_depth_
                                       + 7) >> 3) + 1) * height_;
    png_info_.decompressed_image_end = image + stride * height_;
    png_info_.decompressed_image_start = png_info_.decompressed_image_end -
                                       png_info_.decompressed_image_size;
    png_info_.decompressed_image = png_info_.decompressed_image_start;
    png_info_.zbuffer.num_bits = 0;
    png_info_.zbuffer.code_buffer = 0;
    // png_info_.unprocessed_zbuffer_size = 0;

    bool succeeded;
    // int count = 0;
    while (png_info_.current_chunk.type != png_IEND) {
        switch (png_info_.current_chunk.type) {
            case png_tIME:
                succeeded = parsetIME(png_info_);
                if (!succeeded) return false;
                break;
            case png_zTXt:
                succeeded = parsezTXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_tEXt:
                succeeded = parsetEXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_iTXt:
                succeeded = parseiTXt(png_info_);
                if (!succeeded) return false;
                break;
            case png_IDAT:
                succeeded = parseIDATs(png_info_, image, stride);
                if (!succeeded) return false;
                // std::cout << "processing IDAT chunk: " << count++ << std::endl;
                break;
            default:
                succeeded = parsetUnknownChunk(png_info_);
                if (!succeeded) return false;
                break;
        }

        // file_data_->skip(png_info_.current_chunk.length);
        // file_data_->skip(4);  // skip crc checking.
        if (!png_info_.header_after_idat) {
            getChunkHeader(png_info_);
        }
    }

    succeeded = parseIEND(png_info_);
    if (!succeeded) {
        return false;
    }

    if (color_type_ != 2 && color_type_ != 6) {  // not truecolour or truecolor with alpha
        succeeded = deFilterImage(png_info_, image, stride);
    }
    else {
        succeeded = deFilterImageTrueColor(png_info_, image, stride);
    }
    if (!succeeded) {
        return false;
    }

    if (color_type_ == 0 || color_type_ == 2) {
        if (encoded_channels_ + 1 == channels_) {
            succeeded = computeTransparency(png_info_, image, stride);
            if (!succeeded) {
                return false;
            }
        }
    }
    else if (color_type_ == 3) {
        succeeded = expandPalette(png_info_, image, stride);
        if (!succeeded) {
            return false;
        }
    }
    else {
    }

    return true;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl