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
#include <immintrin.h>

#include "ppl/cv/types.h"
#include "ppl/common/log.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace x86 {

#define MAKE_CHUNK_TYPE(s0, s1, s2, s3) (((uint32_t)(s0) << 24) | \
                                         ((uint32_t)(s1) << 16) | \
                                         ((uint32_t)(s2) << 8)  | \
                                         ((uint32_t)(s3)))
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

enum FilterTypes {
    FILTER_NONE    = 0,
    FILTER_SUB     = 1,
    FILTER_UP      = 2,
    FILTER_AVERAGE = 3,
    FILTER_PAETH   = 4,
};

static const uint8_t depth_scales[9] = {0, 0xff, 0x55, 0, 0x11, 0, 0, 0, 0x01};

static const uint8_t default_length_sizes[SYMBOL_NUMBER] = {
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8
};

static const uint32_t default_distance_sizes = 5;

static const int length_base[31] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0};

static const int length_extra_bits[31] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5,
    5, 5, 5, 0, 0 ,0};

static const int distance_base[32] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0};

static const int distance_extra_bits[32] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    11, 11, 12, 12, 13, 13};

PngDecoder::PngDecoder(BytesReader& file_data) {
    file_data_ = &file_data;
    png_info_.palette_length = 0;
    png_info_.palette = nullptr;
    png_info_.chunk_status = 0;
    encoded_channels_ = 0;

    file_data_->setCrcChecking(&crc32_);
    png_info_.fixed_huffman_done = false;
    png_info_.header_after_idat = false;
    // crc32_.turnOff();
}

PngDecoder::~PngDecoder() {
    if (png_info_.palette != nullptr) {
        delete [] png_info_.palette;
    }

    file_data_->unsetCrcChecking();
}

void PngDecoder::getChunkHeader(PngInfo& png_info) {
    png_info.current_chunk.length = file_data_->getDWordBigEndian();
    png_info.current_chunk.type   = file_data_->getDWordBigEndian();
}

bool PngDecoder::parseIHDR(PngInfo& png_info) {
    if (png_info.current_chunk.length != 13) {
        LOG(ERROR) << "The IHDR length: " << png_info.current_chunk.length
                   << ", correct bytes: 13.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    width_  = file_data_->getDWordBigEndian();
    height_ = file_data_->getDWordBigEndian();
    if (height_ < 1 || width_ < 1) {
        LOG(ERROR) << "Invalid image height/width: " << height_ << ", "
                   << width_;
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
    if (color_type_ != GRAY && color_type_ != TRUE_COLOR &&
        color_type_ != INDEXED_COLOR && color_type_ != GRAY_WITH_ALPHA &&
        color_type_ != TRUE_COLOR_WITH_ALPHA) {
        LOG(ERROR) << "Invalid color type: " << color_type_
                   << ", valid value: greyscale(0)/true color(2)/"
                   << "indexed-color(3)/greyscale with alpha(4)/"
                   << "true color with alpha(6).";
        return false;
    }

    compression_method_ = file_data_->getByte();
    if (compression_method_ != DEFLATE) {
        LOG(ERROR) << "Invalid compression method: " << compression_method_
                   << ", only deflate(0) compression is supported.";
        return false;
    }

    filter_method_ = file_data_->getByte();
    if (filter_method_ != ADAPTIVE) {
        LOG(ERROR) << "Invalid filter method: " << filter_method_
                   << ", only adaptive filtering(0) is supported.";
        return false;
    }

    interlace_method_ = file_data_->getByte();
    if (interlace_method_ != NO_INTRLACE && interlace_method_ != ADAM7) {
        LOG(ERROR) << "Invalid interlace method: " << interlace_method_
                   << ", only no interlace(0) and Adam7 interlace(1) are "
                   << "supported.";
        return false;
    }

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    if (color_type_ == GRAY) {
        channels_ = 1;
        encoded_channels_ = 1;
    }
    else if (color_type_ == TRUE_COLOR) {
        channels_ = 3;  // further check tRNS for alpha channel
        encoded_channels_ = 3;
    }
    else if (color_type_ == INDEXED_COLOR) {
        channels_ = 3;  // further check tRNS for alpha channel
        encoded_channels_ = 1;
    }
    else if (color_type_ == GRAY_WITH_ALPHA) {
        channels_ = 2;
        encoded_channels_ = 2;
    }
    else {  // TRUE_COLOR_WITH_ALPHA
        channels_ = 4;
        encoded_channels_ = 4;
    }

    if (width_ * channels_ * height_ > MAX_IMAGE_SIZE) {
        LOG(ERROR) << "The target image: " << width_ << " * " << channels_
                   << " * " << height_ << ", it is too large to be decoded.";
        return false;
    }

    if (color_type_ == TRUE_COLOR && (bit_depth_ == 1 || bit_depth_ == 2 ||
                                      bit_depth_ == 4)) {
        LOG(ERROR) << "Invalid bit depth for true color: " << bit_depth_
                   << ", valid value: 8/16.";
        return false;
    }
    if (color_type_ == INDEXED_COLOR && bit_depth_ == 16) {
        LOG(ERROR) << "Invalid bit depth for indexed color: " << bit_depth_
                   << ", valid value: 1/2/4/8.";
        return false;
    }
    if ((color_type_ == GRAY_WITH_ALPHA ||
         color_type_ == TRUE_COLOR_WITH_ALPHA) &&
        (bit_depth_ == 1 || bit_depth_ == 2 || bit_depth_ == 4)) {
        LOG(ERROR) << "Invalid bit depth for greyscale/true color with alpha: "
                   << bit_depth_ << ", valid value: 8/16.";
        return false;
    }

    if (bit_depth_ == 16) {
        LOG(ERROR) << "This implementation does not support bit depth 16.";
        return false;
    }

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

    file_data_->skipBytes(7);
    // LOG(INFO) << "The tIME chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(png_info.current_chunk.length);
    // LOG(INFO) << "A zTXt chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(png_info.current_chunk.length);
    // LOG(INFO) << "A tEXt chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(png_info.current_chunk.length);
    // LOG(INFO) << "A iTXt chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(9);
    // LOG(INFO) << "The pHYs chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(png_info.current_chunk.length);
    // LOG(INFO) << "A sPLT chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_iCCP;
    // LOG(INFO) << "The iCCP chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(1);
    png_info.chunk_status |= HAVE_sRGB;
    // LOG(INFO) << "The sRGB chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(png_info.current_chunk.length);
    // LOG(INFO) << "The sBIT chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(4);
    // LOG(INFO) << "The gAMA chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    file_data_->skipBytes(32);
    // LOG(INFO) << "The cHRM chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

// reorder the channels from RGB to BGR for truecolour.
bool PngDecoder::parsePLTE(PngInfo& png_info) {
    if (color_type_ == GRAY_WITH_ALPHA ||
        color_type_ == TRUE_COLOR_WITH_ALPHA) {
        LOG(ERROR) << "The PLTE chunk must not come with greyscale/greyscale "
                   << "with alpha.";
        return false;
    }

    if ((png_info.chunk_status & HAVE_tRNS) ||
        (png_info.chunk_status & HAVE_hIST) ||
        (png_info.chunk_status & HAVE_bKGD)) {
        LOG(ERROR) << "The PLTE chunk must come before the tRNS/hIST/bKGD "
                   << "chunk.";
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
        png_info.palette[index]     = value2;
        png_info.palette[index + 1] = value1;
        png_info.palette[index + 2] = value0;
        png_info.palette[index + 3] = 255;
    }
    png_info.chunk_status |= HAVE_PLTE;

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsetRNS(PngInfo& png_info) {
    if (color_type_ == GRAY_WITH_ALPHA ||
        color_type_ == TRUE_COLOR_WITH_ALPHA) {
        LOG(ERROR) << "The tRNS chunk must not come with greyscale/truecolor "
                   << "with alpha.";
        return false;
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    if (color_type_ == 3) {
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

        for (uint32_t index = 3; index < png_info.current_chunk.length * 4;
             index += 4) {
            png_info.palette[index] = file_data_->getByte();
        }
        channels_ = 4;
    }
    else {
        if (color_type_ & GRAY_WITH_ALPHA) {
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
            uint16_t* values = (uint16_t*)png_info.alpha_values;
            if (color_type_ == 0) {
                values[0] = file_data_->getWordBigEndian();
            }
            else {  // color_type_ == TRUE_COLOR
                values[2] = file_data_->getWordBigEndian();
                values[1] = file_data_->getWordBigEndian();
                values[0] = file_data_->getWordBigEndian();
            }
        }
        else {
            uint8_t* values = png_info.alpha_values;
            if (color_type_ == 0) {
                values[0] = (file_data_->getWordBigEndian() & 0xFF) *
                             depth_scales[bit_depth_];
            }
            else {  // color_type_ == TRUE_COLOR
                values[2] = (file_data_->getWordBigEndian() & 0xFF) *
                             depth_scales[bit_depth_];
                values[1] = (file_data_->getWordBigEndian() & 0xFF) *
                             depth_scales[bit_depth_];
                values[0] = (file_data_->getWordBigEndian() & 0xFF) *
                             depth_scales[bit_depth_];
            }
        }
        channels_++;
    }
    png_info.chunk_status |= HAVE_tRNS;

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsehIST(PngInfo& png_info) {
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

    file_data_->skipBytes(png_info.current_chunk.length);
    // LOG(INFO) << "The hIST chunk is skipped.";
    png_info.chunk_status |= HAVE_hIST;

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::parsebKGD(PngInfo& png_info) {
    if (color_type_ == GRAY || color_type_ == GRAY_WITH_ALPHA) {
        if (png_info.current_chunk.length != 2) {
            LOG(ERROR) << "The bKGD length: " << png_info.current_chunk.length
                       << ", correct value: 2.";
            return false;
        }
    }
    else if (color_type_ == TRUE_COLOR ||
             color_type_ == TRUE_COLOR_WITH_ALPHA) {
        if (png_info.current_chunk.length != 6) {
            LOG(ERROR) << "The bKGD length: " << png_info.current_chunk.length
                       << ", correct value: 6.";
            return false;
        }
    }
    else {  // indexed color
        if (png_info.current_chunk.length != 1) {
            LOG(ERROR) << "The bKGD length: " << png_info.current_chunk.length
                       << ", correct value: 1.";
            return false;
        }
    }

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    file_data_->skipBytes(png_info.current_chunk.length);
    png_info.chunk_status |= HAVE_bKGD;
    // LOG(INFO) << "The bKGD chunk is skipped.";

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
    succeeded = isCrcCorrect();
    if (!succeeded) return false;

    return true;
}

// This function processes one or more IDAT chunk.
bool PngDecoder::parseIDATs(PngInfo& png_info, uint8_t* image,
                            uint32_t stride) {
    if (color_type_ == INDEXED_COLOR &&
        (!(png_info.chunk_status & HAVE_PLTE))) {
        LOG(ERROR) << "A color palette chunk is needed for indexed color, but "
                   << " no one appears befor an IDAT chunk.";
        return false;
    }

    // processing crc of the first or current IDAT chunk.
    bool succeeded = setCrc32();
    if (!succeeded) return false;

    if (!(png_info.chunk_status & HAVE_IDAT)) {  // first IDAT chunk
        succeeded = parseDeflateHeader();
        if (!succeeded) return false;
    }

    succeeded = inflateImage(png_info, image, stride);
    if (!succeeded) return false;
    // skipping 32 bits of the optional ADLER32 checksum in zlib data stream.
    if (png_info.current_chunk.type == png_IDAT) {
        file_data_->skipBytes(png_info.current_chunk.length);
    }

    // process crc of the current or the last IDAT chunk.
    if (png_info.current_chunk.type == png_IDAT) {
        png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

    bool succeeded = setCrc32();
    if (!succeeded) return false;

    png_info.current_chunk.crc = file_data_->getDWordBigEndian();
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

        file_data_->skipBytes(png_info.current_chunk.length);
        // LOG(INFO) << "Encountering an unknown ancillary chunk: " << chunk_name;

        png_info.current_chunk.crc = file_data_->getDWordBigEndian();
        succeeded = isCrcCorrect();
        if (!succeeded) return false;

        return true;
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

bool PngDecoder::setCrc32() {
    if (!crc32_.isChecking()) {
        return true;
    }

    if (png_info_.current_chunk.length != 0) {
        uint8_t* buffer = file_data_->getCurrentPosition();
        uint32_t buffer_size = file_data_->getValidSize();
        crc32_.setCrc(buffer, buffer_size, 0, png_info_.current_chunk.length);

        bool succeeded = crc32_.calculateCrc(png_info_.current_chunk.type);
        if (!succeeded) return false;
        succeeded = crc32_.calculateCrc();
        if (!succeeded) return false;
    }
    else {
        crc32_.setCrc(nullptr, 0, 0, 0);
        bool succeeded = crc32_.calculateCrc(png_info_.current_chunk.type);
        if (!succeeded) return false;
    }

    return true;
}

bool PngDecoder::isCrcCorrect() {
    if (!crc32_.isChecking()) return true;

    uint32_t calculated_crc = crc32_.getCrcValue();
    uint32_t transmitted_crc = png_info_.current_chunk.crc;
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
    if (fdict) {
        LOG(ERROR) << "A preset dictonary is not allowed in the PNG "
                   << "specification.";
        return false;
    }
    png_info_.current_chunk.length -= 2;
    png_info_.zlib_buffer.window_size = 1 << (compresson_info + 8);

    return true;
}

bool PngDecoder::fillBits(ZlibBuffer *zlib_buffer) {
    uint64_t segment;
    if (png_info_.current_chunk.length > 8) {
        memcpy(&segment, file_data_->getCurrentPosition(), sizeof(uint64_t));

        zlib_buffer->code_buffer |= segment << zlib_buffer->bit_number;
        int32_t shift = 7 - ((zlib_buffer->bit_number >> 3) & 7);
        file_data_->skipBytes(shift);
        png_info_.current_chunk.length -= shift;
        zlib_buffer->bit_number |= 56;

        return true;
    }

    do {
        if (png_info_.current_chunk.length == 0) {
            png_info_.current_chunk.crc = file_data_->getDWordBigEndian();
            bool succeeded = isCrcCorrect();
            if (!succeeded) return false;

            getChunkHeader(png_info_);
            if (png_info_.current_chunk.type == png_IDAT) {
                succeeded = setCrc32();
                if (!succeeded) return false;
            }
            else {  // (png_info_.current_chunk.type != png_IDAT)
                png_info_.header_after_idat = true;
                return true;
            }
        }

        zlib_buffer->code_buffer |= ((uint64_t)file_data_->getByte() <<
                                     zlib_buffer->bit_number);
        zlib_buffer->bit_number += 8;
        png_info_.current_chunk.length -= 1;
    } while (zlib_buffer->bit_number <= PNG_SHIFT_SIZE);

    return true;
}

uint32_t PngDecoder::getNumber(ZlibBuffer *zlib_buffer, uint32_t bit_number) {
    if (zlib_buffer->bit_number < bit_number) {
        bool succeeded = fillBits(zlib_buffer);
        if (!succeeded) return UINT32_MAX;
    }

    uint32_t value = zlib_buffer->code_buffer & ((1 << bit_number) - 1);
    zlib_buffer->code_buffer >>= bit_number;
    zlib_buffer->bit_number   -= bit_number;

    return value;
}

bool PngDecoder::parseZlibUncompressedBlock(PngInfo& png_info) {
    ZlibBuffer &zlib_buffer = png_info.zlib_buffer;
    uint8_t header[4];
    uint32_t ignored_bits = zlib_buffer.bit_number & 7;
    if (ignored_bits) {
        getNumber(&zlib_buffer, ignored_bits);
    }
    uint32_t index = 0;
    while (zlib_buffer.bit_number > 0 && index < 4) {
        header[index++] = (uint8_t)(zlib_buffer.code_buffer & 255);
        zlib_buffer.code_buffer >>= 8;
        zlib_buffer.bit_number -= 8;
    }
    if (zlib_buffer.bit_number == 0 && index < 4) {
        while (index < 4) {
            header[index++] = file_data_->getByte();
            png_info_.current_chunk.length -= 1;
        }
    }
    uint32_t len  = header[1] * 256 + header[0];
    uint32_t nlen = header[3] * 256 + header[2];
    if (nlen != (len ^ 0xffff)) {
        LOG(ERROR) << "Non-compressed block in zlib is corrupt.";
        return false;
    }
    if (png_info.decompressed_image + len > png_info.decompressed_image_end) {
        LOG(ERROR) << "No space stores decompressed blocks in zlib.";
        return false;
    }

    if (zlib_buffer.bit_number > 0) {
        uint32_t size = zlib_buffer.bit_number >> 3;
        memcpy(png_info.decompressed_image, &(zlib_buffer.code_buffer), size);
        png_info.decompressed_image += size;
        zlib_buffer.code_buffer = 0;
        zlib_buffer.bit_number = 0;
        len -= size;
    }

    do {
        if (len <= png_info.current_chunk.length) {
            file_data_->getBytes(png_info.decompressed_image, len);
            png_info.decompressed_image += len;
            png_info.current_chunk.length -= len;
            len = 0;
        }
        else {
            file_data_->getBytes(png_info.decompressed_image,
                                 png_info.current_chunk.length);
            png_info.decompressed_image += png_info.current_chunk.length;
            len -= png_info.current_chunk.length;

            png_info_.current_chunk.crc = file_data_->getDWordBigEndian();
            bool succeeded = isCrcCorrect();
            if (!succeeded) return false;

            getChunkHeader(png_info_);
            if (png_info_.current_chunk.type == png_IDAT) {
                succeeded = setCrc32();
                if (!succeeded) return false;
            }
            if (png_info_.current_chunk.type != png_IDAT) {
                png_info_.header_after_idat = true;
                return true;
            }
        }
    } while (len > 0);

    return true;
}

static int32_t reverseBits(int32_t input, int32_t bits) {
    input = ((input & 0xAAAA) >> 1) | ((input & 0x5555) << 1);
    input = ((input & 0xCCCC) >> 2) | ((input & 0x3333) << 2);
    input = ((input & 0xF0F0) >> 4) | ((input & 0x0F0F) << 4);
    input = ((input & 0xFF00) >> 8) | ((input & 0x00FF) << 8);

    int32_t value = input >> (16 - bits);

    return value;
}

static int32_t reverse16Bits(uint16_t input) {
    input = ((input & 0xAAAA) >> 1) | ((input & 0x5555) << 1);
    input = ((input & 0xCCCC) >> 2) | ((input & 0x3333) << 2);
    input = ((input & 0xF0F0) >> 4) | ((input & 0x0F0F) << 4);
    input = ((input & 0xFF00) >> 8) | ((input & 0x00FF) << 8);

    return input;
}

bool PngDecoder::buildHuffmanCode(ZlibHuffman *huffman_coding,
                                  const uint8_t *size_list, int32_t number) {
    int32_t i, code = 0, next_code[16], sizes[17];

    memset(sizes, 0, sizeof(sizes));
    memset(huffman_coding->fast, 0, sizeof(huffman_coding->fast));
    for (i = 0; i < number; ++i) {
        ++sizes[size_list[i]];
    }
    sizes[0] = 0;
    for (i = 1; i < 16; ++i) {
        if (sizes[i] > (1 << i)) {
            LOG(ERROR) << "The size of code lengths in huffman is wrong.";
            return false;
        }
    }

    uint32_t symbol = 0;
    for (i = 1; i < 16; ++i) {
        next_code[i] = code;
        huffman_coding->first_code[i]   = (uint16_t)code;
        huffman_coding->first_symbol[i] = (uint16_t)symbol;
        code = (code + sizes[i]);
        if (sizes[i]) {
            if (code - 1 >= (1 << i)) {
                LOG(ERROR) << "The code lengths of huffman are wrong.";
                return false;
            }
        }
        huffman_coding->max_code[i] = code << (16 - i);
        code <<= 1;
        symbol += sizes[i];
    }

    huffman_coding->max_code[16] = 0x10000;
    for (i = 0; i < number; ++i) {
        int32_t size = size_list[i];
        if (size) {
            int index = next_code[size] - huffman_coding->first_code[size] +
                        huffman_coding->first_symbol[size];
            uint16_t fast_value = (uint16_t) ((size << 9) | i);
            huffman_coding->size [index] = (uint8_t)size;
            huffman_coding->value[index] = (uint16_t)i;
            if (size <= ZLIB_FAST_BITS) {
                int32_t j = reverseBits(next_code[size], size);
                while (j < (1 << ZLIB_FAST_BITS)) {
                    huffman_coding->fast[j] = fast_value;
                    j += (1 << size);
                }
            }
            ++next_code[size];
        }
    }

    return true;
}

bool PngDecoder::buildHuffmanCode(ZlibHuffman *huffman_coding,
                                  const uint32_t size, int32_t number) {
    int32_t i, code = 0, next_code[16], sizes[17];

    memset(sizes, 0, sizeof(sizes));
    memset(huffman_coding->fast, 0, sizeof(huffman_coding->fast));
    for (i = 0; i < number; ++i) {
        ++sizes[size];
    }
    sizes[0] = 0;
    for (i = 1; i < 16; ++i) {
        if (sizes[i] > (1 << i)) {
            LOG(ERROR) << "The size of code lengths in huffman is wrong.";
            return false;
        }
    }

    uint32_t symbol = 0;
    for (i = 1; i < 16; ++i) {
        next_code[i] = code;
        huffman_coding->first_code[i]   = (uint16_t)code;
        huffman_coding->first_symbol[i] = (uint16_t)symbol;
        code = (code + sizes[i]);
        if (sizes[i]) {
            if (code - 1 >= (1 << i)) {
                LOG(ERROR) << "The code lengths of huffman are wrong.";
                return false;
            }
        }
        huffman_coding->max_code[i] = code << (16 - i);
        code <<= 1;
        symbol += sizes[i];
    }

    huffman_coding->max_code[16] = 0x10000;
    for (i = 0; i < number; ++i) {
        int index = next_code[size] - huffman_coding->first_code[size] +
                    huffman_coding->first_symbol[size];
        uint16_t fast_value = (uint16_t) ((size << 9) | i);
        huffman_coding->size [index] = (uint8_t)size;
        huffman_coding->value[index] = (uint16_t)i;
        if (size <= ZLIB_FAST_BITS) {
            int32_t j = reverseBits(next_code[size], size);
            while (j < (1 << ZLIB_FAST_BITS)) {
                huffman_coding->fast[j] = fast_value;
                j += (1 << size);
            }
        }
        ++next_code[size];
    }

    return true;
}

int32_t PngDecoder::huffmanDecodeSlowly(ZlibBuffer *zlib_buffer,
                                        ZlibHuffman *huffman_coding) {
    int32_t bits, size, index;
    bits = reverse16Bits((uint16_t)(zlib_buffer->code_buffer & 0xFFFF));
    for (size = ZLIB_FAST_BITS + 1; ; ++size) {
        if (bits < huffman_coding->max_code[size]) break;
    }
    if (size >= 16) return -1; // invalid code!
    index = (bits >> (16 - size)) - huffman_coding->first_code[size] +
            huffman_coding->first_symbol[size];
    if (index >= SYMBOL_NUMBER) return -1;
    if (huffman_coding->size[index] != size) return -1;
    zlib_buffer->code_buffer >>= size;
    zlib_buffer->bit_number   -= size;

    return huffman_coding->value[index];
}

int32_t PngDecoder::huffmanDecode(ZlibBuffer *zlib_buffer,
                                  ZlibHuffman *huffman_coding) {
    if (zlib_buffer->bit_number < 16) {
        fillBits(zlib_buffer);
    }

    uint32_t fast_bits, size;
    fast_bits = huffman_coding->fast[zlib_buffer->code_buffer & ZLIB_FAST_MASK];
    if (fast_bits) {
        size = fast_bits >> 9;
        zlib_buffer->code_buffer >>= size;
        zlib_buffer->bit_number   -= size;
        return fast_bits & 511;
    }

    int32_t value = huffmanDecodeSlowly(zlib_buffer, huffman_coding);

    return value;
}

bool PngDecoder::computeDynamicHuffman(ZlibBuffer* zlib_buffer) {
    const uint8_t length_dezigzag[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
                                         11, 4, 12, 3, 13, 2, 14, 1, 15 };
    uint8_t length_codes[286 + 32 + 137];  //padding for maximum single op
    uint8_t codelength_sizes[19];

    uint32_t hlit  = getNumber(zlib_buffer, 5) + 257;
    uint32_t hdist = getNumber(zlib_buffer, 5) + 1;
    uint32_t hclen = getNumber(zlib_buffer, 4) + 4;
    uint32_t total_number = hlit + hdist;

    memset(codelength_sizes, 0, sizeof(codelength_sizes));
    for (uint32_t i = 0; i < hclen; ++i) {
        uint32_t size = getNumber(zlib_buffer, 3);
        codelength_sizes[length_dezigzag[i]] = (uint8_t)size;
    }

    ZlibHuffman code_length;
    bool succeeded = buildHuffmanCode(&code_length, codelength_sizes, 19);
    if (!succeeded) return false;

    uint32_t number = 0;
    while (number < total_number) {
        int32_t length = huffmanDecode(zlib_buffer, &code_length);
        if (length < 0 || length >= 19) {
            LOG(ERROR) << "Invalid code length: " << length
                       << ", valid value: 0-19.";
            return false;
        }
        if (length < 16) {
            length_codes[number++] = (uint8_t)length;
        }
        else {
            uint8_t fill = 0;
            if (length == 16) {
                length = getNumber(zlib_buffer, 2) + 3;
                if (number == 0) {
                    LOG(ERROR) << "Invalid code length " << length
                               << " for the first code.";
                    return false;
                }
                fill = length_codes[number - 1];
            } else if (length == 17) {
                length = getNumber(zlib_buffer, 3) + 3;
            } else if (length == 18) {
                length = getNumber(zlib_buffer, 7) + 11;
            } else {
                LOG(ERROR) << "Invalid code length: " << length
                           << ", valid value: 0-19.";
                return false;
            }
            if (total_number - number < (uint32_t)length) {
                LOG(ERROR) << "Invalid repeating count: " << length
                           << ", valid value: " << total_number - number;
                return false;
            }
            memset(length_codes + number, fill, length);
            number += length;
        }
    }
    if (number != total_number) {
            LOG(ERROR) << "Invalid count of code length: " << number
                       << ", valid value: " << total_number;
            return false;
    }

    succeeded = buildHuffmanCode(&zlib_buffer->length_huffman, length_codes,
                                 hlit);
    if (!succeeded) return false;

    succeeded = buildHuffmanCode(&zlib_buffer->distance_huffman,
                                 length_codes + hlit, hdist);
    if (!succeeded) return false;

    return true;
}

bool PngDecoder::decodeHuffmanData(ZlibBuffer* zlib_buffer) {
    uint8_t *output = png_info_.decompressed_image;
    uint32_t rounded_length;

    while (1) {
        int32_t value = huffmanDecode(zlib_buffer,
                                      &zlib_buffer->length_huffman);
        if (value < 256) {
            if (value < 0) {
                LOG(ERROR) << "Invalid decoded literal/length code: " << value
                           << ", valid value: 0-285.";
                return false;
            }
            *output++ = (uint8_t)value;
        } else {
            uint32_t length, distance;
            if (value == 256) {
                png_info_.decompressed_image = output;
                return true;
            }
            value -= 257;
            length = length_base[value];
            if (length_extra_bits[value]) {
                length += getNumber(zlib_buffer, length_extra_bits[value]);
            }
            if (png_info_.decompressed_image_end - output < length) {
                LOG(ERROR) << "Invalid decoded length: " << length
                           << ", valid value: 0-"
                           << png_info_.decompressed_image_end - output;
                return false;
            }
            value = huffmanDecode(zlib_buffer, &zlib_buffer->distance_huffman);
            if (value < 0) {
                LOG(ERROR) << "Invalid decoded distance code: " << value
                           << ", valid value: 0-31.";
                return false;
            }
            distance = distance_base[value];
            if (distance_extra_bits[value]) {
                distance += getNumber(zlib_buffer, distance_extra_bits[value]);
            }
            if (output - png_info_.decompressed_image_start < distance) {
                LOG(ERROR) << "Invalid decoded distance: " << distance
                           << ", valid value: 0-"
                           << output - png_info_.decompressed_image_start;
                return false;
            }
            if (distance > zlib_buffer->window_size) {
                LOG(ERROR) << "Invalid decoded distance: " << distance
                           << ", valid value: 0-" << zlib_buffer->window_size;
                return false;
            }
            uint8_t *copy_address = (uint8_t *) (output - distance);
            if (distance == 1) {  // run of one byte, common in images.
                uint8_t source = *copy_address;
                if (output + length + 8 < png_info_.decompressed_image_end) {
                    rounded_length = (length + 7) & -8;
                    memset(output, source, rounded_length);
                }
                else {
                    memset(output, source, length);
                }
                output += length;
            } else {
                if (distance >= length) {
                    if (output + length + 8 <
                        png_info_.decompressed_image_end) {
                        rounded_length = (length + 7) & -8;
                        memcpy(output, copy_address, rounded_length);
                    }
                    else {
                        memcpy(output, copy_address, length);
                    }
                    output += length;
                }
                else {  // 2 <= distance < length
                    if (length) {
                        do {
                            *output++ = *copy_address++;
                        } while (--length);
                    }
                }
            }
        }
    }
}

bool PngDecoder::inflateImage(PngInfo& png_info, uint8_t* image,
                              uint32_t stride) {
    if (png_info.current_chunk.length == 0) {
        LOG(ERROR) << "No data in the IDAT chunk is needed to be decompressed.";
        return false;
    }

    bool succeeded;
    ZlibBuffer &zlib_buffer = png_info.zlib_buffer;
    do {
        zlib_buffer.is_final_block = getNumber(&zlib_buffer, 1);
        zlib_buffer.encoding_method = (EncodingMethod)getNumber(&zlib_buffer,
                                                                2);

        switch (zlib_buffer.encoding_method) {
            case STORED_SECTION:
                succeeded = parseZlibUncompressedBlock(png_info);
                if (!succeeded) return false;
                break;
            case STATIC_HUFFMAN:
                if (!png_info.fixed_huffman_done) {
                    succeeded = buildHuffmanCode(&zlib_buffer.length_huffman,
                                  default_length_sizes, SYMBOL_NUMBER);
                    if (!succeeded) return false;
                    succeeded = buildHuffmanCode(&zlib_buffer.distance_huffman,
                                  default_distance_sizes, 32);
                    if (!succeeded) return false;
                    png_info.fixed_huffman_done = true;
                }
                succeeded = decodeHuffmanData(&zlib_buffer);
                if (!succeeded) return false;
                break;
            case DYNAMIC_HUFFMAN:
                succeeded = computeDynamicHuffman(&zlib_buffer);
                if (!succeeded) return false;

                succeeded = decodeHuffmanData(&zlib_buffer);
                if (!succeeded) return false;
                break;
            default:
                LOG(ERROR) << "Invalid zlib encoding method: "
                           << (uint32_t)zlib_buffer.encoding_method
                           << ", valid values: stored section(0)/"
                           << "static huffman(1)/dynamic huffman(2).";
                break;
        }

    } while (!zlib_buffer.is_final_block);

    return true;
}

inline
void rowDefilter1(uint8_t* input_row, uint32_t pixel_bytes, uint32_t row_bytes,
                  uint8_t* recon_row) {
    uint32_t i = 0;
    for (; i < pixel_bytes; i++) {
        recon_row[i] = input_row[i];
    }

    uint32_t j = 0;
    for (; i < row_bytes; i++, j++) {
        recon_row[i] = input_row[i] + recon_row[j];
    }
}

inline
void rowDefilter1C2(uint8_t* input_row, uint32_t row_bytes,
                    uint8_t* recon_row) {
    recon_row[0] = input_row[0];
    recon_row[1] = input_row[1];

    uint32_t i = 2;
    uint32_t j = 0;
    for (; i < row_bytes; i += 2, j += 2) {
        recon_row[i]     = input_row[i] + recon_row[j];
        recon_row[i + 1] = input_row[i + 1] + recon_row[j + 1];
    }
}

inline
void rowDefilter2(uint8_t* input_row, uint8_t* prior_row, uint32_t row_bytes,
                  uint8_t* recon_row, bool overflowed) {
    uint32_t i = 0;
    row_bytes -= 16;
    for (; i < row_bytes; i+= 16) {
        __m128i input_data = _mm_loadu_si128((__m128i*)(input_row + i));
        __m128i prior_data = _mm_loadu_si128((__m128i*)(prior_row + i));
        __m128i current_data = _mm_add_epi8(input_data, prior_data);
        _mm_storeu_si128((__m128i*)(recon_row + i), current_data);
    }

    if (overflowed) {
        __m128i input_data = _mm_loadu_si128((__m128i*)(input_row + i));
        __m128i prior_data = _mm_loadu_si128((__m128i*)(prior_row + i));
        __m128i current_data = _mm_add_epi8(input_data, prior_data);
        _mm_storeu_si128((__m128i*)(recon_row + i), current_data);
    }
    else {
        row_bytes += 16;
        for (; i < row_bytes; i++) {
            recon_row[i] = input_row[i] + prior_row[i];
        }
    }
}

inline
void rowDefilter3(uint8_t* input_row, uint8_t* prior_row, uint32_t pixel_bytes,
                  uint32_t row_bytes, uint8_t* recon_row) {
    uint32_t i = 0;
    for (; i < pixel_bytes; i++) {
        recon_row[i] = input_row[i] + (prior_row[i] >> 1);
    }

    uint32_t j = 0;
    for (; i < row_bytes; i++, j++) {
        recon_row[i] = input_row[i] + ((recon_row[j] + prior_row[i]) >> 1);
    }
}

inline
void rowDefilter3C2(uint8_t* input_row, uint8_t* prior_row, uint32_t row_bytes,
                    uint8_t* recon_row) {
    recon_row[0] = input_row[0] + (prior_row[0] >> 1);
    recon_row[1] = input_row[1] + (prior_row[1] >> 1);

    uint32_t i = 2;
    uint32_t j = 0;
    for (; i < row_bytes; i += 2, j += 2) {
        recon_row[i]     = input_row[i] + ((recon_row[j] + prior_row[i]) >> 1);
        recon_row[i + 1] = input_row[i + 1] +
                           ((recon_row[j + 1] + prior_row[i + 1]) >> 1);
    }
}

inline
void row0Defilter3(uint8_t* input_row, uint32_t pixel_bytes,
                   uint32_t row_bytes, uint8_t* recon_row) {
    uint32_t i = 0;
    for (; i < pixel_bytes; i++) {
        recon_row[i] = input_row[i];
    }

    uint32_t j = 0;
    for (; i < row_bytes; i++, j++) {
        recon_row[i] = input_row[i] + (recon_row[j] >> 1);
    }
}

inline
int filterPaeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = abs(p - a);
    int pb = abs(p - b);
    int pc = abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;

    return c;
}

inline
void rowDefilter4(uint8_t* input_row, uint8_t* prior_row, uint32_t pixel_bytes,
                  uint32_t row_bytes, uint8_t* recon_row) {
    uint32_t i = 0;
    for (; i < pixel_bytes; i++) {
        recon_row[i] = input_row[i] + prior_row[i];
    }

    uint32_t j = 0;
    for (; i < row_bytes; i++, j++) {
        recon_row[i] = input_row[i] + filterPaeth(recon_row[j], prior_row[i],
                       prior_row[j]);
    }
}

inline
void filterPaethC2(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* src,
                   uint8_t* result) {
    int a0 = a[0];
    int a1 = a[1];
    int b0 = b[0];
    int b1 = b[1];
    int c0 = c[0];
    int c1 = c[1];
    int p0 = a0 + b0 -c0;
    int p1 = a1 + b1 -c1;
    int pa0 = abs(p0 - a0);
    int pa1 = abs(p1 - a1);
    int pb0 = abs(p0 - b0);
    int pb1 = abs(p1 - b1);
    int pc0 = abs(p0 - c0);
    int pc1 = abs(p1 - c1);
    int target0, target1;
    if (pa0 <= pb0 && pa0 <= pc0) {
        target0 = a0;
    }
    else if (pb0 <= pc0) {
        target0 = b0;
    }
    else {
        target0 = c0;
    }
    if (pa1 <= pb1 && pa1 <= pc1) {
        target1 = a1;
    }
    else if (pb1 <= pc1) {
        target1 = b1;
    }
    else {
        target1 = c1;
    }

    target0 += src[0];
    target1 += src[1];

    result[0] = target0;
    result[1] = target1;
}

inline
void rowDefilter4C2(uint8_t* input_row, uint8_t* prior_row,
                    uint32_t row_bytes, uint8_t* recon_row) {
    recon_row[0] = input_row[0] + prior_row[0];
    recon_row[1] = input_row[1] + prior_row[1];

    uint32_t i = 2;
    uint32_t j = 0;
    for (; i < row_bytes; i += 2, j += 2) {
        filterPaethC2(recon_row + j, prior_row + i, prior_row + j,
                      input_row + i, recon_row + i);
    }
}

/* process an image with color type 0/3/4, which does not need rgb-> bgr
 * transformation.
 */
bool PngDecoder::deFilterImage(PngInfo& png_info, uint8_t* image,
                               uint32_t stride) {
    int bytes = (bit_depth_ == 16 ? 2 : 1);
    int pixel_bytes = bytes * encoded_channels_;  // 1, 2, 4
    uint32_t scanline_bytes = ((width_ * encoded_channels_ * bit_depth_ +
                                7) >> 3) + 1;
    uint32_t row_bytes = scanline_bytes - 1;
    uint32_t stride_offset0 = 0;
    if (bit_depth_ < 8 || color_type_ == 3 ||
        (color_type_ == 0 && encoded_channels_ + 1 == channels_)) {
        stride_offset0 = stride - scanline_bytes;  // can't align with 16 bytes
    }

    uint8_t *input = png_info.decompressed_image_start;
    uint8_t *recon_row = image + stride_offset0;
    uint8_t *prior_row;
    uint8_t filter_type;
    uint32_t aligned_bytes = (row_bytes + 15) & -16;

    // process the first row.
    filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                   << "Average(3)/Paeth(4)";
        return false;
    }
    switch (filter_type) {
        case FILTER_NONE:
        case FILTER_UP:
            memcpy(recon_row, input, aligned_bytes);
            break;
        case FILTER_SUB:
        case FILTER_PAETH:
            rowDefilter1(input, pixel_bytes, row_bytes, recon_row);
            break;
        case FILTER_AVERAGE:
            row0Defilter3(input, pixel_bytes, row_bytes, recon_row);
            break;
    }
    input += row_bytes;
    recon_row += stride;

    // process the most rows.
    for (uint32_t row = 1; row < height_ - 1; ++row) {
        prior_row = recon_row - stride;
        filter_type = *input++;
        if (filter_type > 4) {
            LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                       << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                       << "Average(3)/Paeth(4)";
            return false;
        }
        switch (filter_type) {
            case FILTER_NONE:
                memcpy(recon_row, input, aligned_bytes);
                break;
            case FILTER_SUB:
                if (pixel_bytes == 2) {
                    rowDefilter1C2(input, row_bytes, recon_row);
                }
                else {
                    rowDefilter1(input, pixel_bytes, row_bytes, recon_row);
                }
                break;
            case FILTER_UP:
                rowDefilter2(input, prior_row, row_bytes, recon_row,
                             row < height_ - 15 ? true : false);
                break;
            case FILTER_AVERAGE:
                if (pixel_bytes == 2) {
                    rowDefilter3C2(input, prior_row, row_bytes, recon_row);
                }
                else {
                    rowDefilter3(input, prior_row, pixel_bytes, row_bytes,
                                 recon_row);
                }
                break;
            case FILTER_PAETH:
                if (pixel_bytes == 2) {
                    rowDefilter4C2(input, prior_row, row_bytes, recon_row);
                }
                else {
                    rowDefilter4(input, prior_row, pixel_bytes, row_bytes,
                                 recon_row);
                }
                break;
        }
        input += row_bytes;
        recon_row += stride;
    }

    // process the last row.
    prior_row = recon_row - stride;
    filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                   << "Average(3)/Paeth(4)";
        return false;
    }
    switch (filter_type) {
        case FILTER_NONE:
            memcpy(recon_row, input, row_bytes);
            break;
        case FILTER_SUB:
            if (pixel_bytes == 2) {
                rowDefilter1C2(input, row_bytes, recon_row);
            }
            else {
                rowDefilter1(input, pixel_bytes, row_bytes, recon_row);
            }
            break;
        case FILTER_UP:
            rowDefilter2(input, prior_row, row_bytes, recon_row, false);
            break;
        case FILTER_AVERAGE:
            if (pixel_bytes == 2) {
                rowDefilter3C2(input, prior_row, row_bytes, recon_row);
            }
            else {
                rowDefilter3(input, prior_row, pixel_bytes, row_bytes,
                             recon_row);
            }
            break;
        case FILTER_PAETH:
            if (pixel_bytes == 2) {
                rowDefilter4C2(input, prior_row, row_bytes, recon_row);
            }
            else {
                rowDefilter4(input, prior_row, pixel_bytes, row_bytes,
                             recon_row);
            }
            break;
    }

    uint32_t stride_offset1 = 0;
    if ((color_type_ == 0 && encoded_channels_ + 1 == channels_) ||
        color_type_ == 3) {
        stride_offset1 = stride - width_ * pixel_bytes;
    }
    if (bit_depth_ < 8) {  // unpack 1/2/4-bit into a 8-bit.
        int index;
        uint8_t scale = (color_type_ == 0) ? depth_scales[bit_depth_] : 1;
        uint8_t value;
        uint8_t* src = image + stride_offset0;
        uint8_t* dst = image + stride_offset1;
        for (uint32_t row = 0; row < height_; ++row) {
            input = src;
            recon_row = dst;

            if (bit_depth_ == 4) {
                for (index = width_; index >= 2; index -= 2, ++input) {
                    value = *input;
                    recon_row[0] = scale * ((value >> 4));
                    recon_row[1] = scale * ((value     ) & 0x0f);
                    recon_row += 2;
                }
                if (index > 0) *recon_row++ = scale * ((*input >> 4));
            } else if (bit_depth_ == 2) {
                for (index = width_; index >= 4; index -= 4, ++input) {
                    value = *input;
                    recon_row[0] = scale * ((value >> 6));
                    recon_row[1] = scale * ((value >> 4) & 0x03);
                    recon_row[2] = scale * ((value >> 2) & 0x03);
                    recon_row[3] = scale * ((value     ) & 0x03);
                    recon_row += 4;
                }
                value = *input;
                if (index > 0) recon_row[0] = scale * ((value >> 6));
                if (index > 1) recon_row[1] = scale * ((value >> 4) & 0x03);
                if (index > 2) recon_row[2] = scale * ((value >> 2) & 0x03);
            } else if (bit_depth_ == 1) {
                for (index = width_; index >= 8; index -= 8, ++input) {
                    value = *input;
                    recon_row[0] = scale * ((value >> 7));
                    recon_row[1] = scale * ((value >> 6) & 0x01);
                    recon_row[2] = scale * ((value >> 5) & 0x01);
                    recon_row[3] = scale * ((value >> 4) & 0x01);
                    recon_row[4] = scale * ((value >> 3) & 0x01);
                    recon_row[5] = scale * ((value >> 2) & 0x01);
                    recon_row[6] = scale * ((value >> 1) & 0x01);
                    recon_row[7] = scale * ((value     ) & 0x01);
                    recon_row += 8;
                }
                value = *input;
                if (index > 0) recon_row[0] = scale * ((value >> 7));
                if (index > 1) recon_row[1] = scale * ((value >> 6) & 0x01);
                if (index > 2) recon_row[2] = scale * ((value >> 5) & 0x01);
                if (index > 3) recon_row[3] = scale * ((value >> 4) & 0x01);
                if (index > 4) recon_row[4] = scale * ((value >> 3) & 0x01);
                if (index > 5) recon_row[5] = scale * ((value >> 2) & 0x01);
                if (index > 6) recon_row[6] = scale * ((value >> 1) & 0x01);
            }
            src += stride;
            dst += stride;
        }
    } else if (bit_depth_ == 16) {
        // force the image data from big-endian to platform-native.
        uint8_t value0, value1;
        uint8_t *recon_row = image + stride_offset0;
        uint16_t *current_row16;
        for (uint32_t row = 0; row < height_; row++) {
            current_row16 = (uint16_t*)recon_row;
            for (uint32_t col = 0, col1 = 0; col < width_; col++, col1 += 2) {
                value0 = recon_row[col1];
                value1 = recon_row[col1 + 1];
                current_row16[col] = (value0 << 8) | value1;
            }
            recon_row += stride;
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

static __m128i rgb2bgrc3_index = _mm_set_epi8(15, 12, 13, 14, 9, 10, 11,
                                              6, 7, 8, 3, 4, 5, 0, 1, 2);
static __m128i rgb2bgrc4_index = _mm_set_epi8(15, 12, 13, 14, 11, 8, 9, 10,
                                              7, 4, 5, 6, 3, 0, 1, 2);
static __m128i trunc_index = _mm_set_epi8(0, 255, 0, 255, 0, 255, 0, 255,
                                          0, 255, 0, 255, 0, 255, 0, 255);
static __m128i pick1_index = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 255, 255, 255, 0, 0, 0);
static __m128i pick2_index = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 255,
                                          255, 255, 0, 0, 0, 0, 0, 0);
static __m128i pick3_index = _mm_set_epi8(0, 0, 0, 0, 255, 255, 255, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0);
static __m128i pick4_index = _mm_set_epi8(0, 255, 255, 255, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0);

inline
void rowDefilter0ColorC3(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row, bool is_last_row) {
    __m128i input, output;
    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        input  = _mm_loadu_si128((__m128i const*)input_row);
        output = _mm_shuffle_epi8(input, rgb2bgrc3_index);
        _mm_storeu_si128((__m128i*)recon_row, output);
        recon_row += 15;
        input_row += 15;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            input  = _mm_loadu_si128((__m128i const*)input_row);
            output = _mm_shuffle_epi8(input, rgb2bgrc3_index);
            _mm_storeu_si128((__m128i*)recon_row, output);

            return;
        }
        else {
            uint32_t value0, value1, value2;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                recon_row[0] = value2;
                recon_row[1] = value1;
                recon_row[2] = value0;
                recon_row += 3;
                input_row += 3;
            }
        }
    }
}

inline
void rowDefilter0ColorC4(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row, bool is_last_row) {
    __m128i input, output;
    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        input  = _mm_loadu_si128((__m128i const*)input_row);
        output = _mm_shuffle_epi8(input, rgb2bgrc4_index);
        _mm_store_si128((__m128i*)recon_row, output);
        recon_row += 16;
        input_row += 16;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            input  = _mm_loadu_si128((__m128i const*)input_row);
            output = _mm_shuffle_epi8(input, rgb2bgrc4_index);
            _mm_store_si128((__m128i*)recon_row, output);

            return;
        }
        else {
            uint32_t value0, value1, value2, value3;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                value3 = input_row[3];
                recon_row[0] = value2;
                recon_row[1] = value1;
                recon_row[2] = value0;
                recon_row[3] = value3;
                recon_row += 4;
                input_row += 4;
            }
        }
    }
}

inline
void rowDefilter0ColorC6(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row) {
    uint8_t* input_end = input_row + row_bytes;
    uint32_t value0, value1, value2, value3, value4, value5;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        recon_row[0] = value4;
        recon_row[1] = value5;
        recon_row[2] = value2;
        recon_row[3] = value3;
        recon_row[4] = value0;
        recon_row[5] = value1;
        recon_row += 6;
        input_row += 6;
    }
}

inline
void rowDefilter0ColorC8(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row) {
    uint8_t* input_end = input_row + row_bytes;
    uint32_t value0, value1, value2, value3, value4, value5, value6, value7;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        value6 = input_row[6];
        value7 = input_row[7];
        recon_row[0] = value4;
        recon_row[1] = value5;
        recon_row[2] = value2;
        recon_row[3] = value3;
        recon_row[4] = value0;
        recon_row[5] = value1;
        recon_row[6] = value6;
        recon_row[7] = value7;
        recon_row += 8;
        input_row += 8;
    }
}

inline
void rowDefilter1ColorC3(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row, bool is_last_row) {
    __m128i input, bgr, result, output, value;
    __m128i index0 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8,
                                  7, 6, 2, 1, 0, 2, 1, 0);
    __m128i index1 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9,
                                  5, 4, 3, 5, 4, 3, 2, 1, 0);
    __m128i index2 = _mm_set_epi8(15, 14, 13, 12, 8, 7, 6, 8,
                                  7, 6, 5, 4, 3, 2, 1, 0);
    __m128i index3 = _mm_set_epi8(15, 11, 10, 9, 11, 10, 9, 8,
                                  7, 6, 5, 4, 3, 2, 1, 0);
    __m128i index4 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8,
                                  7, 6, 5, 4, 3, 14, 13, 12);
    output = _mm_set1_epi8(0);

    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        input  = _mm_loadu_si128((__m128i const*)input_row);
        bgr    = _mm_shuffle_epi8(input, rgb2bgrc3_index);
        result = _mm_add_epi8(bgr, output);
        output = result;

        value  = _mm_shuffle_epi8(result, index0);
        result = _mm_add_epi8(bgr, value);
        output = _mm_blendv_epi8(output, result, pick1_index);

        value  = _mm_shuffle_epi8(result, index1);
        result = _mm_add_epi8(bgr, value);
        output = _mm_blendv_epi8(output, result, pick2_index);

        value  = _mm_shuffle_epi8(result, index2);
        result = _mm_add_epi8(bgr, value);
        output = _mm_blendv_epi8(output, result, pick3_index);

        value  = _mm_shuffle_epi8(result, index3);
        result = _mm_add_epi8(bgr, value);
        output = _mm_blendv_epi8(output, result, pick4_index);

        _mm_storeu_si128((__m128i*)recon_row, output);
        output = _mm_shuffle_epi8(result, index4);
        recon_row += 15;
        input_row += 15;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            input  = _mm_loadu_si128((__m128i const*)input_row);
            bgr    = _mm_shuffle_epi8(input, rgb2bgrc3_index);
            result = _mm_add_epi8(bgr, output);
            output = result;

            value  = _mm_shuffle_epi8(result, index0);
            result = _mm_add_epi8(bgr, value);
            output = _mm_blendv_epi8(output, result, pick1_index);

            value  = _mm_shuffle_epi8(result, index1);
            result = _mm_add_epi8(bgr, value);
            output = _mm_blendv_epi8(output, result, pick2_index);

            value  = _mm_shuffle_epi8(result, index2);
            result = _mm_add_epi8(bgr, value);
            output = _mm_blendv_epi8(output, result, pick3_index);

            value  = _mm_shuffle_epi8(result, index3);
            result = _mm_add_epi8(bgr, value);
            output = _mm_blendv_epi8(output, result, pick4_index);

            _mm_storeu_si128((__m128i*)recon_row, output);
            output = _mm_shuffle_epi8(result, index4);
        }
        else {
            uint32_t value0, value1, value2;
            uint8_t* before_bytes = recon_row - 3;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                recon_row[0] = value2 + before_bytes[0];
                recon_row[1] = value1 + before_bytes[1];
                recon_row[2] = value0 + before_bytes[2];
                recon_row    += 3;
                input_row    += 3;
                before_bytes += 3;
            }
        }
    }
}

inline
void rowDefilter1ColorC4(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row, bool is_last_row) {
    __m128i input, bgr, output, a4, b4, c4, d4, value;
    __m128i index0 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8,
                                  3, 2, 1, 0, 3, 2, 1, 0);
    __m128i index1 = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4,
                                  7, 6, 5, 4, 3, 2, 1, 0);
    __m128i index2 = _mm_set_epi8(11, 10, 9, 8, 11, 10, 9, 8,
                                  7, 6, 5, 4, 3, 2, 1, 0);
    __m128i index3 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8,
                                  7, 6, 5, 4, 15, 14, 13, 12);
    output = _mm_set1_epi8(0);

    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        input  = _mm_loadu_si128((__m128i const*)input_row);
        bgr    = _mm_shuffle_epi8(input, rgb2bgrc4_index);
        a4     = _mm_add_epi8(bgr, output);
        output = a4;

        value  = _mm_shuffle_epi8(a4, index0);
        b4     = _mm_add_epi8(bgr, value);
        output = _mm_blend_epi16(output, b4, 12);

        value  = _mm_shuffle_epi8(b4, index1);
        c4     = _mm_add_epi8(bgr, value);
        output = _mm_blend_epi16(output, c4, 48);

        value  = _mm_shuffle_epi8(c4, index2);
        d4     = _mm_add_epi8(bgr, value);
        output = _mm_blend_epi16(output, d4, 192);

        _mm_store_si128((__m128i*)recon_row, output);
        output = _mm_shuffle_epi8(d4, index3);
        recon_row += 16;
        input_row += 16;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            input  = _mm_loadu_si128((__m128i const*)input_row);
            bgr    = _mm_shuffle_epi8(input, rgb2bgrc4_index);
            a4     = _mm_add_epi8(bgr, output);
            output = a4;

            value  = _mm_shuffle_epi8(a4, index0);
            b4     = _mm_add_epi8(bgr, value);
            output = _mm_blend_epi16(output, b4, 12);

            value  = _mm_shuffle_epi8(b4, index1);
            c4     = _mm_add_epi8(bgr, value);
            output = _mm_blend_epi16(output, c4, 48);

            value  = _mm_shuffle_epi8(c4, index2);
            d4     = _mm_add_epi8(bgr, value);
            output = _mm_blend_epi16(output, d4, 192);

            _mm_store_si128((__m128i*)recon_row, output);

            return;
        }
        else {
            uint32_t value0, value1, value2, value3;
            uint8_t* before_bytes = recon_row - 4;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                value3 = input_row[3];
                recon_row[0] = value2 + before_bytes[0];
                recon_row[1] = value1 + before_bytes[1];
                recon_row[2] = value0 + before_bytes[2];
                recon_row[3] = value3 + before_bytes[3];
                recon_row    += 4;
                input_row    += 4;
                before_bytes += 4;
            }
        }
    }
}

inline
void rowDefilter1ColorC6(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    recon_row[0] = value4;
    recon_row[1] = value5;
    recon_row[2] = value2;
    recon_row[3] = value3;
    recon_row[4] = value0;
    recon_row[5] = value1;

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 6;
    recon_row += 6;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        recon_row[0] = value4 + before_bytes[0];
        recon_row[1] = value5 + before_bytes[1];
        recon_row[2] = value2 + before_bytes[2];
        recon_row[3] = value3 + before_bytes[3];
        recon_row[4] = value0 + before_bytes[4];
        recon_row[5] = value1 + before_bytes[5];
        recon_row    += 6;
        input_row    += 6;
        before_bytes += 6;
    }
}

inline
void rowDefilter1ColorC8(uint8_t* input_row, uint32_t row_bytes,
                         uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5, value6, value7;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    value6 = input_row[6];
    value7 = input_row[7];
    recon_row[0] = value4;
    recon_row[1] = value5;
    recon_row[2] = value2;
    recon_row[3] = value3;
    recon_row[4] = value0;
    recon_row[5] = value1;
    recon_row[6] = value6;
    recon_row[7] = value7;

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 8;
    recon_row += 8;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        value6 = input_row[6];
        value7 = input_row[7];
        recon_row[0] = value4 + before_bytes[0];
        recon_row[1] = value5 + before_bytes[1];
        recon_row[2] = value2 + before_bytes[2];
        recon_row[3] = value3 + before_bytes[3];
        recon_row[4] = value0 + before_bytes[4];
        recon_row[5] = value1 + before_bytes[5];
        recon_row[6] = value6 + before_bytes[6];
        recon_row[7] = value7 + before_bytes[7];
        recon_row    += 8;
        input_row    += 8;
        before_bytes += 8;
    }
}

inline
void rowDefilter2ColorC3(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row,
                         bool is_last_row) {
    __m128i input0, input1, bgr, output;
    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        input0  = _mm_loadu_si128((__m128i const*)input_row);
        input1  = _mm_loadu_si128((__m128i const*)prior_row);
        bgr     = _mm_shuffle_epi8(input0, rgb2bgrc3_index);
        output  = _mm_add_epi8(bgr, input1);
        _mm_storeu_si128((__m128i*)recon_row, output);
        recon_row += 15;
        input_row += 15;
        prior_row += 15;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            input0  = _mm_loadu_si128((__m128i const*)input_row);
            input1  = _mm_loadu_si128((__m128i const*)prior_row);
            bgr     = _mm_shuffle_epi8(input0, rgb2bgrc3_index);
            output  = _mm_add_epi8(bgr, input1);
            _mm_storeu_si128((__m128i*)recon_row, output);
        }
        else {
            uint32_t value0, value1, value2;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                recon_row[0] = value2 + prior_row[0];
                recon_row[1] = value1 + prior_row[1];
                recon_row[2] = value0 + prior_row[2];
                recon_row += 3;
                input_row += 3;
                prior_row += 3;
            }
        }
    }
}

inline
void rowDefilter2ColorC4(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row,
                         bool is_last_row) {
    __m128i input0, input1, bgr, output;
    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        input0  = _mm_loadu_si128((__m128i const*)input_row);
        input1  = _mm_loadu_si128((__m128i const*)prior_row);
        bgr     = _mm_shuffle_epi8(input0, rgb2bgrc4_index);
        output  = _mm_add_epi8(bgr, input1);
        _mm_store_si128((__m128i*)recon_row, output);
        recon_row += 16;
        input_row += 16;
        prior_row += 16;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            input0  = _mm_loadu_si128((__m128i const*)input_row);
            input1  = _mm_loadu_si128((__m128i const*)prior_row);
            bgr     = _mm_shuffle_epi8(input0, rgb2bgrc4_index);
            output  = _mm_add_epi8(bgr, input1);
            _mm_store_si128((__m128i*)recon_row, output);

            return;
        }
        else {
            uint32_t value0, value1, value2, value3;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                value3 = input_row[3];
                recon_row[0] = value2 + prior_row[0];
                recon_row[1] = value1 + prior_row[1];
                recon_row[2] = value0 + prior_row[2];
                recon_row[3] = value3 + prior_row[3];
                recon_row += 4;
                input_row += 4;
                prior_row += 4;
            }
        }
    }
}

inline
void rowDefilter2ColorC6(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row) {
    uint8_t* input_end = input_row + row_bytes;
    uint32_t value0, value1, value2, value3, value4, value5;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        recon_row[0] = value4 + prior_row[0];
        recon_row[1] = value5 + prior_row[1];
        recon_row[2] = value2 + prior_row[2];
        recon_row[3] = value3 + prior_row[3];
        recon_row[4] = value0 + prior_row[4];
        recon_row[5] = value1 + prior_row[5];
        recon_row += 6;
        input_row += 6;
        prior_row += 6;
    }
}

inline
void rowDefilter2ColorC8(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row) {
    uint8_t* input_end = input_row + row_bytes;
    uint32_t value0, value1, value2, value3, value4, value5, value6, value7;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        value6 = input_row[6];
        value7 = input_row[7];
        recon_row[0] = value4 + prior_row[0];
        recon_row[1] = value5 + prior_row[1];
        recon_row[2] = value2 + prior_row[2];
        recon_row[3] = value3 + prior_row[3];
        recon_row[4] = value0 + prior_row[4];
        recon_row[5] = value1 + prior_row[5];
        recon_row[6] = value6 + prior_row[6];
        recon_row[7] = value7 + prior_row[7];
        recon_row += 8;
        input_row += 8;
        prior_row += 8;
    }
}

inline
void row0Defilter3ColorC3(uint8_t* input_row, uint32_t row_bytes,
                          uint8_t* recon_row) {
    uint32_t value0, value1, value2;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    recon_row[0] = value2;
    recon_row[1] = value1;
    recon_row[2] = value0;

    uint8_t* input_end    = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 3;
    recon_row += 3;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        recon_row[0] = value2 + (before_bytes[0] >> 1);
        recon_row[1] = value1 + (before_bytes[1] >> 1);
        recon_row[2] = value0 + (before_bytes[2] >> 1);
        recon_row    += 3;
        input_row    += 3;
        before_bytes += 3;
    }
}

inline
void row0Defilter3ColorC4(uint8_t* input_row, uint32_t row_bytes,
                          uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    recon_row[0] = value2;
    recon_row[1] = value1;
    recon_row[2] = value0;
    recon_row[3] = value3;

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 4;
    recon_row += 4;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        recon_row[0] = value2 + (before_bytes[0] >> 1);
        recon_row[1] = value1 + (before_bytes[1] >> 1);
        recon_row[2] = value0 + (before_bytes[2] >> 1);
        recon_row[3] = value3 + (before_bytes[3] >> 1);
        recon_row    += 4;
        input_row    += 4;
        before_bytes += 4;
    }
}

inline
void row0Defilter3ColorC6(uint8_t* input_row, uint32_t row_bytes,
                          uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    recon_row[0] = value4;
    recon_row[1] = value5;
    recon_row[2] = value2;
    recon_row[3] = value3;
    recon_row[4] = value0;
    recon_row[5] = value1;

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 6;
    recon_row += 6;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        recon_row[0] = value4 + (before_bytes[0] >> 1);
        recon_row[1] = value5 + (before_bytes[1] >> 1);
        recon_row[2] = value2 + (before_bytes[2] >> 1);
        recon_row[3] = value3 + (before_bytes[3] >> 1);
        recon_row[4] = value0 + (before_bytes[4] >> 1);
        recon_row[5] = value1 + (before_bytes[5] >> 1);
        recon_row    += 6;
        input_row    += 6;
        before_bytes += 6;
    }
}

inline
void row0Defilter3ColorC8(uint8_t* input_row, uint32_t row_bytes,
                          uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5, value6, value7;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    value6 = input_row[6];
    value7 = input_row[7];
    recon_row[0] = value4;
    recon_row[1] = value5;
    recon_row[2] = value2;
    recon_row[3] = value3;
    recon_row[4] = value0;
    recon_row[5] = value1;
    recon_row[6] = value6;
    recon_row[7] = value7;

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 8;
    recon_row += 8;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        value6 = input_row[6];
        value7 = input_row[7];
        recon_row[0] = value4 + (before_bytes[0] >> 1);
        recon_row[1] = value5 + (before_bytes[1] >> 1);
        recon_row[2] = value2 + (before_bytes[2] >> 1);
        recon_row[3] = value3 + (before_bytes[3] >> 1);
        recon_row[4] = value0 + (before_bytes[4] >> 1);
        recon_row[5] = value1 + (before_bytes[5] >> 1);
        recon_row[6] = value6 + (before_bytes[6] >> 1);
        recon_row[7] = value7 + (before_bytes[7] >> 1);
        recon_row    += 8;
        input_row    += 8;
        before_bytes += 8;
    }
}

inline
__m128i caculateDefilter3(__m128i a4_i16, __m128i b4_i16, __m128i input) {
    __m128i add_i16    = _mm_add_epi16(a4_i16, b4_i16);
    __m128i shift_i16  = _mm_srli_epi16(add_i16, 1);
    __m128i result_i16 = _mm_add_epi16(input, shift_i16);
    __m128i result     = _mm_and_si128(result_i16, trunc_index);

    return result;
}

// de-filter a 3-byte pixel
inline
void rowDefilter3ColorC3(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row,
                         bool is_last_row) {
    __m128i prior_pixels, current_pixels, bgr_pixels, prior_i16, current_i16;
    __m128i a4_i16, b4_i16, input, result_i16, result_i8, output0, output1;
    a4_i16 = _mm_set1_epi16(0);

    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
        current_pixels = _mm_loadu_si128((__m128i const*)input_row);
        bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc3_index);
        prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
        current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

        // the first 3 8-bit output
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        result_i8  = _mm_packus_epi16(result_i16, result_i16);

        // the second 3 8-bit output
        a4_i16     = result_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 6);
        input      = _mm_bsrli_si128(current_i16, 6);
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 3);
        result_i8  = _mm_blendv_epi8(result_i8, output1, pick1_index);

        output0     = _mm_bsrli_si128(prior_pixels, 6);
        output1     = _mm_bsrli_si128(bgr_pixels, 6);
        prior_i16   = _mm_cvtepu8_epi16(output0);
        current_i16 = _mm_cvtepu8_epi16(output1);

        // the third 3 8-bit output
        a4_i16     = result_i16;
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 6);
        result_i8  = _mm_blendv_epi8(result_i8, output1, pick2_index);

        // the fourth 3 8-bit output
        a4_i16     = result_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 6);
        input      = _mm_bsrli_si128(current_i16, 6);
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 9);
        result_i8  = _mm_blendv_epi8(result_i8, output1, pick3_index);

        // the fifth 3 8-bit output
        a4_i16      = result_i16;
        output0     = _mm_bsrli_si128(prior_pixels, 12);
        output1     = _mm_bsrli_si128(bgr_pixels, 12);
        prior_i16   = _mm_cvtepu8_epi16(output0);
        current_i16 = _mm_cvtepu8_epi16(output1);
        b4_i16      = prior_i16;
        input       = current_i16;
        result_i16  = caculateDefilter3(a4_i16, b4_i16, input);
        output0     = _mm_packus_epi16(result_i16, result_i16);
        output1     = _mm_bslli_si128(output0, 12);
        result_i8   = _mm_blendv_epi8(result_i8, output1, pick4_index);

        _mm_storeu_si128((__m128i*)recon_row, result_i8);
        a4_i16 = result_i16;
        recon_row += 15;
        input_row += 15;
        prior_row += 15;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
            current_pixels = _mm_loadu_si128((__m128i const*)input_row);
            bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc3_index);
            prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
            current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

            // the first 3 8-bit output
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            result_i8  = _mm_packus_epi16(result_i16, result_i16);

            // the second 3 8-bit output
            a4_i16     = result_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 6);
            input      = _mm_bsrli_si128(current_i16, 6);
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 3);
            result_i8  = _mm_blendv_epi8(result_i8, output1, pick1_index);

            output0     = _mm_bsrli_si128(prior_pixels, 6);
            output1     = _mm_bsrli_si128(bgr_pixels, 6);
            prior_i16   = _mm_cvtepu8_epi16(output0);
            current_i16 = _mm_cvtepu8_epi16(output1);

            // the third 3 8-bit output
            a4_i16     = result_i16;
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 6);
            result_i8  = _mm_blendv_epi8(result_i8, output1, pick2_index);

            // the fourth 3 8-bit output
            a4_i16     = result_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 6);
            input      = _mm_bsrli_si128(current_i16, 6);
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 9);
            result_i8  = _mm_blendv_epi8(result_i8, output1, pick3_index);

            // the fifth 3 8-bit output
            a4_i16      = result_i16;
            output0     = _mm_bsrli_si128(prior_pixels, 12);
            output1     = _mm_bsrli_si128(bgr_pixels, 12);
            prior_i16   = _mm_cvtepu8_epi16(output0);
            current_i16 = _mm_cvtepu8_epi16(output1);
            b4_i16      = prior_i16;
            input       = current_i16;
            result_i16  = caculateDefilter3(a4_i16, b4_i16, input);
            output0     = _mm_packus_epi16(result_i16, result_i16);
            output1     = _mm_bslli_si128(output0, 12);
            result_i8   = _mm_blendv_epi8(result_i8, output1, pick4_index);

            _mm_storeu_si128((__m128i*)recon_row, result_i8);
        }
        else {
            uint32_t value0, value1, value2;
            uint8_t* before_bytes = recon_row - 3;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                recon_row[0] = value2 + ((before_bytes[0] + prior_row[0]) >> 1);
                recon_row[1] = value1 + ((before_bytes[1] + prior_row[1]) >> 1);
                recon_row[2] = value0 + ((before_bytes[2] + prior_row[2]) >> 1);
                recon_row    += 3;
                input_row    += 3;
                before_bytes += 3;
                prior_row    += 3;
            }
        }
    }
}

// de-filter a 4-byte pixel
inline
void rowDefilter3ColorC4(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row,
                         bool is_last_row) {
    __m128i prior_pixels, current_pixels, bgr_pixels, prior_i16, current_i16;
    __m128i a4_i16, b4_i16, input, result_i16, result_i8, output0, output1;
    a4_i16 = _mm_set1_epi16(0);

    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
        current_pixels = _mm_loadu_si128((__m128i const*)input_row);
        bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc4_index);
        prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
        current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

        // first 4 8-bit output
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        result_i8  = _mm_packus_epi16(result_i16, result_i16);

        // second 4 8-bit output
        a4_i16     = result_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 8);
        input      = _mm_bsrli_si128(current_i16, 8);
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 4);
        result_i8  = _mm_blend_epi16(result_i8, output1, 12);

        output0     = _mm_bsrli_si128(prior_pixels, 8);
        output1     = _mm_bsrli_si128(bgr_pixels, 8);
        prior_i16   = _mm_cvtepu8_epi16(output0);
        current_i16 = _mm_cvtepu8_epi16(output1);

        // third 4 8-bit output
        a4_i16     = result_i16;
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 8);
        result_i8  = _mm_blend_epi16(result_i8, output1, 48);

        // fourth 4 8-bit output
        a4_i16     = result_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 8);
        input      = _mm_bsrli_si128(current_i16, 8);
        result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 12);
        result_i8  = _mm_blend_epi16(result_i8, output1, 192);

        _mm_store_si128((__m128i*)recon_row, result_i8);
        a4_i16 = result_i16;
        recon_row += 16;
        input_row += 16;
        prior_row += 16;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
            current_pixels = _mm_loadu_si128((__m128i const*)input_row);
            bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc4_index);
            prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
            current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

            // first 4 8-bit output
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            result_i8  = _mm_packus_epi16(result_i16, result_i16);

            // second 4 8-bit output
            a4_i16     = result_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 8);
            input      = _mm_bsrli_si128(current_i16, 8);
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 4);
            result_i8  = _mm_blend_epi16(result_i8, output1, 12);

            output0     = _mm_bsrli_si128(prior_pixels, 8);
            output1     = _mm_bsrli_si128(bgr_pixels, 8);
            prior_i16   = _mm_cvtepu8_epi16(output0);
            current_i16 = _mm_cvtepu8_epi16(output1);

            // third 4 8-bit output
            a4_i16     = result_i16;
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 8);
            result_i8  = _mm_blend_epi16(result_i8, output1, 48);

            // fourth 4 8-bit output
            a4_i16     = result_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 8);
            input      = _mm_bsrli_si128(current_i16, 8);
            result_i16 = caculateDefilter3(a4_i16, b4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 12);
            result_i8  = _mm_blend_epi16(result_i8, output1, 192);

            _mm_store_si128((__m128i*)recon_row, result_i8);

            return;
        }
        else {
            uint32_t value0, value1, value2, value3;
            uint8_t* before_bytes = recon_row - 4;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                value3 = input_row[3];
                recon_row[0] = value2 + ((before_bytes[0] + prior_row[0]) >> 1);
                recon_row[1] = value1 + ((before_bytes[1] + prior_row[1]) >> 1);
                recon_row[2] = value0 + ((before_bytes[2] + prior_row[2]) >> 1);
                recon_row[3] = value3 + ((before_bytes[3] + prior_row[3]) >> 1);
                recon_row    += 4;
                input_row    += 4;
                before_bytes += 4;
                prior_row    += 4;
            }
        }
    }
}

inline
void rowDefilter3ColorC6(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    recon_row[0] = value4 + (prior_row[0] >> 1);
    recon_row[1] = value5 + (prior_row[1] >> 1);
    recon_row[2] = value2 + (prior_row[2] >> 1);
    recon_row[3] = value3 + (prior_row[3] >> 1);
    recon_row[4] = value0 + (prior_row[4] >> 1);
    recon_row[5] = value1 + (prior_row[5] >> 1);

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 6;
    prior_row += 6;
    recon_row += 6;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        recon_row[0] = value4 + ((before_bytes[0] + prior_row[0]) >> 1);
        recon_row[1] = value5 + ((before_bytes[1] + prior_row[1]) >> 1);
        recon_row[2] = value2 + ((before_bytes[2] + prior_row[2]) >> 1);
        recon_row[3] = value3 + ((before_bytes[3] + prior_row[3]) >> 1);
        recon_row[4] = value0 + ((before_bytes[4] + prior_row[4]) >> 1);
        recon_row[5] = value1 + ((before_bytes[5] + prior_row[5]) >> 1);
        recon_row    += 6;
        input_row    += 6;
        before_bytes += 6;
        prior_row    += 6;
    }
}

inline
void rowDefilter3ColorC8(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5, value6, value7;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    value6 = input_row[6];
    value7 = input_row[7];
    recon_row[0] = value4 + (prior_row[0] >> 1);
    recon_row[1] = value5 + (prior_row[1] >> 1);
    recon_row[2] = value2 + (prior_row[2] >> 1);
    recon_row[3] = value3 + (prior_row[3] >> 1);
    recon_row[4] = value0 + (prior_row[4] >> 1);
    recon_row[5] = value1 + (prior_row[5] >> 1);
    recon_row[6] = value6 + (prior_row[6] >> 1);
    recon_row[7] = value7 + (prior_row[7] >> 1);

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    input_row += 8;
    prior_row += 8;
    recon_row += 8;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        value6 = input_row[6];
        value7 = input_row[7];
        recon_row[0] = value4 + ((before_bytes[0] + prior_row[0]) >> 1);
        recon_row[1] = value5 + ((before_bytes[1] + prior_row[1]) >> 1);
        recon_row[2] = value2 + ((before_bytes[2] + prior_row[2]) >> 1);
        recon_row[3] = value3 + ((before_bytes[3] + prior_row[3]) >> 1);
        recon_row[4] = value0 + ((before_bytes[4] + prior_row[4]) >> 1);
        recon_row[5] = value1 + ((before_bytes[5] + prior_row[5]) >> 1);
        recon_row[6] = value6 + ((before_bytes[6] + prior_row[6]) >> 1);
        recon_row[7] = value7 + ((before_bytes[7] + prior_row[7]) >> 1);
        recon_row    += 8;
        input_row    += 8;
        before_bytes += 8;
        prior_row    += 8;
    }
}

inline
__m128i caculatePaeth(__m128i a4_i16, __m128i b4_i16, __m128i c4_i16,
                      __m128i input) {
    __m128i pab = _mm_add_epi16(a4_i16, b4_i16);
    __m128i pabc = _mm_sub_epi16(pab, c4_i16);
    __m128i pa = _mm_sub_epi16(pabc, a4_i16);
    __m128i pb = _mm_sub_epi16(pabc, b4_i16);
    __m128i pc = _mm_sub_epi16(pabc, c4_i16);
    __m128i pa_abs = _mm_abs_epi16(pa);
    __m128i pb_abs = _mm_abs_epi16(pb);
    __m128i pc_abs = _mm_abs_epi16(pc);
    __m128i min_ab = _mm_min_epi16(pa_abs, pb_abs);
    __m128i target_ab = _mm_blendv_epi8(b4_i16, a4_i16,
                                        _mm_cmpeq_epi16(min_ab, pa_abs));
    __m128i min_abc = _mm_min_epi16(min_ab, pc_abs);
    __m128i target_abc = _mm_blendv_epi8(c4_i16, target_ab,
                                         _mm_cmpeq_epi16(min_ab, min_abc));
    __m128i value = _mm_add_epi16(input, target_abc);
    __m128i result = _mm_and_si128(value, trunc_index);

    return result;
}

// de-filter a 3-byte pixel
inline
void rowDefilter4ColorC3(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row,
                         bool is_last_row) {
    __m128i prior_pixels, current_pixels, bgr_pixels, prior_i16, current_i16;
    __m128i a4_i16, b4_i16, c4_i16, input, result_i16, result_i8;
    __m128i output0, output1;
    a4_i16 = _mm_set1_epi16(0);
    c4_i16 = _mm_set1_epi16(0);

    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
        current_pixels = _mm_loadu_si128((__m128i const*)input_row);
        bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc3_index);
        prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
        current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

        // the first 3 8-bit output
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        result_i8  = _mm_packus_epi16(result_i16, result_i16);

        // the second 3 8-bit output
        a4_i16     = result_i16;
        c4_i16     = b4_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 6);
        input      = _mm_bsrli_si128(current_i16, 6);
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 3);
        result_i8  = _mm_blendv_epi8(result_i8, output1, pick1_index);

        output0      = _mm_bsrli_si128(prior_pixels, 6);
        output1      = _mm_bsrli_si128(bgr_pixels, 6);
        prior_i16   = _mm_cvtepu8_epi16(output0);
        current_i16 = _mm_cvtepu8_epi16(output1);

        // the third 3 8-bit output
        a4_i16     = result_i16;
        c4_i16     = b4_i16;
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 6);
        result_i8  = _mm_blendv_epi8(result_i8, output1, pick2_index);

        // the fourth 3 8-bit output
        a4_i16     = result_i16;
        c4_i16     = b4_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 6);
        input      = _mm_bsrli_si128(current_i16, 6);
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 9);
        result_i8  = _mm_blendv_epi8(result_i8, output1, pick3_index);

        // the fifth 3 8-bit output
        a4_i16      = result_i16;
        c4_i16      = b4_i16;
        output0     = _mm_bsrli_si128(prior_pixels, 12);
        output1     = _mm_bsrli_si128(bgr_pixels, 12);
        prior_i16   = _mm_cvtepu8_epi16(output0);
        current_i16 = _mm_cvtepu8_epi16(output1);
        b4_i16      = prior_i16;
        input       = current_i16;
        result_i16  = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        output0     = _mm_packus_epi16(result_i16, result_i16);
        output1     = _mm_bslli_si128(output0, 12);
        result_i8   = _mm_blendv_epi8(result_i8, output1, pick4_index);

        _mm_storeu_si128((__m128i*)recon_row, result_i8);
        a4_i16 = result_i16;
        c4_i16 = b4_i16;
        recon_row += 15;
        input_row += 15;
        prior_row += 15;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
            current_pixels = _mm_loadu_si128((__m128i const*)input_row);
            bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc3_index);
            prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
            current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

            // the first 3 8-bit output
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            result_i8  = _mm_packus_epi16(result_i16, result_i16);

            // the second 3 8-bit output
            a4_i16     = result_i16;
            c4_i16     = b4_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 6);
            input      = _mm_bsrli_si128(current_i16, 6);
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 3);
            result_i8  = _mm_blendv_epi8(result_i8, output1, pick1_index);

            output0      = _mm_bsrli_si128(prior_pixels, 6);
            output1      = _mm_bsrli_si128(bgr_pixels, 6);
            prior_i16   = _mm_cvtepu8_epi16(output0);
            current_i16 = _mm_cvtepu8_epi16(output1);

            // the third 3 8-bit output
            a4_i16     = result_i16;
            c4_i16     = b4_i16;
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 6);
            result_i8  = _mm_blendv_epi8(result_i8, output1, pick2_index);

            // the fourth 3 8-bit output
            a4_i16     = result_i16;
            c4_i16     = b4_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 6);
            input      = _mm_bsrli_si128(current_i16, 6);
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 9);
            result_i8  = _mm_blendv_epi8(result_i8, output1, pick3_index);

            // the fifth 3 8-bit output
            a4_i16      = result_i16;
            c4_i16      = b4_i16;
            output0     = _mm_bsrli_si128(prior_pixels, 12);
            output1     = _mm_bsrli_si128(bgr_pixels, 12);
            prior_i16   = _mm_cvtepu8_epi16(output0);
            current_i16 = _mm_cvtepu8_epi16(output1);
            b4_i16      = prior_i16;
            input       = current_i16;
            result_i16  = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            output0     = _mm_packus_epi16(result_i16, result_i16);
            output1     = _mm_bslli_si128(output0, 12);
            result_i8   = _mm_blendv_epi8(result_i8, output1, pick4_index);

            _mm_storeu_si128((__m128i*)recon_row, result_i8);
        }
        else {
            uint32_t value0, value1, value2;
            uint8_t* before_bytes = recon_row - 3;
            uint8_t* prior_before = prior_row - 3;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                recon_row[0] = value2 + filterPaeth(before_bytes[0],
                               prior_row[0], prior_before[0]);
                recon_row[1] = value1 + filterPaeth(before_bytes[1],
                               prior_row[1], prior_before[1]);
                recon_row[2] = value0 + filterPaeth(before_bytes[2],
                               prior_row[2], prior_before[2]);
                recon_row    += 3;
                input_row    += 3;
                before_bytes += 3;
                prior_row    += 3;
                prior_before += 3;
            }
        }
    }
}

// de-filter a 4-byte pixel
inline
void rowDefilter4ColorC4(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row,
                         bool is_last_row) {
    __m128i prior_pixels, current_pixels, bgr_pixels, prior_i16, current_i16;
    __m128i a4_i16, b4_i16, c4_i16, input, result_i16, result_i8;
    __m128i output0, output1;
    a4_i16 = _mm_set1_epi16(0);
    c4_i16 = _mm_set1_epi16(0);

    uint8_t* input_end = input_row + row_bytes - 16;
    while (input_row < input_end) {
        prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
        current_pixels = _mm_loadu_si128((__m128i const*)input_row);
        bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc4_index);
        prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
        current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

        // the first 4 8-bit output
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        result_i8  = _mm_packus_epi16(result_i16, result_i16);

        // the second 4 8-bit output
        a4_i16     = result_i16;
        c4_i16     = b4_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 8);
        input      = _mm_bsrli_si128(current_i16, 8);
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 4);
        result_i8  = _mm_blend_epi16(result_i8, output1, 12);

        output0      = _mm_bsrli_si128(prior_pixels, 8);
        output1      = _mm_bsrli_si128(bgr_pixels, 8);
        prior_i16   = _mm_cvtepu8_epi16(output0);
        current_i16 = _mm_cvtepu8_epi16(output1);

        // the third 4 8-bit output
        a4_i16     = result_i16;
        c4_i16     = b4_i16;
        b4_i16     = prior_i16;
        input      = current_i16;
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 8);
        result_i8  = _mm_blend_epi16(result_i8, output1, 48);

        // the fourth 4 8-bit output
        a4_i16     = result_i16;
        c4_i16     = b4_i16;
        b4_i16     = _mm_bsrli_si128(prior_i16, 8);
        input      = _mm_bsrli_si128(current_i16, 8);
        result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
        output0    = _mm_packus_epi16(result_i16, result_i16);
        output1    = _mm_bslli_si128(output0, 12);
        result_i8  = _mm_blend_epi16(result_i8, output1, 192);

        _mm_store_si128((__m128i*)recon_row, result_i8);
        a4_i16 = result_i16;
        c4_i16 = b4_i16;
        recon_row += 16;
        input_row += 16;
        prior_row += 16;
    }

    input_end += 16;
    if (input_row < input_end) {
        if ((!is_last_row) || input_row + 16 == input_end) {
            prior_pixels   = _mm_loadu_si128((__m128i const*)prior_row);
            current_pixels = _mm_loadu_si128((__m128i const*)input_row);
            bgr_pixels     = _mm_shuffle_epi8(current_pixels, rgb2bgrc4_index);
            prior_i16      = _mm_cvtepu8_epi16(prior_pixels);
            current_i16    = _mm_cvtepu8_epi16(bgr_pixels);

            // the first 4 8-bit output
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            result_i8  = _mm_packus_epi16(result_i16, result_i16);

            // the second 4 8-bit output
            a4_i16     = result_i16;
            c4_i16     = b4_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 8);
            input      = _mm_bsrli_si128(current_i16, 8);
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 4);
            result_i8  = _mm_blend_epi16(result_i8, output1, 12);

            output0      = _mm_bsrli_si128(prior_pixels, 8);
            output1      = _mm_bsrli_si128(bgr_pixels, 8);
            prior_i16   = _mm_cvtepu8_epi16(output0);
            current_i16 = _mm_cvtepu8_epi16(output1);

            // the third 4 8-bit output
            a4_i16     = result_i16;
            c4_i16     = b4_i16;
            b4_i16     = prior_i16;
            input      = current_i16;
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 8);
            result_i8  = _mm_blend_epi16(result_i8, output1, 48);

            // the fourth 4 8-bit output
            a4_i16     = result_i16;
            c4_i16     = b4_i16;
            b4_i16     = _mm_bsrli_si128(prior_i16, 8);
            input      = _mm_bsrli_si128(current_i16, 8);
            result_i16 = caculatePaeth(a4_i16, b4_i16, c4_i16, input);
            output0    = _mm_packus_epi16(result_i16, result_i16);
            output1    = _mm_bslli_si128(output0, 12);
            result_i8  = _mm_blend_epi16(result_i8, output1, 192);

            _mm_store_si128((__m128i*)recon_row, result_i8);

            return;
        }
        else {
            uint32_t value0, value1, value2, value3;
            uint8_t* before_bytes = recon_row - 4;
            uint8_t* prior_before = prior_row - 4;
            while (input_row < input_end) {
                value0 = input_row[0];
                value1 = input_row[1];
                value2 = input_row[2];
                value3 = input_row[3];
                recon_row[0] = value2 + filterPaeth(before_bytes[0],
                               prior_row[0], prior_before[0]);
                recon_row[1] = value1 + filterPaeth(before_bytes[1],
                               prior_row[1], prior_before[1]);
                recon_row[2] = value0 + filterPaeth(before_bytes[2],
                               prior_row[2], prior_before[2]);
                recon_row[3] = value3 + filterPaeth(before_bytes[3],
                               prior_row[3], prior_before[3]);
                recon_row    += 4;
                input_row    += 4;
                before_bytes += 4;
                prior_row    += 4;
                prior_before += 4;
            }
        }
    }
}

inline
void rowDefilter4ColorC6(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    recon_row[0] = value4 + prior_row[0];
    recon_row[1] = value5 + prior_row[1];
    recon_row[2] = value2 + prior_row[2];
    recon_row[3] = value3 + prior_row[3];
    recon_row[4] = value0 + prior_row[4];
    recon_row[5] = value1 + prior_row[5];

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    uint8_t* prior_before = prior_row;
    input_row += 6;
    prior_row += 6;
    recon_row += 6;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        recon_row[0] = value4 + filterPaeth(before_bytes[0], prior_row[0],
                                            prior_before[0]);
        recon_row[1] = value5 + filterPaeth(before_bytes[1], prior_row[1],
                                            prior_before[1]);
        recon_row[2] = value2 + filterPaeth(before_bytes[2], prior_row[2],
                                            prior_before[2]);
        recon_row[3] = value3 + filterPaeth(before_bytes[3], prior_row[3],
                                            prior_before[3]);
        recon_row[4] = value0 + filterPaeth(before_bytes[4], prior_row[4],
                                            prior_before[4]);
        recon_row[5] = value1 + filterPaeth(before_bytes[5], prior_row[5],
                                            prior_before[5]);
        recon_row    += 6;
        input_row    += 6;
        before_bytes += 6;
        prior_row    += 6;
        prior_before += 6;
    }
}

inline
void rowDefilter4ColorC8(uint8_t* input_row, uint8_t* prior_row,
                         uint32_t row_bytes, uint8_t* recon_row) {
    uint32_t value0, value1, value2, value3, value4, value5, value6, value7;
    value0 = input_row[0];
    value1 = input_row[1];
    value2 = input_row[2];
    value3 = input_row[3];
    value4 = input_row[4];
    value5 = input_row[5];
    value6 = input_row[6];
    value7 = input_row[7];
    recon_row[0] = value4 + prior_row[0];
    recon_row[1] = value5 + prior_row[1];
    recon_row[2] = value2 + prior_row[2];
    recon_row[3] = value3 + prior_row[3];
    recon_row[4] = value0 + prior_row[4];
    recon_row[5] = value1 + prior_row[5];
    recon_row[6] = value6 + prior_row[6];
    recon_row[7] = value7 + prior_row[7];

    uint8_t* input_end = input_row + row_bytes;
    uint8_t* before_bytes = recon_row;
    uint8_t* prior_before = prior_row;
    input_row += 8;
    prior_row += 8;
    recon_row += 8;
    while (input_row < input_end) {
        value0 = input_row[0];
        value1 = input_row[1];
        value2 = input_row[2];
        value3 = input_row[3];
        value4 = input_row[4];
        value5 = input_row[5];
        value6 = input_row[6];
        value7 = input_row[7];
        recon_row[0] = value4 + filterPaeth(before_bytes[0], prior_row[0],
                                            prior_before[0]);
        recon_row[1] = value5 + filterPaeth(before_bytes[1], prior_row[1],
                                            prior_before[1]);
        recon_row[2] = value2 + filterPaeth(before_bytes[2], prior_row[2],
                                            prior_before[2]);
        recon_row[3] = value3 + filterPaeth(before_bytes[3], prior_row[3],
                                            prior_before[3]);
        recon_row[4] = value0 + filterPaeth(before_bytes[4], prior_row[4],
                                            prior_before[4]);
        recon_row[5] = value1 + filterPaeth(before_bytes[5], prior_row[5],
                                            prior_before[5]);
        recon_row[6] = value6 + filterPaeth(before_bytes[6], prior_row[6],
                                            prior_before[6]);
        recon_row[7] = value7 + filterPaeth(before_bytes[7], prior_row[7],
                                            prior_before[7]);
        recon_row    += 8;
        input_row    += 8;
        before_bytes += 8;
        prior_row    += 8;
        prior_before += 8;
    }
}

static
bool deFilterC3TureColor(uint8_t* input, uint32_t height, uint32_t row_bytes,
                         uint32_t stride, uint8_t* recon_row) {
    // process the first row.
    uint8_t filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/Average(3)/"
                   << "Paeth(4)";
        return false;
    }
    switch (filter_type) {
        case FILTER_NONE:
            rowDefilter0ColorC3(input, row_bytes, recon_row, false);
            break;
        case FILTER_SUB:
            rowDefilter1ColorC3(input, row_bytes, recon_row, false);
            break;
        case FILTER_UP:
            rowDefilter0ColorC3(input, row_bytes, recon_row, false);
            break;
        case FILTER_PAETH:
            row0Defilter3ColorC3(input, row_bytes, recon_row);
            break;
        case FILTER_AVERAGE:
            rowDefilter1ColorC3(input, row_bytes, recon_row, false);
            break;
    }
    input += row_bytes;
    recon_row += stride;

    // process the most rows.
    uint8_t *prior_row;
    for (uint32_t row = 1; row < height - 1; ++row) {
        prior_row = recon_row - stride;
        filter_type = *input++;
        if (filter_type > 4) {
            LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                       << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                       << "Average(3)/Paeth(4)";
            return false;
        }
        switch (filter_type) {
            case FILTER_NONE:
                rowDefilter0ColorC3(input, row_bytes, recon_row, false);
                break;
            case FILTER_SUB:
                rowDefilter1ColorC3(input, row_bytes, recon_row, false);
                break;
            case FILTER_UP:
                rowDefilter2ColorC3(input, prior_row, row_bytes, recon_row,
                                    false);
                break;
            case FILTER_AVERAGE:
                rowDefilter3ColorC3(input, prior_row, row_bytes, recon_row,
                                    false);
                break;
            case FILTER_PAETH:
                rowDefilter4ColorC3(input, prior_row, row_bytes, recon_row,
                                    false);
                break;
        }
        input += row_bytes;
        recon_row += stride;
    }

    // process the last row.
    prior_row = recon_row - stride;
    filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                   << "Average(3)/Paeth(4)";
        return false;
    }
    switch (filter_type) {
        case FILTER_NONE:
            rowDefilter0ColorC3(input, row_bytes, recon_row, true);
            break;
        case FILTER_SUB:
            rowDefilter1ColorC3(input, row_bytes, recon_row, true);
            break;
        case FILTER_UP:
            rowDefilter2ColorC3(input, prior_row, row_bytes, recon_row, true);
            break;
        case FILTER_AVERAGE:
            rowDefilter3ColorC3(input, prior_row, row_bytes, recon_row, true);
            break;
        case FILTER_PAETH:
            rowDefilter4ColorC3(input, prior_row, row_bytes, recon_row, true);
            break;
    }

    return true;
}

static
bool deFilterC4TureColor(uint8_t* input, uint32_t height, uint32_t row_bytes,
                         uint32_t stride, uint8_t* recon_row) {
    // process the first row.
    uint8_t filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/Average(3)/"
                   << "Paeth(4)";
        return false;
    }
    switch (filter_type) {
        case FILTER_NONE:
            rowDefilter0ColorC4(input, row_bytes, recon_row, false);
            break;
        case FILTER_SUB:
            rowDefilter1ColorC4(input, row_bytes, recon_row, false);
            break;
        case FILTER_UP:
            rowDefilter0ColorC4(input, row_bytes, recon_row, false);
            break;
        case FILTER_AVERAGE:
            row0Defilter3ColorC4(input, row_bytes, recon_row);
            break;
        case FILTER_PAETH:
            rowDefilter1ColorC4(input, row_bytes, recon_row, false);
            break;
    }
    input += row_bytes;
    recon_row += stride;

    // process the most rows.
    uint8_t *prior_row;
    for (uint32_t row = 1; row < height - 1; ++row) {
        prior_row = recon_row - stride;
        filter_type = *input++;
        if (filter_type > 4) {
            LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                       << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                       << "Average(3)/Paeth(4)";
            return false;
        }
        switch (filter_type) {
            case FILTER_NONE:
                rowDefilter0ColorC4(input, row_bytes, recon_row, false);
                break;
            case FILTER_SUB:
                rowDefilter1ColorC4(input, row_bytes, recon_row, false);
                break;
            case FILTER_UP:
                rowDefilter2ColorC4(input, prior_row, row_bytes, recon_row,
                                    false);
                break;
            case FILTER_AVERAGE:
                rowDefilter3ColorC4(input, prior_row, row_bytes, recon_row,
                                    false);
                break;
            case FILTER_PAETH:
                rowDefilter4ColorC4(input, prior_row, row_bytes, recon_row,
                                    false);
                break;
        }
        input += row_bytes;
        recon_row += stride;
    }

    // process the last row.
    prior_row = recon_row - stride;
    filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                   << "Average(3)/Paeth(4)";
        return false;
    }
    switch (filter_type) {
        case FILTER_NONE:
            rowDefilter0ColorC4(input, row_bytes, recon_row, true);
            break;
        case FILTER_SUB:
            rowDefilter1ColorC4(input, row_bytes, recon_row, true);
            break;
        case FILTER_UP:
            rowDefilter2ColorC4(input, prior_row, row_bytes, recon_row, true);
            break;
        case FILTER_AVERAGE:
            rowDefilter3ColorC4(input, prior_row, row_bytes, recon_row, true);
            break;
        case FILTER_PAETH:
            rowDefilter4ColorC4(input, prior_row, row_bytes, recon_row, true);
            break;
    }

    return true;
}

static
bool deFilterC6TureColor(uint8_t* input, uint32_t height, uint32_t row_bytes,
                         uint32_t stride, uint8_t* recon_row) {
    uint8_t filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/Average(3)/"
                   << "Paeth(4)";
        return false;
    }

    switch (filter_type) {
        case FILTER_NONE:
            rowDefilter0ColorC6(input, row_bytes, recon_row);
            break;
        case FILTER_SUB:
            rowDefilter1ColorC6(input, row_bytes, recon_row);
            break;
        case FILTER_UP:
            rowDefilter0ColorC6(input, row_bytes, recon_row);
            break;
        case FILTER_AVERAGE:
            row0Defilter3ColorC6(input, row_bytes, recon_row);
            break;
        case FILTER_PAETH:
            rowDefilter1ColorC6(input, row_bytes, recon_row);
            break;
    }
    input += row_bytes;
    recon_row += stride;

    uint8_t *prior_row;
    for (uint32_t row = 1; row < height; ++row) {
        prior_row = recon_row - stride;
        filter_type = *input++;
        if (filter_type > 4) {
            LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                       << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                       << "Average(3)/Paeth(4)";
            return false;
        }
        switch (filter_type) {
            case FILTER_NONE:
                rowDefilter0ColorC6(input, row_bytes, recon_row);
                break;
            case FILTER_SUB:
                rowDefilter1ColorC6(input, row_bytes, recon_row);
                break;
            case FILTER_UP:
                rowDefilter2ColorC6(input, prior_row, row_bytes, recon_row);
                break;
            case FILTER_AVERAGE:
                rowDefilter3ColorC6(input, prior_row, row_bytes, recon_row);
                break;
            case FILTER_PAETH:
                rowDefilter4ColorC6(input, prior_row, row_bytes, recon_row);
                break;
        }
        input += row_bytes;
        recon_row += stride;
    }

    return true;
}

static
bool deFilterC8TureColor(uint8_t* input, uint32_t height, uint32_t row_bytes,
                         uint32_t stride, uint8_t* recon_row) {
    uint8_t filter_type = *input++;
    if (filter_type > 4) {
        LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                   << ", correct filter type: None(0)/Sub(1)/Up(2)/Average(3)/"
                   << "Paeth(4)";
        return false;
    }

    switch (filter_type) {
        case FILTER_NONE:
            rowDefilter0ColorC8(input, row_bytes, recon_row);
            break;
        case FILTER_SUB:
            rowDefilter1ColorC8(input, row_bytes, recon_row);
            break;
        case FILTER_UP:
            rowDefilter0ColorC8(input, row_bytes, recon_row);
            break;
        case FILTER_AVERAGE:
            row0Defilter3ColorC8(input, row_bytes, recon_row);
            break;
        case FILTER_PAETH:
            rowDefilter1ColorC8(input, row_bytes, recon_row);
            break;
    }
    input += row_bytes;
    recon_row += stride;

    uint8_t *prior_row;
    for (uint32_t row = 1; row < height; ++row) {
        prior_row = recon_row - stride;
        filter_type = *input++;
        if (filter_type > 4) {
            LOG(ERROR) << "The readed filter type: " << (uint32_t)filter_type
                       << ", correct filter type: None(0)/Sub(1)/Up(2)/"
                       << "Average(3)/Paeth(4)";
            return false;
        }
        switch (filter_type) {
            case FILTER_NONE:
                rowDefilter0ColorC8(input, row_bytes, recon_row);
                break;
            case FILTER_SUB:
                rowDefilter1ColorC8(input, row_bytes, recon_row);
                break;
            case FILTER_UP:
                rowDefilter2ColorC8(input, prior_row, row_bytes, recon_row);
                break;
            case FILTER_AVERAGE:
                rowDefilter3ColorC8(input, prior_row, row_bytes, recon_row);
                break;
            case FILTER_PAETH:
                rowDefilter4ColorC8(input, prior_row, row_bytes, recon_row);
                break;
        }
        input += row_bytes;
        recon_row += stride;
    }

    return true;
}

/*
 * reorder channels in a pixel form RGB to BGR for truecolor or truecolor with
 * alpha.
 */
bool PngDecoder::deFilterImageTrueColor(PngInfo& png_info, uint8_t* image,
                                        uint32_t stride) {
    int bytes = (bit_depth_ == 16 ? 2 : 1);
    int pixel_bytes = bytes * encoded_channels_;  // 3, 4, 6, 8
    uint32_t row_bytes = width_ * pixel_bytes;
    uint32_t scanline_bytes = row_bytes + 1;
    uint32_t stride_offset = 0;
    if (png_info.chunk_status & HAVE_tRNS) {
        stride_offset = (stride - scanline_bytes) & -16;  // align with 16 bytes
    }

    uint8_t *input = png_info.decompressed_image_start;
    uint8_t *recon_row = image + stride_offset;
    if (pixel_bytes == 3) {
        deFilterC3TureColor(input, height_, row_bytes, stride, recon_row);
    }
    else if (pixel_bytes == 4) {
        deFilterC4TureColor(input, height_, row_bytes, stride, recon_row);
    }
    else if (pixel_bytes == 6) {
        deFilterC6TureColor(input, height_, row_bytes, stride, recon_row);
    }
    else if (pixel_bytes == 8) {
        deFilterC8TureColor(input, height_, row_bytes, stride, recon_row);
    }
    else {
        LOG(ERROR) << "The pixel bytes: " << pixel_bytes
                   << ", correct value: 3/4/6/8";
        return false;
    }

    if (bit_depth_ == 16) {
        // force the image data from big-endian to platform-native.
        uint8_t value0, value1;
        recon_row = image + stride_offset;
        uint16_t *current_row16;
        for (uint32_t row = 0; row < height_; row++) {
            current_row16 = (uint16_t*)recon_row;
            for (uint32_t col = 0, col1 = 0; col < width_; col++, col1 += 2) {
                value0 = recon_row[col1];
                value1 = recon_row[col1 + 1];
                current_row16[col] = (value0 << 8) | value1;
            }
            recon_row += stride;
        }
    }
    png_info.defiltered_image = image + stride_offset;

    return true;
}

bool PngDecoder::computeTransparency(PngInfo& png_info, uint8_t* image,
                                     uint32_t stride) {
    if (color_type_ != 0 && color_type_ != 2) {
        LOG(ERROR) << "The expected color type for expanding alpha: "
                   << "greyscale(0)/true color(2).";
        return false;
    }

    if (bit_depth_ <= 8) {
        uint8_t* input_row = png_info.defiltered_image;
        uint8_t* output_row = image;
        uint8_t value0, value1, value2, value3;

        if (color_type_ == GRAY) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_; col0++) {
                     value0 = input_row[col0];
                     value1 = value0 == png_info.alpha_values[0] ? 0 : 255;
                     output_row[col1]     = value0;
                     output_row[col1 + 1] = value1;
                     col1 += 2;
                }
                input_row  += stride;
                output_row += stride;
            }
        }
        else if (color_type_ == TRUE_COLOR) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_ * 3;
                     col0 += 3) {
                     value0 = input_row[col0];
                     value1 = input_row[col0 + 1];
                     value2 = input_row[col0 + 2];
                     if (value0 == png_info.alpha_values[0] &&
                         value1 == png_info.alpha_values[1] &&
                         value2 == png_info.alpha_values[2]) {
                         value3 = 0;
                     }
                     else {
                         value3 = 255;
                     }
                     output_row[col1]     = value0;
                     output_row[col1 + 1] = value1;
                     output_row[col1 + 2] = value2;
                     output_row[col1 + 3] = value3;
                     col1 += 4;
                }
                input_row  += stride;
                output_row += stride;
            }
        }
        else {
        }
    }
    else {  // bit_depth_ == 16
        uint16_t* input_row = (uint16_t*)(png_info.defiltered_image);
        uint16_t* output_row = (uint16_t*)image;
        uint16_t* alpha_values = (uint16_t*)png_info.alpha_values;
        uint16_t value0, value1, value2, value3;

        if (color_type_ == GRAY) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_; col0++) {
                     value0 = input_row[col0];
                     value1 = value0 == alpha_values[0] ? 0 : 65535;
                     output_row[col1]     = value0;
                     output_row[col1 + 1] = value1;
                     col1 += 2;
                }
                input_row  += stride;
                output_row += stride;
            }
        }
        else if (color_type_ == TRUE_COLOR) {
            for (uint32_t row = 0; row < height_; row++) {
                for (uint32_t col0 = 0, col1 = 0; col0 < width_ * 3;
                     col0 += 3) {
                     value0 = input_row[col0];
                     value1 = input_row[col0 + 1];
                     value2 = input_row[col0 + 2];
                     if (value0 == alpha_values[0] &&
                         value1 == alpha_values[1] &&
                         value2 == alpha_values[2]) {
                         value3 = 0;
                     }
                     else {
                         value3 = 65535;
                     }
                     output_row[col1]     = value0;
                     output_row[col1 + 1] = value1;
                     output_row[col1 + 2] = value2;
                     output_row[col1 + 3] = value3;
                     col1 += 4;
                }
                input_row  += stride;
                output_row += stride;
            }
        }
        else {
        }
    }

    return true;
}

bool PngDecoder::expandPalette(PngInfo& png_info, uint8_t* image,
                               uint32_t stride) {
    if (color_type_ != INDEXED_COLOR) {
        LOG(ERROR) << "The expected color type for expanding palette: "
                   << "indexed-colour(3).";
        return false;
    }

    uint8_t* input_row = png_info_.defiltered_image;
    uint8_t* palette = png_info.palette;
    uint8_t* output_row = image;
    uint32_t index;

    if (channels_ == 3) {
        for (uint32_t row = 0; row < height_; row++) {
            uint8_t* src = input_row;
            uint8_t* dst = output_row;
            uint32_t col = 0;
            for (; col < width_ - 1; col++) {
                index = src[col];
                index *= 4;
                __m128i value = _mm_loadu_si128((__m128i*)(palette + index));
                __m128 fvalue = _mm_castsi128_ps(value);
                _mm_store_ss((float*)dst, fvalue);
                dst += 3;
            }

            index = src[col];
            index *= 4;
            uint8_t value0 = palette[index];
            uint8_t value1 = palette[index + 1];
            uint8_t value2 = palette[index + 2];
            dst[0] = value0;
            dst[1] = value1;
            dst[2] = value2;

            input_row  += stride;
            output_row += stride;
        }
    }
    else if (channels_ == 4) {
        for (uint32_t row = 0; row < height_; row++) {
            uint8_t* src = input_row;
            uint8_t* dst = output_row;
            for (uint32_t col = 0; col < width_; col++) {
                index = src[col];
                index *= 4;
                __m128i value = _mm_loadu_si128((__m128i*)(palette + index));
                __m128 fvalue = _mm_castsi128_ps(value);
                _mm_store_ss((float*)dst, fvalue);
                dst += 4;
            }
            input_row  += stride;
            output_row += stride;
        }
    }
    else {
    }

    return true;
}

bool PngDecoder::readHeader() {
    getChunkHeader(png_info_);
    if (png_info_.current_chunk.type != png_IHDR) {
        std::string chunk_name = getChunkName(png_info_.current_chunk.type);
        LOG(ERROR) << "encountering the chunk: " << chunk_name
                   << ", expecting an IHDR chunk.";
        LOG(ERROR) << "The first chunk must be IHDR.";
        return false;
    }

    bool succeeded = parseIHDR(png_info_);
    if (!succeeded) return false;

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

        getChunkHeader(png_info_);
    }

    if (png_info_.current_chunk.type == png_IEND) {
        LOG(ERROR) << "No image data chunk appears.";
        return false;
    }

    return true;
}

bool PngDecoder::decodeData(uint32_t stride, uint8_t* image) {
    if (png_info_.current_chunk.type != png_IDAT) {
        LOG(ERROR) << "encountering the chunk: "
                   << getChunkName(png_info_.current_chunk.type)
                   << ", expecting the IDAT chunk.";
        return false;
    }

    uint32_t decompressed_image_size = (((width_ * encoded_channels_ *
                                          bit_depth_ + 7) >> 3) + 1) * height_;
    png_info_.decompressed_image_end = image + stride * height_;
    png_info_.decompressed_image_start = png_info_.decompressed_image_end -
                                         decompressed_image_size;
    png_info_.decompressed_image = png_info_.decompressed_image_start;
    png_info_.zlib_buffer.bit_number = 0;
    png_info_.zlib_buffer.code_buffer = 0;

    bool succeeded;
    while (png_info_.current_chunk.type != png_IEND) {
        switch (png_info_.current_chunk.type) {
            case png_tIME:
                succeeded = parsetIME(png_info_);
                if (!succeeded) return false;
                if (png_info_.header_after_idat) {
                    png_info_.header_after_idat = false;
                }
                break;
            case png_zTXt:
                succeeded = parsezTXt(png_info_);
                if (!succeeded) return false;
                if (png_info_.header_after_idat) {
                    png_info_.header_after_idat = false;
                }
                break;
            case png_tEXt:
                succeeded = parsetEXt(png_info_);
                if (!succeeded) return false;
                if (png_info_.header_after_idat) {
                    png_info_.header_after_idat = false;
                }
                break;
            case png_iTXt:
                succeeded = parseiTXt(png_info_);
                if (!succeeded) return false;
                if (png_info_.header_after_idat) {
                    png_info_.header_after_idat = false;
                }
                break;
            case png_IDAT:
                succeeded = parseIDATs(png_info_, image, stride);
                if (!succeeded) return false;
                break;
            default:
                succeeded = parsetUnknownChunk(png_info_);
                if (!succeeded) return false;
                break;
        }

        if (!png_info_.header_after_idat) {
            getChunkHeader(png_info_);
        }
    }

    succeeded = parseIEND(png_info_);
    if (!succeeded) return false;

    if (color_type_ != TRUE_COLOR && color_type_ != TRUE_COLOR_WITH_ALPHA) {
        succeeded = deFilterImage(png_info_, image, stride);
    }
    else {  // truecolour or truecolor with alpha.
        succeeded = deFilterImageTrueColor(png_info_, image, stride);
    }
    if (!succeeded) return false;

    if (color_type_ == GRAY || color_type_ == TRUE_COLOR) {
        if (encoded_channels_ + 1 == channels_) {
            succeeded = computeTransparency(png_info_, image, stride);
            if (!succeeded) return false;
        }
    }
    else if (color_type_ == INDEXED_COLOR) {
        succeeded = expandPalette(png_info_, image, stride);
        if (!succeeded) return false;
    }
    else {
    }

    return true;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl