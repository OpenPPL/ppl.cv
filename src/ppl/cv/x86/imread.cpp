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

#include "ppl/cv/x86/imread.h"
#include "imgcodecs/bytesreader.h"
#include "imgcodecs/imagecodecs.h"
#include "imgcodecs/bmp.h"
#include "imgcodecs/jpeg.h"
#include "imgcodecs/png.h"
#include "imgcodecs/codecs.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "ppl/common/log.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace x86 {

bool detectFormat(BytesReader& file_data, ImageFormats* image_format) {
    const char* bmp_signature  = "BM";
    const char* jpeg_signature = "\xFF\xD8\xFF";
    const char png_signature[] = {(int8_t)0x89, (int8_t)0x50, (int8_t)0x4E,
                                  (int8_t)0x47, (int8_t)0x0D, (int8_t)0x0A,
                                  (int8_t)0x1A, (int8_t)0x0A};

    const char* file_signature = (char*)file_data.data();

    int matched;
    matched = memcmp(bmp_signature, file_signature, 2);
    if (matched == 0) {
        *image_format = BMP;
        return true;
    }
    matched = memcmp(jpeg_signature, file_signature, 3);
    if (matched == 0) {
        *image_format = JPEG;
        return true;
    }
    matched = memcmp(png_signature, file_signature, 8);
    if (matched == 0) {
        *image_format = PNG;
        file_data.skipBytes(8);
        return true;
    }

    *image_format = UNSUPPORTED;
    return false;
}

RetCode Imread(const char* file_name, int* height, int* width, int* channels,
               int* stride, uchar** image) {
    assert(file_name != nullptr);
    assert(height != nullptr);
    assert(width != nullptr);
    assert(channels != nullptr);
    assert(stride != nullptr);
    assert(image != nullptr);

    RetCode code = RC_SUCCESS;
    FILE* fp = fopen(file_name, "rb");
    if (fp == nullptr) {
        LOG(ERROR) << "failed to open the input file: " << file_name;
        return RC_OTHER_ERROR;
    }

    BytesReader file_data(fp);
    ImageFormats image_format;
    bool succeeded = detectFormat(file_data, &image_format);
    if (succeeded == false) {
        if (image_format == UNSUPPORTED) {
            LOG(ERROR) << "unsupported image format.";
        }
        fclose(fp);
        return RC_OTHER_ERROR;
    }

    ImageDecoder* decoder = nullptr;
    if (image_format == BMP) {
        decoder = new BmpDecoder(file_data);
    }
    else if (image_format == JPEG) {
        decoder = new JpegDecoder(file_data);
    }
    else if (image_format == PNG) {
        decoder = new PngDecoder(file_data);
    }
    else {
    }

    succeeded = decoder->readHeader();
    if (succeeded == false) {
        LOG(ERROR) << "failed to read file header.";
        fclose(fp);
        return RC_OTHER_ERROR;
    }

    *height   = decoder->height();
    *width    = decoder->width();
    *channels = decoder->channels();
    if (image_format == PNG) {
        int bytes = (decoder->depth() == 16 ? 2 : 1);
        *stride = (decoder->width() * decoder->channels() * bytes + 1 + 15) &
                   -16;
    }
    else {
        *stride = (decoder->width() * decoder->channels() + 3) & -4;
    }
    size_t size = (*stride) * (*height);
    assert(size < MAX_IMAGE_SIZE);
    (*image) = (uchar*)malloc(size);
    if (*image == nullptr) {
        LOG(ERROR) << "failed to allocate memory for the image.";
        fclose(fp);
        return RC_OUT_OF_MEMORY;
    }

    succeeded = decoder->decodeData(*stride, (*image));
    if (succeeded == false) {
        LOG(ERROR) << "failed to decode the file data.";
        fclose(fp);
        return RC_OTHER_ERROR;
    }
    fclose(fp);

    return code;
}

}  // namespace x86
}  // namespace cv
}  // namespace ppl
