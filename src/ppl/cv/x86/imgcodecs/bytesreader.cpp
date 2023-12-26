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

#include "bytesreader.h"

#include <string.h>

#include "codecs.h"
#include "ppl/common/log.h"

namespace ppl {
namespace cv {
namespace x86 {

BytesReader::BytesReader(FILE* fp) {
    fp_ = fp;
    start_ = new uint8_t[FILE_BLOCK_SIZE];
    current_ = start_;
    block_position_ = 0;
    is_last_block_ = false;
    crc_ = nullptr;

    readBlock();
}

BytesReader::~BytesReader() {
    delete [] start_;
}

uint8_t* BytesReader::data() const {
    return start_;
}

uint32_t BytesReader::getPosition() {
    uint32_t position = current_ - start_ + block_position_;

    return position;
}

void BytesReader::setPosition(uint32_t position) {
    uint32_t offset = position % FILE_BLOCK_SIZE;
    uint32_t prev_block_position = block_position_;
    block_position_ = position - offset;
    current_ = start_ + offset;
    if (prev_block_position != block_position_) {
        readBlock();
    }
}

bool BytesReader::skipBytes(uint32_t size) {
    if (size <= FILE_BLOCK_SIZE - (current_ - start_)) {
        current_ += size;
    }
    else {
        uint32_t left = size - (FILE_BLOCK_SIZE - (current_ - start_));
        uint32_t offset = left % FILE_BLOCK_SIZE;
        left -= offset;
        int code = fseek(fp_, left, SEEK_CUR);
        if (code != 0) {
            LOG(ERROR) << "Error in skipping " << size
                       << " bytes in input file.";
            return false;
        }
        block_position_ += left;

        readBlock();
        current_ = start_ + offset;
    }

    return true;
}

void BytesReader::readBlock() {
    uint32_t readed = fread(start_, 1, FILE_BLOCK_SIZE, fp_);
    if (ferror(fp_)) {
        LOG(ERROR) << "Error in reading the input file.";
    }
    if (feof(fp_)) {
        is_last_block_ = true;
    }
    end_ = start_ + readed;
    current_ = start_;
    if (readed == FILE_BLOCK_SIZE) {
        block_position_ += FILE_BLOCK_SIZE;
    }

    if (crc_ != nullptr && crc_->isChecking()) {
        crc_->resetData(start_, readed);
        crc_->calculateCrc();
    }
}

int32_t BytesReader::getByte() {
    if (current_ >= end_) {
        if (is_last_block_) return -2;
        readBlock();
    }

    return *current_++;
}

int32_t BytesReader::getWordLittleEndian() {
    uint8_t *current = current_;
    int32_t value;

    if (current + 1 < end_) {
        value = current[0] + (current[1] << 8);
        current_ = current + 2;
    }
    else {
        value  = getByte();
        value |= getByte() << 8;
    }

    return value;
}

int32_t BytesReader::getWordBigEndian() {
    uint8_t *current = current_;
    int32_t value;

    if (current + 1 < end_) {
        value = (current[0] << 8) + current[1];
        current_ = current + 2;
    }
    else {
        value = getByte() << 8;
        value|= getByte();
    }

    return value;
}

int32_t BytesReader::getDWordLittleEndian() {
    uint8_t *current = current_;
    int32_t value;

    if (current + 3 < end_) {
        value = current[0] + (current[1] << 8) + (current[2] << 16) +
                (current[3] << 24);
        current_ = current + 4;
    }
    else {
        value = getByte();
        value |= getByte() << 8;
        value |= getByte() << 16;
        value |= getByte() << 24;
    }

    return value;
}

int32_t BytesReader::getDWordBigEndian() {
    uint8_t *current = current_;
    int32_t value;

    if (current + 3 < end_) {
        value = (current[0] << 24) + (current[1] << 16) + (current[2] << 8) +
                 current[3];
        current_ = current + 4;
    }
    else {
        value  = getByte() << 24;
        value |= getByte() << 16;
        value |= getByte() << 8;
        value |= getByte();
    }

    return value;
}

void BytesReader::getBytes(void* buffer, int32_t count) {
    if (count >= FILE_BLOCK_SIZE) {
        LOG(ERROR) << "The bytes are too big than the buffer.";
    }

    int32_t left = (int32_t)(end_ - current_);
    if (count <= left) {
        memcpy(buffer, current_, count);
        current_ += count;
    }
    else {
        uint8_t* data = (uint8_t*)buffer;
        memcpy(data, current_, left);
        data += left;
        count -= left;

        readBlock();
        memcpy(data, current_, count);
        current_ = start_ + count;
    }
}

uint32_t BytesReader::getValidSize() const {
    return end_ - current_;
}

void BytesReader::setCrcChecking(Crc32* crc) {
    crc_ = crc;
}

void BytesReader::unsetCrcChecking() {
    crc_ = nullptr;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl
