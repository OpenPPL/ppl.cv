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
#include <iostream> // debug

#include "codecs.h"
#include "ppl/common/log.h"

namespace ppl {
namespace cv {
namespace x86 {

BytesReader::BytesReader(FILE* fp) {
    fp_ = fp;
    start_ = new uchar[FILE_BLOCK_SIZE];
    current_ = start_;
    block_position_ = 0;
    is_last_block_ = false;

    readBlock();
}

uchar* BytesReader::data() const {
    return start_;
}

int BytesReader::getPosition() {
    int position = current_ - start_ + block_position_;

    return position;
}

void BytesReader::setPosition(int position) {
    int offset = position % FILE_BLOCK_SIZE;
    int prev_block_position = block_position_;
    block_position_ = position - offset;
    current_ = start_ + offset;
    if (prev_block_position != block_position_)
        readBlock();
}

void BytesReader::skip(int bytes) {
    // if (bytes <= FILE_BLOCK_SIZE - (current_ - start_))
    current_ += bytes;
    // else ...
}

void BytesReader::readBlock() {
    setPosition(getPosition());

    fseek(fp_, block_position_, SEEK_SET);
    size_t readed = fread(start_, 1, FILE_BLOCK_SIZE, fp_);
    if (feof(fp_)) {
        is_last_block_ = true;
        // LOG(INFO) << "block_position_: " << block_position_ << ", readed "
        //           << readed << " bytes, Reaching the end of the file.";
    }
    if (ferror(fp_)) {
        LOG(ERROR) << "Error in reading the file.";
    }
    end_ = start_ + readed;
}

int BytesReader::getByte() {
    if (current_ >= end_) {
        if (is_last_block_) return -2;
        readBlock();
        // current = current_;
    }

    // std::cout << "getByte: " << std::showbase << std::hex << value << std::noshowbase << std::endl;
    return *current_++;
}
/*
int BytesReader::getByte() {
    uchar* current = current_;
    int value;

    if (current >= end_) {
        if (is_last_block_) return -2;
        readBlock();
        current = current_;
    }

    value = *current;
    current_ = current + 1;

    // std::cout << "getByte: " << std::showbase << std::hex << value << std::noshowbase << std::endl;
    return value;
} */

int BytesReader::getBytes(void* buffer, int count) {
    uchar* data = (uchar*)buffer;
    int readed = 0;

    while (count > 0) {
        int left;

        for (;;) {
            left = (int)(end_ - current_);
            if (left > count) left = count;
            if (left > 0) break;
            readBlock();
        }
        memcpy(data, current_, left);
        current_ += left;
        data += left;
        count -= left;
        readed += left;
    }

    // std::cout << "getBytes readed: " << std::dec << readed << std::endl;
    return readed;
}

int BytesReader::getWord() {
    uchar *current = current_;
    int value;

    if (current + 1 < end_) {
        value = current[0] + (current[1] << 8);
        current_ = current + 2;
    }
    else {
        value = getByte();
        value|= getByte() << 8;
    }

    // std::cout << "getWord: " << std::dec << value << std::endl;
    return value;
}

int BytesReader::getWordBigEndian() {
    uchar *current = current_;
    int value;

    if (current + 1 < end_) {
        value = (current[0] << 8) + current[1];
        current_ = current + 2;
    }
    else {
        value = getByte() << 8;
        value|= getByte();
    }

    // std::cout << "getWordBigEndian: " << std::dec << value << std::endl;
    return value;
}

int BytesReader::getDWord() {
    uchar *current = current_;
    int value;

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

    // std::cout << "getDWord: " << std::dec << value << std::endl;
    return value;
}

int BytesReader::getDWordBigEndian() {
    uchar *current = current_;
    int value;

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

    // std::cout << "getDWordBigEndian: " << std::dec << value << std::endl;
    return value;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl
