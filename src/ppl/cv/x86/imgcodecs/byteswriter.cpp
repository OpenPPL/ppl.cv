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

#include "byteswriter.h"

#include <string.h>
#include <assert.h>

#include "codecs.h"

namespace ppl {
namespace cv {
namespace x86 {

BytesWriter::BytesWriter(FILE* fp) {
    fp_ = fp;
    start_ = new uchar[FILE_BLOCK_SIZE];
    end_ = start_ + FILE_BLOCK_SIZE;
    current_ = start_;
    block_position_ = 0;
}

int BytesWriter::getPosition() {
    int position = current_ - start_ + block_position_;

    return position;
}

void BytesWriter::writeBlock() {
    int size = (int)(current_ - start_);

    if (size == 0) {
        return;
    }

    fwrite(start_, 1, size, fp_);
    current_ = start_;
    block_position_ += size;
}

void BytesWriter::putByte(int value) {
    *current_++ = (uchar)value;
    if (current_ >= end_) {
        writeBlock();
    }
}

void BytesWriter::putBytes(const void* buffer, int count) {
    uchar* data = (uchar*)buffer;
    assert(data && current_ && count >= 0);

    while (count) {
        int left = (int)(end_ - current_);

        if (left > count) {
            left = count;
        }

        if (left > 0) {
            memcpy(current_, data, left);
            current_ += left;
            data += left;
            count -= left;
        }
        if (current_ == end_) {
            writeBlock();
        }
    }
}

void BytesWriter::putWord(int value) {
    uchar *current = current_;

    if (current+1 < end_) {
        current[0] = (uchar)value;
        current[1] = (uchar)(value >> 8);
        current_ = current + 2;
        if (current_ == end_) {
            writeBlock();
        }
    }
    else {
        putByte(value);
        putByte(value >> 8);
    }
}

void BytesWriter::putDWord(int value) {
    uchar *current = current_;

    if (current+3 < end_) {
        current[0] = (uchar)value;
        current[1] = (uchar)(value >> 8);
        current[2] = (uchar)(value >> 16);
        current[3] = (uchar)(value >> 24);
        current_ = current + 4;
        if (current_ == end_) {
            writeBlock();
        }
    }
    else {
        putByte(value);
        putByte(value >> 8);
        putByte(value >> 16);
        putByte(value >> 24);
    }
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl
