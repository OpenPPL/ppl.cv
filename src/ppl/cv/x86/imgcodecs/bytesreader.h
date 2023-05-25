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

#ifndef __ST_HPC_PPL_CV_X86_BYTES_READER_H_
#define __ST_HPC_PPL_CV_X86_BYTES_READER_H_

#include "crc32.h"

#include <stdio.h>

#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

class BytesReader {
  public:
    BytesReader(FILE* fp);
    ~BytesReader() {
        delete [] start_;
    }

    uchar* data() const;
    int getPosition();
    void setPosition(int position);
    void skip(int bytes);
    void readBlock();
    int getByte();
    int getBytes(void* buffer, int count);
    int getWord();
    int getWordBigEndian();
    int getDWord();
    int getDWordBigEndian();
    uint32_t getDWordBigEndian1();
    uchar* getCurrentPosition() const;
    uint32_t getValidSize() const;
    void setCrcChecking(Crc32* crc);
    void unsetCrcChecking();

  private:
    FILE* fp_;
    Crc32* crc_;
    uchar* start_;
    uchar* end_;
    uchar* current_;
    int block_position_;
    bool is_last_block_;
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_BYTES_READER_H_
