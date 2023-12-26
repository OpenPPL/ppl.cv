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
    ~BytesReader();

    uint8_t* data() const;
    uint32_t getPosition();
    void setPosition(uint32_t position);
    uint8_t* getCurrentPosition() const {return current_;}
    bool skipBytes(uint32_t size);
    void readBlock();
    int32_t getByte();
    int32_t getWordLittleEndian();
    int32_t getWordBigEndian();
    int32_t getDWordLittleEndian();
    int32_t getDWordBigEndian();
    void getBytes(void* buffer, int32_t count);
    uint32_t getValidSize() const;
    void setCrcChecking(Crc32* crc);
    void unsetCrcChecking();

  private:
    FILE* fp_;
    uint8_t* start_;
    uint8_t* end_;
    uint8_t* current_;
    uint32_t block_position_;
    bool is_last_block_;
    Crc32* crc_;
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_BYTES_READER_H_
