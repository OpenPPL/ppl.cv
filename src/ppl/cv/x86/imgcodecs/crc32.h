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

#ifndef __ST_HPC_PPL_CV_X86_IMGCODECS_CRC32_H_
#define __ST_HPC_PPL_CV_X86_IMGCODECS_CRC32_H_

#include <stdint.h>

namespace ppl {
namespace cv {
namespace x86 {

enum ByteOrder {
  LITTLE_ENDIAN_ORDER = 0,
  BIG_ENDIAN_ORDER    = 1,
};

class Crc32 {
  public:
    Crc32();
    Crc32(uint8_t* data_poiter, uint32_t data_size, uint32_t crc_value,
          uint32_t crc_length, bool is_checking);
    ~Crc32() {}

    ByteOrder checkByteOrder();
    void setCrc(uint8_t* data_poiter, uint32_t data_size,
                uint32_t crc_value, uint32_t crc_length);
    bool calculateCrc(uint32_t data);
    bool calculateCrc();
    bool resetData(uint8_t* data_poiter, uint32_t data_size);
    bool isChecking() const;
    void turnOn();
    void turnOff();
    uint32_t getCrcValue() const;
    uint32_t getCrcLength() const;

  private:
    const uint8_t* data_poiter_;
    uint32_t data_size_;
    uint32_t crc_value_;
    uint32_t crc_length_;
    ByteOrder byte_order_;
    bool is_checking_;
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMGCODECS_CRC32_H_
