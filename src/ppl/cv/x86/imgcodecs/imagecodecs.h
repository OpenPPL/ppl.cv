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

#ifndef __ST_HPC_PPL_CV_X86_IMAGE_CODECS_H_
#define __ST_HPC_PPL_CV_X86_IMAGE_CODECS_H_

#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

class ImageDecoder {
  public:
    ImageDecoder();
    virtual ~ImageDecoder();

    uint32_t height() const;
    uint32_t width() const;
    uint32_t channels() const;
    uint32_t depth() const;
    virtual bool readHeader() = 0;
    virtual bool decodeData(uint32_t stride, uint8_t* image) = 0;

  protected:
    uint32_t height_;
    uint32_t width_;
    uint32_t channels_;
    uint32_t depth_;
};

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMAGE_CODECS_H_
