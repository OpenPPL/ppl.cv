/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef _ST_HPC_PPL_CV_OCL_KERNEL_TYPES_H_
#define _ST_HPC_PPL_CV_OCL_KERNEL_TYPES_H_

enum BorderType {
    BORDER_TYPE_CONSTANT       = 0, //!< `iiiiii|abcdefgh|iiiiii` with some specified `i`
    BORDER_TYPE_REPLICATE      = 1, //!< `aaaaaa|abcdefgh|hhhhhh`
    BORDER_TYPE_REFLECT        = 2, //!< `fedcba|abcdefgh|hgfedc`
    BORDER_TYPE_WRAP           = 3, //!< `cdefgh|abcdefgh|abcdef`
    BORDER_TYPE_REFLECT_101    = 4, //!< `gfedcb|abcdefgh|gfedcb`
    BORDER_TYPE_TRANSPARENT    = 5, //!< `uvwxyz|abcdefgh|ijklmn`
    BORDER_TYPE_REFLECT101     = BORDER_TYPE_REFLECT_101,
    BORDER_TYPE_DEFAULT        = BORDER_TYPE_REFLECT_101,
    BORDER_TYPE_ISOLATED       = 16
};

inline int divideUp(int total, int shift) {
  return (total + ((1 << shift) - 1)) >> shift;
}

#endif