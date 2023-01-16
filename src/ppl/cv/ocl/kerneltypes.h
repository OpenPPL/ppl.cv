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
    BORDER_CONSTANT       = 0, //!< `iiiiii|abcdefgh|iiiiii` with some specified `i`
    BORDER_REPLICATE      = 1, //!< `aaaaaa|abcdefgh|hhhhhh`
    BORDER_REFLECT        = 2, //!< `fedcba|abcdefgh|hgfedc`
    BORDER_WRAP           = 3, //!< `cdefgh|abcdefgh|abcdef`
    BORDER_REFLECT_101    = 4, //!< `gfedcb|abcdefgh|gfedcb`
    BORDER_TRANSPARENT    = 5, //!< `uvwxyz|abcdefgh|ijklmn`
    BORDER_REFLECT101     = BORDER_REFLECT_101,
    BORDER_DEFAULT        = BORDER_REFLECT_101,
    BORDER_ISOLATED       = 16
};

inline int divideUp(int total, int shift) {
  return (total + ((1 << shift) - 1)) >> shift;
}

inline int divideUp1(int total, int shift) {
  return (total + (1 << (shift - 1))) >> shift;
}

inline int interpolateConstantBorder(int range, int radius, int index) {
  if (index < 0) {
    return -1;
  }
  else if (index < range) {
    return index;
  }
  else {
    return -1;
  }
}

inline int interpolateReplicateBorder(int range, int radius, int index) {
  if (index < 0) {
    return 0;
  }
  else if (index < range) {
    return index;
  }
  else {
    return range - 1;
  }
}

inline int interpolateReflectBorder(int range, int radius, int index) {
  if (range >= radius) {
    if (index < 0) {
      return -1 - index;
    }
    else if (index < range) {
      return index;
    }
    else {
      return (range << 1) - index - 1;
    }
  }
  else {
    if (index >= 0 && index < range) {
      return index;
    }
    else {
      if (range == 1) {
        index = 0;
      }
      else {
        do {
          if (index < 0)
            index = -1 - index;
          else
            index = (range << 1) - index - 1;
        } while (index >= range || index < 0);
      }

      return index;
    }
  }
}

inline int interpolateWarpBorder(int range, int radius, int index) {
  if (range >= radius) {
    if (index < 0) {
      return index + range;
    }
    else if (index < range) {
      return index;
    }
    else {
      return index - range;
    }
  }
  else {
    if (index >= 0 && index < range) {
      return index;
    }
    else {
      if (range == 1) {
        index = 0;
      }
      else {
        do {
          if (index < 0)
            index += range;
          else
            index -= range;
        } while (index >= range || index < 0);
      }

      return index;
    }
  }
}

inline int interpolateReflect101Border(int range, int radius, int index) {
  if (range > radius) {
    if (index < 0) {
      return 0 - index;
    }
    else if (index < range) {
      return index;
    }
    else {
      return (range << 1) - index - 2;
    }
  }
  else {
    if (index >= 0 && index < range) {
      return index;
    }
    else {
      if (range == 1) {
        index = 0;
      }
      else {
        do {
          if (index < 0)
            index = 0 - index;
          else
            index = (range << 1) - index - 2;
        } while (index >= range || index < 0);
      }

      return index;
    }
  }
}

#endif
