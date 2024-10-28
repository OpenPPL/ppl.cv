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

enum InterpolationType {
    INTERPOLATION_LINEAR,       //!< Linear interpolation
    INTERPOLATION_NEAREST_POINT, //!< Nearest point interpolation
    INTERPOLATION_AREA //!< Area interpolation
};

enum AdaptiveThresholdTypes {
    /** the threshold value \f$T(x,y)\f$ is a mean of the \f$\texttt{blockSize} \times
    \texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$ minus C */
    ADAPTIVE_THRESH_MEAN_C     = 0,
    /** the threshold value \f$T(x, y)\f$ is a weighted sum (cross-correlation with a Gaussian
    window) of the \f$\texttt{blockSize} \times \texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$
    minus C . The default sigma (standard deviation) is used for the specified blockSize . See
    #getGaussianKernel*/
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
};

enum {
    CV_THRESH_BINARY      =0,  /**< value = value > threshold ? max_value : 0       */
    CV_THRESH_BINARY_INV  =1,  /**< value = value > threshold ? 0 : max_value       */
    CV_THRESH_TRUNC       =2,  /**< value = value > threshold ? threshold : value   */
    CV_THRESH_TOZERO      =3,  /**< value = value > threshold ? value : 0           */
    CV_THRESH_TOZERO_INV  =4,  /**< value = value > threshold ? 0 : value           */
    CV_THRESH_MASK        =7,
    CV_THRESH_OTSU        =8, /**< use Otsu algorithm to choose the optimal threshold value;
                                 combine the flag with one of the above CV_THRESH_* values */
    CV_THRESH_TRIANGLE    =16  /**< use Triangle algorithm to choose the optimal threshold value;
                                 combine the flag with one of the above CV_THRESH_* values, but not
                                 with CV_THRESH_OTSU */
};

enum ThresholdTypes {
    THRESH_BINARY     = 0,
    THRESH_BINARY_INV = 1
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
