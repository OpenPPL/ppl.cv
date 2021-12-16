// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/cv/x86/invert.h"
#include "ppl/cv/types.h"
#include <string.h>
#include <cmath>
#include <limits.h>
#include <type_traits>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

#define det2(m) ((double)m(0, 0) * m(1, 1) - (double)m(0, 1) * m(1, 0))
#define det3(m) (m(0, 0) * ((double)m(1, 1) * m(2, 2) - (double)m(1, 2) * m(2, 1)) - \
                 m(0, 1) * ((double)m(1, 0) * m(2, 2) - (double)m(1, 2) * m(2, 0)) + \
                 m(0, 2) * ((double)m(1, 0) * m(2, 1) - (double)m(1, 1) * m(2, 0)))

template <typename _Tp>
static inline bool CholImpl(_Tp *A, int32_t astep, int32_t m, _Tp *b, int32_t bstep, int32_t n)
{
    _Tp *L = A;
    int32_t i, j, k;
    double s;
    astep /= sizeof(A[0]);
    bstep /= sizeof(b[0]);

    for (i = 0; i < m; i++) {
        for (j = 0; j < i; j++) {
            s = A[i * astep + j];
            for (k = 0; k < j; k++)
                s -= L[i * astep + k] * L[j * astep + k];
            L[i * astep + j] = (_Tp)(s * L[j * astep + j]);
        }
        s = A[i * astep + i];
        for (k = 0; k < j; k++) {
            double t = L[i * astep + k];
            s -= t * t;
        }
        if (s < std::numeric_limits<_Tp>::epsilon())
            return false;
        L[i * astep + i] = (_Tp)(1. / std::sqrt(s));
    }

    if (!b)
        return true;
    // LLt x = b
    // 1: L y = b
    // 2. Lt x = y

    /*
         [ L00             ]  y0   b0
         [ L10 L11         ]  y1 = b1
         [ L20 L21 L22     ]  y2   b2
         [ L30 L31 L32 L33 ]  y3   b3

         [ L00 L10 L20 L30 ]  x0   y0
         [     L11 L21 L31 ]  x1 = y1
         [         L22 L32 ]  x2   y2
         [             L33 ]  x3   y3
     */

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            s = b[i * bstep + j];
            for (k = 0; k < i; k++)
                s -= L[i * astep + k] * b[k * bstep + j];
            b[i * bstep + j] = (_Tp)(s * L[i * astep + i]);
        }
    }

    for (i = m - 1; i >= 0; i--) {
        for (j = 0; j < n; j++) {
            s = b[i * bstep + j];
            for (k = m - 1; k > i; k--)
                s -= L[k * astep + i] * b[k * bstep + j];
            b[i * bstep + j] = (_Tp)(s * L[i * astep + i]);
        }
    }

    return true;
}

#define Sf(y, x) ((float *)(srcdata + y * srcstep))[x]
#define Sd(y, x) ((double *)(srcdata + y * srcstep))[x]
#define Df(y, x) ((float *)(dstdata + y * dststep))[x]
#define Dd(y, x) ((double *)(dstdata + y * dststep))[x]

template <typename T>
::ppl::common::RetCode Invert(
    int32_t m,
    int32_t n,
    int32_t inWidthStride,
    const T *src,
    int32_t outWidthStride,
    T *dst,
    InvertMethod decompTypes)
{
    if(m != n) {
        return ppl::common::RC_INVALID_VALUE;
    }

    bool result     = false;
    int32_t srcstep = inWidthStride * sizeof(T);
    int32_t dststep = outWidthStride * sizeof(T);
    if (n <= 3) {
        const uint8_t *srcdata = reinterpret_cast<const uint8_t *>(src);
        uint8_t *dstdata       = reinterpret_cast<uint8_t *>(dst);

        if (n == 2) {
            if (std::is_same<float, T>::value) {
                double d = det2(Sf);
                if (d != 0.) {
                    result = true;
                    d      = 1. / d;

                    double t0, t1;
                    t0       = Sf(0, 0) * d;
                    t1       = Sf(1, 1) * d;
                    Df(1, 1) = (float)t0;
                    Df(0, 0) = (float)t1;
                    t0       = -Sf(0, 1) * d;
                    t1       = -Sf(1, 0) * d;
                    Df(0, 1) = (float)t0;
                    Df(1, 0) = (float)t1;
                }
            } else if (std::is_same<double, T>::value) {
                double d = det2(Sd);
                if (d != 0.) {
                    result = true;
                    d      = 1. / d;
                    double t0, t1;
                    t0       = Sd(0, 0) * d;
                    t1       = Sd(1, 1) * d;
                    Dd(1, 1) = t0;
                    Dd(0, 0) = t1;
                    t0       = -Sd(0, 1) * d;
                    t1       = -Sd(1, 0) * d;
                    Dd(0, 1) = t0;
                    Dd(1, 0) = t1;
                }
            }
        } else if (n == 3) {
            if (std::is_same<float, T>::value) {
                double d = det3(Sf);

                if (d != 0.) {
                    double t[12];

                    result = true;
                    d      = 1. / d;
                    t[0]   = (((double)Sf(1, 1) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 1)) * d);
                    t[1]   = (((double)Sf(0, 2) * Sf(2, 1) - (double)Sf(0, 1) * Sf(2, 2)) * d);
                    t[2]   = (((double)Sf(0, 1) * Sf(1, 2) - (double)Sf(0, 2) * Sf(1, 1)) * d);

                    t[3] = (((double)Sf(1, 2) * Sf(2, 0) - (double)Sf(1, 0) * Sf(2, 2)) * d);
                    t[4] = (((double)Sf(0, 0) * Sf(2, 2) - (double)Sf(0, 2) * Sf(2, 0)) * d);
                    t[5] = (((double)Sf(0, 2) * Sf(1, 0) - (double)Sf(0, 0) * Sf(1, 2)) * d);

                    t[6] = (((double)Sf(1, 0) * Sf(2, 1) - (double)Sf(1, 1) * Sf(2, 0)) * d);
                    t[7] = (((double)Sf(0, 1) * Sf(2, 0) - (double)Sf(0, 0) * Sf(2, 1)) * d);
                    t[8] = (((double)Sf(0, 0) * Sf(1, 1) - (double)Sf(0, 1) * Sf(1, 0)) * d);

                    Df(0, 0) = (float)t[0];
                    Df(0, 1) = (float)t[1];
                    Df(0, 2) = (float)t[2];
                    Df(1, 0) = (float)t[3];
                    Df(1, 1) = (float)t[4];
                    Df(1, 2) = (float)t[5];
                    Df(2, 0) = (float)t[6];
                    Df(2, 1) = (float)t[7];
                    Df(2, 2) = (float)t[8];
                }
            } else if (std::is_same<double, T>::value) {
                double d = det3(Sd);
                if (d != 0.) {
                    result = true;
                    d      = 1. / d;
                    double t[9];

                    t[0] = (Sd(1, 1) * Sd(2, 2) - Sd(1, 2) * Sd(2, 1)) * d;
                    t[1] = (Sd(0, 2) * Sd(2, 1) - Sd(0, 1) * Sd(2, 2)) * d;
                    t[2] = (Sd(0, 1) * Sd(1, 2) - Sd(0, 2) * Sd(1, 1)) * d;

                    t[3] = (Sd(1, 2) * Sd(2, 0) - Sd(1, 0) * Sd(2, 2)) * d;
                    t[4] = (Sd(0, 0) * Sd(2, 2) - Sd(0, 2) * Sd(2, 0)) * d;
                    t[5] = (Sd(0, 2) * Sd(1, 0) - Sd(0, 0) * Sd(1, 2)) * d;

                    t[6] = (Sd(1, 0) * Sd(2, 1) - Sd(1, 1) * Sd(2, 0)) * d;
                    t[7] = (Sd(0, 1) * Sd(2, 0) - Sd(0, 0) * Sd(2, 1)) * d;
                    t[8] = (Sd(0, 0) * Sd(1, 1) - Sd(0, 1) * Sd(1, 0)) * d;

                    Dd(0, 0) = t[0];
                    Dd(0, 1) = t[1];
                    Dd(0, 2) = t[2];
                    Dd(1, 0) = t[3];
                    Dd(1, 1) = t[4];
                    Dd(1, 2) = t[5];
                    Dd(2, 0) = t[6];
                    Dd(2, 1) = t[7];
                    Dd(2, 2) = t[8];
                }
            }
        } else {
            if(n != 1) {
                return ppl::common::RC_INVALID_VALUE;
            }

            if (std::is_same<float, T>::value) {
                double d = Sf(0, 0);
                if (d != 0.) {
                    result   = true;
                    Df(0, 0) = (float)(1. / d);
                }
            } else if (std::is_same<double, T>::value) {
                double d = Sd(0, 0);
                if (d != 0.) {
                    result   = true;
                    Dd(0, 0) = 1. / d;
                }
            }
        }
        if (!result) {
            for (int32_t i = 0; i < m; i++) {
                memset(dstdata + i * dststep, 0, n * sizeof(T));
            }
        }
        return ppl::common::RC_SUCCESS;
    }

    T *src1 = new T[n * n];
    for (int32_t i = 0; i < n; i++) {
        memcpy(src1 + i * n, src + i * inWidthStride, n * sizeof(T));
    }

    //set identity
    for (int32_t i = 0; i < m; i++) {
        memset(dst + i * outWidthStride, 0, n * sizeof(T));
    }
    for (int32_t i = 0; i < m; i++) {
        dst[i * outWidthStride + i] = 1;
    }

    result = CholImpl<T>(src1, n * sizeof(T), n, dst, dststep, n);

    delete[] src1;
    if (!result) {
        for (int32_t i = 0; i < m; i++) {
            memset(dst + i * outWidthStride, 0, n * sizeof(T));
        }
    }

    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Invert<float>(
    int32_t m,
    int32_t n,
    int32_t inWidthStride,
    const float *src,
    int32_t outWidthStride,
    float *dst,
    InvertMethod decompTypes);
template ::ppl::common::RetCode Invert<double>(
    int32_t m,
    int32_t n,
    int32_t inWidthStride,
    const double *src,
    int32_t outWidthStride,
    double *dst,
    InvertMethod decompTypes);
}
}
} // namespace ppl::cv::x86
