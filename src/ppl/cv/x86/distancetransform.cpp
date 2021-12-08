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

#include "ppl/cv/x86/distancetransform.h"
#include "ppl/cv/types.h"
#include <string.h>
#include <float.h>
#include <limits>
#include <cmath>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {
template <typename T>
inline T abs(T v)
{
    return v < T() ? (-v) : v;
}

template <>
inline float abs<float>(float v)
{
    return fabsf(v);
}

template <>
inline double abs<double>(double v)
{
    return fabs(v);
}

template <typename _Tp>
static inline _Tp* align_Ptr(_Tp* ptr, int32_t n = (int32_t)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

static void trueDistTrans_float(
    int32_t height,
    int32_t width,
    int32_t inStrid,
    const uint8_t* ptrIn,
    int32_t outStride,
    float* ptrOut)
{
    const float inf = 1e15f;

    int32_t i, m = height, n = width;

    float* sqr_tab   = new float[(std::max(m * 2 * sizeof(float) + (m * 3 + 1) * sizeof(int32_t), n * 2 * sizeof(float))) >> 2];
    // stage 1: compute 1d distance transform of each column
    int32_t* sat_tab = align_Ptr((int32_t*)(sqr_tab + m * 2), sizeof(int32_t));
    int32_t shift    = m * 2;

    for (i = 0; i < m; i++)
        sqr_tab[i] = (float)(i * i);
    for (i = m; i < m * 2; i++)
        sqr_tab[i] = inf;
    for (i = 0; i < shift; i++)
        sat_tab[i] = 0;
    for (; i <= m * 3; i++)
        sat_tab[i] = i - shift;

    size_t sstep = inStrid, dstep = outStride;

    {
        int32_t i;
        sat_tab    = sat_tab + height * 2 + 1;
        int32_t* d = new int32_t[m];

        for (i = 0; i < n; i++) {
            const uint8_t* sptr = (uint8_t*)ptrIn + (m - 1) * inStrid + i;
            float* dptr         = (float*)ptrOut + i;
            ;
            int32_t j, dist = m - 1;

            for (j = m - 1; j >= 0; j--, sptr -= sstep) {
                dist = (dist + 1) & (sptr[0] == 0 ? 0 : -1);
                d[j] = dist;
            }

            dist = m - 1;
            for (j = 0; j < m; j++, dptr += dstep) {
                dist    = dist + 1 - sat_tab[dist - d[j]];
                d[j]    = dist;
                dptr[0] = sqr_tab[dist];
            }
        }
        delete[] d;
    }

    // stage 2: compute modified distance transform for each row
    float* inv_tab = sqr_tab + n;

    inv_tab[0] = sqr_tab[0] = 0.f;
    for (i = 1; i < n; i++) {
        inv_tab[i] = (float)(0.5 / i);
        sqr_tab[i] = (float)(i * i);
    }

    {
        const float inf = 1e15f;
        int32_t n       = width;
        float* f        = new float[((n + 2) * 2 * sizeof(float) + (n + 2) * sizeof(int32_t)) >> 2];

        float* z = f + n;

        int32_t* v = align_Ptr((int32_t*)(z + n + 1), sizeof(int32_t));

        for (int32_t i = 0; i < m; i++) {
            float* d = (float*)ptrOut + i * dstep;
            int32_t p, q, k;

            v[0] = 0;
            z[0] = -inf;
            z[1] = inf;
            f[0] = d[0];

            for (q = 1, k = 0; q < n; q++) {
                float fq = d[q];
                f[q]     = fq;

                for (;; k--) {
                    p       = v[k];
                    float s = (fq + sqr_tab[q] - d[p] - sqr_tab[p]) * inv_tab[q - p];
                    if (s > z[k]) {
                        k++;
                        v[k]     = q;
                        z[k]     = s;
                        z[k + 1] = inf;
                        break;
                    }
                }
            }

            for (q = 0, k = 0; q < n; q++) {
                while (z[k + 1] < q)
                    k++;
                p    = v[k];
                d[q] = std::sqrt(sqr_tab[abs(q - p)] + f[p]);
            }
        }
        delete[] f;
    }
    delete[] sqr_tab;
}

template <>
::ppl::common::RetCode DistanceTransform<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    float* outData,
    DistTypes distanceType,
    DistanceTransformMasks maskSize)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (distanceType != DIST_L2 || maskSize != DIST_MASK_PRECISE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    trueDistTrans_float(height, width, inWidthStride, inData, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
