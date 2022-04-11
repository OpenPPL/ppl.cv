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

#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"

static void Sscal(
    int32_t n,
    double da,
    double *x,
    int32_t incx)
{
    int32_t i = 0, j = 0;

    while (j < n) {
        if (da == 0.0) {
            x[i] = 0.0;
        } else {
            x[i] = da * x[i];
        }
        i += incx;
        j++;
    }
}

static int32_t Isamax(
    int32_t n,
    double *x,
    int32_t incx)
{
    int32_t i = 0;
    int32_t ix = 0;
    double maxf = 0.0;
    int32_t max = 0;

    maxf = fabsf(x[0]);
    ix += incx;
    i++;

    while (i < n) {
        if (fabsf(x[ix]) > maxf) {
            max = i;
            maxf = fabsf(x[ix]);
        }
        ix += incx;
        i++;
    }
    return max + 1;
}

static void Strmv(
    const char uplo,
    const char trans,
    const char diag,
    int32_t n,
    double *a,
    int32_t lda,
    double *x,
    int32_t incx)
{
    int32_t a_dim1, a_offset, i__1, i__2;

    int32_t i__, j, ix, jx, kx;
    double temp;
    int32_t nounit;

    // Parameter adjustments
    a_dim1 = lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;

    if (n == 0) {
        return;
    }

    nounit = (diag == 'N');

    if (incx <= 0) {
        kx = 1 - (n - 1) * incx;
    } else if (incx != 1) {
        kx = 1;
    }

    // Start the operations. In this version the elements of A are
    // accessed sequentially with one pass through A.
    if (trans == 'N') {
        if (uplo == 'U') {
            if (incx == 1) {
                i__1 = n;
                for (j = 1; j <= i__1; ++j) {
                    if (x[j] != 0.f) {
                        temp = x[j];
                        i__2 = j - 1;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            x[i__] += temp * a[i__ + j * a_dim1];
                        }
                        if (nounit) {
                            x[j] *= a[j + j * a_dim1];
                        }
                    }
                }
            } else {
                jx = kx;
                i__1 = n;
                for (j = 1; j <= i__1; ++j) {
                    if (x[jx] != 0.f) {
                        temp = x[jx];
                        ix = kx;
                        i__2 = j - 1;
                        for (i__ = 1; i__ <= i__2; ++i__) {
                            x[ix] += temp * a[i__ + j * a_dim1];
                            ix += incx;
                        }
                        if (nounit) {
                            x[jx] *= a[j + j * a_dim1];
                        }
                    }
                    jx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (j = n; j >= 1; --j) {
                    if (x[j] != 0.f) {
                        temp = x[j];
                        i__1 = j + 1;
                        for (i__ = n; i__ >= i__1; --i__) {
                            x[i__] += temp * a[i__ + j * a_dim1];
                        }
                        if (nounit) {
                            x[j] *= a[j + j * a_dim1];
                        }
                    }
                }
            } else {
                kx += (n - 1) * incx;
                jx = kx;
                for (j = n; j >= 1; --j) {
                    if (x[jx] != 0.f) {
                        temp = x[jx];
                        ix = kx;
                        i__1 = j + 1;
                        for (i__ = n; i__ >= i__1; --i__) {
                            x[ix] += temp * a[i__ + j * a_dim1];
                            ix -= incx;
                        }
                        if (nounit) {
                            x[jx] *= a[j + j * a_dim1];
                        }
                    }
                    jx -= incx;
                }
            }
        }
    } else {
        if (uplo == 'U') {
            if (incx == 1) {
                for (j = n; j >= 1; --j) {
                    temp = x[j];
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    for (i__ = j - 1; i__ >= 1; --i__) {
                        temp += a[i__ + j * a_dim1] * x[i__];
                    }
                    x[j] = temp;
                }
            } else {
                jx = kx + (n - 1) * incx;
                for (j = n; j >= 1; --j) {
                    temp = x[jx];
                    ix = jx;
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    for (i__ = j - 1; i__ >= 1; --i__) {
                        ix -= incx;
                        temp += a[i__ + j * a_dim1] * x[ix];
                    }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                i__1 = n;
                for (j = 1; j <= i__1; ++j) {
                    temp = x[j];
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    i__2 = n;
                    for (i__ = j + 1; i__ <= i__2; ++i__) {
                        temp += a[i__ + j * a_dim1] * x[i__];
                    }
                    x[j] = temp;
                }
            } else {
                jx = kx;
                i__1 = n;
                for (j = 1; j <= i__1; ++j) {
                    temp = x[jx];
                    ix = jx;
                    if (nounit) {
                        temp *= a[j + j * a_dim1];
                    }
                    i__2 = n;
                    for (i__ = j + 1; i__ <= i__2; ++i__) {
                        ix += incx;
                        temp += a[i__ + j * a_dim1] * x[ix];
                    }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        }
    }
}

static void SgemvColmajor(
    int32_t m,
    int32_t n,
    double alpha,
    const double *matrix,
    int32_t lda,
    const double *v,
    double beta,
    double *r)
{
    for (int32_t i = 0; i < m; i++) {
        double ret = 0.f;
        for (int32_t j = 0; j < n; j++) {
            ret += matrix[j * lda + i] * v[j];
        }
        r[i] = (0 == beta ? 0 : beta * r[i]) + alpha * ret;
    }
}

static ::ppl::common::RetCode SlufactSimple(
    int32_t n,
    double *A,
    int32_t lda,
    int32_t *ipiv)
{
    int32_t i, j, k, jp, ii, jj;
    for (i = 0; i < n; i++) {
        //select max from col
        j = i;
        jp = j + Isamax(n - j, A + j + j * lda, 1) - 1; // Isamax returns 1..n
        ipiv[j] = jp + 1; // ipiv is from 1 to n
        if (A[jp + j * lda] != 0.f) {
            if (j != jp) {
                //swap
                for (k = 0; k < n; k++) {
                    double tmp;
                    tmp = A[j + k * lda];
                    A[j + k * lda] = A[jp + k * lda];
                    A[jp + k * lda] = tmp;
                }
            }
            if (j < n - 1) {
                double recip = 1.f / A[j + j * lda];
                Sscal(n - j - 1, recip, A + j + 1 + j * lda, 1);
            }
        } else {
            // a[j, j] is 0, error!
            return ppl::common::RC_INVALID_VALUE;
        }
        if (j < n - 1) {
            //trailing matrix update A = A - x*y', x is col, y is row
            for (ii = j + 1; ii < n; ii++) {
                for (jj = j + 1; jj < n; jj++) {
                    A[jj + ii * lda] = A[jj + ii * lda] - A[jj + j * lda] * A[j + ii * lda];
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

// LU decomposition
static ::ppl::common::RetCode LUDecomp(
    int32_t m,
    int32_t n,
    double *A,
    int32_t lda,
    double *X,
    int32_t incx,
    double *B,
    int32_t incb)
{
    // the layout of A, B is column major and no trans
    int32_t i, j;
    double *lu;
    int32_t *ipiv;

    //use tmp buffer in factor
    lu = (double *)ppl::common::AlignedAlloc(n * n * sizeof(double) + n * sizeof(int32_t), 128);
    ipiv = (int32_t *)(lu + n * n);

    //lu is col major no trans, copy A into it first
    for (i = 0; i < n; i++) {
        memcpy(lu + i * n, A + i * lda, n * sizeof(double));
    }
    // small case, not use BLAS3 & block

    ppl::common::RetCode slufact_rc = SlufactSimple(n, lu, n, ipiv);
    if (slufact_rc != ppl::common::RC_SUCCESS) {
        ppl::common::AlignedFree(lu);
        return slufact_rc;
    }

    // backward Solve A * X = B.
    // now we simply use for loop to do it...no func call
    // X copy from B
    memcpy(X, B, n * sizeof(double));
    // note : ipiv is in 1..n i is in 0..n-1
    for (i = 0; i < n; i++) {
        double tmp;
        if (ipiv[i] != (i + 1)) {
            tmp = X[i];
            X[i] = X[ipiv[i] - 1];
            X[ipiv[i] - 1] = tmp;
        }
    }
    for (i = 0; i < n; i++) {
        double temp = X[i];
        for (j = i + 1; j < n; j++) {
            X[j] -= temp * lu[j + i * n];
        }
    } // L
    for (i = n - 1; i >= 0; i--) {
        double temp;
        X[i] /= lu[i + i * n];
        temp = X[i];
        for (j = i - 1; j >= 0; j--) {
            X[j] -= temp * lu[j + i * n];
        }
    } // U
    ppl::common::AlignedFree(lu);
    return ppl::common::RC_SUCCESS;
}

static ::ppl::common::RetCode InverseMatLU(
    int32_t n,
    double *A,
    int32_t lda,
    double *inva,
    int32_t ldinv)
{
    double *lu;
    int32_t *ipiv;
    int32_t i, j, jp;
    double temp;
    double *work;

    //use tmp buffer in factor
    lu = (double *)ppl::common::AlignedAlloc((n * n + n) * sizeof(double) + n * sizeof(int32_t), 128);
    ipiv = (int32_t *)(lu + n * n);
    work = (double *)(ipiv + n);
    //trans row major to col major
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            lu[i * n + j] = A[j * lda + i];
        }
    }
    // small case, not use BLAS3 & block
    ppl::common::RetCode slufact_rc = SlufactSimple(n, lu, n, ipiv);
    if (slufact_rc != ppl::common::RC_SUCCESS) {
        ppl::common::AlignedFree(lu);
        return slufact_rc;
    }

    //form U^-1
    for (i = 0; i < n; i++) {
        lu[i + i * n] = 1.f / lu[i + i * n];
        temp = -lu[i + i * n];
        //Strmv
        Strmv('U', 'N', 'N', i, lu, n, lu + i * n, 1);
        //scal
        for (j = 0; j < i; j++) {
            lu[i * n + j] *= temp;
        }
    }

    //solve inv(A)*L=inv(U)
    for (i = n - 1; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {
            work[j] = lu[i * n + j];
            lu[i * n + j] = 0.f;
        }

        //sgemv
        if (i < n - 1) {
            SgemvColmajor(n, n - 1 - i, -1.0f, lu + (i + 1) * n, n, work + i + 1, 1.f, lu + i * n);
        }
    }

    // apply col swap, inverse order
    for (i = n - 2; i >= 0; i--) {
        jp = ipiv[i] - 1;
        if (jp != i) {
            //swap
            for (j = 0; j < n; j++) {
                double tmp;
                tmp = lu[j + i * n];
                lu[j + i * n] = lu[j + jp * n];
                lu[j + jp * n] = tmp;
            }
        }
    }
    //copy lu int32_to inva
    // trans col major to row major
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            inva[i * n + j] = lu[j * n + i];
        }
    }
    ppl::common::AlignedFree(lu);
    return ppl::common::RC_SUCCESS;
}

namespace ppl {
namespace cv {
namespace riscv {

::ppl::common::RetCode GetAffineTransform(
    const double *src_points,
    const double *dst_points,
    double *mat,
    double *inverse_mat)
{
    ppl::common::RetCode rc;

    double A[36]; // 6x6 mat
    double X[6]; // 1x6
    double tmp[9]; // store temp results
    double inverse_tmp[9];

    for (int32_t i = 0; i < 36; i++) {
        A[i] = 0;
    }
    // A:
    // x0, x1, x2, 0,  0,  0
    // y0, y1, y2, 0,  0,  0
    // 1,  1,  1,  0,  0,  0
    // 0,  0,  0,  x0, x1, x2
    // 0,  0,  0,  y0, y1, y2
    // 0,  0,  0,  1,  1,  1
    for (int32_t i = 0; i < 3; i++) {
        A[i] = src_points[i * 2 + 0];
        A[6 + i] = src_points[i * 2 + 1];
        A[12 + i] = 1;
        A[18 + i + 3] = src_points[i * 2 + 0];
        A[24 + i + 3] = src_points[i * 2 + 1];
        A[30 + i + 3] = 1;
        X[i] = dst_points[i * 2 + 0];
        X[3 + i] = dst_points[i * 2 + 1];
    }
    rc = LUDecomp(6, 6, A, 6, tmp, 1, X, 1);
    if (rc != ppl::common::RC_SUCCESS) {
        return rc;
    }

    if (mat != nullptr) {
        mat[0] = tmp[0];
        mat[1] = tmp[1];
        mat[2] = tmp[2];
        mat[3] = tmp[3];
        mat[4] = tmp[4];
        mat[5] = tmp[5];
    }

    if (inverse_mat != nullptr) {
        tmp[6] = 0;
        tmp[7] = 0;
        tmp[8] = 1;

        // inverse mat
        rc = InverseMatLU(3, tmp, 3, inverse_tmp, 3);
        if (rc != ppl::common::RC_SUCCESS) {
            return rc;
        }

        inverse_mat[0] = inverse_tmp[0];
        inverse_mat[1] = inverse_tmp[1];
        inverse_mat[2] = inverse_tmp[2];
        inverse_mat[3] = inverse_tmp[3];
        inverse_mat[4] = inverse_tmp[4];
        inverse_mat[5] = inverse_tmp[5];
    }

    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::riscv
