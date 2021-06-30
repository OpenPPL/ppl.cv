#include "ppl/cv/x86/get_affine_transform.h"
#include "ppl/cv/debug.h"
#include <memory>
#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>

namespace {

void BM_getAffineTransform_ppl_x86(benchmark::State &state) {
    double src[6], dst[6];
    double mat[6];
    double inverse_mat[6];
    src[0] = 5;
    src[1] = 9;
    src[2] = 223;
    src[3] = 13;
    src[4] = 49;
    src[5] = 146;

    dst[0] = 27;
    dst[1] = 19;
    dst[2] = 103;
    dst[3] = 47;
    dst[4] = 18;
    dst[5] = 91;
    for (auto _ : state) {
        ppl::cv::x86::GetAffineTransform(src, dst, mat, inverse_mat);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK(BM_getAffineTransform_ppl_x86);

#ifdef PPLCV_BENCHMARK_OPENCV
void BM_getAffineTransform_opencv_x86(benchmark::State &state) {
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];
    srcTri[0] = cv::Point2f(5, 9);
    srcTri[1] = cv::Point2f(223, 13);
    srcTri[2] = cv::Point2f(49, 146);

    dstTri[0] = cv::Point2f(27, 19);
    dstTri[1] = cv::Point2f(103, 47);
    dstTri[2] = cv::Point2f(18, 91);
    for (auto _ : state) {
        cv::getAffineTransform(srcTri, dstTri);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK(BM_getAffineTransform_opencv_x86);
#endif //! PPLCV_BENCHMARK_OPENCV
}
