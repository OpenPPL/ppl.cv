#include <benchmark/benchmark.h>
#include "ppl/cv/x86/cvtcolor.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include "ppl/cv/debug.h"

namespace {

enum NV2ColorMode {NV122RGB_MODE, NV122BGR_MODE, NV212RGB_MODE, NV212BGR_MODE};

template<typename T, NV2ColorMode mode>
void BM_NV2BGR_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3 / 2, 0, 255);
    for (auto _ : state) {
        if (mode == NV122RGB_MODE) { 
            ppl::cv::x86::NV122RGB<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        } else if (mode == NV122BGR_MODE) {
            ppl::cv::x86::NV122BGR<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        } else if (mode == NV212RGB_MODE) { 
            ppl::cv::x86::NV212RGB<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        } else if (mode == NV212BGR_MODE) {
            ppl::cv::x86::NV212BGR<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_NV2BGR_ppl_x86, uint8_t, NV122RGB_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_NV2BGR_ppl_x86, uint8_t, NV122BGR_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_NV2BGR_ppl_x86, uint8_t, NV212RGB_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_NV2BGR_ppl_x86, uint8_t, NV212BGR_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV

template<typename T, NV2ColorMode mode>
void BM_NV2BGR_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3 / 2, 0, 255);
    cv::Mat srcMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 3), dst.get());
    for (auto _ : state) {
        if (mode == NV122RGB_MODE) { 
            cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGB_NV12);
        } else if (mode == NV122BGR_MODE) {
            cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGR_NV12);
        } else if (mode == NV212RGB_MODE) { 
            cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGB_NV21);
        } else if (mode == NV212BGR_MODE) {
            cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGR_NV21);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_NV2BGR_opencv_x86, uint8_t, NV122RGB_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_NV2BGR_opencv_x86, uint8_t, NV122BGR_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_NV2BGR_opencv_x86, uint8_t, NV212RGB_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_NV2BGR_opencv_x86, uint8_t, NV212BGR_MODE)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}
