#include <benchmark/benchmark.h>

#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/debug.h"

namespace {

template<typename T>
void BM_BGR2GRAY_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * 3]);
    std::unique_ptr<T[]> dst(new T[width * height]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * 3, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::BGR2GRAY<T>(height, width, width * 3, src.get(), width, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_BGR2GRAY_ppl_x86, float)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2GRAY_ppl_x86, uint8_t)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T>
void BM_BGR2GRAY_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * 3]);
    std::unique_ptr<T[]> dst(new T[width * height]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * 3, 0, 255);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 3), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), dst.get());
    for (auto _ : state) {
        cv::cvtColor(srcMat, dstMat, cv::COLOR_BGR2GRAY);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_BGR2GRAY_opencv_x86, float)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2GRAY_opencv_x86, uint8_t)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}
