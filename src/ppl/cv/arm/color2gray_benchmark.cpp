#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include <stdlib.h>
#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/debug.h"

namespace {
template <typename T, int32_t input_channels, int32_t output_channels>
class BGR_GRAY_Benchmark {
public:
    T *dev_iImage;
    T *dev_oImage;
    int32_t width;
    int32_t height;

    BGR_GRAY_Benchmark(int32_t width, int32_t height)
        : dev_iImage(nullptr)
        , dev_oImage(nullptr)
        , width(width)
        , height(height)
    {
        dev_iImage = (T *)malloc(sizeof(T) * width * height * input_channels);
        dev_oImage = (T *)malloc(sizeof(T) * width * height * output_channels);
        memset(dev_iImage, 0, sizeof(T) * width * height * input_channels);
        memset(dev_oImage, 0, sizeof(T) * width * height * output_channels);
    }

    ~BGR_GRAY_Benchmark()
    {
        free(dev_iImage);
        free(dev_oImage);
    }

    void BGR2GRAYapply()
    {
        ppl::cv::arm::BGR2GRAY<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void GRAY2BGRapply()
    {
        ppl::cv::arm::GRAY2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGRA2GRAYapply()
    {
        ppl::cv::arm::BGRA2GRAY<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void GRAY2BGRAapply()
    {
        ppl::cv::arm::GRAY2BGRA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
};
} // namespace
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2GRAY_ppl_aarch64(benchmark::State &state)
{
    BGR_GRAY_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2GRAYapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_GRAY2BGR_ppl_aarch64(benchmark::State &state)
{
    BGR_GRAY_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.GRAY2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2GRAY_ppl_aarch64(benchmark::State &state)
{
    BGR_GRAY_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2GRAYapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_GRAY2BGRA_ppl_aarch64(benchmark::State &state)
{
    BGR_GRAY_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.GRAY2BGRAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

constexpr int32_t c1 = 1;
constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

BENCHMARK_TEMPLATE(BM_BGR2GRAY_ppl_aarch64, float, c3, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2GRAY_ppl_aarch64, float, c4, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2GRAY_ppl_aarch64, uint8_t, c3, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2GRAY_ppl_aarch64, uint8_t, c4, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGR_ppl_aarch64, float, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGRA_ppl_aarch64, float, c1, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGR_ppl_aarch64, uint8_t, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGRA_ppl_aarch64, uint8_t, c1, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2GRAY_opencv_aarch64(benchmark::State &state)
{
    cv::setNumThreads(1);
    std::unique_ptr<T[]> src(new T[state.range(0) * state.range(1) * input_channels]);
    memset(src.get(), 0, state.range(0) * state.range(1) * input_channels * sizeof(T));
    std::unique_ptr<T[]> dst(new T[state.range(0) * state.range(1) * output_channels]);
    cv::Mat src_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * state.range(0) * input_channels);
    cv::Mat dst_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst.get(), sizeof(T) * state.range(0) * output_channels);

    for (auto _ : state) {
        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2GRAY);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_GRAY2BGR_opencv_aarch64(benchmark::State &state)
{
    cv::setNumThreads(1);
    std::unique_ptr<T[]> src(new T[state.range(0) * state.range(1) * input_channels]);
    memset(src.get(), 0, state.range(0) * state.range(1) * input_channels * sizeof(T));
    std::unique_ptr<T[]> dst(new T[state.range(0) * state.range(1) * output_channels]);
    cv::Mat src_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * state.range(0) * input_channels);
    cv::Mat dst_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst.get(), sizeof(T) * state.range(0) * output_channels);

    for (auto _ : state) {
        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_GRAY2BGR);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2GRAY_opencv_aarch64(benchmark::State &state)
{
    cv::setNumThreads(1);
    std::unique_ptr<T[]> src(new T[state.range(0) * state.range(1) * input_channels]);
    memset(src.get(), 0, state.range(0) * state.range(1) * input_channels * sizeof(T));
    std::unique_ptr<T[]> dst(new T[state.range(0) * state.range(1) * output_channels]);
    cv::Mat src_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * state.range(0) * input_channels);
    cv::Mat dst_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst.get(), sizeof(T) * state.range(0) * output_channels);

    for (auto _ : state) {
        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2GRAY);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_GRAY2BGRA_opencv_aarch64(benchmark::State &state)
{
    cv::setNumThreads(1);
    std::unique_ptr<T[]> src(new T[state.range(0) * state.range(1) * input_channels]);
    memset(src.get(), 0, state.range(0) * state.range(1) * input_channels * sizeof(T));
    std::unique_ptr<T[]> dst(new T[state.range(0) * state.range(1) * output_channels]);
    cv::Mat src_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * state.range(0) * input_channels);
    cv::Mat dst_opencv(state.range(1), state.range(0), CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst.get(), sizeof(T) * state.range(0) * output_channels);

    for (auto _ : state) {
        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_GRAY2BGRA);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
BENCHMARK_TEMPLATE(BM_BGR2GRAY_opencv_aarch64, float, c3, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2GRAY_opencv_aarch64, float, c4, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2GRAY_opencv_aarch64, uint8_t, c3, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2GRAY_opencv_aarch64, uint8_t, c4, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGR_opencv_aarch64, float, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGRA_opencv_aarch64, float, c1, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGR_opencv_aarch64, uint8_t, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_GRAY2BGRA_opencv_aarch64, uint8_t, c1, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
