/**
 * @file   warpaffine_benchmark.cpp
 * @brief  benchmark suites for applying an affine transformation to an image.
 * @author Liheng Jian(jianliheng@sensetime.com)
 *
 * @copyright Copyright (c) 2014-2021 SenseTime Group Limited.
 */

#include "warpaffine.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

enum InterpolationTypes {
  kInterLinear,
  kInterNearest,
};

template <typename T, int channels, InterpolationTypes inter_type,
          BorderType border_type>
void BM_WarpAffine_ppl_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::Mat M = createSourceImage(2, 3, CV_32FC1);

  int iterations = 3000;
  struct timeval start, end;

  // warp up the GPU
  for (int i = 0; i < iterations; i++) {
    WarpAffineNearestPoint<T, channels>(0, src.rows, src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, (float*)M.data,
        border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (inter_type == kInterLinear) {
        WarpAffineLinear<T, channels>(0, src.rows, src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data, (float*)M.data,
            border_type);
      }
      else if (inter_type == kInterNearest) {
        WarpAffineNearestPoint<T, channels>(0, src.rows, src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data, (float*)M.data,
            border_type);
      }
      else {
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, InterpolationTypes inter_type,
          BorderType border_type>
void BM_WarpAffine_opencv_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::Mat M = createSourceImage(2, 3, CV_32FC1);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_TYPE_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_TYPE_TRANSPARENT) {
    cv_border = cv::BORDER_TRANSPARENT;
  }
  else {
  }

  int iterations = 3000;
  struct timeval start, end;

  // warp up the GPU
  for (int i = 0; i < iterations; i++) {
    cv::cuda::warpAffine(gpu_src, gpu_dst, M, cv::Size(dst_width, dst_height),
                         cv::WARP_INVERSE_MAP | cv::INTER_LINEAR, cv_border);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (inter_type == kInterLinear) {
        cv::cuda::warpAffine(gpu_src, gpu_dst, M,
                             cv::Size(dst_width, dst_height),
                             cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                             cv_border);
      }
      else if (inter_type == kInterNearest) {
        cv::cuda::warpAffine(gpu_src, gpu_dst, M,
                             cv::Size(dst_width, dst_height),
                             cv::WARP_INVERSE_MAP | cv::INTER_NEAREST,
                             cv_border);
      }
      else {
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, InterpolationTypes inter_type,
          BorderType border_type>
void BM_WarpAffine_opencv_x86_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::Mat M = createSourceImage(2, 3, CV_32FC1);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_TYPE_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_TYPE_TRANSPARENT) {
    cv_border = cv::BORDER_TRANSPARENT;
  }
  else {
  }

  for (auto _ : state) {
    if (inter_type == kInterLinear) {
      cv::warpAffine(src, dst, M, cv::Size(dst_width, dst_height),
                     cv::WARP_INVERSE_MAP | cv::INTER_LINEAR, cv_border);
    }
    else if (inter_type == kInterNearest) {
      cv::warpAffine(src, dst, M, cv::Size(dst_width, dst_height),
                     cv::WARP_INVERSE_MAP | cv::INTER_NEAREST, cv_border);
    }
    else {
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, inter_type, src_width, src_height, dst_width,  \
                      dst_height)                                              \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, channels,             \
                   inter_type, BORDER_TYPE_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, channels,                 \
                   inter_type, BORDER_TYPE_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, channels,                    \
                   inter_type, BORDER_TYPE_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, channels,             \
                   inter_type, BORDER_TYPE_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, channels,                 \
                   inter_type, BORDER_TYPE_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, channels,                    \
                   inter_type, BORDER_TYPE_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, channels,             \
                   inter_type, BORDER_TYPE_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, channels,                 \
                   inter_type, BORDER_TYPE_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, channels,                    \
                   inter_type, BORDER_TYPE_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, channels,             \
                   inter_type, BORDER_TYPE_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, channels,                 \
                   inter_type, BORDER_TYPE_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, channels,                    \
                   inter_type, BORDER_TYPE_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK(c1, kInterLinear, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, kInterLinear, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, kInterLinear, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, kInterLinear, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, kInterLinear, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, kInterLinear, 640, 480, 1280, 960)

// RUN_BENCHMARK(c1, kInterNearest, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, kInterNearest, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, kInterNearest, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, kInterNearest, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, kInterNearest, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, kInterNearest, 640, 480, 1280, 960)

#define RUN_OPENCV_X86_TYPE_FUNCTIONS(inter_type, border_type)                 \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c1, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c1, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c3, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c3, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c4, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c4, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c1, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c1, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c3, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c3, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c4, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c4, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});

#define RUN_OPENCV_CUDA_TYPE_FUNCTIONS(inter_type, border_type)                \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c1, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c1, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c3, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c3, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c4, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c4, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c1, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c1, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c3, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c3, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c4, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c4, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(inter_type, border_type)                \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c1, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c1, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c3, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c3, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c4, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c4, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c1, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c1, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c3, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c3, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c4, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c4, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);

// RUN_OPENCV_X86_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_CONSTANT)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_REPLICATE)
RUN_OPENCV_X86_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_TRANSPARENT)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_CONSTANT)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_REPLICATE)
RUN_OPENCV_X86_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_TRANSPARENT)

RUN_OPENCV_CUDA_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_CONSTANT)
RUN_OPENCV_CUDA_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_REPLICATE)
RUN_OPENCV_CUDA_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_CONSTANT)
RUN_OPENCV_CUDA_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_REPLICATE)

RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(kInterLinear, BORDER_TYPE_TRANSPARENT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(kInterNearest, BORDER_TYPE_TRANSPARENT)
