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

#include "ppl/cv/x86/imread.h"

#include <time.h>
#include <sys/time.h>

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "ppl/cv/x86/imgcodecs/byteswriter.h"
#include "ppl/cv/x86/imgcodecs/bmp.h"
#include "ppl/cv/x86/imgcodecs/jpeg.h"
#include "ppl/cv/cuda/utility/infrastructure.hpp"

using namespace ppl::cv::debug;

struct EncodeType {
    int channels;
    int bits_per_pixel;
    ppl::cv::x86::BmpCompression compression;
};

static const char* bmp_signature = "BM";

void fillGrayPalette(ppl::cv::x86::PaletteEntry* palette, int bits_per_pixel,
                     int channels) {
    int length = 1 << bits_per_pixel;

    for (int i = 0; i < length; i++) {
        int value = i * 255 / (length - 1);
        if (channels == 1) {
            palette[i].b = (uchar)value;
            palette[i].g = (uchar)value;
            palette[i].r = (uchar)value;
        }
        else {
            palette[i].b = (uchar)value;
            palette[i].g = (uchar)((value + 64) & 0xff);
            palette[i].r = (uchar)((value + 128) & 0xff);
        }
        palette[i].a = 0;
    }
}

/***************************** Bmp benchmark *****************************/

bool createBmpTestFile(cv::Mat& image, EncodeType& encode_type, FILE* fp) {
    int file_step = ((image.cols * (encode_type.bits_per_pixel != 15 ?
                      encode_type.bits_per_pixel : 16) + 31) >> 5) << 2;
    uchar padding[] = "\0\0\0\0";
    ppl::cv::x86::BytesWriter bytes_writer(fp);

    int header_size = 14;
    int info_size = 40;
    int used_colors = encode_type.bits_per_pixel <= 8 ?
                      (1 << encode_type.bits_per_pixel) : 0;
    int palette_size = used_colors * 4;
    int mask_size = encode_type.compression == ppl::cv::x86::BMP_BITFIELDS ?
                                               16 : 0;
    int headers_size = header_size + info_size + palette_size + mask_size;
    size_t file_size = headers_size + (size_t)file_step * image.rows;
    ppl::cv::x86::PaletteEntry palette[256];

    // write bitmap file header.
    bytes_writer.putBytes(bmp_signature, strlen(bmp_signature));
    bytes_writer.putDWord(file_size);
    bytes_writer.putDWord(0);
    bytes_writer.putDWord(headers_size);

    // write bitmap information header.
    bytes_writer.putDWord(info_size);
    bytes_writer.putDWord(image.cols);
    bytes_writer.putDWord(image.rows);
    bytes_writer.putWord(1);
    bytes_writer.putWord(encode_type.bits_per_pixel == 15 ? 16 :
                         encode_type.bits_per_pixel);
    bytes_writer.putDWord(encode_type.compression);
    bytes_writer.putDWord(0);
    bytes_writer.putDWord(0);
    bytes_writer.putDWord(0);
    bytes_writer.putDWord(used_colors);
    bytes_writer.putDWord(0);

    // write color palette.
    if (encode_type.bits_per_pixel <= 8) {
        fillGrayPalette(palette, encode_type.bits_per_pixel,
                        encode_type.channels);
        bytes_writer.putBytes(palette, sizeof(palette));
    }

    // write bit mask.
    if (encode_type.compression == ppl::cv::x86::BMP_BITFIELDS) {
        if (encode_type.bits_per_pixel == 15) {
            bytes_writer.putDWord(0x7c00);
            bytes_writer.putDWord(0x3e0);
            bytes_writer.putDWord(0x1f);
            bytes_writer.putDWord(0x00);
        }
        else if (encode_type.bits_per_pixel == 16) {
            bytes_writer.putDWord(0xf800);
            bytes_writer.putDWord(0x7e0);
            bytes_writer.putDWord(0x1f);
            bytes_writer.putDWord(0x00);
        }
        else {
            return false;
        }
    }

    // write pixel data.
    int stride = 0;
    switch (encode_type.bits_per_pixel) {
        case 1:
            stride = (image.cols + 7) >> 3;
            for (int row = image.rows - 1; row >= 0; row--) {
                bytes_writer.putBytes(image.ptr(row), stride);
                if (file_step > stride) {
                    bytes_writer.putBytes(padding, file_step - stride);
                }
            }
            break;
        case 4:
            stride = (image.cols + 1) >> 1;
            for (int row = image.rows - 1; row >= 0; row--) {
                bytes_writer.putBytes(image.ptr(row), stride);
                if (file_step > stride) {
                    bytes_writer.putBytes(padding, file_step - stride);
                }
            }
            break;
        case 8:
            stride = image.cols;
            for (int row = image.rows - 1; row >= 0; row--) {
                bytes_writer.putBytes(image.ptr(row), stride);
                if (file_step > stride) {
                    bytes_writer.putBytes(padding, file_step - stride);
                }
            }
            break;
        case 15:
        case 16:
            stride = image.cols * 2;
            for (int row = image.rows - 1; row >= 0; row--) {
                bytes_writer.putBytes(image.ptr(row), stride);
                if (file_step > stride) {
                    bytes_writer.putBytes(padding, file_step - stride);
                }
            }
            break;
        case 24:
            for (int row = image.rows - 1; row >= 0; row--) {
                bytes_writer.putBytes(image.ptr(row), file_step );
            }
            break;
        case 32:
            for (int row = image.rows - 1; row >= 0; row--) {
                bytes_writer.putBytes(image.ptr(row), file_step );
            }
            break;
        default:
            return false;
    }
    bytes_writer.writeBlock();

    return true;
}

template <int channels, int bits_per_pixel,
          ppl::cv::x86::BmpCompression compression>
void BM_ImreadBmp_ppl_x86(benchmark::State &state) {
    int width  = state.range(0);
    int height = state.range(1);

    int src_channels = 0;
    if (bits_per_pixel <= 8) {
        src_channels = 1;
    }
    else if (bits_per_pixel == 15 || bits_per_pixel == 16) {
        src_channels = 2;
    }
    else if (bits_per_pixel == 24 || bits_per_pixel == 32) {
        src_channels = channels;
    }
    else {
    }
    cv::Mat src, gray_image;
    if (src_channels == 1) {
        if (bits_per_pixel <= 8) {
            src = createSourceImage(height, width,
                                    CV_MAKETYPE(cv::DataType<uchar>::depth,
                                    src_channels));
        }
        else {
            gray_image = createSourceImage(height, width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
            cv::cvtColor(gray_image, src, cv::COLOR_GRAY2BGR);
        }
    }
    else {
        src = createSourceImage(height, width,
                                CV_MAKETYPE(cv::DataType<uchar>::depth,
                                src_channels));
    }

    std::string file_name("test.bmp");
    FILE* fp = fopen(file_name.c_str(), "wb");
    if (fp == NULL) {
        std::cout << "failed to open test.bmp." << std::endl;
        return;
    }

    EncodeType encode_type{channels, bits_per_pixel, compression};
    bool succeeded = createBmpTestFile(src, encode_type, fp);
    if (succeeded == false) {
        std::cout << "failed to create test.bmp." << std::endl;
        fclose(fp);
        return;
    }
    int code = fclose(fp);
    if (code != 0) {
        std::cout << "failed to close test.bmp." << std::endl;
        return;
    }

    int height1, width1, channels1, stride;
    uchar* image;

    for (auto _ : state) {
        ppl::cv::x86::Imread(file_name.c_str(), &height1, &width1, &channels1,
                             &stride, &image);
        free(image);
    }
    state.SetItemsProcessed(state.iterations() * 1);

    code = remove(file_name.c_str());
    if (code != 0) {
        std::cout << "failed to delete test.bmp." << std::endl;
    }
}

template <int channels, int bits_per_pixel,
          ppl::cv::x86::BmpCompression compression>
void BM_ImreadBmp_opencv_x86(benchmark::State &state) {
    int width  = state.range(0);
    int height = state.range(1);

    int src_channels = 0;
    if (bits_per_pixel <= 8) {
        src_channels = 1;
    }
    else if (bits_per_pixel == 15 || bits_per_pixel == 16) {
        src_channels = 2;
    }
    else if (bits_per_pixel == 24 || bits_per_pixel == 32) {
        src_channels = channels;
    }
    else {
    }
    cv::Mat src, gray_image;
    if (src_channels == 1) {
        if (bits_per_pixel <= 8) {
            src = createSourceImage(height, width,
                                    CV_MAKETYPE(cv::DataType<uchar>::depth,
                                    src_channels));
        }
        else {
            gray_image = createSourceImage(height, width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
            cv::cvtColor(gray_image, src, cv::COLOR_GRAY2BGR);
        }
    }
    else {
        src = createSourceImage(height, width,
                                CV_MAKETYPE(cv::DataType<uchar>::depth,
                                src_channels));
    }

    std::string file_name("test.bmp");
    FILE* fp = fopen(file_name.c_str(), "wb");
    if (fp == NULL) {
        std::cout << "failed to open test.bmp." << std::endl;
        return;
    }

    EncodeType encode_type{channels, bits_per_pixel, compression};
    bool succeeded = createBmpTestFile(src, encode_type, fp);
    if (succeeded == false) {
        std::cout << "failed to create test.bmp." << std::endl;
        fclose(fp);
        return;
    }
    int code = fclose(fp);
    if (code != 0) {
        std::cout << "failed to close test.bmp." << std::endl;
        return;
    }

    for (auto _ : state) {
        cv::Mat dst = cv::imread(file_name, cv::IMREAD_UNCHANGED);
    }
    state.SetItemsProcessed(state.iterations() * 1);

    code = remove(file_name.c_str());
    if (code != 0) {
        std::cout << "failed to delete test.bmp." << std::endl;
    }
}

#define RUN_BMP_BENCHMARK(channels, bits_per_pixel, compression)               \
BENCHMARK_TEMPLATE(BM_ImreadBmp_opencv_x86, channels, bits_per_pixel,          \
                   compression)->Args({320, 240});                             \
BENCHMARK_TEMPLATE(BM_ImreadBmp_ppl_x86, channels, bits_per_pixel,             \
                   compression)->Args({320, 240});                             \
BENCHMARK_TEMPLATE(BM_ImreadBmp_opencv_x86, channels, bits_per_pixel,          \
                   compression)->Args({640, 480});                             \
BENCHMARK_TEMPLATE(BM_ImreadBmp_ppl_x86, channels, bits_per_pixel,             \
                   compression)->Args({640, 480});                             \
BENCHMARK_TEMPLATE(BM_ImreadBmp_opencv_x86, channels, bits_per_pixel,          \
                   compression)->Args({1280, 720});                            \
BENCHMARK_TEMPLATE(BM_ImreadBmp_ppl_x86, channels, bits_per_pixel,             \
                   compression)->Args({1280, 720});                            \
BENCHMARK_TEMPLATE(BM_ImreadBmp_opencv_x86, channels, bits_per_pixel,          \
                   compression)->Args({1920, 1080});                           \
BENCHMARK_TEMPLATE(BM_ImreadBmp_ppl_x86, channels, bits_per_pixel,             \
                   compression)->Args({1920, 1080});

RUN_BMP_BENCHMARK(1, 1, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(3, 1, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(1, 4, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(3, 4, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(1, 8, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(3, 8, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(3, 16, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(3, 24, ppl::cv::x86::BMP_RGB)
RUN_BMP_BENCHMARK(4, 32, ppl::cv::x86::BMP_RGB)

/***************************** Jpeg benchmark *****************************/

template <int channels>
void BM_ImreadJPEG0_ppl_x86(benchmark::State &state) {
    int width  = state.range(0);
    int height = state.range(1);

    cv::Mat src = createSourceImage(height, width,
                                    CV_MAKETYPE(cv::DataType<uchar>::depth,
                                    channels));
    std::string file_name("test.jpeg");
    bool succeeded = cv::imwrite(file_name.c_str(), src);
    if (succeeded == false) {
        std::cout << "failed to write the image to test.jpeg." << std::endl;
        return;
    }

    struct timeval start, end;
    int height1, width1, channels1, stride;
    uchar* image;
    for (auto _ : state) {
        gettimeofday(&start, NULL);
        ppl::cv::x86::Imread(file_name.c_str(), &height1, &width1, &channels1,
                             &stride, &image);
        gettimeofday(&end, NULL);
        int time = (end.tv_sec * 1000000 + end.tv_usec) -
                   (start.tv_sec * 1000000 + start.tv_usec);
        state.SetIterationTime(time * 1e-6);

        free(image);
    }
    state.SetItemsProcessed(state.iterations() * 1);

    int code = remove(file_name.c_str());
    if (code != 0) {
        std::cout << "failed to delete test.bmp." << std::endl;
    }
}

template <int channels>
void BM_ImreadJPEG0_opencv_x86(benchmark::State &state) {
    int width  = state.range(0);
    int height = state.range(1);

    cv::Mat src = createSourceImage(height, width,
                                    CV_MAKETYPE(cv::DataType<uchar>::depth,
                                    channels));
    std::string file_name("test.jpeg");

    bool succeeded = cv::imwrite(file_name.c_str(), src);
    if (succeeded == false) {
        std::cout << "failed to write the image to test.jpeg." << std::endl;
        return;
    }

    struct timeval start, end;
    for (auto _ : state) {
        gettimeofday(&start, NULL);
        cv::Mat dst = cv::imread(file_name, cv::IMREAD_UNCHANGED);
        gettimeofday(&end, NULL);
        int time = (end.tv_sec * 1000000 + end.tv_usec) -
                   (start.tv_sec * 1000000 + start.tv_usec);
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);

    int code = remove(file_name.c_str());
    if (code != 0) {
        std::cout << "failed to delete test.bmp." << std::endl;
    }
}

#define RUN_JPEG_BENCHMARK0(channels)                                          \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_opencv_x86, channels)->Args({320, 240})->    \
                   UseManualTime();                                            \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_ppl_x86, channels)->Args({320, 240})->       \
                   UseManualTime();                                            \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_opencv_x86, channels)->Args({640, 480})->    \
                   UseManualTime();                                            \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_ppl_x86, channels)->Args({640, 480})->       \
                   UseManualTime();                                            \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_opencv_x86, channels)->Args({1280, 720})->   \
                   UseManualTime();                                            \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_ppl_x86, channels)->Args({1280, 720})->      \
                   UseManualTime();                                            \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_opencv_x86, channels)->Args({1920, 1080})->  \
                   UseManualTime();                                            \
BENCHMARK_TEMPLATE(BM_ImreadJPEG0_ppl_x86, channels)->Args({1920, 1080})->     \
                   UseManualTime();

RUN_JPEG_BENCHMARK0(1)
RUN_JPEG_BENCHMARK0(3)

void BM_ImreadJPEG1_ppl_x86(benchmark::State &state) {
    int index = state.range(0);
    std::string file_name = "data/jpegs/progressive" + std::to_string(index) +
                            ".jpg";

    struct timeval start, end;
    int height, width, channels, stride;
    uchar* image;
    for (auto _ : state) {
        gettimeofday(&start, NULL);
        ppl::cv::x86::Imread(file_name.c_str(), &height, &width, &channels,
                             &stride, &image);
        gettimeofday(&end, NULL);
        int time = (end.tv_sec * 1000000 + end.tv_usec) -
                   (start.tv_sec * 1000000 + start.tv_usec);
        state.SetIterationTime(time * 1e-6);

        free(image);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

void BM_ImreadJPEG1_opencv_x86(benchmark::State &state) {
    int index = state.range(0);
    std::string file_name = "data/jpegs/progressive" + std::to_string(index) +
                            ".jpg";

    struct timeval start, end;
    for (auto _ : state) {
        gettimeofday(&start, NULL);
        cv::Mat dst = cv::imread(file_name, cv::IMREAD_UNCHANGED);
        gettimeofday(&end, NULL);
        int time = (end.tv_sec * 1000000 + end.tv_usec) -
                   (start.tv_sec * 1000000 + start.tv_usec);
        state.SetIterationTime(time * 1e-6);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_JPEG_BENCHMARK1()                                                  \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(0)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(0)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(1)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(1)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(2)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(2)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(3)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(3)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(4)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(4)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(5)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(5)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(6)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(6)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(7)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(7)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(8)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(8)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(9)->UseManualTime();                 \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(9)->UseManualTime();                    \
BENCHMARK(BM_ImreadJPEG1_opencv_x86)->Arg(10)->UseManualTime();                \
BENCHMARK(BM_ImreadJPEG1_ppl_x86)->Arg(10)->UseManualTime();

RUN_JPEG_BENCHMARK1()

/***************************** Png benchmark *****************************/

template <typename T>
void BM_ImreadPNG_ppl_x86(benchmark::State &state) {
    int index = state.range(0);

    std::string png_image = "data/pngs/png" + std::to_string(index) + ".png";
    int height, width, channels, stride;
    T* image = nullptr;

    struct timeval start, end;
    for (auto _ : state) {
        gettimeofday(&start, NULL);
        ppl::cv::x86::Imread(png_image.c_str(), &height, &width, &channels,
                             &stride, &image);
        gettimeofday(&end, NULL);
        int time = (end.tv_sec * 1000000 + end.tv_usec) -
                   (start.tv_sec * 1000000 + start.tv_usec);
        state.SetIterationTime(time * 1e-6);

        if (image != nullptr) {
            free(image);
            image = nullptr;
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T>
void BM_ImreadPNG_opencv_x86(benchmark::State &state) {
    int index = state.range(0);

    std::string png_image = "data/pngs/png" + std::to_string(index) + ".png";
    for (auto _ : state) {
        cv::Mat cv_dst = cv::imread(png_image, cv::IMREAD_UNCHANGED);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_PNG_BENCHMARK(uchar)                                               \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({0});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({0})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({1});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({1})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({2});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({2})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({3});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({3})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({4});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({4})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({5});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({5})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({6});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({6})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({7});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({7})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({8});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({8})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({9});                 \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({9})->UseManualTime();   \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({10});                \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({10})->UseManualTime();  \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({11});                \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({11})->UseManualTime();  \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({12});                \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({12})->UseManualTime();  \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({13});                \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({13})->UseManualTime();  \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({14});                \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({14})->UseManualTime();  \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({15});                \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({15})->UseManualTime();  \
BENCHMARK_TEMPLATE(BM_ImreadPNG_opencv_x86, uchar)->Args({16});                \
BENCHMARK_TEMPLATE(BM_ImreadPNG_ppl_x86, uchar)->Args({16})->UseManualTime();

RUN_PNG_BENCHMARK(uchar)