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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <string>
#include <assert.h>

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "gtest/gtest.h"

#include "ppl/cv/x86/imgcodecs/byteswriter.h"
#include "ppl/cv/x86/imgcodecs/bmp.h"
#include "ppl/cv/cuda/utility/infrastructure.hpp"

template <typename T>
bool checkDataIdentity(const T* image0, const T* image1, int height, int width,
                       int channels, int stride0, int stride1, float epsilon,
                       bool display = false) {
  assert(image0 != nullptr);
  assert(image1 != nullptr);
  assert(height >= 1 && width >= 1);
  assert(channels == 1 || channels == 2 || channels == 3 || channels == 4);
  assert(stride0 >= width);
  assert(stride1 >= width);
  assert(epsilon > 0.f);

  float difference, max = 0.0f;
  const T *element0, *element1;

  std::cout.precision(7);
  for (int row = 0; row < height; ++row) {
    element0 = (T*)((uchar*)image0 + row * stride0);
    element1 = (T*)((uchar*)image1 + row * stride1);

    for (int col = 0; col < width; ++col) {
      difference = fabs((float)element0[0] - (float)element1[0]);
      if (difference > max) max = difference;
      if (difference > epsilon || display) {
        std::cout << "[" << row << ", " << col <<"].0: " << (float)element0[0]
                  << ", " << (float)element1[0] << std::endl;
      }
      if (channels >= 2) {
        difference = fabs((float)element0[1] - (float)element1[1]);
        if (difference > max) max = difference;
        if (difference > epsilon || display) {
          std::cout << "[" << row << ", " << col <<"].1: " << (float)element0[1]
                    << ", " << (float)element1[1] << std::endl;
        }
      }
      if (channels >= 3) {
        difference = fabs((float)element0[2] - (float)element1[2]);
        if (difference > max) max = difference;
        if (difference > epsilon || display) {
          std::cout << "[" << row << ", " << col <<"].2: " << (float)element0[2]
                    << ", " << (float)element1[2] << std::endl;
        }
      }
      if (channels == 4) {
        difference = fabs((float)element0[3] - (float)element1[3]);
        if (difference > max) max = difference;
        if (difference > epsilon || display) {
          std::cout << "[" << row << ", " << col <<"].3: " << (float)element0[3]
                    << ", " << (float)element1[3] << std::endl;
        }
      }

      element0 += channels;
      element1 += channels;
    }
  }

  if (max <= epsilon) {
    return true;
  }
  else {
    std::cout << "Max difference between elements of the two images: "
              << max << std::endl;
    return false;
  }
}

void BGRA2GRAYA(cv::Mat& src, cv::Mat& dst) {
    assert(src.type() == CV_8UC4);
    assert(dst.type() == CV_8UC2);

    int rows = src.rows;
    int cols = src.cols;
    int src_channels = src.channels();
    int dst_channels = dst.channels();
    uchar *element0, *element1;

    for (int row = 0; row < rows; row++) {
        element0 = src.ptr<uchar>(row);
        element1 = dst.ptr<uchar>(row);
        for (int col = 0; col < cols; col++) {
            element1[0] = element0[0];
            element1[1] = element0[3];
            element0 += src_channels;
            element1 += dst_channels;
        }
    }
}

/***************************** Bmp unittest *****************************/

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
                bytes_writer.putBytes(image.ptr(row), file_step);
            }
            break;
        case 32:
            for (int row = image.rows - 1; row >= 0; row--) {
                bytes_writer.putBytes(image.ptr(row), file_step);
            }
            break;
        default:
            return false;
    }
    bytes_writer.writeBlock();

    return true;
}

using Parameters0 = std::tuple<EncodeType, cv::Size>;
inline std::string convertToStringBmp(const Parameters0& parameters) {
    std::ostringstream formatted;

    EncodeType encode_type = std::get<0>(parameters);
    formatted << "Channels" << encode_type.channels << "_";
    formatted << "BitPerPexil" << encode_type.bits_per_pixel << "_";
    if (encode_type.compression == ppl::cv::x86::BMP_RGB) {
        formatted << "Compression_RGB" << "_";
    }
    else if (encode_type.compression == ppl::cv::x86::BMP_RLE4) {
        formatted << "Compression_RLE4" << "_";
    }
    else if (encode_type.compression == ppl::cv::x86::BMP_RLE8) {
        formatted << "Compression_RLE8" << "_";
    }
    else {
        formatted << "Compression_BITFIELDS" << "_";
    }

    cv::Size size = std::get<1>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

class PplCvX86ImreadBmpTest : public ::testing::TestWithParam<Parameters0> {
  public:
    PplCvX86ImreadBmpTest() {
        const Parameters0& parameters = GetParam();
        encode_type = std::get<0>(parameters);
        size        = std::get<1>(parameters);
    }

    ~PplCvX86ImreadBmpTest() {
    }

    bool apply();

  private:
    EncodeType encode_type;
    cv::Size size;
};

bool PplCvX86ImreadBmpTest::apply() {
    int src_channels = 0;
    if (encode_type.bits_per_pixel <= 8) {
        src_channels = 1;
    }
    else if (encode_type.bits_per_pixel == 15 ||
             encode_type.bits_per_pixel == 16) {
        src_channels = 2;
    }
    else if (encode_type.bits_per_pixel == 24 ||
             encode_type.bits_per_pixel == 32) {
        src_channels = encode_type.channels;
    }
    else {
    }
    cv::Mat src, gray_image;
    if (src_channels == 1) {
        if (encode_type.bits_per_pixel <= 8) {
            src = createSourceImage(size.height, size.width,
                                    CV_MAKETYPE(cv::DataType<uchar>::depth,
                                    src_channels));
        }
        else {
            gray_image = createSourceImage(size.height, size.width,
                             CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
            cv::cvtColor(gray_image, src, cv::COLOR_GRAY2BGR);
        }
    }
    else {
        src = createSourceImage(size.height, size.width,
                                CV_MAKETYPE(cv::DataType<uchar>::depth,
                                src_channels));
    }

    std::string file_name("test.bmp");
    FILE* fp = fopen(file_name.c_str(), "wb");
    if (fp == NULL) {
        std::cout << "failed to open test.bmp." << std::endl;
        return false;
    }

    bool succeeded = createBmpTestFile(src, encode_type, fp);
    if (succeeded == false) {
        std::cout << "failed to create test.bmp." << std::endl;
        fclose(fp);
        return false;
    }
    int code = fclose(fp);
    if (code != 0) {
        std::cout << "failed to close test.bmp." << std::endl;
        return false;
    }

    cv::Mat cv_dst = cv::imread(file_name, cv::IMREAD_UNCHANGED);

    int height, width, channels, stride;
    uchar* image;
    ppl::cv::x86::Imread(file_name.c_str(), &height, &width, &channels, &stride,
                         &image);

    float epsilon = EPSILON_1F;
    bool identity = checkDataIdentity<uchar>(cv_dst.data, image, height, width,
                                             channels, cv_dst.step, stride,
                                             epsilon);

    free(image);
    code = remove(file_name.c_str());
    if (code != 0) {
        std::cout << "failed to delete test.bmp." << std::endl;
    }

    return identity;
}

TEST_P(PplCvX86ImreadBmpTest, Standard) {
    bool identity = this->apply();
    EXPECT_TRUE(identity);
}

INSTANTIATE_TEST_CASE_P(IsEqual, PplCvX86ImreadBmpTest,
    ::testing::Combine(
        ::testing::Values(EncodeType{1, 1, ppl::cv::x86::BMP_RGB},
                          EncodeType{3, 1, ppl::cv::x86::BMP_RGB},
                          EncodeType{1, 4, ppl::cv::x86::BMP_RGB},
                          EncodeType{3, 4, ppl::cv::x86::BMP_RGB},
                          EncodeType{1, 8, ppl::cv::x86::BMP_RGB},
                          EncodeType{3, 8, ppl::cv::x86::BMP_RGB},
                          EncodeType{3, 15, ppl::cv::x86::BMP_BITFIELDS},
                          EncodeType{3, 16, ppl::cv::x86::BMP_BITFIELDS},
                          EncodeType{3, 24, ppl::cv::x86::BMP_RGB},
                          EncodeType{4, 32, ppl::cv::x86::BMP_RGB}),
        ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},
                          cv::Size{1283, 720}, cv::Size{1976, 1080},
                          cv::Size{320, 240}, cv::Size{640, 480},
                          cv::Size{1280, 720}, cv::Size{1920, 1080})),
    [](const testing::TestParamInfo<PplCvX86ImreadBmpTest::ParamType>& info) {
        return convertToStringBmp(info.param);
    }
);

/***************************** Jpeg unittest *****************************/

using Parameters1 = std::tuple<int, cv::Size>;
inline std::string convertToStringJpeg(const Parameters1& parameters) {
    std::ostringstream formatted;

    int channels = std::get<0>(parameters);
    formatted << "Channels" << channels << "_";

    cv::Size size = std::get<1>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

class PplCvX86ImreadJpegTest0 : public ::testing::TestWithParam<Parameters1> {
  public:
    PplCvX86ImreadJpegTest0() {
        const Parameters1& parameters = GetParam();
        channels = std::get<0>(parameters);
        size     = std::get<1>(parameters);
    }

    ~PplCvX86ImreadJpegTest0() {
    }

    bool apply();

  private:
    int channels;
    cv::Size size;
};

bool PplCvX86ImreadJpegTest0::apply() {
    cv::Mat src = createSourceImage(size.height, size.width,
                                    CV_MAKETYPE(cv::DataType<uchar>::depth,
                                    channels));
    std::string file_name("test.jpeg");
    bool succeeded = cv::imwrite(file_name.c_str(), src);
    if (succeeded == false) {
        std::cout << "failed to write the image to test.jpeg." << std::endl;
        return false;
    }

    cv::Mat cv_dst = cv::imread(file_name, cv::IMREAD_UNCHANGED);

    int height, width, channels, stride;
    uchar* image;
    ppl::cv::x86::Imread(file_name.c_str(), &height, &width, &channels, &stride,
                         &image);

    float epsilon = EPSILON_3F;
    bool identity = checkDataIdentity<uchar>(cv_dst.data, image, height, width,
                                             channels, cv_dst.step, stride,
                                             epsilon);

    free(image);
    int code = remove(file_name.c_str());
    if (code != 0) {
        std::cout << "failed to delete test.jpeg." << std::endl;
    }

    return identity;
}

TEST_P(PplCvX86ImreadJpegTest0, Standard) {
    bool identity = this->apply();
    EXPECT_TRUE(identity);
}

INSTANTIATE_TEST_CASE_P(IsEqual, PplCvX86ImreadJpegTest0,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},
                          cv::Size{1283, 720}, cv::Size{1976, 1080},
                          cv::Size{320, 240}, cv::Size{640, 480},
                          cv::Size{1280, 720}, cv::Size{1920, 1080})),
    [](const testing::TestParamInfo<PplCvX86ImreadJpegTest0::ParamType>& info) {
        return convertToStringJpeg(info.param);
    }
);

class PplCvX86ImreadJpegTest1 : public ::testing::TestWithParam<Parameters1> {
  public:
    PplCvX86ImreadJpegTest1() {
        const Parameters1& parameters = GetParam();
        channels = std::get<0>(parameters);
        size     = std::get<1>(parameters);
    }

    ~PplCvX86ImreadJpegTest1() {
    }

    bool apply();

  private:
    int channels;
    cv::Size size;
};

bool PplCvX86ImreadJpegTest1::apply() {
    int height, width, channels, stride;
    uchar* image = nullptr;
    bool identity = true;
    for (int i = 0; i < 11; i++) {
        std::cout << "****** decoding image " << i << " ******" << std::endl;
        std::string jpeg_image = "data/jpegs/progressive" + std::to_string(i) +
                                 ".jpg";
        cv::Mat cv_dst = cv::imread(jpeg_image, cv::IMREAD_UNCHANGED);
        ppl::cv::x86::Imread(jpeg_image.c_str(), &height, &width, &channels,
                             &stride, &image);

        float epsilon = EPSILON_3F;
        identity = checkDataIdentity<uchar>(cv_dst.data, image, height, width,
                                            channels, cv_dst.step, stride,
                                            epsilon);
        if (image != nullptr) {
            free(image);
            image = nullptr;
        }

        if (!identity) return false;
    }

    return true;
}

TEST_P(PplCvX86ImreadJpegTest1, Standard) {
    bool identity = this->apply();
    EXPECT_TRUE(identity);
}

INSTANTIATE_TEST_CASE_P(IsEqual, PplCvX86ImreadJpegTest1,
    ::testing::Combine(
        ::testing::Values(1),
        ::testing::Values(cv::Size{1, 1})),
    [](const testing::TestParamInfo<PplCvX86ImreadJpegTest1::ParamType>&
        info) {
        return convertToStringJpeg(info.param);
    }
);

/***************************** Png unittest *****************************/

using Parameters1 = std::tuple<int, cv::Size>;
inline std::string convertToStringPng(const Parameters1& parameters) {
    std::ostringstream formatted;

    int channels = std::get<0>(parameters);
    formatted << "Channels" << channels << "_";

    cv::Size size = std::get<1>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

template <typename T>
class PplCvX86ImreadPngTest0 : public ::testing::TestWithParam<Parameters1> {
  public:
    PplCvX86ImreadPngTest0() {
        const Parameters1& parameters = GetParam();
        channels = std::get<0>(parameters);
        size     = std::get<1>(parameters);
    }

    ~PplCvX86ImreadPngTest0() {
    }

    bool apply();

  private:
    int channels;
    cv::Size size;
};

template <typename T>
bool PplCvX86ImreadPngTest0<T>::apply() {
    cv::Mat src = createSourceImage(size.height, size.width,
                                    CV_MAKETYPE(cv::DataType<T>::depth,
                                    channels));
    std::string file_name("test.png");
    bool succeeded = cv::imwrite(file_name.c_str(), src);
    if (succeeded == false) {
        std::cout << "failed to write the image to test.png." << std::endl;
        return false;
    }

    cv::Mat cv_dst = cv::imread(file_name, cv::IMREAD_UNCHANGED);
    int height, width, channels, stride;
    T* image = nullptr;
    ppl::cv::x86::Imread(file_name.c_str(), &height, &width, &channels, &stride,
                         &image);

    float epsilon = EPSILON_1F;
    bool identity = checkDataIdentity<T>(cv_dst.data, image, height, width,
                                         channels, cv_dst.step, stride,
                                         epsilon);
    if (image != nullptr) {
        free(image);
    }
    int code = remove(file_name.c_str());
    if (code != 0) {
        std::cout << "failed to delete test.png." << std::endl;
    }

    return identity;
}

#define PNG_UNITTEST0(T)                                                       \
using PplCvX86ImreadPngTest0 ## T = PplCvX86ImreadPngTest0<T>;                 \
TEST_P(PplCvX86ImreadPngTest0 ## T, Standard) {                                \
    bool identity = this->apply();                                             \
    EXPECT_TRUE(identity);                                                     \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvX86ImreadPngTest0 ## T,                  \
    ::testing::Combine(                                                        \
        ::testing::Values(1, 3, 4),                                            \
        ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},              \
                          cv::Size{1283, 720}, cv::Size{1976, 1080},           \
                          cv::Size{320, 240}, cv::Size{640, 480},              \
                          cv::Size{1280, 720}, cv::Size{1920, 1080})),         \
    [](const testing::TestParamInfo<PplCvX86ImreadPngTest0 ## T::ParamType>&   \
        info) {                                                                \
        return convertToStringPng(info.param);                                 \
    }                                                                          \
);

PNG_UNITTEST0(uchar)

template <typename T>
class PplCvX86ImreadPngTest1 : public ::testing::TestWithParam<Parameters1> {
  public:
    PplCvX86ImreadPngTest1() {
        const Parameters1& parameters = GetParam();
        channels = std::get<0>(parameters);
        size     = std::get<1>(parameters);
    }

    ~PplCvX86ImreadPngTest1() {
    }

    bool apply();

  private:
    int channels;
    cv::Size size;
};

template <typename T>
bool PplCvX86ImreadPngTest1<T>::apply() {
    int height, width, channels, stride;
    T* image = nullptr;
    bool identity;
    for (int i = 0; i < 17; i++) {
        std::string png_image = "data/pngs/png" + std::to_string(i) + ".png";
        cv::Mat cv_dst = cv::imread(png_image, cv::IMREAD_UNCHANGED);
        ppl::cv::x86::Imread(png_image.c_str(), &height, &width, &channels,
                             &stride, &image);

        float epsilon = EPSILON_1F;
        if (channels != 2) {
            identity = checkDataIdentity<T>(cv_dst.data, image, height, width,
                                            channels, cv_dst.step, stride,
                                            epsilon);
        }
        else {
            cv::Mat dst(height, width, CV_8UC2);
            BGRA2GRAYA(cv_dst, dst);
            identity = checkDataIdentity<T>(dst.data, image, height, width,
                                            channels, dst.step, stride,
                                            epsilon);
        }
        if (image != nullptr) {
            free(image);
            image = nullptr;
        }

        if (!identity) return false;
    }

    return true;
}

#define PNG_UNITTEST1(T)                                                       \
using PplCvX86ImreadPngTest1 ## T = PplCvX86ImreadPngTest1<T>;                 \
TEST_P(PplCvX86ImreadPngTest1 ## T, Standard) {                                \
    bool identity = this->apply();                                             \
    EXPECT_TRUE(identity);                                                     \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvX86ImreadPngTest1 ## T,                  \
    ::testing::Combine(                                                        \
        ::testing::Values(1),                                                  \
        ::testing::Values(cv::Size{1, 1})),                                    \
    [](const testing::TestParamInfo<PplCvX86ImreadPngTest1 ## T::ParamType>&   \
        info) {                                                                \
        return convertToStringPng(info.param);                                 \
    }                                                                          \
);

PNG_UNITTEST1(uchar)
