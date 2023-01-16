/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "infrastructure.h"
#include "utility.hpp"

schar randomChar() {
  int flag   = rand() % 2;
  int number = rand() % 128;
  if (flag) {
    return number;
  }
  else {
    return (0 - number - 1);
  }
}

template <typename T>
void randomImage(cv::Mat& image, int basic_type, int channels) {
  AUX_ASSERT(image.data != nullptr);
  AUX_ASSERT(image.rows >= 1 && image.cols >= 1);
  AUX_ASSERT(basic_type == CV_8U || basic_type == CV_8S ||
             basic_type == CV_32F || basic_type == CV_64F);
  AUX_ASSERT(channels == 1 || channels == 2 || channels == 3 || channels == 4);

  int rows = image.rows;
  int cols = image.cols;
  T *element;

  for (int row = 0; row < rows; ++row) {
    element = image.ptr<T>(row);

    for (int col = 0; col < cols; ++col) {
      if (basic_type == CV_8U) {
        element[0] = rand() % 256;
        if (channels >= 2) {
          element[1] = rand() % 256;
        }
        if (channels >= 3) {
          element[2] = rand() % 256;
        }
        if (channels == 4) {
          element[3] = 255;
        }
      }
      else if (basic_type == CV_8S) {
        element[0] = randomChar();
        if (channels >= 2) {
          element[1] = randomChar();
        }
        if (channels >= 3) {
          element[2] = randomChar();
        }
        if (channels == 4) {
          element[3] = 255;
        }
      }
      else if (basic_type == CV_32F) {
        element[0] = (float) rand() / RAND_MAX;
        if (channels >= 2) {
          element[1] = (float) rand() / RAND_MAX;
        }
        if (channels >= 3) {
          element[2] = (float) rand() / RAND_MAX;
        }
        if (channels == 4) {
          element[3] = 1.0f;
        }
      }
      else if (basic_type == CV_64F) {
        element[0] = (double) rand() / RAND_MAX;
        if (channels >= 2) {
          element[1] = (double) rand() / RAND_MAX;
        }
        if (channels >= 3) {
          element[2] = (double) rand() / RAND_MAX;
        }
        if (channels == 4) {
          element[3] = 1.0;
        }
      }
      else {
      }
      element += channels;
    }
  }
}

template <typename T>
T clamp(T value, T begin, T end) {
  if (value < begin) {
    return begin;
  }
  else if (value > end) {
    return end;
  }
  else {
    return value;
  }
}

template <typename T>
void randomImage(cv::Mat& image, int basic_type, int channels, T begin, T end) {
  AUX_ASSERT(image.data != nullptr);
  AUX_ASSERT(image.rows >= 1 && image.cols >= 1);
  AUX_ASSERT(basic_type == CV_8U || basic_type == CV_8S ||
             basic_type == CV_32F || basic_type == CV_64F);
  AUX_ASSERT(channels == 1 || channels == 2 || channels == 3 || channels == 4);
  AUX_ASSERT(begin <= end);

  int rows = image.rows;
  int cols = image.cols;
  T *element;

  for (int row = 0; row < rows; ++row) {
    element = image.ptr<T>(row);

    for (int col = 0; col < cols; ++col) {
      if (basic_type == CV_8U) {
        element[0] = clamp<T>(rand() % 256, begin, end);
        if (channels >= 2) {
          element[1] = clamp<T>(rand() % 256, begin, end);
        }
        if (channels >= 3) {
          element[2] = clamp<T>(rand() % 256, begin, end);
        }
        if (channels == 4) {
          element[3] = 255;
        }
      }
      else if (basic_type == CV_8S) {
        element[0] = clamp<T>(randomChar(), begin, end);
        if (channels >= 2) {
          element[1] = clamp<T>(randomChar(), begin, end);
        }
        if (channels >= 3) {
          element[2] = clamp<T>(randomChar(), begin, end);
        }
        if (channels == 4) {
          element[3] = 255;
        }
      }
      else if (basic_type == CV_32F) {
        element[0] = clamp<T>((float) rand() / RAND_MAX, begin, end);
        if (channels >= 2) {
          element[1] = clamp<T>((float) rand() / RAND_MAX, begin, end);
        }
        if (channels >= 3) {
          element[2] = clamp<T>((float) rand() / RAND_MAX, begin, end);
        }
        if (channels == 4) {
          element[3] = 1.0f;
        }
      }
      else if (basic_type == CV_64F) {
        element[0] = clamp<T>((double) rand() / RAND_MAX, begin, end);
        if (channels >= 2) {
          element[1] = clamp<T>((double) rand() / RAND_MAX, begin, end);
        }
        if (channels >= 3) {
          element[2] = clamp<T>((double) rand() / RAND_MAX, begin, end);
        }
        if (channels == 4) {
          element[3] = 1.0;
        }
      }
      else {
      }
      element += channels;
    }
  }
}

cv::Mat createSourceImage(int rows, int cols, int type) {
  AUX_ASSERT(rows >= 1 && cols >= 1);
  AUX_ASSERT(type == CV_8UC1 || type == CV_8UC2 ||
             type == CV_8UC3 || type == CV_8UC4 ||
             type == CV_8SC1 || type == CV_8SC2 ||
             type == CV_8SC3 || type == CV_8SC4 ||
             type == CV_32FC1 || type == CV_32FC2 ||
             type == CV_32FC3 || type == CV_32FC4 ||
             type == CV_64FC1 || type == CV_64FC2 ||
             type == CV_64FC3 || type == CV_64FC4);

  cv::Mat image(rows, cols, type);

  if (type == CV_8UC1) {
    randomImage<unsigned char>(image, CV_8U, 1);
  }
  else if (type == CV_8UC2) {
    randomImage<unsigned char>(image, CV_8U, 2);
  }
  else if (type == CV_8UC3) {
    randomImage<unsigned char>(image, CV_8U, 3);
  }
  else if (type == CV_8UC4) {
    randomImage<unsigned char>(image, CV_8U, 4);
  }
  else if (type == CV_8SC1) {
    randomImage<char>(image, CV_8S, 1);
  }
  else if (type == CV_8SC2) {
    randomImage<char>(image, CV_8S, 2);
  }
  else if (type == CV_8SC3) {
    randomImage<char>(image, CV_8S, 3);
  }
  else if (type == CV_8SC4) {
    randomImage<char>(image, CV_8S, 4);
  }
  else if (type == CV_32FC1) {
    randomImage<float>(image, CV_32F, 1);
  }
  else if (type == CV_32FC2) {
    randomImage<float>(image, CV_32F, 2);
  }
  else if (type == CV_32FC3) {
    randomImage<float>(image, CV_32F, 3);
  }
  else if (type == CV_32FC4) {
    randomImage<float>(image, CV_32F, 4);
  }
  else if (type == CV_64FC1) {
    randomImage<double>(image, CV_64F, 1);
  }
  else if (type == CV_64FC2) {
    randomImage<double>(image, CV_64F, 2);
  }
  else if (type == CV_64FC3) {
    randomImage<double>(image, CV_64F, 3);
  }
  else if (type == CV_64FC4) {
    randomImage<double>(image, CV_64F, 4);
  }
  else {
  }

  return image;
}

cv::Mat createSourceImage(int rows, int cols, int type, float begin,
                          float end) {
  AUX_ASSERT(rows >= 1 && cols >= 1);
  AUX_ASSERT(type == CV_8UC1 || type == CV_8UC2 ||
             type == CV_8UC3 || type == CV_8UC4 ||
             type == CV_8SC1 || type == CV_8SC2 ||
             type == CV_8SC3 || type == CV_8SC4 ||
             type == CV_32FC1 || type == CV_32FC2 ||
             type == CV_32FC3 || type == CV_32FC4 ||
             type == CV_64FC1 || type == CV_64FC2 ||
             type == CV_64FC3 || type == CV_64FC4);
  AUX_ASSERT(begin <= end);

  cv::Mat image(rows, cols, type);

  if (type == CV_8UC1) {
    randomImage<unsigned char>(image, CV_8U, 1, begin, end);
  }
  else if (type == CV_8UC2) {
    randomImage<unsigned char>(image, CV_8U, 2, begin, end);
  }
  else if (type == CV_8UC3) {
    randomImage<unsigned char>(image, CV_8U, 3, begin, end);
  }
  else if (type == CV_8UC4) {
    randomImage<unsigned char>(image, CV_8U, 4, begin, end);
  }
  else if (type == CV_8SC1) {
    randomImage<char>(image, CV_8S, 1, begin, end);
  }
  else if (type == CV_8SC2) {
    randomImage<char>(image, CV_8S, 2, begin, end);
  }
  else if (type == CV_8SC3) {
    randomImage<char>(image, CV_8S, 3, begin, end);
  }
  else if (type == CV_8SC4) {
    randomImage<char>(image, CV_8S, 4, begin, end);
  }
  else if (type == CV_32FC1) {
    randomImage<float>(image, CV_32F, 1, begin, end);
  }
  else if (type == CV_32FC2) {
    randomImage<float>(image, CV_32F, 2, begin, end);
  }
  else if (type == CV_32FC3) {
    randomImage<float>(image, CV_32F, 3, begin, end);
  }
  else if (type == CV_32FC4) {
    randomImage<float>(image, CV_32F, 4, begin, end);
  }
  else if (type == CV_64FC1) {
    randomImage<double>(image, CV_64F, 1, begin, end);
  }
  else if (type == CV_64FC2) {
    randomImage<double>(image, CV_64F, 2, begin, end);
  }
  else if (type == CV_64FC3) {
    randomImage<double>(image, CV_64F, 3, begin, end);
  }
  else if (type == CV_64FC4) {
    randomImage<double>(image, CV_64F, 4, begin, end);
  }
  else {
  }

  return image;
}

cv::Mat createBinaryImage(int rows, int cols, int type) {
  AUX_ASSERT(rows >= 1 && cols >= 1);
  AUX_ASSERT(type == CV_8UC1);

  cv::Mat image(rows, cols, type);

  uchar *element;
  for (int row = 0; row < rows; ++row) {
    element = image.ptr<uchar>(row);

    for (int col = 0; col < cols; ++col) {
      element[0] = rand() % 1;
      element++;
    }
  }

  return image;
}

template <typename T>
void copyMatToArray(const cv::Mat& image0, T* image1) {
  AUX_ASSERT(image0.data != nullptr);
  AUX_ASSERT(image1 != nullptr);
  AUX_ASSERT(image0.data != (uchar*)image1);
  AUX_ASSERT(image0.rows >= 1 && image0.cols >= 1);
  AUX_ASSERT(image0.channels() == 1 || image0.channels() == 2 ||
             image0.channels() == 3 || image0.channels() == 4);

  int rows = image0.rows;
  int cols = image0.cols;
  int channels = image0.channels();
  const T *element0;
  T *element1;

  for (int row = 0; row < rows; ++row) {
    element0 = image0.ptr<const T>(row);
    element1 = (T*)((uchar*)image1 + row * cols * channels * sizeof(T));

    for (int col = 0; col < cols; ++col) {
      element1[0] = element0[0];
      if (channels >= 2) {
        element1[1] = element0[1];
      }
      if (channels >= 3) {
        element1[2] = element0[2];
      }
      if (channels == 4) {
        element1[3] = element0[3];
      }

      element0 += channels;
      element1 += channels;
    }
  }
}

template
void copyMatToArray(const cv::Mat& image0, uchar* image1);

template
void copyMatToArray(const cv::Mat& image0, schar* image1);

template
void copyMatToArray(const cv::Mat& image0, float* image1);

inline
void findMax(float& max, const float& value) {
  if (value > max) {
    max = value;
  }
}

inline
void findMax(double& max, const double& value) {
  if (value > max) {
    max = value;
  }
}

template <typename T>
bool checkMatricesIdentity(const T* src0, int rows, int cols, int channels,
                           int src0_stride, const T* src1, int src1_stride,
                           float epsilon, bool display) {
  AUX_ASSERT(src0 != nullptr);
  AUX_ASSERT(src1 != nullptr);
  AUX_ASSERT(rows >= 1 && cols >= 1);
  AUX_ASSERT(channels == 1 || channels == 2 || channels == 3 || channels == 4);
  AUX_ASSERT(src0_stride >= cols * channels * (int)sizeof(T));
  AUX_ASSERT(src1_stride >= cols * channels * (int)sizeof(T));
  AUX_ASSERT(epsilon > 0.f);

  float difference, max = 0.f;
  T *element0, *element1;

  std::cout.precision(7);
  for (int row = 0; row < rows; ++row) {
    element0 = (T*)((uchar*)src0 + row * src0_stride);
    element1 = (T*)((uchar*)src1 + row * src1_stride);

    for (int col = 0; col < cols; ++col) {
      difference = fabs((float) element0[0] - (float) element1[0]);
      findMax(max, difference);
      if (difference > epsilon || display)  {
        std::cout << "[" << row << ", " << col <<"].0: " << (float)element0[0]
                  << ", " << (float)element1[0] << std::endl;
      }
      if (channels >= 2) {
        difference = fabs((float) element0[1] - (float) element1[1]);
        findMax(max, difference);
        if (difference > epsilon || display)  {
          std::cout << "[" << row << ", " << col <<"].1: " << (float)element0[1]
                    << ", " << (float)element1[1] << std::endl;
        }
      }
      if (channels >= 3) {
        difference = fabs((float) element0[2] - (float) element1[2]);
        findMax(max, difference);
        if (difference > epsilon || display)  {
          std::cout << "[" << row << ", " << col <<"].2: " << (float)element0[2]
                    << ", " << (float)element1[2] << std::endl;
        }
      }
      if (channels == 4) {
        difference = fabs((float) element0[3] - (float) element1[3]);
        findMax(max, difference);
        if (difference > epsilon || display)  {
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

template
bool checkMatricesIdentity(const uchar* src0, int rows, int cols, int channels,
                           int src0_stride, const uchar* src1, int src1_stride,
                           float epsilon, bool display);

template
bool checkMatricesIdentity(const schar* src0, int rows, int cols, int channels,
                           int src0_stride, const schar* src1, int src1_stride,
                           float epsilon, bool display);

template
bool checkMatricesIdentity(const float* src0, int rows, int cols, int channels,
                           int src0_stride, const float* src1, int src1_stride,
                           float epsilon, bool display);
