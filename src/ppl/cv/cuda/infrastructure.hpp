/**
 * @file   infrastructure.hpp
 * @brief  Infrastructure functions for the convenience of unittest and
 *         benchmark.
 * @author Liheng Jian(jianliheng@sensetime.com)
 *
 * @copyright Copyright (c) 2014-2021 SenseTime Group Limited.
 */

#ifndef _ST_HPC_PPL3_CV_CUDA_TEST_INFRASTRUCTURE_HPP_
#define _ST_HPC_PPL3_CV_CUDA_TEST_INFRASTRUCTURE_HPP_

#include <cstdlib>
#include <cmath>

#include <iostream>

#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"

#define EPSILON_1F 1.1f
#define EPSILON_E4 1e-4
#define EPSILON_E6 1e-6

inline
char randomChar() {
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
  if (image.data == nullptr || (basic_type != CV_8U && basic_type != CV_8S &&
      basic_type != CV_32F && basic_type != CV_64F) ||
      (channels != 1 && channels != 2 && channels != 3 && channels != 4)) {
    std::cout << "Invalid parameters in randomImage()." << std::endl;
    return;
  }

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
void randomImage(cv::Mat& image, int basic_type, int channels, T begin,
                 T end) {
  if (image.data == nullptr || (basic_type != CV_8U && basic_type != CV_8S &&
      basic_type != CV_32F && basic_type != CV_64F) ||
      (channels != 1 && channels != 2 && channels != 3 && channels != 4) ||
      begin > end) {
    std::cout << "Invalid parameters in randomImage()." << std::endl;
    return;
  }

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

inline
cv::Mat createSourceImage(int rows, int cols, int type) {
  if (rows < 1 || cols < 1 ||
      (type != CV_8UC1 && type != CV_8UC2 &&
       type != CV_8UC3 && type != CV_8UC4 &&
       type != CV_8SC1 && type != CV_8SC3 && type != CV_8SC4 &&
       type != CV_32FC1 && type != CV_32FC3 && type != CV_32FC4 &&
       type != CV_64FC1 && type != CV_64FC3 && type != CV_64FC4)) {
    std::cout << "Invalid parameters in createSourceImage()." << std::endl;
    return cv::Mat(0, 0, type);
  }

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
  else if (type == CV_8SC3) {
    randomImage<char>(image, CV_8S, 3);
  }
  else if (type == CV_8SC4) {
    randomImage<char>(image, CV_8S, 4);
  }
  else if (type == CV_32FC1) {
    randomImage<float>(image, CV_32F, 1);
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

inline
cv::Mat createSourceImage(int rows, int cols, int type, float begin,
                          float end) {
  if (rows < 1 || cols < 1 || begin > end ||
      (type != CV_8UC1 && type != CV_8UC2 &&
       type != CV_8UC3 && type != CV_8UC4 &&
       type != CV_8SC1 && type != CV_8SC2 &&
       type != CV_8SC3 && type != CV_8SC4 &&
       type != CV_32FC1 && type != CV_32FC2 &&
       type != CV_32FC3 && type != CV_32FC4 &&
       type != CV_64FC1 && type != CV_64FC2 &&
       type != CV_64FC3 && type != CV_64FC4)) {
    std::cout << "Invalid parameters in createSourceImage()." << std::endl;
    return cv::Mat(0, 0, type);
  }

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

template <typename T>
void copyMatToArray(const cv::Mat& image0, T *image1) {
  if (image0.data == nullptr || image1 == nullptr) {
    std::cout << "Invalid parameters in copyMatToArray()." << std::endl;
    return;
  }

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
bool checkMatricesIdentity(const cv::Mat& image0, const cv::Mat& image1,
                           float epsilon, bool display = false) {
  if (image0.data == nullptr || image1.data == nullptr ||
      image0.rows != image1.rows || image0.cols != image1.cols ||
      image0.channels() != image1.channels() || epsilon < 0) {
    std::cout << "Invalid parameters in checkMatricesIdentity()." << std::endl;
    return false;
  }

  int rows = image0.rows;
  int cols = image0.cols;
  int type = image0.type();
  int channels = image0.channels();
  float difference, max = 0.0f;
  const T *element0, *element1;

  if (rows != image1.rows || cols != image1.cols) {
    std::cout << "checkMatricesIdentity(): two images don't have the same size!"
              << std::endl;
    return false;
  }

  if (type != image1.type()) {
    std::cout << "checkMatricesIdentity(): two images don't have the same type!"
              << std::endl;
    return false;
  }

  if (channels != image1.channels()) {
    std::cout << "checkMatricesIdentity(): two images have different channels!"
              << std::endl;
    return false;
  }

  std::cout.precision(7);
  for (int row = 0; row < rows; ++row) {
    element0 = image0.ptr<const T>(row);
    element1 = image1.ptr<const T>(row);

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

template <typename T>
bool checkMatArrayIdentity(const cv::Mat& image0, const T* image1,
                           float epsilon, bool display = false) {
  if (image0.data == nullptr || image1 == nullptr || epsilon < 0) {
    std::cout << "Invalid parameters in checkMatArrayIdentity()." << std::endl;
    return false;
  }

  int rows = image0.rows;
  int cols = image0.cols;
  int channels = image0.channels();
  float difference, max = 0.0f;
  const T *element0, *element1;

  std::cout.precision(7);
  for (int row = 0; row < rows; ++row) {
    element0 = image0.ptr<const T>(row);
    element1 = (T*)((uchar*)image1 + row * cols * channels * sizeof(T));

    for (int col = 0; col < cols; ++col) {
      difference = fabs((float) element0[0] - (float) element1[0]);
      findMax(max, difference);
      if (difference > epsilon || display) {
        std::cout << "[" << row << ", " << col <<"].0: " << (float)element0[0]
                  << ", " << (float)element1[0] << std::endl;
      }
      if (channels >= 3) {
        difference = fabs((float) element0[1] - (float) element1[1]);
        findMax(max, difference);
        if (difference > epsilon || display) {
          std::cout << "[" << row << ", " << col <<"].1: " << (float)element0[1]
                    << ", " << (float)element1[1] << std::endl;
        }
        difference = fabs((float) element0[2] - (float) element1[2]);
        findMax(max, difference);
        if (difference > epsilon || display) {
          std::cout << "[" << row << ", " << col <<"].2: " << (float)element0[2]
                    << ", " << (float)element1[2] << std::endl;
        }
      }
      if (channels == 4) {
        difference = fabs((float) element0[3] - (float) element1[3]);
        findMax(max, difference);
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

#endif  // _ST_HPC_PPL3_CV_CUDA_TEST_INFRASTRUCTURE_HPP_
