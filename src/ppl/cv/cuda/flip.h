/**
 * @file   flip.h
 * @brief  Interface declarations of image flipping operation.
 * @author Liheng Jian(jianliheng@sensetime.com)
 *
 * @copyright Copyright (c) 2014-2021 SenseTime Group Limited.
 */

#ifndef _ST_HPC_PPL3_CV_CUDA_FLIP_H_
#define _ST_HPC_PPL3_CV_CUDA_FLIP_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Flips a 2D image around vertical, horizontal, or both axes.
 * @tparam T The data type, used for both source matrix and destination matrix,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param inHeight       input image's height.
 * @param inWidth        input image's width need to be processed.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @param flipCode       a flag to specify how to flip the array; 0 means
 *                       flipping around the x-axis and positive value (for
 *                       example, 1) means flipping around y-axis. Negative
 *                       value (for example, -1) means flipping around both
 *                       axes.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/flip.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/flip.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* dev_input;
 *   float* dev_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&dev_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&dev_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Flip<float, 3>(stream, height, width, input_pitch / sizeof(float),
 *                  dev_input, output_pitch / sizeof(float), dev_output, 0);
 *   cudaStreamSynchronize(stream);
 *
 *   cudaFree(dev_input);
 *   cudaFree(dev_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Flip(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const T* inData,
                          int outWidthStride,
                          T* outData,
                          int flipCode);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif // _ST_HPC_PPL3_CV_CUDA_FLIP_H_
