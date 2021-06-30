#ifndef __ST_HPC_PPL_CV_X86_ERODE_H_
#define __ST_HPC_PPL_CV_X86_ERODE_H_

#include "ppl/common/retcode.h"
#include <ppl/cv/types.h>
namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Denoise or obscure an image with erode alogrithm.
 * @tparam T The data type of input and output image, currently only \a uint8_t and \a float are supported.
 * @tparam channels The number of channels of input and output image, 1, 3 and 4 are supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param kernelx_len       the length of mask , x direction.
 * @param kernely_len       the length of mask , y direction.
 * @param kernel           the data of the mask.
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @param border_type       ways to deal with border. Only BORDER_TYPE_CONSTANT is supported now.
 * @param border_value      filling border_value for BORDER_TYPE_CONSTANT
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
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
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/erode.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/erode.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t C = 3;
 *     const int32_t kernelx_len = 3;
 *     const int32_t kernely_len = 3;
 *     (float*)dev_iImage = (float*)malloc(W * H * C * sizeof(T));
 *     (float*)dev_oImage = (float*)malloc(W * H * C * sizeof(T));
 *     (unsigned char*)kernel = (unsigned char*)malloc(kernel_len * kernel_len * sizeof(unsigned char));
 *     ppl::cv::x86::Erode<float, 3>(H, W, W * C, dev_iImage, kernelx_len, kernely_len, kernel, W * C, dev_oImage, ppl::cv::BORDER_TYPE_CONSTANT);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     free(kernel);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template<typename T, int32_t numChannels>
::ppl::common::RetCode Erode(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const unsigned char* kernel,
    int32_t outWidthStride,
    T* outData,
    BorderType border_type = BORDER_TYPE_CONSTANT,
    T border_value = 0);

} //! namespace x86
} //! namespace cv
} //! namespace ppl
#endif //! __ST_HPC_PPL3_CV_X86_ERODE_H_

