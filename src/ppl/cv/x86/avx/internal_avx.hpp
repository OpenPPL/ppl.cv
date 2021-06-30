#ifndef PPL_CV_X86_INTERNAL_AVX_H_
#define PPL_CV_X86_INTERNAL_AVX_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

template <int32_t dcn, int32_t bIdx>
::ppl::common::RetCode YUV420ptoRGB_avx(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData);

template <typename Tsrc, int32_t ncSrc, typename Tdst, int32_t ncDst>
::ppl::common::RetCode BGR2GRAYImage_avx(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData);

template <typename Tsrc, int32_t ncSrc, typename Tdst, int32_t ncDst>
::ppl::common::RetCode RGB2GRAYImage_avx(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData);

}
}
} // namespace ppl::cv::x86
#endif //! PPL_CV_X86_INTERNAL_AVX_H_
