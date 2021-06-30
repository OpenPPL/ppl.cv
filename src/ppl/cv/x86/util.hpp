#ifndef PPL_CV_X86_UTIL_H_
#define PPL_CV_X86_UTIL_H_

#include "ppl/cv/types.h"
#include <stdint.h>
#include <vector>
#include <cstring>
#include <cassert>

namespace ppl {
namespace cv {
namespace x86 {
inline const uint8_t sat_cast_u8(int32_t data)
{
    return data > 255 ? 255 : (data < 0 ? 0 : data);
}
template<typename T>
inline const T round(const T &a, const T &b)
{
    return a / b * b;
}

template<typename T>
inline const T round_up(const T &a, const T &b)
{
    return (a + b - static_cast<T>(1)) / b * b;
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data, int32_t n, FUNCTION f, T operand0) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = f(in_data[i], operand0);
    }
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data0, const T *in_data1, int32_t n, FUNCTION f) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = f(in_data0[i], in_data1[i]);
    }
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data, float alpha, int32_t n, FUNCTION f, T operand0) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = alpha * f(in_data[i], operand0);
    }
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data0, const T *in_data1, float alpha, int32_t n, FUNCTION f) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = alpha * f(in_data0[i], in_data1[i]);
    }
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! PPL_CV_X86_UTIL_H_
