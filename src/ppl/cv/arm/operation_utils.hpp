#ifndef OP_UTILS_HPP
#define OP_UTILS_HPP

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>

namespace HPC { namespace utils {
template <typename T>
static inline T min(T x, T y)
{
    return (x < y) ? x : y;
}

template <typename T>
static inline T max(T x, T y)
{
    return (x > y) ? x : y;
}

static inline int round(int v)
{
    return v;
}
// !!! ONLY AVAILABLE WHEN v >= 0
static inline int round(float v)
{
    return (int)(v + 0.5f);
}
// !!! ONLY AVAILABLE WHEN v >= 0
static inline int round(double v)
{
    return (int)(v + 0.5);
}

template <typename _Tp>
static inline _Tp saturate_cast(uint8_t v)
{
    return _Tp(v);
}
template <typename _Tp>
static inline _Tp saturate_cast(int8_t v)
{
    return _Tp(v);
}
template <typename _Tp>
static inline _Tp saturate_cast(uint16_t v)
{
    return _Tp(v);
}
template <typename _Tp>
static inline _Tp saturate_cast(int16_t v)
{
    return _Tp(v);
}
template <typename _Tp>
static inline _Tp saturate_cast(unsigned v)
{
    return _Tp(v);
}
template <typename _Tp>
static inline _Tp saturate_cast(int v)
{
    return _Tp(v);
}
template <typename _Tp>
static inline _Tp saturate_cast(float v)
{
    return _Tp(v);
}
template <typename _Tp>
static inline _Tp saturate_cast(double v)
{
    return _Tp(v);
}

template <>
inline uint8_t saturate_cast<uint8_t>(int8_t v)
{
    return (uint8_t)utils::max((int)v, 0);
}
template <>
inline uint8_t saturate_cast<uint8_t>(uint16_t v)
{
    return (uint8_t)utils::min((unsigned)v, (unsigned)UCHAR_MAX);
}
template <>
inline uint8_t saturate_cast<uint8_t>(int v)
{
    return (uint8_t)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX
                                                          : 0);
}
template <>
inline uint8_t saturate_cast<uint8_t>(int16_t v)
{
    return v >= UCHAR_MAX ? UCHAR_MAX : v < 0 ? 0
                                              : v;
}
template <>
inline uint8_t saturate_cast<uint8_t>(unsigned v)
{
    return (uint8_t)utils::min(v, (unsigned)UCHAR_MAX);
}
template <>
inline uint8_t saturate_cast<uint8_t>(float v)
{
    int iv = utils::round(v);
    return saturate_cast<uint8_t>(iv);
}
template <>
inline uint8_t saturate_cast<uint8_t>(double v)
{
    int iv = utils::round(v);
    return saturate_cast<uint8_t>(iv);
}

template <>
inline int8_t saturate_cast<int8_t>(uint8_t v)
{
    return (int8_t)utils::min((int)v, SCHAR_MAX);
}
template <>
inline int8_t saturate_cast<int8_t>(uint16_t v)
{
    return (int8_t)utils::min((unsigned)v, (unsigned)SCHAR_MAX);
}
template <>
inline int8_t saturate_cast<int8_t>(int v)
{
    return (int8_t)((unsigned)(v - SCHAR_MIN) <= (unsigned)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX
                                                                                 : SCHAR_MIN);
}
template <>
inline int8_t saturate_cast<int8_t>(int16_t v)
{
    return saturate_cast<int8_t>((int)v);
}
template <>
inline int8_t saturate_cast<int8_t>(unsigned v)
{
    return (int8_t)utils::min(v, (unsigned)SCHAR_MAX);
}

template <>
inline int8_t saturate_cast<int8_t>(float v)
{
    int iv = utils::round(v);
    return saturate_cast<int8_t>(iv);
}
template <>
inline int8_t saturate_cast<int8_t>(double v)
{
    int iv = utils::round(v);
    return saturate_cast<int8_t>(iv);
}

template <>
inline uint16_t saturate_cast<uint16_t>(int8_t v)
{
    return (uint16_t)utils::max((int)v, 0);
}
template <>
inline uint16_t saturate_cast<uint16_t>(int16_t v)
{
    return (uint16_t)utils::max((int)v, 0);
}
template <>
inline uint16_t saturate_cast<uint16_t>(int v)
{
    return (uint16_t)((unsigned)v <= (unsigned)USHRT_MAX ? v : v > 0 ? USHRT_MAX
                                                                     : 0);
}
template <>
inline uint16_t saturate_cast<uint16_t>(unsigned v)
{
    return (uint16_t)utils::min(v, (unsigned)USHRT_MAX);
}
template <>
inline uint16_t saturate_cast<uint16_t>(float v)
{
    int iv = utils::round(v);
    return saturate_cast<uint16_t>(iv);
}
template <>
inline uint16_t saturate_cast<uint16_t>(double v)
{
    int iv = utils::round(v);
    return saturate_cast<uint16_t>(iv);
}

template <>
inline int16_t saturate_cast<int16_t>(uint16_t v)
{
    return (int16_t)utils::min((int)v, SHRT_MAX);
}
template <>
inline int16_t saturate_cast<int16_t>(int v)
{
    return (int16_t)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX
                                                                                 : SHRT_MIN);
}
template <>
inline int16_t saturate_cast<int16_t>(unsigned v)
{
    return (int16_t)utils::min(v, (unsigned)SHRT_MAX);
}
template <>
inline int16_t saturate_cast<int16_t>(float v)
{
    int iv = utils::round(v);
    return saturate_cast<int16_t>(iv);
}
template <>
inline int16_t saturate_cast<int16_t>(double v)
{
    int iv = utils::round(v);
    return saturate_cast<int16_t>(iv);
}

template <>
inline int saturate_cast<int>(float v)
{
    return utils::round(v);
}
template <>
inline int saturate_cast<int>(double v)
{
    return utils::round(v);
}

// we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
template <>
inline unsigned saturate_cast<unsigned>(float v)
{
    return utils::round(v);
}
template <>
inline unsigned saturate_cast<unsigned>(double v)
{
    return utils::round(v);
}
}
} // namespace HPC::utils

#define INTER_RESIZE_COEF_BITS  (11)
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define MAX_ESIZE               16
#define FUNC_MIN(a, b)          ((a) > (b) ? (b) : (a))
#define FUNC_MAX(a, b)          ((a) < (b) ? (b) : (a))
#ifndef SHRT_MIN
#define SHRT_MIN -32768
#endif
#ifndef SHRT_MAX
#define SHRT_MAX 32767
#endif

inline void prefetch(const void *ptr, size_t offset = 32 * 10)
{
    __builtin_prefetch(reinterpret_cast<const char *>(ptr) + offset);
}

inline void prefetch_l1(const void *ptr, size_t offset)
{
    asm volatile(
        "prfm pldl1keep, [%0, %1]\n\t"
        :
        : "r"(ptr), "r"(offset)
        : "cc", "memory");
}

static inline uint32_t align_size(int32_t sz, int32_t n)
{
    return (sz + n - 1) & -n;
}

static inline int32_t img_floor(float a)
{
    return (((a) >= 0) ? ((int32_t)a) : ((int32_t)a - 1));
}

static inline int32_t img_floor(double a)
{
    return (((a) >= 0) ? ((int32_t)a) : ((int32_t)a - 1));
}

static inline int32_t img_clip(int32_t x, int32_t a, int32_t b)
{
    return (x >= a ? (x < b ? x : b - 1) : a);
}

static inline unsigned char img_cast_op(int32_t val)
{
    int32_t bits  = INTER_RESIZE_COEF_BITS * 2;
    int32_t SHIFT = bits;
    int32_t DELTA = 1 << (bits - 1);
    int32_t temp  = FUNC_MIN(255, FUNC_MAX(0, (val + DELTA) >> SHIFT));
    return (unsigned char)(temp);
};

static inline int32_t img_round(double value)
{
    double intpart, fractpart;
    fractpart = modf(value, &intpart);
    if (fabs(fractpart) != 0.5 || ((((int32_t)intpart) % 2) != 0))
        return (int32_t)(value + (value >= 0 ? 0.5 : -0.5));
    else
        return (int32_t)intpart;
}
static inline int16_t img_saturate_cast_short(float x)
{
    int32_t iv = img_round(x);

    //return (x > SHRT_MIN ? (x < SHRT_MAX ? x : SHRT_MAX) : SHRT_MIN);
    return (iv > SHRT_MIN ? (iv < SHRT_MAX ? iv : SHRT_MAX) : SHRT_MIN);
}

inline int32_t Floor(double value)
{
    int32_t i = (int32_t)value;
    return i - (i > value);
}

typedef struct {
    int32_t height;
    int32_t width;
} Size;
#endif
