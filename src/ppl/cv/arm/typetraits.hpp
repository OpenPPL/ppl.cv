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

#ifndef __ST_HPC_PPL_CV_AARCH64_TYPETRAITS_H_
#define __ST_HPC_PPL_CV_AARCH64_TYPETRAITS_H_
#include <stdint.h>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

template <int nc, typename T>
struct DT;

template <>
struct DT<1, uint8_t> {
    typedef uint8x8_t vec_DT;
};
template <>
struct DT<2, uint8_t> {
    typedef uint8x8x2_t vec_DT;
};
template <>
struct DT<3, uint8_t> {
    typedef uint8x8x3_t vec_DT;
};
template <>
struct DT<4, uint8_t> {
    typedef uint8x8x4_t vec_DT;
};
template <>
struct DT<1, float> {
    typedef float32x4_t vec_DT;
};
template <>
struct DT<2, float> {
    typedef float32x4x2_t vec_DT;
};
template <>
struct DT<3, float> {
    typedef float32x4x3_t vec_DT;
};
template <>
struct DT<4, float> {
    typedef float32x4x4_t vec_DT;
};

template <int nc, typename Tptr, typename T>
inline void vstx_u8_f32(Tptr *ptr, T vec);
template <int nc, typename Tptr, typename T>
inline T vldx_u8_f32(const Tptr *ptr);

template <>
inline uint8x8_t vldx_u8_f32<1, uint8_t, uint8x8_t>(const uint8_t *ptr)
{
    return vld1_u8(ptr);
};
template <>
inline uint8x8x2_t vldx_u8_f32<2, uint8_t, uint8x8x2_t>(const uint8_t *ptr)
{
    return vld2_u8(ptr);
};
template <>
inline uint8x8x3_t vldx_u8_f32<3, uint8_t, uint8x8x3_t>(const uint8_t *ptr)
{
    return vld3_u8(ptr);
};
template <>
inline uint8x8x4_t vldx_u8_f32<4, uint8_t, uint8x8x4_t>(const uint8_t *ptr)
{
    return vld4_u8(ptr);
};
template <>
inline float32x4_t vldx_u8_f32<1, float, float32x4_t>(const float *ptr)
{
    return vld1q_f32(ptr);
};
template <>
inline float32x4x2_t vldx_u8_f32<2, float, float32x4x2_t>(const float *ptr)
{
    return vld2q_f32(ptr);
};
template <>
inline float32x4x3_t vldx_u8_f32<3, float, float32x4x3_t>(const float *ptr)
{
    return vld3q_f32(ptr);
};
template <>
inline float32x4x4_t vldx_u8_f32<4, float, float32x4x4_t>(const float *ptr)
{
    return vld4q_f32(ptr);
};

template <>
inline void vstx_u8_f32<1, uint8_t, uint8x8_t>(uint8_t *ptr, uint8x8_t vec)
{
    vst1_u8(ptr, vec);
}
template <>
inline void vstx_u8_f32<2, uint8_t, uint8x8x2_t>(uint8_t *ptr, uint8x8x2_t vec)
{
    vst2_u8(ptr, vec);
}
template <>
inline void vstx_u8_f32<3, uint8_t, uint8x8x3_t>(uint8_t *ptr, uint8x8x3_t vec)
{
    vst3_u8(ptr, vec);
}
template <>
inline void vstx_u8_f32<4, uint8_t, uint8x8x4_t>(uint8_t *ptr, uint8x8x4_t vec)
{
    vst4_u8(ptr, vec);
}
template <>
inline void vstx_u8_f32<1, float, float32x4_t>(float *ptr, float32x4_t vec)
{
    vst1q_f32(ptr, vec);
}
template <>
inline void vstx_u8_f32<2, float, float32x4x2_t>(float *ptr, float32x4x2_t vec)
{
    vst2q_f32(ptr, vec);
}
template <>
inline void vstx_u8_f32<3, float, float32x4x3_t>(float *ptr, float32x4x3_t vec)
{
    vst3q_f32(ptr, vec);
}
template <>
inline void vstx_u8_f32<4, float, float32x4x4_t>(float *ptr, float32x4x4_t vec)
{
    vst4q_f32(ptr, vec);
}
template <>
inline void vstx_u8_f32<3, uint8_t, uint8x16x3_t>(uint8_t *ptr, uint8x16x3_t vec)
{
    vst3q_u8(ptr, vec);
}
template <>
inline void vstx_u8_f32<4, uint8_t, uint8x16x4_t>(uint8_t *ptr, uint8x16x4_t vec)
{
    vst4q_u8(ptr, vec);
}

}
}
} // namespace ppl::cv::arm

#endif //! __ST_HPC_PPL_CV_AARCH64_TYPETRAITS_H_
