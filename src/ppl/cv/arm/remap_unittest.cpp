// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for mulitional information
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

#include "ppl/cv/arm/remap.h"
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include "ppl/cv/arm/test.h"
#include <math.h>


struct remapSize {
    int32_t inWidth;
    int32_t inHeight;
    int32_t outWidth;
    int32_t outHeight;
};

template<typename T, int32_t c, bool inter_linear, ppl::cv::BorderType borderMode>
class Remap : public ::testing::TestWithParam<std::tuple<remapSize, float>> {
public:
    using RemapParam = std::tuple<remapSize, float>;
    Remap()
    {
    }

    ~Remap()
    {
    }

    void apply(const RemapParam &param) {
        remapSize size = std::get<0>(param);
        const float diff = std::get<1>(param);
        std::unique_ptr<T[]> src(new T[size.inWidth * size.inHeight * c]);
        std::unique_ptr<T[]> dst_ref(new T[size.outWidth * size.outHeight * c]);
        std::unique_ptr<T[]> dst(new T[size.outWidth * size.outHeight * c]);
        std::unique_ptr<float[]> map_x(new float[size.outWidth * size.outHeight * 1]);
        std::unique_ptr<float[]> map_y(new float[size.outWidth * size.outHeight * 1]);
        memset(dst.get(), 0, sizeof(T) * size.outWidth * size.outHeight * c);
        memset(dst_ref.get(), 0, sizeof(T) * size.outWidth * size.outHeight * c);

        ppl::cv::debug::randomFill<T>(src.get(), size.inWidth * size.inHeight * c, 0, 255);
        ppl::cv::debug::randomFill<float>(map_x.get(), size.outWidth * size.outHeight * 1, -10, size.inWidth+10);
        ppl::cv::debug::randomFill<float>(map_y.get(), size.outWidth * size.outHeight * 1, -10, size.inHeight+10);
        double border_value;
        ppl::cv::debug::randomFill<double>(&border_value, 1, 0, 255);

        
        if(inter_linear == true) {
            ppl::cv::arm::RemapLinear<T, c>(size.inHeight, size.inWidth, size.inWidth * c, src.get(), 
                                            size.outHeight, size.outWidth, size.outWidth * c, dst.get(), 
                                            map_x.get(), map_y.get(), borderMode, (T)border_value);    
        } else {
            ppl::cv::arm::RemapNearestPoint<T, c>(size.inHeight, size.inWidth, size.inWidth * c, src.get(), 
                                            size.outHeight, size.outWidth, size.outWidth * c, dst.get(), 
                                            map_x.get(), map_y.get(), borderMode, (T)border_value);
        }
        
        cv::Mat src_opencv(size.inHeight, size.inWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), src.get(), sizeof(T) * size.inWidth * c);
        cv::Mat dst_opencv(size.outHeight, size.outWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), dst_ref.get(), sizeof(T) * size.outWidth * c);
        cv::Mat mapy_opencv(size.outHeight, size.outWidth, CV_MAKETYPE(cv::DataType<float>::depth, 1), map_y.get(), sizeof(float) * size.outWidth);
        cv::Mat mapx_opencv(size.outHeight, size.outWidth, CV_MAKETYPE(cv::DataType<float>::depth, 1), map_x.get(), sizeof(float) * size.outWidth);
        cv::Scalar borderValue = {border_value, border_value, border_value, border_value};
        
        int32_t border_mode = borderMode == ppl::cv::BORDER_CONSTANT ? cv::BORDER_CONSTANT : 
                         (borderMode == ppl::cv::BORDER_REPLICATE ? cv::BORDER_REPLICATE : cv::BORDER_TRANSPARENT);
        cv::remap(src_opencv, dst_opencv, mapx_opencv, mapy_opencv, (inter_linear ? cv::INTER_LINEAR : cv::INTER_NEAREST), 
                  border_mode, borderValue);
        checkResult<T, c>(dst_ref.get(), dst.get(), size.outHeight, 
                          size.outWidth, size.outWidth * c, size.outWidth * c, diff);
    }
};

#define R(name, t, c, inter_linear, border_type, diff)\
    using name = Remap<t, c, inter_linear, border_type>;\
    TEST_P(name, abc)\
    {\
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
        ::testing::Combine(::testing::Values(remapSize{320, 240, 643, 481}, remapSize{640, 480, 321, 241}),\
                           ::testing::Values(diff)));
    
R(Remap_u8c1_linear_constant, uint8_t, 1, true, ppl::cv::BORDER_CONSTANT, 1.01f)
R(Remap_u8c3_linear_constant, uint8_t, 3, true, ppl::cv::BORDER_CONSTANT, 1.01f)
R(Remap_u8c4_linear_constant, uint8_t, 4, true, ppl::cv::BORDER_CONSTANT, 1.01f)
R(Remap_f32c1_linear_constant, float, 1, true, ppl::cv::BORDER_CONSTANT, 1e-3)  
R(Remap_f32c3_linear_constant, float, 3, true, ppl::cv::BORDER_CONSTANT, 1e-3)
R(Remap_f32c4_linear_constant, float, 4, true, ppl::cv::BORDER_CONSTANT, 1e-3)

R(Remap_u8c1_nearest_constant, uint8_t, 1, false, ppl::cv::BORDER_CONSTANT, 1.01f)
R(Remap_u8c3_nearest_constant, uint8_t, 3, false, ppl::cv::BORDER_CONSTANT, 1.01f)
R(Remap_u8c4_nearest_constant, uint8_t, 4, false, ppl::cv::BORDER_CONSTANT, 1.01f)
R(Remap_f32c1_nearest_constant, float, 1, false, ppl::cv::BORDER_CONSTANT, 1e-3) 
R(Remap_f32c3_nearest_constant, float, 3, false, ppl::cv::BORDER_CONSTANT, 1e-3)
R(Remap_f32c4_nearest_constant, float, 4, false, ppl::cv::BORDER_CONSTANT, 1e-3)

R(Remap_u8c1_linear_replicate, uint8_t, 1, true, ppl::cv::BORDER_REPLICATE, 1.01f)
R(Remap_u8c3_linear_replicate, uint8_t, 3, true, ppl::cv::BORDER_REPLICATE, 1.01f)
R(Remap_u8c4_linear_replicate, uint8_t, 4, true, ppl::cv::BORDER_REPLICATE, 1.01f)
R(Remap_f32c1_linear_replicate, float, 1, true, ppl::cv::BORDER_REPLICATE, 1e-3)  
R(Remap_f32c3_linear_replicate, float, 3, true, ppl::cv::BORDER_REPLICATE, 1e-3)
R(Remap_f32c4_linear_replicate, float, 4, true, ppl::cv::BORDER_REPLICATE, 1e-3)

R(Remap_u8c1_nearest_replicate, uint8_t, 1, false, ppl::cv::BORDER_REPLICATE, 1.01f)
R(Remap_u8c3_nearest_replicate, uint8_t, 3, false, ppl::cv::BORDER_REPLICATE, 1.01f)
R(Remap_u8c4_nearest_replicate, uint8_t, 4, false, ppl::cv::BORDER_REPLICATE, 1.01f)
R(Remap_f32c1_nearest_replicate, float, 1, false, ppl::cv::BORDER_REPLICATE, 1e-3) 
R(Remap_f32c3_nearest_replicate, float, 3, false, ppl::cv::BORDER_REPLICATE, 1e-3)
R(Remap_f32c4_nearest_replicate, float, 4, false, ppl::cv::BORDER_REPLICATE, 1e-3)

R(Remap_u8c1_linear_transparent, uint8_t, 1, true, ppl::cv::BORDER_TRANSPARENT, 1.01f)
R(Remap_u8c3_linear_transparent, uint8_t, 3, true, ppl::cv::BORDER_TRANSPARENT, 1.01f)
R(Remap_u8c4_linear_transparent, uint8_t, 4, true, ppl::cv::BORDER_TRANSPARENT, 1.01f)
R(Remap_f32c1_linear_transparent, float, 1, true, ppl::cv::BORDER_TRANSPARENT, 1e-3) 
R(Remap_f32c3_linear_transparent, float, 3, true, ppl::cv::BORDER_TRANSPARENT, 1e-3)
R(Remap_f32c4_linear_transparent, float, 4, true, ppl::cv::BORDER_TRANSPARENT, 1e-3)

R(Remap_u8c1_nearest_transparent, uint8_t, 1, false, ppl::cv::BORDER_TRANSPARENT, 1.01f)
R(Remap_u8c3_nearest_transparent, uint8_t, 3, false, ppl::cv::BORDER_TRANSPARENT, 1.01f)
R(Remap_u8c4_nearest_transparent, uint8_t, 4, false, ppl::cv::BORDER_TRANSPARENT, 1.01f)
R(Remap_f32c1_nearest_transparent, float, 1, false, ppl::cv::BORDER_TRANSPARENT, 1e-3)
R(Remap_f32c3_nearest_transparent, float, 3, false, ppl::cv::BORDER_TRANSPARENT, 1e-3)
R(Remap_f32c4_nearest_transparent, float, 4, false, ppl::cv::BORDER_TRANSPARENT, 1e-3)



