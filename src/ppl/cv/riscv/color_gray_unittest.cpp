#include "ppl/cv/riscv/cvtcolor.h"
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include "ppl/cv/riscv/test.h"

template <typename T>
class BGR2GRAY : public ::testing::TestWithParam<std::tuple<Size, int32_t, float>> {
public:
    using BGR2GRAYParam = std::tuple<Size, int32_t, float>;
    BGR2GRAY()
    {
    }

    ~BGR2GRAY()
    {
    }

    void apply(const BGR2GRAYParam &param)
    {
        Size size = std::get<0>(param);
        int32_t mode = std::get<1>(param);
        const float diff = std::get<2>(param);

        if (mode == 0) {
            int32_t input_channels = 3;
            int32_t output_channels = 1;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2GRAY);

            ppl::cv::riscv::BGR2GRAY<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 1>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }

        if (mode == 1) {
            int32_t input_channels = 1;
            int32_t output_channels = 3;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_GRAY2BGR);

            ppl::cv::riscv::GRAY2BGR<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 3>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }

        if (mode == 2) {
            int32_t input_channels = 4;
            int32_t output_channels = 1;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2GRAY);

            ppl::cv::riscv::BGRA2GRAY<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 1>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }

        if (mode == 3) {
            int32_t input_channels = 1;
            int32_t output_channels = 4;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_GRAY2BGRA);

            ppl::cv::riscv::GRAY2BGRA<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 4>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }

        if (mode == 4) {
            int32_t input_channels = 3;
            int32_t output_channels = 1;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2GRAY);

            ppl::cv::riscv::RGB2GRAY<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 1>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }

        if (mode == 5) {
            int32_t input_channels = 1;
            int32_t output_channels = 3;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_GRAY2RGB);

            ppl::cv::riscv::GRAY2RGB<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 3>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }

        if (mode == 6) {
            int32_t input_channels = 4;
            int32_t output_channels = 1;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2GRAY);

            ppl::cv::riscv::RGBA2GRAY<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 1>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }

        if (mode == 7) {
            int32_t input_channels = 1;
            int32_t output_channels = 4;
            std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
            std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
            std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

            ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

            cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
            cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_GRAY2RGBA);

            ppl::cv::riscv::GRAY2RGBA<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, 4>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }
    }
};

constexpr int32_t c1 = 1;
constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

#define R(name, t, ic, oc, mode, diff) \
    using name = BGR2GRAY<t>;          \
    TEST_P(name, abc)                  \
    {                                  \
        this->apply(GetParam());       \
    }                                  \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(mode), ::testing::Values(diff)));
R(UT_BGR2GRAY_float_riscv, float, c3, c1, 0, 1e-4)
R(UT_BGR2GRAY_uint8_t_riscv, uint8_t, c3, c1, 0, 1.01)
R(UT_GRAY2BGR_float_riscv, float, c1, c3, 1, 1e-4)
R(UT_GRAY2BGR_uint8_t_riscv, uint8_t, c1, c3, 1, 1.01)
R(UT_BGRA2GRAY_float_riscv, float, c4, c1, 2, 1e-4)
R(UT_BGRA2GRAY_uint8_t_riscv, uint8_t, c4, c1, 2, 1.01)
R(UT_GRAY2BGRA_float_riscv, float, c1, c4, 3, 1e-4)
R(UT_GRAY2BGRA_uint8_t_riscv, uint8_t, c1, c4, 3, 1.01)
R(UT_RGB2GRAY_float_riscv, float, c3, c1, 4, 1e-4)
R(UT_RGB2GRAY_uint8_t_riscv, uint8_t, c3, c1, 4, 1.01)
R(UT_GRAY2RGB_float_riscv, float, c1, c3, 5, 1e-4)
R(UT_GRAY2RGB_uint8_t_riscv, uint8_t, c1, c3, 5, 1.01)
R(UT_RGBA2GRAY_float_riscv, float, c4, c1, 6, 1e-4)
R(UT_RGBA2GRAY_uint8_t_riscv, uint8_t, c4, c1, 6, 1.01)
R(UT_GRAY2RGBA_float_riscv, float, c1, c4, 7, 1e-4)
R(UT_GRAY2RGBA_uint8_t_riscv, uint8_t, c1, c4, 7, 1.01)
