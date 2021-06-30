#include "ppl/cv/x86/get_rotation_matrix2d.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <time.h>
#include "ppl/common/retcode.h"
#include "ppl/cv/debug.h"

#define CHECK_RESULT(a, b) \
        EXPECT_LT(abs(a-b), diff_THR);

void GetRotationMatrix2DTest(float center_y, float center_x, double angle, double scale) {
    
    std::unique_ptr<double[]> dst(new double[6]);
    ppl::cv::x86::GetRotationMatrix2D(center_y, center_x, angle, scale, dst.get());
    
    cv::Point2f center;
    center.x = center_x;
    center.y = center_y;
    cv::Matx23d oMat = cv::getRotationMatrix2D(center, angle, scale);
    float diff_THR = 1e-4;
    CHECK_RESULT(oMat(0, 0), dst[0]);
    CHECK_RESULT(oMat(0, 1), dst[1]);
    CHECK_RESULT(oMat(0, 2), dst[2]);
    CHECK_RESULT(oMat(1, 0), dst[3]);
    CHECK_RESULT(oMat(1, 1), dst[4]);
    CHECK_RESULT(oMat(1, 2), dst[5]);
}
	

TEST(GetRotationMatrix2DTest, x86)
{
    GetRotationMatrix2DTest(320, 360, 45, 1.0);
    GetRotationMatrix2DTest(320, 360, 30, 1.0);
    GetRotationMatrix2DTest(320, 360, 90, 1.0);
    GetRotationMatrix2DTest(320, 360, -30, 1.0);
    GetRotationMatrix2DTest(320, 360, -45, 1.0);
    GetRotationMatrix2DTest(320, 360, -90, 1.0);
}