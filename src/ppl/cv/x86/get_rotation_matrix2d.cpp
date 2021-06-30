#include "ppl/cv/x86/get_rotation_matrix2d.h"

#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

#include <cmath>

#define CV_PI 3.1415926535897932384626433832795

namespace ppl {
namespace cv { 
namespace x86 {

::ppl::common::RetCode GetRotationMatrix2D(
    float center_y,
    float center_x,
    double angle,
    double scale,
    double* out_data) {

    if (nullptr == out_data) {
        return ppl::common::RC_INVALID_VALUE;
    }

    angle *= CV_PI / 180.0;
    double alpha = std::cos(angle) * scale;
    double beta = std::sin(angle) * scale;

    out_data[0] = alpha;
    out_data[1] = beta;
    out_data[2] = (1 - alpha) * center_x - beta * center_y;
    out_data[3] = -beta;
    out_data[4] = alpha;
    out_data[5] = beta * center_x + (1 - alpha) * center_y;

    return ppl::common::RC_SUCCESS;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl
