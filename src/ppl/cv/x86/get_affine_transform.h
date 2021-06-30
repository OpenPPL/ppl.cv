#ifndef __ST_HPC_PPL3_CV_X86_GET_AFFINE_TRANSFORM_H_
#define __ST_HPC_PPL3_CV_X86_GET_AFFINE_TRANSFORM_H_

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Calculates an affine transform from three pairs of the corresponding points.
* @param src_points               src keypoints data, which has 3 points and each points has 2 double values (x, y)
* @param dst_points               dst keypoints data, which has 3 points and each points has 2 double values (x, y)
* @param mat                      transform matrix, 2x3 matrix. can be nullptr.
* @param inverse_mat              transform matrix inversed, 2x3 matrix. can be nullptr.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* <caption align="left">Requirements</caption>
* <tr><td>x86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/get_affine_transform.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/get_affine_transform.h>
* int32_t main(int32_t argc, char** argv) {
*     double src_points[6], dst_points[6], mat[6], inverse_mat[6];
*     src_points[0] = 5;   src_points[1] = 9;
*     src_points[2] = 223; src_points[3] = 13;
*     src_points[4] = 49;  src_points[5] = 146;
*
*     dst_points[0] = 27;  dst_points[1] = 19;
*     dst_points[2] = 103; dst_points[3] = 47;
*     dst_points[4] = 18;  dst_points[5] = 91;
*
*     ppl::cv::x86::GetAffineTransform(src_points, dst_points, mat, inverse_mat);
*     return 0;
* }
* @endcode
***************************************************************************************************/

void GetAffineTransform(
    const double *src_points,
    const double *dst_points,
    double *mat,
    double *inverse_mat);

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //!__ST_HPC_PPL3_CV_X86_GET_AFFINE_TRANSFORM_H_

