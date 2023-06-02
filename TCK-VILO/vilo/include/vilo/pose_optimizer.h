#ifndef VILO_POSE_OPTIMIZER_H_
#define VILO_POSE_OPTIMIZER_H_

#include <vilo/global.h>
#include <vilo/feature.h>


namespace vilo {

using namespace Eigen;
using namespace Sophus;
using namespace std;

typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,2,6> Matrix26d;
typedef Matrix<double,6,1> Vector6d;


/// Motion-only bundle adjustment. Minimize the reprojection error of a single frame.
namespace pose_optimizer {

// void optimizeGaussNewton(
//     const double reproj_thresh,
//     const size_t n_iter,
//     const bool verbose,
//     FramePtr& frame,
//     double& estimated_scale,
//     double& error_init,
//     double& error_final,
//     size_t& num_obs);


// void optimizeLevenbergMarquardt2nd(
//     const double reproj_thresh, const size_t n_iter, const bool verbose,
//     FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
//     size_t& num_obs);

void optimizeLevenbergMarquardt3rd(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs);

// void optimizeLevenbergMarquardtMagnitude(
//     const double reproj_thresh, const size_t n_iter, const bool verbose,
//     FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
//     size_t& num_obs);


    // distribution
    static int residual_buffer[10000]={0};

} // namespace pose_optimizer
} // namespace vilo

#endif // VILO_POSE_OPTIMIZER_H_
