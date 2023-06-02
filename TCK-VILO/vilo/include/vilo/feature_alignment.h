#ifndef VILO_FEATURE_ALIGNMENT_H_
#define VILO_FEATURE_ALIGNMENT_H_

#include <vilo/global.h>

namespace vilo {

class Point;
struct Feature;
/// Subpixel refinement of a reference feature patch with the current image.
/// Implements the inverse-compositional approach (see "Lucas-Kanade 20 Years on"
/// paper by Baker.
namespace feature_alignment {

bool align1D(
    const cv::Mat& cur_img,
    const Vector2f& dir,                  // direction in which the patch is allowed to move
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    double& h_inv);

bool align1D(
    const cv::Mat& cur_img,
    const Vector2f& dir,                  // direction in which the patch is allowed to move
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    double& h_inv,
    float* cur_patch = NULL);

bool align2D(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool no_simd = true,
    float* cur_patch = NULL);

bool align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool no_simd = true,
    float* cur_patch = NULL);

bool align2D_SSE2(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate);

bool align2D_NEON(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate);

} // namespace feature_alignment
} // namespace vilo

#endif // VILO_FEATURE_ALIGNMENT_H_
