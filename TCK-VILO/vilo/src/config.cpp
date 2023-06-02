#include <vilo/config.h>

namespace vilo {

Config::Config() :

    trace_name("vilo"),
    trace_dir("/tmp"),
    n_pyr_levels(3),
    use_imu(false),
    core_n_kfs(7),
    map_scale(1.0),
    grid_size(36),
    init_min_disparity(5.0),
    init_min_tracked(50),
    init_min_inliers(40),
    klt_max_level(4),
    klt_min_level(0),
    reproj_thresh(2.0),
    poseoptim_thresh(2.0),
    poseoptim_num_iter(10),
    structureoptim_max_pts(30),
    structureoptim_num_iter(5),
    loba_thresh(2.0),
    loba_robust_huber_width(1.0),
    loba_num_iter(10),
    kfselect_mindist(0.12),
    triang_min_corner_score(20.0),
    triang_half_patch_size(4),
    subpix_n_iter(10),
    max_n_kfs(2000),
    img_imu_delay(0.0),
    max_fts(200),
    quality_min_fts(0),
    quality_max_drop_fts(40),

    edgelet_angle(0.86),

    n_max_drop_keyframe(13)

{}

Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}

} // namespace vilo

