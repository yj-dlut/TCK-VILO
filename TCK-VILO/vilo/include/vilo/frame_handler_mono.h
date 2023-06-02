#ifndef VILO_FRAME_HANDLER_H_
#define VILO_FRAME_HANDLER_H_

#include <set>
#include <boost/thread.hpp>

#include "vilo/vikit/performance_monitor.h"
#include "vilo/frame_handler_base.h"
#include "vilo/reprojector.h"
#include "vilo/initialization.h"
#include "vilo/PhotomatricCalibration.h"
#include "vilo/camera.h"
#include "vilo/imu.h"
#include "vilo/CoarseTracker.h"
#include "vilo/lidar.h"
#include "vilo/config.h"
#include "vilo/map.h"
#include "vilo/frame.h"
#include "vilo/feature.h"
#include "vilo/point.h"
#include "vilo/pose_optimizer.h"
#include "vilo/matcher.h"
#include "vilo/feature_alignment.h"
#include "vilo/global.h"
#include "vilo/depth_filter.h"

namespace vilo {

class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  FrameHandlerMono(vilo::AbstractCamera* cam, bool _use_pc=false);
  FrameHandlerMono(std::string& settingfile, bool _use_pc=false);//vio
  virtual ~FrameHandlerMono();

  void addImageImu(const cv::Mat& image, 
                  const std::vector<vilo::IMUPoint>& imu_vec, 
                  const double& img_time,
                  UpdateResult& res);


  void addLidarImu(pcl::PointCloud<pcl::PointXYZI>::Ptr p_cloud,
                   const std::vector<vilo::IMUPoint>& imu_vec, 
                   const double& timestamp,
                   std::vector<double>& pts_ts_vec);

  void addLidarImageImu(pcl::PointCloud<pcl::PointXYZI>::Ptr lidar,
                        cv::Mat& image,
                        const std::vector<vilo::IMUPoint>& imu_vec,
                        std::vector<double>& pts_ts_vec, 
                        const double& lidar_ts,
                        const double& image_ts,
                        int imu_id1, int imu_id2,
                        int imu_id3, int imu_id4,
                        int type);

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the last lidar that has been processed.
  LidarPtr lastLidar() { return last_lidar_; }

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

public:

  //std::vector<int> ids_; 
  std::vector<double> times_;
  std::vector<Sophus::SE3> gt_poses_; 
  std::list< pair<long, Sophus::SE3> > visual_motions_;
  std::list< pair<long, Sophus::SE3> > lidar_motions_;
  list< LidarPtr > key_lidars_list_; 
  Lidar* last_key_lidar_; 

  std::string setting_file_;
  vilo::IMU* imu_from_last_keyframe_;               
  vilo::IMU* imu_from_last_keylidar_; 
  bool is_imu_used_;
  // bool is_imu_initialized_;
  bool is_vio_initialized_;
  bool is_lio_initialized_;
  bool is_livo_initialized_;
  bool is_lidar_img_pair_;
  long livo_init_lidar_id_;
  
  float imu_freq_, imu_na_, imu_ng_, imu_naw_, imu_ngw_;
  bool is_third_opt_ = false; 
  //bool is_map_changed_;

  FramePtr last_keyframe_;
  SE3 T_b_c_;
  //SE3 T_c_b_;
  vilo::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  Reprojector* reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  FramePtr first_frame_;
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  //vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.
  // Sophus::SE3 motion_model_;
  Sophus::SE3 lidar_motion_model_, visual_motion_model_;
  bool after_init_ = false;
  PhotomatricCalibration* photomatric_calib_;
  vector<Frame*> covisible_kfs_; 
  set<Frame*> sub_map_;
  size_t n_matched_fts_, n_edges_final_; 
  double distance_mean_, depth_min_;

  LidarPtr last_lidar_;
  LidarPtr new_lidar_;
  int n_lidar_;
  Sophus::SE3 T_b_l_;
  long lio_init_id_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_flat_pts_;
  int step_ = 5;
  bool is_first_lidar_;
  

  bool readConfigFile(std::string& settingfile);
  
  void preintegrate(const std::vector<vilo::IMUPoint>& imu_vec, int type); 
  void predictWithImu();
  // void calcInitialTransform(const std::vector<vilo::IMUPoint>& imu_vec, Sophus::SE3& dT); 
  void trackNewLidarFrame(LidarPtr& new_lidar_, LidarPtr& last_lidar_, Sophus::SE3& Trc); 
  // void trackLidarLocalMap(); 


  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(const SE3& T_cur_ref, FramePtr ref_keyframe);

  // void processKeyFrame(bool is_tracking_good, double distance_mean, double depth_min);
  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(
  const double& scene_depth_mean, 
  const size_t& num_observations);

  static bool frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs);
  // static bool frameCovisibilityComparatorF(pair<float, Frame*>& lhs, pair<float, Frame*>& rhs);

  void createCovisibilityGraph(FramePtr currentFrame, size_t n_closest, bool is_keyframe);

  // void prepareForPhotomatricCalibration();

  // void changeFrameEdgeLetNormal();
  // void calcMotionModel();
  // void setCoreKfs(size_t n_closest);

  // bool kfOverView();

  // void refineAffineWarp();
  // double getAffineResidual(
  //   FramePtr target, Matrix2d& hessian, Vector2d& Jres, vector<double>& errors,
  //   vector< pair<float, float> >& photometric, vector< pair<float, float> >& photoKF, vector< pair<float, float> >& photoRF);
 
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
  // in at least other 3 keyframes (in the same or finer scale)
  // We only consider close stereo points
  // void KeyFrameCulling();
 
  // int getDir(std::string dir, std::vector<std::string> &files);

  
  /// An external place recognition module may know where to relocalize.
  // bool relocalizeFrameAtPose(const int keyframe_id,
  //                            const SE3& T_kf_f,
  //                            const cv::Mat& img,
  //                            const double timestamp);

  /// Provide an image.
  // void addImage(const cv::Mat& img, double timestamp, string* timestamp_s=NULL);

  // void addLidar(pcl::PointCloud<pcl::PointXYZI>::Ptr p_cloud,
  //               const double& timestamp,
  //               std::vector<double>& pts_ts_vec);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  // void setFirstFrame(const FramePtr& first_frame);

  // void solveGyroscopeBias(list<LidarPtr>& lidars_list);
  //********************************FOR PAPER!!!**************************************//
  vector< pair<string, double> > m_stamp_et;
  vector<double> m_grad_mean;
  vector<Vector3d> m_vig;

  // void saveTimes(string dir);
  // void saveVignettingTotal(string dir);
  // void saveVignettingOnce();

};

} // namespace vilo

#endif // VILO_FRAME_HANDLER_H_
