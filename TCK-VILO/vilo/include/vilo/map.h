#ifndef VILO_MAP_H_
#define VILO_MAP_H_

#include <queue>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <vilo/global.h>

namespace vilo {

class Point;
class Feature;
class Seed;

/// Container for converged 3D points that are not already assigned to two keyframes.
class MapPointCandidates
{
public:
  typedef pair<Point*, Feature*> PointCandidate;
  // typedef pair< Point*, std::list<Feature*> > PointCandidate;
  typedef list<PointCandidate> PointCandidateList;

  /// The depth-filter is running in a parallel thread and fills the canidate list.
  /// This mutex controls concurrent access to point_candidates.
  boost::mutex mut_;
  boost::mutex candi_mutex_;

  /// Candidate points are created from converged seeds.
  /// Until the next keyframe, these points can be used for reprojection and pose optimization.
  PointCandidateList candidates_;
  list< Point* > trash_points_;
  list< pair<Point*, Feature*> > temporaryPoints_;

  MapPointCandidates();
  ~MapPointCandidates();

  /// Add a candidate point.
  void newCandidatePoint(Point* point, double depth_sigma2);

  /// Finish a candidate point.
  void addPauseSeedPoint(Point* point);

  /// Adds the feature to the frame and deletes candidate from list.
  void addCandidatePointToFrame(FramePtr frame);

  /// Remove a candidate point from the list of candidates.
  bool deleteCandidatePoint(Point* point);

  /// Remove all candidates that belong to a frame.
  void removeFrameCandidates(FramePtr frame);

  /// Reset the candidate list, remove and delete all points.
  void reset();

  void deleteCandidate(PointCandidate& c);

  void emptyTrash();


  void changeCandidatePosition(Frame* frame);
};

/// Map object which saves all keyframes which are in a map.
class Map : boost::noncopyable
{
public:
  list< FramePtr > keyframes_;          //!< List of keyframes in the map.
  list< Point* > trash_points_;         //!< A deleted point is moved to the trash bin. Now and then this is cleaned. One reason is that the visualizer must remove the points also.
  MapPointCandidates point_candidates_;

  long init_id_ = 0;
  list< LidarPtr > lidars_list_;
  bool is_map_changed_;
  double last_avr_li_err_;
  double last_avr_lidar_err_;
  double last_avr_visual_err_;
  double last_avr_vi_err_;
  int n_lidar_err_, n_visual_err_; 

  double avr_err_pt_;
  double avr_err_ls_;
  double avr_err_lr_;
  int v_err_num_, l_err_num_; 
  std::deque<double> avr_err_pt_deque_, avr_err_lr_deque_;
  std::deque<int> v_err_num_deque_, l_err_num_deque_;
  bool is_camera_very_bad_ = false, is_lidar_very_bad_ = false;
  int n_v_recovery_ = 0, n_l_recovery_ = 0; 
  
  boost::mutex map_mutex_; 
  
  std::map<long, Sophus::SE3> lid_poses_map_;
protected:
  boost::mutex frame_mutex_; 
  boost::mutex point_mutex_; 
  boost::mutex flag_mutex_; 
  

public:
  
  Map();
  ~Map();
  void scaleRotate(const Eigen::Matrix3d& Rwg, const double& scale, const Eigen::Vector3d& t);

  void scaleRotate(Sophus::SE3& Tbc, Eigen::Matrix3d& Rcl_orient, Eigen::Matrix3d& Rlc_position, Eigen::Vector3d& tlc_position, double& scale);

  /// void forceChangeVisualMap(Sophus::SE3& Tbc, float imu_freq, float imu_na, float imu_ng, float imu_naw, float imu_ngw);

  void rotateLidar(const Eigen::Matrix3d& Rgw, const Eigen::Vector3d& t);

  /// Reset the map. Delete all keyframes and reset the frame and point counters.
  void reset();

  /// Delete a point in the map and remove all references in keyframes to it.
  void safeDeletePoint(Point* pt);

  /// Moves the point to the trash queue which is cleaned now and then.
  void deletePoint(Point* pt);

  /// Moves the frame to the trash queue which is cleaned now and then.
  bool safeDeleteFrame(FramePtr frame);
  bool safeDeleteFrameID(int id);

  /// Remove the references between a point and a frame.
  void removePtFrameRef(Frame* frame, Feature* ftr);

  /// Add a new keyframe to the map.
  void addKeyframe(FramePtr new_keyframe);

  /// Given a frame, return all keyframes which have an overlapping field of view.
  void getCloseKeyframes(const FramePtr& frame, list< pair<FramePtr,double> >& close_kfs);// const;

  /// Return the keyframe which is spatially closest and has overlapping field of view.
  FramePtr getClosestKeyframe(const FramePtr& frame); // const;

  /// Return the keyframe which is furthest apart from pos.
  FramePtr getFurthestKeyframe(const Vector3d& pos); // const;

  bool getKeyframeById(const int id, FramePtr& frame); // const;

  /// Empty trash bin of deleted keyframes and map points. We don't delete the
  /// points immediately to ensure proper cleanup and to provide the visualizer
  /// a list of objects which must be removed.
  void emptyTrash();

  /// Return the keyframe which was last inserted in the map.
  inline FramePtr lastKeyframe() { return keyframes_.back(); }

  /// Return the number of keyframes in the map
  inline size_t size() const { return keyframes_.size(); }

  void safeDeleteTempPoint(pair<Point*, Feature*>& p);

  bool getMapChangedFlag();
  void setMapChangedFlag(bool flag);

  void addNewLidar(LidarPtr lidar);

};

/// A collection of debug functions to check the data consistency.
namespace map_debug {

void mapStatistics(Map* map);
void mapValidation(Map* map, int id);
void frameValidation(Frame* frame, int id);
void pointValidation(Point* point, int id);

} // namespace map_debug
} // namespace vilo

#endif // VILO_MAP_H_
