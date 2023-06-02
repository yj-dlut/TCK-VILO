#ifndef VILO_FRAME_HANDLER_BASE_H_
#define VILO_FRAME_HANDLER_BASE_H_

#include <queue>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <vilo/global.h>
#include <vilo/map.h>

#include "vilo/vikit/timer.h"
#include "vilo/vikit/ringbuffer.h"

namespace vilo {
class PerformanceMonitor;}

namespace vilo
{
class Point;
class Matcher;
class DepthFilter;

/// Base class for various VO pipelines. Manages the map and the state machine.
class FrameHandlerBase : boost::noncopyable
{
public:
  enum Stage {
    STAGE_PAUSED,
    STAGE_FIRST_FRAME,
    STAGE_SECOND_FRAME,
    STAGE_DEFAULT_FRAME,
    STAGE_RELOCALIZING
  };
  enum TrackingQuality {
    TRACKING_INSUFFICIENT,
    TRACKING_BAD,
    TRACKING_GOOD
  };
  enum UpdateResult {
    RESULT_NO_KEYFRAME,
    RESULT_IS_KEYFRAME,
    RESULT_FAILURE
  };

  FrameHandlerBase();

  virtual ~FrameHandlerBase();

  /// Get the current map.
  Map* map() const { return map_; }

  /// Will reset the map as soon as the current frame is finished processing.
  void reset() { set_reset_ = true; }

  /// Start processing.
  void start() { set_start_ = true; }

  /// Get the current stage of the algorithm.
  Stage stage() const { return stage_; }

  /// Get tracking quality.
  TrackingQuality trackingQuality() const { return tracking_quality_; }

  /// Get the processing time of the previous iteration.
  double lastProcessingTime() const { return timer_.getTime(); }

  /// Get the number of feature observations of the last frame.
  size_t lastNumObservations() const { return num_obs_last_; }

public:
  Stage stage_;                 //!< Current stage of the algorithm.
  bool set_reset_;              //!< Flag that the user can set. Will reset the system before the next iteration.
  bool set_start_;              //!< Flag the user can set to start the system when the next image is received.
  vilo::Timer timer_;            //!< Stopwatch to measure time to process frame.
  vilo::RingBuffer<double> acc_frame_timings_;    //!< Total processing time of the last 10 frames, used to give some user feedback on the performance.
  vilo::RingBuffer<size_t> acc_num_obs_;          //!< Number of observed features of the last 10 frames, used to give some user feedback on the tracking performance.
  size_t num_obs_last_;                         //!< Number of observations in the previous frame.
  TrackingQuality tracking_quality_;            //!< An estimate of the tracking quality based on the number of tracked features.
  size_t regular_counter_;
  int n_acc_relocalization_; 
  Map* map_;                     //!< Map of keyframes created by the slam system.

  /// Before a frame is processed, this function is called.
  bool startFrameProcessingCommon(const double timestamp);

  /// When a frame is finished processing, this function is called.
  int finishFrameProcessingCommon(
      const size_t update_id,
      const UpdateResult dropout,
      const size_t num_observations);

  /// Reset the map and frame handler to start from scratch.
  void resetCommon();

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll() { resetCommon(); }

  /// Set the tracking quality based on the number of tracked features.
  virtual void setTrackingQuality(const size_t num_observations);

};

} // namespace nslam

#endif // VILO_FRAME_HANDLER_BASE_H_
