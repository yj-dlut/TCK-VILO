#include <stdlib.h>
#include <Eigen/StdVector>
#include <boost/bind.hpp>
#include <fstream>
#include <vilo/frame_handler_base.h>
#include <vilo/config.h>
#include <vilo/feature.h>
#include <vilo/matcher.h>
#include <vilo/map.h>
#include <vilo/point.h>

namespace vilo
{
#ifdef VILO_TRACE
vilo::PerformanceMonitor* g_permon = NULL;
#endif

FrameHandlerBase::FrameHandlerBase() :
                                      stage_(STAGE_PAUSED),
                                      set_reset_(false),
                                      set_start_(false),
                                      acc_frame_timings_(10),
                                      acc_num_obs_(10),
                                      num_obs_last_(0),
                                      tracking_quality_(TRACKING_INSUFFICIENT),
                                      regular_counter_(0),
                                      n_acc_relocalization_(0)
{
#ifdef VILO_TRACE
  g_permon = new vilo::PerformanceMonitor();
  g_permon->addTimer("pyramid_creation");
  g_permon->addTimer("sparse_img_align");
  g_permon->addTimer("reproject");
  g_permon->addTimer("reproject_kfs");
  g_permon->addTimer("reproject_candidates");
  g_permon->addTimer("feature_align");
  g_permon->addTimer("pose_optimizer");
  g_permon->addTimer("point_optimizer");
  g_permon->addTimer("local_ba");
  g_permon->addTimer("tot_time");
  g_permon->addLog("timestamp");
  g_permon->addLog("img_align_n_tracked");
  g_permon->addLog("repr_n_mps");
  g_permon->addLog("repr_n_new_references");
  g_permon->addLog("sfba_thresh");
  g_permon->addLog("sfba_error_init");
  g_permon->addLog("sfba_error_final");
  g_permon->addLog("sfba_n_edges_final");
  g_permon->addLog("loba_n_erredges_init");
  g_permon->addLog("loba_n_erredges_fin");
  g_permon->addLog("loba_err_init");
  g_permon->addLog("loba_err_fin");
  g_permon->addLog("n_candidates");
  g_permon->addLog("dropout");
  g_permon->init(Config::traceName(), Config::traceDir());
#endif

  VILO_INFO_STREAM("VILO initialized");
}

FrameHandlerBase::~FrameHandlerBase()
{
  VILO_INFO_STREAM("VILO destructor invoked");
#ifdef VILO_TRACE
  delete g_permon;
#endif
}

bool FrameHandlerBase::startFrameProcessingCommon(const double timestamp)
{
  if(set_start_)
  {
    resetAll();
    stage_ = STAGE_FIRST_FRAME;
  }

  if(stage_ == STAGE_PAUSED)
    return false;

  VILO_LOG(timestamp);
  VILO_START_TIMER("tot_time");
  timer_.start();

  map_->emptyTrash();
  return true;
}

int FrameHandlerBase::finishFrameProcessingCommon(
    const size_t update_id,
    const UpdateResult dropout,
    const size_t num_observations)
{
  VILO_DEBUG_STREAM("Frame: "<<update_id<<"\t fps-avg = "<< 1.0/acc_frame_timings_.getMean()<<"\t nObs = "<<acc_num_obs_.getMean());
  VILO_LOG(dropout);

  acc_frame_timings_.push_back(timer_.stop());
  if(stage_ == STAGE_DEFAULT_FRAME)
    acc_num_obs_.push_back(num_observations);
  num_obs_last_ = num_observations;
  VILO_STOP_TIMER("tot_time");

#ifdef VILO_TRACE
  g_permon->writeToFile();
  {
    boost::unique_lock<boost::mutex> lock(map_->point_candidates_.mut_);
    size_t n_candidates = map_->point_candidates_.candidates_.size();
    VILO_LOG(n_candidates);
  }
#endif

  if(dropout == RESULT_FAILURE &&
      (stage_ == STAGE_DEFAULT_FRAME || stage_ == STAGE_RELOCALIZING ))
  {
    stage_ = STAGE_RELOCALIZING;
    tracking_quality_ = TRACKING_INSUFFICIENT;
    
    n_acc_relocalization_++;
    if(n_acc_relocalization_ >= 20)
    {
      set_start_ = true;
      n_acc_relocalization_ = 0;
    }
  }
  else if (dropout == RESULT_FAILURE)
  {
    resetAll();
  }
    
  if(set_reset_)
  {
    resetAll();
  }
    

  return 0;
}

void FrameHandlerBase::resetCommon()
{
  map_->reset();
  stage_ = STAGE_PAUSED;
  set_reset_ = false;
  set_start_ = false;
  tracking_quality_ = TRACKING_INSUFFICIENT;
  num_obs_last_ = 0;
  VILO_INFO_STREAM("RESET");
}

void FrameHandlerBase::setTrackingQuality(const size_t num_observations)
{
  tracking_quality_ = TRACKING_GOOD;
  if(num_observations < Config::qualityMinFts())
  {
    VILO_WARN_STREAM_THROTTLE(0.5, "Tracking less than "<< Config::qualityMinFts() <<" features!");
    tracking_quality_ = TRACKING_INSUFFICIENT;
  }
  const int feature_drop = static_cast<int>(std::min(num_obs_last_, Config::maxFts())) - num_observations;
  if(feature_drop > Config::qualityMaxFtsDrop())
  {
    VILO_WARN_STREAM("Lost "<< feature_drop <<" features!");
    tracking_quality_ = TRACKING_BAD;
  }
}

bool ptLastOptimComparator(Point* lhs, Point* rhs)
{
  return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

} // namespace vilo
