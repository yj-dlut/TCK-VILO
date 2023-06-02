#ifndef VILO_GLOBAL_H_
#define VILO_GLOBAL_H_

#include <list>
#include <vector>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <boost/shared_ptr.hpp>
#include <Eigen/StdVector>

#include "vilo/vikit/performance_monitor.h"

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

#ifndef RPG_VILO_VIKIT_IS_VECTOR_SPECIALIZED //Guard for rpg_vikit
#define RPG_VILO_VIKIT_IS_VECTOR_SPECIALIZED
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2d)
#endif

// #ifdef VILO_USE_ROS
//   #include <ros/console.h>
//   #define VILO_DEBUG_STREAM(x) ROS_DEBUG_STREAM(x)
//   #define VILO_INFO_STREAM(x) ROS_INFO_STREAM(x)
//   #define VILO_WARN_STREAM(x) ROS_WARN_STREAM(x)
//   #define VILO_WARN_STREAM_THROTTLE(rate, x) ROS_WARN_STREAM_THROTTLE(rate, x)
//   #define VILO_ERROR_STREAM(x) ROS_ERROR_STREAM(x)
// #else
  #define VILO_INFO_STREAM(x) std::cerr<<"\033[0;0m[INFO] "<<x<<"\033[0;0m"<<std::endl;
  #ifdef VILO_DEBUG_OUTPUT
    #define VILO_DEBUG_STREAM(x) VILO_INFO_STREAM(x)
  #else
    #define VILO_DEBUG_STREAM(x)
  #endif
  #define VILO_WARN_STREAM(x) std::cerr<<"\033[0;33m[WARN] "<<x<<"\033[0;0m"<<std::endl;
  #define VILO_ERROR_STREAM(x) std::cerr<<"\033[1;31m[ERROR] "<<x<<"\033[0;0m"<<std::endl;
  #include <chrono> // Adapted from rosconsole. Copyright (c) 2008, Willow Garage, Inc.
  #define VILO_WARN_STREAM_THROTTLE(rate, x) \
    do { \
      static double __log_stream_throttle__last_hit__ = 0.0; \
      std::chrono::time_point<std::chrono::system_clock> __log_stream_throttle__now__ = \
      std::chrono::system_clock::now(); \
      if (__log_stream_throttle__last_hit__ + rate <= \
          std::chrono::duration_cast<std::chrono::seconds>( \
          __log_stream_throttle__now__.time_since_epoch()).count()) { \
        __log_stream_throttle__last_hit__ = \
        std::chrono::duration_cast<std::chrono::seconds>( \
        __log_stream_throttle__now__.time_since_epoch()).count(); \
        VILO_WARN_STREAM(x); \
      } \
    } while(0)
// #endif

namespace vilo
{
  using namespace Eigen;
  using namespace Sophus;

  const double EPS = 0.0000000001;
  const double PI = 3.14159265;

#ifdef VILO_TRACE
  extern vilo::PerformanceMonitor* g_permon;
  #define VILO_LOG(value) g_permon->log(std::string((#value)),(value))
  #define VILO_LOG2(value1, value2) VILO_LOG(value1); VILO_LOG(value2)
  #define VILO_LOG3(value1, value2, value3) VILO_LOG2(value1, value2); VILO_LOG(value3)
  #define VILO_LOG4(value1, value2, value3, value4) VILO_LOG2(value1, value2); VILO_LOG2(value3, value4)
  #define VILO_START_TIMER(name) g_permon->startTimer((name))
  #define VILO_STOP_TIMER(name) g_permon->stopTimer((name))
#else
  #define VILO_LOG(v)
  #define VILO_LOG2(v1, v2)
  #define VILO_LOG3(v1, v2, v3)
  #define VILO_LOG4(v1, v2, v3, v4)
  #define VILO_START_TIMER(name)
  #define VILO_STOP_TIMER(name)
#endif

  
  class Frame;
  typedef boost::shared_ptr<Frame> FramePtr;
  
  class Lidar;
  typedef boost::shared_ptr<Lidar> LidarPtr;

} // namespace vilo

#endif // VILO_GLOBAL_H_
