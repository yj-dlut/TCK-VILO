#ifndef VILO_FEATURE_H_
#define VILO_FEATURE_H_

#include <vilo/frame.h>

namespace vilo {

/// A salient image region that is tracked across frames.
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType {CORNER, EDGELET, GRADIENT};

  FeatureType type;     //!< Type can be corner or edgelet.
  Frame* frame;         //!< Pointer to frame in which the feature was detected.
  Vector2d px;          //!< Coordinates in pixels on pyramid level 0.
  Vector3d f;           //!< Unit-bearing vector of the feature.
  int level;            //!< Image pyramid level where feature was extracted.
  Point* point;         //!< Pointer to 3D point which corresponds to the feature.
  Vector2d grad;        //!< Dominant gradient direction for edglets, normalized.

  // used in photometric calibration thread
  vector<double> outputs;
  vector<double> radiances;
  vector<double> outputs_grad;
  vector<double> rad_mean;
  Feature* m_prev_feature = NULL;  
  Feature* m_next_feature = NULL;
  bool m_added = false;  // Flag, used in photomatric calibration
  // bool m_is_seed = false;
  bool m_non_point = false;
  Matrix2d rotate_plane;


  // corner
  Feature(Frame* _frame, const Vector2d& _px, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {
  }

  Feature(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {
  }

  Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(_point),
    grad(1.0,0.0)
  {
  }

  // edgelet
  Feature(Frame* _frame, const Vector2d& _px, const Vector2d& _grad, int _level) :
    type(EDGELET),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(NULL),
    grad(_grad)
  {
  }

  Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, const Vector2d& _grad, int _level) :
    type(EDGELET),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(_point),
    grad(_grad)
  {
  }

  // gradient
  Feature(Frame* _frame, const Vector2d& _px, int _level, FeatureType _type) :
    type(_type),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {
  }

  Feature(Frame* _frame, Point* _point, const Vector2d& _px, int _level, FeatureType _type) :
    type(_type),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(_point),
    grad(1.0,0.0)
  {
  }

};

} // namespace vilo

#endif // VILO_FEATURE_H_
