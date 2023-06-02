#include <set>
#include "vilo/map.h"
#include "vilo/point.h"
#include "vilo/frame.h"
#include "vilo/feature.h"
#include <boost/bind.hpp>
#include "vilo/lidar.h"
#include "vilo/vikit/so3_functions.h"

namespace vilo {

Map::Map() {}

Map::~Map()
{
  reset();
  VILO_INFO_STREAM("Map destructed");
}

void Map::scaleRotate(const Eigen::Matrix3d& Rgw, const double& scale, const Eigen::Vector3d& t)
{
    boost::unique_lock<boost::mutex> lock(frame_mutex_);
    bool vel_scale = true;
    SE3 T_y_w = SE3(Rgw, t);
    
    int i=0;
    auto iter = keyframes_.begin();
    while(iter != keyframes_.end())
    {
      FramePtr kf = *iter;
      SE3 Twc = kf->getFramePose().inverse();

      Eigen::Matrix3d rwc = Twc.rotation_matrix();
      Eigen::Vector3d twc = scale * Twc.translation();

      SE3 new_Twc = SE3(rwc, twc);      
      SE3 Tcy = (T_y_w * new_Twc).inverse();
      kf->setFramePose(Tcy);
      
      Eigen::Vector3d vel = kf->getVelocity();
      if(!vel_scale)
          kf->setVelocity(T_y_w * vel);
      else
          kf->setVelocity(T_y_w * vel * scale);
            
      for(auto& ft: kf->fts_)
      {
          if(ft->point == NULL) 
              continue;
          if(ft->point->is_imu_rectified)
              continue;

          double idist = ft->point->getPointIdist();
          idist /= scale;
          ft->point->setPointIdistAndPose(idist);

          ft->point->is_imu_rectified = true;                   
          ft->point->initNormal();     
      }

      ++iter;
    }

    iter = keyframes_.begin();
    while(iter != keyframes_.end())
    {
      FramePtr kf = *iter;
      for(auto& ft: kf->fts_)
      {
          if(ft->point == NULL) 
              continue;

          ft->point->is_imu_rectified = false;                   
      }

      ++iter;
    }  
}

void Map::scaleRotate(Sophus::SE3& Tbc,
                      Eigen::Matrix3d& Rcl_orient, 
                      Eigen::Matrix3d& Rlc_position, 
                      Eigen::Vector3d& tlc_position, double& scale)
{
    boost::unique_lock<boost::mutex> lock(frame_mutex_);
    bool vel_scale = true;
    Sophus::SE3 Tcb = Tbc.inverse();
    int i=0;
    auto iter = keyframes_.begin();
    while(iter != keyframes_.end())
    {
      FramePtr kf = *iter;
        
      Eigen::Matrix3d rwb_c = kf->getImuRotation();
      Eigen::Vector3d twb_c = kf->getImuPosition();
      Eigen::Matrix3d new_rwb_c = rwb_c * Rcl_orient;
      Eigen::Vector3d new_twb_c = scale * Rlc_position * twb_c + tlc_position;
      Sophus::SE3 new_Twb_c = SE3(new_rwb_c, new_twb_c); 
      Sophus::SE3 new_Tcw = (new_Twb_c * Tbc).inverse();
      kf->setFramePose(new_Tcw);
           
      Eigen::Vector3d vel = kf->getVelocity();
      if(!vel_scale)
          kf->setVelocity(Rlc_position * vel);
      else
          kf->setVelocity(scale * Rlc_position * vel);

      for(auto& ft: kf->fts_)
      {
          if(ft->point == NULL) 
              continue;
          if(ft->point->is_imu_rectified)
              continue;

          double idist = ft->point->getPointIdist();
          idist /= scale;
          ft->point->setPointIdistAndPose(idist);

          ft->point->is_imu_rectified = true;                   

      }
      ++iter;
    }

    iter = keyframes_.begin();
    while(iter != keyframes_.end())
    {
      FramePtr kf = *iter;
      for(auto& ft: kf->fts_)
      {
          if(ft->point == NULL) 
              continue;

          ft->point->initNormal(); 
          ft->point->is_imu_rectified = false;
      }
      
      ++iter;
    }
}

void Map::rotateLidar(const Eigen::Matrix3d& Rgw, const Eigen::Vector3d& t)
{
    boost::unique_lock<boost::mutex> lock(frame_mutex_);
    SE3 T_y_w = SE3(Rgw, t);

    int i=0;
    auto iter = lidars_list_.begin();

    while(iter != lidars_list_.end())
    {
      LidarPtr lidar = *iter;
      SE3 Twl = lidar->getLidarPose().inverse();
      SE3 Tly = (T_y_w * Twl).inverse();
      lidar->setLidarPose(Tly);
            
      Eigen::Vector3d vel = lidar->getVelocity();
      lidar->setVelocity(T_y_w * vel);
      ++iter;
    }
}

void Map::reset()
{  
  {
    boost::unique_lock<boost::mutex> lock(frame_mutex_);
    keyframes_.clear();
  }
  point_candidates_.reset();
  emptyTrash();

  lidars_list_.clear();
}

bool Map::safeDeleteFrame(FramePtr frame)
{
  bool found = false;
  {
    boost::unique_lock<boost::mutex> lock(frame_mutex_);  
    for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
    {
      if(*it == frame)
      {
        std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
          removePtFrameRef(it->get(), ftr);
        });
        keyframes_.erase(it);
        found = true;
        break;
      }
    }
  }

  point_candidates_.removeFrameCandidates(frame);

  if(found)
    return true;

  VILO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

bool Map::safeDeleteFrameID(int id)
{
  bool found = false;
  FramePtr delete_frame = NULL;
  
  {
    boost::unique_lock<boost::mutex> lock(frame_mutex_);
    
    for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
    {
      if((*it)->id_ == id)
      {
        delete_frame = *it;

        std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
          removePtFrameRef(it->get(), ftr);
        });
        keyframes_.erase(it);
        found = true;
        break;
      }
    }
  }

  if(delete_frame != NULL)
    point_candidates_.removeFrameCandidates(delete_frame);

  if(found)
    return true;

  VILO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

void Map::removePtFrameRef(Frame* frame, Feature* ftr)
{
    if(ftr->point == NULL)
      return;
    Point* pt = ftr->point;
    ftr->point = NULL;
    int n_obs = pt->getObsSize();
    if(n_obs <= 2)
    {
      safeDeletePoint(pt);
      return;
    }
    pt->deleteFrameRef(frame);  
    frame->removeKeyPoint(ftr); 
}

void Map::safeDeletePoint(Point* pt)
{
  {
      boost::unique_lock<boost::mutex> lock(pt->obs_mutex_);
      std::for_each(pt->obs_.begin(), pt->obs_.end(), [&](Feature* ftr)
      {
        ftr->point=NULL;
        ftr->frame->removeKeyPoint(ftr);
      });
      pt->obs_.clear();
  }
  
  deletePoint(pt);
}

void Map::safeDeleteTempPoint(pair<Point*, Feature*>& p)
{
    if(p.first->seedStates_ == -1)
    {
        if(p.first->isBad_)
            safeDeletePoint(p.first);
        else
        {
            assert(p.first->hostFeature_ == p.second);

            double idist = p.first->getPointIdist();
            p.first->setPointIdistAndPose(idist);

            if(p.first->obs_.size() == 1)
            {
                p.first->setPointState(2);
                p.first->n_failed_reproj_ = 0;
                p.first->n_succeeded_reproj_ = 0;

                point_candidates_.candidates_.push_back(MapPointCandidates::PointCandidate(p.first, p.first->obs_.front()));
            }
            else
            {
                p.first->setPointState(3);
                p.first->n_failed_reproj_ = 0;
                p.first->n_succeeded_reproj_ = 0;
                p.second->frame->addFeature(p.second);
            }
        }
    }
    else 
    {   
        assert(p.first->seedStates_ == 1 && 
               p.first->obs_.back()->point->id_ == p.second->point->id_);

        for(auto it = p.first->obs_.begin(); it != p.first->obs_.end(); ++it)
            if((*it)->point->id_ != p.second->point->id_) 
            {
                (*it)->point=NULL;
                (*it)->frame->removeKeyPoint(*it);
            }

        p.first->obs_.clear();
        deletePoint(p.first);
    }
}


void Map::deletePoint(Point* pt)
{
  boost::unique_lock<boost::mutex> lock(point_mutex_);
  pt->setPointState(0);
  trash_points_.push_back(pt);
}

void Map::addKeyframe(FramePtr new_keyframe)
{
  boost::unique_lock<boost::mutex> lock(frame_mutex_);
  keyframes_.push_back(new_keyframe);
}

void Map::getCloseKeyframes(
    const FramePtr& frame,
    std::list< std::pair<FramePtr,double> >& close_kfs)
{
    Sophus::SE3 Tfw_cur = frame->getFramePose();
    Eigen::Vector3d tfw_cur = Tfw_cur.translation();

    boost::unique_lock<boost::mutex> lock(frame_mutex_);
    for(auto kf : keyframes_)
    {   
        boost::unique_lock<boost::mutex> lock(frame->keypoints_mutex_);
        for(auto keypoint : kf->key_pts_)
        {   
            if(keypoint == nullptr) continue;
  
            assert(keypoint->point != NULL);
            Eigen::Vector3d p_w = keypoint->point->getPointPose();
            
            if(frame->isVisible(p_w))
            {
                Sophus::SE3 Tfw_k = kf->getFramePose();
                Eigen::Vector3d tfw_k = Tfw_k.translation();
                close_kfs.push_back( std::make_pair( kf, (tfw_cur - tfw_k).norm() ) );
                break; 
        }
    }
}

FramePtr Map::getClosestKeyframe(const FramePtr& frame) 
{
  list< pair<FramePtr,double> > close_kfs;
  getCloseKeyframes(frame, close_kfs);
  if(close_kfs.empty())
  {
    return nullptr;
  }

  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  if(close_kfs.front().first != frame)
    return close_kfs.front().first;
  close_kfs.pop_front();
  return close_kfs.front().first;
}

FramePtr Map::getFurthestKeyframe(const Vector3d& pos) 
{
  boost::unique_lock<boost::mutex> lock(frame_mutex_);
  FramePtr furthest_kf;
  double maxdist = 0.0;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    double dist = ((*it)->pos()-pos).norm();
    if(dist > maxdist) {
      maxdist = dist;
      furthest_kf = *it;
    }
  }
  return furthest_kf;
}

bool Map::getKeyframeById(const int id, FramePtr& frame) 
{
    boost::unique_lock<boost::mutex> lock(frame_mutex_);
    bool found = false;
    for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
        if((*it)->id_ == id) 
        {
            found = true;
            frame = *it;
            break;
        }
    return found;
}

void Map::emptyTrash()
{
  {
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& pt){
      
      pt=NULL;
    });
    trash_points_.clear();
  }
  
  point_candidates_.emptyTrash();
}

bool Map::getMapChangedFlag()
{
    boost::unique_lock<boost::mutex> lock(flag_mutex_);
    return is_map_changed_;
}

void Map::setMapChangedFlag(bool flag)
{
    boost::unique_lock<boost::mutex> lock(flag_mutex_);
    is_map_changed_ = flag;
}


void Map::addNewLidar(LidarPtr lidar)
{
  lidars_list_.push_back(lidar);
  while(lidars_list_.size()>=100)
  {
    lidars_list_.pop_front();
  }
      
}

MapPointCandidates::MapPointCandidates()
{}

MapPointCandidates::~MapPointCandidates()
{
  reset();
}

void MapPointCandidates::newCandidatePoint(Point* point, double depth_sigma2)
{
  point->setPointState(2);
  boost::unique_lock<boost::mutex> lock(mut_);

  candidates_.push_back(PointCandidate(point, point->obs_.front()));
}

void MapPointCandidates::addPauseSeedPoint(Point* point)
{
  assert(point->getPointState() == Point::TYPE_TEMPORARY);
  boost::unique_lock<boost::mutex> lock(mut_);

  assert(point->hostFeature_ == point->obs_.front());

  temporaryPoints_.push_back(make_pair(point, point->obs_.front()));
}

void MapPointCandidates::addCandidatePointToFrame(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    PointCandidateList::iterator it=candidates_.begin();
    while(it != candidates_.end())
    {
        if(it->first->obs_.front()->frame == frame.get())
        {
            assert(it->first->obs_.size() == 2);
            it->first->setPointState(3);
            it->first->n_failed_reproj_ = 0;

            it->second->frame->addFeature(it->second);


            it = candidates_.erase(it);
        }
        else
            ++it;
    }
}

bool MapPointCandidates::deleteCandidatePoint(Point* point)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    for(auto it=candidates_.begin(), ite=candidates_.end(); it!=ite; ++it)
    {
        if(it->first == point)
        {
            deleteCandidate(*it);
            candidates_.erase(it);
            return true;
        }
    }
    return false;
}

void MapPointCandidates::changeCandidatePosition(Frame* frame)
{
    Sophus::SE3 Twf = frame->getFramePose().inverse();
    boost::unique_lock<boost::mutex> lock(mut_);
    
    for(PointCandidateList::iterator it = candidates_.begin(); it != candidates_.end(); ++it)
    {
        Point* point = it->first;
        Feature* ft = it->second;
        int pt_type = point->getPointState();
        assert(point != NULL && 
               pt_type == Point::TYPE_CANDIDATE &&
               point->obs_.size() == 1 &&
               point->vPoint_ == NULL);

        if(ft->frame->id_ == frame->id_)
        {
            double idist = point->getPointIdist();
            point->setPointIdistAndPose(idist);
        }
            
    }
}

void MapPointCandidates::removeFrameCandidates(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  auto it=candidates_.begin();
  while(it!=candidates_.end())
  {
    if(it->second->frame == frame.get())
    {
      deleteCandidate(*it);
      it = candidates_.erase(it);
    }
    else
      ++it;
  }
}

void MapPointCandidates::reset()
{
  boost::unique_lock<boost::mutex> lock(mut_);
  std::for_each(candidates_.begin(), candidates_.end(), [&](PointCandidate& c){
    delete c.first;
    delete c.second;
  });
  candidates_.clear();
}

void MapPointCandidates::deleteCandidate(PointCandidate& c)
{
  boost::unique_lock<boost::mutex> lock(candi_mutex_);
  delete c.second; c.second=NULL;
  c.first->setPointState(0);
  trash_points_.push_back(c.first);
}

void MapPointCandidates::emptyTrash()
{
  boost::unique_lock<boost::mutex> lock(candi_mutex_);
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& p)
  {
    p=NULL;
  });
  trash_points_.clear();
}

namespace map_debug {

void mapValidation(Map* map, int id)
{
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    frameValidation(it->get(), id);
}

void frameValidation(Frame* frame, int id)
{
  for(auto it = frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point==NULL)
      continue;

    if((*it)->point->getPointState() == Point::TYPE_DELETED)
      printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);

    if(!(*it)->point->findFrameRef(frame))
      printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);

    pointValidation((*it)->point, id);
  }
  for(auto it=frame->key_pts_.begin(); it!=frame->key_pts_.end(); ++it)
    if(*it != NULL)
      if((*it)->point == NULL)
        printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

void pointValidation(Point* point, int id)
{
  for(auto it=point->obs_.begin(); it!=point->obs_.end(); ++it)
  {
    bool found=false;
    for(auto it_ftr=(*it)->frame->fts_.begin(); it_ftr!=(*it)->frame->fts_.end(); ++it_ftr)
     if((*it_ftr)->point == point) {
       found=true; break;
     }
    if(!found)
      printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
  }
}

void mapStatistics(Map* map)
{
  size_t n_pt_obs(0);
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    n_pt_obs += (*it)->nObs();
  printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs/map->size());

  size_t n_frame_obs(0);
  size_t n_pts(0);
  std::set<Point*> points;
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
  {
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
      if(points.insert((*ftr)->point).second) {
        ++n_pts;
        n_frame_obs += (*ftr)->point->nRefs();
      }
    }
  }
  printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs/n_pts);
}

} // namespace map_debug

} // namespace vilo
