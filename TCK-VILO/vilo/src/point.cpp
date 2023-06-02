#include <stdexcept>
#include <vilo/point.h>
#include <vilo/frame.h>
#include <vilo/feature.h>
#include <vilo/config.h>

#include "vilo/vikit/math_utils.h"


namespace vilo {

int Point::point_counter_ = 0;

Point::Point(const Vector3d& pos) :
    id_(point_counter_++),
    pos_(pos),
    normal_set_(false),
    n_obs_(0),
    v_pt_(NULL),
    last_published_ts_(0),
    last_projected_kf_id_(-1),
    type_(TYPE_UNKNOWN),
    n_failed_reproj_(0),
    n_succeeded_reproj_(0),
    last_structure_optim_(0)
{

    is_imu_rectified = false;

    isBad_ = false;

    vPoint_ = NULL;
    nBA_ = 0;
}

Point::Point(const Vector3d& pos, Feature* ftr) :
    id_(point_counter_++),
    pos_(pos),
    normal_set_(false),
    n_obs_(1),
    v_pt_(NULL),
    last_published_ts_(0),
    last_projected_kf_id_(-1),
    type_(TYPE_UNKNOWN),
    n_failed_reproj_(0),
    n_succeeded_reproj_(0),
    last_structure_optim_(0)
{
    is_imu_rectified = false;
    
    obs_.push_front(ftr);
    isBad_ = false;

    vPoint_ = NULL;

    nBA_ = 0;
}

Point::~Point()
{}

void Point::addFrameRef(Feature* ftr)
{
    boost::unique_lock<boost::mutex> lock(obs_mutex_);
    obs_.push_front(ftr);
    ++n_obs_;
}

int Point::getObsSize()
{
    boost::unique_lock<boost::mutex> lock(obs_mutex_);
    return obs_.size();
}

std::list<Feature*> Point::getObs()
{
    boost::unique_lock<boost::mutex> lock(obs_mutex_);
    return obs_;
}

Feature* Point::findFrameRef(Frame* frame)
{
    boost::unique_lock<boost::mutex> lock(obs_mutex_);
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
        if((*it)->frame == frame)
            return *it;
    return NULL;    
}

bool Point::deleteFrameRef(Frame* frame)
{
    boost::unique_lock<boost::mutex> lock(obs_mutex_);
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
        if((*it)->frame == frame)
        {
            obs_.erase(it);
            return true;
        }

    return false;
}

void Point::initNormal()
{
    boost::unique_lock<boost::mutex> lock(obs_mutex_);
    assert(!obs_.empty());
    const Feature* ftr = obs_.back();
    assert(ftr->frame != NULL);
    normal_ = ftr->frame->getFramePose().rotation_matrix().transpose()*(-ftr->f);
    
    {
        boost::unique_lock<boost::mutex> lock(point_mutex_);
        normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos()).norm(),2), 1.0, 1.0);
    }
    normal_set_ = true;
}

bool Point::getCloseViewObs(const Vector3d& framepos, Feature*& ftr)
{
    Vector3d p_w;
    {
        boost::unique_lock<boost::mutex> lock(point_mutex_);
        p_w = pos_;
    }
    
    Vector3d obs_dir(framepos - p_w); 
    obs_dir.normalize();

    boost::unique_lock<boost::mutex> lock(obs_mutex_);
    auto min_it=obs_.begin();
    double min_cos_angle = 0;
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    {
        Vector3d dir((*it)->frame->pos() - p_w);
        dir.normalize();
        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
            min_cos_angle = cos_angle;
            min_it = it;
        }
    }
    ftr = *min_it;
    if(min_cos_angle < 0.5) return false; 

    return true;
}

double Point::getPointIdist()
{
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    return idist_;
}
    
Eigen::Vector3d Point::getPointPose()
{
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    return pos_;
}

void Point::setPointIdistAndPose(double idist) 
{
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    idist_ = idist;
    Eigen::Vector3d p_host = hostFeature_->f * (1.0/idist_);
    Sophus::SE3 Tfw_h = hostFeature_->frame->getFramePose();
    pos_ = Tfw_h.inverse() * p_host;
}

void Point::setPointIdist(double idist)
{
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    idist_ = idist;
}

void Point::setPointPose(Vector3d& pos) 
{
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    pos_ = pos;
}
void Point::setPointState(int state)
{
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    switch (state)
    {
        case 0 : 
        {
           type_ = TYPE_DELETED; 
           break;
        }
        case 1 : 
        {
            type_ = TYPE_TEMPORARY;  
            break;
        }
        case 2 : 
        {
            type_ = TYPE_CANDIDATE;
            break;
        }
        case 3 : 
        {
            type_ = TYPE_UNKNOWN;
            break;
        }
        case 4 : 
        {
           type_ = TYPE_GOOD; 
           break;
        }
        default :
        {
            break;
        }
    }
    
}

int Point::getPointState()
{
    boost::unique_lock<boost::mutex> lock(point_mutex_);
    return type_;
}

} // namespace vilo
