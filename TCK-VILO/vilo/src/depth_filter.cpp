#include <algorithm>

#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <vilo/global.h>
#include <vilo/depth_filter.h>
#include <vilo/frame.h>
#include <vilo/point.h>
#include <vilo/feature.h>
#include <vilo/matcher.h>
#include <vilo/config.h>
#include <vilo/feature_detection.h>
#include <vilo/IndexThreadReduce.h>
#include <vilo/matcher.h>
#include <vilo/feature_alignment.h>

#include "vilo/vikit/robust_cost.h"
#include "vilo/vikit/math_utils.h"

namespace vilo {

int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

Seed::Seed(Feature* ftr, float depth_mean, float depth_min, float converge_threshold) :
    batch_id(batch_counter),
    id(seed_counter++),
    ftr(ftr),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36),
    isValid(true),
    eplStart(Vector2i(0,0)),
    eplEnd(Vector2i(0,0)),
    haveReprojected(false),
    temp(NULL)
{
    vec_distance.push_back(depth_mean);
    vec_sigma2.push_back(sigma2);

    converge_thresh = converge_threshold;
}

DepthFilter::DepthFilter(
    feature_detection::FeatureExtractor* featureExtractor, callback_t seed_converged_cb) :
    featureExtractor_(featureExtractor),
    seed_converged_cb_(seed_converged_cb),
    seeds_updating_halt_(false),
    thread_(NULL),
    new_keyframe_set_(false),
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0),
    px_error_angle_(-1)
{
    ot_.open("/home/jun/Code/vilo_all/vilo/result/depthfilter.txt",ios::app);
    frame_prior_.resize(100000);

    threadReducer_ = new lsd_slam::IndexThreadReduce();

    runningStats_ = new RunningStats();
    n_update_last_ = 100;

    n_pre_update_ = 0;
    n_pre_try_ = 0;

    nPonits = 1;
    nSkipFrame = 0;

    nMeanConvergeFrame_ = 6;
    convergence_sigma2_thresh_ = 200;

    is_visual_opt_ = false;
    frameCount = 0;
    laserCloudCenWidth = 10;
    laserCloudCenHeight = 10;
    laserCloudCenDepth = 5;
    laserCloudWidth = 21;
    laserCloudHeight = 21;
    laserCloudDepth = 11;
    feature_array_num = 4851;
    for (int i = 0; i < feature_array_num; i++)
	{
		corner_features_array[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
		surf_features_array[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
	}
}

DepthFilter::~DepthFilter()
{
    stopThread();
    VILO_INFO_STREAM("DepthFilter destructed.");
}

void DepthFilter::startThread()
{
    thread_ = new boost::thread(&DepthFilter::run, this);
}

void DepthFilter::stopThread()
{
    VILO_INFO_STREAM("DepthFilter stop thread invoked.");
    if(thread_ != NULL)
    {
        VILO_INFO_STREAM("DepthFilter interrupt and join thread... ");
        setHaltFlag(true);
        thread_->interrupt();
        thread_->join();
        thread_ = NULL;
    }

    delete threadReducer_;
    delete runningStats_;
}

void DepthFilter::addFrame(FramePtr frame)
{
    if(thread_ != NULL)
    {
        {
            lock_t lock(frame_queue_mut_);
            if(frame_queue_.size() > 5)
                frame_queue_.pop();
            frame_queue_.push(frame);
        }
        setHaltFlag(false);
        frame_queue_cond_.notify_one();
    }
    else
        updateSeeds(frame);
}

void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min, float converge_thresh)
{
    new_keyframe_min_depth_ = depth_min;
    new_keyframe_mean_depth_ = depth_mean;
    convergence_sigma2_thresh_ = converge_thresh;

    if(thread_ != NULL)
    {
        new_keyframe_ = frame;
        setNewKeyframeFlag(true);
        setHaltFlag(true);
        frame_queue_cond_.notify_one();
    }
    else
        initializeSeeds(frame);
}

void DepthFilter::initializeSeeds(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock_c(detector_mut_);
    Features new_features;

    featureExtractor_->setExistingFeatures(frame->fts_);
    featureExtractor_->detect(frame.get(), 20, frame->gradMean_, new_features, frame->m_last_frame.get());

    lock_c.unlock();

    setHaltFlag(true);

    lock_t lock(seeds_mut_);
    ++Seed::batch_counter;

    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr)
    {
        Seed seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_, convergence_sigma2_thresh_);
       
        for(auto it = frame_prior_[Seed::batch_counter-1].begin(); it != frame_prior_[Seed::batch_counter-1].end(); ++it)
        {
            seed.pre_frames.push_back(*it);
        }

        seeds_.push_back(seed);
    });

    setHaltFlag(false);
    frame->finish();
}

void DepthFilter::removeKeyframe(FramePtr frame)
{
    setHaltFlag(true);
    lock_t lock(seeds_mut_);
    std::list<Seed>::iterator it=seeds_.begin();
    size_t n_removed = 0;
    while(it!=seeds_.end())
    {
        if(it->ftr->frame == frame.get())
        {
        it = seeds_.erase(it);
        ++n_removed;
        }
        else
        ++it;
    }

    setHaltFlag(false);
}

void DepthFilter::reset()
{
  setHaltFlag(true);
  {
    lock_t lock(seeds_mut_);
    seeds_.clear();
  }
  lock_t lock();
  while(!frame_queue_.empty())
    frame_queue_.pop();

  setHaltFlag(false);

  if(options_.verbose)
    VILO_INFO_STREAM("DepthFilter: RESET.");
}

void DepthFilter::setNewKeyframeFlag(bool flag)
{
    boost::unique_lock<boost::mutex> lock(new_keyframe_mut_);
    new_keyframe_set_ = flag;
}

bool DepthFilter::getNewKeyframeFlag()
{
    boost::unique_lock<boost::mutex> lock(new_keyframe_mut_);
    return new_keyframe_set_;
}

void DepthFilter::run()
{
    while(!boost::this_thread::interruption_requested())
    {
        FramePtr frame;
        {
            lock_t lock(frame_queue_mut_);
            if(seeds_.empty())
            {
                while(frame_queue_.empty() && getNewKeyframeFlag() == false)
                    frame_queue_cond_.wait(lock);
            }
            else
            {
                std::list<Seed>::iterator it=seeds_.begin();

                while(frame_queue_.empty() && getNewKeyframeFlag() == false && it != seeds_.end())
                {
                    observeDepthWithPreviousFrameOnce(it);
                    it++;
                }
                
            }

            if(!frame_queue_.empty() || getNewKeyframeFlag())
            {
                if(getNewKeyframeFlag())
                {
                    setNewKeyframeFlag(false);
                    setHaltFlag(false);
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }
            else
            {
                while(frame_queue_.empty() && getNewKeyframeFlag() == false)
                {
                    frame_queue_cond_.wait(lock);
                }
                    
                if(getNewKeyframeFlag())
                {
                    setNewKeyframeFlag(false);
                    setHaltFlag(false);
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }

            n_pre_update_ = 0;
            n_pre_try_ = 0;

        }

        LidarPtr lidar;
        {
            lock_t lock(lidar_queue_mut_);
            if(!lidar_queue_.empty())
            {
                lidar = lidar_queue_.front();
                lidar_queue_.pop();
                is_new_lidar_ = true;
                new_lidar_ = lidar;
                
                if(lidar->is_keylidar_)
                {
                    if(frame_handler_->is_lio_initialized_)
                    {
                        while(key_lidars_list_.size()>=8)
                            key_lidars_list_.pop_front();
                    }
                    key_lidars_list_.push_back(lidar);
                }
            }

        }

        active_frame_ = frame;
        sub_map_ = frame_handler_->sub_map_;

        if(!active_frame_->isKeyframe())
        {
            is_visual_opt_ = false;
        }
        else
        {
            if(!frame_handler_->is_vio_initialized_)
                is_visual_opt_ = true;
            else
            {               
                if(active_frame_->align_lid_ != -1)
                    is_visual_opt_ = false;
                else
                    is_visual_opt_ = true; 
            }

        }

        updateSeeds(frame);
        
        if(frame->isKeyframe())
        {
            initializeSeeds(frame);
        }

        localMapping(is_track_good_);
    }
}

void DepthFilter::updateSeeds(FramePtr frame)
{
    printf("updateSeeds 0 ");
    if(!frame->isKeyframe())
        frame_prior_[Seed::batch_counter].push_front(frame);
    else
        frame_prior_[Seed::batch_counter+1].push_front(frame);

    if(Seed::batch_counter > 30 && frame->isKeyframe() && frame->m_pc == NULL)
    {
        list<FramePtr>::iterator it = frame_prior_[Seed::batch_counter-30].begin();
        while(it != frame_prior_[Seed::batch_counter-30].end())
        {
            Frame* dframe = (*it).get();
            if(!dframe->isKeyframe())
            {
                delete dframe;
                dframe = NULL;
            }
            ++it;
        }
    }
    
    lock_t lock(seeds_mut_);

    if(this->px_error_angle_ == -1)
    {
        const double focal_length = frame->cam_->errorMultiplier2();
        double px_noise = 1.0;
        double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0;
        this->px_error_angle_ = px_error_angle;
    }

    std::list<Seed>::iterator it=seeds_.begin();
    int nn=0;
    while(it!=seeds_.end())
    {
        if(getHaltFlag())
            return;

        if((Seed::batch_counter - it->batch_id) > options_.max_n_kfs)
        {
            assert(it->ftr->point == NULL);

            if(it->temp != NULL && it->haveReprojected)
                it->temp->seedStates_ = -1;
            else
            {
                assert(it->ftr != NULL);
                nn++;

                delete it->ftr;
                it->ftr = NULL;
            }

            it->pre_frames.clear();
            it->optFrames_P.clear();
            it->optFrames_A.clear();

            it = seeds_.erase(it);
            continue;
        }

        it++;
    }
    
    observeDepth();
    
    if(frame->id_ >= 100)
        assert(seeds_.size()!=0);
    it = seeds_.begin();
    while(it!=seeds_.end())
    {
        if(getHaltFlag())
        {
            return;
        }

        if(sqrt(it->sigma2) < it->z_range/it->converge_thresh)
        {
            bool isValid = true;
            if(activatePoint(*it, isValid))
                it->mu = it->opt_id;

            Vector3d pHost = it->ftr->f * (1.0/it->mu);
            if(it->mu < 1e-10 || pHost[2] < 1e-10)  isValid = false;

            if(!isValid)
            {
                if(it->temp != NULL && it->haveReprojected)
                    it->temp->seedStates_ = -1;

                it = seeds_.erase(it);
                continue;
            }

            {
                if(m_v_n_converge.size() > 1000)
                    m_v_n_converge.erase(m_v_n_converge.begin());

                m_v_n_converge.push_back(it->vec_distance.size());
            }

            Sophus::SE3 Twf_h = it->ftr->frame->getFramePose().inverse();
            Vector3d xyz_world = Twf_h * pHost;
            Point* point = new Point(xyz_world, it->ftr);

            point->idist_ = it->mu;
            point->hostFeature_ = it->ftr;
            point->color_ = it->ftr->frame->img_pyr_[0].at<uchar>((int)it->ftr->px[1], (int)it->ftr->px[0]);

            if(it->ftr->type == Feature::EDGELET)
                point->ftr_type_ = Point::FEATURE_EDGELET;
            else if(it->ftr->type == Feature::CORNER)
                point->ftr_type_ = Point::FEATURE_CORNER;
            else
                point->ftr_type_ = Point::FEATURE_GRADIENT;

            it->ftr->point = point;

            if(it->temp != NULL && it->haveReprojected)
                it->temp->seedStates_ = 1;
            else
                assert(it->temp == NULL && !it->haveReprojected);

            it->pre_frames.clear();
            it->optFrames_P.clear();
            it->optFrames_A.clear();

            seed_converged_cb_(point, it->sigma2); 

            it = seeds_.erase(it);
        }
        else if(!it->isValid)
        {
            VILO_WARN_STREAM("z_min is NaN");
            it = seeds_.erase(it);
        }
        else
            ++it;
    }

    lock_t lock_converge(mean_mutex_);
    if(m_v_n_converge.size() > 500)
        nMeanConvergeFrame_ = std::accumulate(m_v_n_converge.begin(), m_v_n_converge.end(), 0) / m_v_n_converge.size();
    else
        nMeanConvergeFrame_ = 6;
}

void DepthFilter::clearFrameQueue()
{
  while(!frame_queue_.empty())
    frame_queue_.pop();
}

void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds)
{
  lock_t lock(seeds_mut_);
  for(std::list<Seed>::iterator it=seeds_.begin(); it!=seeds_.end(); ++it)
  {
    if (it->ftr->frame == frame.get())
      seeds.push_back(*it);
  }
}

#define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))
void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
    float id_var = seed->sigma2*1.01f;
    float w = tau2 / (tau2 + id_var);
    float new_idepth = (1-w)*x + w*seed->mu;
    seed->mu = UNZERO(new_idepth);
    id_var *= w;

    if(id_var < seed->sigma2) seed->sigma2 = id_var;
}

double DepthFilter::computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm);
    double beta = acos(a.dot(-t)/(t_norm*a_norm));
    double beta_plus = beta + px_error_angle;
    double gamma_plus = PI-alpha-beta_plus; 
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus);
    return (z_plus - z);
}

void DepthFilter::observeDepth()
{
    size_t n_seeds = seeds_.size();
    int n_step = n_seeds / 10;
    threadReducer_->reduce(boost::bind(&DepthFilter::observeDepthRow, this, _1, _2, _3), 0, (int)seeds_.size(), runningStats_, n_step);

    runningStats_->n_seeds = seeds_.size();
    n_update_last_ = runningStats_->n_updates;
    runningStats_->setZero();
}

void DepthFilter::observeDepthRow(int yMin, int yMax, RunningStats* stats)
{
    if(getHaltFlag())
        return;

    std::list<Seed>::iterator it=seeds_.begin();
    for(int i = 0; i < yMin; ++i) it++;

    Sophus::SE3 Twf_cur = active_frame_->getFramePose().inverse();
    for(int idx = yMin; idx < yMax; ++idx, ++it)
    {
        if(getHaltFlag())
            return;
        
        Sophus::SE3 Tfw_ref = it->ftr->frame->getFramePose();
        SE3 T_ref_cur = Tfw_ref * Twf_cur;

        Vector3d xyz_f = T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f);
        if(xyz_f.z() < 0.0)
        {
            stats->n_out_views++;
            it->is_update = false;
            continue;
        }

        if(!active_frame_->cam_->isInFrame(active_frame_->f2c(xyz_f).cast<int>())) 
        {
            stats->n_out_views++;
            it->is_update = false;
            continue;
        }
        
        it->is_update = true;

        if(it->optFrames_A.size() < 15)
            it->optFrames_A.push_back(active_frame_);

        float z_inv_min = it->mu + 2*sqrt(it->sigma2);
        float z_inv_max = max(it->mu - 2*sqrt(it->sigma2), 0.00000001f);
        if(isnan(z_inv_min)) it->isValid = false;


        Matcher matcher;
        double z;
        int res = matcher.doLineStereo(*it->ftr->frame,
                                       *active_frame_,
                                       *it->ftr,
                                        1.0/z_inv_min,
                                        1.0/it->mu,
                                        1.0/z_inv_max,
                                        z,
                                        it->eplStart,
                                        it->eplEnd);
    
        if(res != 1)
        {
            it->b++;
            it->eplStart = Vector2i(0,0);
            it->eplEnd   = Vector2i(0,0);

            stats->n_failed_matches++;

            if(res == -1)
                stats->n_fail_lsd++;
            else if(res == -2)
                stats->n_fail_triangulation++;
            else if(res == -3)
                stats->n_fail_alignment++;
            else if(res == -4)
                stats->n_fail_score++;

            continue;
        }
        
        double tau = computeTau(T_ref_cur, it->ftr->f, z, this->px_error_angle_);
        double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));
        
        updateSeed(1./z, tau_inverse*tau_inverse, &*it);

        it->vec_distance.push_back(1.0/it->mu);
        it->vec_sigma2.push_back(it->sigma2);

        it->last_update_frame = active_frame_;
        it->last_matched_px = matcher.px_cur_;
        it->last_matched_level = matcher.search_level_;

        stats->n_updates++;

        if(active_frame_->isKeyframe())
        {
            boost::unique_lock<boost::mutex> lock(detector_mut_);
            featureExtractor_->setGridOccpuancy(matcher.px_cur_, it->ftr);
        }
    }
}

void DepthFilter::observeDepthWithPreviousFrameOnce(std::list<Seed>::iterator& ite)
{
    if(ite->pre_frames.empty() || this->px_error_angle_ == -1)
        return;
    
    FramePtr preFrame = *(ite->pre_frames.begin());
    assert(preFrame->id_ < ite->ftr->frame->id_);

    n_pre_try_++;
    Sophus::SE3 Tfw_ref = ite->ftr->frame->getFramePose();
    Sophus::SE3 Twf_cur = preFrame->getFramePose().inverse();
    SE3 T_ref_cur = Tfw_ref * Twf_cur;
    Vector3d xyz_f = T_ref_cur.inverse()*(1.0/ite->mu * ite->ftr->f);

    if(xyz_f.z() < 0.0)
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return;
    }
    if(!preFrame->cam_->isInFrame(preFrame->f2c(xyz_f).cast<int>()))
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return;
    }

    if(ite->optFrames_P.size() < 15)
        ite->optFrames_P.push_back(preFrame);


    float z_inv_min = ite->mu + 2*sqrt(ite->sigma2);
    float z_inv_max = max(ite->mu - 2*sqrt(ite->sigma2), 0.00000001f);
    double z;

    if(!matcher_.findEpipolarMatchPrevious(*ite->ftr->frame, *preFrame, *ite->ftr, 1.0/ite->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return;
    }

    n_pre_update_++;

    double tau = computeTau(T_ref_cur, ite->ftr->f, z, this->px_error_angle_);
    double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

    updateSeed(1./z, tau_inverse*tau_inverse, &*ite);
    ite->pre_frames.erase(ite->pre_frames.begin());

}

bool DepthFilter::activatePoint(Seed& seed, bool& isValid)
{
    seed.opt_id = seed.mu;

    const int halfPatchSize = 4;
    const int patchSize = halfPatchSize*2;
    const int patchArea = patchSize*patchSize;

    Frame* host = seed.ftr->frame;
    Vector3d pHost = seed.ftr->f*(1.0/seed.mu);
    Sophus::SE3 Twf_h = host->getFramePose().inverse();

    vector< pair<FramePtr, Vector2d> > targets;
    targets.reserve(seed.optFrames_P.size()+seed.optFrames_A.size());
    for(size_t i = 0; i < seed.optFrames_P.size(); ++i)
    {
        FramePtr target = seed.optFrames_P[i];
        Sophus::SE3 Tfw_t = target->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Twf_h;
        Vector3d pTarget = Tth*pHost;
        if(pTarget[2] < 0.0001) continue;

        Vector2d px(target->cam_->world2cam(pTarget));
        if(!target->cam_->isInFrame(px.cast<int>(), 8))
            continue;

        targets.push_back(make_pair(target, px));
    }

    for(size_t i = 0; i < seed.optFrames_A.size(); ++i)
    {
        FramePtr target = seed.optFrames_A[i];
        Sophus::SE3 Tfw_t = target->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Twf_h;
        Vector3d pTarget = Tth*pHost;
        if(pTarget[2] < 0.0001) continue;

        Vector2d px(target->cam_->world2cam(pTarget));
        if(!target->cam_->isInFrame(px.cast<int>(), 8))
            continue;

        targets.push_back(make_pair(target, px));
    }

    float n_frame_thresh = nMeanConvergeFrame_*0.7;
    if(n_frame_thresh > 8)  n_frame_thresh = 8;
    if(n_frame_thresh < 3)  n_frame_thresh = 3;

    if(targets.size() < n_frame_thresh) return false;

    double distMean = 0;
    vector< pair<FramePtr, Vector2d> > targetResult;
    targetResult.reserve(targets.size());
    vector<Vector2d> targetNormal; targetNormal.reserve(targets.size());
    for(size_t i = 0; i < targets.size(); ++i)
    {
        Vector2d beforePx(targets[i].second);
        Matcher matcher;
        if(matcher.findMatchSeed(seed, *(targets[i].first.get()), targets[i].second, 0.65))
        {
            Vector2d afterPx(targets[i].second);

            if(seed.ftr->type != Feature::EDGELET)
            {
                double err = (beforePx-afterPx).norm();
                err /= (1<<matcher.search_level_);
                distMean += err;
            }
            else
            {
                Vector2d normal(matcher.A_cur_ref_*seed.ftr->grad);
                normal.normalize();
                targetNormal.push_back(normal);
                double err = fabs(normal.transpose()*(beforePx-afterPx));
                err /= (1<<matcher.search_level_);
                distMean += err;
            }

            Vector3d f(targets[i].first->cam_->cam2world(targets[i].second));
            Vector2d obs(vilo::project2d(f));
            targetResult.push_back(make_pair(targets[i].first, obs));
        }
    }

    if(targetResult.size() < n_frame_thresh)
        return false;

    distMean /= targetResult.size();
    if(seed.ftr->type != Feature::EDGELET && distMean > 3.2)
    {
        isValid = false;
        return false;
    }
    if(seed.ftr->type == Feature::EDGELET && distMean > 2.5)
    {
        isValid = false;
        return false;
    }
    isValid = true;

    if(seed.ftr->type != Feature::EDGELET && distMean > 2.5)
        return false;
    if(seed.ftr->type == Feature::EDGELET && distMean > 2.0)
        return false;

    #ifdef ACTIVATE_DBUG
        cout << "======================" << endl;
    #endif

    seedOptimizer(seed, targetResult, targetNormal);

    return true;
}

void DepthFilter::seedOptimizer(
    Seed& seed, const vector<pair<FramePtr, Vector2d> >& targets, const vector<Vector2d>& normals)
{
    if(seed.ftr->type == Feature::EDGELET)
        assert(targets.size() == normals.size());


    double oldEnergy = 0.0, rho = 0, mu = 0.1, nu = 2.0;
    bool stop = false;
    int n_trials = 0;

    const int n_trials_max = 5;


    double old_id = seed.mu;
    vector<SE3> Tths; Tths.resize(targets.size());
    Vector3d pHost(seed.ftr->f * (1.0/old_id));

    vector<float> errors; errors.reserve(targets.size());
    Sophus::SE3 Twf_h = seed.ftr->frame->getFramePose().inverse();
    for(size_t i = 0; i < targets.size(); ++i)
    {
        FramePtr target = targets[i].first;
        Sophus::SE3 Tfw_t = target->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Twf_h;
        Vector2d residual = targets[i].second-vilo::project2d(Tth*pHost);

        if(seed.ftr->type == Feature::EDGELET)
            errors.push_back(fabs(normals[i].transpose()*residual));
        else
            errors.push_back(residual.norm());
    }

    vilo::robust_cost::MADScaleEstimator mad_estimator;
    const double huberTH = mad_estimator.compute(errors);

    for(size_t i = 0; i < targets.size(); ++i)
    {
        FramePtr target = targets[i].first;
        Sophus::SE3 Tfw_t = target->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Twf_h;
        Vector2d residual = targets[i].second-vilo::project2d(Tth*pHost);

        if(seed.ftr->type == Feature::EDGELET)
        {
            double resEdgelet = normals[i].transpose()*residual;
            double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
            oldEnergy += resEdgelet*resEdgelet * hw;
        }
        else
        {
            double res_dist = residual.norm();
            double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
            oldEnergy += res_dist*res_dist * hw;
        }

        Tths[i] = Tth;
    }

    double H = 0, b = 0;
    for(int iter = 0; iter < 5; ++iter)
    {
        n_trials = 0;
        do
        {
            double new_id = old_id;
            double newEnergy = 0;
            H = b = 0;

            pHost = seed.ftr->f * (1.0/old_id);
            for(size_t i = 0; i < targets.size(); ++i)
            {
                FramePtr target = targets[i].first;
                SE3 Tth = Tths[i];

                Vector3d pTarget = Tth*pHost;
                Vector2d residual = targets[i].second-vilo::project2d(pTarget);

                if(seed.ftr->type == Feature::EDGELET)
                {
                    double resEdgelet = normals[i].transpose()*residual;
                    double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
                    
                    Vector2d Jxidd;
                    Point::jacobian_id2uv(pTarget, Tth, old_id, seed.ftr->f, Jxidd);
                    double JEdge = normals[i].transpose()*Jxidd;
                    H += JEdge*JEdge*hw;
                    b -= JEdge*resEdgelet*hw;
                }
                else
                {
                    double res_dist = residual.norm();
                    double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
                    
                    Vector2d Jxidd;
                    Point::jacobian_id2uv(pTarget, Tth, old_id, seed.ftr->f, Jxidd);

                    H += (Jxidd[0]*Jxidd[0] + Jxidd[1]*Jxidd[1])*hw;
                    b -= (Jxidd[0]*residual[0] + Jxidd[1]*residual[1])*hw;
                }
            }

            H *= 1.0+mu;
            double step = b/H;

            if(!(bool)std::isnan(step))
            {
                new_id = old_id+step;

                pHost = seed.ftr->f * (1.0/new_id);
                for(size_t i = 0; i < targets.size(); ++i)
                {
                    FramePtr target = targets[i].first;
                    SE3 Tth = Tths[i];
                    Vector2d residual = targets[i].second-vilo::project2d(Tth*pHost);

                    if(seed.ftr->type == Feature::EDGELET)
                    {
                        double resEdgelet = normals[i].transpose()*residual;
                        double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
                        newEnergy += resEdgelet*resEdgelet * hw;
                    }
                    else
                    {
                        double res_dist = residual.norm();
                        double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
                        newEnergy += res_dist*res_dist * hw;
                    }
                }

                rho = oldEnergy - newEnergy;
            }
            else
            {
                #ifdef ACTIVATE_DBUG
                    cout << "Matrix is close to singular!" << endl;
                    cout << "H = " << H << endl;
                    cout << "b = " << b << endl;
                #endif

                rho = -1;
            }

            if(rho > 0)
            {
                #ifdef ACTIVATE_DBUG
                    if(seed.ftr->type == Feature::EDGELET)
                        cout<< "EDGELET:  ";
                    else
                        cout<< "CORNER:  ";
                    cout<< "It. " << iter
                        << "\t Trial " << n_trials
                        << "\t Succ"
                        << "\t old Energy = " << oldEnergy
                        << "\t new Energy = " << newEnergy
                        << "\t lambda = " << mu
                        << endl;
                #endif

                oldEnergy = newEnergy;
                old_id = new_id;
                seed.opt_id = new_id;
                stop = fabsf(step) < 0.00001*new_id;

                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                ++n_trials;
                if (n_trials >= n_trials_max) stop = true;

                #ifdef ACTIVATE_DBUG
                    if(seed.ftr->type == Feature::EDGELET)
                        cout<< "EDGELET:  ";
                    else
                        cout<< "CORNER:  ";
                    cout<< "It. " << iter
                        << "\t Trial " << n_trials
                        << "\t Fail"
                        << "\t old Energy = " << oldEnergy
                        << "\t new Energy = " << newEnergy
                        << "\t lambda = " << mu
                        << endl;
                #endif
            }
        }while(!(rho>0 || stop));

        if(stop) break;
    }
}

void DepthFilter::directPromotionFeature()
{
    lock_t lock(m_converge_seed_mut);

    list<Seed>::iterator seed = m_converge_seed.begin();
    while(seed != m_converge_seed.end())
    {
        FramePtr obs_frame = seed->last_update_frame;
        assert(obs_frame->isKeyframe());
        Sophus::SE3 Tfw_t = obs_frame->getFramePose();
        Vector3d p_world = seed->ftr->point->getPointPose();
        Vector3d p_target = Tfw_t * p_world;
        if(p_target[2] < 0.000001)
        {
            seed_converged_cb_(seed->ftr->point, seed->sigma2);
            seed = m_converge_seed.erase(seed);
            continue;
        }

        Feature* ftr = new Feature(obs_frame.get(), seed->last_matched_px, seed->last_matched_level);
        if(seed->ftr->type == Feature::CORNER)
            ftr->type = Feature::CORNER;
        else
            ftr->type = Feature::EDGELET;

        assert(seed->ftr->point->obs_.size() == 1);

        ftr->point = seed->ftr->point;
        ftr->point->addFrameRef(ftr);

        seed->ftr->frame->addFeature(seed->ftr);

        obs_frame->addFeature(ftr);

        seed = m_converge_seed.erase(seed);
    }
}

bool DepthFilter::getHaltFlag()
{
    boost::unique_lock<boost::mutex> lock(halt_mutex_);
    return seeds_updating_halt_;
}

void DepthFilter::setHaltFlag(bool flag)
{
    boost::unique_lock<boost::mutex> lock(halt_mutex_);
    seeds_updating_halt_ = flag;
}

void DepthFilter::setFrameHandler(FrameHandlerMono* frame_handler)
{
    frame_handler_ = frame_handler;
    map_ = frame_handler_->map_;
    imu_freq_ = frame_handler_->imu_freq_;
    imu_na_ = frame_handler_->imu_na_;
    imu_ng_ = frame_handler_->imu_ng_;
    imu_naw_ = frame_handler_->imu_naw_;
    imu_ngw_ = frame_handler_->imu_ngw_;
}

void DepthFilter::addKeyframe(FramePtr frame, bool is_track_good, double depth_mean, double depth_min, float converge_thresh)
{
    new_keyframe_min_depth_ = depth_min;
    new_keyframe_mean_depth_ = depth_mean;
    convergence_sigma2_thresh_ = converge_thresh;
    is_track_good_ = is_track_good;

    if(thread_ != NULL)
    {
        new_keyframe_ = frame;
        setNewKeyframeFlag(true);
        setHaltFlag(true);
        frame_queue_cond_.notify_one();
    }
    else
        initializeSeeds(frame);
}


void DepthFilter::processKeyFrame(bool is_tracking_good)
{
    if(!is_visual_opt_)
        return;
        
    size_t n_edges_init = 0, n_edges_fin = 0;
    double err_init = 0, err_final = 0;

    if(frame_handler_->is_vio_initialized_ )
    {
        ba::visualImuLocalBundleAdjustment(active_frame_.get(),
                                        &sub_map_,
                                        map_,
                                        is_tracking_good,
                                        n_edges_init,
                                        n_edges_fin,
                                        err_init,
                                        err_final);

    }
    else
    {
        ba::visualLocalBundleAdjustment(active_frame_.get(),
                                        &sub_map_,
                                        map_,
                                        n_edges_init,
                                        n_edges_fin,
                                        err_init,
                                        err_final);
    }

    if(!frame_handler_->is_vio_initialized_)
    {
        vioInitialization();
    }

    map_->setMapChangedFlag(true);
}


void DepthFilter::addLidar(LidarPtr lidar)
{
    if(thread_ != NULL)
    {
        {
            lock_t lock(lidar_queue_mut_);
            if(lidar_queue_.size() > 30)
            {
                LidarPtr poplidar = lidar_queue_.front();
                lidar_queue_.pop();
            }
              
            lidar_queue_.push(lidar);
        }
    }
}

void DepthFilter::trackLidarLocalMap()
{
    if(new_lidar_->id_==0)
    {
        last_lidar_ = new_lidar_;
        map_->addNewLidar(new_lidar_);
        return;
    }

    map_->addNewLidar(new_lidar_);
    if(frame_handler_->is_lio_initialized_ )
        predictWithImu();
    
    int valid_id[125];
    int surround_id[125];

    pcl::PointCloud<pcl::PointXYZI>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr local_map_kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> surf_downsize_filter;

    Sophus::SE3 T_map_wf = new_lidar_->getLidarPose().inverse();
    Eigen::Vector3d twf = T_map_wf.translation();
    Eigen::Matrix3d Rwf = T_map_wf.rotation_matrix();

    int centerCubeI = int((twf.x() + 25.0) / 50.0) + laserCloudCenWidth;
    int centerCubeJ = int((twf.y() + 25.0) / 50.0) + laserCloudCenHeight;
    int centerCubeK = int((twf.z() + 25.0) / 50.0) + laserCloudCenDepth;

    if (twf.x() + 25.0 < 0)
        centerCubeI--;
    if (twf.y() + 25.0 < 0)
        centerCubeJ--;
    if (twf.z() + 25.0 < 0)
        centerCubeK--;
        
    while (centerCubeI < 3)
    {
        for (int j = 0; j < laserCloudHeight; j++)
        {
            for (int k = 0; k < laserCloudDepth; k++)
            {
                int i = laserCloudWidth - 1;
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeCornerPointer =
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeSurfPointer =
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
 
                for (; i >= 1; i--)
                {
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        corner_features_array[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        surf_features_array[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                }
  
                corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeCornerPointer;
                surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeSurfPointer;
                CubeCornerPointer->clear();
                CubeSurfPointer->clear();
            }
        }

        centerCubeI++;
        laserCloudCenWidth++;
    }

    while (centerCubeI >= laserCloudWidth - 3)
    {
        for (int j = 0; j < laserCloudHeight; j++)
        {
            for (int k = 0; k < laserCloudDepth; k++)
            {
                int i = 0;
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeCornerPointer =
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeSurfPointer =
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                for (; i < laserCloudWidth - 1; i++)
                {
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        corner_features_array[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        surf_features_array[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                }
                corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeCornerPointer;
                surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeSurfPointer;
                CubeCornerPointer->clear();
                CubeSurfPointer->clear();
            }
        }

        centerCubeI--;
        laserCloudCenWidth--;
    }

    while (centerCubeJ < 3)
    {
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int k = 0; k < laserCloudDepth; k++)
            {
                int j = laserCloudHeight - 1;
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeCornerPointer =
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeSurfPointer =
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                for (; j >= 1; j--)
                {
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        corner_features_array[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        surf_features_array[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                }
                corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeCornerPointer;
                surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeSurfPointer;
                CubeCornerPointer->clear();
                CubeSurfPointer->clear();
            }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
    }

    while (centerCubeJ >= laserCloudHeight - 3)
    {
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int k = 0; k < laserCloudDepth; k++)
            {
                int j = 0;
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeCornerPointer =
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeSurfPointer =
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                for (; j < laserCloudHeight - 1; j++)
                {
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        corner_features_array[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        surf_features_array[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                }
                corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeCornerPointer;
                surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeSurfPointer;
                CubeCornerPointer->clear();
                CubeSurfPointer->clear();
            }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
    }

    while (centerCubeK < 3)
    {
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                int k = laserCloudDepth - 1;
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeCornerPointer =
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeSurfPointer =
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                for (; k >= 1; k--)
                {
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                }
                corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeCornerPointer;
                surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeSurfPointer;
                CubeCornerPointer->clear();
                CubeSurfPointer->clear();
            }
        }

        centerCubeK++;
        laserCloudCenDepth++;
    }

    while (centerCubeK >= laserCloudDepth - 3)
    {
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                int k = 0;
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeCornerPointer =
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                pcl::PointCloud<pcl::PointXYZI>::Ptr CubeSurfPointer =
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                for (; k < laserCloudDepth - 1; k++)
                {
                    corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                    surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                        surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                }
                corner_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeCornerPointer;
                surf_features_array[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                    CubeSurfPointer;
                CubeCornerPointer->clear();
                CubeSurfPointer->clear();
            }
        }

        centerCubeK--;
        laserCloudCenDepth--;
    }

    for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
    {
        for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
        {
            for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
            {
                if (i >= 0 && i < laserCloudWidth &&
                    j >= 0 && j < laserCloudHeight &&
                    k >= 0 && k < laserCloudDepth)
                {
                    valid_id[valid_num] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                    valid_num++;
                }
            }
        }
    }
    
    local_map->clear();
    for (int i = 0; i < valid_num; i++)
    {
        assert(valid_id[i]>=0 && valid_id[i]<=4850);
        *local_map += *surf_features_array[valid_id[i]];
    }
    
    int n_local_map_surf_features = local_map->points.size();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_surf_features_all = new_lidar_->ptr_less_surfs_;
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_surf_features(new pcl::PointCloud<pcl::PointXYZI>());
	surf_downsize_filter.setLeafSize(0.8, 0.8, 0.8);
    surf_downsize_filter.setInputCloud(cur_surf_features_all);
    surf_downsize_filter.filter(*cur_surf_features);
    int n_cur_surf_features = cur_surf_features->points.size();
    
    pcl::PointXYZI p_l, p_w;
    if (n_local_map_surf_features > 50)
    {
        local_map_kd_tree->setInputCloud(local_map);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        for (int iterCount = 0; iterCount < 1; iterCount++)
        {
            std::vector<Eigen::Vector3d> ps, nms;
            std::vector<double> ds;
            int surf_num = 0;
            for (int i = 0; i < n_cur_surf_features; i++)
            {
                p_l = cur_surf_features->points[i];

                Eigen::Vector3d pOri(p_l.x, p_l.y, p_l.z);
                Eigen::Vector3d pW = Rwf * pOri + twf;
                p_w.x = pW.x(); p_w.y = pW.y(); p_w.z = pW.z();
                p_w.intensity = p_l.intensity;

                local_map_kd_tree->nearestKSearch(p_w, 5, pointSearchInd, pointSearchSqDis);

                Eigen::Matrix<double, 5, 3> matA0;
                Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                if (pointSearchSqDis[4] < 1.0)
                {

                    for (int j = 0; j < 5; j++)
                    {
                        matA0(j, 0) = local_map->points[pointSearchInd[j]].x;
                        matA0(j, 1) = local_map->points[pointSearchInd[j]].y;
                        matA0(j, 2) = local_map->points[pointSearchInd[j]].z;
                    }
                    
                    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();

                    bool planeValid = true;
                    for (int j = 0; j < 5; j++)
                    {
                        if (fabs(norm(0) * local_map->points[pointSearchInd[j]].x +
                                    norm(1) * local_map->points[pointSearchInd[j]].y +
                                    norm(2) * local_map->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                        {
                            planeValid = false;
                            break;
                        }
                    }
                    Eigen::Vector3d cur_point(p_l.x, p_l.y, p_l.z);
                    if (planeValid)
                    {
                        ps.push_back(cur_point);
                        nms.push_back(norm);
                        ds.push_back(negative_OA_dot_norm);

                        surf_num++;
                    }
                }
            }
            
            if(!frame_handler_->is_lio_initialized_)
                ba::lidarOnePoseOptimization(T_map_wf, ds, ps, nms);
            else if(!new_lidar_->previous_lidar_->is_keylidar_)
            {
                ba::lidarImuOnePoseOptimization(new_lidar_, T_map_wf, ds, ps, nms);
            }
            else
            {
                ba::lidarImuOnePoseOptimizationFromKF(new_lidar_, T_map_wf, ds, ps, nms);
            }
            
            Sophus::SE3 Tlw_opt = T_map_wf.inverse();
            new_lidar_->setLidarPose(Tlw_opt);
        }
    }

    if(new_lidar_->is_keylidar_ && frame_handler_->is_lio_initialized_ )
    {
        new_lidar_->ptr_less_surfs_ = cur_surf_features;
        liLocalMapping(key_lidars_list_, local_map, local_map_kd_tree);
        T_map_wf = new_lidar_->getLidarPose().inverse();
    }

    for (int i = 0; i < n_cur_surf_features; i++)
    {
        Eigen::Vector3d pOri(cur_surf_features->points[i].x, cur_surf_features->points[i].y, cur_surf_features->points[i].z);
        Eigen::Vector3d pW = T_map_wf * pOri;
        p_w.x = pW.x(); p_w.y = pW.y(); p_w.z = pW.z();
        p_w.intensity = p_l.intensity;

        int cubeI = int((p_w.x + 25.0) / 50.0) + laserCloudCenWidth;
        int cubeJ = int((p_w.y + 25.0) / 50.0) + laserCloudCenHeight;
        int cubeK = int((p_w.z + 25.0) / 50.0) + laserCloudCenDepth;

        if (p_w.x + 25.0 < 0)
            cubeI--;
        if (p_w.y + 25.0 < 0)
            cubeJ--;
        if (p_w.z + 25.0 < 0)
            cubeK--;

        if (cubeI >= 0 && cubeI < laserCloudWidth &&
            cubeJ >= 0 && cubeJ < laserCloudHeight &&
            cubeK >= 0 && cubeK < laserCloudDepth)
        {
            int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
            surf_features_array[cubeInd]->push_back(p_w);
        }
    }

    for (int i = 0; i < valid_num; i++)
    {
        int ind = valid_id[i];

        pcl::PointCloud<pcl::PointXYZI>::Ptr tmpSurf(new pcl::PointCloud<pcl::PointXYZI>());
        surf_downsize_filter.setInputCloud(surf_features_array[ind]);
        surf_downsize_filter.setLeafSize(0.8, 0.8,0.8);
        surf_downsize_filter.filter(*tmpSurf);
        surf_features_array[ind] = tmpSurf;
    }

    lidar_local_map_ = local_map;
    Sophus::SE3 Tlw_last = last_lidar_->getLidarPose();
    frame_handler_->lidar_motion_model_ = Tlw_last * T_map_wf;

    Sophus::SE3 Tlw_ini = new_lidar_->getLidarPose();
    last_lidar_ = new_lidar_;
    if(new_lidar_->is_keylidar_)
        last_keylidar_ = new_lidar_;

    frameCount++;
}

void DepthFilter::predictWithImu()
{
    assert(last_lidar_->id_ + 1 == new_lidar_->id_);
    Eigen::Vector3d last_bg = last_lidar_->getImuGyroBias();
    Eigen::Vector3d last_ba = last_lidar_->getImuAccBias();
    Eigen::Vector3d cur_bg = new_lidar_->getImuGyroBias();
    Eigen::Vector3d cur_ba = new_lidar_->getImuAccBias();
    if( (last_bg - cur_bg).norm() >0.001 || (last_ba - cur_ba).norm() >0.001)
    {
        new_lidar_->setNewBias(last_ba,last_bg);

        if(new_lidar_->imu_from_last_lidar_)
            new_lidar_->imu_from_last_lidar_->reintegrate();
    }
    Eigen::Vector3d G; G << 0,0,-vilo::GRAVITY_VALUE;
    
    Eigen::Matrix3d q1 = last_lidar_->getImuRotation();
    Eigen::Vector3d p1 = last_lidar_->getImuPosition();
    Eigen::Vector3d v1 = last_lidar_->getVelocity();
    Eigen::Vector3d ba = last_lidar_->getImuAccBias();
    Eigen::Vector3d bg = last_lidar_->getImuGyroBias();

    Eigen::Matrix3d dq = new_lidar_->imu_from_last_lidar_->getDeltaRotation(bg);
    Eigen::Vector3d dp = new_lidar_->imu_from_last_lidar_->getDeltaPosition(ba,bg);
    Eigen::Vector3d dv = new_lidar_->imu_from_last_lidar_->getDeltaVelocity(ba,bg);
    double dt = new_lidar_->imu_from_last_lidar_->delta_t_;

    Eigen::Matrix3d q2 = vilo::normalizeRotation(q1*dq);
    Eigen::Vector3d v2 = v1 + G*dt + q1*dv;
    Eigen::Vector3d p2 = p1 + v1*dt + 0.5*dt*dt*G + q1 * dp;

    Sophus::SE3 Twb2 = Sophus::SE3(q2, p2);
    Sophus::SE3 Tbl = frame_handler_->T_b_l_;
    Sophus::SE3 Tlw_cur = (Twb2 * Tbl).inverse();
    new_lidar_->setLidarPose(Tlw_cur);
    new_lidar_->setVelocity(v2);
}

void DepthFilter::liLocalMapping(list< LidarPtr >& key_lidars_list,
                                pcl::PointCloud<pcl::PointXYZI>::Ptr local_map,
                                pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr local_map_kd_tree,
                                bool is_opt_each_bias)
{
    const int N = key_lidars_list.size();
    std::vector <std::vector<Eigen::Vector3d>> pts_vec_vec(N);
    std::vector <std::vector<Eigen::Vector3d>> nms_vec_vec(N);
    std::vector <std::vector<double>> ds_vec_vec(N);

    int n_scan = 0;
    auto lit = key_lidars_list.begin();
    while(lit != key_lidars_list.end())
    {
        if(lit == key_lidars_list.begin())
        {
            ++lit;
            ++n_scan;
            continue;
        }

        LidarPtr cur_lidar = *lit;
        pcl::PointCloud<pcl::PointXYZI>::Ptr features = cur_lidar->ptr_less_surfs_;
        Sophus::SE3 Twf = cur_lidar->getLidarPose().inverse();
        size_t n_pts = features->points.size();

        std::vector<int> ids;
        std::vector<float> sq_dis;
        std::vector<Eigen::Vector3d> pts_vec;
        std::vector<Eigen::Vector3d> nms_vec;
        std::vector<double> ds_vec;
        for(size_t i=0; i<n_pts; i=i+4)
        {
            pcl::PointXYZI p_l = features->points[i];

            Eigen::Vector3d pL(p_l.x, p_l.y, p_l.z);
            Eigen::Vector3d pW = Twf * pL;

            pcl::PointXYZI p_w;
            p_w.x = pW.x(); p_w.y = pW.y(); p_w.z = pW.z();
            p_w.intensity = p_l.intensity;

            local_map_kd_tree->nearestKSearch(p_w, 5, ids, sq_dis);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if (sq_dis[4] < 1.0)
            {

                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = local_map->points[ids[j]].x;
                    matA0(j, 1) = local_map->points[ids[j]].y;
                    matA0(j, 2) = local_map->points[ids[j]].z;
                }

                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double negative_OA_dot_norm = 1 / norm.norm();
                norm.normalize();

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    if (fabs(norm(0) * local_map->points[ids[j]].x +
                                norm(1) * local_map->points[ids[j]].y +
                                norm(2) * local_map->points[ids[j]].z + negative_OA_dot_norm) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }
                Eigen::Vector3d cur_point(p_l.x, p_l.y, p_l.z);
                if (planeValid)
                {
                    pts_vec.push_back(cur_point);
                    nms_vec.push_back(norm);
                    ds_vec.push_back(negative_OA_dot_norm);
                }
            }

        }

        pts_vec_vec[n_scan] = pts_vec;
        nms_vec_vec[n_scan] = nms_vec;
        ds_vec_vec[n_scan] = ds_vec;

        ++lit;
        ++n_scan;
    }

    bool is_synchronous = new_keyframe_->getTimeStamp()==new_lidar_->getTimeStamp()?true:false;
    if( (!frame_handler_->is_livo_initialized_) || (!is_synchronous) )
    {
        ba::lidarImuLocalBundleAdjustment(map_,
                                          key_lidars_list,
                                          pts_vec_vec,
                                          nms_vec_vec,
                                          ds_vec_vec,
                                          is_opt_each_bias);
    }
    else
    {
        LidarPtr head_lidar = *key_lidars_list.begin();
        if( 0!=frame_handler_->livo_init_lidar_id_
             && head_lidar->id_ >= frame_handler_->livo_init_lidar_id_)
        {
            ba::lidarVisualImuLocalBundleAdjustment(key_lidars_list,
                                                    pts_vec_vec,
                                                    nms_vec_vec,
                                                    ds_vec_vec,
                                                    new_keyframe_.get(),
                                                    map_);

        }
        else
        {
            ba::lidarVisualImuLocalBundleAdjustment(key_lidars_list,
                                                pts_vec_vec,
                                                nms_vec_vec,
                                                ds_vec_vec,
                                                new_keyframe_.get(),
                                                &sub_map_,
                                                map_);
        }
    }

    map_->setMapChangedFlag(true);
}

void DepthFilter::lioInitialization()
{
    list<LidarPtr> lidars_list = key_lidars_list_;
    auto iter = lidars_list.begin();
    Eigen::Vector3d drt_g(0,0,0);
    Eigen::Matrix3d Rwg;
    double scale; 
    double priorG;
    double priorA; 

    if(10==key_lidars_list_.size())
    {
        if(10==lidars_list.size())
        {
            while(iter != lidars_list.end() )
            {
                if (!(*iter)->imu_from_last_keylidar_)
                {
                    (*iter)->setVelocity( Eigen::Vector3d(0.0, 0.0, 0.0));
                    ++iter;
                    continue;
                }

                if (!(*iter)->last_key_lidar_)
                {
                    (*iter)->setVelocity( Eigen::Vector3d(0.0, 0.0, 0.0));
                    ++iter;
                    continue;
                }
   
                Eigen::Vector3d pos1 = (*iter)->getImuPosition();
                Eigen::Vector3d pos2 = (*iter)->last_key_lidar_->getImuPosition();
                double dt = (*iter)->ts_ - (*iter)->last_key_lidar_->ts_;
                Eigen::Vector3d vel = (pos1 - pos2) / dt;
                
                (*iter)->setVelocity(vel);
                (*iter)->last_key_lidar_->setVelocity(vel);
                drt_g -= (*iter)->last_key_lidar_->getImuRotation() * (*iter)->imu_from_last_keylidar_->getUpdateDeltaVelocity();
                ++iter;
            }

            drt_g = drt_g / drt_g.norm();
            Eigen::Vector3d drt_z(0,0,-1);
            Eigen::Vector3d v = drt_z.cross(drt_g);
            double cosg = drt_z.dot(drt_g);
            double ang = acos(cosg);
            Eigen::Vector3d vzg = v*ang/v.norm();
            Rwg = vilo::expSO3(vzg);

            priorG = 1e3;
            priorA = 1e6;
            scale = 1; 
        }
        
        ba::lidarImuAlign(lidars_list, Rwg, scale, false, priorG, priorA);
        Eigen::Matrix3d Rgw = Rwg.transpose();
    
        {
            boost::unique_lock<boost::mutex> lock(map_->map_mutex_);
            map_->rotateLidar(Rgw,Eigen::Vector3d(0,0,0));

            for (int i = 0; i < feature_array_num; i++)
            {
                size_t n_pts = surf_features_array[i]->points.size();
                for(size_t j=0; j<n_pts;j++)
                {
                    Eigen::Vector3d pt_w; pt_w << surf_features_array[i]->points[j].x,
                                                surf_features_array[i]->points[j].y,
                                                surf_features_array[i]->points[j].z;
                    Eigen::Vector3d pt_g = Rgw * pt_w;
                    surf_features_array[i]->points[j].x = pt_g[0];
                    surf_features_array[i]->points[j].y = pt_g[1];
                    surf_features_array[i]->points[j].z = pt_g[2];
                }
            }

            for(size_t i=0, iend=lidar_local_map_->points.size();i<iend;i++)
            {
                Eigen::Vector3d pt_w;
                pt_w << lidar_local_map_->points[i].x,
                        lidar_local_map_->points[i].y,
                        lidar_local_map_->points[i].z;

                Eigen::Vector3d pt_g = Rgw * pt_w;
                lidar_local_map_->points[i].x = pt_g[0];
                lidar_local_map_->points[i].y = pt_g[1];
                lidar_local_map_->points[i].z = pt_g[2];
            }

        }

        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr local_map_kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        local_map_kd_tree->setInputCloud(lidar_local_map_);
        liLocalMapping(lidars_list, lidar_local_map_, local_map_kd_tree, false);

        Sophus::SE3 Tlw_cur = new_lidar_->getLidarPose();
        Sophus::SE3 Tlw_ref = last_lidar_->getLidarPose();
        Sophus::SE3 T_ref_cur = Tlw_ref * Tlw_cur.inverse();
        frame_handler_->lidar_motion_model_ = T_ref_cur;

        frame_handler_->is_lio_initialized_ = true;
        frame_handler_->lio_init_id_ = new_lidar_->id_;

        map_->setMapChangedFlag(true);
    }

}

void DepthFilter::vioInitialization()
{
    if(map_->size() == 10)
    {
        Eigen::Vector3d drt_g(0,0,0);
        Eigen::Matrix3d Rwg;
        double scale = 1.0;
        double priorG = 1.0;
        double priorA = 1.0;

        list< FramePtr > keyframes = map_->keyframes_;
        auto iter = keyframes.begin();

        if(map_->size() == 10)
        {
            while(iter != keyframes.end() )
            {
                if (!(*iter)->imu_from_last_keyframe_)
                {
                    (*iter)->setVelocity( Eigen::Vector3d(0.0, 0.0, 0.0));
                    ++iter;
                    continue;
                }

                if (!(*iter)->last_kf_)
                {
                    (*iter)->setVelocity( Eigen::Vector3d(0.0, 0.0, 0.0));
                    ++iter;
                    continue;
                }

                Eigen::Vector3d pos1 = (*iter)->getImuPosition();
                Eigen::Vector3d pos2 = (*iter)->last_kf_->getImuPosition();
                Eigen::Vector3d vel = (pos1 - pos2) / (*iter)->imu_from_last_keyframe_->delta_t_;

                (*iter)->setVelocity(vel);
                (*iter)->last_kf_->setVelocity(vel);

                drt_g -= (*iter)->last_kf_->getImuRotation() * (*iter)->imu_from_last_keyframe_->getUpdateDeltaVelocity();

                ++iter;

            }

            drt_g = drt_g / drt_g.norm();
            Eigen::Vector3d drt_z(0,0,-1);
            Eigen::Vector3d v = drt_z.cross(drt_g);
            double cosg = drt_z.dot(drt_g);
            double ang = acos(cosg); 
            Eigen::Vector3d vzg = v*ang/v.norm(); 
            Rwg = vilo::expSO3(vzg);

            priorG = 1e2;
            priorA = 1e10;
            scale = 1.0;

        }
        else
        {
            Rwg.setIdentity();
            scale = 1.0;
            if(map_->size() == 80)
            {
                priorG = 1.0;
                priorA = 1e5;
            }
            if(map_->size() == 150)
            {
                priorG = 0.0;
                priorA = 0.0;
            }

        }

        ba::visualImuAlign(keyframes, Rwg, scale, false, priorG, priorA);
        if(scale < 0.1)
        {
            return;
        }

        iter = keyframes.begin();
        while(iter != keyframes.end() )
        {
            FramePtr kf = *iter;
            kf->is_opt_imu_ = true;
            ++iter;
        }

        {
            boost::unique_lock<boost::mutex> lock(map_->map_mutex_);
            map_->scaleRotate(Rwg.transpose(),scale, Eigen::Vector3d(0,0,0));
            map_->setMapChangedFlag(true);
            frame_handler_->is_vio_initialized_ = true;
        }
        
        if(map_->size() == 10)
            ba::visualImuFullBundleAdjustment(map_, false, priorG, priorA, 20);
        else
        {
            int n_its = 0;
            if(abs(scale-1.0)<=0.01)
                n_its = 2;
            else if(abs(scale-1.0)<=0.05)
                n_its = 4;
            else
                n_its = 8;
            ba::visualImuFullBundleAdjustment(map_, true, priorG, priorA, n_its);
        }

        map_->setMapChangedFlag(true);
    }

}

void DepthFilter::alignVioLio()
{
    boost::unique_lock<boost::mutex> lock(map_->map_mutex_);
    std::vector<LidarPtr> lidar_vec;
    std::vector<FramePtr> frame_vec;

    {
        if(map_->lidars_list_.size() < 20)
            return;

        for(auto it=map_->lidars_list_.begin();it!=map_->lidars_list_.end();++it)
        {
                lidar_vec.push_back(*it);
        }
        for(auto it=map_->keyframes_.begin();it!=map_->keyframes_.end();++it)
        {
            frame_vec.push_back(*it);
        }
    }

    int n_lidar = lidar_vec.size();
    int n_frame = frame_vec.size();
   
    std::vector<Sophus::SE3> frame_pose_vec, lidar_pose_vec;
    int cursor = 0;
    for(int i=0;i<n_frame;i++)
    {
        FramePtr frame = frame_vec[i];
        double image_ts = frame->getTimeStamp();
        for(int j=cursor;j<n_lidar-1;j++)
        {
            LidarPtr lidar1 = lidar_vec[j];
            double lidar_ts1 = lidar1->getTimeStamp();
            LidarPtr lidar2 = lidar_vec[j+1];
            double lidar_ts2 = lidar2->getTimeStamp();

            if(lidar_ts1 == image_ts)
            {
                Eigen::Matrix3d Rwb_l, Rwb_f;
                Eigen::Vector3d twb_l, twb_f;

                Rwb_l = lidar1->getImuRotation();
                Rwb_f = frame->getImuRotation();
                twb_l = lidar1->getImuPosition();
                twb_f = frame->getImuPosition();
                Sophus::SE3 Twb_lidar(Rwb_l, twb_l);
                Sophus::SE3 Twb_frame(Rwb_f, twb_f);
                frame_pose_vec.push_back(Twb_frame);
                lidar_pose_vec.push_back(Twb_lidar);

                cursor = j+1;
                break;
            }
            else
            {
                if(image_ts<lidar_ts2 && image_ts>lidar_ts1)
                {
                    double total_dt = image_ts - lidar_ts1;
                    assert(total_dt>0);
                    if(!lidar2->imu_from_last_lidar_)
                    {
                        cursor = j+1;
                        break;
                    }
                    vilo::IMU* imu_from_last_lidar = lidar2->imu_from_last_lidar_;
                    std::vector<double> dt_vec = imu_from_last_lidar->dt_vec_;
                    int n_dt = dt_vec.size();
                    double picked_dt = 0;
                    int n_pick = 0;
                    std::vector<double> imu_dt_vec;
                    std::vector<Eigen::Vector3d> imu_acc_vec, imu_gyro_vec;

                    for(int m=0;m<n_dt;m++)
                    {
                        if(picked_dt + dt_vec[m] <= total_dt)
                        {
                            picked_dt += dt_vec[m];
                            imu_dt_vec.push_back(dt_vec[m]);
                            imu_acc_vec.push_back(imu_from_last_lidar->acc_vec_[m]);
                            imu_gyro_vec.push_back(imu_from_last_lidar->gyro_vec_[m]);
                            n_pick++;
                        }
                        else
                        {
                            if(0==imu_acc_vec.size())
                            {
                                imu_acc_vec.push_back(imu_from_last_lidar->acc_vec_[m]);
                                imu_gyro_vec.push_back(imu_from_last_lidar->gyro_vec_[m]);
                                imu_dt_vec.push_back(total_dt - picked_dt);
                            }
                            else
                            {
                                double s = (total_dt - picked_dt) / dt_vec[m];
                                Eigen::Vector3d acc = (1-s)*imu_acc_vec.back()
                                                    + s*imu_from_last_lidar->acc_vec_[m];
                                Eigen::Vector3d gyro = (1-s)*imu_gyro_vec.back()
                                                    + s*imu_from_last_lidar->gyro_vec_[m];
                                imu_acc_vec.push_back(acc);
                                imu_gyro_vec.push_back(gyro);
                                imu_dt_vec.push_back(total_dt - picked_dt);
                            }

                            n_pick++;
                            break;
                        }

                    }

                    Eigen::Vector3d ba = imu_from_last_lidar->ba_;
                    Eigen::Vector3d bg = imu_from_last_lidar->bg_;
                    vilo::IMU* imu_image_lidar = new vilo::IMU(imu_freq_,
                                                    imu_na_, imu_ng_,
                                                    imu_naw_, imu_ngw_,
                                                    ba, bg);

                    for(int m=0;m<n_pick;m++)
                    {
                        imu_image_lidar->addImuPoint(imu_acc_vec[m],
                                                        imu_gyro_vec[m],
                                                        ba, bg, imu_dt_vec[m]);
                    }

                    Eigen::Vector3d G; G << 0,0,-vilo::GRAVITY_VALUE;
                    Eigen::Matrix3d q1 = lidar1->getImuRotation();
                    Eigen::Vector3d p1 = lidar1->getImuPosition();
                    Eigen::Vector3d v1 = lidar1->getVelocity();

                    Eigen::Matrix3d dq = imu_image_lidar->getDeltaRotation(bg);
                    Eigen::Vector3d dp = imu_image_lidar->getDeltaPosition(ba,bg);
                    Eigen::Vector3d dv = imu_image_lidar->getDeltaVelocity(ba,bg);
                    double dt = imu_image_lidar->delta_t_;

                    Eigen::Matrix3d q2 = vilo::normalizeRotation(q1*dq);
                    Eigen::Vector3d v2 = v1 + G*dt + q1*dv;
                    Eigen::Vector3d p2 = p1 + v1*dt + 0.5*dt*dt*G + q1 * dp;

                    Sophus::SE3 Twb_l = Sophus::SE3(q2, p2);

                    Eigen::Matrix3d Rwb_f = frame->getImuRotation();
                    Eigen::Vector3d twb_f = frame->getImuPosition();
                    Sophus::SE3 Twb_f(Rwb_f, twb_f);
                    frame_pose_vec.push_back(Twb_f);
                    lidar_pose_vec.push_back(Twb_l);
                    cursor = j+1;
                    break;
                }
            }

        }
    }

    Eigen::Matrix3d Ril_orient, Rli_position;
    Eigen::Vector3d tli_position;
    double scale = 0.0;
    ba::alignLidarImagePosition(Rli_position, tli_position, scale,
                                frame_pose_vec, lidar_pose_vec);
    ba::alignLidarImageOrientation(Ril_orient, frame_pose_vec, lidar_pose_vec);

    Sophus::SE3 Tbc = frame_handler_->T_b_c_;
    map_->scaleRotate(Tbc, Ril_orient, Rli_position, tli_position, scale);

    cursor = 0;
    for(int i=0;i<n_frame;i++)
    {
        FramePtr frame = frame_vec[i];
        double image_ts = frame->getTimeStamp();

        for(int j=cursor;j<n_lidar-1;j++)
        {
            LidarPtr lidar0 = lidar_vec[j];
            double lidar_ts0 = lidar0->getTimeStamp();
            LidarPtr lidar1 = lidar_vec[j+1];
            double lidar_ts1 = lidar1->getTimeStamp();

            if(image_ts>=lidar_ts0 && image_ts<lidar_ts1)
            {
                Eigen::Vector3d ba_lidar = lidar0->getImuAccBias();
                Eigen::Vector3d bg_lidar = lidar0->getImuGyroBias();
                Eigen::Vector3d ba_frame = frame->getImuAccBias();
                Eigen::Vector3d bg_frame = frame->getImuGyroBias();

                if ( (ba_lidar - ba_frame).norm() >0.01 ||
                        (bg_lidar - bg_frame).norm() >0.01 )
                {
                    frame->setNewBias(ba_lidar, bg_lidar);
                    if (frame->imu_from_last_keyframe_)
                        frame->imu_from_last_keyframe_->reintegrate();
                }

                cursor = j+1;
                break;
            }

        }
    }

    ba::visualMapOptimization(map_, lidar_pose_vec, Tbc);

    frame_handler_->is_livo_initialized_ = true;
    frame_handler_->livo_init_lidar_id_ = last_lidar_->id_;
    map_->init_id_ = last_lidar_->id_;
    map_->setMapChangedFlag(true);
}

void DepthFilter::localMapping(bool is_tracking_good)
{
    if(is_visual_opt_)
    {
        is_visual_opt_ = false;
        size_t n_edges_init = 0, n_edges_fin = 0;
        double err_init = 0, err_final = 0;

        if(frame_handler_->is_vio_initialized_ )
        {
            if(sub_map_.find(new_keyframe_.get()) != sub_map_.end())
                sub_map_.insert(new_keyframe_.get());

            ba::visualImuLocalBundleAdjustment(new_keyframe_.get(),
                                            &sub_map_,
                                            map_,
                                            is_tracking_good,
                                            n_edges_init,
                                            n_edges_fin,
                                            err_init,
                                            err_final);

        }
        else
        {
            assert(sub_map_.find(new_keyframe_.get()) != sub_map_.end());
            ba::visualLocalBundleAdjustment(new_keyframe_.get(),
                                            &sub_map_,
                                            map_,
                                            n_edges_init,
                                            n_edges_fin,
                                            err_init,
                                            err_final);
        }

        if(!frame_handler_->is_vio_initialized_)
        {
            vioInitialization();
        }

        map_->setMapChangedFlag(true);
    }
    if(is_new_lidar_)
    {
        is_new_lidar_ = false;
        trackLidarLocalMap();
        
        if( !frame_handler_->is_lio_initialized_ )
            lioInitialization();

        if(frame_handler_->is_lio_initialized_
                && frame_handler_->is_vio_initialized_
                    && (!frame_handler_->is_livo_initialized_))
                    alignVioLio();
    }

}


} // namespace vilo
