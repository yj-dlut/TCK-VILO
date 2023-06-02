#include <dirent.h>
#include <boost/timer.hpp>
#include <boost/thread.hpp>
#include "vilo/frame_handler_mono.h"
#include "vilo/bundle_adjustment.h"

namespace vilo {

FrameHandlerMono::FrameHandlerMono(vilo::AbstractCamera* cam, bool _use_pc) :
                                  FrameHandlerBase(),
                                  cam_(cam),
                                  depth_filter_(NULL)
{
    map_ = new Map();
    reprojector_ = new Reprojector(cam_, map_);

    if(_use_pc)
        photomatric_calib_ = new PhotomatricCalibration(2, cam_->width(), cam_->height());
    else
        photomatric_calib_ = NULL;

    initialize();
}

FrameHandlerMono::FrameHandlerMono(std::string& settingfile, bool _use_pc) :
                                  FrameHandlerBase(),
                                  cam_(NULL),
                                  depth_filter_(NULL)
{
    setting_file_ = settingfile;
    is_imu_used_ = true;

    if(!readConfigFile(settingfile))
        return;

    map_ = new Map();
    reprojector_ = new Reprojector(map_);
    reprojector_->initializeGrid(cam_);

    if(_use_pc)
        photomatric_calib_ = new PhotomatricCalibration(2, cam_->width(), cam_->height());
    else
        photomatric_calib_ = NULL;

    initialize();
}

void FrameHandlerMono::initialize()
{
    n_matched_fts_ = 0;
    n_edges_final_ = 0;
    distance_mean_ = 0;
    depth_min_ = 0;

    is_vio_initialized_ = false;
    is_lio_initialized_ = false;
    is_livo_initialized_ = false;
    is_first_lidar_ = false;
    lio_init_id_ = 0;
    livo_init_lidar_id_ = 0;

    lidar_motion_model_ = Sophus::SE3(Matrix3d::Identity(), Vector3d::Zero());
    visual_motion_model_ = Sophus::SE3(Matrix3d::Identity(), Vector3d::Zero());

    imu_from_last_keyframe_ = new vilo::IMU(imu_freq_,
                                           imu_na_,
                                           imu_ng_,
                                           imu_naw_,
                                           imu_ngw_);

    imu_from_last_keylidar_ = new vilo::IMU(imu_freq_,
                                           imu_na_,
                                           imu_ng_,
                                           imu_naw_,
                                           imu_ngw_);

    feature_detection::FeatureExtractor* featureExt(
        new feature_detection::FeatureExtractor(cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));

    DepthFilter::callback_t depth_filter_cb = boost::bind(&MapPointCandidates::newCandidatePoint, &(map_->point_candidates_), _1, _2);

    depth_filter_ = new DepthFilter(featureExt, depth_filter_cb);
    depth_filter_->startThread();
    depth_filter_->setFrameHandler(this);
    reprojector_->depth_filter_ = depth_filter_;
    n_lidar_ = 0;
}

FrameHandlerMono::~FrameHandlerMono()
{
    delete depth_filter_;
    if(photomatric_calib_ != NULL) delete photomatric_calib_;
}

void FrameHandlerMono::addLidarImageImu(pcl::PointCloud<pcl::PointXYZI>::Ptr lidar,
                                        cv::Mat& image,
                                        const std::vector<vilo::IMUPoint>& imu_vec,
                                        std::vector<double>& pts_ts_vec,
                                        const double& lidar_ts,
                                        const double& image_ts,
                                        int imu_id1, int imu_id2,
                                        int imu_id3, int imu_id4,
                                        int type)
{
    struct timeval st,et;
    gettimeofday(&st,NULL);
    {
        boost::unique_lock<boost::mutex> lock(map_->map_mutex_);
        UpdateResult res = RESULT_FAILURE;

        if(0 == type)
        {
            is_lidar_img_pair_ = false;
            std::vector<vilo::IMUPoint> frame_imu_vec;
            for(int i=imu_id1; i<=imu_id2; i++)
                frame_imu_vec.push_back(imu_vec[i]);

            addImageImu(image, frame_imu_vec, image_ts, res);

        }
        else if(1 == type)
        {
            is_lidar_img_pair_ = true;
            std::vector<vilo::IMUPoint> frame_imu_vec, lidar_imu_vec;
            for(int i=imu_id1; i<=imu_id2; i++)
                frame_imu_vec.push_back(imu_vec[i]);

            double new_lidar_ts;
            if(!is_lio_initialized_)
            {
                for(int i=imu_id3; i<=imu_id4; i++)
                    lidar_imu_vec.push_back(imu_vec[i]);
                new_lidar_ts = lidar_ts;
            }
            else
            {
                assert(imu_id3 < imu_id2);
                for(int i=imu_id3; i<=imu_id2; i++)
                    lidar_imu_vec.push_back(imu_vec[i]);

                new_lidar_ts = image_ts;

            }
            
            boost::thread t_lidar(&FrameHandlerMono::addLidarImu, this,
                                lidar, lidar_imu_vec, new_lidar_ts, pts_ts_vec);

            addImageImu(image, frame_imu_vec, image_ts, res);
            t_lidar.join();

            new_frame_->align_lid_ = new_lidar_->id_;
            new_lidar_->align_vid_ = new_frame_->id_;

        }

        if(STAGE_DEFAULT_FRAME == stage_)
        {
            if(RESULT_IS_KEYFRAME == res)
            {
                bool is_good_track = (n_matched_fts_ >= 100 ? true : false);
                map_->addKeyframe(new_frame_);

                if(is_vio_initialized_)
                {
                    Eigen::Vector3d ba = new_frame_->ba_;
                    Eigen::Vector3d bg = new_frame_->bg_;
                    imu_from_last_keyframe_ = new vilo::IMU(imu_freq_,
                                                            imu_na_,  imu_ng_,
                                                            imu_naw_,  imu_ngw_,
                                                            ba, bg);
                }
                else
                {
                    imu_from_last_keyframe_ = new vilo::IMU(imu_freq_,
                                                            imu_na_, imu_ng_,
                                                            imu_naw_, imu_ngw_);
                }

                last_keyframe_ = new_frame_;
                
                if(n_edges_final_ <= 70)
                    depth_filter_->addKeyframe(new_frame_, is_good_track, distance_mean_, 0.5*depth_min_, 100);
                else
                    depth_filter_->addKeyframe(new_frame_, is_good_track, distance_mean_, 0.5*depth_min_, 200);
                
            }
            else if(RESULT_NO_KEYFRAME == res)
            {
                depth_filter_->addFrame(new_frame_);
            }

            Sophus::SE3 Tfw_cur = new_frame_->getFramePose();
            Sophus::SE3 Tfw_last = last_frame_->getFramePose();
            visual_motion_model_ = Tfw_cur * Tfw_last.inverse();
            
            while(visual_motions_.size() >= 20)
                visual_motions_.pop_front();
            visual_motions_.push_back(make_pair(last_frame_->id_, visual_motion_model_));

        }

        last_frame_ = new_frame_;
        new_frame_.reset();
        finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->m_n_inliers);

        if(1 == type)
        {
            if(is_first_lidar_)
            {
                is_first_lidar_ = false;
            }

            n_lidar_++;
            last_lidar_ = new_lidar_;

            if(new_lidar_->is_keylidar_)
                last_key_lidar_ = new_lidar_.get();
            
            depth_filter_->addLidar(new_lidar_);
        }

    }
    
    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
    if(time_use/1000 < 20)
    {
        usleep(1000 * (20-time_use/1000));
    }

    usleep(50000);
    gettimeofday(&et,NULL);
    time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

void FrameHandlerMono::addImageImu(const cv::Mat& img,
                                   const std::vector<vilo::IMUPoint>& imu_vec,
                                   const double& timestamp,
                                   UpdateResult& res)
{
    struct timeval st,et;
    gettimeofday(&st,NULL);

    if(!startFrameProcessingCommon(timestamp))
        return;

    if(is_imu_used_)
        new_frame_.reset(new Frame(cam_, img.clone(), timestamp, T_b_c_, photomatric_calib_));
    else
        new_frame_.reset(new Frame(cam_, img.clone(), timestamp, photomatric_calib_));

    if(map_->size() == 0)
        new_frame_->keyFrameId_ = 0;
    else
        new_frame_->keyFrameId_ = map_->lastKeyframe()->keyFrameId_;

    new_frame_->setTimeStamp(timestamp);
    if(last_frame_ && is_imu_used_)
    {
        new_frame_->setImuAccBias(last_frame_->getImuAccBias());
        new_frame_->setImuGyroBias(last_frame_->getImuGyroBias());
    }

    preintegrate(imu_vec, 0);

    if(stage_ == STAGE_DEFAULT_FRAME)
        res = processFrame();
    else if(stage_ == STAGE_SECOND_FRAME)
        res = processSecondFrame();
    else if(stage_ == STAGE_FIRST_FRAME)
        res = processFirstFrame();
    else if(stage_ == STAGE_RELOCALIZING)
        res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()), map_->getClosestKeyframe(last_frame_));

    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

void FrameHandlerMono::addLidarImu(pcl::PointCloud<pcl::PointXYZI>::Ptr p_cloud,
                                const std::vector<vilo::IMUPoint>& imu_vec,
                                const double& timestamp,
                                std::vector<double>& pts_ts_vec)
{
    struct timeval st,et;
    gettimeofday(&st,NULL);
    new_lidar_.reset(new Lidar(p_cloud));
    new_lidar_->setTbl(T_b_l_);
    new_lidar_->setTimeStamp(timestamp);
     
    if(n_lidar_ == 0)
    {
        new_lidar_->registerPoints();
        new_lidar_->extractFeatures();
        Eigen::Vector3d new_ba = Eigen::Vector3d(0,0,0);
        Eigen::Vector3d new_bg = Eigen::Vector3d(0,0,0);
        new_lidar_->setNewBias(new_ba, new_bg);

        key_lidars_list_.push_back(new_lidar_);
        last_key_lidar_ = new_lidar_.get();
        is_first_lidar_ = true;
        return;
    }
    else
    {
        Eigen::Vector3d new_ba = last_lidar_->getImuAccBias();
        Eigen::Vector3d new_bg = last_lidar_->getImuGyroBias();
        new_lidar_->setNewBias(new_ba, new_bg);
    }
    
    if(last_lidar_)
    {
        new_lidar_->previous_lidar_ = last_lidar_.get();
    }
    
    preintegrate(imu_vec, 1); 
    
    bool is_undist_pts = true;
    if(is_undist_pts && is_lio_initialized_)
        new_lidar_->undistPoints(lidar_motion_model_, pts_ts_vec);
        
    new_lidar_->registerPoints();
    
    new_lidar_->extractFeatures();
    
    if(!is_lio_initialized_)
    {
        Sophus::SE3 dTrc = lidar_motion_model_;
        trackNewLidarFrame(new_lidar_, last_lidar_, dTrc);
        Sophus::SE3 Tlw_last = last_lidar_->getLidarPose();
        Sophus::SE3 Tlw_cur = dTrc.inverse() * Tlw_last;
        new_lidar_->setLidarPose(Tlw_cur);

        Eigen::Vector3d p_w1 = (T_b_l_ * Tlw_cur).inverse().translation();
        Eigen::Vector3d p_w2 = (T_b_l_ * Tlw_last).inverse().translation();
        Eigen::Vector3d vel = (p_w1 - p_w2)/new_lidar_->imu_from_last_lidar_->delta_t_;
        new_lidar_->setVelocity(vel);
    }
    else 
    {
        predictWithImu();
    }
    
    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
    new_frame_->imu_from_last_keyframe_ = NULL;
    Sophus::SE3 Tfw(Matrix3d::Identity(), Vector3d::Zero());
    new_frame_->setFramePose(Tfw);

    if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
        return RESULT_NO_KEYFRAME;
    new_frame_->setKeyframe();
    map_->addKeyframe(new_frame_);
    stage_ = STAGE_SECOND_FRAME;
    VILO_INFO_STREAM("Init: Selected first frame.");
    first_frame_ = new_frame_;

    first_frame_->m_exposure_time = 1.0;
    first_frame_->m_exposure_finish = true;


    m_stamp_et.push_back(make_pair(first_frame_->m_timestamp_s, 1));
    m_grad_mean.push_back(first_frame_->gradMean_);

    imu_from_last_keyframe_ = new vilo::IMU(imu_freq_,
                                           imu_na_,
                                           imu_ng_,
                                           imu_naw_,
                                           imu_ngw_);

    return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
    initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);

    m_stamp_et.push_back(make_pair(new_frame_->m_timestamp_s, 1));
    m_grad_mean.push_back(new_frame_->gradMean_);

    if(res == initialization::FAILURE)
        return RESULT_FAILURE;
    else if(res == initialization::NO_KEYFRAME)
    {
        if(new_frame_->id_ - first_frame_->id_ > 20) 
            set_start_ = true;
        return RESULT_NO_KEYFRAME;
    }

    stage_ = STAGE_DEFAULT_FRAME;
    klt_homography_init_.reset();
    VILO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");

    after_init_ = true;
    first_frame_->setKeyPoints();

    return RESULT_NO_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
    struct timeval st,et;
    gettimeofday(&st,NULL);

    Sophus::SE3 Tfw_last = last_frame_->getFramePose();
    Sophus::SE3 Tfw_cur = visual_motion_model_ * Tfw_last;
    new_frame_->setFramePose(Tfw_cur);

    if(after_init_)
        last_frame_ = first_frame_;

    new_frame_->m_last_frame = last_frame_;

    bool is_map_changed = map_->getMapChangedFlag();
    if(is_vio_initialized_)
    {
        Eigen::Vector3d G; G << 0,0,-vilo::GRAVITY_VALUE;

        FramePtr last_kf = map_->lastKeyframe();
        new_frame_->last_kf_ = last_kf.get();
        if(new_frame_->imu_from_last_keyframe_ && is_map_changed)
        {
            Eigen::Matrix3d q1 = last_keyframe_->getImuRotation();
            Eigen::Vector3d p1 = last_keyframe_->getImuPosition();
            Eigen::Vector3d v1 = last_keyframe_->getVelocity();

            Eigen::Vector3d ba = last_keyframe_->getImuAccBias();
            Eigen::Vector3d bg = last_keyframe_->getImuGyroBias();
 
            Eigen::Matrix3d dq = new_frame_->imu_from_last_keyframe_->getDeltaRotation(bg);
            Eigen::Vector3d dp = new_frame_->imu_from_last_keyframe_->getDeltaPosition(ba,bg);
            Eigen::Vector3d dv = new_frame_->imu_from_last_keyframe_->getDeltaVelocity(ba,bg);
            double dt = new_frame_->imu_from_last_keyframe_->delta_t_;

            Eigen::Matrix3d q2 = vilo::normalizeRotation(q1*dq);
            Eigen::Vector3d p2 = p1 + v1*dt + 0.5*dt*dt*G + q1 * dp;
            Eigen::Vector3d v2 = v1 + G*dt + q1*dv;

            Sophus::SE3 Twb2 = Sophus::SE3(q2, p2);
            Tfw_cur = (Twb2 * T_b_c_).inverse();
            new_frame_->setFramePose(Tfw_cur);
            new_frame_->setVelocity(v2);

            new_frame_->setImuAccBias(ba);
            new_frame_->setImuGyroBias(bg);
        }
        else
        {
            Eigen::Matrix3d q1 = last_frame_->getImuRotation();
            Eigen::Vector3d p1 = last_frame_->getImuPosition();
            Eigen::Vector3d v1 = last_frame_->getVelocity();

            Eigen::Vector3d ba = last_frame_->getImuAccBias();
            Eigen::Vector3d bg = last_frame_->getImuGyroBias();

            Eigen::Matrix3d dq = new_frame_->imu_from_last_frame_->getDeltaRotation(bg);
            Eigen::Vector3d dp = new_frame_->imu_from_last_frame_->getDeltaPosition(ba,bg);
            Eigen::Vector3d dv = new_frame_->imu_from_last_frame_->getDeltaVelocity(ba,bg);
            double dt = new_frame_->imu_from_last_frame_->delta_t_;

            Eigen::Matrix3d q2 = vilo::normalizeRotation(q1*dq);
            Eigen::Vector3d p2 = p1 + v1*dt + 0.5*dt*dt*G + q1 * dp;
            Eigen::Vector3d v2 = v1 + G*dt + q1*dv;

            Sophus::SE3 Twb2 = Sophus::SE3(q2, p2);
            Tfw_cur = (Twb2 * T_b_c_).inverse();
            new_frame_->setFramePose(Tfw_cur);
            new_frame_->setVelocity(v2);
        }

    }
    else
    {
        if(is_map_changed)
        {
            FramePtr last_kf = map_->lastKeyframe();
            assert(last_kf != NULL);
            assert(last_kf->id_ <= last_frame_->id_);
            Sophus::SE3 Tfw = last_kf->getFramePose();
            
            for(auto lit=visual_motions_.begin();lit!=visual_motions_.end();++lit)
            {
                if(lit->first < last_kf->id_)
                    continue;

                if(lit->first < last_frame_->id_)
                { 
                    Tfw = lit->second * Tfw; 
                }
                else
                {
                    last_frame_->setFramePose(Tfw);
                    break;
                }
                
            }
        }
        if(new_frame_->gradMean_ > last_frame_->gradMean_ + 1.2)
        {
            CoarseTracker Tracker(false, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);
            size_t n_tracked = Tracker.run(last_frame_, new_frame_);
        }
        else
        {
            CoarseTracker Tracker(true, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);
            size_t n_tracked = Tracker.run(last_frame_, new_frame_);
        }

    }

    m_stamp_et.push_back(make_pair(new_frame_->m_timestamp_s, new_frame_->m_exposure_time));
    m_grad_mean.push_back(new_frame_->gradMean_);

    reprojector_->reprojectMap(new_frame_);

    n_matched_fts_ = reprojector_->n_matches_; 
    if(n_matched_fts_ < Config::qualityMinFts())
    {
        new_frame_->setFramePose(Tfw_last);
        tracking_quality_ = TRACKING_INSUFFICIENT;
        return RESULT_FAILURE;
    }
    
    double opt_thresh = 0, error_init = 0, error_final = 0;
    if(is_vio_initialized_)
    {
        if(is_map_changed)
        {
            ba::visualImuOnePoseOptimizationFromKF(new_frame_,
                                                  error_init,
                                                  error_final,
                                                  n_edges_final_);
        }
        else
            ba::visualImuOnePoseOptimization(new_frame_,
                                             error_init,
                                             error_final,
                                             n_edges_final_);

    }
    else
    {
        pose_optimizer::optimizeLevenbergMarquardt3rd(
            Config::poseOptimThresh(), 12, false,
            new_frame_, opt_thresh, error_init, error_final, n_edges_final_);

    }
    new_frame_->m_n_inliers = n_edges_final_;

    if(is_map_changed)
        map_->setMapChangedFlag(false);

    setTrackingQuality(n_edges_final_);
    if(tracking_quality_ == TRACKING_INSUFFICIENT)
    {
        new_frame_->setFramePose(Tfw_last);
        return RESULT_FAILURE;
    }

    double depth_mean;
    frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min_);
    frame_utils::getSceneDistance(*new_frame_, distance_mean_);

    bool is_kf = needNewKf(distance_mean_, n_matched_fts_);
    if(!is_kf && !after_init_)
    {
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), false);
        regular_counter_++;

        Tfw_cur = new_frame_->getFramePose();
        visual_motion_model_ = Tfw_cur * Tfw_last.inverse();
        return RESULT_NO_KEYFRAME;
    }
    else
    {
        if(after_init_)
            after_init_ = false;

        regular_counter_ = 0;
        new_frame_->setKeyframe();

        for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
            if((*it)->point != NULL)
                (*it)->point->addFrameRef(*it);

        map_->point_candidates_.addCandidatePointToFrame(new_frame_);

        createCovisibilityGraph(new_frame_, Config::coreNKfs(), true);

        return RESULT_IS_KEYFRAME;
    }

}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(const SE3& T_cur_ref, FramePtr ref_keyframe)
{
    if(ref_keyframe == nullptr)
        return RESULT_FAILURE;

    CoarseTracker Tracker(true, Config::kltMaxLevel(), Config::kltMinLevel(), 15, false);
    size_t n_tracked = Tracker.run(ref_keyframe, last_frame_);

    if(n_tracked > 30)
    {
        SE3 T_f_w_last = last_frame_->getFramePose();
        last_frame_ = ref_keyframe;
        FrameHandlerMono::UpdateResult res = processFrame();
        if(res != RESULT_FAILURE)
        {
            stage_ = STAGE_DEFAULT_FRAME;
        }
        else
            new_frame_->setFramePose(T_f_w_last); 

        return res;
    }
    return RESULT_FAILURE;
}

void FrameHandlerMono::resetAll()
{
    resetCommon();
    last_frame_.reset();
    new_frame_.reset();
    depth_filter_->reset();
}

bool FrameHandlerMono::needNewKf(const double& scene_depth_mean, const size_t& num_observations)
{
    if( is_lidar_img_pair_
         && is_livo_initialized_
           && (last_lidar_->id_+1) % step_==0)
        return true;
        
    if(is_imu_used_ && !is_vio_initialized_)
    {
        if(regular_counter_ > 5)
            return true;
    }

    if(is_imu_used_ && regular_counter_ > 6)
        return true;

    

    if(new_frame_->m_n_inliers > 0 && last_frame_->m_n_inliers > 0)
    {
        if(1.0 * new_frame_->m_n_inliers / last_frame_->m_n_inliers < 0.3)
        {
            return true;
        }

    }

    const FramePtr last_kf = map_->lastKeyframe();
    Sophus::SE3 Tfw_cur = new_frame_->getFramePose();
    Sophus::SE3 Tfw_last = last_frame_->getFramePose();
    const SE3 T_c_r_full = Tfw_cur * Tfw_last.inverse();
    const SE3 T_c_r_nR(Matrix3d::Identity(), T_c_r_full.translation());

    float optical_flow_full = 0, optical_flow_nR = 0, optical_flow_nt = 0;
    size_t optical_flow_num = 0;

    for(auto& ft_kf: last_kf->fts_)
    {
        if(ft_kf->point == NULL)
            continue;
        
        Vector3d p_w = ft_kf->point->getPointPose();
        Vector3d p_ref(ft_kf->f * (p_w - last_kf->pos()).norm());
        Vector3d p_cur_full(T_c_r_full * p_ref);
        Vector3d p_cur_nR(T_c_r_nR * p_ref);

        Vector2d uv_cur_full(new_frame_->cam_->world2cam(p_cur_full));
        Vector2d uv_cur_nR(new_frame_->cam_->world2cam(p_cur_nR));

        optical_flow_full += (uv_cur_full - ft_kf->px).squaredNorm();
        optical_flow_nR += (uv_cur_nR - ft_kf->px).squaredNorm();

        optical_flow_num++;
    }

    optical_flow_full /= optical_flow_num;
    if(optical_flow_full < 133)
        return false;

    optical_flow_full = sqrtf(optical_flow_full);
    optical_flow_nR /= optical_flow_num;
    optical_flow_nR = sqrtf(optical_flow_nR);

    size_t n_mean_converge_frame;
    {
        boost::unique_lock<boost::mutex> lock(depth_filter_->mean_mutex_);
        n_mean_converge_frame = depth_filter_->nMeanConvergeFrame_;
    }

    if(int(regular_counter_) < std::min(3, int(n_mean_converge_frame*0.8))) return false;

    const int defult_resolution = 752+480;
    const float setting_maxShiftWeightT = 0.04*defult_resolution;
    const float setting_maxShiftWeightRT = 0.02*defult_resolution;
    const float setting_kfGlobalWeight = 0.75;

    const int wh = new_frame_->cam_->width() + new_frame_->cam_->height();

    float DSO = setting_kfGlobalWeight*setting_maxShiftWeightT* optical_flow_nR   / wh+
                setting_kfGlobalWeight*setting_maxShiftWeightRT*optical_flow_full / wh;

    return DSO > 1;

    return false;
}

bool FrameHandlerMono::frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs)
{
    if(lhs.first != rhs.first)
        return (lhs.first > rhs.first);
    else
        return (lhs.second->id_ < rhs.second->id_);

}

void FrameHandlerMono::createCovisibilityGraph(FramePtr currentFrame, size_t n_closest, bool is_keyframe)
{
    std::map<Frame*, int> KFcounter;
    int n_linliers = 0;
    for(Features::iterator it = currentFrame->fts_.begin(); it != currentFrame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;

        n_linliers++;

        for(auto ite = (*it)->point->obs_.begin(); ite != (*it)->point->obs_.end(); ++ite)
        {
            if((*ite)->frame->id_== currentFrame->id_) continue;
            KFcounter[(*ite)->frame]++;
        }
    }

    if(KFcounter.empty()) return;

    int nmax=0;
    Frame* pKFmax=NULL;
    const int th = n_linliers > 30? 5 : 3;

    vector< pair<int, Frame*> > vPairs;
    vPairs.reserve(KFcounter.size());


    for(std::map<Frame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }

        if(mit->second>=th)
            vPairs.push_back(make_pair(mit->second,mit->first));

        if(mit->first->keyFrameId_+5 < currentFrame->keyFrameId_)
            if(!mit->first->sobelX_.empty())
            {
                mit->first->sobelX_.clear();
                mit->first->sobelY_.clear();
            }
    }

    if(vPairs.empty())
        vPairs.push_back(make_pair(nmax,pKFmax));

    std::partial_sort(vPairs.begin(), vPairs.begin()+vPairs.size(), vPairs.end(), boost::bind(&FrameHandlerMono::frameCovisibilityComparator, _1, _2));

    const size_t nCovisibility = 5;
    size_t k = min(nCovisibility, vPairs.size());
    for(size_t i = 0; i < k; ++i)
        currentFrame->connectedKeyFrames.push_back(vPairs[i].second);

    if(is_keyframe)
    {
        sub_map_.clear();

        size_t n = min(n_closest, vPairs.size());
        std::for_each(vPairs.begin(), vPairs.begin()+n, [&](pair<int, Frame*>& i){ sub_map_.insert(i.second); });

        FramePtr LastKF = map_->lastKeyframe();
        if(sub_map_.find(LastKF.get()) == sub_map_.end())
            sub_map_.insert(LastKF.get());

        assert(sub_map_.find(currentFrame.get()) == sub_map_.end());
        sub_map_.insert(currentFrame.get());

        currentFrame->last_kf_ = LastKF.get();
    }
}

bool FrameHandlerMono::readConfigFile(std::string& settingfile)
{
    cv::FileStorage fSettings(settingfile.c_str(), cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << settingfile << endl;
       exit(-1);
    }
    bool b_miss_params = false;

    cout << endl << "Camera Parameters: " << endl;
    string sCameraName = fSettings["Camera.type"];

    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        float k1, k2, p1, p2, k3;

        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            p1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            p2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << k1 << std::endl;
        std::cout << "- k2: " << k2 << std::endl;
        std::cout << "- p1: " << p1 << std::endl;
        std::cout << "- p2: " << p2 << std::endl;

        if(b_miss_params)
        {
            return false;
        }

        int width_i = 0, height_i = 0;
        node = fSettings["Camera.width"];
        if(!node.empty() && node.isInt())
        {
            width_i = node.operator int();
        }
        else
        {
            std::cerr << "*Camera.weight parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.height"];
        if(!node.empty() && node.isInt())
        {
            height_i = node.operator int();
        }
        else
        {
            std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        if(b_miss_params)
        {
            return false;
        }
        else
        {
            std::cout << "- width: " << width_i << std::endl;
            std::cout << "- height: " << height_i << std::endl;
            cam_ = new vilo::PinholeCamera(width_i, height_i, fx, fy, cx, cy, k1, k2, p1, p2);
            std::cout<<"Pinhole Camera model initialized!"<<std::endl;
        }
    }
    else
    {
        cout<<"Please use PinHole model!"<<endl;
        return false;
    }

    {
        Eigen::Matrix3d rbc;
        Eigen::Vector3d tbc;
        cv::Mat Tbc;
        cv::FileNode node = fSettings["Tbc"];
        if(!node.empty())
        {
            Tbc = node.mat();
            if(Tbc.rows != 4 || Tbc.cols != 4)
            {
                std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;

                return false;
            }
            for(int i=0;i<3;i++)
            {
                for(int j=0;j<3;j++)
                {
                    rbc(i,j) = Tbc.at<float>(i,j);
                }
            }

            tbc << Tbc.at<float>(0,3), Tbc.at<float>(1,3), Tbc.at<float>(2,3);

            T_b_c_ = SE3(rbc,tbc);
        }
        else
        {
            std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
            return false;
        }
    }

    {
        Eigen::Matrix3d rbl;
        Eigen::Vector3d tbl;
        cv::Mat Tbl;
        cv::FileNode node = fSettings["Tbl"];
        if(!node.empty())
        {
            Tbl = node.mat();
            if(Tbl.rows != 4 || Tbl.cols != 4)
            {
                std::cerr << "*Tbl matrix have to be a 4x4 transformation matrix*" << std::endl;
                return false;
            }
            for(int i=0;i<3;i++)
            {
                for(int j=0;j<3;j++)
                {
                    rbl(i,j) = Tbl.at<float>(i,j);
                }
            }

            tbl << Tbl.at<float>(0,3), Tbl.at<float>(1,3), Tbl.at<float>(2,3);

            T_b_l_ = SE3(rbl,tbl);
        }
        else
        {
            std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
            return false;
        }
    }

    if(is_imu_used_)
    {
        cv::FileNode node = fSettings["IMU.Frequency"];
        if(!node.empty() && node.isInt())
        {
            imu_freq_ = node.operator int();
        }
        else
        {
            std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseGyro"];
        if(!node.empty() && node.isReal())
        {
            imu_ng_ = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseAcc"];
        if(!node.empty() && node.isReal())
        {
            imu_na_ = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.GyroWalk"];
        if(!node.empty() && node.isReal())
        {
            imu_ngw_ = node.real();
        }
        else
        {
            std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.AccWalk"];
        if(!node.empty() && node.isReal())
        {
            imu_naw_ = node.real();
        }
        else
        {
            std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        if(b_miss_params)
            return false;
        else
        {
            cout<<"IMU parameters:"<<endl;
            cout<<"Frequency:"<<imu_freq_<<endl;
            cout<<"Acc Noise:"<<imu_na_<<endl;
            cout<<"Gyro Noise:"<<imu_ng_<<endl;
            cout<<"Acc walk:"<<imu_naw_<<endl;
            cout<<"Gyro walk:"<<imu_ngw_<<endl;
        }


    }

    return true;
}

void FrameHandlerMono::preintegrate(const std::vector<vilo::IMUPoint>& imu_vec, int type)
{
    if(0 == type)
    {
        vilo::IMU* imu_from_last_frame = NULL;
        if(last_frame_ && last_frame_->is_opt_imu_)
            imu_from_last_frame = new vilo::IMU(imu_freq_,
                                            imu_na_, imu_ng_,
                                            imu_naw_, imu_ngw_,
                                            last_frame_->ba_, last_frame_->bg_);
        else
            imu_from_last_frame = new vilo::IMU(imu_freq_,
                                            imu_na_, imu_ng_,
                                            imu_naw_, imu_ngw_);

        double t_s, t_e;
        t_e = new_frame_->getTimeStamp();
        if(last_frame_ != NULL)
            t_s = last_frame_->getTimeStamp();
        else
            t_s = imu_vec[0].t;

        int N = imu_vec.size()-1;
        for(int i=0;i<N;i++)
        {
            Eigen::Vector3d acc, gyro;
            double dt;
            if((i==0) && (i<(N-1)))
            {
                double tab = imu_vec[i+1].t-imu_vec[i].t;
                double tini = imu_vec[i].t-t_s;
                acc = (imu_vec[i].acc + imu_vec[i+1].acc - (imu_vec[i+1].acc-imu_vec[i].acc)*(tini/tab))  *0.5f;
                gyro = (imu_vec[i].gyro + imu_vec[i+1].gyro - (imu_vec[i+1].gyro-imu_vec[i].gyro)*(tini/tab))*0.5f;
                dt = imu_vec[i+1].t-t_s;
            }
            else if(i<(N-1))
            {
                acc = (imu_vec[i].acc + imu_vec[i+1].acc)*0.5f;
                gyro = (imu_vec[i].gyro + imu_vec[i+1].gyro)*0.5f;
                dt = imu_vec[i+1].t-imu_vec[i].t;
            }
            else if((i>0) && (i==(N-1)))
            {
                float tab = imu_vec[i+1].t-imu_vec[i].t;
                float tend = imu_vec[i+1].t-t_e;
                acc = (imu_vec[i].acc + imu_vec[i+1].acc - (imu_vec[i+1].acc-imu_vec[i].acc)*(tend/tab))*0.5f;
                gyro = (imu_vec[i].gyro + imu_vec[i+1].gyro- (imu_vec[i+1].gyro-imu_vec[i].gyro)*(tend/tab))*0.5f;
                dt = t_e - imu_vec[i].t;
            }
            else if((i==0) && (i==(N-1)))
            {
                acc = imu_vec[i].acc;
                gyro = imu_vec[i].gyro;
                dt = t_e - t_s;
            }

            imu_from_last_frame->addImuPoint(acc, gyro, imu_from_last_frame->ba_, imu_from_last_frame->bg_, dt); 
            imu_from_last_keyframe_->addImuPoint(acc, gyro, imu_from_last_keyframe_->ba_, imu_from_last_keyframe_->bg_, dt); 
        }

        new_frame_->imu_from_last_frame_ = imu_from_last_frame;
        new_frame_->imu_from_last_keyframe_ = imu_from_last_keyframe_;

    }

    if(1 == type)
    {
        vilo::IMU* imu_from_last_lidar = NULL;
        if(last_lidar_)
            imu_from_last_lidar = new vilo::IMU(imu_freq_,
                                            imu_na_, imu_ng_,
                                            imu_naw_, imu_ngw_,
                                            last_lidar_->ba_, last_lidar_->bg_);
        else
            imu_from_last_lidar = new vilo::IMU(imu_freq_,
                                            imu_na_, imu_ng_,
                                            imu_naw_, imu_ngw_);

        double t_s = last_lidar_->getTimeStamp();
        double t_e = new_lidar_->getTimeStamp();
        
        int N = imu_vec.size()-1;
        for(int i=0;i<N;i++)
        {
            Eigen::Vector3d acc, gyro;
            double dt;
            if((i==0) && (i<(N-1)))
            {
                double tab = imu_vec[i+1].t-imu_vec[i].t;
                double tini = imu_vec[i].t-t_s;
                acc = (imu_vec[i].acc + imu_vec[i+1].acc - (imu_vec[i+1].acc-imu_vec[i].acc)*(tini/tab))  *0.5f;
                gyro = (imu_vec[i].gyro + imu_vec[i+1].gyro - (imu_vec[i+1].gyro-imu_vec[i].gyro)*(tini/tab))*0.5f;
                dt = imu_vec[i+1].t-t_s;
            }
            else if(i<(N-1))
            {
                acc = (imu_vec[i].acc + imu_vec[i+1].acc)*0.5f;
                gyro = (imu_vec[i].gyro + imu_vec[i+1].gyro)*0.5f;
                dt = imu_vec[i+1].t-imu_vec[i].t;
            }
            else if((i>0) && (i==(N-1)))
            {
                float tab = imu_vec[i+1].t-imu_vec[i].t;
                float tend = imu_vec[i+1].t-t_e;
                acc = (imu_vec[i].acc + imu_vec[i+1].acc - (imu_vec[i+1].acc-imu_vec[i].acc)*(tend/tab))*0.5f;
                gyro = (imu_vec[i].gyro + imu_vec[i+1].gyro- (imu_vec[i+1].gyro-imu_vec[i].gyro)*(tend/tab))*0.5f;
                dt = t_e - imu_vec[i].t;
            }
            else if((i==0) && (i==(N-1)))
            {
                acc = imu_vec[i].acc;
                gyro = imu_vec[i].gyro;
                dt = t_e - t_s;
            }

            imu_from_last_lidar->addImuPoint(acc, gyro, imu_from_last_lidar->ba_, imu_from_last_lidar->bg_, dt);
            imu_from_last_keylidar_->addImuPoint(acc, gyro, imu_from_last_keylidar_->ba_, imu_from_last_keylidar_->bg_, dt);
        }

        new_lidar_->imu_from_last_lidar_ = imu_from_last_lidar;
        new_lidar_->last_key_lidar_ = last_key_lidar_;
        new_lidar_->imu_from_last_keylidar_ = imu_from_last_keylidar_;

        Eigen::AngleAxisd aa(imu_from_last_keylidar_->delta_q_);
        double angle = aa.angle() / 3.14 * 180;

        if(new_lidar_->id_ % step_==0 || angle>=10)
        {
            new_lidar_->is_keylidar_ = true;

            if(is_lio_initialized_)
                imu_from_last_keylidar_ = new vilo::IMU(imu_freq_,
                                                        imu_na_, imu_ng_,
                                                        imu_naw_, imu_ngw_,
                                                        last_lidar_->ba_,
                                                        last_lidar_->bg_);
            else
                imu_from_last_keylidar_ = new vilo::IMU(imu_freq_,
                                                        imu_na_, imu_ng_,
                                                        imu_naw_, imu_ngw_);

        }

    }

}

void FrameHandlerMono::trackNewLidarFrame(LidarPtr& new_lidar_, LidarPtr& last_lidar_, Sophus::SE3& Trc)
{
    struct timeval st,et;
    gettimeofday(&st,NULL);

    const float DISTANCE_SQ_THRESHOLD = 5;
    const float NEARBY_SCAN = 2.5;
    bool DISTORTION = false;
    float SCAN_PERIOD = 0.1;
    const int N_ITER = 2;

    const int Horizon_SCAN = new_lidar_->Horizon_SCAN_;
    const int N_SCAN = new_lidar_->N_SCAN_;
    const float Vertical_Range = new_lidar_->Vertical_Range_;
    const float Vertical_Bottom = new_lidar_->Vertical_Bottom_;
    const float ang_res_x = 360.0 / Horizon_SCAN;
    const float ang_res_y = Vertical_Range / (N_SCAN-1);

    pcl::PointXYZI p_w;
    std::vector<int> nbr_id_vec;
    std::vector<float> nbr_dist_vec;

    const int N_surf = new_lidar_->ptr_surfs_->points.size();
    const int N_line = new_lidar_->ptr_corners_->points.size();

    for(int n_it=0; n_it<N_ITER; n_it++)
    {
        Eigen::Quaterniond q_last_cur = Trc.unit_quaternion();
        Eigen::Vector3d t_last_cur = Trc.translation();

        std::vector<Eigen::Vector3d> ps, nms;
        std::vector<double> ds;

        int n_close = 0; int n0=0, n1=0, n2=0,n3=0, n4=0; int nm3=0;
        for (int i = 0; i < N_surf; ++i)
        {
            pcl::PointXYZI surf_ftr = new_lidar_->ptr_surfs_->points[i];
            new_lidar_->transformToStart(surf_ftr, p_w , SCAN_PERIOD, q_last_cur, t_last_cur);

            float ptx = p_w.x;
            float pty = p_w.y;
            float ptz = p_w.z;

            float range = sqrt(ptx*ptx + pty*pty);
            float v_angle = atan2(ptz, range) * 180 / M_PI;

            int n_row = round(N_SCAN - 1 - (v_angle - Vertical_Bottom) / ang_res_y);
            int n_col = round(0.5 * Horizon_SCAN - atan2(pty, ptx)/ ang_res_x* 180 / M_PI);

            if(n_row<0 || n_row>N_SCAN-1 || n_col<0 || n_col>Horizon_SCAN-1)
                continue;

            int id = -1;
            if(0 == last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col)[3])
            {
                int close_id1 = -1, close_id2 = -1;
                for(int k=0;k<3;k++) 
                {

                    if( n_col+k > last_lidar_->id_img_.cols-1)
                        continue;
                    if(0 != last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col + k)[3])
                    {
                        close_id1 = last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col + k)[3];
                        break;
                    }
                }
                for(int k=0;k<3;k++)
                {
                    if(n_col-k < 0)
                        continue;
                    if(0 != last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col - k)[3])
                    {
                        close_id2 = last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col - k)[3];
                        break;
                    }
                }
                
                if(-1 == close_id1 && -1 == close_id2)
                {
                    n0++;
                    continue;
                }
                else if(-1 == close_id1 && -1 != close_id2)
                {
                    id = close_id2;
                    n1++;
                }
                else if(-1 != close_id1 && -1 == close_id2)
                {
                    n2++;
                    id = close_id1;
                }
                else
                {
                    n3++;
                    id = close_id2;
                }
            }
            else
            {
                n4++;
                id = last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col)[3];
            }

            float ptx1 = last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col)[0];
            float pty1 = last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col)[1];
            float ptz1 = last_lidar_->id_img_.at<cv::Vec4f>(n_row, n_col)[2];

            float dist = sqrt( (ptx1-ptx)*(ptx1-ptx) + (pty1-pty)*(pty1-pty) + (ptz1-ptz)*(ptz1-ptz) );

            n_close++;
            int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
            if (dist < DISTANCE_SQ_THRESHOLD)
            {
                closestPointInd = id;
                int closestPointScanID = int(last_lidar_->ptr_less_surfs_->points[closestPointInd].intensity);
                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                for (int j = closestPointInd + 1; j < (int)last_lidar_->ptr_less_surfs_->points.size(); ++j)
                {
                    if (int(last_lidar_->ptr_less_surfs_->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                        break;

                    double pointSqDis = (last_lidar_->ptr_less_surfs_->points[j].x - p_w.x) *
                                            (last_lidar_->ptr_less_surfs_->points[j].x - p_w.x) +
                                        (last_lidar_->ptr_less_surfs_->points[j].y - p_w.y) *
                                            (last_lidar_->ptr_less_surfs_->points[j].y - p_w.y) +
                                        (last_lidar_->ptr_less_surfs_->points[j].z - p_w.z) *
                                            (last_lidar_->ptr_less_surfs_->points[j].z - p_w.z);

                    if (int(last_lidar_->ptr_less_surfs_->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                    else if (int(last_lidar_->ptr_less_surfs_->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                    {
                        minPointSqDis3 = pointSqDis;
                        minPointInd3 = j;
                    }
                }

                for (int j = closestPointInd - 1; j >= 0; --j)
                {
                    if (int(last_lidar_->ptr_less_surfs_->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                        break;

                    double pointSqDis = (last_lidar_->ptr_less_surfs_->points[j].x - p_w.x) *
                                            (last_lidar_->ptr_less_surfs_->points[j].x - p_w.x) +
                                        (last_lidar_->ptr_less_surfs_->points[j].y - p_w.y) *
                                            (last_lidar_->ptr_less_surfs_->points[j].y - p_w.y) +
                                        (last_lidar_->ptr_less_surfs_->points[j].z - p_w.z) *
                                            (last_lidar_->ptr_less_surfs_->points[j].z - p_w.z);

                    if (int(last_lidar_->ptr_less_surfs_->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                    else if (int(last_lidar_->ptr_less_surfs_->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                    {
                        minPointSqDis3 = pointSqDis;
                        minPointInd3 = j;
                    }
                }

                if (minPointInd2 >= 0 && minPointInd3 >= 0)
                {

                    Eigen::Vector3d curr_point(new_lidar_->ptr_surfs_->points[i].x,
                                                new_lidar_->ptr_surfs_->points[i].y,
                                                new_lidar_->ptr_surfs_->points[i].z);
                    Eigen::Vector3d last_point_a(last_lidar_->ptr_less_surfs_->points[closestPointInd].x,
                                                    last_lidar_->ptr_less_surfs_->points[closestPointInd].y,
                                                    last_lidar_->ptr_less_surfs_->points[closestPointInd].z);
                    Eigen::Vector3d last_point_b(last_lidar_->ptr_less_surfs_->points[minPointInd2].x,
                                                    last_lidar_->ptr_less_surfs_->points[minPointInd2].y,
                                                    last_lidar_->ptr_less_surfs_->points[minPointInd2].z);
                    Eigen::Vector3d last_point_c(last_lidar_->ptr_less_surfs_->points[minPointInd3].x,
                                                    last_lidar_->ptr_less_surfs_->points[minPointInd3].y,
                                                    last_lidar_->ptr_less_surfs_->points[minPointInd3].z);

                    Eigen::Vector3d nm = (last_point_a - last_point_b).cross(last_point_a - last_point_c);
                    nm = nm/nm.norm();
                    double d = - last_point_a.dot(nm);
                    ps.push_back(curr_point);
                    nms.push_back(nm);
                    ds.push_back(d);
                }
            }
        }

        std::vector<Eigen::Vector3d> ps1, lines1, lines2;
        for (int i = 0; i < N_line; ++i)
        {
            pcl::PointXYZI line_ftr = new_lidar_->ptr_corners_->points[i];
            new_lidar_->transformToStart(line_ftr, p_w , SCAN_PERIOD, q_last_cur, t_last_cur);
            last_lidar_->kdtree_corner_->nearestKSearch(p_w, 1, nbr_id_vec, nbr_dist_vec);

            int closestPointInd = -1, minPointInd2 = -1;
            if (nbr_dist_vec[0] < DISTANCE_SQ_THRESHOLD)
            {
                closestPointInd = nbr_id_vec[0];
                int closestPointScanID = int(last_lidar_->ptr_less_corners_->points[closestPointInd].intensity);
                double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                
                for (int j = closestPointInd + 1; j < (int)last_lidar_->ptr_less_corners_->points.size(); ++j)
                {
                    if (int(last_lidar_->ptr_less_corners_->points[j].intensity) <= closestPointScanID)
                        continue;
                        
                    if (int(last_lidar_->ptr_less_corners_->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                        break;

                    double pointSqDis = (last_lidar_->ptr_less_corners_->points[j].x - p_w.x) *
                                            (last_lidar_->ptr_less_corners_->points[j].x - p_w.x) +
                                        (last_lidar_->ptr_less_corners_->points[j].y - p_w.y) *
                                            (last_lidar_->ptr_less_corners_->points[j].y - p_w.y) +
                                        (last_lidar_->ptr_less_corners_->points[j].z - p_w.z) *
                                            (last_lidar_->ptr_less_corners_->points[j].z - p_w.z);

                    if (pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                }

                for (int j = closestPointInd - 1; j >= 0; --j)
                {
                    if (int(last_lidar_->ptr_less_corners_->points[j].intensity) >= closestPointScanID)
                        continue;

                    if (int(last_lidar_->ptr_less_corners_->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                        break;

                    double pointSqDis = (last_lidar_->ptr_less_corners_->points[j].x - p_w.x) *
                                            (last_lidar_->ptr_less_corners_->points[j].x - p_w.x) +
                                        (last_lidar_->ptr_less_corners_->points[j].y - p_w.y) *
                                            (last_lidar_->ptr_less_corners_->points[j].y - p_w.y) +
                                        (last_lidar_->ptr_less_corners_->points[j].z - p_w.z) *
                                            (last_lidar_->ptr_less_corners_->points[j].z - p_w.z);

                    if (pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }

                }

            }
        }

        ba::lidarOnePoseOptimization2(Trc, ds, ps, nms, lines1, lines2, ps1);
    }

    lidar_motion_model_ = Trc;

    pcl::PointCloud<pcl::PointXYZI>::Ptr sparse_surf(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> filter;
    filter.setLeafSize(0.8, 0.8, 0.8); 
    filter.setInputCloud(last_lidar_->ptr_less_surfs_);
    filter.filter(*sparse_surf);
    last_lidar_->ptr_less_surfs_ = sparse_surf;

    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

void FrameHandlerMono::predictWithImu()
{
    assert(last_lidar_->id_ == new_lidar_->previous_lidar_->id_);
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
    Sophus::SE3 Tlw_cur = (Twb2 * T_b_l_).inverse();
    new_lidar_->setLidarPose(Tlw_cur);
    new_lidar_->setVelocity(v2);
}

} // namespace vilo
