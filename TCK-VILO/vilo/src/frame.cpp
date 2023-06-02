#include <stdexcept>
#include <vilo/frame.h>
#include <vilo/feature.h>
#include <vilo/point.h>
#include <vilo/config.h>
#include <boost/bind.hpp>
#include <fast/fast.h>

#include "vilo/PhotomatricCalibration.h"
#include "vilo/vikit/math_utils.h"
#include "vilo/vikit/vision.h"

using namespace cv;

namespace vilo {

int Frame::frame_counter_ = 0;
int Frame::keyFrameCounter_ = 0;


Frame::Frame(vilo::AbstractCamera* cam, const cv::Mat& img, double timestamp, PhotomatricCalibration* opc) :
             id_(frame_counter_++), timestamp_(timestamp), 
             cam_(cam), key_pts_(5), is_keyframe_(false), v_kf_(NULL), gradMean_(0)
{
    if(opc != NULL) m_pc = opc;

    initFrame(img);
}

Frame::Frame(vilo::AbstractCamera* cam, const cv::Mat& img, double timestamp, SE3& Tbc, PhotomatricCalibration* opc) :
             id_(frame_counter_++), timestamp_(timestamp), 
             cam_(cam), key_pts_(5), is_keyframe_(false), v_kf_(NULL), gradMean_(0)
{
    ba_ << 0.0, 0.0, 0.0;
    bg_ << 0.0, 0.0, 0.0;    
    T_b_c_ = Tbc;
    T_c_b_ = T_b_c_.inverse();
    lba_id_ = -1;
    align_lid_ = -1;
    last_kf_ = NULL;
    is_opt_imu_ = false;
    prior_constraint_ = NULL;

    if(opc != NULL) m_pc = opc;

    initFrame(img);
}

Frame::~Frame()
{
    int nn=0;
    std::for_each(fts_.begin(), fts_.end(), [&](Feature* i)
    {
        if(i->m_prev_feature != NULL)
        {
            assert(i->m_prev_feature->frame->isKeyframe());
            i->m_prev_feature->m_next_feature = NULL;
            i->m_prev_feature = NULL;
        }

        if(i->m_next_feature != NULL)
        {
            i->m_next_feature->m_prev_feature = NULL;
            i->m_next_feature = NULL;
        }

        nn++;
        delete i; i=NULL;
    });

    img_pyr_.clear();
    grad_pyr_.clear();
    sobelX_.clear();
    sobelY_.clear();
    canny_.clear();
    m_pyr_raw.clear();
}

void Frame::initFrame(const cv::Mat& img)
{
    if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
        throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

    std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature* ftr){ ftr=NULL; });

    if(m_pc == NULL)
        frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
    else
        photometricallyCorrectPyramid(img, img_pyr_, m_pyr_raw, max(Config::nPyrLevels(), Config::kltMaxLevel()+1));

    prepareForFeatureDetect();    
}

void Frame::setKeyframe()
{
    is_keyframe_ = true;
    setKeyPoints();

    keyFrameCounter_++;
    keyFrameId_ = keyFrameCounter_;
}

void Frame::addFeature(Feature* ftr)
{
    boost::unique_lock<boost::mutex> lock(fts_mutex_);
    fts_.push_back(ftr);
}

void Frame::getFeaturesCopy(Features& list_copy)
{
    boost::unique_lock<boost::mutex> lock(fts_mutex_);
    for(auto it = fts_.begin(); it != fts_.end(); ++it)
        list_copy.push_back(*it);
}

void Frame::setKeyPoints()
{
    {
        boost::unique_lock<boost::mutex> lock(keypoints_mutex_);
        for(size_t i = 0; i < 5; ++i)
            if(key_pts_[i] != NULL)
            if(key_pts_[i]->point == NULL)
                key_pts_[i] = NULL;
    }
    boost::unique_lock<boost::mutex> lock(fts_mutex_);
    std::for_each(fts_.begin(), fts_.end(), [&](Feature* ftr){ if(ftr->point != NULL) checkKeyPoints(ftr); });
}

void Frame::checkKeyPoints(Feature* ftr)
{
    boost::unique_lock<boost::mutex> lock(keypoints_mutex_);
    const int cu = cam_->width()/2;
    const int cv = cam_->height()/2;
    const Vector2d uv = ftr->px;

    if(key_pts_[0] == NULL)
        key_pts_[0] = ftr;
    else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))  
            < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
        key_pts_[0] = ftr;
        
    if(uv[0] >= cu && uv[1] >= cv)
    {
        if(key_pts_[1] == NULL)
        key_pts_[1] = ftr;
        else if((uv[0] - cu) * (uv[1] - cv)
            >(key_pts_[1]->px[0] - cu) * (key_pts_[1]->px[1] - cv))
        key_pts_[1] = ftr;
    }
    
    if(uv[0] >= cu && uv[1] < cv)
    {
        if(key_pts_[2] == NULL)
        key_pts_[2] = ftr;
        else if((uv[0] - cu) * -(uv[1] - cv)
            >(key_pts_[2]->px[0] - cu) * -(key_pts_[2]->px[1] - cv))
        key_pts_[2] = ftr;
    }
    
    if(uv[0] < cu && uv[1] >= cv)
    {
        if(key_pts_[3] == NULL)
        key_pts_[3] = ftr;
        else if(-(uv[0] - cu) * (uv[1] - cv)
            >-(key_pts_[3]->px[0] - cu) * (key_pts_[3]->px[1] - cv))
        key_pts_[3] = ftr;
    }
    
    if(uv[0] < cu && uv[1] < cv)
    {
        if(key_pts_[4] == NULL)
        key_pts_[4] = ftr;
        else if(-(uv[0] - cu) * -(uv[1] - cv)
            >-(key_pts_[4]->px[0] - cu) * -(key_pts_[4]->px[1] - cv))
        key_pts_[4] = ftr;
    }
}

void Frame::removeKeyPoint(Feature* ftr)
{
    bool found = false;
    {
        boost::unique_lock<boost::mutex> lock(keypoints_mutex_);
        std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature*& i){
        if(i == ftr) {
            i = NULL;
            found = true;
        }
        });
    }

    if(found) setKeyPoints();
}

bool Frame::isVisible(const Vector3d& xyz_w)
{
    Vector3d xyz_f;
    {
        boost::unique_lock<boost::mutex> lock(pose_mutex_);
        xyz_f = T_f_w_ * xyz_w;
    } 
    if(xyz_f.z() < 0.0) return false;

    Vector2d px = f2c(xyz_f);
    if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
        return true;

    return false;
}

void Frame::prepareForFeatureDetect()
{  
    sobelX_.resize(Config::nPyrLevels());
    sobelY_.resize(Config::nPyrLevels());

    assert(Config::nPyrLevels() == 3);

    for(int i = 0; i < 3; ++i)
    {
        cv::Sobel(img_pyr_[i], sobelX_[i], CV_16S, 1, 0, 5, 1, 0, BORDER_REPLICATE);
        cv::Sobel(img_pyr_[i], sobelY_[i], CV_16S, 0, 1, 5, 1, 0, BORDER_REPLICATE);
    }


    float intSum = 0, gradSum = 0;
    int sum = 0;
    for(int y=16;y<img_pyr_[0].rows-16;y++)
        for(int x=16;x<img_pyr_[0].cols-16;x++)
        {
            sum++;
            float gradx = sobelX_[0].at<short>(y,x);
            float grady = sobelY_[0].at<short>(y,x);
            gradSum += sqrtf(gradx*gradx + grady*grady);

            intSum += img_pyr_[0].ptr<uchar>(y)[x];

        }

    integralImage_ = intSum/sum;

    gradMean_ = gradSum/sum;
    gradMean_ /= 30;
    if(gradMean_ > 20) gradMean_ = 20;
    if(gradMean_ < 7)  gradMean_ = 7;
}

void Frame::photometricallyCorrectPyramid(const cv::Mat& img_level_0, ImgPyr& pyr_correct, ImgPyr& pyr_raw, int n_levels)
{
    pyr_correct.resize(n_levels);
    cv::Mat image_corrected_0 = img_level_0.clone();
    m_pc->photometricallyCorrectImage(image_corrected_0);
    pyr_correct[0] = image_corrected_0;
    for(size_t L=1; L<pyr_correct.size(); ++L)
    {
        if(img_level_0.cols % 16 == 0 && img_level_0.rows % 16 == 0)
        {
            pyr_correct[L] = cv::Mat(pyr_correct[L-1].rows/2, pyr_correct[L-1].cols/2, CV_8U);
            vilo::halfSample(pyr_correct[L-1], pyr_correct[L]);
        }
        else
        {
            float scale = 1.0/(1<<L);
            cv::Size sz(cvRound((float)img_level_0.cols*scale), cvRound((float)img_level_0.rows*scale));
            cv::resize(pyr_correct[L-1], pyr_correct[L], sz, 0, 0, cv::INTER_LINEAR);
        }
    }
    
    pyr_raw.resize(Config::nPyrLevels());
    pyr_raw[0] = img_level_0.clone();
    for(size_t L=1; L<pyr_raw.size(); ++L)
    {
        if(img_level_0.cols % 16 == 0 && img_level_0.rows % 16 == 0)
        {
            pyr_raw[L] = cv::Mat(pyr_raw[L-1].rows/2, pyr_raw[L-1].cols/2, CV_8U);
            vilo::halfSample(pyr_raw[L-1], pyr_raw[L]);
        }
        else
        {
            float scale = 1.0/(1<<L);
            cv::Size sz(cvRound((float)img_level_0.cols*scale), cvRound((float)img_level_0.rows*scale));
            cv::resize(pyr_raw[L-1], pyr_raw[L], sz, 0, 0, cv::INTER_LINEAR);
        }
    }
}

void Frame::finish()
{
    grad_pyr_.clear();
    canny_.clear();   
}

Sophus::SE3 Frame::getFramePose()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return T_f_w_;
}
    
void Frame::setFramePose(Sophus::SE3& Tfw)
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    T_f_w_ = Tfw;
}

Eigen::Vector3d Frame::getImuAccBias()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return ba_;
}

Eigen::Vector3d Frame::getImuGyroBias()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return bg_;
}

void Frame::setImuAccBias(const Eigen::Vector3d& ba)
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    ba_ = ba;
}

void Frame::setImuGyroBias(const Eigen::Vector3d& bg)
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    bg_ = bg;
}

Eigen::Matrix3d Frame::getImuRotation()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return T_f_w_.inverse().rotation_matrix() * T_c_b_.rotation_matrix();
    
}
    
Eigen::Vector3d Frame::getImuPosition()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return (T_f_w_.inverse() * T_c_b_).translation(); 
}

Sophus::SE3 Frame::getImuPose()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return T_f_w_.inverse() * T_c_b_;
}

void Frame::setVelocity(const Eigen::Vector3d& vel)
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    velocity_ = vel;
}

void Frame::setVelocity()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    if(m_last_frame)
        velocity_ = m_last_frame->velocity_;
    else
        velocity_ << 0.0, 0.0, 0.0;
        
    
}

Eigen::Vector3d Frame::getVelocity()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return velocity_;
}

void Frame::setNewBias(Eigen::Vector3d& ba, Eigen::Vector3d& bg)
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    ba_ = ba;
    bg_ = bg;
    if(imu_from_last_frame_)
        imu_from_last_frame_->setNewBias(ba, bg);
    if(imu_from_last_keyframe_)
        imu_from_last_keyframe_->setNewBias(ba, bg);
}

namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
    pyr.resize(n_levels);
    pyr[0] = img_level_0;
    for(int i=1; i<n_levels; ++i)
    {
        if(img_level_0.cols % 16 == 0 && img_level_0.rows % 16 == 0)
        {
            pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
            vilo::halfSample(pyr[i-1], pyr[i]);
        }
        else
        {
            float scale = 1.0/(1<<i);
            cv::Size sz(cvRound((float)img_level_0.cols*scale), cvRound((float)img_level_0.rows*scale));
            cv::resize(pyr[i-1], pyr[i], sz, 0, 0, cv::INTER_LINEAR);
        }     
    }
}

void createImgGrad(const ImgPyr& pyr_img, ImgPyr& scharr, int n_levels)
{ 
    scharr.resize(n_levels);
    for(int i = 0; i < n_levels; ++i)
        vilo::calcSharrDeriv(pyr_img[i], scharr[i]);
}

bool getSceneDepth(Frame& frame, double& depth_mean, double& depth_min)
{
    vector<double> depth_vec;
    depth_vec.reserve(frame.fts_.size());
    depth_min = std::numeric_limits<double>::max();
    for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it)
    {
        if((*it)->point != NULL)
        {
            Eigen::Vector3d p_w = (*it)->point->getPointPose();
            const double z = frame.w2f(p_w).z();
            depth_vec.push_back(z);
            depth_min = fmin(z, depth_min);
        }
    }
    if(depth_vec.empty())
    {
        VILO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
        return false;
    }

    depth_mean = vilo::getMedian(depth_vec);
    return true;
}

bool getSceneDistance(Frame& frame, double& distance_mean)
{
    vector<double> distance_vec;
    distance_vec.reserve(frame.fts_.size());
    for(auto& ft: frame.fts_)
    {
        if(ft->point == NULL) continue;
        Eigen::Vector3d p_w = ft->point->getPointPose();
        const double distance = frame.w2f(p_w).norm();
        distance_vec.push_back(distance);
    }
    if(distance_vec.empty())
    {
        VILO_WARN_STREAM("Cannot set scene distance. Frame has no point-observations!");
        return false;
    }
    
    distance_mean = vilo::getMedian(distance_vec);
    return true;
}

void createIntegralImage(const cv::Mat& image, float& integralImage)
{
    float sum = 0;
    int num = 0;
    int height = image.rows;
    int weight = image.cols;
    for(int y=8;y<height-8;y++)
        for(int x=8;x<weight-8;x++)
        {
            sum += image.ptr<uchar>(y)[x];
            num++; 
        }

    integralImage = sum/num;
}
} // namespace frame_utils
} // namespace vilo
