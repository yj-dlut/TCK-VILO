#include <vilo/config.h>
#include <vilo/frame.h>
#include <vilo/point.h>
#include <vilo/feature.h>
#include <vilo/initialization.h>
#include <vilo/feature_detection.h>

#include "vilo/vikit/homography.h"
#include "vilo/vikit/math_utils.h"

namespace vilo {
namespace initialization {

InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
    reset();

    detectFeatures(frame_ref, px_ref_, f_ref_, ftr_type_);

    if(px_ref_.size() < 200)
    {
        VILO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
        return FAILURE;
    }

    VILO_INFO_STREAM("Init: First Frame: "<< f_ref_.size() <<" features");
    frame_ref_ = frame_ref;
    px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
    img_prev_ = frame_ref_->img_pyr_[0].clone();
    px_prev_ = px_ref_;

    return SUCCESS;
}

InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
    trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_, img_prev_, px_prev_, ftr_type_);
  
    VILO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

    if(disparities_.size() < Config::initMinTracked())
        return FAILURE;

    double disparity = vilo::getMedian(disparities_);
    VILO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
    if(disparity < Config::initMinDisparity())
        return NO_KEYFRAME;


    computeInitializeMatrix(
        f_ref_, f_cur_, frame_ref_->cam_->errorMultiplier2(), 
        Config::poseOptimThresh(), inliers_, xyz_in_cur_, T_cur_from_ref_);

    if(inliers_.size() < Config::initMinInliers())
    {
        VILO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
        return FAILURE;
    }

    vector<double> depth_vec;
    for(size_t i=0; i<xyz_in_cur_.size(); ++i)
        depth_vec.push_back((xyz_in_cur_[i]).z());
    double scene_depth_median = vilo::getMedian(depth_vec);
    double scale = Config::mapScale()/scene_depth_median; 

    Sophus::SE3 Tfw_ref = frame_ref_->getFramePose();
    Sophus::SE3 Tfw_cur = T_cur_from_ref_ * Tfw_ref;
    Eigen::Vector3d ref_pos = Tfw_ref.inverse().translation();
    Eigen::Vector3d cur_pos = Tfw_cur.inverse().translation();
    Tfw_cur.translation() = -Tfw_cur.rotation_matrix()*(ref_pos + scale*(cur_pos - ref_pos));
    frame_cur->setFramePose(Tfw_cur);
    
    SE3 T_world_cur = Tfw_cur.inverse();

    for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
    {
        Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
        Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
        Vector3d fts_type(ftr_type_[*it][0], ftr_type_[*it][1], ftr_type_[*it][2]);

        if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
        {
            Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
            Point* new_point = new Point(pos);
            new_point->idist_ = 1.0/pos.norm();

            if(fts_type[2] == 0)
            {
                new_point->ftr_type_ = Point::FEATURE_CORNER;
                Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
                frame_cur->addFeature(ftr_cur);

                Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
                frame_ref_->addFeature(ftr_ref);

                new_point->addFrameRef(ftr_ref);
                new_point->hostFeature_ = ftr_ref;
            }
            else if(fts_type[2] == 1)
            {
                new_point->ftr_type_ = Point::FEATURE_EDGELET;

                Vector2d grad(fts_type[0], fts_type[1]);
                Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], grad, 0));
                frame_cur->addFeature(ftr_cur);
                
                Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], grad, 0));
                frame_ref_->addFeature(ftr_ref);

                new_point->addFrameRef(ftr_ref);
                new_point->hostFeature_ = ftr_ref;
            }
            else
            {
                new_point->ftr_type_ = Point::FEATURE_GRADIENT;

                Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, 0, Feature::GRADIENT));
                frame_cur->addFeature(ftr_cur);
                
                Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, 0, Feature::GRADIENT));
                frame_ref_->addFeature(ftr_ref);

                new_point->addFrameRef(ftr_ref);
                new_point->hostFeature_ = ftr_ref;
            }
        }
    }

    return SUCCESS;
}

void KltHomographyInit::reset()
{
    px_cur_.clear();
    frame_ref_.reset();
}

void detectFeatures(FramePtr frame, vector<cv::Point2f>& px_vec, vector<Vector3d>& f_vec, vector<Vector3d>& ftr_type)
{
    Features new_features;

    feature_detection::FeatureExtractor* featureExt(
        new feature_detection::FeatureExtractor(frame->img().cols, frame->img().rows, 20, 1, true));
    featureExt->detect(frame.get(), 20, frame->gradMean_ + 0.5f, new_features);

    px_vec.clear(); px_vec.reserve(new_features.size());
    f_vec.clear(); f_vec.reserve(new_features.size());
    Vector3d fts_type_temp;

    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr)
    {
        if(ftr->type == Feature::EDGELET)
        {
            fts_type_temp[0] = ftr->grad[0];
            fts_type_temp[1] = ftr->grad[1];
            fts_type_temp[2] = 1;
        }
        else if(ftr->type == Feature::CORNER)
        {
            fts_type_temp[0] = ftr->grad[0];
            fts_type_temp[1] = ftr->grad[1];
            fts_type_temp[2] = 0;
        }
        else
        {
            fts_type_temp[0] = ftr->grad[0];
            fts_type_temp[1] = ftr->grad[1];
            fts_type_temp[2] = 2;
        }

        ftr_type.push_back(fts_type_temp);
        px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
        f_vec.push_back(ftr->f);

        delete ftr;
    });

    delete featureExt;
}


void trackKlt(FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities,
    cv::Mat& img_prev,
    vector<cv::Point2f>& px_prev,
    vector<Vector3d>& fts_type)
{
    const double klt_win_size = 30.0;
    const int klt_max_iter = 30;
    const double klt_eps = 0.0001;
    vector<unsigned char> status;
    vector<float> error;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
    cv::calcOpticalFlowPyrLK(
        img_prev, frame_cur->img_pyr_[0], px_prev, px_cur, status, error,
        cv::Size2i(klt_win_size, klt_win_size), 4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
    vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
    vector<cv::Point2f>::iterator px_pre_it = px_prev.begin();

    vector<Vector3d>::iterator f_ref_it = f_ref.begin();
    vector<Vector3d>::iterator fts_type_it = fts_type.begin();

    f_cur.clear(); f_cur.reserve(px_cur.size());
    disparities.clear(); disparities.reserve(px_cur.size());

    for(size_t i=0; px_ref_it != px_ref.end(); ++i)
    {
        if(!status[i] || !patchCheck(img_prev, frame_cur->img_pyr_[0], *px_pre_it, *px_cur_it))
        {
            px_ref_it = px_ref.erase(px_ref_it);
            px_cur_it = px_cur.erase(px_cur_it);
            f_ref_it = f_ref.erase(f_ref_it);
            fts_type_it = fts_type.erase(fts_type_it);

            px_pre_it = px_prev.erase(px_pre_it);

            continue;
        }

        f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
        disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());

        ++px_ref_it;
        ++px_cur_it;
        ++f_ref_it;
        ++fts_type_it;

        ++px_pre_it;
    }
    
    img_prev = frame_cur->img_pyr_[0].clone();
    px_prev = px_cur;
}

void computeInitializeMatrix(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
    vector<cv::Point2f> x1(f_ref.size()), x2(f_cur.size());
    for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
    {
        x1[i] = cv::Point2f(f_ref[i][0] / f_ref[i][2], f_ref[i][1] / f_ref[i][2]);
        x2[i] = cv::Point2f(f_cur[i][0] / f_cur[i][2], f_cur[i][1] / f_cur[i][2]);
    }
    const cv::Point2d pp(0,0);
    const double focal = 1.0;
    cv::Mat E = findEssentialMat(x1, x2, focal, pp, CV_RANSAC, 0.99, 2.0/focal_length, cv::noArray() );

    cv::Mat R_cf,t_cf;
    cv::recoverPose(E, x1, x2, R_cf, t_cf, focal, pp);
    Vector3d t;
    Matrix3d R;
    R <<    R_cf.at<double>(0,0), R_cf.at<double>(0,1), R_cf.at<double>(0,2),
            R_cf.at<double>(1,0), R_cf.at<double>(1,1), R_cf.at<double>(1,2),
            R_cf.at<double>(2,0), R_cf.at<double>(2,1), R_cf.at<double>(2,2);
    t << t_cf.at<double>(0), t_cf.at<double>(1), t_cf.at<double>(2);

    vector<int> outliers_E, inliers_E;
    vector<Vector3d> xyz_E;
    
    double E_error = computeP3D(
        f_cur, f_ref, R, t, reprojection_threshold, focal_length, xyz_E, inliers_E);
    SE3 T_E = SE3(R,t);


    vector<Vector2d> uv_ref(f_ref.size());
    vector<Vector2d> uv_cur(f_cur.size());
    for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
    {
        uv_ref[i] = vilo::project2d(f_ref[i]);
        uv_cur[i] = vilo::project2d(f_cur[i]);
    }
    vilo::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
    Homography.computeSE3fromMatches();

    vector<int> outliers_H, inliers_H;
    vector<Vector3d> xyz_H;
    
    double H_error = computeP3D(
        f_cur, f_ref, Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
        reprojection_threshold, focal_length, xyz_H, inliers_H);


    VILO_INFO_STREAM("Init: H error =  " << H_error); 
    VILO_INFO_STREAM("Init: E error =  " << E_error); 

    if(H_error < E_error)
    {
        inliers = inliers_H;
        xyz_in_cur = xyz_H;
        T_cur_from_ref = Homography.T_c2_from_c1;

        VILO_INFO_STREAM("Init: Homography RANSAC "<<inliers.size()<<" inliers.");
    }
    else
    {
        inliers = inliers_E;
        xyz_in_cur = xyz_E;
        T_cur_from_ref = T_E;

        VILO_INFO_STREAM("Init: Essential RANSAC "<<inliers.size()<<" inliers.");
    }
}

double computeP3D(
    const vector<Vector3d>& vBearing1, 
    const vector<Vector3d>& vBearing2,  
    const Matrix3d& R,
    const Vector3d& t,
    const double reproj_thresh,
    double error_multiplier2,
    vector<Vector3d>& vP3D,
    vector<int>& inliers)
{
    inliers.clear(); inliers.reserve(vBearing1.size());
    vP3D.clear(); vP3D.reserve(vBearing1.size());

    SE3 T_c_r = SE3(R,t);
    SE3 T_r_c = T_c_r.inverse();
    double totalEnergy = 0;
    for(size_t i = 0; i < vBearing1.size(); ++i)
    {
        Vector3d p3d_cur_old(vilo::triangulateFeatureNonLin(R, t, vBearing1[i], vBearing2[i]));
        Vector3d p3d_ref_old(T_r_c*p3d_cur_old);
        Vector3d pWorld_new(distancePointOnce(p3d_ref_old, vBearing2[i], vBearing1[i], T_c_r));
        Vector3d pTarget_new(T_c_r*pWorld_new);

        double e1 = vilo::reprojError(vBearing1[i], pTarget_new, error_multiplier2);
    
        totalEnergy += e1;
        vP3D.push_back(pTarget_new);

        if(pWorld_new[2] < 0.01 || pTarget_new[2] < 0.01) continue;    
        float ratio = p3d_ref_old.norm()/pWorld_new.norm();
        if(ratio < 0.9 || ratio > 1.1) continue;
        
        if(e1 < reproj_thresh)
            inliers.push_back(i);
    }

    return totalEnergy;
}


Vector3d distancePointOnce(
    const Vector3d pointW, Vector3d bearingRef, Vector3d bearingCur, SE3 T_c_r)
{
    double idist_old = 1./pointW.norm();
    double idist_new = idist_old;
    Vector3d pHost(bearingRef*(1.0/idist_old));

    double oldEnergy = 0;
    double H=0,b=0;
    for(int iter = 0; iter < 3; ++iter)
    {
        double newEnergy = 0;
        H=0; b=0;

        Vector3d pTarget(T_c_r*pHost);
        Vector2d e(vilo::project2d(bearingCur) - vilo::project2d(pTarget));
        newEnergy += e.squaredNorm();

        Vector2d Juvdd;
        Point::jacobian_id2uv(pTarget, T_c_r, idist_new, bearingRef, Juvdd);
        H += Juvdd.transpose()*Juvdd;
        b -= Juvdd.transpose()*e;

        double step = (1.0/H)*b;
        if((iter > 0 && newEnergy > oldEnergy) || (bool)std::isnan(step))
        {
            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "it " << iter << "\t FAILURE \t new_chi2 = " << newEnergy << endl;
            #endif

            idist_new = idist_old; 
            break;
        }

        idist_old = idist_new;
        idist_new += step;
        oldEnergy = newEnergy;
        pHost = Vector3d(bearingRef*(1.0/idist_new));


        #ifdef POINT_OPTIMIZER_DEBUG
            cout << "it " << iter
            << "\t Success \t new_chi2 = " << newEnergy
            << "\t idist step = " << step
            << endl;
        #endif

        if(step <= 0.000001*idist_new) break;
    }

    return Vector3d(bearingRef*(1.0/idist_new));
}

bool patchCheck(
    const cv::Mat& imgPre, const cv::Mat& imgCur, const cv::Point2f& pxPre, const cv::Point2f& pxCur)
{
    const int patchArea = 64;

    float patch_pre[patchArea], patch_cur[patchArea];
    if(!createPatch(imgPre, pxPre, patch_pre) || !createPatch(imgCur, pxCur, patch_cur)) return false;

    return checkSSD(patch_pre, patch_cur);
}   

bool createPatch(const cv::Mat& img, const cv::Point2f& px, float* patch)
{
    const int halfPatchSize = 4;
    const int patchSize = halfPatchSize*2;
    const int stride = img.cols;

    float u = px.x;
    float v = px.y;
    int ui = floorf(u);
    int vi = floorf(v);

    if(ui < halfPatchSize || ui >= img.cols-halfPatchSize || vi < halfPatchSize || vi >= img.rows-halfPatchSize)
        return false;

    float subpix_u = u - ui;
    float subpix_v = v - vi;
    float w_ref_tl = (1.0-subpix_u) * (1.0-subpix_v);
    float w_ref_tr = subpix_u * (1.0-subpix_v);
    float w_ref_bl = (1.0-subpix_u) * subpix_v;
    float w_ref_br = subpix_u * subpix_v; 

    float* patch_ptr = patch;
    for(int y = 0; y < patchSize; ++y)
    {
        uint8_t* cur_patch_ptr = img.data + (vi - halfPatchSize + y) * stride + (ui - halfPatchSize);
        for(int x = 0; x < patchSize; ++x, ++patch_ptr, ++cur_patch_ptr)
            *patch_ptr = w_ref_tl*cur_patch_ptr[0]      + w_ref_tr*cur_patch_ptr[1] + 
                         w_ref_bl*cur_patch_ptr[stride] + w_ref_br*cur_patch_ptr[stride+1];
    }

    return true;
}

bool checkSSD(float* patch1, float* patch2)
{
    const int patchArea = 64;
    const float threshold = 0.8f;

    float mean1 = 0, mean2 = 0;
    for(int i = 0; i < patchArea; ++i)
    {
        mean1 += patch1[i];
        mean2 += patch2[i];
    }

    mean1 /= patchArea;
    mean2 /= patchArea;

    float numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patchArea; i++)
    {
        numerator   += (patch1[i]-mean1)*(patch2[i]-mean2);
        demoniator1 += (patch1[i]-mean1)*(patch1[i]-mean1);
        demoniator2 += (patch2[i]-mean2)*(patch2[i]-mean2);
    }

    return numerator / (sqrt(demoniator1*demoniator2) + 1e-12) > threshold;
}

} // namespace initialization
} // namespace vilo
