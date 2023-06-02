#include <ctime>
#include <sys/time.h>
#include <stdlib.h>

#include <eigen3/Eigen/Dense>
#include<opencv2/core/eigen.hpp>

#include <fstream>
#include <string>
#include <sstream>

#include "vilo/lidar.h"
#include "vilo/vikit/so3_functions.h"

using namespace std;
namespace vilo {

int Lidar::frame_counter_ = 0;
int Lidar::keyFrameCounter_ = 0;

Lidar::Lidar()
{

}

Lidar::Lidar(vilo::AbstractCamera* cam, int w, int h, pcl::PointCloud<pcl::PointXYZI>::Ptr p_cloud ):
                       id_(frame_counter_++),w_(w) ,h_(h) ,cam_(cam)
{
    N_ = 0;
    pcloud_ = std::move(p_cloud);
    oringin_lidar_img_map_ = cv::Mat::zeros(h_,w_, CV_32SC2);
    grid_w_num_=10, grid_h_num_=10;
    extrinsic_ = cam_->getLidarCamExtrinsic();
    is_vertical_ = false;
    z_max_ = -10000.0;
    z_min_ = 99999.0;

    setLidarParams();
}

Lidar::Lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr p_cloud ):
                       id_(frame_counter_++)
{
    N_ = 0;
    pcloud_ = std::move(p_cloud);

    imu_from_last_lidar_ = NULL;
    prior_constraint_ = NULL;
    imu_from_last_keylidar_ = NULL;
    is_keylidar_ = false;
    previous_lidar_ = NULL;
    align_vid_ = -1;
    ba_ = Eigen::Vector3d(0,0,0);
    bg_ = Eigen::Vector3d(0,0,0);

    is_undist_pts_ = true;
    T_f_w_ = Sophus::SE3(Eigen::Matrix3d::Identity(),Eigen::Vector3d(0,0,0));

    pcl::PointCloud<pcl::PointXYZI>::Ptr ptmp(new pcl::PointCloud<pcl::PointXYZI>);
    ptr_corners_ = ptmp;
    ptr_surfs_ = ptmp;
    ptr_less_corners_ = ptmp;
    ptr_less_surfs_ = ptmp;

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr treetmp(new pcl::KdTreeFLANN<pcl::PointXYZI>);
    kdtree_corner_ = treetmp;
    kdtree_surf_ = treetmp;

    lidar_type_ = 0;
    if(0 == lidar_type_)
    {
        N_SCAN_ = 64;
        Horizon_SCAN_ = 1024;
        Vertical_Range_ = 33.3372;
        Vertical_Bottom_ = -15.594071221098034;
        Scan_Period_ = 0.1;
        ang_res_x_ = 360.0 / Horizon_SCAN_;
        ang_res_y_ = Vertical_Range_ / (N_SCAN_-1);
    }
    if(1 == lidar_type_)
    {
        N_SCAN_ = 64;
        Horizon_SCAN_ = 2000;
        Scan_Period_ = 0.1;
        ang_res_x_ = 360.0 / Horizon_SCAN_;
    }    
    
}

Lidar::~Lidar()
{
    project_img_.release();
    id_img_.release();
    oringin_lidar_img_map_.release();
    xy_mapimg_.release();
    label_mat_.release();
}

void Lidar::setCamParams(vilo::AbstractCamera* cam)
{
    w_ = cam->width();
    h_ = cam->height();
    cam_ = cam;
    extrinsic_ = cam_->getLidarCamExtrinsic();
}

void Lidar::setLidarParams()
{

}

void Lidar::registerPoints()
{
    std::vector<int> scan_sid_vec, scan_eid_vec;
    scan_sid_vec.resize(N_SCAN_);
    scan_eid_vec.resize(N_SCAN_);
   
    cv::Mat project_img = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC3);
    std::vector<pcl::PointCloud<pcl::PointXYZI>> pcs_ring_vec;
    pcs_ring_vec.resize(N_SCAN_);
  
    bool halfPassed = false;
    float startOri = -atan2(pcloud_->points[0].y, pcloud_->points[0].x);
    float endOri = -atan2(pcloud_->points[pcloud_->points.size()-1].y,
                          pcloud_->points[pcloud_->points.size()-1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI) 
    {
        endOri += 2 * M_PI;
    }

    for(size_t i=0,iend=pcloud_->points.size(); i<iend; i++)
    {
        float ptx = pcloud_->points[i].x;
        float pty = pcloud_->points[i].y;
        float ptz = pcloud_->points[i].z;

        float s_dist = ptx*ptx + pty*pty + ptz*ptz;
        if(s_dist>2500 || s_dist<1) 
            continue;

        float d = sqrt(ptx*ptx + pty*pty);
        float v_angle = atan2(ptz, d) * 180 / M_PI;

        int n_row = round(N_SCAN_ - 1 - (v_angle - Vertical_Bottom_) / ang_res_y_);
        int n_col = round(0.5 * Horizon_SCAN_ - atan2(pty, ptx)/ ang_res_x_* 180 / M_PI);

        if(n_row<0 || n_row>N_SCAN_-1 || n_col<0 || n_col>Horizon_SCAN_-1)
            continue;

        project_img.at<cv::Vec3f>(n_row,n_col)[0] = ptx;
        project_img.at<cv::Vec3f>(n_row,n_col)[1] = pty;
        project_img.at<cv::Vec3f>(n_row,n_col)[2] = ptz;

        float ori = -atan2(pty, ptx);
        if (!halfPassed)
        { 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        float relTime = (ori - startOri) / (endOri - startOri);

        pcl::PointXYZI point;
        point.x = ptx;
        point.y = pty;
        point.z = ptz;
        point.intensity = n_row + Scan_Period_*relTime;
        pcs_ring_vec[n_row].push_back(point); 
    }
    project_img_ = project_img;

    pcloud_->clear();
    for (int i = 0; i < N_SCAN_; i++)
    { 
        scan_sid_vec[i] = pcloud_->size() + 5;
        *pcloud_ += pcs_ring_vec[i];
        scan_eid_vec[i] = pcloud_->size() - 6;
    }
    scan_eid_vec_ = scan_eid_vec;
    scan_sid_vec_ = scan_sid_vec;

    assert(pcloud_->points.size() >= 10);
}

void Lidar::undistPoints(Sophus::SE3& dT, std::vector<double>& pts_ts_vec)
{
    if(!previous_lidar_)
        return;
    
    Sophus::SE3 Tlw = previous_lidar_->getLidarPose();

    Eigen::Matrix3d q1 = previous_lidar_->getImuRotation();
    Eigen::Vector3d p1 = previous_lidar_->getImuPosition();
    Eigen::Vector3d v1 = previous_lidar_->getVelocity();

    std::vector<double> dt_vec = imu_from_last_lidar_->dt_vec_;
    std::vector<Eigen::Vector3d> dp_vec = imu_from_last_lidar_->dp_vec_;
    std::vector<Eigen::Vector3d> dv_vec = imu_from_last_lidar_->dv_vec_;
    std::vector<Eigen::Matrix3d> dq_vec = imu_from_last_lidar_->dq_vec_;

    Eigen::Vector3d G; G << 0,0,-vilo::GRAVITY_VALUE;

    double dt = 0;
    std::vector<Eigen::Matrix3d> Rlc_vec;
    std::vector<Eigen::Vector3d> tlc_vec;
    
    for(size_t i=0,iend=dt_vec.size(); i<iend; i++)
    {
        dt += dt_vec[i];
        Eigen::Matrix3d dq = dq_vec[i];
        Eigen::Vector3d dp = dp_vec[i];
        Eigen::Vector3d dv = dv_vec[i];

        Eigen::Matrix3d q2 = vilo::normalizeRotation(q1*dq);
        Eigen::Vector3d v2 = v1 + G*dt + q1*dv;
        Eigen::Vector3d p2 = p1 + v1*dt + 0.5*dt*dt*G + q1 * dp;

        Sophus::SE3 Twb2 = Sophus::SE3(q2, p2);
        Sophus::SE3 Twc2 = Twb2 * T_b_l_;
        Sophus::SE3 Tlc = Tlw * Twc2;

        Eigen::Matrix3d Rlc = Tlc.rotation_matrix();
        Eigen::Vector3d tlc = Tlc.translation();
        
        Rlc_vec.push_back(Rlc);
        tlc_vec.push_back(tlc);
        dt_vec[i] = dt;
    }

    Sophus::SE3 Tlc_final(Rlc_vec.back(), tlc_vec.back());
    Sophus::SE3 Tcl_final = Tlc_final.inverse();

    int cursor = 0;
    int n_dt = dt_vec.size();
    double min_t = pts_ts_vec[0];
    for(size_t i=0,iend=pcloud_->points.size(); i<iend; i++)
    {
        double ptx = pcloud_->points[i].x;
        double pty = pcloud_->points[i].y;
        double ptz = pcloud_->points[i].z;
        
        if(0==ptx && 0==pty && 0==ptz)
            continue; 
            
        double t_pt = pts_ts_vec[i];
        for(int j=cursor;j<n_dt-1;j++)
        {
            if(t_pt >= dt_vec[j] && t_pt < dt_vec[j+1])
            {
                cursor = j+1;
                break;
            }
        }

        double t_c = dt_vec[cursor];
        Eigen::Quaterniond qlc(Rlc_vec[cursor]);
        Eigen::Vector3d tlc = tlc_vec[cursor];

        double s;
        if (is_undist_pts_)
            s = (t_pt - min_t) / t_c;
        else
            s = 1.0;

        s = s>1 ? 1:s;

        Eigen::Quaterniond q_lt = Eigen::Quaterniond::Identity().slerp(s, qlc); 
        Eigen::Vector3d t_lt = s * tlc;
        Eigen::Vector3d point(ptx, pty, ptz);
        Eigen::Vector3d un_point_last = q_lt * point + t_lt; 
        Eigen::Vector3d un_point_cur = Tcl_final * un_point_last;

        pcloud_->points[i].x = un_point_cur[0];
        pcloud_->points[i].y = un_point_cur[1];
        pcloud_->points[i].z = un_point_cur[2];
    }
}

size_t Lidar::extractFeatures()
{
    struct timeval st,et;
    gettimeofday(&st,NULL);
    
    const int N = pcloud_->points.size();
    assert(N>5);

    std::vector<float> diff_vec; diff_vec.resize(N);
    std::vector<int> id_vec; id_vec.resize(N);
    std::vector<int> nbr_pick_vec; nbr_pick_vec.resize(N);
    std::vector<int> label_vec; label_vec.resize(N);
    
    for (size_t i=5, iend=N-5; i<iend; i++)
    { 
        float diffX = pcloud_->points[i - 5].x + pcloud_->points[i - 4].x + pcloud_->points[i - 3].x + pcloud_->points[i - 2].x + pcloud_->points[i - 1].x - 10 * pcloud_->points[i].x + pcloud_->points[i + 1].x + pcloud_->points[i + 2].x + pcloud_->points[i + 3].x + pcloud_->points[i + 4].x + pcloud_->points[i + 5].x;
        float diffY = pcloud_->points[i - 5].y + pcloud_->points[i - 4].y + pcloud_->points[i - 3].y + pcloud_->points[i - 2].y + pcloud_->points[i - 1].y - 10 * pcloud_->points[i].y + pcloud_->points[i + 1].y + pcloud_->points[i + 2].y + pcloud_->points[i + 3].y + pcloud_->points[i + 4].y + pcloud_->points[i + 5].y;
        float diffZ = pcloud_->points[i - 5].z + pcloud_->points[i - 4].z + pcloud_->points[i - 3].z + pcloud_->points[i - 2].z + pcloud_->points[i - 1].z - 10 * pcloud_->points[i].z + pcloud_->points[i + 1].z + pcloud_->points[i + 2].z + pcloud_->points[i + 3].z + pcloud_->points[i + 4].z + pcloud_->points[i + 5].z;

        diff_vec[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        id_vec[i] = i;
        nbr_pick_vec[i] = 0;
        label_vec[i] = 0;
    }
    diff_vec_ = diff_vec; 

    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPointsSharp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsFlat(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>);

    for (int i = 0; i < N_SCAN_; i++)
    {
        if( scan_eid_vec_[i] - scan_sid_vec_[i] < 6)
            continue;

        pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<pcl::PointXYZI>);
        for (int j = 0; j < 6; j++) 
        {
            int sp = scan_sid_vec_[i] + (scan_eid_vec_[i] - scan_sid_vec_[i]) * j / 6; 
            int ep = scan_sid_vec_[i] + (scan_eid_vec_[i] - scan_sid_vec_[i]) * (j + 1) / 6 - 1;

            std::sort (id_vec.begin() + sp, 
                       id_vec.begin() + ep + 1, 
                       [&](int i, int j){return diff_vec[i] < diff_vec[j];}
                      );

            int nCorner = 0;
            for (int k = ep; k >= sp; k--) 
            {
                int ind = id_vec[k]; 

                if (nbr_pick_vec[ind] == 0 &&
                    diff_vec[ind] > 0.1)
                {

                    nCorner++;
                    if (nCorner <= 2)
                    {                        
                        label_vec[ind] = 2;
                        cornerPointsSharp->points.push_back(pcloud_->points[ind]);
                        cornerPointsLessSharp->points.push_back(pcloud_->points[ind]);
                    }
                    else if (nCorner <= 20)
                    {                        
                        label_vec[ind] = 1; 
                        cornerPointsLessSharp->points.push_back(pcloud_->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    nbr_pick_vec[ind] = 1; 

                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = pcloud_->points[ind + l].x - pcloud_->points[ind + l - 1].x;
                        float diffY = pcloud_->points[ind + l].y - pcloud_->points[ind + l - 1].y;
                        float diffZ = pcloud_->points[ind + l].z - pcloud_->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        nbr_pick_vec[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = pcloud_->points[ind + l].x - pcloud_->points[ind + l + 1].x;
                        float diffY = pcloud_->points[ind + l].y - pcloud_->points[ind + l + 1].y;
                        float diffZ = pcloud_->points[ind + l].z - pcloud_->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        nbr_pick_vec[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = id_vec[k];

                if (nbr_pick_vec[ind] == 0 &&
                    diff_vec[ind] < 0.1)
                {

                    label_vec[ind] = -1; 
                    surfPointsFlat->points.push_back(pcloud_->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    nbr_pick_vec[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = pcloud_->points[ind + l].x - pcloud_->points[ind + l - 1].x;
                        float diffY = pcloud_->points[ind + l].y - pcloud_->points[ind + l - 1].y;
                        float diffZ = pcloud_->points[ind + l].z - pcloud_->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        nbr_pick_vec[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = pcloud_->points[ind + l].x - pcloud_->points[ind + l + 1].x;
                        float diffY = pcloud_->points[ind + l].y - pcloud_->points[ind + l + 1].y;
                        float diffZ = pcloud_->points[ind + l].z - pcloud_->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        nbr_pick_vec[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                if (label_vec[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(pcloud_->points[k]);
                }
            }
        }
    
        pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScanDS(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(*surfPointsLessFlatScanDS);

        *surfPointsLessFlat += *surfPointsLessFlatScanDS;
    }

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCorner(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurf(new pcl::KdTreeFLANN<pcl::PointXYZI>());

    kdtree_corner_->setInputCloud(cornerPointsLessSharp);
    kdtree_surf_->setInputCloud(surfPointsLessFlat);

    ptr_corners_ = cornerPointsSharp;
    ptr_surfs_ = surfPointsFlat;
    ptr_less_corners_ = cornerPointsLessSharp;
    ptr_less_surfs_ = surfPointsLessFlat;
    
    cv::Mat id_img = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC4);
    for(size_t i=0,iend=ptr_less_surfs_->points.size();i<iend;i++)
    {
        float ptx = ptr_less_surfs_->points[i].x;
        float pty = ptr_less_surfs_->points[i].y;
        int n_row = int(ptr_less_surfs_->points[i].intensity);
        int n_col = round(0.5 * Horizon_SCAN_ - atan2(pty, ptx)/ ang_res_x_* 180 / M_PI);
       
        if(n_row<0 || n_row>N_SCAN_-1 || n_col<0 || n_col>Horizon_SCAN_-1)
            continue;

        id_img.at<cv::Vec4f>(n_row,n_col)[0] = ptx;
        id_img.at<cv::Vec4f>(n_row,n_col)[1] = pty;
        id_img.at<cv::Vec4f>(n_row,n_col)[2] = ptr_less_surfs_->points[i].z;
        id_img.at<cv::Vec4f>(n_row,n_col)[3] = i;
    }
    id_img_ = id_img;
    
    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);  
    
    return surfPointsFlat->points.size();
}

void Lidar::transformToStart(pcl::PointXYZI& pi, pcl::PointXYZI& po, float SCAN_PERIOD, 
                             Eigen::Quaterniond& q_last_curr, Eigen::Vector3d& t_last_curr)
{
    bool DISTORTION = false;
    double s;
    if (DISTORTION)
        s = (pi.intensity - int(pi.intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi.x, pi.y, pi.z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po.x = un_point.x();
    po.y = un_point.y();
    po.z = un_point.z();
    po.intensity = pi.intensity;
}

void Lidar::releaseMemory(int flag)
{
    
}

void Lidar::projectLaserToImage(const Eigen::Matrix<double,3,4>& velo_to_img,const cv::Mat& image)
{
    cv::Mat img; image.copyTo(img);
    
    struct timeval st,et;
    gettimeofday(&st,NULL);

	velo_to_img_ = velo_to_img;
    cv::Mat xy_img = cv::Mat::zeros(N_SCAN_,Horizon_SCAN_,CV_32FC3);
    cv::Mat label_img = cv::Mat::zeros(N_SCAN_,Horizon_SCAN_,CV_32SC1);
    Eigen::Matrix<double,4,1> pt;
    Eigen::Matrix<double,3,1> pt_proj;
    int x,y;
    int count = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_on_img(new pcl::PointCloud<PcsType>);

    int alread = 0;
    int n_all=0;
    int x_tl = 10000,y_tl = 10000,x_br = -1,y_br = -1;
    
    for(size_t i=0,iend=pcloud_->points.size(); i<iend; i++)
    {
        if(pcloud_->points[i].x > 0)
            continue;
        float ptx = pcloud_->points[i].x;
        float pty = pcloud_->points[i].y;
        float ptz = pcloud_->points[i].z;

        float s_dist = ptx*ptx + pty*pty + ptz*ptz;
        if( s_dist>1600 || s_dist<0.25)
            continue;

        pt<<ptx,pty,ptz,1.0;
        pt_proj = velo_to_img_*pt;
        x = round(pt_proj(0,0)/pt_proj(2,0));
        y = round(pt_proj(1,0)/pt_proj(2,0));

        if(x >= 0 && x < w_ && y >= 0 && y < h_)
        {            
            float d = sqrt(ptx*ptx + pty*pty);
            float v_angle = atan2(ptz, d) * 180 / M_PI;
            int n_row, n_col;
            if(1 == lidar_type_)
            {
                n_row = round(N_SCAN_ - (v_angle + Vertical_Bottom_) / ang_res_y_);
                n_col = round(0.5 * Horizon_SCAN_ - atan2(pty, ptx)/ ang_res_x_* 180 / M_PI);
            }
            else if(0 == lidar_type_)
            {
                n_row = round(pcloud_->points[i].intensity);
                n_col = round(0.5 * Horizon_SCAN_ - atan2(pty, ptx)/ ang_res_x_* 180 / M_PI);
            }
            else
            {
                n_row = 0;
                n_col = 0;
            }

            if(n_col<0 || n_col>Horizon_SCAN_-1 || n_row<0 || n_row>N_SCAN_-1)
                continue;

            if(n_col<x_tl)
                x_tl = n_col;
            if(n_col>x_br)
                x_br = n_col;
            if(n_row<y_tl)
                y_tl = n_row;
            if(n_row>y_br)
                y_br = n_row;

            n_all++;

            if(0!=xy_img.at<cv::Vec3f>(n_row,n_col)[0])
            {
                float prev_x = xy_img.at<cv::Vec3f>(n_row,n_col)[0];
                float prev_y = xy_img.at<cv::Vec3f>(n_row,n_col)[1];
                float prev_r2 = prev_x*prev_x + prev_y*prev_y;
                if(prev_r2 > d*d)
                {
                    xy_img.at<cv::Vec3f>(n_row,n_col)[0] = ptx;
                    xy_img.at<cv::Vec3f>(n_row,n_col)[1] = pty;
                    xy_img.at<cv::Vec3f>(n_row,n_col)[2] = ptz;
                    label_img.at<int>(n_row,n_col) = pcloud_->points[i].intensity;
                }
                alread++;
            }
            else
            {
                xy_img.at<cv::Vec3f>(n_row,n_col)[0] = ptx;
                xy_img.at<cv::Vec3f>(n_row,n_col)[1] = pty;
                xy_img.at<cv::Vec3f>(n_row,n_col)[2] = ptz;
                label_img.at<int>(n_row,n_col) = pcloud_->points[i].intensity;

                if(ptz > z_max_)
                    z_max_ = ptz;
                if(ptz < z_min_)
                    z_min_ = ptz;
                count++;

            }

        }
       
    }

    cv::Rect rec(x_tl,y_tl,x_br-x_tl+1,y_br-y_tl+1);
    xy_img(rec).copyTo(xy_mapimg_);
    label_img(rec).copyTo(label_mat_);
    N_ = count;

    const int n_cols = xy_mapimg_.cols;
	const int n_rows = xy_mapimg_.rows;
    oringin_lidar_img_map_ = cv::Mat::zeros(h_,w_, CV_32SC2);
    
    for(int col=0;col<n_cols;col++)
    {     
        for(int row=0;row<n_rows;row++)
		{
            float x = xy_mapimg_.at<cv::Vec3f>(row,col)[0];
			float y = xy_mapimg_.at<cv::Vec3f>(row,col)[1];
			float z = xy_mapimg_.at<cv::Vec3f>(row,col)[2];
			
			if(0 == xy_mapimg_.at<cv::Vec3f>(row,col)[1])
			  continue;
			
            Eigen::Vector4d tmp_pt(x,y,z,1.0);
            Eigen::Vector3d pt_proj;
            pt_proj = velo_to_img_ * tmp_pt; 

			int x_orin_img = round(pt_proj(0,0)/pt_proj(2,0));
	        int y_orin_img = round(pt_proj(1,0)/pt_proj(2,0)); 
            
            assert(x_orin_img>=0 && x_orin_img<=w_-1 && y_orin_img>=0 && y_orin_img<=h_-1);
		
			oringin_lidar_img_map_.at<cv::Vec2i>(y_orin_img,x_orin_img)[0] = row;
			oringin_lidar_img_map_.at<cv::Vec2i>(y_orin_img,x_orin_img)[1] = col;       
        }
    }
}

int Lidar::findFeatureDepth(std::vector<cv::Point2f>& pixels ,
                            std::vector<bool>& is_depth_found,
                            std::vector<Eigen::Vector3d>& intersections)
{
    struct timeval st,et;
    gettimeofday(&st,NULL);
    
    const size_t N = pixels.size();
    const int n_threshold = 3;
    const int COLS = xy_mapimg_.cols;
    const int ROWS = xy_mapimg_.rows;
    const int n_cols = oringin_lidar_img_map_.cols;
    const int n_rows = oringin_lidar_img_map_.rows;

    int nbignbrs = 0 , nsmallnbrs = 0, ncorners = 0;
    int ndepthfound = 0 , ndepthsuspect = 0;
    for(size_t i=0;i<N;i++)
    {
        double px = pixels[i].x ,  py = pixels[i].y;
        int pxi = round(px); int pyi = round(py);
        int kpxi = -1 , kpyi = -1;

        for(int p=0;p<25;p++)
        {
            int ppxi = pxi+ pattern_25_[p][0]; int ppyi = pyi+ pattern_25_[p][1];
            if(ppxi<0 || ppxi>n_cols-1 || ppyi<0 || ppyi>n_rows-1)
                continue;
            if(0 == oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[0] &&
                    0 == oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[1])
                    continue;
            kpxi = oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[1];
            kpyi = oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[0];
            break;
        }

        if(-1 == kpxi)
            continue;
        
        assert(kpxi>=0 && kpxi<=COLS-1 && kpyi>=0 && kpyi<=ROWS-1);

        int px_label = label_mat_.at<int>(kpyi,kpxi);
        if(0==px_label || px_label>9999)
            continue;

        std::vector<Eigen::Matrix<double,5,1> > nbrs;       
        {
            for(int h=-1;h<=1;h++)
            {
                for(int w=-2;w<=2;w++)
                {
                    int x = kpxi+w;  int y = kpyi+h;

                    if(x<0 || x>COLS-1 || y<0 || y>ROWS-1 )
                        continue;
                    
                    if(0 == label_mat_.at<int>(y,x))
                        continue;

                    double laser_x =  xy_mapimg_.at<cv::Vec3f>(y,x)[0];
                    double laser_y =  xy_mapimg_.at<cv::Vec3f>(y,x)[1];
                    double laser_z =  xy_mapimg_.at<cv::Vec3f>(y,x)[2];

                    Eigen::Vector4d laser_pt(laser_x,laser_y,laser_z,1.0);
                    Eigen::Vector4d cam_pt = extrinsic_ * laser_pt;

                    if(cam_pt.head(3).norm()<1)
                        continue;

                    Eigen::Matrix<double,5,1> l_x_p;
                    l_x_p[0] = x; l_x_p[1] = y;
                    l_x_p[2] = cam_pt[0]; l_x_p[3] = cam_pt[1]; l_x_p[4] = cam_pt[2];

                    nbrs.push_back(l_x_p);
                }
            }

        }
        
        if(nbrs.size()<3)
            continue;
        nbignbrs++;
        
        std::vector<Eigen::Matrix<double,5,1> > best_nbrs;

        
        {
            if( !findBestNbrs(nbrs , best_nbrs , px_label) )
                continue;
        }

        const float ang_res_y_ = 0.427;
        Vertical_Bottom_ = 24.9;
        int last_row = -1;
        bool is_samering = true;
        for(size_t k=0;k<best_nbrs.size();k++)
        {
            double cpt_x = best_nbrs[k][2];
            double cpt_y = best_nbrs[k][3];
            double cpt_z = best_nbrs[k][4];

            float rad = sqrt(cpt_x*cpt_x + cpt_z*cpt_z);
            float v_angle = atan2(cpt_y, rad) * 180 / M_PI;
            int n_row = round(v_angle / ang_res_y_);
            if(0==k)
            {
                last_row = n_row;
                continue;
            }
            if(last_row != n_row)
            {
                is_samering = false;
                break;
            }

        }
        if(is_samering)
        {
            continue;
        }
        nsmallnbrs++;

        Eigen::Vector3d pts_norm;
        std::vector<Eigen::Matrix<double,5,1> > real_best_nbrs;
        int countor = 0;
        double cpt_x_sum = 0, cpt_y_sum = 0, cpt_z_sum = 0;
        Eigen::MatrixXd matA(best_nbrs.size(),3);
        Eigen::MatrixXd matb(best_nbrs.size(),1);
        for(size_t k=0;k<best_nbrs.size();k++)
        {
            double cpt_x = best_nbrs[k][2];
            double cpt_y = best_nbrs[k][3];
            double cpt_z = best_nbrs[k][4];

            matA(k,0) = cpt_x;
            matA(k,1) = cpt_y;
            matA(k,2) = cpt_z;
        }
        matb.setOnes();
        matb = -1 * matb;
        Eigen::Vector3d matX = matA.colPivHouseholderQr().solve(matb);
        double nrm = matX.norm();
        matX /= nrm;
        double D = 1 / nrm;

        for(size_t k=0;k<best_nbrs.size();k++)
        {
            double cpt_x = best_nbrs[k][2];
            double cpt_y = best_nbrs[k][3];
            double cpt_z = best_nbrs[k][4];

            double d_plane = abs(cpt_x*matX[0] + cpt_y*matX[1] + cpt_z*matX[2] + D);

            if(abs(cpt_x*matX[0] + cpt_y*matX[1] + cpt_z*matX[2] + D)>0.005*D )
            {
                countor++;
                continue;
            }
            real_best_nbrs.push_back(best_nbrs[k]);
            cpt_x_sum += cpt_x;
            cpt_y_sum += cpt_y;
            cpt_z_sum += cpt_z;

        }
        if(countor>=2 || real_best_nbrs.size()<3)
        {
            continue;
        }
        pca(real_best_nbrs , pts_norm);
        ncorners++;

        Eigen::Vector3d direction = cam_->cam2world(px,py);
        Eigen::Vector3d drt(direction[0],direction[1],direction[2]);
        Eigen::Vector3d oringin(0,0,0);

        cpt_x_sum /= real_best_nbrs.size();
        cpt_y_sum /= real_best_nbrs.size();
        cpt_z_sum /= real_best_nbrs.size();
        Eigen::Vector3d center(cpt_x_sum,cpt_y_sum,cpt_z_sum);

        Eigen::Vector3d nml = pts_norm;
        nml.normalize();

        Eigen::ParametrizedLine<double, 3> line = Eigen::ParametrizedLine<double, 3>::Through(oringin, drt);
        Eigen::Hyperplane<double, 3> plane = Eigen::Hyperplane<double, 3>(nml, center);

        Eigen::Vector3d intersection = line.intersectionPoint(plane);
        Eigen::Vector2d intersection_2d = cam_->world2cam(intersection);

        ndepthsuspect++;
        Eigen::Vector3d delta_dist(intersection - center);
        double max_local_dist = 0;
        for(size_t k=0;k<real_best_nbrs.size();k++)
        {
            Eigen::Vector3d cpt(best_nbrs[k][2], best_nbrs[k][3], best_nbrs[k][4]);
            double delta_d = (cpt - center).norm();
            if(max_local_dist < delta_d)
                max_local_dist = delta_d;

        }
        if((intersection - center).norm() > max_local_dist)
        {
            continue;
        }
        
        ndepthfound++;
        
        is_depth_found[i] = true;
        intersections[i] = intersection;
    }

    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
   
    return ndepthfound;
}


bool Lidar::pca(std::vector<Eigen::Matrix<double,5,1> >& best_nbrs , Eigen::Vector3d& pts_norm)
{
    double treshold_2_1_rel_min = 0.5;
    double treshold_3_2_rel_max = 15.0;
    double treshold_3_abs_min = 0.005;


    Eigen::MatrixXd pts(3,best_nbrs.size());
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> eigenVectorsValues;
    for(size_t k=0;k<best_nbrs.size();k++)
    {
        pts(0,k) = best_nbrs[k][2];
        pts(1,k) = best_nbrs[k][3];
        pts(2,k) = best_nbrs[k][4];
    }
    Eigen::Vector3d p_cent = pts.rowwise().mean();
    Eigen::MatrixXd pts_centered = pts.colwise() - p_cent;
    Eigen::MatrixXd cov = pts_centered * pts_centered.adjoint();
    assert(cov.rows() == 3 && cov.cols() == 3);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

    Eigen::Vector4d CurComp;
    for (size_t i = 0; i < 3; ++i) {
        CurComp << eig.eigenvectors().col(i) / eig.eigenvectors().col(i).norm(), eig.eigenvalues()(i);
        eigenVectorsValues.push_back(CurComp);
    }
    assert(eigenVectorsValues[0](3) < eigenVectorsValues[1](3));
    pts_norm = eigenVectorsValues[0].head(3);

    double ev1 = eigenVectorsValues[0](3);
    double ev2 = eigenVectorsValues[1](3);
    double ev3 = eigenVectorsValues[2](3);

    double planarity = (ev2 - ev1) / ev3;
    double linearity = (ev3 - ev2) / ev3;
    
    if (planarity < treshold_2_1_rel_min)
        return false;

    if (linearity > treshold_3_2_rel_max)
        return false;

    if (ev3 < treshold_3_abs_min)
        return false;

    return true;
}

bool Lidar::findBestNbrs(std::vector<Eigen::Matrix<double,5,1> >& nbrs ,
                                        std::vector<Eigen::Matrix<double,5,1> >& best_nbrs,
                                         int px_label)
{
    const int N = nbrs.size();

    double bin_width = 0.4;
    int min_pts_thrd = 3;
    int depthCount = N;
    int maxDist = 0;
    int minDist = 99999.0;
    std::vector<double> z_dist;

    for(int i=0;i<N;i++)
    {
        z_dist.push_back(nbrs[i][4]);

        if(nbrs[i][4] > maxDist)
            maxDist = ceil(nbrs[i][4]);
        if(nbrs[i][4] < minDist)
            minDist = floor(nbrs[i][4]);
    }

    double lowerBorder = -1.0;
    double higherBorder = -1.0;
    int binCount;

    binCount = (maxDist - minDist) / bin_width + 1;
    std::multimap<int,int> map_bin_index;

    for (int i = 0; i < depthCount; i++)
    {
        double depth = z_dist[i];
        double value = std::min(depth, 1e10);
        int bin_ind = floor((value - minDist)/bin_width);
        map_bin_index.insert(std::make_pair(bin_ind,i));
    }

    int binMaxId = -1;
    int binMaxVal = -1;
    int binSecondMaxId = -1;
    int binSecondMaxVal = -1;
    int binValue = 0;

    for(int i=0;i<binCount;i++)
    {
        binValue = map_bin_index.count(i);
        if(binValue > binMaxVal && binValue >= min_pts_thrd)
        {
            binSecondMaxId = binMaxId;
            binSecondMaxVal = binMaxVal;
            binMaxId = i;
            binMaxVal = binValue;
        }
    }

    if(-1==binMaxId && -1==binSecondMaxId)
        return false;
    else if(-1!=binMaxId && -1==binSecondMaxId)
    {
        std::multimap<int,int>::iterator sit,eit,it;
        sit = map_bin_index.lower_bound(binMaxId);
        eit = map_bin_index.upper_bound(binMaxId);
        for(it=sit; it!=eit; ++it)
            best_nbrs.push_back(nbrs[it->second]);
        return true;

    }
    else if(-1!=binMaxId && -1!=binSecondMaxId)
    {
        if(1 == abs(binMaxId - binSecondMaxId))
        {
            std::multimap<int,int>::iterator sit,eit,it;
            sit = map_bin_index.lower_bound(binMaxId);
            eit = map_bin_index.upper_bound(binMaxId);
            for(it=sit; it!=eit; ++it)
                best_nbrs.push_back(nbrs[it->second]);

            sit = map_bin_index.lower_bound(binSecondMaxId);
            eit = map_bin_index.upper_bound(binSecondMaxId);
            for(it=sit; it!=eit; ++it)
                best_nbrs.push_back(nbrs[it->second]);
        }
        else 
        {
            int bin_id = min(binMaxId , binSecondMaxId);
            std::multimap<int,int>::iterator sit,eit,it;
            sit = map_bin_index.lower_bound(bin_id);
            eit = map_bin_index.upper_bound(bin_id);
            for(it=sit; it!=eit; ++it)
                best_nbrs.push_back(nbrs[it->second]);
        }
        return true;
    }

    return false;
}

bool Lidar::isPointInTriangle(const Eigen::Vector2d& A, const Eigen::Vector2d& B, const Eigen::Vector2d& C, const Eigen::Vector2d& P)
{
    Eigen::Vector2d v0 = C - A ;
    Eigen::Vector2d v1 = B - A ;
    Eigen::Vector2d v2 = P - A ;

    double dot00 = v0.dot(v0) ;
    double dot01 = v0.dot(v1) ;
    double dot02 = v0.dot(v2) ;
    double dot11 = v1.dot(v1) ;
    double dot12 = v1.dot(v2) ;

    double inverDeno = 1 / (dot00 * dot11 - dot01 * dot01) ;

    double u = (dot11 * dot02 - dot01 * dot12) * inverDeno ;
    if (u < 0 || u > 1.2) 
    {
        return false ;
    }

    double v = (dot00 * dot12 - dot01 * dot02) * inverDeno ;
    if (v < 0 || v > 1.2) 
    {
        return false ;
    }

    return u + v <= 1.2 ;
}

bool Lidar::calculatePlaneCorners(const std::vector<Eigen::Matrix<double,5,1> >& label_xy_pos,
                                             const Eigen::Vector2d& px,
                                             Eigen::Matrix<double,5,1>& corner1,
                                             Eigen::Matrix<double,5,1>& corner2,
                                             Eigen::Matrix<double,5,1>& corner3)
{
    const double _distTreshold = 0.1;
    int pointsCount = label_xy_pos.size();

    if (pointsCount < 3) {
        return false;
    }

    int maxDist_i = -1;
    int maxDist_j = -1;
    float maxdist = -1;

    for (int i = 0; i < pointsCount - 1; i++) {
        for (int j = i + 1; j < pointsCount; j++) {
            double dist = (label_xy_pos[i].segment<3>(2) - label_xy_pos[j].segment<3>(2)).squaredNorm();

            if (dist > maxdist) {
                maxdist = dist;
                maxDist_i = i;
                maxDist_j = j;
            }
        }
    }
    
    if (maxdist <= _distTreshold)
        return false;

    double maxdist2 = -1;
    double maxDist_k = -1;
    for (int k = 0; k < pointsCount - 1; k++) {
        if (k == maxDist_i || k == maxDist_j)
            continue;

        double dist1 = (label_xy_pos[k].segment<3>(2) - label_xy_pos[maxDist_i].segment<3>(2)).norm();

        if (dist1 <= _distTreshold)
            continue;

        double dist2 = (label_xy_pos[k].segment<3>(2) - label_xy_pos[maxDist_j].segment<3>(2)).norm();

        if (dist2 <= _distTreshold)
            continue;

        double dist = dist1 + dist2;

        if (dist > maxdist2) {
            maxdist2 = dist;
            maxDist_k = k;
        }


    }
    if ((maxDist_i == -1) || (maxDist_j == -1) || (maxDist_k == -1))
        return false;

    corner1 = label_xy_pos[maxDist_i].head(5);
    corner2 = label_xy_pos[maxDist_j].head(5);
    corner3 = label_xy_pos[maxDist_k].head(5);

    return true;


}

bool Lidar::checkPlanar(const Eigen::Vector3d& corner1,
                                             const Eigen::Vector3d& corner2,
                                             const Eigen::Vector3d& corner3)
{
    double threshold = 0.1;
    Eigen::Vector3d edge1 = corner2 - corner1;
    Eigen::Vector3d edge2 = corner3 - corner1;
    Eigen::Vector3d edge3 = corner3 - corner2;
    edge1.normalize();
    edge2.normalize();
    edge3.normalize();

    Eigen::Vector3d cross12 = edge1.cross(edge2);
    Eigen::Vector3d cross13 = edge1.cross(edge3);
    Eigen::Vector3d cross23 = edge2.cross(edge3);

    double length12 = cross12.norm();
    double length13 = cross13.norm();
    double length23 = cross23.norm();

    bool check12 = (length12 >= threshold);
    bool check13 = (length13 >= threshold);
    bool check23 = (length23 >= threshold);

    return (check12 && check13 && check23);
}

bool Lidar::findDepthOnce(Eigen::Vector2d pixel , Eigen::Vector3d& intersection)
{
    const int n_threshold = 3;
    const int COLS = xy_mapimg_.cols;
    const int ROWS = xy_mapimg_.rows;
    const int n_cols = oringin_lidar_img_map_.cols;
    const int n_rows = oringin_lidar_img_map_.rows;

	double px = pixel[0] ,  py = pixel[1];
    int pxi = round(px); int pyi = round(py);
    int kpxi = -1 , kpyi = -1;

    for(int p=0;p<25;p++)
    {
        int ppxi = pxi+ pattern_25_[p][0]; int ppyi = pyi+ pattern_25_[p][1];
        if(ppxi<0 || ppxi>n_cols-1 || ppyi<0 || ppyi>n_rows-1)
            continue;
        if(0 == oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[0] &&
                0 == oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[1])
                continue;
        kpxi = oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[1];
        kpyi = oringin_lidar_img_map_.at<cv::Vec2i>(ppyi,ppxi)[0];
        break;
    }

	if(-1 == kpxi || -1 == kpyi)
    {
        return false;
    }

    assert(kpxi>=0 && kpxi<=COLS-1 && kpyi>=0 && kpyi<=ROWS-1);

    int px_label = label_mat_.at<int>(kpyi,kpxi);
    if(0==px_label || px_label>9999)
    {
        return false;
    }

    std::vector<Eigen::Matrix<double,5,1> > nbrs;
    if(px_label==1)
    {
        for(int h=-1;h<=1;h++)
        {
            for(int w=-2;w<=2;w++)
            {
                int x = kpxi+w;  int y = kpyi+h;

                if(x<0 || x>COLS-1 || y<0 || y>ROWS-1 )
                    continue;
                if(px_label != label_mat_.at<int>(y,x))
                    continue;

                double laser_x =  xy_mapimg_.at<cv::Vec3f>(y,x)[0];
                double laser_y =  xy_mapimg_.at<cv::Vec3f>(y,x)[1];
                double laser_z =  xy_mapimg_.at<cv::Vec3f>(y,x)[2];

                Eigen::Vector4d laser_pt(laser_x,laser_y,laser_z,1.0);
                Eigen::Vector4d cam_pt = extrinsic_ * laser_pt;

                if(cam_pt.head(3).norm()<1)
                    continue;

                Eigen::Matrix<double,5,1> l_x_p;
                l_x_p[0] = x; l_x_p[1] = y;
                l_x_p[2] = cam_pt[0]; l_x_p[3] = cam_pt[1]; l_x_p[4] = cam_pt[2];

                nbrs.push_back(l_x_p);

            }
        }

    }
    else
    {
        for(int h=-1;h<=1;h++)
        {
            for(int w=-2;w<=2;w++)
            {
                int x = kpxi+w;  int y = kpyi+h;

                if(x<0 ||x>COLS-1 || y<0 || y>ROWS-1 )
                    continue;

                int label = label_mat_.at<int>(y,x);
                if(0==label || 1==label || label>9999)
                    continue;

                double laser_x =  xy_mapimg_.at<cv::Vec3f>(y,x)[0];
                double laser_y =  xy_mapimg_.at<cv::Vec3f>(y,x)[1];
                double laser_z =  xy_mapimg_.at<cv::Vec3f>(y,x)[2];

                Eigen::Vector4d laser_pt(laser_x,laser_y,laser_z,1.0);
                Eigen::Vector4d cam_pt = extrinsic_ * laser_pt;

                if(cam_pt.head(3).norm()<1)
                    continue;

                Eigen::Matrix<double,5,1> l_x_p;
                l_x_p[0] = x; l_x_p[1] = y;

                l_x_p[2] = cam_pt[0]; l_x_p[3] = cam_pt[1]; l_x_p[4] = cam_pt[2];

                nbrs.push_back(l_x_p);
            }
        }
    }

    if(nbrs.size()<3)
    {
        return false;
    }

    std::vector<Eigen::Matrix<double,5,1> > best_nbrs;

    if(px_label==1)
        best_nbrs = nbrs;
    else
    {
        if( !findBestNbrs(nbrs , best_nbrs , px_label) )
        {
            return false;
        }

    }

    ang_res_y_ = 0.427;
    Vertical_Bottom_ = 24.9;
    int last_row = -1;
    bool is_samering = true;
    for(size_t k=0;k<best_nbrs.size();k++)
    {
        double cpt_x = best_nbrs[k][2];
        double cpt_y = best_nbrs[k][3];
        double cpt_z = best_nbrs[k][4];

        float rad = sqrt(cpt_x*cpt_x + cpt_z*cpt_z);
        float v_angle = atan2(cpt_y, rad) * 180 / M_PI;
        int n_row = round(v_angle / ang_res_y_);
        if(0==k)
        {
            last_row = n_row;
            continue;
        }
        if(last_row != n_row)
        {
            is_samering = false;
            break;
        }

    }
    if(is_samering)
    {
        return false;
    }

    Eigen::Vector3d pts_norm;
    
    std::vector<Eigen::Matrix<double,5,1> > real_best_nbrs;
    int countor = 0;
    double cpt_x_sum = 0, cpt_y_sum = 0, cpt_z_sum = 0;
    Eigen::MatrixXd matA(best_nbrs.size(),3);
    Eigen::MatrixXd matb(best_nbrs.size(),1);
    for(size_t k=0;k<best_nbrs.size();k++)
    {
        double cpt_x = best_nbrs[k][2];
        double cpt_y = best_nbrs[k][3];
        double cpt_z = best_nbrs[k][4];

        matA(k,0) = cpt_x;
        matA(k,1) = cpt_y;
        matA(k,2) = cpt_z;
    }
    matb.setOnes();
    matb = -1 * matb;
    Eigen::Vector3d matX = matA.colPivHouseholderQr().solve(matb);
    double nrm = matX.norm();
    matX /= nrm;
    double D = 1 / nrm;

    for(size_t k=0;k<best_nbrs.size();k++)
    {
        double cpt_x = best_nbrs[k][2];
        double cpt_y = best_nbrs[k][3];
        double cpt_z = best_nbrs[k][4];

        double d_plane = abs(cpt_x*matX[0] + cpt_y*matX[1] + cpt_z*matX[2] + D);

        if(abs(cpt_x*matX[0] + cpt_y*matX[1] + cpt_z*matX[2] + D)>0.01*D )
        {
            countor++;
            continue;
        }
        real_best_nbrs.push_back(best_nbrs[k]);
        cpt_x_sum += cpt_x;
        cpt_y_sum += cpt_y;
        cpt_z_sum += cpt_z;

    }
    if(countor>=2 || real_best_nbrs.size()<3)
    {
        return false;
    }
    pca(real_best_nbrs , pts_norm);

    Eigen::Vector3d direction = cam_->cam2world(px,py);
    Eigen::Vector3d drt(direction[0],direction[1],direction[2]);
    Eigen::Vector3d oringin(0,0,0);

    cpt_x_sum /= real_best_nbrs.size();
    cpt_y_sum /= real_best_nbrs.size();
    cpt_z_sum /= real_best_nbrs.size();
    Eigen::Vector3d center(cpt_x_sum,cpt_y_sum,cpt_z_sum);

    Eigen::Vector3d nml = pts_norm;
    nml.normalize();

    Eigen::ParametrizedLine<double, 3> line = Eigen::ParametrizedLine<double, 3>::Through(oringin, drt);
    Eigen::Hyperplane<double, 3> plane = Eigen::Hyperplane<double, 3>(nml, center);

    intersection = line.intersectionPoint(plane);
    Eigen::Vector2d intersection_2d = cam_->world2cam(intersection);

    Eigen::Vector3d delta_dist(intersection - center);
    double max_local_dist = 0;
    for(size_t k=0;k<real_best_nbrs.size();k++)
    {
        Eigen::Vector3d cpt(best_nbrs[k][2], best_nbrs[k][3], best_nbrs[k][4]);
        double delta_d = (cpt - center).norm();
        if(max_local_dist < delta_d)
            max_local_dist = delta_d;

    }
    if((intersection - center).norm() > max_local_dist)
    {
        return false;
    }
    
    for(size_t k=0;k<real_best_nbrs.size();k++)
    {
        Eigen::Vector3d cpt(best_nbrs[k][2], best_nbrs[k][3], best_nbrs[k][4]);
        Eigen::Vector3d cpx = cam_->K() * cpt;
    }
    return true;
}

void Lidar::finish()
{
    xy_mapimg_.release();
    label_mat_.release();
    oringin_lidar_img_map_.release();
}

bool Lidar::setGridOccpuancy(const std::vector<Eigen::Vector2d>& pxs)
{
    resetGrid();

    const int W = xy_mapimg_.cols; const int H = xy_mapimg_.rows;
    const int n_cols = oringin_lidar_img_map_.cols;
    const int n_rows = oringin_lidar_img_map_.rows;
    if(0==W || 0==H)
        return false;
    w_grid_size_ = floor(1.f * W / grid_w_num_); h_grid_size_ = floor(1.f * H / grid_h_num_);
  
    bool is_found = false; int n_nbr=0;
    for(size_t i=0,iend=pxs.size(); i<iend; i++)
    {
        double px = pxs[i][0] ,  py = pxs[i][1];
        int pxi = round(px); int pyi = round(py);
        int kpxi = -1 , kpyi = -1;
        for(int m=-2;m<=2;m++)
        {
            for(int n=-2;n<=2;n++)
            {
                if(pxi+m<0 || pxi+m>n_cols-1 || pyi+n<0 || pyi+n>n_rows-1)
                    continue;
                if(0 == oringin_lidar_img_map_.at<cv::Vec2i>(pyi+n,pxi+m)[0] &&
                    0 == oringin_lidar_img_map_.at<cv::Vec2i>(pyi+n,pxi+m)[1])
                    continue;
                kpxi = oringin_lidar_img_map_.at<cv::Vec2i>(pyi+n,pxi+m)[1];
                kpyi = oringin_lidar_img_map_.at<cv::Vec2i>(pyi+n,pxi+m)[0];

                break;
            }
        }

        if(-1 == kpxi || -1 == kpyi)
            continue;

        int w_id = kpxi % grid_w_num_; int h_id = kpyi / grid_w_num_;
        grid_occupancy_.at(w_id * grid_w_num_+ h_id) = true;
    }
    return true;
}

void Lidar::chooseExtraLaserFeatrues(std::vector<Eigen::Vector3d>& features)
{
    int n_new = 0; int n_fail=0;
    for(size_t i=0,iend=grid_occupancy_.size();i<iend;i++)
    {
        if(grid_occupancy_[i])
            continue;
        int w_id = i % grid_w_num_; int h_id = i / grid_w_num_;
        int w_s = w_id * w_grid_size_; int w_e = (w_id+1) * w_grid_size_;
        int h_s = h_id * h_grid_size_; int h_e = (h_id+1) * h_grid_size_;

        bool is_found = false;
        for(int m=w_s;m<w_e;m++)
        {
            for(int n=h_s;n<h_e;n++)
            {
                if(0==label_mat_.at<int>(n,m) || label_mat_.at<int>(n,m)>9999)
                    continue;
                    
                float x = xy_mapimg_.at<cv::Vec3f>(n,m)[0];
			    float y = xy_mapimg_.at<cv::Vec3f>(n,m)[1];
			    float z = xy_mapimg_.at<cv::Vec3f>(n,m)[2];

                Eigen::Vector4d lpt; lpt<<x,y,z,1.0;
                Eigen::Vector4d cpt(extrinsic_ * lpt);
                features.push_back(cpt.head(3));
                n_new++;
                is_found = true;
                break;
            }
            if(is_found)
                break;
        }
    }
}

void Lidar::resetGrid()
{
    grid_occupancy_.resize(grid_w_num_*grid_h_num_);
    std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

Sophus::SE3 Lidar::getLidarPose()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return T_f_w_;
}
    
void Lidar::setLidarPose(Sophus::SE3& Tfw)
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    T_f_w_ = Tfw;
}

Eigen::Vector3d Lidar::getImuAccBias()
{
    return ba_;
}

Eigen::Vector3d Lidar::getImuGyroBias()
{
    return bg_;
}

void Lidar::setImuAccBias(const Eigen::Vector3d& ba)
{
    ba_ = ba;
}

void Lidar::setImuGyroBias(const Eigen::Vector3d& bg)
{
    bg_ = bg;
}

Eigen::Matrix3d Lidar::getImuRotation()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return T_f_w_.inverse().rotation_matrix() * T_l_b_.rotation_matrix();
}
    
Eigen::Vector3d Lidar::getImuPosition()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return (T_f_w_.inverse() * T_l_b_).translation(); 
}

Sophus::SE3 Lidar::getImuPose()
{
    boost::unique_lock<boost::mutex> lock(pose_mutex_);
    return T_f_w_.inverse() * T_l_b_;
}

void Lidar::setVelocity(const Eigen::Vector3d& vel)
{
    velocity_ = vel;
}

void Lidar::setVelocity()
{
    if(previous_lidar_)
        velocity_ = previous_lidar_->velocity_;
    else
        velocity_ << 0.0, 0.0, 0.0;
}

Eigen::Vector3d Lidar::getVelocity()
{
    return velocity_;
}

void Lidar::setNewBias(Eigen::Vector3d& ba, Eigen::Vector3d& bg)
{
    ba_ = ba;
    bg_ = bg;
    if(imu_from_last_lidar_)
        imu_from_last_lidar_->setNewBias(ba, bg);
    if(imu_from_last_keylidar_)
        imu_from_last_keylidar_->setNewBias(ba, bg);
}

} // namespace vilo
