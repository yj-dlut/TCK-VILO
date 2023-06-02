#ifndef LMVO_LASER_POINTS_H_
#define LMVO_LASER_POINTS_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
//#include <pcl/visualization/cloud_viewer.h>  
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/conditional_removal.h>    
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/feature.h>
//#include <pcl/segmentation/region_growing.h>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.h>

#include "vilo/camera.h"
#include "vilo/imu.h"

namespace vilo {

class IMUPriorConstraint;
typedef pcl::PointXYZI PcsType;
class Lidar: boost::noncopyable{


public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   
   static int frame_counter_;         
   static int keyFrameCounter_;

   int id_;
   int lidar_type_;
   int w_;
   int h_;
   //int w_cut_;
   //int h_cut_;
   float z_max_;
   float z_min_;
   double ts_;
   
   size_t N_;
   int labelCount_;

   float kw_,kh_;
   vilo::AbstractCamera* cam_;
   Eigen::Matrix<double,4,4> extrinsic_;
   Eigen::Matrix<double,3,4> velo_to_img_;
   pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_;
   //pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_on_img_;
   //cv::Mat laser_img_;
   cv::Mat xy_mapimg_;
   cv::Mat label_mat_;
   //cv::Mat normals_;
   //Eigen::Vector3d ground_vertical_;
   //Eigen::Vector3d ground_vertical_cam_;
   bool is_vertical_;
   cv::Mat oringin_lidar_img_map_;
   //std::vector<cv::Point2f> pixels_;
   //std::vector<int> vec_label_;
   //std::vector<std::pair<int, int> > neighbors_;

   //std::vector<Eigen::Vector3d> ground_pts_;
   //Eigen::Vector3d ground_center_;
   
   std::vector<bool> grid_occupancy_;
   int grid_w_num_, grid_h_num_; 
   int w_grid_size_, h_grid_size_;

   int Horizon_SCAN_;
   int N_SCAN_;
   float Vertical_Range_;
   float Vertical_Bottom_;
   float Scan_Period_;
   float ang_res_x_;
   float ang_res_y_;

   bool is_undist_pts_; 
   cv::Mat project_img_;
   cv::Mat id_img_;
   std::vector<int> scan_sid_vec_;
   std::vector<int> scan_eid_vec_;
   std::vector<float> diff_vec_; 
   //std::vector<Eigen::Vector2i> lidar_ftr_vec_; 
   pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_corners_;
   pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_surfs_;
   pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_less_corners_;
   pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_less_surfs_;
   pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_corner_;
   pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_surf_;
   Lidar* previous_lidar_;

   IMU* imu_from_last_lidar_;
   IMUPriorConstraint* prior_constraint_; //priori of imu
   
   IMU* imu_from_last_keylidar_; 
   Lidar* last_key_lidar_; 
   bool is_keylidar_;
   
   Sophus::SE3 T_b_l_, T_l_b_; 
   Sophus::SE3 T_f_w_;
   boost::mutex pose_mutex_;
   
   Eigen::Vector3d ba_, bg_;
   Eigen::Vector3d velocity_;
   int lba_id_;
   long align_vid_;
   bool is_opt_imu_; 

   Sophus::SE3 getLidarPose();    
   void setLidarPose(Sophus::SE3& Tfw);
   Eigen::Vector3d getImuAccBias();
   Eigen::Vector3d getImuGyroBias();
   void setImuAccBias(const Eigen::Vector3d& ba);
   void setImuGyroBias(const Eigen::Vector3d& bg);
   void setVelocity(const Eigen::Vector3d& vel);
   void setVelocity();
   Eigen::Vector3d getVelocity();
   Eigen::Matrix3d getImuRotation();
   Eigen::Vector3d getImuPosition();  
   Sophus::SE3 getImuPose();  
   void setNewBias(Eigen::Vector3d& ba, Eigen::Vector3d& bg);
   void setTimeStamp(double ts) {ts_ = ts;};
   double getTimeStamp() {return ts_;};  
   
   //20210219
   // cv::Mat image_;
   int pattern_25_[25][2] = {
		             {0,0},  {-1,0},  {1,0},   {0,-1},  {0,1}, 
                             {-1,1}, {-1,-1}, {1,1},   {1,-1},  {-2,0},  
                             {2,0},  {0,-2},  {0,2},   {-2,1},  {-2,-1},
                             {2,1},  {2,-1},  {-1,2},  {-1,-2}, {1,2},
                             {1,-2}, {-2,2},  {-2,-2}, {2,2},   {2,-2}
	                   };
 
   Lidar();
   Lidar(vilo::AbstractCamera* cam, int w, int h, pcl::PointCloud<pcl::PointXYZI>::Ptr p_cloud );
   Lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr p_cloud );
   ~Lidar();

   void projectLaserToImage(const Eigen::Matrix<double,3,4>& extrinsic,const cv::Mat& image);
   //void segmentLaserPts(pcl::PointCloud<pcl::PointXYZI>& cloud_);
   //void voxelFilter(pcl::PointCloud<PcsType>::Ptr& cloud);
   //void labelground(cv::Mat &laserimg , std::vector<cv::Point2f> &pixels);
   void labelground( );
   //void labelComponents(int row, int col);
   //bool checkgoodnbr(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1);
   //void computeNormals();
   
   void setCamParams(vilo::AbstractCamera* cam);
   void setLidarParams();
   void setTbl(Sophus::SE3& Tbl){T_b_l_ = Tbl; T_l_b_ = Tbl.inverse();};
   void registerPoints();
   void undistPoints(Sophus::SE3& dT, std::vector<double>& pts_ts_vec);
   size_t extractFeatures();
   bool comp (int i,int j) {return (diff_vec_[i] < diff_vec_[j]);}
   void transformToStart(pcl::PointXYZI& pi, pcl::PointXYZI& po, float SCAN_PERIOD, 
                             Eigen::Quaterniond& q_last_curr, Eigen::Vector3d& t_last_curr);
   void releaseMemory(int flag=0);

   int findFeatureDepth(std::vector<cv::Point2f>& pixels , 
                          std::vector<bool>& is_depth_found,
                           std::vector<Eigen::Vector3d>& intersections);

   bool calculatePlaneCorners(const std::vector<Eigen::Matrix<double,5,1> >& label_xy_pos, 
                               const Eigen::Vector2d& px,             
                                 Eigen::Matrix<double,5,1>& corner1,
                                  Eigen::Matrix<double,5,1>& corner2,
                                   Eigen::Matrix<double,5,1>& corner3);

   bool findBestNbrs(std::vector<Eigen::Matrix<double,5,1> >& nbrs,
                      std::vector<Eigen::Matrix<double,5,1> >& best_nbrs,
                       int px_label);

   bool checkPlanar(const Eigen::Vector3d& corner1,
                     const Eigen::Vector3d& corner2,
                       const Eigen::Vector3d& corner3);

   bool isPointInTriangle(const Eigen::Vector2d& A, 
                           const Eigen::Vector2d& B, 
                            const Eigen::Vector2d& C, 
                             const Eigen::Vector2d& P);
   bool pca(std::vector<Eigen::Matrix<double,5,1> >& best_nbrs , Eigen::Vector3d& pts_norm);

   bool findDepthOnce(Eigen::Vector2d pixel , Eigen::Vector3d& intersection);

   void findGroundFeatureDepth(std::vector<Eigen::Vector2d>& pixels , 
                                std::vector<int>& index_in_pxs,
                                 std::vector<Eigen::Vector3d>& intersections);
   
   void finish();

   bool setGridOccpuancy(const std::vector<Eigen::Vector2d>& pxs);
   void resetGrid();
   void chooseExtraLaserFeatrues(std::vector<Eigen::Vector3d>& features);

   

};


} // namespace vilo

#endif // vilo_LASER_POINTS_H_
