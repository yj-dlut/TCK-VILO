#include <vilo/imu.h>
#include <ctime>
#include <sys/time.h>
#include <stdlib.h>

#include <eigen3/Eigen/Dense>
#include<opencv2/core/eigen.hpp>

#include <fstream>
#include <string>
#include <sstream>

#include "vilo/vikit/so3_functions.h"

using namespace std;
namespace vilo {


IMU::IMU(float& freq, float& na, float& ng, float& naw, float& ngw)   
{
    delta_t_ = 0;
    delta_p_.setZero();
    delta_v_.setZero();
    delta_q_.setIdentity();
    ba_.setZero();
    bg_.setZero();
    new_ba_.setZero();
    new_bg_.setZero();
    db_.setZero();
    cov_mat_.setZero();
    noise_.setIdentity();
    bias_walk_.setIdentity();
    J_p_ba_.setZero(); 
    J_p_bg_.setZero(); 
    J_q_bg_.setZero(); 
    J_v_bg_.setZero(); 
    J_v_ba_.setZero();
    
    acc_vec_.clear();
    gyro_vec_.clear();
    dt_vec_.clear();
    dp_vec_.clear();
    dv_vec_.clear();
    dq_vec_.clear();

    double sfreq = sqrt(freq);
    double gyroscope_noise_density = ng; 
    double accelerometer_noise_density = na;
    double gyroscope_random_walk = ngw;
    double accelerometer_random_walk = naw;

    double gyroscope_noise = gyroscope_noise_density * sfreq;
    double accelerometer_noise = accelerometer_noise_density * sfreq;
    double gyroscope_bias = gyroscope_random_walk / sfreq;
    double accelerometer_bias = accelerometer_random_walk / sfreq;
    
    noise_(0,0) = gyroscope_noise * gyroscope_noise;
    noise_(1,1) = gyroscope_noise * gyroscope_noise;
    noise_(2,2) = gyroscope_noise * gyroscope_noise;
    noise_(3,3) = accelerometer_noise * accelerometer_noise;
    noise_(4,4) = accelerometer_noise * accelerometer_noise;
    noise_(5,5) = accelerometer_noise * accelerometer_noise;

    bias_walk_(0,0) = gyroscope_bias * gyroscope_bias;
    bias_walk_(1,1) = gyroscope_bias * gyroscope_bias;
    bias_walk_(2,2) = gyroscope_bias * gyroscope_bias;
    bias_walk_(3,3) = accelerometer_bias * accelerometer_bias;
    bias_walk_(4,4) = accelerometer_bias * accelerometer_bias;
    bias_walk_(5,5) = accelerometer_bias * accelerometer_bias;

}

IMU::IMU(float& freq, float& na, float& ng, float& naw, float& ngw,
         Eigen::Vector3d ba, Eigen::Vector3d bg):ba_(ba),bg_(bg)   
{
    delta_t_ = 0;
    delta_p_.setZero();
    delta_v_.setZero();
    delta_q_.setIdentity();
    
    new_ba_.setZero();
    new_bg_.setZero();
    db_.setZero();
    cov_mat_.setZero();
    noise_.setIdentity();
    bias_walk_.setIdentity();
    J_p_ba_.setZero(); 
    J_p_bg_.setZero(); 
    J_q_bg_.setZero(); 
    J_v_bg_.setZero(); 
    J_v_ba_.setZero();
    
    acc_vec_.clear();
    gyro_vec_.clear();
    dt_vec_.clear();
    dp_vec_.clear();
    dv_vec_.clear();
    dq_vec_.clear();

    double sfreq = sqrt(freq);
    double gyroscope_noise_density = ng; 
    double accelerometer_noise_density = na; 
    double gyroscope_random_walk = ngw; 
    double accelerometer_random_walk = naw;

    double gyroscope_noise = gyroscope_noise_density * sfreq;
    double gyroscope_bias = gyroscope_random_walk / sfreq;
    double accelerometer_noise = accelerometer_noise_density * sfreq;
    double accelerometer_bias = accelerometer_random_walk / sfreq;
    noise_(0,0) = gyroscope_noise * gyroscope_noise;
    noise_(1,1) = gyroscope_noise * gyroscope_noise;
    noise_(2,2) = gyroscope_noise * gyroscope_noise;
    noise_(3,3) = accelerometer_noise * accelerometer_noise;
    noise_(4,4) = accelerometer_noise * accelerometer_noise;
    noise_(5,5) = accelerometer_noise * accelerometer_noise;

    bias_walk_(0,0) = gyroscope_bias * gyroscope_bias;
    bias_walk_(1,1) = gyroscope_bias * gyroscope_bias;
    bias_walk_(2,2) = gyroscope_bias * gyroscope_bias;
    bias_walk_(3,3) = accelerometer_bias * accelerometer_bias;
    bias_walk_(4,4) = accelerometer_bias * accelerometer_bias;
    bias_walk_(5,5) = accelerometer_bias * accelerometer_bias;
}

IMU::~IMU()
{}

void IMU::reset()   
{
    delta_t_ = 0;
    delta_p_.setZero();
    delta_v_.setZero();
    delta_q_.setIdentity();
    cov_mat_.setZero();
    J_p_ba_.setZero(); 
    J_p_bg_.setZero(); 
    J_q_bg_.setZero(); 
    J_v_bg_.setZero(); 
    J_v_ba_.setZero();

    acc_vec_.clear();
    gyro_vec_.clear();
    dt_vec_.clear();
    dp_vec_.clear();
    dv_vec_.clear();
    dq_vec_.clear();
}

void IMU::integrateDeltaR(Eigen::Matrix3d& dR, Eigen::Matrix3d& rightJac, 
               const Eigen::Vector3d& gyro, const Eigen::Vector3d& bg, double t)
{
    const double eps = 1e-4;

    double x = (gyro[0] - bg[0])*t;
    double y = (gyro[1] - bg[1])*t;
    double z = (gyro[2] - bg[2])*t;

    double d2 = x*x+y*y+z*z;
    double d = sqrt(d2);

    Eigen::Matrix3d skew_gyro;
    skew_gyro<< 0.0,  -z,   y,
                  z, 0.0,  -x,
                 -y,   x, 0.0;
                
    if(d<eps)
    {
        dR = Eigen::MatrixXd::Identity(3,3) + skew_gyro;
        rightJac = Eigen::MatrixXd::Identity(3,3);
    }
    else
    {
        dR = Eigen::MatrixXd::Identity(3,3) + skew_gyro*sin(d)/d + skew_gyro*skew_gyro*(1.0f-cos(d))/d2;
        rightJac = Eigen::MatrixXd::Identity(3,3) - skew_gyro*(1.0f-cos(d))/d2 + skew_gyro*skew_gyro*(d-sin(d))/(d2*d);
    }
}

void IMU::addImuPoint(const Eigen::Vector3d& acc_measurement, const Eigen::Vector3d& gyro_measurement,
                      const Eigen::Vector3d& ba, const Eigen::Vector3d& bg, const double& dt)
{
    Eigen::Vector3d avg_acc, avg_gyro;
    acc_vec_.push_back(acc_measurement);
    gyro_vec_.push_back(gyro_measurement);
    dt_vec_.push_back(dt);

    Eigen::Matrix<double,9,9> A; A.setIdentity();
    Eigen::Matrix<double,9,6> B; B.setZero();
    Eigen::Vector3d acc = acc_measurement - ba;
    Eigen::Vector3d gyro = gyro_measurement - bg;

    avg_acc = (delta_t_*avg_acc + delta_q_*acc*dt)/(delta_t_+dt);
    avg_gyro = (delta_t_*avg_gyro + gyro*dt)/(delta_t_+dt);

    delta_p_ = delta_p_ + delta_v_*dt + 0.5*delta_q_*acc*dt*dt;
    delta_v_ = delta_v_ + delta_q_*acc*dt;
    Eigen::Matrix3d skew_acc; skew_acc<<    0.0, -acc[2],  acc[1],
                                         acc[2],     0.0, -acc[0],
                                        -acc[1],  acc[0],     0.0;

    A.block(3,0,3,3) = -delta_q_*dt*skew_acc;
    A.block(6,0,3,3) = -0.5f*delta_q_*dt*dt*skew_acc;
    A.block(6,3,3,3) = Eigen::MatrixXd::Identity(3,3)*dt;
    B.block(3,3,3,3) = delta_q_*dt;
    B.block(6,3,3,3) = 0.5f*delta_q_*dt*dt;
 
    J_p_ba_ = J_p_ba_ + J_v_ba_*dt -0.5f*delta_q_*dt*dt;
    J_p_bg_ = J_p_bg_ + J_v_bg_*dt -0.5f*delta_q_*dt*dt*skew_acc*J_q_bg_;
    J_v_ba_ = J_v_ba_ - delta_q_*dt;
    J_v_bg_ = J_v_bg_ - delta_q_*dt*skew_acc*J_q_bg_;

    Eigen::Matrix3d dR, right_jac;
    integrateDeltaR(dR, right_jac, gyro_measurement, bg, dt);

    delta_q_ = vilo::normalizeRotation(delta_q_ * dR);
    Eigen::Matrix3d U,V;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(delta_q_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = svd.matrixU();
    V = svd.matrixV();
    delta_q_ = U * V.transpose();

    A.block(0,0,3,3) = dR.transpose();
    B.block(0,0,3,3) = right_jac*dt;

    cov_mat_.block(0,0,9,9) = A*cov_mat_.block(0,0,9,9)*A.transpose() + B*noise_*B.transpose();
    cov_mat_.block(9,9,6,6) = cov_mat_.block(9,9,6,6) + bias_walk_;

    J_q_bg_ = dR.transpose()*J_q_bg_ - right_jac*dt;

    delta_t_ += dt;

    dp_vec_.push_back(delta_p_);
    dv_vec_.push_back(delta_v_);
    dq_vec_.push_back(delta_q_);
}

Eigen::Matrix3d IMU::getDeltaRotation(const Eigen::Vector3d& bg)
{   
    Eigen::Vector3d dbg = bg - bg_;
    return normalizeRotation(delta_q_*expSO3(J_q_bg_*dbg));
}

Eigen::Vector3d IMU::getDeltaVelocity(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg)
{
    Eigen::Vector3d dba, dbg;
    dba = ba - ba_;
    dbg = bg - bg_;
    return delta_v_ + J_v_bg_*dbg + J_v_ba_*dba;
}

Eigen::Vector3d IMU::getDeltaPosition(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg)
{
    Eigen::Vector3d dba, dbg;
    dba = ba - ba_;
    dbg = bg - bg_;
    return delta_p_ + J_p_bg_*dbg + J_p_ba_*dba;
}

Eigen::Matrix3d IMU::getUpdateDeltaRotation()
{   
    return normalizeRotation(delta_q_*expSO3(J_q_bg_*db_.tail(3)));
}

Eigen::Vector3d IMU::getUpdateDeltaVelocity()
{
    return delta_v_ + J_v_bg_*db_.tail(3) + J_v_ba_*db_.head(3);
}

Eigen::Vector3d IMU::getUpdateDeltaPosition()
{
    return delta_p_ + J_p_bg_*db_.tail(3) + J_p_ba_*db_.head(3);
}


Eigen::Matrix<double,6,1> IMU::getDeltaBias(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg)
{
    db_.head(3) = ba - ba_;
    db_.tail(3) = bg - bg_;
    return db_;
}
Eigen::Matrix<double,6,1> IMU::getDeltaBias()
{
    return db_;
}

void IMU::setNewBias(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg)
{
    new_ba_ = ba;
    new_bg_ = bg;
    db_.head(3) = ba - ba_;
    db_.tail(3) = bg - bg_;
}

void IMU::reintegrate()
{
    std::vector<double> dt_vec = dt_vec_;
    std::vector<Eigen::Vector3d> acc_vec = acc_vec_;
    std::vector<Eigen::Vector3d> gyro_vec = gyro_vec_;
    
    reset();
    ba_ = new_ba_;
    bg_ = new_bg_;
    if( (dt_vec.size() != acc_vec.size()) || (dt_vec.size() != gyro_vec.size()) || (acc_vec.size() != gyro_vec.size()))
    {
        cout<<"reintegrate failed !"<<endl;
        return;
    }
    const int N = dt_vec.size();
    for(int i=0;i<N;i++)
    {
        addImuPoint(acc_vec[i], gyro_vec[i], new_ba_, new_bg_, dt_vec[i]);
    }
}

} // namespace vilo
