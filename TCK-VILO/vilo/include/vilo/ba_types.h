#ifndef VILO_BA_TYPES_H_
#define VILO_BA_TYPES_H_

#include <vilo/global.h>
#include <vilo/frame.h>
#include "vilo/camera.h"
#include "vilo/imu.h"

#include <g2o/core/robust_kernel.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>

#include <g2o/types/types_six_dof_expmap.h>
#include <g2o/types/types_seven_dof_expmap.h>
#include <g2o/types/se3quat.h>
#include <g2o/types/types_sba.h>



#include "vilo/vikit/math_utils.h"
#include "vilo/vikit/so3_functions.h"

using namespace g2o;

namespace vilo {


class Frame;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;



class ImuCamPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuCamPose(){}
    ImuCamPose(Frame* pF);
    ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, Frame* pKF);
    ImuCamPose(Sophus::SE3& T);
    ImuCamPose(Sophus::SE3& Twb, Sophus::SE3& Tbc, Sophus::SE3& Tbl, Frame* pF);

    void SetParam(const std::vector<Eigen::Matrix3d> &_Rcw, const std::vector<Eigen::Vector3d> &_tcw, const std::vector<Eigen::Matrix3d> &_Rbc,
                  const std::vector<Eigen::Vector3d> &_tbc);//, const double &_bf

    void Update(const double *pu); // update in the imu reference
    void UpdateW(const double *pu); // update in the world reference
    Eigen::Vector2d Project(const Eigen::Vector3d &Xw, int cam_idx=0) const; // Mono
    //Eigen::Vector3d ProjectStereo(const Eigen::Vector3d &Xw, int cam_idx=0) const; // Stereo
    bool isDepthPositive(const Eigen::Vector3d &Xw, int cam_idx=0) const;

public:
    // For IMU
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    // For set of cameras
    std::vector<Eigen::Matrix3d> Rcw;
    std::vector<Eigen::Vector3d> tcw;
    std::vector<Eigen::Matrix3d> Rcb, Rbc;
    std::vector<Eigen::Vector3d> tcb, tbc;
    // double bf; 
    std::vector<AbstractCamera*> pCamera;
    Eigen::Matrix3d Rlb, Rbl;
    Eigen::Vector3d tlb, tbl;

    // For posegraph 4DoF
    Eigen::Matrix3d Rwb0;
    Eigen::Matrix3d DR;

    int its;
};


// Optimizable parameters are IMU pose
class VertexPose : public g2o::BaseVertex<6,ImuCamPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose(){}

    VertexPose(Frame* pF){
        setEstimate(ImuCamPose(pF));
    }

    VertexPose(Eigen::Matrix3d& Rwc, Eigen::Vector3d& twc, Frame* pF){
        setEstimate(ImuCamPose(Rwc, twc, pF));
    }
    
    // for lidar-imu optimization
    VertexPose(Sophus::SE3& T){
        setEstimate(ImuCamPose(T));
    }

    // for lidar-visual-imu optimization
    VertexPose(Sophus::SE3& Twb, Sophus::SE3& Tbc, Sophus::SE3& Tbl, Frame* pF){
        setEstimate(ImuCamPose(Twb, Tbc, Tbl, pF));
    }

    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

class VertexVelocity : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity(){}
    VertexVelocity(Frame* pF);
    VertexVelocity(Eigen::Vector3d& velocity);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uv;
        uv << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uv);
    }
};


class VertexGyroBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias(){}
    VertexGyroBias(Frame* pF);
    VertexGyroBias(Eigen::Vector3d& bg);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d ubg;
        ubg << update_[0], update_[1], update_[2];
        setEstimate(estimate()+ubg);
    }
};


class VertexAccBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias(){}
    VertexAccBias(Frame* pF);
    VertexAccBias(Eigen::Vector3d& ba);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uba;
        uba << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uba);
    }
};


// Gravity direction vertex
class GDirection
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GDirection(){}
    GDirection(Eigen::Matrix3d pRwg): Rwg(pRwg){}

    void Update(const double *pu)
    {
        Rwg=Rwg*expSO3(pu[0],pu[1],0.0);
    }

    Eigen::Matrix3d Rwg, Rgw;

    int its;
};

class VertexGDir : public g2o::BaseVertex<2,GDirection>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGDir(){}
    VertexGDir(Eigen::Matrix3d pRwg){
        setEstimate(GDirection(pRwg));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

// scale vertex
class VertexScale : public g2o::BaseVertex<1,double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexScale(){
        setEstimate(1.0);
    }
    VertexScale(double ps){
        setEstimate(ps);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl(){
        setEstimate(1.0);
    }

    virtual void oplusImpl(const double *update_)
    {
        setEstimate(estimate()*exp(*update_));
    }
};

// 
class Rot
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Rot(){}
    Rot(Eigen::Matrix3d& R): R_(R){}

    void Update(const double *pu)
    {
        R_=R_*expSO3(pu[0],pu[1],pu[2]);
    }

    Eigen::Matrix3d R_;

};

class VertexRot : public g2o::BaseVertex<3,Rot>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexRot(){}
    VertexRot(Eigen::Matrix3d R){
        setEstimate(Rot(R));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

// Edge inertial whre gravity is included as optimizable variable and it is not supposed to be pointing in -z axis, as well as scale
class EdgeInertialGS : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // EdgeInertialGS(IMU::Preintegrated* pInt);
    EdgeInertialGS(vilo::IMU* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    // IMU::Preintegrated* mpInt;
    IMU* mpInt;
    const double dt;
    Eigen::Vector3d g, gI;

    Eigen::Matrix<double,27,27> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,27> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        J.block<9,2>(0,24) = _jacobianOplus[6];
        J.block<9,1>(0,26) = _jacobianOplus[7];

        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,27,27> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,27> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        J.block<9,1>(0,8) = _jacobianOplus[7];
        J.block<9,3>(0,9) = _jacobianOplus[1];
        J.block<9,3>(0,12) = _jacobianOplus[5];
        J.block<9,6>(0,15) = _jacobianOplus[0];
        J.block<9,6>(0,21) = _jacobianOplus[4];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,9,9> GetHessian3(){
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        J.block<9,1>(0,8) = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }



    Eigen::Matrix<double,1,1> GetHessianScale(){
        linearizeOplus();
        Eigen::Matrix<double,9,1> J = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,3,3> GetHessianBiasGyro(){
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[2];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,3,3> GetHessianBiasAcc(){
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[3];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,2,2> GetHessianGDir(){
        linearizeOplus();
        Eigen::Matrix<double,9,2> J = _jacobianOplus[6];
        return J.transpose()*information()*J;
    }
};

// Edge inertial whre gravity is included as optimizable variable and it is not supposed to be pointing in -z axis, as well as scale
class EdgeInertialG : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInertialG(vilo::IMU* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    // IMU::Preintegrated* mpInt;
    IMU* mpInt;
    const double dt;
    Eigen::Vector3d g, gI;

    Eigen::Matrix<double,26,26> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,26> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        J.block<9,2>(0,24) = _jacobianOplus[6];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,26,26> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,26> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        J.block<9,3>(0,8) = _jacobianOplus[1];
        J.block<9,3>(0,11) = _jacobianOplus[5];
        J.block<9,6>(0,14) = _jacobianOplus[0];
        J.block<9,6>(0,20) = _jacobianOplus[4];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,8,8> GetHessian3(){
        linearizeOplus();
        Eigen::Matrix<double,9,8> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,3,3> GetHessianBiasGyro(){
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[2];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,3,3> GetHessianBiasAcc(){
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[3];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,2,2> GetHessianGDir(){
        linearizeOplus();
        Eigen::Matrix<double,9,2> J = _jacobianOplus[6];
        return J.transpose()*information()*J;
    }
};

// Edge inertial
class EdgeInertial : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInertial(vilo::IMU* pImu);
    EdgeInertial(vilo::IMU* pImu, bool is_verbose);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double,24,24> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,24> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,18,18> GetHessianNoPose1(){
        linearizeOplus();
        Eigen::Matrix<double,9,18> J;
        J.block<9,3>(0,0) = _jacobianOplus[1];
        J.block<9,3>(0,3) = _jacobianOplus[2];
        J.block<9,3>(0,6) = _jacobianOplus[3];
        J.block<9,6>(0,9) = _jacobianOplus[4];
        J.block<9,3>(0,15) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,9,9> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,6>(0,0) = _jacobianOplus[4];
        J.block<9,3>(0,6) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    // IMU::Preintegrated* mpInt;
    vilo::IMU* mpImu;
    const double dt;
    Eigen::Vector3d g;
    bool verbose;
};

// Priors for biases
// class EdgeScale : public g2o::BaseUnaryEdge<9,Vector9d,VertexScale>
class EdgeScale : public g2o::BaseUnaryEdge<6,Vector6d,VertexScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeScale(vilo::IMU* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    Sophus::SE3 pos1_, pos2_;
    Eigen::Vector3d vel1_, vel2_;
    Eigen::Vector3d G_;
    double dt_;
    IMU* mpInt;

    void setPoses(Sophus::SE3& pos1, Sophus::SE3& pos2)
    {
        pos1_ = pos1;
        pos2_ = pos2;
    }

    void setVels(Eigen::Vector3d& vel1, Eigen::Vector3d& vel2)
    {
        vel1_ = vel1;
        vel2_ = vel2;
    }
    void setGravity(Eigen::Vector3d& G)
    {
        G_ = G;
    }

};


// Priors for biases
class EdgePriorAcc : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorAcc(const Eigen::Vector3d& bprior_):bprior(bprior_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[0]);
        _error = bprior - VA->estimate();
    }
    virtual void linearizeOplus();

    Eigen::Matrix<double,3,3> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

    const Eigen::Vector3d bprior;
};


class EdgePriorGyro : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorGyro(const Eigen::Vector3d& bprior_):bprior(bprior_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[0]);
        _error = bprior - VG->estimate();
    }
    virtual void linearizeOplus();

    Eigen::Matrix<double,3,3> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

    const Eigen::Vector3d bprior;
};


class EdgeGyroRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexGyroBias,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGyroRW(){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[0]);
        const VertexGyroBias* VG2= static_cast<const VertexGyroBias*>(_vertices[1]);
        _error = VG2->estimate()-VG1->estimate();
    }

    virtual void linearizeOplus(){
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
    }

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,3,6> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,3>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

    Eigen::Matrix3d GetHessian2(){
        linearizeOplus();
        return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
    }
};


class EdgeAccRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexAccBias,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeAccRW(){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[0]);
        const VertexAccBias* VA2= static_cast<const VertexAccBias*>(_vertices[1]);
        _error = VA2->estimate()-VA1->estimate();
    }

    virtual void linearizeOplus(){
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
    }

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,3,6> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,3>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

    Eigen::Matrix3d GetHessian2(){
        linearizeOplus();
        return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
    }
};

class IMUPriorConstraint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUPriorConstraint(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_,
                       const Eigen::Vector3d &bg_, const Eigen::Vector3d &ba_, const Matrix15d &H_):
                       Rwb(Rwb_), twb(twb_), vwb(vwb_), bg(bg_), ba(ba_), H(H_)
    {
        H = (H+H)/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15> > es(H);
        Eigen::Matrix<double,15,1> eigs = es.eigenvalues();
        for(int i=0;i<15;i++)
            if(eigs[i]<1e-12)
                eigs[i]=0;
        H = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    }

    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Vector3d vwb;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Matrix15d H;
};

class EdgePriorPoseImu : public g2o::BaseMultiEdge<15,Vector15d>
{
public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgePriorPoseImu(IMUPriorConstraint* c);

        virtual bool read(std::istream& is){return false;}
        virtual bool write(std::ostream& os) const{return false;}

        void computeError();
        virtual void linearizeOplus();

        Eigen::Matrix<double,15,15> GetHessian(){
            linearizeOplus();
            Eigen::Matrix<double,15,15> J;
            J.block<15,6>(0,0) = _jacobianOplus[0];
            J.block<15,3>(0,6) = _jacobianOplus[1];
            J.block<15,3>(0,9) = _jacobianOplus[2];
            J.block<15,3>(0,12) = _jacobianOplus[3];
            return J.transpose()*information()*J;
        }

        Eigen::Matrix<double,9,9> GetHessianNoPose(){
            linearizeOplus();
            Eigen::Matrix<double,15,9> J;
            J.block<15,3>(0,0) = _jacobianOplus[1];
            J.block<15,3>(0,3) = _jacobianOplus[2];
            J.block<15,3>(0,6) = _jacobianOplus[3];
            return J.transpose()*information()*J;
        }
        Eigen::Matrix3d Rwb;
        Eigen::Vector3d twb, vwb;
        Eigen::Vector3d bg, ba;
};

class EdgeCornerOnlyPose : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeCornerOnlyPose(const Eigen::Vector3d& Xw, int cam_idx=0):_Xw(Xw),
        _cam_idx(cam_idx){}
    EdgeCornerOnlyPose(bool verbose, const Eigen::Vector3d& Xw, int cam_idx=0):_Xw(Xw),
        _cam_idx(cam_idx){_verbose = verbose;}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - _Fxy * VPose->estimate().Project(_Xw,_cam_idx);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

    void setCameraParams(Vector2d fxy)
    {
        _Fxy(0,0) = fxy[0]; _Fxy(0,1) = 0;
        _Fxy(1,0) = 0; _Fxy(1,1) = fxy[1];
    }

public:
    const Eigen::Vector3d _Xw;
    const int _cam_idx;
    Eigen::Matrix2d _Fxy;
    bool _verbose;
};

class EdgeEdgeletOnlyPose : public g2o::BaseUnaryEdge<1,double,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeEdgeletOnlyPose(const Eigen::Vector3d& Xw, int cam_idx=0):_Xw(Xw),
        _cam_idx(cam_idx){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        const double obs = _measurement;
        _error(0,0) = obs - _normal.transpose() * _Fxy * VPose->estimate().Project(_Xw,_cam_idx);
        // _error(0,0) = obs - _normal.transpose() * VPose->estimate().Project(_Xw,_cam_idx);

    }

    virtual void linearizeOplus();

    // bool isDepthPositive()
    // {
    //     const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
    //     return VPose->estimate().isDepthPositive(Xw,cam_idx);
    // }

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

    void setTargetNormal(Eigen::Vector2d n) {_normal = n;}

    void setCameraParams(Vector2d fxy)
    {
        _Fxy(0,0) = fxy[0]; _Fxy(0,1) = 0;
        _Fxy(1,0) = 0; _Fxy(1,1) = fxy[1];
    }

public:
    const Eigen::Vector3d _Xw;
    Eigen::Vector2d _normal;
    const int _cam_idx;
    Eigen::Matrix2d _Fxy;
};

class VertexPointID : public BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPointID() : BaseVertex<1, double>()
    {}

    virtual bool read(std::istream& is) { return true; }
    virtual bool write(std::ostream& os) const { return true; }

    virtual void setToOriginImpl() {
        _estimate = 0;
    }

    virtual void oplusImpl(const double* update) {
        _estimate += (*update);
    }
};


class EdgeIdistOnlyCorner: public g2o::BaseUnaryEdge<2, Vector2d, VertexPointID>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeIdistOnlyCorner(Sophus::SE3& Th, Sophus::SE3& Tt)
    {
        Th_ = Th;
        Tt_ = Th;
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}
    void computeError();
    virtual void linearizeOplus();

    void setCameraParams(Vector2d fxy)
    {
        _Fxy(0,0) = fxy[0]; _Fxy(0,1) = 0;
        _Fxy(1,0) = 0; _Fxy(1,1) = fxy[1];
    }
    void setHostBearing(Vector3d f) {_fH = f;}

    // CameraParameters * _cam;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return vilo::project2d(trans_xyz);
    }

public:
    Sophus::SE3 Th_, Tt_;
    Eigen::Vector3d _fH;
    Eigen::Matrix2d _Fxy;
};

class EdgeIdistOnlyEdgeLet : public g2o::BaseUnaryEdge<1, double, VertexPointID>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeIdistOnlyEdgeLet(Sophus::SE3& Th, Sophus::SE3& Tt)
    {
        Th_ = Th;
        Tt_ = Th;
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}
    void computeError();
    virtual void linearizeOplus();

    void setHostBearing(Vector3d f) {_fH = f;}
    void setTargetNormal(Vector2d n) {_normal = n;}
    void setCameraParams(Vector2d fxy)
    {
        _Fxy(0,0) = fxy[0]; _Fxy(0,1) = 0;
        _Fxy(1,0) = 0; _Fxy(1,1) = fxy[1];
    }

    // CameraParameters * _cam;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return vilo::project2d(trans_xyz);
    }

public:
    Vector2d _normal;
    Vector3d _fH;
    Eigen::Matrix2d _Fxy;
    Sophus::SE3 Th_, Tt_;
};

class EdgeIdistCorner : public BaseMultiEdge<2, Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeIdistCorner()
    {
        // _cam = 0;
        // resizeParameters(1);
        // installParameter(_cam, 0);
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}

    void computeError();
    virtual void linearizeOplus();

    void setHostBearing(Vector3d f) {_fH = f;}
    void setCameraParams(Vector2d fxy)
    {
        _Fxy(0,0) = fxy[0]; _Fxy(0,1) = 0;
        _Fxy(1,0) = 0; _Fxy(1,1) = fxy[1];
    }

    // CameraParameters * _cam;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return vilo::project2d(trans_xyz);
    }

public:
    Eigen::Vector3d _fH;
    // double _fx, _fy;
    Eigen::Matrix2d _Fxy;
};

class EdgeIdistEdgeLet : public BaseMultiEdge<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeIdistEdgeLet()
    {
        // _cam = 0;
        // resizeParameters(1);
        // installParameter(_cam, 0);
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}
    void computeError();
    virtual void linearizeOplus();

    void setHostBearing(Vector3d f) {_fH = f;}
    void setTargetNormal(Vector2d n) {_normal = n;}
    void setCameraParams(Vector2d fxy)
    {
        _Fxy(0,0) = fxy[0]; _Fxy(0,1) = 0;
        _Fxy(1,0) = 0; _Fxy(1,1) = fxy[1];
    }

    // CameraParameters * _cam;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return vilo::project2d(trans_xyz);
    }

public:
    Vector2d _normal;
    Vector3d _fH;
    //double _fx, _fy;
    Eigen::Matrix2d _Fxy;
};

class EdgeLidarPointPlane : public g2o::BaseUnaryEdge<1,double,VertexPose>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeLidarPointPlane(){}
    EdgeLidarPointPlane(Eigen::Vector3d& nml, double d)
    {
        _nml = nml;
        _d = d;
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}
    void computeError();
    virtual void linearizeOplus();
    // void setPreviousPose(Sophus::SE3& T1){_T1 = T1;};
    void setPlaneNormal( Eigen::Vector3d& nml){_nml = nml;};
    void setPlaneDist(double d){_d = d;}
    void setPoint(Eigen::Vector3d& measurement){_measurement = measurement;}
    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }
    public:
    // Sophus::SE3 _T1;
    Eigen::Vector3d _nml;
    Eigen::Vector3d _measurement;
    double _d;
};

class EdgeLidarPointLine : public g2o::BaseUnaryEdge<1,double,VertexPose>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeLidarPointLine(){}
    EdgeLidarPointLine(Eigen::Vector3d& p1, Eigen::Vector3d& p2)
    {
        _p1 = p1;
        _p2 = p2;
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}
    void computeError();
    virtual void linearizeOplus();
    void setPreviousPose(Sophus::SE3& T1){_T1 = T1;};
    void setPoint(Eigen::Vector3d& measurement){_measurement = measurement;}

    public:
    Sophus::SE3 _T1;
    Eigen::Vector3d _p1, _p2;
    Eigen::Vector3d _measurement;
};

class  EdgeSim3XYZ: public  g2o::BaseUnaryEdge<3, Vector3d, g2o::VertexSim3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSim3XYZ( Eigen::Vector3d& pl, Eigen::Vector3d& pf)
  {
      pf_ = pf;
      pl_ = pl;
  }

  virtual bool read(std::istream& is) {return true;}
  virtual bool write(std::ostream& os) const {return true;}

  void computeError()  
  {
    const VertexSim3Expmap* v1 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
    _error = pl_ - v1->estimate().map(pf_);
  }

//   virtual void linearizeOplus(){}

  Eigen::Vector3d pf_, pl_;
};

class  EdgeRot: public  g2o::BaseUnaryEdge<3, Vector3d, VertexRot>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeRot( Eigen::Matrix3d& Rl, Eigen::Matrix3d& Rf)
  {
      Rf_ = Rf;
      Rl_ = Rl;
  }

  virtual bool read(std::istream& is) {return true;}
  virtual bool write(std::ostream& os) const {return true;}

  void computeError()  
  {
      const VertexRot* Rlf = static_cast<const VertexRot*>(_vertices[0]);
      _error = logSO3(Rf_.transpose()*Rl_*Rlf->estimate().R_);
   }

  virtual void linearizeOplus()
  {
      const VertexRot* Rlf = static_cast<const VertexRot*>(_vertices[0]);
      Eigen::Vector3d er = logSO3(Rf_.transpose()*Rl_*Rlf->estimate().R_);
      Eigen::Matrix3d invJr = inverseRightJacobianSO3(er);
      _jacobianOplusXi = invJr;
  }

  Eigen::Matrix3d Rf_, Rl_;
};

class EdgeSim3: public g2o::BaseUnaryEdge<6, Vector6d, g2o::VertexSim3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSim3( Sophus::SE3& Tl, Sophus::SE3& Tf)
  {
      Tf_ = Tf;
      Tl_ = Tl;
  }

  virtual bool read(std::istream& is) {return true;}
  virtual bool write(std::ostream& os) const {return true;}

  void computeError()  
  {
    const VertexSim3Expmap* v1 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
    Eigen::Vector3d er, ep;
    Eigen::Matrix3d Rl, Rf;
    Eigen::Vector3d tl, tf;
    Rl = Tl_.rotation_matrix();
    Rf = Tf_.rotation_matrix();
    tl = Tl_.translation();
    tf = Tf_.translation();
    ep = tl - v1->estimate().map(tf);
    er = logSO3(Rf.transpose() * Rl * v1->estimate().rotation());
    _error << er, ep;
  }

//   virtual void linearizeOplus(){}

  Sophus::SE3 Tf_, Tl_;
};

} // namespace vilo

#endif // VILO_BA_TYPES_H_
