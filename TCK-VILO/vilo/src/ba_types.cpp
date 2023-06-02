#include <boost/thread.hpp>

#include "vilo/ba_types.h"
#include "vilo/vikit/math_utils.h"

#define SCHUR_TRICK 1

namespace vilo {


ImuCamPose::ImuCamPose(Frame *pF):its(0)
{
    twb = pF->getImuPosition();
    Rwb = pF->getImuRotation();
    int num_cams = 1;
 
    tcw.resize(num_cams);
    Rcw.resize(num_cams);
    tcb.resize(num_cams);
    Rcb.resize(num_cams);
    Rbc.resize(num_cams);
    tbc.resize(num_cams);
    pCamera.resize(num_cams);

    Sophus::SE3 Tfw = pF->getFramePose();
    tcw[0] = Tfw.translation();
    Rcw[0] = Tfw.rotation_matrix();
    tcb[0] = pF->T_c_b_.translation();
    Rcb[0] = pF->T_c_b_.rotation_matrix();
    Rbc[0] = pF->T_b_c_.rotation_matrix();
    tbc[0] = pF->T_b_c_.translation();
    pCamera[0] = pF->cam_;

    Rwb0 = Rwb;
    DR.setIdentity();
}

ImuCamPose::ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, Frame* pF): its(0)
{
    tcw.resize(1);
    Rcw.resize(1);
    tcb.resize(1);
    Rcb.resize(1);
    Rbc.resize(1);
    tbc.resize(1);
    pCamera.resize(1);

    tcb[0] = pF->T_c_b_.translation();
    Rcb[0] = pF->T_c_b_.rotation_matrix();
    Rbc[0] = Rcb[0].transpose();
    tbc[0] = pF->T_b_c_.translation();
    twb = _Rwc*tcb[0]+_twc;
    Rwb = _Rwc*Rcb[0];
    Rcw[0] = _Rwc.transpose();
    tcw[0] = -Rcw[0]*_twc;
    pCamera[0] = pF->cam_;
   
}

ImuCamPose::ImuCamPose(Sophus::SE3& T)
{
    Rwb = T.rotation_matrix();
    twb = T.translation();
}

ImuCamPose::ImuCamPose(Sophus::SE3& Twb, Sophus::SE3& Tbc, Sophus::SE3& Tbl, Frame* pF)
{
    tcw.resize(1);
    Rcw.resize(1);
    tcb.resize(1);
    Rcb.resize(1);
    Rbc.resize(1);
    tbc.resize(1);
    pCamera.resize(1);

    Rcb[0] = Tbc.inverse().rotation_matrix();
    Rbc[0] = Tbc.rotation_matrix();
    tcb[0] = Tbc.inverse().translation();
    tbc[0] = Tbc.translation();
    twb = Twb.translation();
    Rwb = Twb.rotation_matrix();
    Rcw[0] = (Twb * Tbc).inverse().rotation_matrix();
    tcw[0] = (Twb * Tbc).inverse().translation();
    pCamera[0] = pF->cam_;
}

void ImuCamPose::SetParam(const std::vector<Eigen::Matrix3d> &_Rcw, const std::vector<Eigen::Vector3d> &_tcw, const std::vector<Eigen::Matrix3d> &_Rbc,
              const std::vector<Eigen::Vector3d> &_tbc)//, const double &_bf
{
    Rbc = _Rbc;
    tbc = _tbc;
    Rcw = _Rcw;
    tcw = _tcw;
    const int num_cams = Rbc.size();
    Rcb.resize(num_cams);
    tcb.resize(num_cams);

    for(size_t i=0; i<tcb.size(); i++)
    {
        Rcb[i] = Rbc[i].transpose();
        tcb[i] = -Rcb[i]*tbc[i];
    }
    Rwb = Rcw[0].transpose()*Rcb[0];
    twb = Rcw[0].transpose()*(tcb[0]-tcw[0]);

}

Eigen::Vector2d ImuCamPose::Project(const Eigen::Vector3d &Xw, int cam_idx) const
{
    Eigen::Vector3d Xc = Rcw[cam_idx]*Xw+tcw[cam_idx];
    return vilo::project2d(Xc);
}

bool ImuCamPose::isDepthPositive(const Eigen::Vector3d &Xw, int cam_idx) const
{
    return (Rcw[cam_idx].row(2)*Xw+tcw[cam_idx](2))>0.0;
}

void ImuCamPose::Update(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];
    
    // Update body pose
    twb += Rwb*ut;
    Rwb = Rwb*expSO3(ur);

    // Normalize rotation after 5 updates
    its++;
    if(its>=3)
    {
        normalizeRotation(Rwb);
        its=0;
    }

    // Update camera poses
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw*twb;

    for(size_t i=0; i<pCamera.size(); i++)
    {
        Rcw[i] = Rcb[i]*Rbw;
        tcw[i] = Rcb[i]*tbw+tcb[i];
    }

}

void ImuCamPose::UpdateW(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];


    const Eigen::Matrix3d delta_q_ = expSO3(ur);
    DR = delta_q_*DR;
    Rwb = DR*Rwb0;
    // Update body pose
    twb += ut;

    // Normalize rotation after 5 updates
    its++;
    if(its>=5)
    {
        DR(0,2)=0.0;
        DR(1,2)=0.0;
        DR(2,0)=0.0;
        DR(2,1)=0.0;
        normalizeRotation(DR);
        its=0;
    }

    // Update camera pose
    const Eigen::Matrix3d Rbw = Rwb.transpose();
    const Eigen::Vector3d tbw = -Rbw*twb;

    for(size_t i=0; i<pCamera.size(); i++)
    {
        Rcw[i] = Rcb[i]*Rbw;
        tcw[i] = Rcb[i]*tbw+tcb[i];
    }
}

bool VertexPose::read(std::istream& is)
{
    std::vector<Eigen::Matrix<double,3,3> > Rcw;
    std::vector<Eigen::Matrix<double,3,1> > tcw;
    std::vector<Eigen::Matrix<double,3,3> > Rbc;
    std::vector<Eigen::Matrix<double,3,1> > tbc;

    const int num_cams = _estimate.Rbc.size();
    for(int idx = 0; idx<num_cams; idx++)
    {
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                is >> Rcw[idx](i,j);
        }
        for (int i=0; i<3; i++){
            is >> tcw[idx](i);
        }

        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                is >> Rbc[idx](i,j);
        }
        for (int i=0; i<3; i++){
            is >> tbc[idx](i);
        }

    }

    _estimate.SetParam(Rcw,tcw,Rbc,tbc);
    updateCache();
    
    return true;
}

bool VertexPose::write(std::ostream& os) const
{
    std::vector<Eigen::Matrix<double,3,3> > Rcw = _estimate.Rcw;
    std::vector<Eigen::Matrix<double,3,1> > tcw = _estimate.tcw;

    std::vector<Eigen::Matrix<double,3,3> > Rbc = _estimate.Rbc;
    std::vector<Eigen::Matrix<double,3,1> > tbc = _estimate.tbc;

    const int num_cams = tcw.size();

    for(int idx = 0; idx<num_cams; idx++)
    {
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                os << Rcw[idx](i,j) << " ";
        }
        for (int i=0; i<3; i++){
            os << tcw[idx](i) << " ";
        }

        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++)
                os << Rbc[idx](i,j) << " ";
        }
        for (int i=0; i<3; i++){
            os << tbc[idx](i) << " ";
        }

    }

    return os.good();
}

VertexVelocity::VertexVelocity(Frame* pF)
{
    setEstimate(pF->velocity_);
}

VertexVelocity::VertexVelocity(Eigen::Vector3d& velocity)
{
    setEstimate(velocity);
}

VertexGyroBias::VertexGyroBias(Frame *pF)
{
    setEstimate(pF->getImuGyroBias());
}

VertexGyroBias::VertexGyroBias(Eigen::Vector3d& bg)
{
    setEstimate(bg);
}

VertexAccBias::VertexAccBias(Frame *pF)
{
    setEstimate(pF->ba_);
}

VertexAccBias::VertexAccBias(Eigen::Vector3d& ba)
{
    setEstimate(ba);
}

EdgeInertialGS::EdgeInertialGS(vilo::IMU* pInt):JRg(pInt->J_q_bg_),
    JVg(pInt->J_v_bg_), JPg(pInt->J_p_bg_), JVa(pInt->J_v_ba_),
    JPa(pInt->J_p_ba_), mpInt(pInt), dt(pInt->delta_t_)
{
    // This edge links 8 vertices
    resize(8);
    gI << 0, 0, -vilo::GRAVITY_VALUE;

    Matrix9d Info = mpInt->cov_mat_.block(0,0,9,9).inverse();//cv::DECOMP_SVD

    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();

    for(int i=0;i<9;i++)
        if(eigs[i]<1e-12)
            eigs[i]=0;

    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}

void EdgeInertialGS::computeError()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale* VS = static_cast<const VertexScale*>(_vertices[7]);
    
    Eigen::Vector3d ba, bg;
    ba << VA->estimate()[0],VA->estimate()[1],VA->estimate()[2];
    bg << VG->estimate()[0],VG->estimate()[1],VG->estimate()[2];
    g = VGDir->estimate().Rwg*gI;
    const double s = VS->estimate();

    const Eigen::Matrix3d delta_q_ = mpInt->getDeltaRotation(bg);
    const Eigen::Vector3d dV = mpInt->getDeltaVelocity(ba,bg);
    const Eigen::Vector3d dP = mpInt->getDeltaPosition(ba,bg);

    const Eigen::Vector3d er = logSO3(delta_q_.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(s*(VV2->estimate() - VV1->estimate()) - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(s*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt) - g*dt*dt/2) - dP;

    Eigen::Vector3d vp = VP1->estimate().Rwb.transpose()*(s*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt) - g*dt*dt/2);
    Eigen::Vector3d v12 = VV2->estimate()-VV1->estimate();
    Eigen::Vector3d v12_g = v12 - g*dt;
    Eigen::Vector3d sv12_g = s*v12_g;
    Eigen::Vector3d evv = VP1->estimate().Rwb.transpose()*sv12_g;
    Eigen::Vector3d vv = VP1->estimate().Rwb.transpose()*(s*(VV2->estimate() - VV1->estimate()) - g*dt);

    _error << er, ev, ep;
}

void EdgeInertialGS::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    const VertexScale* VS = static_cast<const VertexScale*>(_vertices[7]);

    Eigen::Vector3d ba, bg;
    ba << VA->estimate()[0],VA->estimate()[1],VA->estimate()[2];
    bg << VG->estimate()[0],VG->estimate()[1],VG->estimate()[2];
    Vector6d db = mpInt->getDeltaBias(ba, bg);

    Eigen::Vector3d dbg = db.tail(3);

    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;
    const Eigen::Matrix3d Rwg = VGDir->estimate().Rwg;
    Eigen::MatrixXd Gm = Eigen::MatrixXd::Zero(3,2);
    Gm(0,1) = -vilo::GRAVITY_VALUE;
    Gm(1,0) = vilo::GRAVITY_VALUE;
    const double s = VS->estimate();
    const Eigen::MatrixXd dGdTheta = Rwg*Gm;
    const Eigen::Matrix3d delta_q_ = mpInt->getDeltaRotation(bg); 
    const Eigen::Matrix3d eR = delta_q_.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = logSO3(eR);
    const Eigen::Matrix3d invJr = inverseRightJacobianSO3(er);

    _jacobianOplus[0].setZero();

    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1;
    _jacobianOplus[0].block<3,3>(3,0) = skew(Rbw1*(s*(VV2->estimate() - VV1->estimate()) - g*dt));
    _jacobianOplus[0].block<3,3>(6,0) = skew(Rbw1*(s*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt) - 0.5*g*dt*dt));
    _jacobianOplus[0].block<3,3>(6,3) = -s*Eigen::Matrix3d::Identity();

    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -s*Rbw1;
    _jacobianOplus[1].block<3,3>(6,0) = -s*Rbw1*dt;

    // Jacobians wrt Gyro bias
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*rightJacobianSO3(JRg*dbg)*JRg;
    _jacobianOplus[2].block<3,3>(3,0) = -JVg;
    _jacobianOplus[2].block<3,3>(6,0) = -JPg;

    // Jacobians wrt Accelerometer bias
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa;
    _jacobianOplus[3].block<3,3>(6,0) = -JPa;

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3,3>(0,0) = invJr;
    // translation
    _jacobianOplus[4].block<3,3>(6,3) = s*Rbw1*Rwb2;

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = s*Rbw1;

    // Jacobians wrt Gravity direction
    _jacobianOplus[6].setZero();
    _jacobianOplus[6].block<3,2>(3,0) = -Rbw1*dGdTheta*dt;
    _jacobianOplus[6].block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;

    // Jacobians wrt scale factor
    _jacobianOplus[7].setZero();
    _jacobianOplus[7].block<3,1>(3,0) = Rbw1*(VV2->estimate()-VV1->estimate());
    _jacobianOplus[7].block<3,1>(6,0) = Rbw1*(VP2->estimate().twb-VP1->estimate().twb-VV1->estimate()*dt);
}

EdgeInertialG::EdgeInertialG(vilo::IMU* pInt):JRg(pInt->J_q_bg_),
    JVg(pInt->J_v_bg_), JPg(pInt->J_p_bg_), JVa(pInt->J_v_ba_),
    JPa(pInt->J_p_ba_), mpInt(pInt), dt(pInt->delta_t_)
{
    resize(7);
    gI << 0, 0, -vilo::GRAVITY_VALUE;
    
    Matrix9d Info = mpInt->cov_mat_.block(0,0,9,9).inverse();//cv::DECOMP_SVD
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
     for(int i=0;i<9;i++)
         if(eigs[i]<1e-12)
             eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    setInformation(Info);
}

void EdgeInertialG::computeError()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);
    
    Eigen::Vector3d ba, bg;
    ba << VA->estimate()[0],VA->estimate()[1],VA->estimate()[2];
    bg << VG->estimate()[0],VG->estimate()[1],VG->estimate()[2];
    g = VGDir->estimate().Rwg*gI;
    
    const Eigen::Matrix3d delta_q_ = mpInt->getDeltaRotation(bg);
    const Eigen::Vector3d dV = mpInt->getDeltaVelocity(ba,bg);
    const Eigen::Vector3d dP = mpInt->getDeltaPosition(ba,bg);
 
    const Eigen::Vector3d er = logSO3(delta_q_.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(VV2->estimate() - VV1->estimate() - g*dt) - dV;
    const Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt - g*dt*dt/2) - dP;

    Eigen::Vector3d vp = VP1->estimate().Rwb.transpose()*(VP2->estimate().twb - VP1->estimate().twb - VV1->estimate()*dt - g*dt*dt/2);
    Eigen::Vector3d v12 = VV2->estimate()-VV1->estimate();
    Eigen::Vector3d v12_g = v12 - g*dt;
    Eigen::Vector3d sv12_g = v12_g;
    Eigen::Vector3d evv = VP1->estimate().Rwb.transpose()*sv12_g;
    Eigen::Vector3d vv = VP1->estimate().Rwb.transpose()*((VV2->estimate() - VV1->estimate()) - g*dt);
   
    _error << er, ev, ep;
}

void EdgeInertialG::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    const VertexGDir* VGDir = static_cast<const VertexGDir*>(_vertices[6]);

    Eigen::Vector3d ba, bg;
    ba << VA->estimate()[0],VA->estimate()[1],VA->estimate()[2];
    bg << VG->estimate()[0],VG->estimate()[1],VG->estimate()[2];
    Vector6d db = mpInt->getDeltaBias(ba, bg);

    Eigen::Vector3d dbg = db.tail(3);

    const Eigen::Matrix3d Rwb1 = VP1->estimate().Rwb;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().Rwb;
    const Eigen::Matrix3d Rwg = VGDir->estimate().Rwg;
    Eigen::MatrixXd Gm = Eigen::MatrixXd::Zero(3,2);
    Gm(0,1) = -vilo::GRAVITY_VALUE;
    Gm(1,0) = vilo::GRAVITY_VALUE;
    const Eigen::MatrixXd dGdTheta = Rwg*Gm;
    const Eigen::Matrix3d delta_q_ = mpInt->getDeltaRotation(bg);
    const Eigen::Matrix3d eR = delta_q_.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = logSO3(eR);
    const Eigen::Matrix3d invJr = inverseRightJacobianSO3(er);

    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1; 
    _jacobianOplus[0].block<3,3>(3,0) = skew(Rbw1*(VV2->estimate() - VV1->estimate() - g*dt));
    _jacobianOplus[0].block<3,3>(6,0) = skew(Rbw1*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt - 0.5*g*dt*dt));
    _jacobianOplus[0].block<3,3>(6,3) = -Eigen::Matrix3d::Identity();

    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -Rbw1;
    _jacobianOplus[1].block<3,3>(6,0) = -Rbw1*dt;

    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*rightJacobianSO3(JRg*dbg)*JRg; 
    _jacobianOplus[2].block<3,3>(3,0) = -JVg;
    _jacobianOplus[2].block<3,3>(6,0) = -JPg;

    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa;
    _jacobianOplus[3].block<3,3>(6,0) = -JPa;

    _jacobianOplus[4].setZero();
    _jacobianOplus[4].block<3,3>(0,0) = invJr;
    _jacobianOplus[4].block<3,3>(6,3) = Rbw1*Rwb2;

    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = Rbw1;

    _jacobianOplus[6].setZero();
    _jacobianOplus[6].block<3,2>(3,0) = -Rbw1*dGdTheta*dt;
    _jacobianOplus[6].block<3,2>(6,0) = -0.5*Rbw1*dGdTheta*dt*dt;
}

EdgeInertial::EdgeInertial(vilo::IMU* pImu):JRg(pImu->J_q_bg_),
    JVg(pImu->J_v_bg_), JPg(pImu->J_p_bg_), JVa(pImu->J_v_ba_),
    JPa(pImu->J_p_ba_), mpImu(pImu), dt(pImu->delta_t_)
{
    resize(6);
    g << 0, 0, -vilo::GRAVITY_VALUE;
    Matrix9d Info = mpImu->cov_mat_.block<9,9>(0,0).inverse();

    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
     for(int i=0;i<9;i++)
         if(eigs[i]<1e-12)
             eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();

    setInformation(Info);
    verbose = false;
}

EdgeInertial::EdgeInertial(vilo::IMU* pImu, bool is_verbose):JRg(pImu->J_q_bg_),
    JVg(pImu->J_v_bg_), JPg(pImu->J_p_bg_), JVa(pImu->J_v_ba_),
    JPa(pImu->J_p_ba_), mpImu(pImu), dt(pImu->delta_t_)
{
    resize(6);
    g << 0, 0, -vilo::GRAVITY_VALUE;
    Matrix9d Info = mpImu->cov_mat_.block<9,9>(0,0).inverse();
    
    Info = (Info+Info.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
    Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
     for(int i=0;i<9;i++)
         if(eigs[i]<1e-12)
             eigs[i]=0;
    Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();

    setInformation(Info);
    verbose = is_verbose;
}

void EdgeInertial::computeError()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
    Eigen::Vector3d ba, bg;
    ba << VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2];
    bg << VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2];
    
    const Eigen::Matrix3d delta_q_ = mpImu->getDeltaRotation(bg);
    const Eigen::Vector3d dV = mpImu->getDeltaVelocity(ba,bg);
    const Eigen::Vector3d dP = mpImu->getDeltaPosition(ba,bg);
  
    const Eigen::Vector3d er = logSO3(delta_q_.transpose()*VP1->estimate().Rwb.transpose()*VP2->estimate().Rwb);
    const Eigen::Vector3d ev = VP1->estimate().Rwb.transpose()*(VV2->estimate() - VV1->estimate() - g*dt) - dV;
   
    Eigen::Vector3d ep = VP1->estimate().Rwb.transpose()*(VP2->estimate().twb - VP1->estimate().twb
                                                               - VV1->estimate()*dt - g*dt*dt/2) - dP;
    
    _error << er, ev, ep; 

}

void EdgeInertial::linearizeOplus()
{
    const VertexPose* VP1 = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV1= static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[3]);
    const VertexPose* VP2 = static_cast<const VertexPose*>(_vertices[4]);
    const VertexVelocity* VV2= static_cast<const VertexVelocity*>(_vertices[5]);

    Eigen::Vector3d ba, bg;
    ba << VA1->estimate()[0],VA1->estimate()[1],VA1->estimate()[2];
    bg << VG1->estimate()[0],VG1->estimate()[1],VG1->estimate()[2];
    Vector6d db = mpImu->getDeltaBias(ba, bg);
    Eigen::Vector3d dbg = db.tail(3);
    
    const Eigen::Matrix3d Rwb1 = normalizeRotation(VP1->estimate().Rwb);
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = normalizeRotation(VP2->estimate().Rwb);

    const Eigen::Matrix3d delta_q_ = mpImu->getDeltaRotation(bg);
    const Eigen::Matrix3d eR = delta_q_.transpose()*Rbw1*Rwb2;
    const Eigen::Vector3d er = logSO3(eR);
    const Eigen::Matrix3d invJr = inverseRightJacobianSO3(er);
    
    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = -invJr*Rwb2.transpose()*Rwb1;
    _jacobianOplus[0].block<3,3>(3,0) = skew(Rbw1*(VV2->estimate() - VV1->estimate() - g*dt));
    _jacobianOplus[0].block<3,3>(6,0) = skew(Rbw1*(VP2->estimate().twb - VP1->estimate().twb
                                                   - VV1->estimate()*dt - 0.5*g*dt*dt));

    _jacobianOplus[0].block<3,3>(6,3) = -Eigen::Matrix3d::Identity();
    
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(3,0) = -Rbw1;
    _jacobianOplus[1].block<3,3>(6,0) = -Rbw1*dt;
    
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(0,0) = -invJr*eR.transpose()*rightJacobianSO3(JRg*dbg)*JRg;
    _jacobianOplus[2].block<3,3>(3,0) = -JVg;
    _jacobianOplus[2].block<3,3>(6,0) = -JPg;
    
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(3,0) = -JVa;
    _jacobianOplus[3].block<3,3>(6,0) = -JPa;
    
    _jacobianOplus[4].setZero();
    _jacobianOplus[4].block<3,3>(0,0) = invJr;
    _jacobianOplus[4].block<3,3>(6,3) = Rbw1*Rwb2;
    
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3,3>(3,0) = Rbw1;
}

void EdgePriorAcc::linearizeOplus()
{
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}

void EdgePriorGyro::linearizeOplus()
{
    _jacobianOplusXi.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}

void EdgeIdistOnlyCorner::computeError()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);
    SE3 Tth = Tt_*Th_.inverse();
    Vector2d obs(_measurement);
    _error = obs - _Fxy * (cam_project( Tth * (_fH*(1.0/point->estimate()))) );
   
}

void EdgeIdistOnlyCorner::linearizeOplus()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);
    double idHost = point->estimate();
    Sophus::SE3 Tth = Tt_*Th_.inverse();

    Eigen::Vector3d pts_camera_h = _fH*(1.0/idHost);
    Eigen::Vector3d pts_camera_t = Tth * pts_camera_h;

    double Z2 = pts_camera_t[2] * pts_camera_t[2];
    Eigen::Matrix<double,2,3> dudp;
    dudp << 1.0/pts_camera_t[2], 0, -pts_camera_t[0]/Z2,
            0, 1.0/pts_camera_t[2], -pts_camera_t[1]/Z2;
    
    dudp.noalias() = _Fxy * dudp;
    _jacobianOplusXi = dudp * Tth.rotation_matrix() * _fH/(idHost*idHost);

}

void EdgeIdistOnlyEdgeLet::computeError()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);

    SE3 Tth = Tt_*Th_.inverse();
    double obs = _measurement;
    _error(0,0) = obs - _normal.transpose()*_Fxy*(cam_project( Tth * (_fH*(1.0/point->estimate()))));
}

void EdgeIdistOnlyEdgeLet::linearizeOplus()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);
    double idHost = point->estimate();

    SE3 Tth = Tt_*Th_.inverse();

    Eigen::Vector3d pts_camera_h = _fH*(1.0/idHost);
    Eigen::Vector3d pts_camera_t = Tth * pts_camera_h;

    double Z2 = pts_camera_t[2] * pts_camera_t[2];
    Eigen::Matrix<double,2,3> dudp;
    dudp << 1.0/pts_camera_t[2], 0, -pts_camera_t[0]/Z2,
            0, 1.0/pts_camera_t[2], -pts_camera_t[1]/Z2;

    dudp.noalias() = _Fxy * dudp;

    _jacobianOplusXi = _normal.transpose() * dudp * Tth.rotation_matrix() * _fH/(idHost*idHost);

}

void EdgeIdistCorner::computeError()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);
    const VertexPose* host   = static_cast<const VertexPose*>(_vertices[1]); 
    const VertexPose* target = static_cast<const VertexPose*>(_vertices[2]); 

    const Eigen::Matrix3d& Rcw_h = host->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_h = host->estimate().tcw[0];
    const Eigen::Matrix3d& Rcw_t = target->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_t = target->estimate().tcw[0];

    SE3 Ttw = SE3(Rcw_t, tcw_t);
    SE3 Thw = SE3(Rcw_h, tcw_h);
    SE3 Tth = Ttw*Thw.inverse();

    Vector2d obs(_measurement);
    _error = obs - _Fxy * (cam_project( Tth * (_fH*(1.0/point->estimate()))) ); 
}

void EdgeIdistCorner::linearizeOplus()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);
    const VertexPose* host   = static_cast<const VertexPose*>(_vertices[1]); 
    const VertexPose* target = static_cast<const VertexPose*>(_vertices[2]); 
    
    double idHost = point->estimate();
    const Eigen::Matrix3d& Rcw_h  = host->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_h = host->estimate().tcw[0];
    const Eigen::Matrix3d& Rcw_t = target->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_t = target->estimate().tcw[0];
    const Eigen::Matrix3d& Rcb = host->estimate().Rcb[0];
    const Eigen::Vector3d& tcb = host->estimate().tcb[0];

    SE3 Ttw = SE3(Rcw_t, tcw_t);
    SE3 Thw = SE3(Rcw_h, tcw_h);
    SE3 Tcb = SE3(Rcb, tcb);
    SE3 Tth = Ttw*Thw.inverse();

    SE3 Ttw_b = Tcb.inverse() * Ttw;
    SE3 Thw_b = Tcb.inverse() * Thw;
    SE3 Tth_b = Ttw_b * Thw_b.inverse();

    Eigen::Vector3d pts_camera_h = _fH*(1.0/idHost);
    Eigen::Vector3d pts_imu_h = Tcb.inverse() * pts_camera_h;
    Eigen::Vector3d pts_w = Thw_b.inverse() * pts_imu_h;
    Eigen::Vector3d pts_imu_t = Tth_b * pts_imu_h;
    Eigen::Vector3d pts_camera_t = Tcb * pts_imu_t;

    double Z2 = pts_camera_t[2] * pts_camera_t[2];
    Eigen::Matrix<double,2,3> dudp;
    dudp << 1.0/pts_camera_t[2], 0, -pts_camera_t[0]/Z2,
            0, 1.0/pts_camera_t[2], -pts_camera_t[1]/Z2;
    
    dudp.noalias() = _Fxy * dudp;

    _jacobianOplus[0] = dudp * Rcb * Tth_b.rotation_matrix() * Rcb.transpose()* _fH/(idHost*idHost);

    Eigen::Matrix<double, 3, 6> jaco_th;
    jaco_th << 0.0, pts_imu_t[2],   -pts_imu_t[1], 1.0, 0.0, 0.0,
            -pts_imu_t[2] , 0.0, pts_imu_t[0], 0.0, 1.0, 0.0,
            pts_imu_t[1] ,  -pts_imu_t[0] , 0.0, 0.0, 0.0, 1.0; 

    Eigen::Matrix<double, 3, 6> jaco_phi;
    jaco_phi << 0.0, pts_imu_h[2],   -pts_imu_h[1], 1.0, 0.0, 0.0,
            -pts_imu_h[2] , 0.0, pts_imu_h[0], 0.0, 1.0, 0.0,
            pts_imu_h[1] ,  -pts_imu_h[0] , 0.0, 0.0, 0.0, 1.0; 
    _jacobianOplus[2] =  dudp * Rcb * jaco_th;
    _jacobianOplus[1] = -dudp * Rcb * Tth_b.rotation_matrix()*jaco_phi;
}

void EdgeIdistEdgeLet::computeError()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);
    const VertexPose* host   = static_cast<const VertexPose*>(_vertices[1]); 
    const VertexPose* target = static_cast<const VertexPose*>(_vertices[2]); 
    
    const Eigen::Matrix3d& Rcw_h = host->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_h = host->estimate().tcw[0];
    const Eigen::Matrix3d& Rcw_t = target->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_t = target->estimate().tcw[0];

    SE3 Ttw = SE3(Rcw_t, tcw_t);
    SE3 Thw = SE3(Rcw_h, tcw_h);
    SE3 Tth = Ttw*Thw.inverse();

    double obs = _measurement;
    _error(0,0) = obs - _normal.transpose()*_Fxy*(cam_project( Tth * (_fH*(1.0/point->estimate()))));   
}

void EdgeIdistEdgeLet::linearizeOplus()
{
    const VertexPointID* point = static_cast<const VertexPointID*>(_vertices[0]);
    const VertexPose* host   = static_cast<const VertexPose*>(_vertices[1]); 
    const VertexPose* target = static_cast<const VertexPose*>(_vertices[2]); 

    double idHost = point->estimate();
    const Eigen::Matrix3d& Rcw_h  = host->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_h = host->estimate().tcw[0];
    const Eigen::Matrix3d& Rcw_t = target->estimate().Rcw[0];
    const Eigen::Vector3d& tcw_t = target->estimate().tcw[0];
    const Eigen::Matrix3d& Rcb = host->estimate().Rcb[0];
    const Eigen::Vector3d& tcb = host->estimate().tcb[0];

    SE3 Ttw = SE3(Rcw_t, tcw_t);
    SE3 Thw = SE3(Rcw_h, tcw_h);
    SE3 Tcb = SE3(Rcb, tcb);
    SE3 Tth = Ttw*Thw.inverse();

    SE3 Ttw_b = Tcb.inverse() * Ttw;
    SE3 Thw_b = Tcb.inverse() * Thw;
    SE3 Tth_b = Ttw_b * Thw_b.inverse();
    
    Eigen::Vector3d pts_camera_h = _fH*(1.0/idHost);
    Eigen::Vector3d pts_imu_h = Tcb.inverse() * pts_camera_h;
    Eigen::Vector3d pts_w = Thw_b.inverse() * pts_imu_h;
    Eigen::Vector3d pts_imu_t = Tth_b * pts_imu_h;
    Eigen::Vector3d pts_camera_t = Tcb * pts_imu_t;

    double Z2 = pts_camera_t[2] * pts_camera_t[2];
    Eigen::Matrix<double,2,3> dudp;
    dudp << 1.0/pts_camera_t[2], 0, -pts_camera_t[0]/Z2,
            0, 1.0/pts_camera_t[2], -pts_camera_t[1]/Z2;

    dudp.noalias() = _Fxy * dudp;

    _jacobianOplus[0] = _normal.transpose() * dudp * Rcb * Tth_b.rotation_matrix() * Rcb.transpose()* _fH/(idHost*idHost);

    Eigen::Matrix<double, 3, 6> jaco_th;
    jaco_th << 0.0, pts_imu_t[2],   -pts_imu_t[1], 1.0, 0.0, 0.0,
            -pts_imu_t[2] , 0.0, pts_imu_t[0], 0.0, 1.0, 0.0,
            pts_imu_t[1] ,  -pts_imu_t[0] , 0.0, 0.0, 0.0, 1.0; 

    Eigen::Matrix<double, 3, 6> jaco_phi;
    jaco_phi << 0.0, pts_imu_h[2],   -pts_imu_h[1], 1.0, 0.0, 0.0,
            -pts_imu_h[2] , 0.0, pts_imu_h[0], 0.0, 1.0, 0.0,
            pts_imu_h[1] ,  -pts_imu_h[0] , 0.0, 0.0, 0.0, 1.0; 
    _jacobianOplus[2] =  _normal.transpose() *  dudp * Rcb * jaco_th;
    _jacobianOplus[1] = -_normal.transpose() *  dudp * Rcb * Tth_b.rotation_matrix()*jaco_phi;
}

EdgePriorPoseImu::EdgePriorPoseImu(IMUPriorConstraint *c)
{
    resize(4);
    Rwb = c->Rwb;
    twb = c->twb;
    vwb = c->vwb;
    bg = c->bg;
    ba = c->ba;
    setInformation(c->H);
}

void EdgePriorPoseImu::computeError()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const VertexVelocity* VV = static_cast<const VertexVelocity*>(_vertices[1]);
    const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[2]);
    const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[3]);

    const Eigen::Vector3d er = logSO3(Rwb.transpose()*VP->estimate().Rwb);
    const Eigen::Vector3d et = Rwb.transpose()*(VP->estimate().twb-twb);
    const Eigen::Vector3d ev = VV->estimate() - vwb;
    const Eigen::Vector3d ebg = VG->estimate() - bg;
    const Eigen::Vector3d eba = VA->estimate() - ba;

    _error << er, et, ev, ebg, eba;
}

void EdgePriorPoseImu::linearizeOplus()
{
    const VertexPose* VP = static_cast<const VertexPose*>(_vertices[0]);
    const Eigen::Vector3d er = logSO3(Rwb.transpose()*VP->estimate().Rwb);

    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3,3>(0,0) = inverseRightJacobianSO3(er);
    _jacobianOplus[0].block<3,3>(3,3) = Rwb.transpose()*VP->estimate().Rwb;
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3,3>(6,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3,3>(9,0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3,3>(12,0) = Eigen::Matrix3d::Identity();
}

void EdgeCornerOnlyPose::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[_cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[_cam_idx];
    const Eigen::Vector3d Xc = Rcw*_Xw + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[_cam_idx]*Xc+VPose->estimate().tbc[_cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[_cam_idx];

    double fx,fy;
    fx = _Fxy(0,0);
    fy = _Fxy(1,1);
    Eigen::Matrix<double,2,3> proj_jac;
    proj_jac << fx/Xc[2], 0.f, -(fx*Xc[0])/(Xc[2]*Xc[2]),
                0.f, fy/Xc[2], -(fy*Xc[1])/(Xc[2]*Xc[2]);

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);
    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;
    _jacobianOplusXi = proj_jac * Rcb * SE3deriv; 
}

void EdgeEdgeletOnlyPose::linearizeOplus()
{
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);

    const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[_cam_idx];
    const Eigen::Vector3d &tcw = VPose->estimate().tcw[_cam_idx];
    const Eigen::Vector3d Xc = Rcw*_Xw + tcw;
    const Eigen::Vector3d Xb = VPose->estimate().Rbc[_cam_idx]*Xc+VPose->estimate().tbc[_cam_idx];
    const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[_cam_idx];

    double fx,fy;
    fx = _Fxy(0,0);
    fy = _Fxy(1,1);
    Eigen::Matrix<double,2,3> proj_jac;
    proj_jac << fx/Xc[2], 0, -(fx*Xc[0])/(Xc[2]*Xc[2]),
                0, fy/Xc[2], -(fy*Xc[1])/(Xc[2]*Xc[2]);

    Eigen::Matrix<double,3,6> SE3deriv;
    double x = Xb(0);
    double y = Xb(1);
    double z = Xb(2);
    SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
            -z , 0.0, x, 0.0, 1.0, 0.0,
            y ,  -x , 0.0, 0.0, 0.0, 1.0;
    _jacobianOplusXi = _normal.transpose()*proj_jac * Rcb * SE3deriv; 

}

void EdgeLidarPointPlane::computeError()
{
    const VertexPose* P   = static_cast<const VertexPose*>(_vertices[0]);

    Sophus::SE3 Twb(P->estimate().Rwb, P->estimate().twb);
    _error(0,0) =  - (_nml.dot(Twb*_measurement) + _d);
}

void EdgeLidarPointPlane::linearizeOplus()
{
    const VertexPose* P = static_cast<const VertexPose*>(_vertices[0]);
    Sophus::SE3 Twb(P->estimate().Rwb, P->estimate().twb);

    Eigen::Vector3d pts_imu_h = _measurement;
    Eigen::Matrix<double, 3, 6> jaco_phi;
    jaco_phi << 0.0, pts_imu_h[2],   -pts_imu_h[1], 1.0, 0.0, 0.0,
            -pts_imu_h[2] , 0.0, pts_imu_h[0], 0.0, 1.0, 0.0,
            pts_imu_h[1] ,  -pts_imu_h[0] , 0.0, 0.0, 0.0, 1.0; 

    _jacobianOplusXi = -_nml.transpose() * Twb.rotation_matrix()*jaco_phi;
}

void EdgeLidarPointLine::computeError()
{
    const VertexPose* P2   = static_cast<const VertexPose*>(_vertices[0]);

    Sophus::SE3 T2(P2->estimate().Rwb, P2->estimate().twb);
    Sophus::SE3 T12 = _T1.inverse() * T2;

    Eigen::Vector3d p_h = T12*_measurement;
    Eigen::Vector3d nu = (p_h - _p1).cross(p_h - _p2);
    Eigen::Vector3d de = _p1 - _p2;

    _error << -nu.norm() / de.norm();
}

void EdgeLidarPointLine::linearizeOplus()
{
    Eigen::Matrix3d Rbl, Rlb;
    Rbl.setIdentity(); Rlb.setIdentity(); 
    const VertexPose* P2 = static_cast<const VertexPose*>(_vertices[0]);
    Sophus::SE3 T2(P2->estimate().Rwb, P2->estimate().twb);
    Sophus::SE3 T12 = _T1.inverse() * T2;

    Eigen::Vector3d pts_imu_h = Rbl * _measurement;
    Eigen::Matrix<double, 3, 6> jaco_phi;
    jaco_phi << 0.0, pts_imu_h[2],   -pts_imu_h[1], 1.0, 0.0, 0.0,
            -pts_imu_h[2] , 0.0, pts_imu_h[0], 0.0, 1.0, 0.0,
            pts_imu_h[1] ,  -pts_imu_h[0] , 0.0, 0.0, 0.0, 1.0; 

    Eigen::Vector3d p_h = T12*_measurement;

    double x0 = p_h[0];
    double y0 = p_h[1];
    double z0 = p_h[2];
    double x1 = _p1[0];
    double y1 = _p1[1];
    double z1 = _p1[2];
    double x2 = _p2[0];
    double y2 = _p2[1];
    double z2 = _p2[2];

    double a012 = ((p_h - _p1).cross(p_h - _p2)).norm();
    double l12 = (_p1 - _p2).norm();      

    double la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
            + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

    double lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

    double lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

    Eigen::Vector3d grad(la, lb, lc);
    _jacobianOplusXi = -grad.transpose()  * Rlb * T12.rotation_matrix()*jaco_phi;
}

} // namespace vilo


