#include <boost/thread.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
// #include <g2o/solvers/linear_solver_cholmod.h>
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/solvers/linear_solver_dense.h"

#include "vilo/vikit/math_utils.h"
#include "vilo/vikit/robust_cost.h"

#include "vilo/bundle_adjustment.h"
#include "vilo/frame.h"
#include "vilo/feature.h"
#include "vilo/point.h"
#include "vilo/config.h"
#include "vilo/map.h"
#include "vilo/matcher.h"
#include "vilo/lidar.h"

#include <stdlib.h>
#include <time.h>
#include <random>


#define SCHUR_TRICK 1

namespace vilo {
namespace ba {

static Eigen::MatrixXd marginalize(const Eigen::MatrixXd &H, const int &start, const int &end);

g2oFrameSE3*
createG2oFrameSE3(Frame* frame, size_t id, bool fixed)
{
    g2oFrameSE3* v = new g2oFrameSE3();
    v->setId(id);
    v->setFixed(fixed);
    
    Sophus::SE3 Tfw = frame->getFramePose();
    v->setEstimate(g2o::SE3Quat(Tfw.unit_quaternion(), Tfw.translation()));

    return v;
}

g2oPoint*
createG2oPoint(Vector3d pos,
               size_t id,
               bool fixed)
{
  g2oPoint* v = new g2oPoint();
  v->setId(id);
#if SCHUR_TRICK
  v->setMarginalized(true);
#endif
  v->setFixed(fixed);
  v->setEstimate(pos);
  return v;
}

g2oEdgeSE3*
createG2oEdgeSE3( g2oFrameSE3* v_frame,
                  g2oPoint* v_point,
                  const Vector2d& f_up,
                  bool robust_kernel,
                  double huber_width,
                  double weight)
{
  g2oEdgeSE3* e = new g2oEdgeSE3();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(f_up);
  
  e->setInformation(Eigen::Matrix2d::Identity()*weight);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();   
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  e->setParameterId(0, 0); 
  return e;
}

rdvoEdgeProjectXYZ2UV*
createG2oEdgeletSE3(g2oFrameSE3* v_frame,
                    g2oPoint* v_point,
                    const Vector2d& f_up,
                    bool robust_kernel,
                    double huber_width,
                    double weight,
                    const Vector2d& grad)
{
  rdvoEdgeProjectXYZ2UV* e = new rdvoEdgeProjectXYZ2UV();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(grad.transpose() * f_up);
  e->information() = weight * Eigen::Matrix<double,1,1>::Identity(1,1);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  e->setParameterId(0, 0);
  e->setGrad(grad);
  return e;
}

void setupG2o(g2o::SparseOptimizer * optimizer)
{
  optimizer->setVerbose(false);

#if SCHUR_TRICK
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
  linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
  g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
#else
  g2o::BlockSolverX::LinearSolverType * linearSolver;
  linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
  g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
#endif

  solver->setMaxTrialsAfterFailure(5);
  optimizer->setAlgorithm(solver);

}

void
runSparseBAOptimizer(g2o::SparseOptimizer* optimizer,
                     unsigned int num_iter,
                     double& init_error, double& final_error)
{
    optimizer->initializeOptimization();
    optimizer->computeActiveErrors();
    init_error = optimizer->activeChi2();
    optimizer->optimize(num_iter);
    final_error = optimizer->activeChi2();
}


Eigen::MatrixXd marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    const int a = start;
    const int b = end-start+1;
    const int c = H.cols() - (end+1);
    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        Hn.block(0,0,a,a) = H.block(0,0,a,a);
        Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
        Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
    }
    if(a>0 && c>0)
    {
        Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
        Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
    }
    if(c>0)
    {
        Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
        Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
        Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
    }
    Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();
    for (int i=0; i<b; ++i)
    {
        if (singularValues_inv(i)>1e-6)
            singularValues_inv(i)=1.0/singularValues_inv(i);
        else singularValues_inv(i)=0;
    }
    Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();
    Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
    Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
    Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
    Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        res.block(0,0,a,a) = Hn.block(0,0,a,a);
        res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
        res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
    }
    if(a>0 && c>0)
    {
        res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
        res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
    }
    if(c>0)
    {
        res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
        res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
        res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
    }

    res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

    return res;
}


void visualImuAlign(list<FramePtr>& keyframes,
                    Eigen::Matrix3d& Rwg,
                    double& scale,
                    bool is_vel_fixed,
                    double priorG,
                    double priorA)
{
    int n_its = 200; 
    int max_id = keyframes.back()->id_;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    if (priorG!=0.f)
        solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    auto iter = keyframes.begin();
    while(iter!=keyframes.end())
    {
        FramePtr kf = *iter;
        if(kf->id_ > max_id)
        {
            ++iter;
            continue;
        }

        VertexPose * VP = new VertexPose(kf.get());
        VP->setId(kf->id_);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(kf.get());
        VV->setId( max_id+(kf->id_)+1 );
        if (is_vel_fixed)
            VV->setFixed(true);
        else
            VV->setFixed(false);

        optimizer.addVertex(VV);
        ++iter;
    }
    
    VertexGyroBias* VG = new VertexGyroBias(keyframes.front().get());
    VG->setId(max_id*2+2);
    if (is_vel_fixed)
        VG->setFixed(true);
    else
        VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(keyframes.front().get());
    VA->setId(max_id*2+3);
    if (is_vel_fixed)
        VA->setFixed(true);
    else
        VA->setFixed(false);

    optimizer.addVertex(VA);

    EdgePriorAcc* epa = new EdgePriorAcc(Eigen::Vector3d(0,0,0));
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(Eigen::Vector3d(0,0,0));
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(max_id*2+4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(max_id*2+5);
    optimizer.addVertex(VS);

    vector<EdgeInertialGS*> ei_vec;
    ei_vec.reserve(keyframes.size());
    vector<pair<Frame*,Frame*> > kf_pair_vec;
    kf_pair_vec.reserve(keyframes.size());

    iter = keyframes.begin();
    while(iter!=keyframes.end())
    {
        FramePtr kf = *iter;
        if(kf->last_kf_ && kf->id_<=max_id)
        {
            if(!kf->imu_from_last_keyframe_)
                std::cout << "Not preintegrated measurement" << std::endl;

            kf->setImuAccBias(kf->last_kf_->getImuAccBias());
            kf->setImuGyroBias(kf->last_kf_->getImuGyroBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(kf->last_kf_->id_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(max_id+(kf->last_kf_->id_)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(kf->id_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(max_id+(kf->id_)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(max_id*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(max_id*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(max_id*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(max_id*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;
                ++iter;
                continue;
            }

            EdgeInertialGS* ei = new EdgeInertialGS(kf->imu_from_last_keyframe_);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            ei_vec.push_back(ei);

            kf_pair_vec.push_back(make_pair(kf->last_kf_,kf.get()));
            optimizer.addEdge(ei);

            ei->computeError();

        }
        ++iter;
    }

    std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err_ini = optimizer.activeRobustChi2();
    optimizer.optimize(n_its);
    float err_end = optimizer.activeRobustChi2();

    scale = VS->estimate();
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_id*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(max_id*2+3));

    Eigen::Vector3d ba = VA->estimate();
    Eigen::Vector3d bg = VG->estimate();
    Rwg = VGDir->estimate().Rwg;

    iter = keyframes.begin();
    while(iter!=keyframes.end())
    {
        FramePtr kf = *iter;
        if(kf->id_ > max_id)
            continue;
        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_id+(kf->id_)+1));
        Eigen::Vector3d Vw = VV->estimate();
        kf->setVelocity(Vw);
        if ( (kf->getImuGyroBias()-bg).norm() >0.01)
        {
          
            kf->setNewBias(ba,bg);
            if (kf->imu_from_last_keyframe_)
            {
                kf->imu_from_last_keyframe_->reintegrate();
            }
        }
        else
            kf->setNewBias(ba,bg);

        ++iter;
    }
}


void visualImuLocalBundleAdjustment(Frame* center_kf,
                                    set<Frame*>* core_kfs,
                                    Map* map,
                                    bool is_tracking_good,
                                    size_t& n_incorrect_edges_1,
                                    size_t& n_incorrect_edges_2,
                                    double& init_error,
                                    double& final_error)
{
    struct timeval st,et;
    gettimeofday(&st,NULL);

    printf("visualImuLocalBundleAdjustment 开始:%d\n",center_kf->id_);

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-1);
    optimizer.setAlgorithm(solver);

    const int n_opt_kfs = 7; 
    int n_features = 0;
    int n_features_threshold = 1500;
    std::vector<Frame*> act_kf_vec;
    Frame* fixed_kf = NULL;
    bool is_in_act_vec = true;
    int fix_id = -1;

    act_kf_vec.push_back(center_kf);
    act_kf_vec.back()->lba_id_ = center_kf->id_;
    while(act_kf_vec.back()->last_kf_)
    {
        act_kf_vec.push_back(act_kf_vec.back()->last_kf_);
        act_kf_vec.back()->lba_id_ = center_kf->id_;

        if(act_kf_vec.size() >= n_opt_kfs)
        {
            fix_id = act_kf_vec.back()->id_;
            break;
        }

    }

    std::vector<Frame*> covisual_kf_vec;
    for(auto it=core_kfs->begin();it!=core_kfs->end();++it)
    {
        if((*it)->lba_id_ == center_kf->id_)
            continue;

        if((*it)->id_ < fix_id)
        {
            fix_id = (*it)->id_;
            is_in_act_vec = false;
            fixed_kf = *it;
        }
    }

    if(is_in_act_vec)
    {
        if(act_kf_vec.back()->last_kf_)
        {
            fixed_kf = act_kf_vec.back()->last_kf_;
        }
        else
        {
            fixed_kf = act_kf_vec.back();
            act_kf_vec.pop_back();
        }
    }
    else
    {
        for(auto it=core_kfs->begin();it!=core_kfs->end();++it)
        {
            if((*it)->lba_id_ == center_kf->id_)
                continue;

            if((*it)->id_ == fix_id)
                continue;

            (*it)->lba_id_ = center_kf->id_;
            covisual_kf_vec.push_back(*it);
        }
    }
    fixed_kf->lba_id_ = center_kf->id_;
   
    const int N = act_kf_vec.size() + covisual_kf_vec.size();
    const int max_id = center_kf->id_;
    set<Point*> mps;

    auto iter = act_kf_vec.begin();
    while(iter != act_kf_vec.end())
    {

        Frame* kf = *iter;
        if(kf->id_ > max_id)
        {
            ++iter;
            continue;
        }

        bool is_fixed = false;

        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        optimizer.addVertex(VP);
        VP->setFixed(is_fixed);

        if(kf->imu_from_last_keyframe_)
        {
            VertexVelocity* VV = new VertexVelocity(kf);
            VV->setId( max_id + 3*kf->id_ + 1 );
            VV->setFixed(is_fixed);
            optimizer.addVertex(VV);

            VertexGyroBias* VG = new VertexGyroBias(kf);
            VG->setId(max_id + 3*kf->id_ + 2);
            VG->setFixed(is_fixed);
            optimizer.addVertex(VG);

            VertexAccBias* VA = new VertexAccBias(kf);
            VA->setId(max_id + 3*kf->id_ + 3);
            VA->setFixed(is_fixed);
            optimizer.addVertex(VA);
        }

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;
                
            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
            mps.insert((*it_pt)->point);
        }

        ++iter;
    }

    iter = covisual_kf_vec.begin();
    while(iter != covisual_kf_vec.end())
    {
        Frame* kf = *iter;
        if(kf->id_ > max_id)
        {
            ++iter;
            continue;
        }

        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL) continue;

            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
            mps.insert((*it_pt)->point);
        }

        ++iter;
    }

    {
        VertexPose * VP = new VertexPose(fixed_kf);
        VP->setId(fixed_kf->id_);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(fixed_kf->imu_from_last_keyframe_)
        {
            VertexVelocity* VV = new VertexVelocity(fixed_kf);
            VV->setId( max_id + 3*fixed_kf->id_ + 1 );
            VV->setFixed(true);
            optimizer.addVertex(VV);

            VertexGyroBias* VG = new VertexGyroBias(fixed_kf);
            VG->setId(max_id + 3*fixed_kf->id_ + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);

            VertexAccBias* VA = new VertexAccBias(fixed_kf);
            VA->setId(max_id + 3*fixed_kf->id_ + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }

        for(Features::iterator it_pt=fixed_kf->fts_.begin(); it_pt!=fixed_kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;
           
            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
            mps.insert((*it_pt)->point);
        }
    }

    float huber_corner = 0, huber_edge = 0;
    double focal_length = center_kf->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = center_kf->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];
    computeVisualWeight(mps, focal_length, huber_corner, huber_edge);

    std::set<Frame*> fix_kf_set;
    list<EdgeFrameFeature> edges;
    list<EdgeLetFrameFeature> edgeLets;
    int n_edges = 0;
    int v_id = 4 * (max_id+1);
    double visual_error = 0;
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexPointID* vPoint = new VertexPointID();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        double idist = (*it_pt)->getPointIdist();
        vPoint->setEstimate(idist);
        assert(optimizer.addVertex(vPoint));
        (*it_pt)->lba_id_ = center_kf->id_;

        if((*it_pt)->hostFeature_->frame->lba_id_ != center_kf->id_)
        {
            (*it_pt)->hostFeature_->frame->lba_id_ = center_kf->id_;
            VertexPose* vHost = new VertexPose((*it_pt)->hostFeature_->frame);
            vHost->setId((*it_pt)->hostFeature_->frame->id_);
            vHost->setFixed(true);
            assert(optimizer.addVertex(vHost));

            fix_kf_set.insert((*it_pt)->hostFeature_->frame);
        }
        
        list<Feature*> obs = (*it_pt)->getObs();
        list<Feature*>::iterator it_obs = obs.begin();
        while(it_obs != obs.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            if((*it_obs)->frame->lba_id_ != center_kf->id_)
            {
                (*it_obs)->frame->lba_id_ = center_kf->id_;
                VertexPose* vTarget = new VertexPose((*it_obs)->frame);
                vTarget->setId((*it_obs)->frame->id_);
                vTarget->setFixed(true);
                assert(optimizer.addVertex(vTarget));
                fix_kf_set.insert((*it_obs)->frame);
            }

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIdistCorner* edge = new EdgeIdistCorner();
                edge->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edge->setHostBearing((*it_pt)->hostFeature_->f);
                edge->setCameraParams(fxy);
                edge->setMeasurement(Fxy* vilo::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(0.1*Eigen::Matrix2d::Identity() );

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_corner);
                edge->setRobustKernel(rk);
                edge->setParameterId(0, 0);

                edges.push_back(EdgeFrameFeature(edge, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edge));

                edge->computeError();
                visual_error += edge->chi2();
            }
            else
            {
                EdgeIdistEdgeLet* edgeLet = new EdgeIdistEdgeLet();
                edgeLet->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f);
                edgeLet->setCameraParams(fxy);
                edgeLet->setTargetNormal((*it_obs)->grad);
                edgeLet->setMeasurement((*it_obs)->grad.transpose()* Fxy* vilo::project2d((*it_obs)->f));
        
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));

                edgeLet->setInformation(0.1*Eigen::Matrix<double,1,1>::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                edgeLet->setRobustKernel(rk);
                edgeLet->setParameterId(0, 0);

                edgeLets.push_back(EdgeLetFrameFeature(edgeLet, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edgeLet));

                edgeLet->computeError();
                visual_error += edgeLet->chi2();
            }

            ++n_edges;
            ++it_obs;
        }

    }

    vector<EdgeInertial*> vi_vec(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> eg_vec(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> ea_vec(N,(EdgeAccRW*)NULL);

    for(size_t i=0, iend=act_kf_vec.size()-1;i<iend;i++)
    {
        Frame* kf = act_kf_vec[i];
        if(kf->id_ > max_id)
            continue;

        if(kf->last_kf_ && kf->last_kf_->id_ < max_id)
        {
            if(!kf->imu_from_last_keyframe_ || !kf->last_kf_->imu_from_last_keyframe_)
            {
                continue;
            }
            
            kf->setImuAccBias(kf->last_kf_->getImuAccBias());
            kf->setImuGyroBias(kf->last_kf_->getImuGyroBias());

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(kf->last_kf_->id_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(max_id + 3*kf->last_kf_->id_ + 1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(max_id + 3*kf->last_kf_->id_ + 2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(max_id + 3*kf->last_kf_->id_ + 3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(kf->id_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(max_id + 3*kf->id_ + 1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(max_id + 3*kf->id_ + 2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(max_id + 3*kf->id_ + 3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            EdgeInertial* vi= NULL;
            vi = new EdgeInertial(kf->imu_from_last_keyframe_, false);
            vi->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vi->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vi->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vi->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vi->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vi->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            optimizer.addEdge(vi);
            vi_vec.push_back(vi);

            EdgeGyroRW* eg = new EdgeGyroRW();
            eg->setVertex(0,VG1);
            eg->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(9,9).inverse();
            eg->setInformation(0.1*InfoG);
            optimizer.addEdge(eg);
            eg_vec.push_back(eg);

            EdgeAccRW* ea = new EdgeAccRW();
            ea->setVertex(0,VA1);
            ea->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(12,12).inverse();
            ea->setInformation(0.1*InfoA);
            optimizer.addEdge(ea);
            ea_vec.push_back(ea);

            vi->computeError();

            if(i==0)
            {
                double factor = vi->chi2() / 200;
                factor = 1.0 / (ceil(factor));
                vi->setInformation(vi->information() * factor);

                ea->setInformation(ea->information()*1e-2);
                eg->setInformation(eg->information()*1e-2);
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vi->setRobustKernel(rki);
                rki->setDelta(sqrt(100));

            }
            double ee_vi = vi->chi2();
            eg->computeError();
            double ee_eg = eg->chi2();

            ea->computeError();
            double ee_ea = ea->chi2();

        }

    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    init_error = optimizer.activeChi2();
    
    optimizer.setVerbose(false);
    int n_its = (n_edges>=4000 ? 10 : 15);
    optimizer.optimize(n_its); 
    final_error = optimizer.activeChi2();

    boost::unique_lock<boost::mutex> lock(map->map_mutex_);
    iter = act_kf_vec.begin();
    while(iter!=act_kf_vec.end())
    {
        Frame* frame = *iter;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex( frame->id_ ) );
        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(  max_id + 3*frame->id_ + 1 ) );
        VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(  max_id + 3*frame->id_ + 2 ) );
        VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(  max_id + 3*frame->id_ + 3 ) );

        Sophus::SE3 Tfw( VP->estimate().Rcw[0], VP->estimate().tcw[0]);
        frame->setFramePose(Tfw);
        frame->setVelocity( VV->estimate() );
        frame->setImuAccBias( VA->estimate() );
        frame->setImuGyroBias( VG->estimate() );
        map->point_candidates_.changeCandidatePosition(*iter);
        frame->lba_id_ = -1;
        ++iter;
    }

    set<Frame*>::iterator sit = core_kfs->begin();
    while(sit!=core_kfs->end())
    {
        VertexPose* VPose = static_cast<VertexPose*>(optimizer.vertex( (*sit)->id_ ) );
        Sophus::SE3 Tfw( VPose->estimate().Rcw[0], VPose->estimate().tcw[0]);
        (*sit)->setFramePose(Tfw);
        map->point_candidates_.changeCandidatePosition(*sit);
        (*sit)->lba_id_ = -1;
        ++sit;
    }

    sit = fix_kf_set.begin();
    while(sit!=fix_kf_set.end())
    {
        (*sit)->lba_id_ = -1;
        ++sit;
    }

    v_id = 4 * (max_id+1);
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it,++v_id)
    {
        VertexPointID* VPoint = static_cast<VertexPointID*>(optimizer.vertex(v_id) );

        double idist = VPoint->estimate();
        (*it)->setPointIdistAndPose(idist);
        (*it)->lba_id_ = -1;
    }

    const double reproj_thresh_2 = 2.0; 
    const double reproj_thresh_1 = 1.2;

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeFrameFeature>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if( it->feature->point == NULL) continue;

        if(it->edge->chi2() > reproj_thresh_2_squared)
        {
            if(it->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeLetFrameFeature>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if(it->feature->point == NULL) continue;


        if(it->edge->chi2() > reproj_thresh_1_squared)
        {
            if(it->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }

    map->setMapChangedFlag(true);
    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}


void visualLocalBundleAdjustment(Frame* center_kf,
                                 set<Frame*>* core_kfs,
                                 Map* map,
                                 size_t& n_incorrect_edges_1,
                                 size_t& n_incorrect_edges_2,
                                 double& init_error,
                                 double& final_error)
{
    bool flag = 1;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setMaxTrialsAfterFailure(5);
    solver->setUserLambdaInit(1e-5);

    optimizer.setAlgorithm(solver);

    list<EdgeFrameFeature*> edges;
    list<EdgeLetFrameFeature*> edgeLets;

    set<Point*> mps;
    std::set<Frame*> kf_set;

    for(set<Frame*>::iterator it_kf = core_kfs->begin(); it_kf != core_kfs->end(); ++it_kf)
    {
        Frame* kf = *it_kf;
        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        optimizer.addVertex(VP);
        if(kf->id_ == 0 || kf->keyFrameId_+20 < center_kf->keyFrameId_)
            VP->setFixed(true);
        else
        {
            VP->setFixed(false);
        }

        kf_set.insert(kf);
        kf->lba_id_ = center_kf->id_;

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;

            assert((*it_pt)->point->type_ != Point::TYPE_CANDIDATE);
            mps.insert((*it_pt)->point);
        }
    }

    const int max_id = center_kf->id_;

    float huber_corner = 0, huber_edge = 0;
    double focal_length = center_kf->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = center_kf->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];

    computeVisualWeight(mps, focal_length, huber_corner, huber_edge);
   
    int n_edges = 0;
    int v_id = max_id+1;
    double error = 0;
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexPointID* vPoint = new VertexPointID();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        double idist = (*it_pt)->getPointIdist();
        vPoint->setEstimate(idist);
        vPoint->setMarginalized(true);
        assert(optimizer.addVertex(vPoint));
        (*it_pt)->lba_id_ = center_kf->id_;

        if((*it_pt)->hostFeature_->frame->lba_id_ != center_kf->id_)
        {
            (*it_pt)->hostFeature_->frame->lba_id_ = center_kf->id_;
            VertexPose* vHost = new VertexPose((*it_pt)->hostFeature_->frame);
            vHost->setId((*it_pt)->hostFeature_->frame->id_);
            vHost->setFixed(true);
            assert(optimizer.addVertex(vHost));

            kf_set.insert( (*it_pt)->hostFeature_->frame);
        }

        list<Feature*> obs = (*it_pt)->getObs();
        list<Feature*>::iterator it_obs = obs.begin();
        while(it_obs != obs.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            if((*it_obs)->frame->lba_id_ != center_kf->id_)
            {
                (*it_obs)->frame->lba_id_ = center_kf->id_;
                VertexPose* vTarget = new VertexPose((*it_obs)->frame);
                vTarget->setId((*it_obs)->frame->id_);
                vTarget->setFixed(true);
                assert(optimizer.addVertex(vTarget));

                kf_set.insert((*it_obs)->frame);
            }

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIdistCorner* edge = new EdgeIdistCorner();
                edge->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));
                edge->setHostBearing((*it_pt)->hostFeature_->f);
                edge->setCameraParams(fxy);

                edge->setMeasurement(Fxy * vilo::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(Eigen::Matrix2d::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_corner);
                edge->setRobustKernel(rk);

                edge->setParameterId(0, 0);

                edges.push_back(new EdgeFrameFeature(edge, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edge));

                edge->computeError();
                error += edge->chi2();
            }

            else
            {
                EdgeIdistEdgeLet* edgeLet = new EdgeIdistEdgeLet();
                edgeLet->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f);
                edgeLet->setTargetNormal((*it_obs)->grad);
                edgeLet->setCameraParams(fxy);

                edgeLet->setMeasurement((*it_obs)->grad.transpose()* Fxy* vilo::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edgeLet->setInformation(Eigen::Matrix<double,1,1>::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                edgeLet->setRobustKernel(rk);

                edgeLet->setParameterId(0, 0);

                edgeLets.push_back(new EdgeLetFrameFeature(edgeLet, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edgeLet));
            }

            ++n_edges;
            ++it_obs;
        }

    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err_ini = optimizer.activeChi2();
    optimizer.optimize(20); 
    float err_end = optimizer.activeChi2();

    float ie = sqrt(err_ini);
    float ee = sqrt(err_end);

    boost::unique_lock<boost::mutex> lock(map->map_mutex_);
    for(auto it=kf_set.begin();it!=kf_set.end();++it)
    {
        Frame* kf = *it;
        if(kf->lba_id_ == center_kf->id_)
            kf->lba_id_ = -1;
    }
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        if((*it_pt)->lba_id_ == center_kf->id_)
            (*it_pt)->lba_id_ = -1;
    }


    for(set<Frame*>::iterator it = core_kfs->begin(); it != core_kfs->end(); ++it)
    {
        VertexPose* VPose = static_cast<VertexPose*>(optimizer.vertex( (*it)->id_ ) );
        Sophus::SE3 Tfw( VPose->estimate().Rcw[0], VPose->estimate().tcw[0]);
        (*it)->setFramePose(Tfw);

        map->point_candidates_.changeCandidatePosition(*it);
    }

    v_id = max_id + 1;
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it,++v_id)
    {
        VertexPointID* VPoint = static_cast<VertexPointID*>(optimizer.vertex(v_id) );

        double idist = VPoint->estimate();
        (*it)->setPointIdistAndPose(idist);
    }

    const double reproj_thresh_2 = 2.0;
    const double reproj_thresh_1 = 1.2;

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeFrameFeature*>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if( (*it)->feature->point == NULL) continue;

        if((*it)->edge->chi2() > reproj_thresh_2_squared)
        {
            if((*it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                (*it)->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef((*it)->frame, (*it)->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeLetFrameFeature*>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if((*it)->feature->point == NULL) continue;


        if((*it)->edge->chi2() > reproj_thresh_1_squared)
        {
            if((*it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                (*it)->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef((*it)->frame, (*it)->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }

    init_error = ie;
    final_error = ee;
    map->setMapChangedFlag(true);
}

void visualMapOptimization(Map* map,
                           std::vector<Sophus::SE3>& lidar_pose_vec,
                           Sophus::SE3& Tbc)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setMaxTrialsAfterFailure(5);
    optimizer.setAlgorithm(solver);

    list<EdgeFrameFeature*> edges;
    list<EdgeLetFrameFeature*> edgeLets;
    set<Point*> mps;
    std::vector<FramePtr> frame_vec;
    
    size_t n_lidar = lidar_pose_vec.size();
    for(auto it=map->keyframes_.begin();it!=map->keyframes_.end();++it)
    {
        FramePtr kf = *it;
        frame_vec.push_back(kf);

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL) 
                continue;
            
            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
            mps.insert((*it_pt)->point);
        }

    }

    for(size_t i=0,iend=frame_vec.size();i<iend;i++)
    {
        if(i<=n_lidar-1)
        {
            Sophus::SE3 Twb = lidar_pose_vec[i];
            Eigen::Matrix3d Rwc = (Twb*Tbc).rotation_matrix();
            Eigen::Vector3d twc = (Twb*Tbc).translation();
            Frame* kf = frame_vec[i].get();

            VertexPose * VP = new VertexPose(Rwc, twc, kf);
            VP->setId(kf->id_);
            optimizer.addVertex(VP);
            VP->setFixed(false);
        }
        else
        {
            Frame* kf = frame_vec[i].get();

            VertexPose * VP = new VertexPose(kf);
            VP->setId(kf->id_);
            optimizer.addVertex(VP);
            VP->setFixed(false);
        }


    }
    FramePtr last_kf = frame_vec.back();
    const int max_id = last_kf->id_;

    float huber_corner = 0, huber_edge = 0;
    double focal_length = last_kf->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = last_kf->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];
    computeVisualWeight(mps, focal_length, huber_corner, huber_edge);

    int n_edges = 0;
    int v_id = max_id+1;
    double error = 0;
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexPointID* vPoint = new VertexPointID();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        double idist = (*it_pt)->getPointIdist();
        vPoint->setEstimate(idist);
        assert(optimizer.addVertex(vPoint));

        list<Feature*> obs = (*it_pt)->getObs();
        list<Feature*>::iterator it_obs = obs.begin();
        while(it_obs != obs.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIdistCorner* edge = new EdgeIdistCorner();
                edge->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edge->setHostBearing((*it_pt)->hostFeature_->f);
                edge->setCameraParams(fxy);
                edge->setMeasurement(Fxy * vilo::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(Eigen::Matrix2d::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_corner);
                edge->setRobustKernel(rk);
                edge->setParameterId(0, 0);

                edges.push_back(new EdgeFrameFeature(edge, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edge));

                edge->computeError();
                error += edge->chi2();
            }
            else
            {
                EdgeIdistEdgeLet* edgeLet = new EdgeIdistEdgeLet();
                edgeLet->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f);
                edgeLet->setTargetNormal((*it_obs)->grad);
                edgeLet->setCameraParams(fxy);
                edgeLet->setMeasurement((*it_obs)->grad.transpose()* Fxy* vilo::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edgeLet->setInformation(Eigen::Matrix<double,1,1>::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                edgeLet->setRobustKernel(rk);
                edgeLet->setParameterId(0, 0);
                edgeLets.push_back(new EdgeLetFrameFeature(edgeLet, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edgeLet));
            }

            ++n_edges;
            ++it_obs;

        }

    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err_ini = optimizer.activeChi2();
    optimizer.optimize(10); 
    float err_end = optimizer.activeChi2();

    float ie = sqrt(err_ini);
    float ee = sqrt(err_end);
    
    for(size_t i=0,iend=frame_vec.size();i<iend;i++)
    {
        Frame* kf = frame_vec[i].get();
        VertexPose* VPose = static_cast<VertexPose*>(optimizer.vertex( kf->id_ ) );
        Sophus::SE3 Tfw( VPose->estimate().Rcw[0], VPose->estimate().tcw[0]);
        kf->setFramePose(Tfw);

        map->point_candidates_.changeCandidatePosition(kf);
    }

    v_id = max_id + 1;
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it,++v_id)
    {
        VertexPointID* VPoint = static_cast<VertexPointID*>(optimizer.vertex(v_id) );

        double idist = VPoint->estimate();
        (*it)->setPointIdistAndPose(idist);
    }

    int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2 = 2.0; 
    const double reproj_thresh_1 = 1.2; 

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeFrameFeature*>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if( (*it)->feature->point == NULL) continue;

        if((*it)->edge->chi2() > reproj_thresh_2_squared)
        {
            if((*it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                (*it)->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef((*it)->frame, (*it)->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeLetFrameFeature*>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if((*it)->feature->point == NULL) continue;

        if((*it)->edge->chi2() > reproj_thresh_1_squared)
        {
            if((*it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                (*it)->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef((*it)->frame, (*it)->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }

}


void visualImuFullBundleAdjustment(Map* map, bool is_opt_each_bias, double priorG, double priorA, int n_its)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if(priorA != 0 && !is_opt_each_bias)
        solver->setUserLambdaInit(1e-5);
    else
        solver->setUserLambdaInit(1e-1);

    optimizer.setAlgorithm(solver);

    std::set<Point*> mps;
    std::vector<Frame*> kf_vec;
    list< FramePtr>::iterator it = map->keyframes_.begin();
    while(it != map->keyframes_.end())
    {
        kf_vec.push_back((*it).get());
        ++it;
    }
    size_t N = kf_vec.size();
    long max_id = kf_vec[N-1]->id_;

    for(size_t i=0; i<N; i++)
    {
        Frame* kf = kf_vec[i];

        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        VP->setFixed(false);

        optimizer.addVertex(VP);

        if(kf->is_opt_imu_)
        {
            VertexVelocity* VV = new VertexVelocity(kf);
            VV->setId( max_id + 3*kf->id_ + 1 );
            VV->setFixed(false);

            optimizer.addVertex(VV);

            if(is_opt_each_bias)
            {
                VertexGyroBias* VG = new VertexGyroBias(kf);
                VG->setId(max_id + 3*kf->id_ + 2);
                VG->setFixed(false);

                optimizer.addVertex(VG);

                VertexAccBias* VA = new VertexAccBias(kf);
                VA->setId(max_id + 3*kf->id_ + 3);
                VA->setFixed(false);

                optimizer.addVertex(VA);
            }

        }

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;

            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;

            mps.insert((*it_pt)->point);
        }


    }
    
    if(!is_opt_each_bias)
    {
        Frame* last_kf = kf_vec[N-1];
        VertexGyroBias* VG = new VertexGyroBias(last_kf);
        VG->setId(4*max_id + 2);
        VG->setFixed(false);
        optimizer.addVertex(VG);

        VertexAccBias* VA = new VertexAccBias(last_kf);
        VA->setId(4*max_id + 3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    double error = 0;
    for(size_t i=0; i<N; i++)
    {
        Frame* kf = kf_vec[i];
        if(kf->id_>max_id)
            continue;
        if(!kf->last_kf_)
            continue;
        if(kf->last_kf_->id_ > max_id)
            continue;

        if(!kf->imu_from_last_keyframe_ || !kf->is_opt_imu_ || !kf->last_kf_->is_opt_imu_)
            continue;

        Eigen::Vector3d new_bg = kf->last_kf_->getImuGyroBias();
        Eigen::Vector3d new_ba = kf->last_kf_->getImuAccBias();
        kf->imu_from_last_keyframe_->setNewBias(new_ba, new_bg);

        g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(kf->last_kf_->id_);
        g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(max_id + 3*kf->last_kf_->id_+1);
        g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(kf->id_);
        g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(max_id + 3*kf->id_+1);

        g2o::HyperGraph::Vertex* VG1;
        g2o::HyperGraph::Vertex* VA1;
        g2o::HyperGraph::Vertex* VG2;
        g2o::HyperGraph::Vertex* VA2;

        if (is_opt_each_bias)
        {
            VG1 = optimizer.vertex(max_id + 3*kf->last_kf_->id_+2);
            VA1 = optimizer.vertex(max_id + 3*kf->last_kf_->id_+3);
            VG2 = optimizer.vertex(max_id + 3*kf->id_+2);
            VA2 = optimizer.vertex(max_id + 3*kf->id_+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cout << "Error:" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            EdgeGyroRW* egr= new EdgeGyroRW();
            egr->setVertex(0,VG1);
            egr->setVertex(1,VG2);

            Eigen::Matrix3d InfoG = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(9,9).inverse();
            egr->setInformation(InfoG);
            egr->computeError();
            optimizer.addEdge(egr);

            EdgeAccRW* ear = new EdgeAccRW();
            ear->setVertex(0,VA1);
            ear->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(12,12).inverse();
            ear->setInformation(InfoA);
            ear->computeError();
            optimizer.addEdge(ear);
        }
        else
        {
            VG1 = optimizer.vertex(4*max_id + 2);
            VA1 = optimizer.vertex(4*max_id + 3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
            {
                cout << "Error:" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;
                continue;
            }
        }

        EdgeInertial* ei = new EdgeInertial(kf->imu_from_last_keyframe_, false);
        ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
        ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
        ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
        ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
        ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
        ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

        g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
        ei->setRobustKernel(rki);
        rki->setDelta(sqrt(16.92));
        optimizer.addEdge(ei);

        ei->computeError();
    }


    if(!is_opt_each_bias)
    {
        g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*max_id+2);
        g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*max_id+3);

        EdgePriorAcc* epa = new EdgePriorAcc(Eigen::Vector3d(0,0,0));
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA;
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro* epg = new EdgePriorGyro(Eigen::Vector3d(0,0,0));
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG;
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        epa->computeError();

        epg->computeError();
    }

    float huber_corner = 0, huber_edge = 0;
    double focal_length = kf_vec[N-1]->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = kf_vec[N-1]->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];

    computeVisualWeight(mps, focal_length, huber_corner, huber_edge);

    list<EdgeFrameFeature*> edges;
    list<EdgeLetFrameFeature*> edgeLets;
    int n_v_edges = 0;
    unsigned long v_id = 4*max_id+4;
    double error_v = 0;

    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexPointID* VPoint = new VertexPointID();
        VPoint->setId(v_id++);
        VPoint->setFixed(false);
        double idist = (*it_pt)->getPointIdist();
        VPoint->setEstimate(idist);
        VPoint->setMarginalized(true);
        assert(optimizer.addVertex(VPoint));

        g2o::HyperGraph::Vertex* VPH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
        list<Feature*> obs = (*it_pt)->getObs();
        list<Feature*>::iterator it_obs = obs.begin();
        while(it_obs != obs.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            g2o::HyperGraph::Vertex* VPT = optimizer.vertex((*it_obs)->frame->id_);
            if(!VPH || !VPT || !VPoint)
            {
                cout << "Error:" << VPH << ", "<< VPT << ", "<< VPoint << endl;
                ++it_obs;
                continue;
            }

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIdistCorner* edge = new EdgeIdistCorner();
                edge->resize(3);

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPoint));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPH));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPT));

                edge->setHostBearing((*it_pt)->hostFeature_->f);
                edge->setCameraParams(fxy);
                edge->setMeasurement(Fxy* vilo::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(Eigen::Matrix2d::Identity()*0.5);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(sqrt(5.991));
                edge->setRobustKernel(rk);
                edge->setParameterId(0, 0);
                edges.push_back(new EdgeFrameFeature(edge, (*it_obs)->frame, *it_obs));

                assert(optimizer.addEdge(edge));

                edge->computeError();
                error_v += edge->chi2();
            }

            else
            {
                EdgeIdistEdgeLet* edgeLet = new EdgeIdistEdgeLet();
                edgeLet->resize(3);

                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPoint));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPH));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VPT));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f);
                edgeLet->setTargetNormal((*it_obs)->grad);
                edgeLet->setCameraParams(fxy);
                edgeLet->setMeasurement((*it_obs)->grad.transpose()* Fxy* vilo::project2d((*it_obs)->f));
                
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edgeLet->setInformation(Eigen::Matrix<double,1,1>::Identity()*0.5);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(sqrt(5.991));
                edgeLet->setRobustKernel(rk);

                edgeLet->setParameterId(0, 0);

                edgeLets.push_back(new EdgeLetFrameFeature(edgeLet, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edgeLet));
            }

            ++n_v_edges;
            ++it_obs;
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err_ini = optimizer.activeChi2();
    optimizer.setVerbose(false);
    optimizer.optimize(n_its); 
    float err_end = optimizer.activeChi2();

    boost::unique_lock<boost::mutex> lock(map->map_mutex_);
    for(size_t i=0; i<N; i++)
    {
        Frame* kf = kf_vec[i];
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(kf->id_));

        Sophus::SE3 Tfw( VP->estimate().Rcw[0], VP->estimate().tcw[0]);
        kf->setFramePose(Tfw);
        if(kf->is_opt_imu_)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_id + 3*kf->id_+1));
            kf->setVelocity( VV->estimate() );

            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (is_opt_each_bias)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_id + 3*kf->id_+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(max_id + 3*kf->id_+3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*max_id+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4*max_id+3));
            }

            Eigen::Vector3d new_ba = VA->estimate();
            Eigen::Vector3d new_bg = VG->estimate();

            kf->setNewBias(new_ba, new_bg);
        }
      
    }

    v_id = 4*max_id + 4;
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it,++v_id)
    {
        VertexPointID* VPoint = static_cast<VertexPointID*>(optimizer.vertex(v_id) );

        double idist = VPoint->estimate();
        (*it)->setPointIdistAndPose(idist);
    }

    int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2 = 2.0; 
    const double reproj_thresh_1 = 1.2; 

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeFrameFeature*>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if( (*it)->feature->point == NULL) continue;

        if((*it)->edge->chi2() > reproj_thresh_2_squared)
        {
            if((*it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                (*it)->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef((*it)->frame, (*it)->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeLetFrameFeature*>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if((*it)->feature->point == NULL) continue;


        if((*it)->edge->chi2() > reproj_thresh_1_squared)
        {
            if((*it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                (*it)->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef((*it)->frame, (*it)->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }
}

void visualImuOnePoseOptimization(FramePtr& frame,
                                  double& sfba_error_init,
                                  double& sfba_error_final,
                                  size_t& sfba_n_edges_final)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    VertexPose * VP = new VertexPose(frame.get());
    VP->setId(0);
    optimizer.addVertex(VP);
    VP->setFixed(false);

    VertexVelocity* VV = new VertexVelocity(frame.get());
    VV->setId( 1 );
    VV->setFixed(false);
    optimizer.addVertex(VV);

    VertexGyroBias* VG = new VertexGyroBias(frame.get());
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(frame.get());
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    EdgeInertial* ei = NULL;
    EdgeGyroRW* egr = NULL;
    EdgeAccRW* ear = NULL;
    EdgePriorPoseImu* ep = NULL;
    if(frame->m_last_frame)
    {
        FramePtr last_frame = frame->m_last_frame;

        VertexPose* VPk = new VertexPose(last_frame.get());
        VPk->setId(4);
        VPk->setFixed(false);
        optimizer.addVertex(VPk);

        VertexVelocity* VVk = new VertexVelocity(last_frame.get());
        VVk->setId(5);
        VVk->setFixed(false);
        optimizer.addVertex(VVk);

        VertexGyroBias* VGk = new VertexGyroBias(last_frame.get());
        VGk->setId(6);
        VGk->setFixed(false);
        optimizer.addVertex(VGk);

        VertexAccBias* VAk = new VertexAccBias(last_frame.get());
        VAk->setId(7);
        VAk->setFixed(false);
        optimizer.addVertex(VAk);

        ei = new EdgeInertial(frame->imu_from_last_frame_,false);
        
        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        optimizer.addEdge(ei);

        egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        Eigen::Matrix3d InfoG = frame->imu_from_last_frame_->cov_mat_.block<3,3>(9,9).inverse();
        egr->setInformation(0.01*InfoG);
        optimizer.addEdge(egr);

        ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        Eigen::Matrix3d InfoA = frame->imu_from_last_frame_->cov_mat_.block<3,3>(12,12).inverse();
        ear->setInformation(0.01*InfoA);
        optimizer.addEdge(ear);

        ear->computeError();
        egr->computeError();
        ei->computeError();
        
        if(frame->m_last_frame->prior_constraint_)
        {
            ep = new EdgePriorPoseImu(frame->m_last_frame->prior_constraint_);
            ep->setVertex(0,VPk);
            ep->setVertex(1,VVk);
            ep->setVertex(2,VGk);
            ep->setVertex(3,VAk);
            g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
            ep->setRobustKernel(rkp);
            rkp->setDelta(5);
            optimizer.addEdge(ep);

            ep->computeError();
        }
        
    }

    std::vector<float> errors_ls, errors_pt;
    Sophus::SE3 Tfw_t = frame->getFramePose();
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        double idist = ft->point->getPointIdist();
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/idist);
        Sophus::SE3 Tfw_h = host->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = vilo::project2d(ft->f) - vilo::project2d(pTarget);
        e *= 1.0 / (1<<ft->level);

        if(ft->type == Feature::EDGELET)
        {
            float error_ls = ft->grad.transpose()*e;
            errors_ls.push_back(fabs(error_ls));
        }
        else
        {
            float error_pt = e.norm();
            errors_pt.push_back(error_pt);
        }

    }

    if(errors_pt.empty() && errors_ls.empty()) return;

    vilo::robust_cost::MADScaleEstimator scale_estimator;

    float huber_corner = 0, huber_edge = 0;
    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = scale_estimator.compute(errors_pt);
        huber_edge = scale_estimator.compute(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = scale_estimator.compute(errors_pt);
        huber_edge = 0.5*huber_corner;
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_edge = scale_estimator.compute(errors_ls);
        huber_corner = 2*huber_edge;
    }
    else
    {
        assert(false);
    }

    double focal_length = frame->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = frame->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];
    huber_corner *= focal_length;
    huber_edge *= focal_length;

    int edge_vision = 0; double vision_chi2 = 0;
    std::vector<EdgeCornerOnlyPose*> corner_ftr_vec; corner_ftr_vec.reserve(frame->fts_.size());
    std::vector<EdgeEdgeletOnlyPose*> edge_ftr_vec; edge_ftr_vec.reserve(frame->fts_.size());
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        double idist = ft->point->getPointIdist();
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/idist);
        Sophus::SE3 Tfw_h = host->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
        Vector3d pTarget = Tth * pHost;
        Eigen::Vector3d wP = Tfw_t.inverse() * pTarget;

        if(ft->type != Feature::EDGELET)
        {
            EdgeCornerOnlyPose* e = new EdgeCornerOnlyPose(wP,0);

            e->setVertex(0,VP);
            e->setMeasurement(Fxy* vilo::project2d(ft->f));
            e->setInformation(0.1*Eigen::Matrix2d::Identity());
            e->setCameraParams(fxy);
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(sqrt(5.991));

            optimizer.addEdge(e);
            corner_ftr_vec.push_back(e);

            edge_vision++;
            e->computeError();
            vision_chi2 += e->chi2();
        }
        else
        {
            EdgeEdgeletOnlyPose* e = new EdgeEdgeletOnlyPose(wP,0);

            e->setVertex(0,VP);
            e->setMeasurement(ft->grad.transpose() * Fxy* vilo::project2d(ft->f));
            e->setInformation(0.1*Eigen::Matrix<double,1,1>::Identity());
            e->setTargetNormal(ft->grad);
            e->setCameraParams(fxy);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(sqrt(5.991));

            optimizer.addEdge(e);
            edge_ftr_vec.push_back(e);

            edge_vision++;
            e->computeError();
            vision_chi2 += e->chi2();
        }

    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    sfba_error_init = optimizer.activeChi2();
    optimizer.optimize(10); 
    sfba_error_final = optimizer.activeChi2();

    Sophus::SE3 T(VP->estimate().Rcw[0],VP->estimate().tcw[0]);
    frame->setFramePose(T);
    frame->setVelocity( VV->estimate() );
    frame->setImuAccBias( VA->estimate() );
    frame->setImuGyroBias( VG->estimate() );
    
    if(frame->m_last_frame)
    {
        Eigen::Matrix<double,30,30> H;
        H.setZero();

        H.block<24,24>(0,0)+= ei->GetHessian();

        Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
        H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
        H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
        H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
        H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

        Eigen::Matrix<double,6,6> Har = ear->GetHessian();
        H.block<3,3>(12,12) += Har.block<3,3>(0,0);
        H.block<3,3>(12,27) += Har.block<3,3>(0,3);
        H.block<3,3>(27,12) += Har.block<3,3>(3,0);
        H.block<3,3>(27,27) += Har.block<3,3>(3,3);

        if(frame->m_last_frame->prior_constraint_)
            H.block<15,15>(0,0) += ep->GetHessian();

        for(size_t i=0, iend=corner_ftr_vec.size(); i<iend; i++)
        {
            EdgeCornerOnlyPose* e = corner_ftr_vec[i];

            if(e->chi2()<=5.99)
                H.block<6,6>(15,15) += e->GetHessian();
        }
        for(size_t i=0, iend=edge_ftr_vec.size(); i<iend; i++)
        {
            EdgeEdgeletOnlyPose* e = edge_ftr_vec[i];

            if(e->chi2()<=5.99 * 0.7)
                H.block<6,6>(15,15) += e->GetHessian();
        }

        H = marginalize(H,0,14);

        frame->prior_constraint_ = new IMUPriorConstraint(VP->estimate().Rwb,
                                                          VP->estimate().twb,
                                                          VV->estimate(),
                                                          VG->estimate(),
                                                          VA->estimate(),
                                                          H.block<15,15>(15,15));
        delete frame->m_last_frame->prior_constraint_;
        frame->m_last_frame->prior_constraint_ = NULL;
    }


    int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2 = 1.0;
    const double reproj_thresh_1 = 0.5;

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    Tfw_t = frame->getFramePose();
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        double idist = ft->point->getPointIdist();
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/idist);
        Sophus::SE3 Tfw_h = host->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
        Vector3d pTarget = Tth * pHost;
        if(ft->type != Feature::EDGELET)
        {
            Eigen::Vector2d e = vilo::project2d(ft->f) -  vilo::project2d(pTarget);
            double chi2 = e[0]*e[0] + e[1]*e[1];
            if(chi2 >= reproj_thresh_2_squared)
                ++n_incorrect_edges_1;
        }
        else
        {
            double e = ft->grad.transpose() * ( vilo::project2d(ft->f) -  vilo::project2d(pTarget) );
            double chi2 = e*e;
            if(chi2 >= reproj_thresh_1_squared)
                ++n_incorrect_edges_2;
        }

    }
    sfba_n_edges_final = size_t( edge_vision - n_incorrect_edges_1 - n_incorrect_edges_2);
}

void visualImuOnePoseOptimizationFromKF(FramePtr& frame,
                                        double& sfba_error_init,
                                        double& sfba_error_final,
                                        size_t& sfba_n_edges_final)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    VertexPose * VP = new VertexPose(frame.get());
    VP->setId(0);
    optimizer.addVertex(VP);
    VP->setFixed(false);

    VertexVelocity* VV = new VertexVelocity(frame.get());
    VV->setId( 1 );
    VV->setFixed(false);
    optimizer.addVertex(VV);

    VertexGyroBias* VG = new VertexGyroBias(frame.get());
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(frame.get());
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    EdgeInertial* ei = NULL;
    EdgeGyroRW* egr = NULL;
    EdgeAccRW* ear = NULL;
    EdgePriorPoseImu* ep = NULL;
    if(frame->last_kf_)
    {
        Frame* last_kf = frame->last_kf_;

        VertexPose* VPk = new VertexPose(last_kf);
        VPk->setId(4);
        VPk->setFixed(true);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(last_kf);
        VVk->setId(5);
        VVk->setFixed(true);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(last_kf);
        VGk->setId(6);
        VGk->setFixed(true);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(last_kf);
        VAk->setId(7);
        VAk->setFixed(true);
        optimizer.addVertex(VAk);

        ei = new EdgeInertial(frame->imu_from_last_keyframe_,true);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        ei->setInformation(0.1*ei->information() );
        optimizer.addEdge(ei);

        egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        Eigen::Matrix3d InfoG = frame->imu_from_last_keyframe_->cov_mat_.block<3,3>(9,9).inverse();
        egr->setInformation(0.01*InfoG);
        optimizer.addEdge(egr);

        ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        Eigen::Matrix3d InfoA = frame->imu_from_last_keyframe_->cov_mat_.block<3,3>(12,12).inverse();
        ear->setInformation(0.01*InfoA);
        optimizer.addEdge(ear);

        ear->computeError();
        egr->computeError();
        ei->computeError();

        Eigen::Vector3d lba = last_kf->getImuAccBias();
        Eigen::Vector3d lbg = last_kf->getImuGyroBias();
        Eigen::Vector3d cba = frame->getImuAccBias();
        Eigen::Vector3d cbg = frame->getImuGyroBias();
    }

    std::vector<float> errors_ls, errors_pt;
    Sophus::SE3 Tfw_t = frame->getFramePose();
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        double idist = ft->point->getPointIdist();
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/idist);
        Sophus::SE3 Tfw_h = host->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = vilo::project2d(ft->f) - vilo::project2d(pTarget);
        e *= 1.0 / (1<<ft->level);

        if(ft->type == Feature::EDGELET)
        {
            float error_ls = ft->grad.transpose()*e;
            errors_ls.push_back(fabs(error_ls));
        }
        else
        {
            float error_pt = e.norm();
            errors_pt.push_back(error_pt);
        }

    }

    if(errors_pt.empty() && errors_ls.empty()) return;

    vilo::robust_cost::MADScaleEstimator scale_estimator;

    float huber_corner = 0, huber_edge = 0;
    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = scale_estimator.compute(errors_pt);
        huber_edge = scale_estimator.compute(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = scale_estimator.compute(errors_pt);
        huber_edge = 0.5*huber_corner;
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_edge = scale_estimator.compute(errors_ls);
        huber_corner = 2*huber_edge;
    }
    else
    {
        assert(false);
    }


    double focal_length = frame->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = frame->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];
    huber_corner *= focal_length;
    huber_edge *= focal_length;

    int edge_vision = 0; double vision_chi2 = 0;
    std::vector<EdgeCornerOnlyPose*> corner_ftr_vec; corner_ftr_vec.reserve(frame->fts_.size());
    std::vector<EdgeEdgeletOnlyPose*> edge_ftr_vec; edge_ftr_vec.reserve(frame->fts_.size());
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        double idist = ft->point->getPointIdist();
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/idist);
        Sophus::SE3 Tfw_h = host->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
        Vector3d pTarget = Tth * pHost;
        Eigen::Vector3d wP = Tfw_t.inverse() * pTarget;

        if(ft->type != Feature::EDGELET)
        {
            EdgeCornerOnlyPose* e = new EdgeCornerOnlyPose(wP,0);

            e->setVertex(0,VP);
            e->setMeasurement(Fxy* vilo::project2d(ft->f));
            e->setInformation(0.1 * Eigen::Matrix2d::Identity());
            e->setCameraParams(fxy);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(sqrt(5.991));

            e->computeError();
            if(e->chi2() >= 500)
                continue;
            optimizer.addEdge(e);
            corner_ftr_vec.push_back(e);

            vision_chi2 += e->chi2();
            edge_vision++;

        }
        else
        {
            EdgeEdgeletOnlyPose* e = new EdgeEdgeletOnlyPose(wP,0);

            e->setVertex(0,VP);
            e->setMeasurement(ft->grad.transpose() * Fxy* vilo::project2d(ft->f));
            e->setInformation(0.1 * Eigen::Matrix<double,1,1>::Identity());
            e->setTargetNormal(ft->grad);
            e->setCameraParams(fxy);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(sqrt(5.991));

            e->computeError();
            if(e->chi2() >= 500)
                continue;
            optimizer.addEdge(e);
            edge_ftr_vec.push_back(e);

            vision_chi2 += e->chi2();
            edge_vision++;
        }

    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    sfba_error_init = optimizer.activeChi2();
    optimizer.optimize(10); 
    sfba_error_final = optimizer.activeChi2();

    Sophus::SE3 T(VP->estimate().Rcw[0],VP->estimate().tcw[0]);
    frame->setFramePose(T);
    frame->setVelocity( VV->estimate() );
    frame->setImuAccBias( VA->estimate() );
    frame->setImuGyroBias( VG->estimate() );
    
    if(frame->last_kf_)
    {
        Eigen::Matrix<double,15,15> H;
        H.setZero();

        H.block<9,9>(0,0)+= ei->GetHessian2();
        H.block<3,3>(9,9) += egr->GetHessian2();
        H.block<3,3>(12,12) += ear->GetHessian2();

        for(size_t i=0, iend=corner_ftr_vec.size(); i<iend; i++)
        {
            EdgeCornerOnlyPose* e = corner_ftr_vec[i];

            if(e->chi2()<=5.99)
                H.block<6,6>(0,0) += e->GetHessian();
        }
        for(size_t i=0, iend=edge_ftr_vec.size(); i<iend; i++)
        {
            EdgeEdgeletOnlyPose* e = edge_ftr_vec[i];

            if(e->chi2()<=5.99 * 0.7)
                H.block<6,6>(0,0) += e->GetHessian();
        }

        frame->prior_constraint_ = new IMUPriorConstraint(VP->estimate().Rwb,
                                                          VP->estimate().twb,
                                                          VV->estimate(),
                                                          VG->estimate(),
                                                          VA->estimate(),
                                                          H);
        delete frame->m_last_frame->prior_constraint_;
        frame->m_last_frame->prior_constraint_ = NULL;
    }


    int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2 = 1.0; 
    const double reproj_thresh_1 = 0.5; 

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    Tfw_t = frame->getFramePose();
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        double idist = ft->point->getPointIdist();
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/idist);
        Sophus::SE3 Tfw_h = host->getFramePose();
        Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
        Vector3d pTarget = Tth * pHost;
        if(ft->type != Feature::EDGELET)
        {
            Eigen::Vector2d e = vilo::project2d(ft->f) -  vilo::project2d(pTarget);
            double chi2 = e[0]*e[0] + e[1]*e[1];
            if(chi2 >= reproj_thresh_2_squared)
                ++n_incorrect_edges_1;
        }
        else
        {
            double e = ft->grad.transpose() * ( vilo::project2d(ft->f) -  vilo::project2d(pTarget) );
            double chi2 = e*e;
            if(chi2 >= reproj_thresh_1_squared)
                ++n_incorrect_edges_2;
        }

    }
}

void lidarImuOnePoseOptimization(LidarPtr& current_lidar,
                                 Sophus::SE3& Twl,
                                 std::vector<double>& d_vec,
                                 std::vector<Eigen::Vector3d>& pts_vec,
                                 std::vector<Eigen::Vector3d>& nml_vec)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    Sophus::SE3 Tlb = current_lidar->T_l_b_;
    Sophus::SE3 Tbl = Tlb.inverse();
    Sophus::SE3 Twb = Twl * Tlb;
    VertexPose * VP = new VertexPose(Twb);
    VP->setId(0);
    optimizer.addVertex(VP);
    VP->setFixed(false);

    Eigen::Vector3d velocity_cur = current_lidar->getVelocity();
    VertexVelocity* VV = new VertexVelocity(velocity_cur);
    VV->setId( 1 );
    VV->setFixed(false);
    optimizer.addVertex(VV);

    Eigen::Vector3d bg_cur = current_lidar->getImuGyroBias();
    VertexGyroBias* VG = new VertexGyroBias(bg_cur);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    Eigen::Vector3d ba_cur = current_lidar->getImuAccBias();
    VertexAccBias* VA = new VertexAccBias(ba_cur);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);
    
    EdgeInertial* ei = NULL;
    EdgeGyroRW* egr = NULL;
    EdgeAccRW* ear = NULL;
    EdgePriorPoseImu* ep = NULL;

    if(current_lidar->previous_lidar_)
    {
        Lidar* last_lidar = current_lidar->previous_lidar_;
        Sophus::SE3 Twb_last = last_lidar->getImuPose();
        VertexPose* VPk = new VertexPose(Twb_last);
        VPk->setId(4);
        VPk->setFixed(false);
        optimizer.addVertex(VPk);

        Eigen::Vector3d velocity_last = last_lidar->getVelocity();
        VertexVelocity* VVk = new VertexVelocity(velocity_last);
        VVk->setId(5);
        VVk->setFixed(false);
        optimizer.addVertex(VVk);

        Eigen::Vector3d bg_last = last_lidar->getImuGyroBias();
        VertexGyroBias* VGk = new VertexGyroBias(bg_last);
        VGk->setId(6);
        VGk->setFixed(false);
        optimizer.addVertex(VGk);

        Eigen::Vector3d ba_last = last_lidar->getImuAccBias();
        VertexAccBias* VAk = new VertexAccBias(ba_last);
        VAk->setId(7);
        VAk->setFixed(false);
        optimizer.addVertex(VAk);

        ei = new EdgeInertial(current_lidar->imu_from_last_lidar_,true);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        optimizer.addEdge(ei);

        egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        Eigen::Matrix3d InfoG = current_lidar->imu_from_last_lidar_->cov_mat_.block<3,3>(9,9).inverse();
        egr->setInformation(InfoG);
        optimizer.addEdge(egr);

        ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        Eigen::Matrix3d InfoA = current_lidar->imu_from_last_lidar_->cov_mat_.block<3,3>(12,12).inverse();
        ear->setInformation(InfoA);
        optimizer.addEdge(ear);

        if(current_lidar->previous_lidar_->prior_constraint_)
        {
            ep = new EdgePriorPoseImu(current_lidar->previous_lidar_->prior_constraint_);
            ep->setVertex(0,VPk);
            ep->setVertex(1,VVk);
            ep->setVertex(2,VGk);
            ep->setVertex(3,VAk);

            optimizer.addEdge(ep);
            ep->computeError();
        }

        ear->computeError();
        egr->computeError();
        ei->computeError();
    }
    
    std::vector<EdgeLidarPointPlane*> lidar_edges_vec;
    for(size_t i=0,iend=d_vec.size();i<iend;i++)
    {
        EdgeLidarPointPlane* e = new EdgeLidarPointPlane(nml_vec[i] , d_vec[i]);

        e->setVertex(0,VP);
        Eigen::Vector3d pt_b = Tbl * pts_vec[i];
        e->setPoint(pt_b);
        e->setInformation(Eigen::Matrix<double,1,1>::Identity());

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(0.5);

        optimizer.addEdge(e);
        lidar_edges_vec.push_back(e);
    }
    
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    double error_init = optimizer.activeChi2();
    optimizer.optimize(10); 
    double error_final = optimizer.activeChi2();
    Sophus::SE3 T(VP->estimate().Rwb, VP->estimate().twb);
    Twl = T * Tbl;

    if(ei)
    {
        Eigen::Vector3d ba = VA->estimate();
        Eigen::Vector3d bg = VG->estimate();
        Eigen::Vector3d vel = VV->estimate();
        current_lidar->setNewBias(ba, bg);
        current_lidar->setVelocity(vel);
    }
    
    if(ep)
    {
        Eigen::Matrix<double,30,30> H;
        H.setZero();

        H.block<24,24>(0,0)+= ei->GetHessian();

        Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
        H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
        H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
        H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
        H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

        Eigen::Matrix<double,6,6> Har = ear->GetHessian();
        H.block<3,3>(12,12) += Har.block<3,3>(0,0);
        H.block<3,3>(12,27) += Har.block<3,3>(0,3);
        H.block<3,3>(27,12) += Har.block<3,3>(3,0);
        H.block<3,3>(27,27) += Har.block<3,3>(3,3);

        H.block<15,15>(0,0) += ep->GetHessian();

        for(size_t i=0, iend=lidar_edges_vec.size(); i<iend; i++)
        {
            EdgeLidarPointPlane* e = lidar_edges_vec[i];
            H.block<6,6>(15,15) += e->GetHessian();
        }
        H = marginalize(H,0,14);

        current_lidar->prior_constraint_ = new IMUPriorConstraint(VP->estimate().Rwb,
                                                          VP->estimate().twb,
                                                          VV->estimate(),
                                                          VG->estimate(),
                                                          VA->estimate(),
                                                          H.block<15,15>(15,15));
        delete current_lidar->previous_lidar_->prior_constraint_;
        current_lidar->previous_lidar_->prior_constraint_ = NULL;
    }
}

void lidarImuOnePoseOptimizationFromKF(LidarPtr& current_lidar,
                                 Sophus::SE3& Twl,
                                 std::vector<double>& d_vec,
                                 std::vector<Eigen::Vector3d>& pts_vec,
                                 std::vector<Eigen::Vector3d>& nml_vec)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    Sophus::SE3 Tlb = current_lidar->T_l_b_;
    Sophus::SE3 Tbl = Tlb.inverse();
    Sophus::SE3 Twb = Twl * Tlb;
    VertexPose * VP = new VertexPose(Twb);
    VP->setId(0);
    optimizer.addVertex(VP);
    VP->setFixed(false);

    Eigen::Vector3d velocity_cur = current_lidar->getVelocity();
    VertexVelocity* VV = new VertexVelocity(velocity_cur);
    VV->setId( 1 );
    VV->setFixed(false);
    optimizer.addVertex(VV);

    Eigen::Vector3d bg_cur = current_lidar->getImuGyroBias();
    VertexGyroBias* VG = new VertexGyroBias(bg_cur);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    Eigen::Vector3d ba_cur = current_lidar->getImuAccBias();
    VertexAccBias* VA = new VertexAccBias(ba_cur);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    EdgeInertial* ei = NULL;
    EdgeGyroRW* egr = NULL;
    EdgeAccRW* ear = NULL;

    if(current_lidar->last_key_lidar_)
    {
        assert(current_lidar->last_key_lidar_->id_ == current_lidar->previous_lidar_->id_);
        Lidar* last_key_lidar = current_lidar->last_key_lidar_;
        Sophus::SE3 Twb_last = last_key_lidar->getImuPose();
        VertexPose* VPk = new VertexPose(Twb_last);
        VPk->setId(4);
        VPk->setFixed(true);
        optimizer.addVertex(VPk);

        Eigen::Vector3d velocity_last = last_key_lidar->getVelocity();
        VertexVelocity* VVk = new VertexVelocity(velocity_last);
        VVk->setId(5);
        VVk->setFixed(true);
        optimizer.addVertex(VVk);

        Eigen::Vector3d bg_last = last_key_lidar->getImuGyroBias();
        VertexGyroBias* VGk = new VertexGyroBias(bg_last);
        VGk->setId(6);
        VGk->setFixed(true);
        optimizer.addVertex(VGk);

        Eigen::Vector3d ba_last = last_key_lidar->getImuAccBias();
        VertexAccBias* VAk = new VertexAccBias(ba_last);
        VAk->setId(7);
        VAk->setFixed(true);
        optimizer.addVertex(VAk);

        if(current_lidar->imu_from_last_lidar_)
        {
            ei = new EdgeInertial(current_lidar->imu_from_last_lidar_,true);

            ei->setVertex(0, VPk);
            ei->setVertex(1, VVk);
            ei->setVertex(2, VGk);
            ei->setVertex(3, VAk);
            ei->setVertex(4, VP);
            ei->setVertex(5, VV);
            ei->setInformation(0.1*ei->information());
            optimizer.addEdge(ei);

            egr = new EdgeGyroRW();
            egr->setVertex(0,VGk);
            egr->setVertex(1,VG);
            Eigen::Matrix3d InfoG = current_lidar->imu_from_last_lidar_->cov_mat_.block<3,3>(9,9).inverse();
            egr->setInformation(0.1*InfoG);
            optimizer.addEdge(egr);

            ear = new EdgeAccRW();
            ear->setVertex(0,VAk);
            ear->setVertex(1,VA);
            Eigen::Matrix3d InfoA = current_lidar->imu_from_last_lidar_->cov_mat_.block<3,3>(12,12).inverse();
            ear->setInformation(0.1*InfoA);
            optimizer.addEdge(ear);

            ear->computeError();
            egr->computeError();
            ei->computeError();
        }

    }
    
    std::vector<EdgeLidarPointPlane*> lidar_edges_vec;
    for(size_t i=0,iend=d_vec.size();i<iend;i++)
    {
        EdgeLidarPointPlane* e = new EdgeLidarPointPlane(nml_vec[i] , d_vec[i]);

        e->setVertex(0,VP);
        Eigen::Vector3d pt_b = Tbl * pts_vec[i];
        e->setPoint(pt_b);
        e->setInformation(Eigen::Matrix<double,1,1>::Identity());

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(0.5);

        optimizer.addEdge(e);
        lidar_edges_vec.push_back(e);
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    double error_init = optimizer.activeChi2();
    optimizer.optimize(10); 
    double error_final = optimizer.activeChi2();

    Sophus::SE3 T(VP->estimate().Rwb, VP->estimate().twb);
    Twl = T * Tbl;
    
    if(ei)
    {
        Eigen::Vector3d ba = VA->estimate();
        Eigen::Vector3d bg = VG->estimate();
        Eigen::Vector3d vel = VV->estimate();
        current_lidar->setNewBias(ba, bg);
        current_lidar->setVelocity(vel);
    }

    Eigen::Matrix<double,15,15> H;
    H.setZero();
    H.block<9,9>(0,0)+= ei->GetHessian2();
    H.block<3,3>(9,9) += egr->GetHessian2();
    H.block<3,3>(12,12) += ear->GetHessian2();
    for(size_t i=0, iend=lidar_edges_vec.size(); i<iend; i++)
    {
        EdgeLidarPointPlane* e = lidar_edges_vec[i];
        H.block<6,6>(0,0) += e->GetHessian();
    }
    current_lidar->prior_constraint_ = new IMUPriorConstraint(VP->estimate().Rwb,
                                                        VP->estimate().twb,
                                                        VV->estimate(),
                                                        VG->estimate(),
                                                        VA->estimate(),
                                                        H);

    if(current_lidar->previous_lidar_->prior_constraint_)
    {
        assert(current_lidar->previous_lidar_->prior_constraint_ != NULL);
        delete current_lidar->previous_lidar_->prior_constraint_;
        current_lidar->previous_lidar_->prior_constraint_ = NULL;
    }
}


void lidarOnePoseOptimization(Sophus::SE3& Trc,
                                 std::vector<double>& d_vec,
                                 std::vector<Eigen::Vector3d>& pts_vec,
                                 std::vector<Eigen::Vector3d>& nml_vec)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    VertexPose * VP = new VertexPose(Trc);
    VP->setId(0);
    optimizer.addVertex(VP);
    VP->setFixed(false);

    Eigen::Matrix3d R;
    R.setIdentity();
    Eigen::Vector3d t(0.0, 0.0, 0.0);
    Sophus::SE3 Tr(R,t);

    VertexPose* VPk = new VertexPose(Tr);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);

    std::vector<EdgeLidarPointPlane*> edges_vec;
    std::vector<bool> outlier_vec;
    for(size_t i=0,iend=d_vec.size();i<iend;i++)
    {
        EdgeLidarPointPlane* e = new EdgeLidarPointPlane(nml_vec[i],d_vec[i]);

        e->setVertex(0,VP);
        e->setPoint(pts_vec[i]);
        e->setInformation(Eigen::Matrix<double,1,1>::Identity());


        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(4);

        optimizer.addEdge(e);
        edges_vec.push_back(e);
        outlier_vec.push_back(false);
    }

    int n_outlier = 0;
    for(int n_it=0; n_it<2; n_it++)
    {
        VP->setEstimate(Trc);
        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();
        double error_init = optimizer.activeChi2();
        optimizer.optimize(10);
        double error_final = optimizer.activeChi2();

        for(size_t i=0, iend=edges_vec.size(); i<iend; i++)
        {
            EdgeLidarPointPlane* e = edges_vec[i];

            if(!outlier_vec[i])
                e->computeError();

            const float chi2 = e->chi2();

            if(chi2>0.25)
            {
                outlier_vec[i]=true;
                e->setLevel(1);
                n_outlier++;
            }
            if(n_it==2)
                e->setRobustKernel(0);
        }
    }


    Sophus::SE3 T(VP->estimate().Rwb, VP->estimate().twb);
    Trc = T;
}

void lidarOnePoseOptimization2(Sophus::SE3& Trc,
                                 std::vector<double>& d_vec,
                                 std::vector<Eigen::Vector3d>& pts_vec1,
                                 std::vector<Eigen::Vector3d>& nml_vec,
                                 std::vector<Eigen::Vector3d>& line_vec1,
                                 std::vector<Eigen::Vector3d>& line_vec2,
                                 std::vector<Eigen::Vector3d>& pts_vec2)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    VertexPose * VP = new VertexPose(Trc);
    VP->setId(0);
    optimizer.addVertex(VP);
    VP->setFixed(false);

    Eigen::Matrix3d R;
    R.setIdentity();
    Eigen::Vector3d t(0.0, 0.0, 0.0);
    Sophus::SE3 Tr(R,t);

    VertexPose* VPk = new VertexPose(Tr);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);

    std::vector<EdgeLidarPointPlane*> edges_vec1;
    std::vector<bool> outlier_vec1;
    for(size_t i=0,iend=d_vec.size();i<iend;i++)
    {

        EdgeLidarPointPlane* e = new EdgeLidarPointPlane(nml_vec[i],d_vec[i]);

        e->setVertex(0,VP);
        e->setPoint(pts_vec1[i]);
        e->setInformation(Eigen::Matrix<double,1,1>::Identity());


        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(4);

        optimizer.addEdge(e);
        edges_vec1.push_back(e);
        outlier_vec1.push_back(false);
    }

    std::vector<EdgeLidarPointLine*> edges_vec2;
    std::vector<bool> outlier_vec2;
    for(size_t i=0,iend=line_vec1.size();i<iend;i++)
    {

        EdgeLidarPointLine* e = new EdgeLidarPointLine(line_vec1[i],line_vec2[i]);

        e->setVertex(0,VP);
        e->setPoint(pts_vec2[i]);
        e->setInformation(Eigen::Matrix<double,1,1>::Identity());


        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(1);

        optimizer.addEdge(e);
        edges_vec2.push_back(e);
        outlier_vec2.push_back(false);
    }

    int n_outlier = 0;
    for(int n_it=0; n_it<2; n_it++)
    {
        VP->setEstimate(Trc);
        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();
        double error_init = optimizer.activeChi2();
        optimizer.optimize(10);
        double error_final = optimizer.activeChi2();

        for(size_t i=0, iend=edges_vec1.size(); i<iend; i++)
        {
            EdgeLidarPointPlane* e = edges_vec1[i];

            if(!outlier_vec1[i])
                e->computeError();

            const float chi2 = e->chi2();

            if(chi2>0.25)
            {
                outlier_vec1[i]=true;
                e->setLevel(1);
                n_outlier++;
            }
            if(n_it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=edges_vec2.size(); i<iend; i++)
        {
            EdgeLidarPointLine* e = edges_vec2[i];

            if(!outlier_vec2[i])
                e->computeError();

            const float chi2 = e->chi2();

            if(chi2>0.25)
            {
                outlier_vec2[i]=true;
                e->setLevel(1);
                n_outlier++;
            }
            if(n_it==2)
                e->setRobustKernel(0);
        }
    }


    Sophus::SE3 T(VP->estimate().Rwb, VP->estimate().twb);
    Trc = T;
}

void lidarImuAlign(list<LidarPtr>& lidar_frames,
                    Eigen::Matrix3d& Rwg,
                    double& scale,
                    bool is_vel_fixed,
                    double priorG,
                    double priorA)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    if (priorG!=0.f)
        solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    int n_its = 200; 
    int max_id = lidar_frames.back()->id_;
    auto iter = lidar_frames.begin();
    while(iter!=lidar_frames.end())
    {
        LidarPtr lidar = *iter;
        if(lidar->id_ > max_id)
        {
            ++iter;
            continue;
        }

        Sophus::SE3 Twb = lidar->getImuPose();
        VertexPose * VP = new VertexPose(Twb);
        VP->setId(lidar->id_);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        Eigen::Vector3d vel = lidar->getVelocity();
        VertexVelocity* VV = new VertexVelocity(vel);
        VV->setId( max_id+(lidar->id_)+1 );
        if (is_vel_fixed)
            VV->setFixed(true);
        else
            VV->setFixed(false);

        optimizer.addVertex(VV);

        ++iter;
    }

    Eigen::Vector3d bg0 = lidar_frames.front()->getImuGyroBias();
    VertexGyroBias* VG = new VertexGyroBias(bg0);
    VG->setId(max_id*2+2);
    if (is_vel_fixed)
        VG->setFixed(true);
    else
        VG->setFixed(false);
    optimizer.addVertex(VG);

    Eigen::Vector3d ba0 = lidar_frames.front()->getImuAccBias();
    VertexAccBias* VA = new VertexAccBias(ba0);
    VA->setId(max_id*2+3);
    if (is_vel_fixed)
        VA->setFixed(true);
    else
        VA->setFixed(false);

    optimizer.addVertex(VA);

    EdgePriorAcc* epa = new EdgePriorAcc(Eigen::Vector3d(0,0,0));
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);

    EdgePriorGyro* epg = new EdgePriorGyro(Eigen::Vector3d(0,0,0));
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(max_id*2+4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);

    iter = lidar_frames.begin();
    while(iter!=lidar_frames.end())
    {
        if(iter==lidar_frames.begin())
        {
            ++iter;
            continue;
        }
        LidarPtr lidar = *iter;
        if(lidar->id_ < 10)
        {
            ++iter;
            continue;
        }
        
        if(lidar->last_key_lidar_ && lidar->id_<=max_id)
        {
            if(!lidar->imu_from_last_keylidar_)
            {
                ++iter;
                continue;
            }

            lidar->setImuAccBias(lidar->last_key_lidar_->getImuAccBias());
            lidar->setImuGyroBias(lidar->last_key_lidar_->getImuGyroBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(lidar->last_key_lidar_->id_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(max_id+(lidar->last_key_lidar_->id_)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(lidar->id_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(max_id+(lidar->id_)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(max_id*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(max_id*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(max_id*2+4);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir)
            {
                ++iter;
                continue;
            }

            EdgeInertialG* ei = new EdgeInertialG(lidar->imu_from_last_keylidar_);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            optimizer.addEdge(ei);

            ei->computeError();
        }
        ++iter;
    }

    std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err_ini = optimizer.activeRobustChi2();
    optimizer.optimize(20);
    float err_end = optimizer.activeRobustChi2();
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_id*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(max_id*2+3));
    Eigen::Vector3d ba = VA->estimate();
    Eigen::Vector3d bg = VG->estimate();
    Rwg = VGDir->estimate().Rwg;

    iter = lidar_frames.begin();
    while(iter!=lidar_frames.end())
    {
        LidarPtr lidar = *iter;
        if(lidar->id_ > max_id)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_id+(lidar->id_)+1));
        Eigen::Vector3d Vw = VV->estimate();
        lidar->setVelocity(Vw);

        if ( (lidar->getImuGyroBias()-bg).norm() >0.01)
        {
            lidar->setNewBias(ba,bg);
            if (lidar->imu_from_last_lidar_)
            {
                lidar->imu_from_last_lidar_->reintegrate();
            }
            if(lidar->imu_from_last_keylidar_)
            {
                lidar->imu_from_last_keylidar_->reintegrate();
            }
        }
        else
            lidar->setNewBias(ba,bg);

        ++iter;
    }
}

void alignLidarImagePosition(Eigen::Matrix3d& Rli, Eigen::Vector3d& tli, double& sli,
                             std::vector<Sophus::SE3>& image_pose_vec,
                             std::vector<Sophus::SE3>& lidar_pose_vec)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    int n_its = 200; 

    assert(image_pose_vec.size()==lidar_pose_vec.size());
    int N = image_pose_vec.size();

    g2o::Sim3 Slf;
    VertexSim3Expmap* sim3_vtx = new VertexSim3Expmap();
    sim3_vtx->setId(0);
    sim3_vtx->setFixed(false);
    sim3_vtx->setEstimate(Slf);
    optimizer.addVertex(sim3_vtx);

    for(int i=0;i<N;i++)
    {
        Sophus::SE3 lidar_pose = lidar_pose_vec[i];
        Sophus::SE3 image_pose = image_pose_vec[i];

        Eigen::Vector3d pl = lidar_pose.translation();
        Eigen::Vector3d pf = image_pose.translation();
        Eigen::Matrix3d fr = image_pose.rotation_matrix();
        Eigen::Matrix3d lr = lidar_pose.rotation_matrix();

        EdgeSim3XYZ* e_sim3 = new EdgeSim3XYZ(pl, pf);
        g2o::HyperGraph::Vertex* VSim3 = optimizer.vertex(0);
        e_sim3->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VSim3));
        e_sim3->setInformation(Eigen::Matrix3d::Identity());
        e_sim3->computeError();
        optimizer.addEdge(e_sim3);
    }

    double init_error, final_error;
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    init_error = optimizer.activeChi2();
    optimizer.setVerbose(false);
    optimizer.optimize(50);
    final_error = optimizer.activeChi2();

    g2o::VertexSim3Expmap* vSim3_new = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    Slf= vSim3_new->estimate();

    Eigen::Quaterniond qlf = Slf.rotation();
    Eigen::Vector3d tlf = Slf.translation();
    double slf = Slf.scale();

    Rli = Slf.rotation().toRotationMatrix();
    tli = Slf.translation();
    sli = Slf.scale();
}

void alignLidarImageOrientation(Eigen::Matrix3d& Ril,
                                std::vector<Sophus::SE3>& image_pose_vec,
                                std::vector<Sophus::SE3>& lidar_pose_vec)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    int n_its = 200; 
    assert(image_pose_vec.size()==lidar_pose_vec.size());
    int N = image_pose_vec.size();

    Eigen::Matrix3d Rlf_delt;
    Eigen::Matrix3d r1 = image_pose_vec.back().rotation_matrix();
    Eigen::Matrix3d r2 = lidar_pose_vec.back().rotation_matrix();
    Rlf_delt = r2 * r1.transpose();
    Rlf_delt.setIdentity();

    vilo::VertexRot* rot_vtx = new vilo::VertexRot();
    rot_vtx->setId(0);
    rot_vtx->setFixed(false);
    rot_vtx->setEstimate(vilo::Rot(Rlf_delt));
    optimizer.addVertex(rot_vtx);

    for(int i=0;i<N;i++)
    {
        Sophus::SE3 lidar_pose = lidar_pose_vec[i];
        Sophus::SE3 image_pose = image_pose_vec[i];

        Eigen::Matrix3d fr = image_pose.rotation_matrix();
        Eigen::Matrix3d lr = lidar_pose.rotation_matrix();

        Eigen::AngleAxisd fr_aa(fr);
        Eigen::AngleAxisd lr_aa(lr);

        EdgeRot* e_rot = new EdgeRot(lr, fr);
        g2o::HyperGraph::Vertex* VRot = optimizer.vertex(0);
        e_rot->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VRot));
        e_rot->setInformation(Eigen::Matrix3d::Identity());
        e_rot->computeError();
        optimizer.addEdge(e_rot);
    }

    double init_error, final_error;
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    init_error = optimizer.activeChi2();
    optimizer.setVerbose(false);
    optimizer.optimize(50);
    final_error = optimizer.activeChi2();

    vilo::VertexRot* vRot_new = static_cast<vilo::VertexRot*>(optimizer.vertex(0));
    Rlf_delt= vRot_new->estimate().R_;
    Ril = Rlf_delt.transpose(); 

    for(int i=0;i<N;i++)
    {
        Sophus::SE3 lidar_pose = lidar_pose_vec[i];
        Eigen::Matrix3d lr = lidar_pose.rotation_matrix();
        Eigen::Matrix3d new_lr = lr * Rlf_delt;

        Eigen::AngleAxisd lr_aa(new_lr);
    }
}


void lidarImuLocalBundleAdjustment(Map* map,
                                list<LidarPtr>& key_lidars_list,
                                std::vector <std::vector<Eigen::Vector3d>>& pts_vec_vec,
                                std::vector <std::vector<Eigen::Vector3d>>& nms_vec_vec,
                                std::vector <std::vector<double>>& ds_vec_vec,
                                bool is_opt_each_bias)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    std::vector<LidarPtr> key_lidars_vec;
    auto lit = key_lidars_list.begin();
    while(lit!=key_lidars_list.end())
    {
        key_lidars_vec.push_back(*lit);
        ++lit;
    }

    int n_its = 200;
    int max_id = key_lidars_vec.back()->id_;

    const int N = key_lidars_vec.size();
    Sophus::SE3 Tlb = key_lidars_vec.back()->T_l_b_;
    Sophus::SE3 Tbl = Tlb.inverse();

    for(int i=0; i<N; i++)
    {
        LidarPtr lidar = key_lidars_vec[i];
        if(lidar->id_ > max_id)
            continue;

        Sophus::SE3 Twb = lidar->getImuPose();
        VertexPose * VP = new VertexPose(Twb);
        VP->setId(lidar->id_);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(lidar->imu_from_last_keylidar_)
        {
            Eigen::Vector3d vel = lidar->getVelocity();
            VertexVelocity* VV = new VertexVelocity(vel);
            VV->setId( max_id + 3*lidar->id_ + 1 );
            VV->setFixed(false);
            optimizer.addVertex(VV);

            if(is_opt_each_bias)
            {
                Eigen::Vector3d bg = lidar->getImuGyroBias();
                VertexGyroBias* VG = new VertexGyroBias(bg);
                VG->setId(max_id + 3*lidar->id_ + 2);
                VG->setFixed(false);
                optimizer.addVertex(VG);

                Eigen::Vector3d ba = lidar->getImuAccBias();
                VertexAccBias* VA = new VertexAccBias(ba);
                VA->setId(max_id + 3*lidar->id_ + 3);
                VA->setFixed(false);
                optimizer.addVertex(VA);
            }

        }
    }

    if(!is_opt_each_bias)
    {
        LidarPtr lidar = key_lidars_vec.back();
        Eigen::Vector3d bg = lidar->getImuGyroBias();
        VertexGyroBias* VG = new VertexGyroBias(bg);
        VG->setId(4*max_id + 2);
        VG->setFixed(false);
        optimizer.addVertex(VG);

        Eigen::Vector3d ba = lidar->getImuAccBias();
        VertexAccBias* VA = new VertexAccBias(ba);
        VA->setId(4*max_id + 3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    int scan_id = 0;
    for(int i=0; i<N; i++)
    {
        std::vector<Eigen::Vector3d> pts_vec = pts_vec_vec[i];
        std::vector<Eigen::Vector3d> nms_vec = nms_vec_vec[i];
        std::vector<double> ds_vec = ds_vec_vec[i];
        LidarPtr lidar = key_lidars_vec[i];

        if(0 == i)
            continue;

        if(lidar->imu_from_last_keylidar_)
        {
            if(lidar->id_>max_id)
                continue;

            if(!lidar->last_key_lidar_)
                continue;

            if(lidar->last_key_lidar_->id_ > max_id)
                continue;

            Eigen::Vector3d new_bg = lidar->last_key_lidar_->getImuGyroBias();
            Eigen::Vector3d new_ba = lidar->last_key_lidar_->getImuAccBias();
            lidar->imu_from_last_keylidar_->setNewBias(new_ba, new_bg);

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(lidar->last_key_lidar_->id_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(max_id + 3*lidar->last_key_lidar_->id_+1);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(lidar->id_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(max_id + 3*lidar->id_+1);

            g2o::HyperGraph::Vertex* VG1(NULL);
            g2o::HyperGraph::Vertex* VA1(NULL);
            g2o::HyperGraph::Vertex* VG2(NULL);
            g2o::HyperGraph::Vertex* VA2(NULL);

            if(is_opt_each_bias)
            {
                VG1 = optimizer.vertex(max_id + 3*lidar->last_key_lidar_->id_+2);
                VA1 = optimizer.vertex(max_id + 3*lidar->last_key_lidar_->id_+3);
                VG2 = optimizer.vertex(max_id + 3*lidar->id_+2);
                VA2 = optimizer.vertex(max_id + 3*lidar->id_+3);

                if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                {
                    continue;
                }

                EdgeGyroRW* egr= new EdgeGyroRW();
                egr->setVertex(0,VG1);
                egr->setVertex(1,VG2);

                Eigen::Matrix3d InfoG = lidar->imu_from_last_keylidar_->cov_mat_.block<3,3>(9,9).inverse();
                egr->setInformation(InfoG);
                egr->computeError();
                optimizer.addEdge(egr);

                EdgeAccRW* ear = new EdgeAccRW();
                ear->setVertex(0,VA1);
                ear->setVertex(1,VA2);
                Eigen::Matrix3d InfoA = lidar->imu_from_last_keylidar_->cov_mat_.block<3,3>(12,12).inverse();
                ear->setInformation(InfoA);
                ear->computeError();
                optimizer.addEdge(ear);
            }
            else
            {
                VG1 = optimizer.vertex(4*max_id + 2);
                VA1 = optimizer.vertex(4*max_id + 3);

                if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                {
                    continue;
                }
            }
            
            Sophus::SE3 Tlw = lidar->last_key_lidar_->getLidarPose();
            Sophus::SE3 Tcw = lidar->getLidarPose();
            Sophus::SE3 Tlc = Tlw * Tcw.inverse();
            double delta_d = Tlc.translation().norm();

            EdgeInertial* ei = new EdgeInertial(lidar->imu_from_last_keylidar_, true);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            
            if(delta_d<0.05)
                ei->setInformation(1e-2 * ei->information());

            optimizer.addEdge(ei);
            ei->computeError();
        }

        double pp_error = 0;
        for(size_t i=0,iend=ds_vec.size();i<iend;i++)
        {
            EdgeLidarPointPlane* e = new EdgeLidarPointPlane(nms_vec[i] , ds_vec[i]);
            g2o::HyperGraph::Vertex* VP = optimizer.vertex(lidar->id_);
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
            Eigen::Vector3d pt_b = Tbl * pts_vec[i];
            e->setPoint(pt_b);
            e->setInformation(Eigen::Matrix<double,1,1>::Identity());

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(0.5);

            e->computeError();
            pp_error += e->chi2();
            optimizer.addEdge(e);
        }

    }

    if(!is_opt_each_bias)
    {
        g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*max_id+2);
        g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*max_id+3);

        EdgePriorAcc* epa = new EdgePriorAcc(Eigen::Vector3d(0,0,0));
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = 1e6;
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro* epg = new EdgePriorGyro(Eigen::Vector3d(0,0,0));
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = 1e3;
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);

        epa->computeError();
        epg->computeError();
    }

    double init_error, final_error;
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    init_error = optimizer.activeChi2();
    optimizer.setVerbose(false);
    optimizer.optimize(10);
    final_error = optimizer.activeChi2();
    
    for(int i=0; i<N; i++)
    {
        if(0==i)
            continue;

        LidarPtr lidar = key_lidars_vec[i];
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex( lidar->id_ ) );
        Sophus::SE3 Twb = SE3( VP->estimate().Rwb, VP->estimate().twb);
        Sophus::SE3 Tlw = (Twb * Tbl).inverse();
        lidar->setLidarPose(Tlw);

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(  max_id + 3*lidar->id_ + 1 ) );
        lidar->setVelocity( VV->estimate() );

        VertexGyroBias* VG;
        VertexAccBias* VA;
        if (is_opt_each_bias)
        {
            VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_id + 3*lidar->id_ + 2 ) );
            VA = static_cast<VertexAccBias*>(optimizer.vertex(max_id + 3*lidar->id_ + 3 ) );
        }
        else
        {
            VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*max_id+2));
            VA = static_cast<VertexAccBias*>(optimizer.vertex(4*max_id+3));
        }

        lidar->setImuAccBias( VA->estimate() );
        lidar->setImuGyroBias( VG->estimate() );
        
        auto mit = map->lid_poses_map_.find(lidar->id_);
        if(mit == map->lid_poses_map_.end())
        {
            map->lid_poses_map_.insert(make_pair(lidar->id_, Twb)); 
        }
        else
        {
            mit->second = Twb;
        }

    }
}

void lidarVisualImuLocalBundleAdjustment(
                list<LidarPtr>& key_lidars_list,
                std::vector <std::vector<Eigen::Vector3d>>& pts_vec_vec,
                std::vector <std::vector<Eigen::Vector3d>>& nms_vec_vec,
                std::vector <std::vector<double>>& ds_vec_vec,
                Frame* center_kf,
                set<Frame*>* core_kfs,
                Map* map)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    LidarPtr center_lidar = key_lidars_list.back();
    assert(center_kf->getTimeStamp()==center_lidar->getTimeStamp());

    const int n_opt_kfs = 7;
    int n_features = 0;
    int n_features_threshold = 1500;
    Frame* fixed_kf = NULL;
    bool is_in_act_vec = true;
    int fix_id = -1;
    std::vector<Frame*> act_kf_vec;

    act_kf_vec.push_back(center_kf);
    act_kf_vec.back()->lba_id_ = center_kf->id_;
    while(act_kf_vec.back()->last_kf_)
    {
        act_kf_vec.push_back(act_kf_vec.back()->last_kf_);
        act_kf_vec.back()->lba_id_ = center_kf->id_;

        if(act_kf_vec.size() >= n_opt_kfs)
        {
            fix_id = act_kf_vec.back()->id_;
            break;
        }
    }

    std::vector<Frame*> covisual_kf_vec;
    for(auto it=core_kfs->begin();it!=core_kfs->end();++it)
    {
        if((*it)->lba_id_ == center_kf->id_)
            continue;

        if((*it)->id_ < fix_id)
        {
            fix_id = (*it)->id_;
            is_in_act_vec = false;
            fixed_kf = *it;
        }
    }

    if(is_in_act_vec)
    {
        if(act_kf_vec.back()->last_kf_)
        {
            fixed_kf = act_kf_vec.back()->last_kf_;
        }
        else
        {
            fixed_kf = act_kf_vec.back();
            act_kf_vec.pop_back();
        }
    }
    else
    {
        for(auto it=core_kfs->begin();it!=core_kfs->end();++it)
        {
            if((*it)->lba_id_ == center_kf->id_)
                continue;

            if((*it)->id_ == fix_id)
                continue;

            (*it)->lba_id_ = center_kf->id_;
            covisual_kf_vec.push_back(*it);
        }

    }
    fixed_kf->lba_id_ = center_kf->id_;

    const int N_image = act_kf_vec.size() + covisual_kf_vec.size();
    const int max_vid = center_kf->id_;

    Sophus::SE3 Tbc = center_kf->T_b_c_;
    Sophus::SE3 Tbl = key_lidars_list.back()->T_b_l_;

    std::set<Point*> mps;
    std::map<double, long> visual_ts_id_map;
    auto iter = act_kf_vec.begin();
    while(iter != act_kf_vec.end())
    {
        Frame* kf = *iter;
        if(kf->id_ > max_vid)
        {
            ++iter;
            continue;
        }

        bool is_fixed = false;
        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        optimizer.addVertex(VP);
        VP->setFixed(is_fixed);
        double ts = kf->getTimeStamp();
        visual_ts_id_map.insert(make_pair(ts, kf->id_));

        if(kf->imu_from_last_keyframe_)
        {
            VertexVelocity* VV = new VertexVelocity(kf);
            VV->setId( max_vid + 3*kf->id_ + 1 );
            VV->setFixed(is_fixed);
            optimizer.addVertex(VV);

            VertexGyroBias* VG = new VertexGyroBias(kf);
            VG->setId(max_vid + 3*kf->id_ + 2);
            VG->setFixed(is_fixed);
            optimizer.addVertex(VG);

            VertexAccBias* VA = new VertexAccBias(kf);
            VA->setId(max_vid + 3*kf->id_ + 3);
            VA->setFixed(is_fixed);
            optimizer.addVertex(VA);
        }

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;

            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
            
            mps.insert((*it_pt)->point);
        }

        ++iter;
    }

    iter = covisual_kf_vec.begin();
    while(iter != covisual_kf_vec.end())
    {
        Frame* kf = *iter;
        if(kf->id_ > max_vid)
        {
            ++iter;
            continue;
        }
        
        VertexPose * VP = new VertexPose(kf);
        VP->setId(kf->id_);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        double ts = kf->getTimeStamp();
        visual_ts_id_map.insert(make_pair(ts, kf->id_));

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;

            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
            mps.insert((*it_pt)->point);
        }

        ++iter;
    }

    {
        VertexPose * VP = new VertexPose(fixed_kf);
        VP->setId(fixed_kf->id_);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        double ts = fixed_kf->getTimeStamp();
        visual_ts_id_map.insert(make_pair(ts, fixed_kf->id_));

        if(fixed_kf->imu_from_last_keyframe_)
        {
            VertexVelocity* VV = new VertexVelocity(fixed_kf);
            VV->setId( max_vid + 3*fixed_kf->id_ + 1 );
            VV->setFixed(true);
            optimizer.addVertex(VV);

            VertexGyroBias* VG = new VertexGyroBias(fixed_kf);
            VG->setId(max_vid + 3*fixed_kf->id_ + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);

            VertexAccBias* VA = new VertexAccBias(fixed_kf);
            VA->setId(max_vid + 3*fixed_kf->id_ + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }

        for(Features::iterator it_pt=fixed_kf->fts_.begin(); it_pt!=fixed_kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;
                
            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
            mps.insert((*it_pt)->point);
        }
    }

    float huber_corner = 0, huber_edge = 0;
    double focal_length = center_kf->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = center_kf->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];
    computeVisualWeight(mps, focal_length, huber_corner, huber_edge);

    vector<EdgeInertial*> vi_vec(N_image,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> eg_vec(N_image,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> ea_vec(N_image,(EdgeAccRW*)NULL);

    for(size_t i=0, iend=act_kf_vec.size()-1;i<iend;i++)
    {
        Frame* kf = act_kf_vec[i];
        if(kf->id_ > max_vid)
            continue;

        if(kf->last_kf_ && kf->last_kf_->id_ < max_vid)
        {
            if(!kf->imu_from_last_keyframe_ || !kf->last_kf_->imu_from_last_keyframe_)
            {
                continue;
            }

            kf->setImuAccBias(kf->last_kf_->getImuAccBias());
            kf->setImuGyroBias(kf->last_kf_->getImuGyroBias());

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(kf->last_kf_->id_);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(max_vid + 3*kf->last_kf_->id_ + 1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(max_vid + 3*kf->last_kf_->id_ + 2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(max_vid + 3*kf->last_kf_->id_ + 3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(kf->id_);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(max_vid + 3*kf->id_ + 1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(max_vid + 3*kf->id_ + 2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(max_vid + 3*kf->id_ + 3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            EdgeInertial* vi= NULL;
            vi = new EdgeInertial(kf->imu_from_last_keyframe_, false);

            vi->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vi->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vi->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vi->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vi->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vi->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            vi->setInformation(0.0001*vi->information());
            optimizer.addEdge(vi);
            vi_vec.push_back(vi);

            EdgeGyroRW* eg = new EdgeGyroRW();
            eg->setVertex(0,VG1);
            eg->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(9,9).inverse();
            eg->setInformation(0.0001*InfoG);
            optimizer.addEdge(eg);
            eg_vec.push_back(eg);

            EdgeAccRW* ea = new EdgeAccRW();
            ea->setVertex(0,VA1);
            ea->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(12,12).inverse();
            ea->setInformation(0.0001*InfoA);
            optimizer.addEdge(ea);
            ea_vec.push_back(ea);

            vi->computeError();
            double ee_vi = vi->chi2();

            eg->computeError();
            double ee_eg = eg->chi2();

            ea->computeError();
            double ee_ea = ea->chi2();
        }

    }

    list<EdgeFrameFeature> edges;
    list<EdgeLetFrameFeature> edgeLets;
    int n_edges = 0;
    long v_id = 4 * (max_vid+1);
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexPointID* vPoint = new VertexPointID();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        double idist = (*it_pt)->getPointIdist();
        vPoint->setEstimate(idist);
        assert(optimizer.addVertex(vPoint));
        (*it_pt)->lba_id_ = center_kf->id_;

        if((*it_pt)->hostFeature_->frame->lba_id_ != center_kf->id_)
        {
            (*it_pt)->hostFeature_->frame->lba_id_ = center_kf->id_;
            VertexPose* vHost = new VertexPose((*it_pt)->hostFeature_->frame);
            vHost->setId((*it_pt)->hostFeature_->frame->id_);
            vHost->setFixed(true);
            assert(optimizer.addVertex(vHost));
            double ts = (*it_pt)->hostFeature_->frame->getTimeStamp();
            visual_ts_id_map.insert(make_pair(ts, (*it_pt)->hostFeature_->frame->id_));
        }

        list<Feature*> obs = (*it_pt)->getObs();
        list<Feature*>::iterator it_obs = obs.begin();
        while(it_obs != obs.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            if((*it_obs)->frame->lba_id_ != center_kf->id_)
            {
                (*it_obs)->frame->lba_id_ = center_kf->id_;
                VertexPose* vTarget = new VertexPose((*it_obs)->frame);
                vTarget->setId((*it_obs)->frame->id_);
                vTarget->setFixed(true);
                assert(optimizer.addVertex(vTarget));

                double ts = (*it_obs)->frame->getTimeStamp();
                visual_ts_id_map.insert(make_pair(ts, (*it_obs)->frame->id_));
            }

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIdistCorner* edge = new EdgeIdistCorner();
                edge->resize(3);
                
                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edge->setHostBearing((*it_pt)->hostFeature_->f);
                edge->setCameraParams(fxy);
                edge->setMeasurement(Fxy* vilo::project2d((*it_obs)->f));
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(0.001*Eigen::Matrix2d::Identity() );

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_corner);
                edge->setRobustKernel(rk);
                edge->setParameterId(0, 0);

                edges.push_back(EdgeFrameFeature(edge, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edge));
            }
            else
            {
                EdgeIdistEdgeLet* edgeLet = new EdgeIdistEdgeLet();
                edgeLet->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(v_id-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex((*it_pt)->hostFeature_->frame->id_);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex((*it_obs)->frame->id_);

                if(!VP || !VH || !VT)
                {
                    cerr << "Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f);
                edgeLet->setCameraParams(fxy);
                edgeLet->setTargetNormal((*it_obs)->grad);
                edgeLet->setMeasurement((*it_obs)->grad.transpose()* Fxy* vilo::project2d((*it_obs)->f));
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edgeLet->setInformation(0.001*Eigen::Matrix<double,1,1>::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                edgeLet->setRobustKernel(rk);
                edgeLet->setParameterId(0, 0);

                edgeLets.push_back(EdgeLetFrameFeature(edgeLet, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edgeLet));
            }

            ++n_edges;
            ++it_obs;
        }

    }

    std::vector<LidarPtr> key_lidars_vec;
    auto lit = key_lidars_list.begin();
    while(lit!=key_lidars_list.end())
    {
        key_lidars_vec.push_back(*lit);
        ++lit;
    }

    int n_its = 200; 
    long max_lid = key_lidars_vec.back()->id_;
    const int N_lidar = key_lidars_vec.size();

    for(int i=0; i<N_lidar; i++)
    {
        LidarPtr lidar = key_lidars_vec[i];
        if(lidar->id_ > max_lid)
            continue;

        if(lidar->id_ != max_lid)
        {
            Sophus::SE3 Twb = lidar->getImuPose();
            VertexPose * VP = new VertexPose(Twb, Tbc, Tbl, center_kf);
            VP->setId(v_id + lidar->id_);
            VP->setFixed(false);
            optimizer.addVertex(VP);

            if(lidar->imu_from_last_keylidar_)
            {
                Eigen::Vector3d vel = lidar->getVelocity();
                VertexVelocity* VV = new VertexVelocity(vel);
                VV->setId( v_id + 3*lidar->id_ + 1 );
                VV->setFixed(false);
                optimizer.addVertex(VV);

                Eigen::Vector3d bg = lidar->getImuGyroBias();
                VertexGyroBias* VG = new VertexGyroBias(bg);
                VG->setId(v_id + 3*lidar->id_ + 2);
                VG->setFixed(false);
                optimizer.addVertex(VG);

                Eigen::Vector3d ba = lidar->getImuAccBias();
                VertexAccBias* VA = new VertexAccBias(ba);
                VA->setId(v_id + 3*lidar->id_ + 3);
                VA->setFixed(false);
                optimizer.addVertex(VA);
            }

        }
        else
        {
            Sophus::SE3 Twb = lidar->getImuPose();
            long visual_id = center_kf->id_;
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(visual_id));
            VP->setEstimate( ImuCamPose(Twb, Tbc, Tbl, center_kf) );

            if(lidar->imu_from_last_keylidar_)
            {
                Eigen::Vector3d vel = lidar->getVelocity();
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_vid + 3*visual_id + 1));
                VV->setEstimate(vel);

                Eigen::Vector3d bg = lidar->getImuGyroBias();
                VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_vid + 3*visual_id + 2));
                VG->setEstimate(bg);

                Eigen::Vector3d ba = lidar->getImuAccBias();
                VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(max_vid + 3*visual_id + 3));
                VA->setEstimate(ba);
            }
        }
    }

    int scan_id = 0;
    for(int i=0; i<N_lidar; i++)
    {
        std::vector<Eigen::Vector3d> pts_vec = pts_vec_vec[i];
        std::vector<Eigen::Vector3d> nms_vec = nms_vec_vec[i];
        std::vector<double> ds_vec = ds_vec_vec[i];
        LidarPtr lidar = key_lidars_vec[i];

        if(0 == i)
            continue;

        long p1_id = -1, p2_id = -1, v1_id = -1, v2_id = -1;
        long a1_id = -1, a2_id = -1, g1_id = -1, g2_id = -1;

        if(lidar->imu_from_last_keylidar_)
        {
            if(lidar->id_>max_lid)
                continue;

            if(!lidar->last_key_lidar_)
                continue;

            if(lidar->last_key_lidar_->id_ > max_lid)
                continue;

            Eigen::Vector3d new_bg = lidar->last_key_lidar_->getImuGyroBias();
            Eigen::Vector3d new_ba = lidar->last_key_lidar_->getImuAccBias();
            lidar->imu_from_last_keylidar_->setNewBias(new_ba, new_bg);
            
            p1_id = v_id + lidar->last_key_lidar_->id_;
            v1_id = v_id + 3*lidar->last_key_lidar_->id_+1;
            g1_id = v_id + 3*lidar->last_key_lidar_->id_+2;
            a1_id = v_id + 3*lidar->last_key_lidar_->id_+3;

            if(lidar->id_ != max_lid)
            {
                p2_id = v_id + lidar->id_;
                v2_id = v_id + 3*lidar->id_+1;
                g2_id = v_id + 3*lidar->id_+2;
                a2_id = v_id + 3*lidar->id_+3;
            }
            else
            {
                p2_id = center_kf->id_;
                v2_id = max_vid + 3*center_kf->id_+1;
                g2_id = max_vid + 3*center_kf->id_+2;
                a2_id = max_vid + 3*center_kf->id_+3;
            }
            
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(p1_id);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(v1_id);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(p2_id);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(v2_id);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(g1_id);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(a1_id);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(g2_id);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(a2_id);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                continue;
            }

            EdgeGyroRW* egr= new EdgeGyroRW();
            egr->setVertex(0,VG1);
            egr->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = lidar->imu_from_last_keylidar_->cov_mat_.block<3,3>(9,9).inverse();
            egr->setInformation(InfoG);
            egr->computeError();
            optimizer.addEdge(egr);

            EdgeAccRW* ear = new EdgeAccRW();
            ear->setVertex(0,VA1);
            ear->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = lidar->imu_from_last_keylidar_->cov_mat_.block<3,3>(12,12).inverse();
            ear->setInformation(InfoA);
            ear->computeError();
            optimizer.addEdge(ear);

            Sophus::SE3 Tlw = lidar->last_key_lidar_->getLidarPose();
            Sophus::SE3 Tcw = lidar->getLidarPose();
            Sophus::SE3 Tlc = Tlw * Tcw.inverse();
            double delta_d = Tlc.translation().norm();

            EdgeInertial* ei = new EdgeInertial(lidar->imu_from_last_keylidar_, false);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            if(delta_d<0.05)
                ei->setInformation(1e-2 * ei->information());

            optimizer.addEdge(ei);

            ei->computeError();
        }

        double pp_error = 0;
        for(size_t i=0,iend=ds_vec.size();i<iend;i++)
        {
            EdgeLidarPointPlane* e = new EdgeLidarPointPlane(nms_vec[i] , ds_vec[i]);
            g2o::HyperGraph::Vertex* VP = optimizer.vertex(p2_id);
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
            Eigen::Vector3d pt_b = Tbl * pts_vec[i];
            e->setPoint(pt_b);
            e->setInformation(10*Eigen::Matrix<double,1,1>::Identity());

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(0.5);

            e->computeError();
            pp_error += e->chi2();
            optimizer.addEdge(e);
        }
    }

    double init_error, final_error;
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    init_error = optimizer.activeChi2();
    optimizer.setVerbose(false);
    optimizer.optimize(5);
    final_error = optimizer.activeChi2();

    boost::unique_lock<boost::mutex> lock(map->map_mutex_);
    iter = act_kf_vec.begin();
    while(iter!=act_kf_vec.end())
    {
        Frame* frame = *iter;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex( frame->id_ ) );
        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_vid + 3*frame->id_ + 1) );
        VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_vid + 3*frame->id_ + 2) );
        VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(max_vid + 3*frame->id_ + 3) );

        Sophus::SE3 Tfw( VP->estimate().Rcw[0], VP->estimate().tcw[0]);
        frame->setFramePose(Tfw);

        frame->setVelocity( VV->estimate() );
        frame->setImuAccBias( VA->estimate() );
        frame->setImuGyroBias( VG->estimate() );

        map->point_candidates_.changeCandidatePosition(*iter);
        ++iter;
    }

    std::set<Frame*>::iterator sit = core_kfs->begin();
    while(sit!=core_kfs->end())
    {
        VertexPose* VPose = static_cast<VertexPose*>(optimizer.vertex( (*sit)->id_ ) );
        Sophus::SE3 Tfw( VPose->estimate().Rcw[0], VPose->estimate().tcw[0]);
        (*sit)->setFramePose(Tfw);
        map->point_candidates_.changeCandidatePosition(*sit);
        ++sit;
    }

    v_id = 4 * (max_vid+1);
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it,++v_id)
    {
        VertexPointID* VPoint = static_cast<VertexPointID*>(optimizer.vertex(v_id) );

        double idist = VPoint->estimate();
        (*it)->setPointIdistAndPose(idist);
    }

    int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2 = 2.0; 
    const double reproj_thresh_1 = 1.2;

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeFrameFeature>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if( it->feature->point == NULL) continue;

        if(it->edge->chi2() > reproj_thresh_2_squared)
        {
            if((it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeLetFrameFeature>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if(it->feature->point == NULL) continue;


        if(it->edge->chi2() > reproj_thresh_1_squared)
        {
            if((it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }

    for(int i=0; i<N_lidar; i++)
    {
        if(0==i)
            continue;

        LidarPtr lidar = key_lidars_vec[i];

        Sophus::SE3 Twb;
        if(lidar->id_ != max_lid)
        {
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(v_id + lidar->id_) );
            Twb = SE3( VP->estimate().Rwb, VP->estimate().twb);
            Sophus::SE3 Tlw = (Twb * Tbl).inverse();
            lidar->setLidarPose(Tlw);

            if(lidar->imu_from_last_keylidar_)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(v_id + 3*lidar->id_+1) );
                lidar->setVelocity( VV->estimate() );

                VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(v_id + 3*lidar->id_+2) );
                VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(v_id + 3*lidar->id_+3) );

                lidar->setImuAccBias( VA->estimate() );
                lidar->setImuGyroBias( VG->estimate() );
            }
        }
        else
        {
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(center_kf->id_) );
            Twb = SE3( VP->estimate().Rwb, VP->estimate().twb);
            Sophus::SE3 Tlw = (Twb * Tbl).inverse();
            lidar->setLidarPose(Tlw);

            if(lidar->imu_from_last_keylidar_)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_vid + 3*center_kf->id_+1) );
                lidar->setVelocity( VV->estimate() );
                VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_vid + 3*center_kf->id_+2) );
                VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(max_vid + 3*center_kf->id_+3) );

                lidar->setImuAccBias( VA->estimate() );
                lidar->setImuGyroBias( VG->estimate() );
            }
        }
         
        auto mit = map->lid_poses_map_.find(lidar->id_);
        if(mit == map->lid_poses_map_.end())
        {
            map->lid_poses_map_.insert(make_pair(lidar->id_, Twb)); 
        }
        else
        {
            mit->second = Twb;
        }
    }
}

void lidarVisualImuLocalBundleAdjustment(
                list<LidarPtr>& key_lidars_list,
                std::vector <std::vector<Eigen::Vector3d>>& pts_vec_vec,
                std::vector <std::vector<Eigen::Vector3d>>& nms_vec_vec,
                std::vector <std::vector<double>>& ds_vec_vec,
                Frame* center_kf,
                Map* map)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    LidarPtr center_lidar = key_lidars_list.back();
    assert(center_kf->getTimeStamp()==center_lidar->getTimeStamp());
    
    std::set<long> lidar_id_set;
    std::vector<LidarPtr> key_lidars_vec;
    auto lit = key_lidars_list.begin();
    while(lit!=key_lidars_list.end())
    {
        key_lidars_vec.push_back(*lit);
        lidar_id_set.insert((*lit)->id_);
        ++lit;
    }
    long v_start_id = key_lidars_vec[0]->align_vid_;
    long v_end_id = key_lidars_vec.back()->align_vid_;

    std::vector<Frame*> key_frames_vec;
    auto fit = map->keyframes_.rbegin();
    while(fit!=map->keyframes_.rend() )
    {
        Frame* kf = (*fit).get();
        if(kf->id_<=v_end_id && kf->id_>=v_start_id)
            key_frames_vec.push_back(kf);
        else
            break;

        ++fit;
    }
    int N_lidar = key_lidars_vec.size();
    int N_frame = key_frames_vec.size();
    std::reverse(key_frames_vec.begin(), key_frames_vec.end());
    long max_lid = key_lidars_vec.back()->id_;
    long max_vid = key_frames_vec.back()->id_;
    Sophus::SE3 Tbc = center_kf->T_b_c_;
    Sophus::SE3 Tbl = key_lidars_vec.back()->T_b_l_;

    for(int i=0; i<N_lidar; i++)
    {
        LidarPtr lidar = key_lidars_vec[i];
        if(lidar->id_ > max_lid)
            continue;

        Sophus::SE3 Twb = lidar->getImuPose();
        VertexPose * VP = new VertexPose(Twb, Tbc, Tbl, center_kf);
        VP->setId(lidar->id_);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(lidar->imu_from_last_keylidar_)
        {
            Eigen::Vector3d vel = lidar->getVelocity();
            VertexVelocity* VV = new VertexVelocity(vel);
            VV->setId( max_lid + 3*lidar->id_ + 1 );
            VV->setFixed(false);
            optimizer.addVertex(VV);

            Eigen::Vector3d bg = lidar->getImuGyroBias();
            VertexGyroBias* VG = new VertexGyroBias(bg);
            VG->setId(max_lid + 3*lidar->id_ + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);

            Eigen::Vector3d ba = lidar->getImuAccBias();
            VertexAccBias* VA = new VertexAccBias(ba);
            VA->setId(max_lid + 3*lidar->id_ + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);

        }

    }

    bool is_visual_good = false;
    int n_total_obs=0;
    long vid = 4*(max_lid+1);
    std::set<Point*> mps;
    std::map<long, Eigen::Vector2d> fid_vid_map;
    for(int i=0; i<N_frame;i++)
    {
        Frame* kf = key_frames_vec[i];
        if(kf->id_ > max_vid)
            continue;
        kf->lba_id_ = center_kf->id_;

        bool is_new_vtx = false;
        if(kf->align_lid_ == -1)
            is_new_vtx = true;
        else
        {
            long align_lid = kf->align_lid_;
            if(lidar_id_set.find(align_lid)==lidar_id_set.end())
                is_new_vtx = true;
            else
                is_new_vtx = false;
        }

        if(is_new_vtx)
        {
            VertexPose * VP = new VertexPose(kf);
            VP->setId(vid + kf->id_);
            optimizer.addVertex(VP);
            VP->setFixed(false);
            
            if(kf->imu_from_last_keyframe_)
            {
                VertexVelocity* VV = new VertexVelocity(kf);
                VV->setId( vid + 3*kf->id_ + 1 );
                VV->setFixed(false);
                optimizer.addVertex(VV);

                VertexGyroBias* VG = new VertexGyroBias(kf);
                VG->setId(vid + 3*kf->id_ + 2);
                VG->setFixed(false);
                optimizer.addVertex(VG);

                VertexAccBias* VA = new VertexAccBias(kf);
                VA->setId(vid + 3*kf->id_ + 3);
                VA->setFixed(false);
                optimizer.addVertex(VA);
            }

            Eigen::Vector2d vids;
            vids << 0, vid + kf->id_;
            fid_vid_map.insert(make_pair(kf->id_, vids));
        }
        else
        {
            Eigen::Vector2d vids;
            vids << 1, kf->align_lid_;
            fid_vid_map.insert(make_pair(kf->id_, vids));
        }

        for(Features::iterator it_pt=kf->fts_.begin(); it_pt!=kf->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL)
                continue;

            if((*it_pt)->point->getPointState() < Point::TYPE_UNKNOWN)
                continue;
                
            mps.insert((*it_pt)->point);
            n_total_obs += (*it_pt)->point->getObsSize();
        }
    }

    double focal_length = center_kf->cam_->errorMultiplier2();
    Eigen::Vector2d fxy = center_kf->cam_->focal_length();
    Eigen::Matrix2d Fxy; Fxy<<fxy[0], 0.0, 0.0, fxy[1];
    
    int v_err_num=0; float avr_err_pt = 0.0, avr_err_ls = 0.0;
    computeVisualAvrError(mps, v_err_num, avr_err_pt, avr_err_ls);

    int l_err_num=0; float avr_err_lr = 0;
    for(int i=0; i<N_lidar; i++)
    {
        std::vector<Eigen::Vector3d> pts_vec = pts_vec_vec[i];
        std::vector<Eigen::Vector3d> nms_vec = nms_vec_vec[i];
        std::vector<double> ds_vec = ds_vec_vec[i];
        LidarPtr lidar = key_lidars_vec[i];
        Sophus::SE3 Twl_tmp = lidar->getLidarPose().inverse();
        for(size_t j=0,jend=ds_vec.size();j<jend;j++)
        {
            Eigen::Vector3d pt_w = Twl_tmp * pts_vec[j];
            double e = nms_vec[j].dot(pt_w) + ds_vec[j];

            avr_err_lr += e * e;
            l_err_num++;
        }

    }
    avr_err_lr = avr_err_lr / l_err_num;
    int old_l_err_num=0, old_v_err_num=0;
    double old_l_err=0, old_v_err=0;
    for(auto qit=map->avr_err_lr_deque_.begin();qit<map->avr_err_lr_deque_.end();++qit)
        old_l_err += *qit;

    for(auto qit=map->avr_err_pt_deque_.begin();qit<map->avr_err_pt_deque_.end();++qit)
        old_v_err += *qit;

    for(auto qit=map->l_err_num_deque_.begin();qit<map->l_err_num_deque_.end();++qit)
        old_l_err_num += *qit;
    
    for(auto qit=map->v_err_num_deque_.begin();qit<map->v_err_num_deque_.end();++qit)
        old_v_err_num += *qit; 
    
    old_l_err = 1.0 * old_l_err / map->avr_err_lr_deque_.size();
    old_v_err = 1.0 * old_v_err / map->avr_err_pt_deque_.size();
    old_l_err_num = 1.0 * old_l_err_num / map->l_err_num_deque_.size();
    old_v_err_num = 1.0 * old_v_err_num / map->v_err_num_deque_.size();

    double reliability_v = (1.0 * v_err_num /  old_v_err_num) * (old_v_err / avr_err_pt);
    double reliability_l = (1.0 * l_err_num /  old_l_err_num) * (old_l_err / avr_err_lr);
    
    double l_weight = (4.0 / PI) * (l_err_num / v_err_num) * atan(sqrt(reliability_l));
    double v_weight = (4.0 / PI) * atan( sqrt(reliability_v) );

    while(map->avr_err_pt_deque_.size()>=10)
        map->avr_err_pt_deque_.pop_front();
    map->avr_err_pt_deque_.push_back(avr_err_pt);

    while(map->avr_err_lr_deque_.size()>=10)
        map->avr_err_lr_deque_.pop_front();
    map->avr_err_lr_deque_.push_back(avr_err_lr);

    while(map->v_err_num_deque_.size()>=10)
        map->v_err_num_deque_.pop_front();
    map->v_err_num_deque_.push_back(v_err_num);

    while(map->l_err_num_deque_.size()>=10)
        map->l_err_num_deque_.pop_front();
    map->l_err_num_deque_.push_back(l_err_num);

    double lidar_error=0, visual_error=0, li_error=0, vi_error=0;
    double lba_error=0, lbg_error=0, vba_error=0, vbg_error=0;
    int n_lidar_error=0, n_visual_error=0, n_li_error=0, n_vi_error=0;

    int scan_id = 0;
    bool is_li_good = true;

    if(std::isnan(l_weight))
        l_weight = 1.0;

    if(std::isnan(v_weight))
        v_weight = 1.0;
    
    for(int i=0; i<N_lidar; i++)
    {
        std::vector<Eigen::Vector3d> pts_vec = pts_vec_vec[i];
        std::vector<Eigen::Vector3d> nms_vec = nms_vec_vec[i];
        std::vector<double> ds_vec = ds_vec_vec[i];
        LidarPtr lidar = key_lidars_vec[i];

        if(0 == i)
            continue;

        long p1_id ,v1_id ,g1_id ,a1_id;
        long p2_id ,v2_id ,g2_id ,a2_id;
        p2_id = lidar->id_;
        double cur_li_err = 0;
        if(lidar->imu_from_last_keylidar_)
        {
            if(lidar->id_>max_lid)
                continue;

            if(!lidar->last_key_lidar_)
                continue;

            if(lidar->last_key_lidar_->id_ > max_lid)
                continue;

            Eigen::Vector3d new_bg = lidar->last_key_lidar_->getImuGyroBias();
            Eigen::Vector3d new_ba = lidar->last_key_lidar_->getImuAccBias();
            lidar->imu_from_last_keylidar_->setNewBias(new_ba, new_bg);
            
            p1_id = lidar->last_key_lidar_->id_;
            v1_id = max_lid + 3*lidar->last_key_lidar_->id_+1;
            g1_id = max_lid + 3*lidar->last_key_lidar_->id_+2;
            a1_id = max_lid + 3*lidar->last_key_lidar_->id_+3;

            v2_id = max_lid + 3*lidar->id_+1;
            g2_id = max_lid + 3*lidar->id_+2;
            a2_id = max_lid + 3*lidar->id_+3;

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(p1_id);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(v1_id);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(p2_id);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(v2_id);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(g1_id);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(a1_id);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(g2_id);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(a2_id);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                continue;
            }

            EdgeGyroRW* egr= new EdgeGyroRW();
            egr->setVertex(0,VG1);
            egr->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = lidar->imu_from_last_keylidar_->cov_mat_.block<3,3>(9,9).inverse();
            egr->setInformation(InfoG);
            egr->computeError();
            optimizer.addEdge(egr);

            EdgeAccRW* ear = new EdgeAccRW();
            ear->setVertex(0,VA1);
            ear->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = lidar->imu_from_last_keylidar_->cov_mat_.block<3,3>(12,12).inverse();
            ear->setInformation(InfoA);
            ear->computeError();
            optimizer.addEdge(ear);

            Sophus::SE3 Tlw = lidar->last_key_lidar_->getLidarPose();
            Sophus::SE3 Tcw = lidar->getLidarPose();
            Sophus::SE3 Tlc = Tlw * Tcw.inverse();
            double delta_d = Tlc.translation().norm();

            EdgeInertial* ei = new EdgeInertial(lidar->imu_from_last_keylidar_, false);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            if(delta_d<0.05)
                ei->setInformation(1e-2 * ei->information());

            optimizer.addEdge(ei);
            ei->computeError();
            cur_li_err = ei->chi2();
            li_error += cur_li_err;
            lba_error += ear->chi2();
            lbg_error += egr->chi2();
            n_li_error++;

        }

        for(size_t j=0,jend=ds_vec.size();j<jend;j++)
        {
            EdgeLidarPointPlane* e = new EdgeLidarPointPlane(nms_vec[j] , ds_vec[j]);
            g2o::HyperGraph::Vertex* VP = optimizer.vertex(p2_id);
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
            Eigen::Vector3d pt_b = Tbl * pts_vec[j];
            e->setPoint(pt_b);
            e->setInformation(l_weight * Eigen::Matrix<double,1,1>::Identity());

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(0.5);

            e->computeError();
            lidar_error += e->chi2();
            optimizer.addEdge(e);
            n_lidar_error++;
        }
    }

    double avg_li_error = li_error / n_li_error;
    double avg_lba_error = lba_error / n_li_error;
    double avg_lbg_error = lbg_error / n_li_error;
    double avg_lidar_error = lidar_error / n_lidar_error;

    double vi_weight = 0.001, eg_weight = 0.001, ea_weight = 0.001;
    vector<EdgeInertial*> vi_vec(N_frame,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> eg_vec(N_frame,(EdgeGyroRW*)NULL); 
    vector<EdgeAccRW*> ea_vec(N_frame,(EdgeAccRW*)NULL); 
    for(int i=0;i<N_frame;i++)
    {
        Frame* kf = key_frames_vec[i];
        if(kf->id_ > max_vid)
            continue;

        if(i==0)
            continue;

        if(kf->last_kf_ && kf->last_kf_->id_ < max_vid)
        {
            if(!kf->imu_from_last_keyframe_ || !kf->last_kf_->imu_from_last_keyframe_)
            {
                continue;
            }

            long p1_id=0, v1_id=0, g1_id=0, a1_id=0;
            long p2_id=0, v2_id=0, g2_id=0, a2_id=0;

            auto mit = fid_vid_map.find(kf->id_);
            if(mit != fid_vid_map.end())
            {
                if(mit->second[0]==0)
                {
                    p2_id = vid + kf->id_;
                    v2_id = vid + 3*kf->id_+1;
                    g2_id = vid + 3*kf->id_+2;
                    a2_id = vid + 3*kf->id_+3;
                }
                else if(mit->second[0]==1)
                {
                    p2_id = kf->align_lid_;
                    v2_id = max_lid + 3*kf->align_lid_+1;
                    g2_id = max_lid + 3*kf->align_lid_+2;
                    a2_id = max_lid + 3*kf->align_lid_+3;
                }
            }

            mit = fid_vid_map.find(kf->last_kf_->id_);
            if(mit != fid_vid_map.end())
            {
                if(mit->second[0]==0)
                {
                    p1_id = vid + kf->last_kf_->id_;
                    v1_id = vid + 3*kf->last_kf_->id_+1;
                    g1_id = vid + 3*kf->last_kf_->id_+2;
                    a1_id = vid + 3*kf->last_kf_->id_+3;
                }
                else if(mit->second[0]==1)
                {
                    p1_id = kf->last_kf_->align_lid_;
                    v1_id = max_lid + 3*kf->last_kf_->align_lid_+1;
                    g1_id = max_lid + 3*kf->last_kf_->align_lid_+2;
                    a1_id = max_lid + 3*kf->last_kf_->align_lid_+3;
                }
            }

            kf->setImuAccBias(kf->last_kf_->getImuAccBias());
            kf->setImuGyroBias(kf->last_kf_->getImuGyroBias());

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(p1_id);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(v1_id);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(g1_id);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(a1_id);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(p2_id);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(v2_id);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(g2_id);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(a2_id);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "vi Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            EdgeInertial* vi= NULL;
            vi = new EdgeInertial(kf->imu_from_last_keyframe_, false);

            vi->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vi->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vi->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vi->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vi->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vi->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            optimizer.addEdge(vi);

            EdgeGyroRW* eg = new EdgeGyroRW();
            eg->setVertex(0,VG1);
            eg->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(9,9).inverse();
            g2o::RobustKernelHuber* rk_eg = new g2o::RobustKernelHuber;
            eg->setRobustKernel(rk_eg);
            rk_eg->setDelta(0.1);
            optimizer.addEdge(eg);

            EdgeAccRW* ea = new EdgeAccRW();
            ea->setVertex(0,VA1);
            ea->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = kf->imu_from_last_keyframe_->cov_mat_.block<3,3>(12,12).inverse();
            g2o::RobustKernelHuber* rk_ea = new g2o::RobustKernelHuber;
            ea->setRobustKernel(rk_ea);
            rk_ea->setDelta(0.1);
            optimizer.addEdge(ea);

            vi->computeError();
            double ee_vi = vi->chi2();
            eg->computeError();
            double ee_eg = eg->chi2();
            ea->computeError();
            double ee_ea = ea->chi2();

            vi->setInformation(vi_weight * vi->information());
            eg->setInformation(eg_weight * InfoG);
            ea->setInformation(ea_weight * InfoA);

            vi_vec.push_back(vi);
            eg_vec.push_back(eg);
            ea_vec.push_back(ea);

            vi_error += vi->chi2();
            vbg_error += eg->chi2();
            vba_error += ea->chi2();
            n_vi_error++;
        }

    }

    list<EdgeFrameFeature> edges;
    list<EdgeLetFrameFeature> edgeLets;
    vid = 4 * (max_lid+max_vid+1);
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexPointID* vPoint = new VertexPointID();
        vPoint->setId(vid++);
        vPoint->setFixed(false);
        vPoint->setMarginalized(true);
        double idist = (*it_pt)->getPointIdist();
        vPoint->setEstimate(idist);
        assert(optimizer.addVertex(vPoint));
        (*it_pt)->lba_id_ = center_kf->id_;

        if((*it_pt)->hostFeature_->frame->lba_id_ != center_kf->id_)
        {
            (*it_pt)->hostFeature_->frame->lba_id_ = center_kf->id_;
            VertexPose* vHost = new VertexPose((*it_pt)->hostFeature_->frame);
            long host_id = 4 * (max_lid+1) + (*it_pt)->hostFeature_->frame->id_;
            vHost->setId(host_id);
            vHost->setFixed(true);
            assert(optimizer.addVertex(vHost));
        }

        long h_vid=0;
        long hid = (*it_pt)->hostFeature_->frame->id_;
        if(hid < v_start_id)
        {
            h_vid = 4 * (max_lid+1) + hid; 
        }
        else 
        {
            auto mit = fid_vid_map.find(hid);
            if(mit != fid_vid_map.end())
            {
                if(mit->second[0]==0)
                {
                    h_vid = 4 * (max_lid+1) + hid;
                }
                else if(mit->second[0]==1)
                {
                    h_vid = (*it_pt)->hostFeature_->frame->align_lid_;
                }
            }
        }

        list<Feature*> obs = (*it_pt)->getObs();
        list<Feature*>::iterator it_obs = obs.begin();
        while(it_obs != obs.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            if(v_start_id - (*it_obs)->frame->id_ >= 20)
            {
                ++it_obs;
                continue;
            }

            if((*it_obs)->frame->lba_id_ != center_kf->id_)
            {
                (*it_obs)->frame->lba_id_ = center_kf->id_;
                VertexPose* vTarget = new VertexPose((*it_obs)->frame);
                long target_id = 4 * (max_lid+1) + (*it_obs)->frame->id_;
                vTarget->setId(target_id);
                vTarget->setFixed(true);
                assert(optimizer.addVertex(vTarget));

            }

            long t_vid=0;
            long tid = (*it_obs)->frame->id_;

            if(tid < v_start_id)
            {
                t_vid = 4 * (max_lid+1) + tid; 
            }
            else
            {
                auto mit = fid_vid_map.find(tid);
                if(mit != fid_vid_map.end())
                {
                    if(mit->second[0]==0) 
                    {
                        t_vid = 4 * (max_lid+1) + tid;
                    }
                    else if(mit->second[0]==1)
                    {
                        t_vid = (*it_obs)->frame->align_lid_;
                    }
                }
            }

            Sophus::SE3 Tfw_t = (*it_obs)->frame->getFramePose();
            Sophus::SE3 Tfw_h = (*it_pt)->hostFeature_->frame->getFramePose();
            Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
            Eigen::Vector3d pHost = (*it_pt)->hostFeature_->f * (1.0/idist);
            Eigen::Vector3d pTarget = Tth * pHost;
            Eigen::Vector2d e_pixel = Fxy * (vilo::project2d( (*it_obs)->f ) - vilo::project2d(pTarget));

            if(e_pixel.norm() >= 10)
            {
                ++it_obs;
                continue;
            } 

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeIdistCorner* edge = new EdgeIdistCorner();
                edge->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(vid-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex(h_vid);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex(t_vid);

                if(!VP || !VH || !VT)
                {
                    cerr << "v Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edge->setHostBearing((*it_pt)->hostFeature_->f);
                edge->setCameraParams(fxy);
                edge->setMeasurement(Fxy* vilo::project2d((*it_obs)->f));
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edge->setInformation(v_weight * Eigen::Matrix2d::Identity() );

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(avr_err_pt);
                edge->setRobustKernel(rk);
                edge->setParameterId(0, 0);

                edges.push_back(EdgeFrameFeature(edge, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edge));

                edge->computeError();                
                if(edge->chi2() >= 20)
                    edge->setLevel(0);
                else
                    visual_error += edge->chi2();
            }
            else
            {
                EdgeIdistEdgeLet* edgeLet = new EdgeIdistEdgeLet();
                edgeLet->resize(3);

                g2o::HyperGraph::Vertex* VP = optimizer.vertex(vid-1);
                g2o::HyperGraph::Vertex* VH = optimizer.vertex(h_vid);
                g2o::HyperGraph::Vertex* VT = optimizer.vertex(t_vid);

                if(!VP || !VH || !VT)
                {
                    cerr << "v Error " << VP << ", "<< VH << ", "<< VT<<endl;
                    ++it_obs;
                    continue;
                }

                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VH));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(VT));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f);
                edgeLet->setCameraParams(fxy);
                edgeLet->setTargetNormal((*it_obs)->grad);
                edgeLet->setMeasurement((*it_obs)->grad.transpose()* Fxy* vilo::project2d((*it_obs)->f));
                
                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                edgeLet->setInformation(v_weight * Eigen::Matrix<double,1,1>::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(avr_err_ls);
                edgeLet->setRobustKernel(rk);
                edgeLet->setParameterId(0, 0);

                edgeLets.push_back(EdgeLetFrameFeature(edgeLet, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(edgeLet));

                edgeLet->computeError();
                visual_error += edgeLet->chi2();
                if(edgeLet->chi2() >= 15)
                    edgeLet->setLevel(0);
                else
                    visual_error += edgeLet->chi2();
            }

            ++n_visual_error;
            ++it_obs;
        }

    }
    double avg_vi_error = vi_error / n_vi_error;
    double avg_vba_error = vba_error / n_vi_error;
    double avg_vbg_error = vbg_error / n_vi_error;
    double avg_visual_error = visual_error / n_visual_error;

    double init_error, final_error;
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    init_error = optimizer.activeChi2();
    optimizer.setVerbose(true);
    optimizer.optimize(10);
    final_error = optimizer.activeChi2();

    boost::unique_lock<boost::mutex> lock(map->map_mutex_);
    for(int i=0; i<N_lidar; i++)
    {
        if(0==i)
            continue;

        LidarPtr lidar = key_lidars_vec[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(lidar->id_) );
        Sophus::SE3 Twb = SE3( VP->estimate().Rwb, VP->estimate().twb);
        Sophus::SE3 Tlw = (Twb * Tbl).inverse();
        lidar->setLidarPose(Tlw);

        if(lidar->imu_from_last_keylidar_)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_lid + 3*lidar->id_+1) );
            lidar->setVelocity( VV->estimate() );
           
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_lid + 3*lidar->id_+2) );
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(max_lid + 3*lidar->id_+3) );

            lidar->setImuAccBias( VA->estimate() );
            lidar->setImuGyroBias( VG->estimate() );
        }

        auto mit = map->lid_poses_map_.find(lidar->id_);
        if(mit == map->lid_poses_map_.end())
        {
            map->lid_poses_map_.insert(make_pair(lidar->id_, Twb)); 
        }
        else
        {
            mit->second = Twb;
        }
    }

    vid = 4*(max_lid+1);
    for(int i=0;i<N_frame;i++)
    {
        Frame* kf = key_frames_vec[i];
        if(kf->id_ > max_vid)
            continue;

        auto mit = fid_vid_map.find(kf->id_);
        if(0 == fid_vid_map[kf->id_][0])
        {
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex( vid+kf->id_) );
            Sophus::SE3 Tfw( VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            kf->setFramePose(Tfw);

            map->point_candidates_.changeCandidatePosition(kf);

            if(kf->imu_from_last_keyframe_)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(vid+3*kf->id_+1) );
                VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(vid+3*kf->id_+2) );
                VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(vid+3*kf->id_+3) );

                kf->setVelocity( VV->estimate() );
                kf->setImuAccBias( VA->estimate() );
                kf->setImuGyroBias( VG->estimate() );
            }
        }
        else
        {
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex( kf->align_lid_ ) );
            Sophus::SE3 Tfw( VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            kf->setFramePose(Tfw);
            map->point_candidates_.changeCandidatePosition(kf);

            if(kf->imu_from_last_keyframe_)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(max_lid+3*kf->align_lid_+1) );
                VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(max_lid+3*kf->align_lid_+2) );
                VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(max_lid+3*kf->align_lid_+3) );

                kf->setVelocity( VV->estimate() );
                kf->setImuAccBias( VA->estimate() );
                kf->setImuGyroBias( VG->estimate() );
            }
        }
    }

    vid = 4 * (max_lid+max_vid+1);
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it,++vid)
    {
        VertexPointID* VPoint = static_cast<VertexPointID*>(optimizer.vertex(vid) );

        double idist = VPoint->estimate();
        (*it)->setPointIdistAndPose(idist);
    }

    int n_incorrect_edges_1 = 0, n_incorrect_edges_2 = 0;
    const double reproj_thresh_2 = 2.0; 
    const double reproj_thresh_1 = 1.2; 

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeFrameFeature>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if( it->feature->point == NULL) continue;

        if(it->edge->chi2() > reproj_thresh_2_squared)
        {
            if((it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeLetFrameFeature>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if(it->feature->point == NULL) continue;


        if(it->edge->chi2() > reproj_thresh_1_squared)
        {
            if((it)->feature->point->getPointState() == Point::TYPE_TEMPORARY)
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }
    
}


void computeVisualWeight(set<Point*>& mps, double focal_length, float& huber_corner, float& huber_edge)
{
    vector<float> errors_pt, errors_ls, errors_tt;
    int n_pt=0, n_ls=0, n_tt=0;

    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        Frame* host_frame = (*it_pt)->hostFeature_->frame;
        Sophus::SE3 Twf_h = host_frame->getFramePose().inverse();
        double idist = (*it_pt)->getPointIdist();
        Vector3d pHost = (*it_pt)->hostFeature_->f * (1.0/idist);
        list<Feature*> obs = (*it_pt)->getObs();
        for(auto it_ft = obs.begin(); it_ft != obs.end(); ++it_ft)
        {
            if((*it_ft)->frame->id_ == host_frame->id_)
                continue;
            if((*it_ft)->point != *it_pt)
                continue;

            Sophus::SE3 Tfw_t = (*it_ft)->frame->getFramePose();
            Sophus::SE3 Tth = Tfw_t * Twf_h;
            Vector3d pTarget = Tth * pHost;
            Vector2d e = vilo::project2d((*it_ft)->f) - vilo::project2d(pTarget);
            e *= 1.0 / (1<<(*it_ft)->level);

            if((*it_ft)->type == Feature::EDGELET)
            {
                errors_ls.push_back(fabs((*it_ft)->grad.transpose()*e));
                n_ls++;
            }
            else
            {
                errors_pt.push_back(e.norm());
                n_pt++;
            }

        }
    }

    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.4826*vilo::getMedian(errors_pt);
        huber_edge = 1.4826*vilo::getMedian(errors_ls);
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.0 / focal_length;
        huber_edge = 1.4826*vilo::getMedian(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = 1.4826*vilo::getMedian(errors_pt);
        huber_edge   = 0.5 / focal_length;
    }
    else
    {
        assert(false);
    }

    huber_corner *= focal_length;
    huber_edge *= focal_length;
}

void computeVisualAvrError(set<Point*>& mps, int& err_num, float& avr_err_pt, float& avr_err_ld)
{
    vector<float> errors_pt, errors_ls;
    int n_pt=0, n_ls=0;
    float sum_err_pt=0, sum_err_ls=0;

    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        Frame* host_frame = (*it_pt)->hostFeature_->frame;
        Sophus::SE3 Twf_h = host_frame->getFramePose().inverse();
        double idist = (*it_pt)->getPointIdist();
        Vector3d pHost = (*it_pt)->hostFeature_->f * (1.0/idist);
        list<Feature*> obs = (*it_pt)->getObs();
        for(auto it_ft = obs.begin(); it_ft != obs.end(); ++it_ft)
        {
            if((*it_ft)->frame->id_ == host_frame->id_)
                continue;

            if((*it_ft)->point != *it_pt)
                continue;

            Sophus::SE3 Tfw_t = (*it_ft)->frame->getFramePose();
            Sophus::SE3 Tth = Tfw_t * Twf_h;
            Vector3d pTarget = Tth * pHost;
            Vector2d e = vilo::project2d((*it_ft)->f) - vilo::project2d(pTarget);
            e *= 1.0 / (1<<(*it_ft)->level);

            if((*it_ft)->type == Feature::EDGELET)
            {
                sum_err_ls += fabs((*it_ft)->grad.transpose()*e);
                n_ls++;
            }
            else
            {
                sum_err_pt += e.norm();
                n_pt++;
            }

        }
    }
    err_num = n_pt + n_ls;
    avr_err_pt = sum_err_pt / n_pt;
    avr_err_ld = sum_err_ls / n_ls;
    if(0==n_pt)
        avr_err_pt = 0;
    if(0==n_ls)
        avr_err_ld = 0;
}

void computeLidarAvrError(std::vector <std::vector<Eigen::Vector3d>>& pts_vec_vec,
                          std::vector <std::vector<Eigen::Vector3d>>& nms_vec_vec,
                          std::vector <std::vector<double>>& ds_vec_vec, 
                          int& n_err, float& avr_err)
{
    const int N = pts_vec_vec.size();
    int err_num = 0;
    float err_sum = 0;
    for(int i=0;i<N;i++)
    {
        std::vector<Eigen::Vector3d> pts_vec = pts_vec_vec[i];
        std::vector<Eigen::Vector3d> nms_vec = nms_vec_vec[i];
        std::vector<double> ds_vec = ds_vec_vec[i];

        for(size_t j=0,jend=ds_vec.size();j<jend;j++)
        {
            double e = nms_vec[j].dot(pts_vec[j]) + ds_vec[j];
            err_sum += e*e;
            err_num++;
        }
    }
    
    n_err = err_num;
    avr_err = err_sum / n_err;
    
}

} // namespace ba
} // namespace vilo


