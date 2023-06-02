#include <algorithm>
#include <stdexcept>
#include <vilo/reprojector.h>
#include <vilo/frame.h>
#include <vilo/point.h>
#include <vilo/feature.h>
#include <vilo/map.h>
#include <vilo/config.h>
#include <vilo/depth_filter.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include "vilo/camera.h"

namespace vilo {

Reprojector::Reprojector(vilo::AbstractCamera* cam, Map* map) :
    map_(map),sum_seed_(0), sum_temp_(0), nFeatures_(0)
{
    initializeGrid(cam);
}

Reprojector::Reprojector(Map* map) :
    map_(map), sum_seed_(0), sum_temp_(0), nFeatures_(0)
{

}

Reprojector::~Reprojector()
{
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
    std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell* s){ delete s; });
}

inline int Reprojector::caculateGridSize(const int wight, const int height, const int N)
{
    return floorf(sqrt(float(wight*height)/N)*0.6);
}

void Reprojector::initializeGrid(vilo::AbstractCamera* cam)
{
    grid_.cell_size = caculateGridSize(cam->width(), cam->height(), Config::maxFts());

    grid_.grid_n_cols = std::ceil(static_cast<double>(cam->width()) /grid_.cell_size);
    grid_.grid_n_rows = std::ceil(static_cast<double>(cam->height())/grid_.cell_size);
    grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);
    grid_.seeds.resize(grid_.grid_n_cols*grid_.grid_n_rows);
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell*& c){ c = new Cell; });
    std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell*& s){ s = new Sell; });
    grid_.cell_order.resize(grid_.cells.size());
    for(size_t i=0; i<grid_.cells.size(); ++i)
        grid_.cell_order[i] = i;

    std::random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); 
}

void Reprojector::resetGrid()
{
    n_matches_ = 0;n_trials_ = 0;n_seeds_ = 0;n_filters_ = 0;
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ c->clear(); });
    std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell* s){ s->clear(); });
    std::random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end());
    nFeatures_ = 0;
}

void Reprojector::reprojectMap( FramePtr frame)
{
    resetGrid();
    std::vector< pair<Vector2d, Point*> > all_px_pts_vec;
    VILO_START_TIMER("reproject_kfs");
    
    if(!map_->point_candidates_.temporaryPoints_.empty())
    {
        DepthFilter::lock_t lock(depth_filter_->seeds_mut_);
        
        size_t n = 0;
        auto ite = map_->point_candidates_.temporaryPoints_.begin();
        while(ite != map_->point_candidates_.temporaryPoints_.end())
        {
            if(ite->first->seedStates_ == 0) 
            {
                ite++;
                continue;
            }

            boost::unique_lock<boost::mutex> lock(map_->point_candidates_.mut_);
            map_->safeDeleteTempPoint(*ite);
            ite = map_->point_candidates_.temporaryPoints_.erase(ite);
            n++;
        }
        sum_seed_ -= n;
    }
  
    FramePtr LastFrame = frame->m_last_frame;
    size_t nCovisibilityGraph = 0;
    for(vector<Frame*>::iterator it = LastFrame->connectedKeyFrames.begin(); it != LastFrame->connectedKeyFrames.end(); ++it)
    {
        Frame* repframe = *it;
        FramePtr repFrame = NULL;
        if(!map_->getKeyframeById(repframe->id_, repFrame))
            continue;

        if(repFrame->lastReprojectFrameId_ == frame->id_)
            continue;
        repFrame->lastReprojectFrameId_ = frame->id_;

        for(auto ite = repFrame->fts_.begin(); ite != repFrame->fts_.end(); ++ite)
        {
            if((*ite)->point == NULL)
                continue;

            int pt_state = (*ite)->point->getPointState();
            if(pt_state== Point::TYPE_TEMPORARY)
                continue;

            if((*ite)->point->last_projected_kf_id_ == frame->id_)
                continue;

            (*ite)->point->last_projected_kf_id_ = frame->id_;

            checkPoint(frame, (*ite)->point, all_px_pts_vec);
            
        }

        nCovisibilityGraph++;
    }
    assert(nCovisibilityGraph == LastFrame->connectedKeyFrames.size());
    LastFrame->connectedKeyFrames.clear();
    list< pair<FramePtr,double> > close_kfs;
    map_->getCloseKeyframes(frame, close_kfs);

    close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) < boost::bind(&std::pair<FramePtr, double>::second, _2));

    size_t n = nCovisibilityGraph;
    for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end(); it_frame!=ite_frame && n<options_.max_n_kfs; ++it_frame)
    {
        FramePtr ref_frame = it_frame->first;

        if(ref_frame->lastReprojectFrameId_ == frame->id_)
            continue;
        ref_frame->lastReprojectFrameId_ = frame->id_;

        for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end(); it_ftr!=ite_ftr; ++it_ftr)
        {
            if((*it_ftr)->point == NULL)
                continue;

            int pt_state = (*it_ftr)->point->getPointState();
            if(pt_state == Point::TYPE_TEMPORARY)
                continue;

            if(pt_state == Point::TYPE_DELETED)
                continue;

            if((*it_ftr)->point->last_projected_kf_id_ == frame->id_)
                continue;

            (*it_ftr)->point->last_projected_kf_id_ = frame->id_;

            checkPoint(frame, (*it_ftr)->point, all_px_pts_vec);

        }

        ++n;
    }
    VILO_STOP_TIMER("reproject_kfs");
    VILO_START_TIMER("reproject_candidates");
    {
        boost::unique_lock<boost::mutex> lock(map_->point_candidates_.mut_);
        auto it=map_->point_candidates_.candidates_.begin();
        while(it!=map_->point_candidates_.candidates_.end())
        {
            if(!checkPoint(frame, it->first, all_px_pts_vec))
            {
                it->first->n_failed_reproj_ += 3;
                if(it->first->n_failed_reproj_ > 30)
                {
                    map_->point_candidates_.deleteCandidate(*it);
                    it = map_->point_candidates_.candidates_.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }
    VILO_STOP_TIMER("reproject_candidates");
    auto itk = map_->point_candidates_.temporaryPoints_.begin();
    while(itk != map_->point_candidates_.temporaryPoints_.end())
    {
        assert(itk->first->last_projected_kf_id_ != frame->id_);

        if(itk->first->isBad_) {
            itk++;
            continue;
        }
        
        itk->first->last_projected_kf_id_ = frame->id_;

        Point* tempPoint = itk->first;
        Feature* tempFeature = itk->second;

        double idist = tempPoint->getPointIdist();
        tempPoint->setPointIdistAndPose(idist);

        if(!checkPoint(frame, itk->first, all_px_pts_vec))
        {
            itk->first->n_failed_reproj_ += 3;
            if(itk->first->n_failed_reproj_ > 30)
                itk->first->isBad_ = true;
        }
        itk++;
    }

    VILO_START_TIMER("feature_align");

    if(all_px_pts_vec.size() < Config::maxFts()+50) 
    {
        reprojectCellAll(all_px_pts_vec, frame);
    }
    else
    {
        for(size_t i=0; i<grid_.cells.size(); ++i)
        {
            if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, false, false))
            {
                ++n_matches_;
            }
            if(n_matches_ >= (size_t) Config::maxFts())
                break;
        }

        if(n_matches_ < (size_t) Config::maxFts())
        {
            for(size_t i=grid_.cells.size()-1; i>0; --i)
            {
                if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, true, false))
                {
                    ++n_matches_;
                }
                if(n_matches_ >= (size_t) Config::maxFts())
                    break;
            }
        }

        if(n_matches_ < (size_t) Config::maxFts())
        {
            for(size_t i=0; i<grid_.cells.size(); ++i)
            {
                reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, true, true);

                if(n_matches_ >= (size_t) Config::maxFts())
                    break;
            }
        }
    }
    
    if(n_matches_ < 100 && options_.reproject_unconverged_seeds)
    {
        DepthFilter::lock_t lock(depth_filter_->seeds_mut_);
        for(auto it = depth_filter_->seeds_.begin(); it != depth_filter_->seeds_.end(); ++it)
        {
            if(sqrt(it->sigma2) < it->z_range/options_.reproject_seed_thresh && !it->haveReprojected)
                checkSeed(frame, *it, it);   
        }

        for(size_t i=0; i<grid_.seeds.size(); ++i)
        {
            if(reprojectorSeeds(*grid_.seeds.at(grid_.cell_order[i]), frame))
            {
                ++n_matches_;
            }
            if(n_matches_ >= (size_t) Config::maxFts())
                break;
        }
    }
    
    VILO_STOP_TIMER("feature_align");
}

bool Reprojector::pointQualityComparator(Candidate& lhs, Candidate& rhs)
{
    int l_type = lhs.pt->getPointState();
    int r_type = rhs.pt->getPointState();
    if(l_type != r_type)
        return (l_type > r_type);
    else
    {
        if(lhs.pt->ftr_type_ > rhs.pt->ftr_type_)
            return true;
        return false;
    }
}

bool Reprojector::seedComparator(SeedCandidate& lhs, SeedCandidate& rhs)
{
  return (lhs.seed.sigma2 < rhs.seed.sigma2);
}


bool Reprojector::reprojectCell(Cell& cell, FramePtr frame, bool is_2nd, bool is_3rd)
{   
    if(cell.empty()) return false;

    if(!is_2nd)
        cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));

    Cell::iterator it=cell.begin();

    int succees = 0;
    
    while(it!=cell.end())
    {
        ++n_trials_;

        int pt_state = it->pt->getPointState();
        if(pt_state == Point::TYPE_DELETED)
        {
            it = cell.erase(it);
            continue;
        }
        
        bool is_matched = matcher_.findMatchDirect(*it->pt, *frame, it->px);

        if(!is_matched)
        {   
            it->pt->n_failed_reproj_++;
            if(pt_state == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
                map_->safeDeletePoint(it->pt);
            
            if(pt_state == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30)
                map_->point_candidates_.deleteCandidatePoint(it->pt);

            if(pt_state == Point::TYPE_TEMPORARY && it->pt->n_failed_reproj_ > 30)
                it->pt->isBad_ = true;

            it = cell.erase(it);
            continue;
        }

        it->pt->n_succeeded_reproj_++;
        if(pt_state == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
            it->pt->setPointState(4);

        Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
        frame->addFeature(new_feature);

        new_feature->point = it->pt;

        if(matcher_.ref_ftr_->type == Feature::EDGELET)
        {
            new_feature->type = Feature::EDGELET;
            new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
            new_feature->grad.normalize();
        }
        else if(matcher_.ref_ftr_->type == Feature::GRADIENT)
            new_feature->type = Feature::GRADIENT;
        else
            new_feature->type = Feature::CORNER;


        it = cell.erase(it);

        if(!is_3rd)
            return true;
        else
        {
            succees++;
            n_matches_++;
            if(succees >= 3 || n_matches_ >= Config::maxFts()) 
                return true;
        }
    }

    return false;
}

bool Reprojector::reprojectorSeeds(Sell& sell, FramePtr frame)
{
    sell.sort(boost::bind(&Reprojector::seedComparator, _1, _2));
    Sell::iterator it=sell.begin();
    while(it != sell.end())
    {
        if(matcher_.findMatchSeed(it->seed, *frame, it->px))
        {
            assert(it->seed.ftr->point == NULL);

            ++n_seeds_;
            sum_seed_++;

            Vector3d pHost = it->seed.ftr->f*(1./it->seed.mu);
            Sophus::SE3 Twf_h = it->seed.ftr->frame->getFramePose().inverse();
            Vector3d xyz_world(Twf_h * pHost);
            Point* point = new Point(xyz_world, it->seed.ftr);

            point->idist_ = it->seed.mu;
            point->hostFeature_ = it->seed.ftr;
            
            point->setPointState(1);

            if(it->seed.ftr->type == Feature::EDGELET)
                point->ftr_type_ = Point::FEATURE_EDGELET;
            else if(it->seed.ftr->type == Feature::CORNER)
                point->ftr_type_ = Point::FEATURE_CORNER;
            else
                point->ftr_type_ = Point::FEATURE_GRADIENT;

            Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);      
            if(matcher_.ref_ftr_->type == Feature::EDGELET)
            {
                new_feature->type = Feature::EDGELET;
                new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
                new_feature->grad.normalize();
            }
            else if(matcher_.ref_ftr_->type == Feature::GRADIENT)
                new_feature->type = Feature::GRADIENT;
            else
                new_feature->type = Feature::CORNER;

            new_feature->point = point;

            frame->addFeature(new_feature);

            it->seed.haveReprojected = true;
            it->seed.temp = point;
            point->seedStates_ = 0;
            map_->point_candidates_.addPauseSeedPoint(point);

            it = sell.erase(it);
            return true;
        }
        else
            ++it;
    }

    return false;
}

bool Reprojector::checkPoint(FramePtr frame, Point* point, vector< pair<Vector2d, Point*> >& cells)
{
    double idist = point->getPointIdist();
    Vector3d pHost = point->hostFeature_->f * (1.0/idist);
    Sophus::SE3 Tfw_t = frame->getFramePose();
    Sophus::SE3 Tfw_h = point->hostFeature_->frame->getFramePose();
    Vector3d pTarget = (Tfw_t * Tfw_h.inverse())*pHost;
    if(pTarget[2] < 0.00001) return false;    

    Vector2d px(frame->cam_->world2cam(pTarget));
    
    if(frame->cam_->isInFrame(px.cast<int>(), 8)) 
    {
        const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                    + static_cast<int>(px[0]/grid_.cell_size);
        grid_.cells.at(k)->push_back(Candidate(point, px));

        cells.push_back(make_pair(px, point));

        nFeatures_++;

        return true;
    }
    return false;
}

bool Reprojector::checkSeed(
    FramePtr frame, Seed& seed, 
    list< Seed, aligned_allocator<Seed> >::iterator index)
{
    Sophus::SE3 Tfw_t = frame->getFramePose();
    Sophus::SE3 Tfw_h = seed.ftr->frame->getFramePose();
    Sophus::SE3 Tth = Tfw_t * Tfw_h.inverse();
    Vector3d pTarget = Tth*(1.0/seed.mu * seed.ftr->f);
    if(pTarget[2] < 0.001) return false;
    
    Vector2d px(frame->cam_->world2cam(pTarget));

    if(frame->cam_->isInFrame(px.cast<int>(), 8))
    {
        const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                    + static_cast<int>(px[0]/grid_.cell_size);
        grid_.seeds.at(k)->push_back(SeedCandidate(seed, px, index));
        return true;
    }
    return false;
}


void Reprojector::reprojectCellAll(vector< pair<Vector2d, Point*> >& cell, FramePtr frame)
{
    if(cell.empty()) return;

    vector< pair<Vector2d, Point*> >::iterator it = cell.begin();
    while(it != cell.end())
    {
        ++n_trials_;

        int pt_state = it->second->getPointState();
        
        if(pt_state == Point::TYPE_DELETED)
        {
            it = cell.erase(it);
            continue;
        }

        bool is_match = matcher_.findMatchDirect(*(it->second), *frame, it->first);
        
        if(!is_match)
        {
            it->second->n_failed_reproj_++;

            if(pt_state == Point::TYPE_UNKNOWN && it->second->n_failed_reproj_ > 15)
                map_->safeDeletePoint(it->second);

            if(pt_state == Point::TYPE_CANDIDATE && it->second->n_failed_reproj_ > 30)
                map_->point_candidates_.deleteCandidatePoint(it->second);
            
            if(pt_state == Point::TYPE_TEMPORARY && it->second->n_failed_reproj_ > 30)
                it->second->isBad_ = true;

            it = cell.erase(it);
            continue;
        }

        it->second->n_succeeded_reproj_++;
        
        if(pt_state == Point::TYPE_UNKNOWN && it->second->n_succeeded_reproj_ > 10)
            it->second->setPointState(4);

        Feature* new_feature = new Feature(frame.get(), it->first, matcher_.search_level_);
        frame->addFeature(new_feature);
        new_feature->point = it->second;

        if(matcher_.ref_ftr_->type == Feature::EDGELET)
        {
            new_feature->type = Feature::EDGELET;
            new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
            new_feature->grad.normalize();
        }
        else if(matcher_.ref_ftr_->type == Feature::GRADIENT)
            new_feature->type = Feature::GRADIENT;
        else
            new_feature->type = Feature::CORNER;

        it = cell.erase(it);

        n_matches_++;
        if(n_matches_ >= (size_t)Config::maxFts()) return;
    }
}

} // namespace vilo
