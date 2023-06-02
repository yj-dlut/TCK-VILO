#include <vilo/feature_detection.h>
#include <vilo/config.h>
#include <vilo/feature.h>
#include <fast/fast.h>
#include <boost/timer.hpp>

#include <thread>

#include "vilo/vikit/vision.h"

using namespace cv;

namespace vilo {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const float& detection_threshold,
    Features& fts)
{
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    vector<fast::fast_xy> fast_corners;
    fast::fast_corner_detect_plain_8(
      (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, img_pyr[L].rows, img_pyr[L].cols, detection_threshold, fast_corners);

    vector<int> scores, nm_corners;
    fast::fast_corner_score_8((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, detection_threshold, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);
      if(grid_occupancy_[k])
        continue;
      const float score = vilo::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      if(score > corners.at(k).score)
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
    }
  }
  
  std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
    if(c.score > detection_threshold)
      fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
  });

  resetGrid();
}

EdgeletDetector::EdgeletDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void EdgeletDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const float& detection_threshold,
    Features& fts)
{
  EdgeLets edgelets(grid_n_cols_*grid_n_rows_, EdgeLet(Vector2i(0,0), Vector2i(0,0), 0, 0.0f, false));
  Vector2d normal;

  for(int L = 0; L < n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    const int cell_size_pyr = cell_size_/scale;

    for(size_t index = 0; index < grid_occupancy_.size(); ++index)
    {
      if(grid_occupancy_.at(index))
        continue;
        
      const int u = index % grid_n_cols_*cell_size_pyr; 
      const int v = index / grid_n_cols_*cell_size_pyr;
      if( u + cell_size_pyr <= frame->canny_[L].cols && v + cell_size_pyr <= frame->canny_[L].rows )
      {
        float grad_max = 0;  
        for(int i = 0; i < cell_size_pyr; ++i)
          for(int j = 0; j < cell_size_pyr; ++j)
          {
            if(frame->canny_[L].ptr<uchar>(v + i)[u + j] == 0)
              continue;

            int gx = frame->grad_pyr_[L].at<cv::Vec2s>(v+i, u+j)[0];
            int gy = frame->grad_pyr_[L].at<cv::Vec2s>(v+i, u+j)[1];
            float grad = sqrtf(gx*gx + gy*gy);

            if(grad > grad_max && grad > 20*detection_threshold && grad > edgelets.at(index).score)
            {
              grad_max = grad;
              edgelets.at(index).is_set = true;
              edgelets.at(index).mid = Vector2i((u+j)*scale, (v+i)*scale); 

              normal = Vector2d(gx, gy);
              normal.normalize();
              edgelets.at(index).grad = normal;
              edgelets.at(index).level = L;
              edgelets.at(index).score = grad_max;
            }
          }
      }
    }
  }

  for(auto it = edgelets.begin(); it != edgelets.end(); ++it)
  {
    if(!it->is_set)
      continue;
    
    Vector2i center = it->mid;
    if(!frame->cam_->isInFrame(center, border_))
      continue;
    fts.push_back( new Feature(frame, Vector2d(center.x(), center.y()), it->grad, it->level) );
  }

  resetGrid();
}

GradientDetector::GradientDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void GradientDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const float& detection_threshold,
    Features& fts)
{
  Gradients gradients(grid_n_cols_*grid_n_rows_, Gradient(0, 0, 0, 0));

  for(int L = 0; L < n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    const int cell_size_pyr = cell_size_/scale;

    const int minBorderX = border_;
    const int minBorderY = minBorderX;
    const int maxBorderX = img_pyr[L].cols-border_;
    const int maxBorderY = img_pyr[L].rows-border_;

    for(size_t index = 0; index < grid_occupancy_.size(); ++index)
    {
        if(grid_occupancy_.at(index))
            continue;
        int iniX = index % grid_n_cols_ * cell_size_pyr; 
        int iniY = index / grid_n_cols_ * cell_size_pyr;

        if(iniX > maxBorderX || iniY > maxBorderY)
            continue;

        int maxX = iniX + cell_size_pyr;
        int maxY = iniY + cell_size_pyr;

        if(maxX > maxBorderX)
            maxX = maxBorderX;
        if(maxY > maxBorderY)
            maxY = maxBorderY;
        if(iniX < minBorderX)
            iniX = minBorderX;
        if(iniY < minBorderY)
            iniY = minBorderY;

        float max_delta = 0;

         for(int y = iniY; y < maxY; ++y)
            for(int x = iniX; x < maxX; ++x)
            {
                const int gx = frame->grad_pyr_[L].at<cv::Vec2s>(y, x)[0];
                const int gy = frame->grad_pyr_[L].at<cv::Vec2s>(y, x)[1];
                float grad = sqrtf(gx*gx + gy*gy); 

                if(grad > max_delta)
                {
                    max_delta = grad;
                    if(grad>20*detection_threshold && grad>gradients.at(index).delta)
                    {
                        gradients.at(index) = Gradient(x*scale, y*scale, grad, L);
                    }
                }
            }
    }
  }

  for(auto it = gradients.begin(); it != gradients.end(); ++it)
  {
    if(it->delta < detection_threshold)
      continue;
      
    fts.push_back(new Feature(frame, Vector2d(it->x, it->y), it->level, Feature::GRADIENT));
  }

  resetGrid();
}

FeatureExtractor::FeatureExtractor(
    const int width, const int height, const int cellSize, const int levels, bool isInit)
{
    cellSize_ = cellSize;

    width_ = width;
    height_ = height;
    vecWidth_.resize(levels);
    vecHeight_.resize(levels);
    vecWidth_[0] = width;
    vecHeight_[0] = height;
    for(int i=1; i<levels; ++i)
    {
        vecWidth_[i] = static_cast<int>(vecWidth_[i-1]/2);
        vecHeight_[i] = static_cast<int>(vecHeight_[i-1]/2);
    }

    nCols_ = std::ceil(static_cast<double>(width)/cellSize);
    nRows_ = std::ceil(static_cast<double>(height)/cellSize);

    nLevels_ = levels;
    featurePerLevel_.resize(nLevels_);
    cornerPerLevel_.resize(nLevels_);
    gradPerLevel_.resize(nLevels_);

    isInit_ = isInit;
    if(isInit)
        nFeatures_ = 2000;
    else
        nFeatures_ = Config::maxFts()+100;


    extFeatures_ = 0;

    vGrids_.resize(levels);
    vGridCols_.resize(levels);
    vGridRows_.resize(levels);
    haveFeatures_.resize(levels);
    for(int i = 0; i < levels; ++i)
    {
        const int gridPrySize = gridSize_/(1<<i);  
        vGrids_[i] = gridPrySize;
        vGridCols_[i] = std::ceil(static_cast<double>(vecWidth_[i])/gridPrySize);
        vGridRows_[i] = std::ceil(static_cast<double>(vecHeight_[i])/gridPrySize);
        haveFeatures_[i].resize(vGridCols_[i]*vGridRows_[i], false);
    }


    m_egde_filter = false;
    
}

void FeatureExtractor::detect(
    Frame* frame, const float initThresh, const float minThresh, Features& fts, Frame* last_frame)
{
    frame_ = frame;
    initThresh_ = initThresh;
    minThresh_ = minThresh;

    needFeatures_ = nFeatures_ - extFeatures_;

    if(last_frame != NULL)
    {
        m_egde_filter = true;
        m_last_frame = last_frame;
        findEpiHole();
    }
    else
    {
        m_egde_filter = false;
        m_last_frame = NULL;
    }

    featurePerLevel_.resize(nLevels_);
    fastDetectMT(frame->img_pyr_);

    if(isInit_)
    {
        fillingHole(frame->img_pyr_[0], 0);
    }
    else
    {
        edgeLetDetectMT(frame->img_pyr_);
    }


    for(size_t level = 0; level < featurePerLevel_.size(); ++level)
        for(size_t j = 0; j < featurePerLevel_[level].size(); ++j)
            allFeturesToDistribute_.push_back(featurePerLevel_[level].at(j));


    resultFeatures_ = computeKeyPointsOctTree(allFeturesToDistribute_, 0, width_, 0, height_, 0);

    for(size_t i=0; i<resultFeatures_.size(); ++i)
    {
        KeyPoint keyPoint = resultFeatures_[i];

        if(keyPoint.species == kCornerHigh)
            fts.push_back(new Feature(frame_, Vector2d(keyPoint.x, keyPoint.y), keyPoint.level));
        else if(keyPoint.species == kGrad)
        {
            Feature* feature = new Feature(frame_, Vector2d(keyPoint.x, keyPoint.y), keyPoint.level, Feature::GRADIENT);

            Vector2d normal(keyPoint.gx, keyPoint.gy);
            normal.normalize();
            feature->grad = normal;

            fts.push_back(feature);
        }
        else
        {
            Feature* feature = new Feature(frame_, Vector2d(keyPoint.x, keyPoint.y), keyPoint.level, Feature::EDGELET);

            Vector2d normal(keyPoint.gx, keyPoint.gy);
            normal.normalize();
            feature->grad = normal;

            fts.push_back(feature);
        }
    }

    resetGrid();

    allFeturesToDistribute_.clear();
    featurePerLevel_.clear();
    resultFeatures_.clear();

    extFeatures_ = 0;
}

void FeatureExtractor::fastDetectMT(const ImgPyr& img_pyr)
{
    if(nLevels_ == 1)
    {
        fastDetect(img_pyr);
    }
    else
    {
        assert(nLevels_ == 3);
        std::thread thread0(&FeatureExtractor::fastDetectST, this, std::ref(img_pyr[0]), 0);
        std::thread thread1(&FeatureExtractor::fastDetectST, this, std::ref(img_pyr[1]), 1);
        std::thread thread2(&FeatureExtractor::fastDetectST, this, std::ref(img_pyr[2]), 2);

        thread0.join();
        thread1.join();
        thread2.join();
    }
}

void FeatureExtractor::fastDetectST(const cv::Mat& imageLevel, const int Level)
{
    const int border = 8;
    const int scale = (1<<Level);
    const short fastThresh = floor(minThresh_);

    vector<fast::fast_xy> fastCorners9;
    fast::fast_corner_detect_9_sse2(
            (fast::fast_byte*)imageLevel.data, imageLevel.cols, imageLevel.rows, imageLevel.cols, fastThresh, fastCorners9);

    vector<int> scores9, nonMaxCornersIndex9;
    fast::fast_corner_score_9((fast::fast_byte*)imageLevel.data, imageLevel.cols, fastCorners9, fastThresh, scores9);
    fast::fast_nonmax_3x3(fastCorners9, scores9, nonMaxCornersIndex9);

    for(vector<int>::iterator it=nonMaxCornersIndex9.begin(), ite=nonMaxCornersIndex9.end(); it!=ite; ++it)
    {
        fast::fast_xy xy = fastCorners9.at(*it);

        if(xy.x < border || xy.x > vecWidth_[Level]-border || xy.y < border || xy.y > vecHeight_[Level]-border)
            continue;

        haveFeatures_[Level].at(getCellIndex(xy.x, xy.y, Level)) = true;

        featurePerLevel_[Level].push_back(
            KeyPoint(xy.x*scale, xy.y*scale, vilo::shiTomasiScore(imageLevel, xy.x, xy.y), Level, kCornerHigh));
    }
}

void FeatureExtractor::fastDetect(const ImgPyr& img_pyr)
{
    const int border = 8;

    for(int L=0; L<nLevels_; ++L)
    {
        const int scale = (1<<L);
        vector<fast::fast_xy> fastCorners;

        fast::fast_corner_detect_9_sse2(
            (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, img_pyr[L].rows, img_pyr[L].cols, minThresh_, fastCorners);
        
        vector<int> scores, nonMaxCornersIndex;
        fast::fast_corner_score_9((fast::fast_byte*)img_pyr[L].data, img_pyr[L].cols, fastCorners, minThresh_, scores);

        fast::fast_nonmax_3x3(fastCorners, scores, nonMaxCornersIndex);

        for(vector<int>::iterator it=nonMaxCornersIndex.begin(), ite=nonMaxCornersIndex.end(); it!=ite; ++it)
        {
            fast::fast_xy xy = fastCorners.at(*it);

            if(xy.x < border || xy.x > vecWidth_[L]-border || xy.y < border || xy.y > vecHeight_[L]-border)
                continue;

            const float response = vilo::shiTomasiScore(img_pyr[L], xy.x, xy.y);

            int index = getCellIndex(xy.x, xy.y, L);
            haveFeatures_[L].at(index) = true;

            featurePerLevel_[L].push_back(KeyPoint(xy.x*scale, xy.y*scale, response, L, kCornerHigh));
        }
    }
}

void FeatureExtractor::gradDetectMT(const ImgPyr& img_pyr)
{
    if(nLevels_ == 1)
    {
        gradDetect(img_pyr);
    }
    else
    {
        assert(nLevels_ == 3);
        std::thread thread0(&FeatureExtractor::gradDetectST, this, std::ref(img_pyr[0]), 0);
        std::thread thread1(&FeatureExtractor::gradDetectST, this, std::ref(img_pyr[1]), 1);
        std::thread thread2(&FeatureExtractor::gradDetectST, this, std::ref(img_pyr[2]), 2);

        thread0.join();
        thread1.join();
        thread2.join();
    }
}

void FeatureExtractor::gradDetectST(const cv::Mat& imageLevel, const int Level)
{
    const int border = 8;
    const int scale = (1<<Level);
    const int gridPrySize = vGrids_[Level];
    const int minBorderX = border;
    const int minBorderY = minBorderX;
    const int maxBorderX = imageLevel.cols-border;
    const int maxBorderY = imageLevel.rows-border;

    for(size_t index = 0; index < haveFeatures_[Level].size(); ++index)
    {
        if(haveFeatures_[Level].at(index)) continue;

        int iniX = index % vGridCols_[Level] * gridPrySize;
        int iniY = index / vGridRows_[Level] * gridPrySize;
        if(iniX > maxBorderX || iniY > maxBorderY) continue;

        int maxX = iniX + gridPrySize;
        int maxY = iniY + gridPrySize;

        if(maxX > maxBorderX) maxX = maxBorderX;
        if(maxY > maxBorderY) maxY = maxBorderY;
        if(iniX < minBorderX) iniX = minBorderX;
        if(iniY < minBorderY) iniY = minBorderY;

        KeyPoint kp;
        bool isSet = false;
        float maxGrad = 0;
        for(int y=iniY; y<maxY; ++y)
        {
            for(int x=iniX; x<maxX; ++x)
            {
                const int gx = frame_->grad_pyr_[Level].at<cv::Vec2s>(y, x)[0];
                const int gy = frame_->grad_pyr_[Level].at<cv::Vec2s>(y, x)[1];

                const float grad = sqrtf(gx*gx + gy*gy);

                if(grad>20.0f*minThresh_ && grad>maxGrad)
                {
                    kp = KeyPoint(x*scale, y*scale, grad, Level, kGrad);
                    kp.gx = gx;
                    kp.gy = gy;

                    isSet = true;
                    maxGrad = grad;
                }
            }
        }

        if(isSet) {
            featurePerLevel_[Level].push_back(kp);
            haveFeatures_[Level].at(index) = true;
        }
    }
}


void FeatureExtractor::gradDetect(const ImgPyr& img_pyr)
{
    const int border = 8;

    for(int L = 0; L < nLevels_; ++L)
    {
        const int scale = (1<<L);
        const int gridPrySize = vGrids_[L];
        const int minBorderX = border;
        const int minBorderY = minBorderX;
        const int maxBorderX = img_pyr[L].cols-border;
        const int maxBorderY = img_pyr[L].rows-border;

        for(size_t index = 0; index < haveFeatures_[L].size(); ++index)
        {
            if(haveFeatures_[L].at(index))
                continue;

            int iniX = index % vGridCols_[L] * gridPrySize;
            int iniY = index / vGridRows_[L] * gridPrySize;

            if(iniX > maxBorderX || iniY > maxBorderY)
                continue;

            int maxX = iniX + gridPrySize;
            int maxY = iniY + gridPrySize;

            if(maxX > maxBorderX) maxX = maxBorderX;
            if(maxY > maxBorderY) maxY = maxBorderY;
            if(iniX < minBorderX) iniX = minBorderX;
            if(iniY < minBorderY) iniY = minBorderY;

            KeyPoint kp;
            bool isSet = false;
            float maxGrad = 0;
            for(int y=iniY; y<maxY; ++y)
            {
                for(int x=iniX; x<maxX; ++x)
                {
                    const int gx = frame_->grad_pyr_[L].at<cv::Vec2s>(y, x)[0];
                    const int gy = frame_->grad_pyr_[L].at<cv::Vec2s>(y, x)[1];
                    const float grad = sqrtf(gx*gx + gy*gy);


                    if(grad > 20.0f*minThresh_ && grad > maxGrad)
                    {
                        kp = KeyPoint(x*scale, y*scale, grad, L, kGrad);
                        kp.gx = gx;
                        kp.gy = gy;

                        isSet = true;
                        maxGrad = grad;
                    }
                }
            }

            if(isSet) {
                featurePerLevel_[L].push_back(kp);
                haveFeatures_[L].at(index) = true;
            }
        }
    }
}

void FeatureExtractor::edgeLetDetectMT(const ImgPyr& img_pyr)
{
    if(nLevels_ == 1)
        edgeLetDetectST(img_pyr[0], 0);
    else
    {
        assert(nLevels_ == 3);

        std::thread thread0(&FeatureExtractor::edgeLetDetectST, this, std::ref(img_pyr[0]), 0);
        std::thread thread1(&FeatureExtractor::edgeLetDetectST, this, std::ref(img_pyr[1]), 1);
        std::thread thread2(&FeatureExtractor::edgeLetDetectST, this, std::ref(img_pyr[2]), 2);

        thread0.join();
        thread1.join();
        thread2.join();
    }
}

void FeatureExtractor::edgeLetDetectST(const cv::Mat& imageLevel, const int Level)
{
    cv::Mat imgEdge;
    cv::Canny(frame_->sobelX_[Level], frame_->sobelY_[Level], imgEdge, 31*minThresh_, 70*minThresh_, true);

    const int border = 8;
    const int scale = (1<<Level);
    const int gridPrySize = vGrids_[Level];
    const int minBorderX = border;
    const int minBorderY = minBorderX;
    const int maxBorderX = imageLevel.cols-border;
    const int maxBorderY = imageLevel.rows-border;

    for(size_t index = 0; index < haveFeatures_[Level].size(); ++index)
    {
        if(haveFeatures_[Level].at(index)) continue;

        int iniX = index % vGridCols_[Level] * gridPrySize;
        int iniY = index / vGridRows_[Level] * gridPrySize;
        if(iniX > maxBorderX || iniY > maxBorderY) continue;

        int maxX = iniX + gridPrySize;
        int maxY = iniY + gridPrySize;

        if(maxX > maxBorderX) maxX = maxBorderX;
        if(maxY > maxBorderY) maxY = maxBorderY;
        if(iniX < minBorderX) iniX = minBorderX;
        if(iniY < minBorderY) iniY = minBorderY;

        KeyPoint kp;
        bool isSet = false;
        float maxGrad = 0;
        for(int y=iniY; y<maxY; ++y)
        {
            for(int x=iniX; x<maxX; ++x)
            {
                if(imgEdge.at<uchar>(y,x) == 0) continue;

                const short gx = frame_->sobelX_[Level].at<short>(y,x);
                const short gy = frame_->sobelY_[Level].at<short>(y,x);

                float grad = sqrtf(gx*gx + gy*gy);
                
                if(grad > maxGrad)
                {
                    kp = KeyPoint(x*scale, y*scale, grad, Level, kEdgeLet);
                    kp.gx = gx;
                    kp.gy = gy;

                    isSet = true;
                    maxGrad = grad;
                }
            }
        }

        if(isSet) {
            featurePerLevel_[Level].push_back(kp);
            haveFeatures_[Level].at(index) = true;
        }
    }
}


vector<KeyPoint> FeatureExtractor::computeKeyPointsOctTree(
    const vector<KeyPoint>& toDistributeKeys, 
    const int &minX, const int &maxX, const int &minY, const int &maxY, const int &level)
{
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; ++i)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),minY);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),minY);
        ni.BL = cv::Point2i(ni.UL.x,maxY);
        ni.BR = cv::Point2i(ni.UR.x,maxY);
        ni.vKeys.reserve(toDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    for(size_t i=0; i<toDistributeKeys.size(); ++i)
    {
        const int x = toDistributeKeys[i].x;

        vpIniNodes[x/hX]->vKeys.push_back(toDistributeKeys[i]);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit != lNodes.end())
    {
        if(lit->vKeys.size() == 1)
        {
            lit->bNoMore = true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit != lNodes.end())
        {
            if(lit->bNoMore)
            {
                ++lit;
                continue;
            }
            else
            {
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                if(n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if(n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }

        if((int)lNodes.size()>=nFeatures_ || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>nFeatures_)
        {
            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                std::sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=nFeatures_)
                        break;
                }

                if((int)lNodes.size()>=nFeatures_ || (int)lNodes.size()==prevSize)
                    bFinish = true;
            }
        }
    }

    std::vector<KeyPoint> vResultKeys;
    vResultKeys.reserve(nFeatures_);
    for(list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); ++lit)
    {
        vector<KeyPoint> &vNodeKeys = lit->vKeys;

        KeyPoint* pKP = &vNodeKeys[0];
        float maxScore = pKP->response;;
        bool haveOccur = false;

        if(pKP->species == kOccur) continue;

        if(vNodeKeys.size() > 1)
        {
            for(size_t k = 1; k < vNodeKeys.size(); ++k)
            {
                KeyPoint* pt = &vNodeKeys[k];

                if(pt->species == kOccur) {
                    haveOccur = true;
                    break;
                }

                if(pKP->species > vNodeKeys[k].species)
                {
                    pKP = &vNodeKeys[k];
                    maxScore = vNodeKeys[k].response;
                }
                else if(pKP->species == vNodeKeys[k].species)
                {
                    if(vNodeKeys[k].response > maxScore)
                    {
                        pKP = &vNodeKeys[k];
                        maxScore = vNodeKeys[k].response;
                    }
                }
            }
        }
        if(!haveOccur) vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void FeatureExtractor::fillingHole(const cv::Mat& imageLevel, const int Level)
{
    const int border = 8;
    const int scale = (1<<Level);
    const short fastThresh = 0.6*minThresh_ > 6? 0.6*minThresh_ : 6;

    vector<fast::fast_xy> fastCorners;
    fast::fast_corner_detect_plain_12(
            (fast::fast_byte*)imageLevel.data, imageLevel.cols, imageLevel.rows, imageLevel.cols, fastThresh, fastCorners);

    vector<int> scores, nonMaxCornersIndex;
    fast::fast_corner_score_12((fast::fast_byte*)imageLevel.data, imageLevel.cols, fastCorners, fastThresh, scores);
    fast::fast_nonmax_3x3(fastCorners, scores, nonMaxCornersIndex);

    for(vector<int>::iterator it=nonMaxCornersIndex.begin(), ite=nonMaxCornersIndex.end(); it!=ite; ++it)
    {
        fast::fast_xy xy = fastCorners.at(*it);
        if(xy.x < border || xy.x > vecWidth_[Level]-border || xy.y < border || xy.y > vecHeight_[Level]-border)
            continue;

        int index = getCellIndex(xy.x, xy.y, Level);
        if(haveFeatures_[Level].at(index)) continue;

        haveFeatures_[Level].at(index) = true;

        featurePerLevel_[Level].push_back(
            KeyPoint(xy.x*scale, xy.y*scale, vilo::shiTomasiScore(imageLevel, xy.x, xy.y), Level, kGrad));
    }
}

void FeatureExtractor::resetGrid()
{
    for(size_t i = 0; i < haveFeatures_.size(); ++i)
        std::fill(haveFeatures_[i].begin(), haveFeatures_[i].end(), false);
}

void FeatureExtractor::setGridOccpuancy(const Vector2d& px, Feature* occurFeature)
{
    allFeturesToDistribute_.push_back(KeyPoint(px.cast<float>().x(), px.cast<float>().y(), 0, 0, kOccur));

    extFeatures_++;
}

void FeatureExtractor::setExistingFeatures(const Features& fts)
{
    for(auto& ftr : fts)
    {
        allFeturesToDistribute_.push_back(KeyPoint(ftr->px.cast<float>().x(), ftr->px.cast<float>().y(), 0, 0, kOccur));
    }   

    extFeatures_ += fts.size();
}


void FeatureExtractor::findEpiHole()
{
    assert(m_last_frame != NULL);
    assert(m_last_frame == frame_->m_last_frame.get());
    epi_hole = frame_->w2c(m_last_frame->pos());
}

bool FeatureExtractor::edgeletFilter(int u_level, int v_level, short gx, short gy, int level, double& angle)
{
    Vector2d grad_dir(gx,gy);
    grad_dir.normalize();

    Vector2d hole_level(epi_hole/(1<<level));
    Vector2d epi_dir = (hole_level - Vector2d(u_level,v_level)).normalized();

    angle = fabs(grad_dir.dot(epi_dir));
    if(angle < 0.1) return false;

    return true;
}

} // namespace feature_detection
} // namespace vilo

