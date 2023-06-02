#pragma once

#include <mutex>
#include <opencv2/core/core.hpp>
#include <sophus/se3.h>
#include <pangolin/pangolin.h>

namespace vilo {
class Frame;
class Map;
class FrameHandlerMono; }


namespace vilo {

class Viewer
{
public:
    Viewer(vilo::FrameHandlerMono* vo);
    void run();
    bool CheckFinish();
    void DrawKeyFrames(const bool bDrawKF);
    void DrawMapRegionPoints();
    void DrawMapSeeds();
    void DrawConstraints();

    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);

private:
    vilo::FrameHandlerMono* _vo;

    std::mutex mMutexCurrentPose;
    std::vector< Sophus::SE3 > visual_pos_vec_;
    std::vector< Sophus::SE3 > lidar_pos_vec_;
    bool m_need_clear = true;
    Sophus::SE3  _CurrentPoseTwc ;
    int _drawedframeID=0;

    void SetFinish();
    bool mbFinished;
    std::mutex mMutexFinish;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
};
}
