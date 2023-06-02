#include <vilo/viewer.h>
#include <vilo/frame_handler_mono.h>
#include <vilo/map.h>
#include <vilo/frame.h>
#include <vilo/feature.h>
#include <vilo/point.h>
#include <vilo/depth_filter.h>
#include <pangolin/gl/gltext.h>

using namespace vilo;


namespace vilo {

Viewer::Viewer(vilo::FrameHandlerMono* vo): _vo(vo)
{
    mbFinished = false;
    mViewpointX =  0;
    mViewpointY =  -1.5;
    mViewpointZ =  5;
    mViewpointF =  200;

    mKeyFrameSize = 0.05;
    mKeyFrameLineWidth = 1.0;
    mCameraSize = 0.08;
    mCameraLineWidth = 3.0;

    mPointSize = 3.0;
}

bool Viewer::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

void Viewer::DrawKeyFrames(const bool bDrawKF)
{
    vilo::FramePtr lastframe = _vo->lastFrame();
    if(lastframe == NULL || lastframe->id_ == _drawedframeID)
    {
        //return;
    }
    else 
    {
        _drawedframeID = lastframe->id_ ;
        _CurrentPoseTwc = lastframe->getFramePose().inverse();
        
        visual_pos_vec_.clear();
        auto it = _vo->map_->keyframes_.begin();
        while(it != _vo->map_->keyframes_.end())
        {
            visual_pos_vec_.push_back((*it)->getFramePose().inverse());
            it++;
        }
    }
    if(visual_pos_vec_.empty()) return;

    std::list<LidarPtr> lidar_list = _vo->map_->lidars_list_;

    if(lidar_list.size() != 0)
    {
        vilo::LidarPtr lastlidar =  lidar_list.back();

        if(lastlidar == NULL || lastlidar->id_ == _drawedframeID)
        {
            //return;
        }
        else
        {
            _drawedframeID = lastlidar->id_ ;
            _CurrentPoseTwc = lastlidar->getLidarPose().inverse();
            lidar_pos_vec_.push_back(_CurrentPoseTwc);
            
            if(m_need_clear && _vo->is_lio_initialized_)
            {
                lidar_pos_vec_.clear();
                auto it = lidar_list.begin();
                while(it != lidar_list.end())
                {
                    lidar_pos_vec_.push_back((*it)->getLidarPose().inverse());
                    it++;
                }
                m_need_clear = false;
            }
            
        }

    }
    
    if(visual_pos_vec_.empty() && lidar_pos_vec_.empty())
        return;

    if(bDrawKF)
    {
        glPointSize(2);
        glBegin(GL_POINTS);
        glColor3f(1.0, 0.0, 0.0); 
        
        for(size_t i = 0; i<visual_pos_vec_.size();i++)
        {
            Sophus::SE3 Twc = visual_pos_vec_[i];
            glVertex3d(Twc.translation()[0], Twc.translation()[1], Twc.translation()[2]);
        }

        glColor3f(0.0, 0.0, 1.0);
        for(size_t i = 0; i<lidar_pos_vec_.size();i++)
        {
            Sophus::SE3 Twl = lidar_pos_vec_[i];
            glVertex3d(Twl.translation()[0], Twl.translation()[1], Twl.translation()[2]);
        }
        glEnd();
    }
}

void Viewer::DrawMapRegionPoints()
{
    glPointSize(mPointSize);
    glBegin(GL_POINTS);

    if( _vo->map_->keyframes_.empty())
        return;

    for(auto kf = _vo->map_->keyframes_.begin(); kf != _vo->map_->keyframes_.end(); ++kf)
        for(auto& ft: (*kf)->fts_)
        {
            if(ft->point == NULL) continue;
            Eigen::Vector3d Pw = ft->point->getPointPose();

            float color = float(ft->point->color_) / 255;
            if(color > 0.9) color = 0.9;

            glColor3f(color,color,color);
            glVertex3f( Pw[0],Pw[1],Pw[2]);
        }
    glEnd();
}

void Viewer::DrawConstraints()
{
    set<Frame*> LocalMap = _vo->sub_map_;
    if(LocalMap.empty())
        return;
    
    Vector3d posCurrent(_vo->lastFrame()->pos());

    if(LocalMap.empty()) return;

    glLineWidth(2.5);
    glColor4f(0.0f,1.0f,0.0f,0.6f);
    glBegin(GL_LINES);
    for(set<Frame*>::iterator it = LocalMap.begin(); it != LocalMap.end(); ++it)
    {
        Frame* target = *it;
        if(target->id_ == _vo->lastFrame()->id_) continue;

        Vector3d posTarget(target->pos());
        glVertex3d(posCurrent[0], posCurrent[1], posCurrent[2]);
        glVertex3d(posTarget[0], posTarget[1], posTarget[2]);
    }
    glEnd();
}

void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif
    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(_drawedframeID != 0)
    {
        Eigen::Matrix3d Rwc = _CurrentPoseTwc.rotation_matrix();
        Eigen::Vector3d twc = _CurrentPoseTwc.translation();

        M.m[0] = Rwc(0,0);
        M.m[1] = Rwc(1,0);
        M.m[2] = Rwc(2,0);
        M.m[3] = 0.0;

        M.m[4] = Rwc(0,1);
        M.m[5] = Rwc(1,1);
        M.m[6] = Rwc(2,1);
        M.m[7] = 0.0;

        M.m[8] = Rwc(0,2);
        M.m[9] = Rwc(1,2);
        M.m[10] = Rwc(2,2);
        M.m[11] = 0.0;

        M.m[12] = twc[0];
        M.m[13] = twc[1];
        M.m[14] = twc[2];
        M.m[15] = 1.0;
    }
    else
        M.SetIdentity();
}

void Viewer::run()
{
    mbFinished = false;
    pangolin::CreateWindowAndBind("VILOM", 848, 480);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(160));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show Trajactory",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",false,true);
    pangolin::Var<bool> menuShowConstrains("menu.Show Constrains",false,true);
    pangolin::OpenGlRenderState s_cam(
              pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,100,600,0.1,1000),
              pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
            ); 

    pangolin::View& d_cam = pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
          .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    bool bFollow = true;
    while(!CheckFinish())
    {
        usleep(10000);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        GetCurrentOpenGLCameraMatrix(Twc);

        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }


        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,0.5f);

        DrawCurrentCamera(Twc);

        DrawKeyFrames(menuShowKeyFrames);

        if(menuShowPoints) 
            DrawMapRegionPoints();

        if(menuShowConstrains)
            DrawConstraints();

        pangolin::FinishFrame();
    }

    pangolin::BindToContext("VILO");

}

}
