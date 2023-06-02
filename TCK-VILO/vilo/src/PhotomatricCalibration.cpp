
#include "vilo/PhotomatricCalibration.h"
#include "vilo/frame.h"
#include "vilo/feature.h"
#include "vilo/config.h"
#include "vilo/point.h"

#include "vilo/vikit/math_utils.h"


namespace vilo{

PhotomatricCalibration::PhotomatricCalibration(int patch_size, int width, int height): m_new_frame_set(false), 
                                                                                       m_width(width), 
                                                                                       m_height(height), 
                                                                                       m_patch_size(patch_size), 
                                                                                       m_max_id(0), 
                                                                                       m_min_id(0), 
                                                                                       m_optimization_block(NULL), 
                                                                                       m_is_optimizing(false)
{
    m_c1 = 6.49595;
    m_c2 = -0.430019;
    m_c3 = 0.266248;
    m_c4 = 0.058124;
    m_v1 = -0.15;
    m_v2 = 0;
    m_v3 = 0;


    m_width_2  = float(width)/2;
    m_height_2 = float(height)/2;
    m_max_radius = sqrt(m_width_2*m_width_2 + m_height_2*m_height_2);
    m_inverse_vignette_image = cv::Mat(height, width, CV_64F, 1.0);

    m_window_size = 12;

    m_keyframe_spacing = 6;

    m_current_frame = NULL;

    nr_optimize = 0;

    m_max_optimize = 80;

    m_finish = false;

    H_marg.setZero();
    b_marg.setZero();

    m_v1_eval = m_v2_eval = m_v3_eval = m_c1_eval = m_c2_eval = m_c3_eval = m_c4_eval = 0;

    setInverseResponseRaw();
    setInverseVignette();

    startThread();
}

void PhotomatricCalibration::startThread()
{
    m_thread = new thread(&vilo::PhotomatricCalibration::Run, this);
}

void PhotomatricCalibration::Run()
{
    while(1)
    {
        {
            std::unique_lock<mutex> lock(m_mutex_frame_list);
            while(!m_new_frame_set)
            {
                m_con_frame_queue.wait(lock);
            }
        }

        m_new_frame_set = false;

        m_current_frame->m_exposure_time = rapidExposureEstimationNonlinear(m_current_frame, false);
        m_current_frame->m_exposure_finish = true;  
        m_current_frame->m_pyr_raw.clear();


        m_et_estimate.push_back(m_current_frame->m_exposure_time);

        std::unique_lock<mutex> lock(m_mutex_opt);
        bool is_optimizing = m_is_optimizing;
        lock.unlock();

        if(is_optimizing) continue;


        if(nr_optimize > 100)
        {
            memoryRelease();
            if(!m_finish) m_finish = true;
            continue;
        }

        if(extractOptimizationBlock(false))
        {
            std::thread thread_optimazation(&PhotomatricCalibration::backEndOptimazation, this);
            thread_optimazation.detach();
        }
    }
}


void PhotomatricCalibration::addFrame(FramePtr frame, int step)
{
    std::unique_lock<mutex> lock(m_mutex_frame_list);

    if(static_cast<int>(m_frame_list.size()) > 150)
    {
        m_trash_frame_buffer.push_back(*m_frame_list.begin());
        m_frame_list.erase(m_frame_list.begin()); 
    }

    if(!frame->isKeyframe())
        m_frame_list.push_back(frame);

    m_new_frame_set = true;

    if(!m_frame_list.empty())
    {
        m_max_id = m_frame_list.back() ->keyFrameId_;  
        m_min_id = m_frame_list.front()->keyFrameId_;
    }
    
    if(m_frame_list.size() > 1)
        assert(m_max_id >= m_min_id);

    m_keyframe_spacing = step;
    m_window_size = m_keyframe_spacing*2;

    m_current_frame = frame;
    m_con_frame_queue.notify_one();
}


void PhotomatricCalibration::backEndOptimazation()
{
    std::unique_lock<mutex> lock_opt(m_mutex_opt);
    m_is_optimizing = true;
    lock_opt.unlock();


    evfOptimization(false);

    evfOptimization(false);

    if(m_type == kActive)
        evfOptimization(false);

    if(m_type == kActive)
        evfOptimization(false);



    if(m_type == kSliding)
        marginalizeFrame();



    setInverseResponseRaw();
    setInverseVignette();
    

    nr_optimize++;
    memoryRelease();


    lock_opt.lock();
    m_is_optimizing = false;
}

void PhotomatricCalibration::memoryRelease()
{
    std::unique_lock<mutex> lock(m_mutex_frame_list);

    auto it = m_trash_frame_buffer.begin();
    while(it != m_trash_frame_buffer.end())
    {
        Frame* frame = (*it).get();

        if(frame->m_kf_pc) break;

        if(frame->keyFrameId_ + Config::maxDropKeyframe() < m_max_id)
        {
            it = m_trash_frame_buffer.erase(it);
            delete frame;
            frame = NULL;
        }
        else
            ++it;
    }
}

double PhotomatricCalibration::rapidExposureEstimation(FramePtr curFrame, bool show_debug_prints)
{
    for(auto it = curFrame->fts_.begin(); it != curFrame->fts_.end(); ++it)
    {
        if((*it)->m_non_point) continue;

        getRadianceOutput(*it);
    }

    if(m_frame_list.size() < 10) return 1.0;

    double e_estimate = 0.0;
    double nr_estimates = 0;

    for(auto ite = curFrame->fts_.begin(); ite != curFrame->fts_.end(); ++ite)
    {
        Feature* curr_feature = *ite;
        if(curr_feature->m_non_point) continue;  

        vector<double> rad_mesurement;
        rad_mesurement.resize(curr_feature->radiances.size(), 0);

        int nr_features_used = 0;
        Feature* prev_feature = curr_feature->m_prev_feature;
        for(int k = 0; k < m_window_size; k++)
        {
            if(prev_feature == NULL) break;
            assert(!prev_feature->m_non_point);
            
            nr_features_used++;

            for(size_t i = 0; i < prev_feature->radiances.size(); ++i)
            {
                rad_mesurement[i] += prev_feature->radiances[i];
                assert(!std::isnan(prev_feature->radiances[i]));
            }

            if(prev_feature->m_prev_feature != NULL)
                assert(prev_feature->m_prev_feature->m_next_feature == prev_feature);

            prev_feature = prev_feature->m_prev_feature;
        }

        if(nr_features_used == 0) continue;

        for(size_t r = 0;r < rad_mesurement.size();r++)
            rad_mesurement.at(r) /= nr_features_used;

        for(size_t k = 0;k < rad_mesurement.size();k++)
        {
            if(fabs(rad_mesurement[k]) < 0.0001)
                continue;

            double weight = 1.0 / (curr_feature->outputs_grad[k]/255+1.0);

            double curr_e_estimate = (curr_feature->radiances[k] / rad_mesurement[k]);

            assert(!std::isnan(curr_feature->radiances[k]));
            assert(!std::isnan(rad_mesurement[k]));

            if(curr_e_estimate < 0.001 || curr_e_estimate > 100)
                continue;
            
            e_estimate   += weight*curr_e_estimate;
            nr_estimates += weight;
        }
    }

    const double exposure_time = e_estimate / nr_estimates;

    for(auto it = curFrame->fts_.begin(); it != curFrame->fts_.end(); ++it)
    {
        if((*it)->m_non_point) continue;

        for(size_t i = 0; i < (*it)->radiances.size(); ++i)
            (*it)->radiances[i] /= exposure_time;
    }

    if(show_debug_prints)
        cout << "[rapidExposureEstimation]: Current frame exposure time = " << exposure_time << endl;

    return exposure_time;
}

double PhotomatricCalibration::rapidExposureEstimationNonlinear(FramePtr cur_frame, bool show_debug_prints)
{
    for(auto it = cur_frame->fts_.begin(); it != cur_frame->fts_.end(); ++it)
    {
        if((*it)->m_non_point) continue;
        getRadianceOutput(*it);
    }

    if(m_frame_list.size() < 10) return 1.0;

    std::vector<vector<double> > radiances_mesurement;
    radiances_mesurement.resize(cur_frame->fts_.size());
    int feature_counter = -1;
    int rad_used = 0;
    std::vector<float> errors;
    const float exposure_init = cur_frame->m_exposure_time;
    for(auto ite = cur_frame->fts_.begin(); ite != cur_frame->fts_.end(); ++ite)
    {
        feature_counter++;
        radiances_mesurement[feature_counter].clear();

        Feature* cur_feature = *ite;
        if(cur_feature->m_non_point) continue;

        int nr_features_used = 0;
        Feature* prev_feature = cur_feature->m_prev_feature;
        for(int k = 0; k < m_window_size; k++)
        {
            if(prev_feature == NULL) break;
            assert(!prev_feature->m_non_point);

            nr_features_used++;

            if(radiances_mesurement[feature_counter].empty())
                for(size_t i = 0; i < prev_feature->radiances.size(); ++i)
                {
                    radiances_mesurement[feature_counter].push_back(prev_feature->radiances[i]);
                    assert(!std::isnan(prev_feature->radiances[i]));
                }
            else
                for(size_t i = 0; i < prev_feature->radiances.size(); ++i)
                {
                    radiances_mesurement[feature_counter].at(i) += prev_feature->radiances[i];
                    assert(!std::isnan(prev_feature->radiances[i]));
                }

            if(prev_feature->m_prev_feature != NULL)
                assert(prev_feature->m_prev_feature->m_next_feature == prev_feature);

            prev_feature = prev_feature->m_prev_feature;
        }

        if(nr_features_used == 0) continue;

        for(size_t r = 0;r < radiances_mesurement[feature_counter].size();r++)
        {
            radiances_mesurement[feature_counter].at(r) /= nr_features_used;
            errors.push_back(fabsf(exposure_init*radiances_mesurement[feature_counter].at(r)-cur_feature->radiances[r]));
        }

        rad_used++;
    }

    float huberTH = 1.4826*vilo::getMedian(errors);
    float outlierTH = 8*huberTH;
    if(outlierTH < 8.0) outlierTH = 8.0;

    if(show_debug_prints)
        cout << "[rapidExposureEstimationNonlinear]: Get " << rad_used << " points" << "  Huber = " << huberTH << endl;

    float energy_old = 0;
    float exposure_estimate = cur_frame->m_exposure_time;
    assert(exposure_estimate > 0 && !std::isnan(exposure_estimate));
    float update = 0;
    for(int iter = 0; iter < 3; ++iter)
    {
        float H = 0, b = 0;
        float energy_new = 0;
        int feature_counter = -1;
        int error_terms = 0;
        for(auto ite = cur_frame->fts_.begin(); ite != cur_frame->fts_.end(); ++ite)
        {
            feature_counter++;

            Feature* cur_feature = *ite;
            if(cur_feature->m_non_point) continue;

            if(radiances_mesurement[feature_counter].empty())
                continue;

            for(size_t k=0; k<radiances_mesurement[feature_counter].size(); ++k)
            {
                float rad_mesurement = radiances_mesurement[feature_counter].at(k);
                float irr_estimate = cur_feature->radiances[k];

                if(cur_feature->outputs[k] > 250 || cur_feature->outputs[k] < 5) continue;

                float residual = irr_estimate - exposure_estimate*rad_mesurement;

                float huber_weight = fabs(residual) < huberTH ? 1 : huberTH / fabs(residual);
                if(fabs(residual) > outlierTH) continue;

                float grad_weight = 1.0 / (cur_feature->outputs_grad[k]/255+1.0);
                float total_weight = huber_weight*grad_weight;

                H +=  rad_mesurement*rad_mesurement*total_weight;
                b -= -rad_mesurement*residual*total_weight;

                energy_new += residual*residual*total_weight;
                error_terms++;
            }
        }

        if(iter > 0 && energy_new > energy_old)
        {
            exposure_estimate -= update;
            break;
        }

        update = b/H;
        exposure_estimate += update;
        energy_old = energy_new;

        if(show_debug_prints)
            cout << "Iter " << iter << ":"
                 << "\t exposure_time = " << exposure_estimate
                 << "\t update = " << update
                 << "\t new chi1 mean = " << energy_new/error_terms << endl;

        if(fabsf(update) < 1e-5) break;
    }

    if(exposure_estimate < 0 && cur_frame->m_exposure_time < 0) assert(false);
    if(exposure_estimate < 0 && cur_frame->m_exposure_time > 0) exposure_estimate = cur_frame->m_exposure_time;

    for(auto it = cur_frame->fts_.begin(); it != cur_frame->fts_.end(); ++it)
    {
        if((*it)->m_non_point) continue;

        for(size_t i = 0; i < (*it)->radiances.size(); ++i)
            (*it)->radiances[i] /= exposure_estimate; 
    }

    return exposure_estimate;
}

void PhotomatricCalibration::bulidRadianceFeature(Feature*& ft)
{
    double u_scaled = ft->px[0]/(1<<ft->level);
    double v_scaled = ft->px[1]/(1<<ft->level);

    assert(u_scaled < ft->frame->img_pyr_[ft->level].cols+m_patch_size);
    assert(v_scaled < ft->frame->img_pyr_[ft->level].rows+m_patch_size);

    ft->radiances = bilinearInterpolateImagePatch(ft->frame->img_pyr_[ft->level], u_scaled, v_scaled, m_patch_size);
}

void PhotomatricCalibration::bulidOutputFeature(Feature*& ft)
{
    double u_scaled = ft->px[0]/(1<<ft->level);
    double v_scaled = ft->px[1]/(1<<ft->level);

    ft->outputs = bilinearInterpolateImagePatch(ft->frame->m_pyr_raw[ft->level], u_scaled, v_scaled, m_patch_size);

    vector<double> output_dx = bilinearInterpolateImagePatch(ft->frame->sobelX_[ft->level], u_scaled, v_scaled, m_patch_size);
    vector<double> output_dy = bilinearInterpolateImagePatch(ft->frame->sobelY_[ft->level], u_scaled, v_scaled, m_patch_size);
    for(size_t i = 0; i < output_dx.size(); ++i)
        ft->outputs_grad.push_back(sqrt(output_dx[i]*output_dx[i]+output_dy[i]*output_dy[i]));
}

void PhotomatricCalibration::getRadianceOutput(Feature*& ft)
{
    double u_scaled = ft->px[0]/(1<<ft->level);
    double v_scaled = ft->px[1]/(1<<ft->level);
    for(int idx=0;idx<n_pattern;idx++)
    {
        double x = u_scaled+m_pattern[idx][0];
        double y = v_scaled+m_pattern[idx][1];

        assert(x <= ft->frame->img_pyr_[ft->level].cols && x >= 0);
        assert(y <= ft->frame->img_pyr_[ft->level].rows && y >= 0);

        ft->radiances.push_back(bilinearInterpolateImage(ft->frame->img_pyr_[ft->level], x, y)); 
        ft->outputs.push_back(bilinearInterpolateImage(ft->frame->m_pyr_raw[ft->level], x, y));

        double gx = bilinearInterpolateImage(ft->frame->sobelX_[ft->level], x, y);
        double gy = bilinearInterpolateImage(ft->frame->sobelY_[ft->level], x, y);
        ft->outputs_grad.push_back(sqrt(gx*gx+gy*gy));
    }
}

std::vector<double> 
PhotomatricCalibration::bilinearInterpolateImagePatch(const cv::Mat& image,double x,double y,int patch_size)
{
    std::vector<double> result;
    for(int y_offset = -patch_size;y_offset <= patch_size;y_offset++)
        for(int x_offset = -patch_size;x_offset <= patch_size;x_offset++)
        {
            double o_value = bilinearInterpolateImage(image,x+x_offset,y+y_offset);
            assert(!std::isnan(o_value));

            result.push_back(o_value);
        }

    return result;
}

double PhotomatricCalibration::bilinearInterpolateImage(const cv::Mat& image,double x,double y)
{
    double floor_x = std::floor(x);
    double ceil_x  = std::ceil(x);
    
    double floor_y = std::floor(y);
    double ceil_y  = std::ceil(y);
    
    double x_normalized = x - floor_x;
    double y_normalized = y - floor_y;
    double w1 = (1.0-x_normalized)*(1.0-y_normalized);
    double w2 = x_normalized*(1.0-y_normalized);
    double w3 = (1.0-x_normalized)*y_normalized;
    double w4 = x_normalized*y_normalized;
    
    double i1 = static_cast<double>(image.at<uchar>(floor_y,floor_x));
    double i2 = static_cast<double>(image.at<uchar>(floor_y,ceil_x));
    double i3 = static_cast<double>(image.at<uchar>(ceil_y,floor_x));
    double i4 = static_cast<double>(image.at<uchar>(ceil_y,ceil_x));
    
    return w1*i1 + w2*i2 + w3*i3 + w4*i4;
}

bool PhotomatricCalibration::extractOptimizationBlock(bool show_debug_prints)
{  
    if(m_optimization_block != NULL)
        delete m_optimization_block;

    m_optimization_block = new OptimizationBlock(m_patch_size);

    if(m_type == kActive)
    {
        std::unique_lock<mutex> lock_list(m_mutex_frame_list);
        int nr_images_in_database = static_cast<int>(m_frame_list.size());

        if(nr_images_in_database < m_keyframe_spacing*8)
            return false;

        int frame_counter = -1;
        for(auto it = m_frame_list.begin(); it != m_frame_list.end(); ++it)
        {
            assert(!(*it)->isKeyframe());

            frame_counter++;

            if(frame_counter % m_keyframe_spacing != 0) continue;

            m_optimization_block->m_keyframes.push_back(*it);

            assert(!(*it)->m_added);
            (*it)->m_added = true;
        }
    }
    else
    {
        assert(!EF_frames.empty());
        for(size_t i=0;i<EF_frames.size();++i)
        {
            assert(!EF_frames[i]->isKeyframe());
            m_optimization_block->m_keyframes.push_back(EF_frames[i]);

            assert(!EF_frames[i]->m_added);
            EF_frames[i]->m_added = true;
        }
    }

    const size_t n_keyframe = m_optimization_block->m_keyframes.size();
    if(show_debug_prints)
        cout << "[extractOptimizationBlock]: Extract " << n_keyframe << " Keyframe." << endl;

    m_optimization_block->m_max_id = m_optimization_block->m_keyframes.back()->id_;
    m_optimization_block->m_min_id = m_optimization_block->m_keyframes.front()->id_;
    assert(m_optimization_block->m_max_id > m_optimization_block->m_min_id);

    for(size_t i = 0; i < n_keyframe; ++i)
    {
        FramePtr keyframe = m_optimization_block->m_keyframes[i];
        assert(keyframe->m_added);

        int n_features = 0;
        for(auto it = keyframe->fts_.begin(); it != keyframe->fts_.end(); ++it)
        {
            Feature* feature = *it;

            if(feature->m_non_point) continue;

            if(feature->m_added) continue;

            Feature* feature_iterator = feature;

            const int min_keyframes_valid = 1;
            int n_keyframe_valid = 0;
            while(feature_iterator != NULL && feature_iterator->frame->id_ <= m_optimization_block->m_max_id)
            {
                assert(!feature_iterator->m_non_point);
                assert(!std::isnan(feature_iterator->radiances[0]));

                if(!feature_iterator->frame->m_added)
                {
                    feature_iterator = feature_iterator->m_next_feature;
                    continue;
                }

                n_keyframe_valid++;
                feature_iterator = feature_iterator->m_next_feature;

                if(n_keyframe_valid > min_keyframes_valid)  break;
            }

            if(n_keyframe_valid <= min_keyframes_valid)
                continue;

            feature_iterator = feature;

            OptimizedPoint opt_p;

            opt_p.output_intensities.resize(n_keyframe);
            opt_p.xy_image_locations.resize(n_keyframe, Vector2d(-1,-1));
            opt_p.radii.resize(n_keyframe, -1);
            opt_p.grad_weights.resize(n_keyframe);

            std::vector<double> radiance_mesurement(feature_iterator->radiances.size(), 0);

            int nr_keyframes_valid = 0;
            int nr_feature_obs = 0;

            while(feature_iterator != NULL && feature_iterator->frame->id_ <= m_optimization_block->m_max_id)
            {
                assert(!feature_iterator->m_non_point);
                assert(!std::isnan(feature_iterator->radiances[0]));

                nr_feature_obs++;

                if(!feature_iterator->frame->m_added)
                {
                    assert(!feature_iterator->radiances.empty());
                    for(size_t i = 0; i < feature_iterator->radiances.size(); ++i)
                        radiance_mesurement[i] += feature_iterator->radiances[i];

                    feature_iterator = feature_iterator->m_next_feature;
                    continue;
                }

                assert(!feature_iterator->m_added);
                feature_iterator->m_added = true;
                m_optimization_block->m_features.push_back(feature_iterator);

                nr_keyframes_valid++;

                assert(!feature_iterator->radiances.empty());
                for(size_t i = 0; i < feature_iterator->radiances.size(); ++i)
                {
                    assert(!std::isnan(feature_iterator->radiances[i]));
                    radiance_mesurement[i] += feature_iterator->radiances[i];
                }

                int keyframe_id = findKeyFrameID(feature_iterator->frame);  

                opt_p.output_intensities[keyframe_id] = feature_iterator->outputs;

                opt_p.xy_image_locations[keyframe_id] = feature_iterator->px;

                double radius = getNormalizedRadius(feature_iterator->px);
                opt_p.radii[keyframe_id] = radius;

                for(size_t r = 0;r < feature_iterator->outputs_grad.size();r++)
                {
                    double grad_value = feature_iterator->outputs_grad[r];
                    double weight = 1.0 / (grad_value/255+1.0);
                    opt_p.grad_weights[keyframe_id].push_back(weight);
                }

                feature_iterator = feature_iterator->m_next_feature;
            }

            assert(nr_keyframes_valid > min_keyframes_valid);

            for(size_t i = 0; i < radiance_mesurement.size(); ++i)
                radiance_mesurement[i] /= (255*nr_feature_obs);

            opt_p.radiances_mesurement = radiance_mesurement;
            m_optimization_block->addOptimizationPoint(opt_p);

            n_features++;
        } 

        if(show_debug_prints)
            cout << "[extractOptimizationBlock]: Keyframe " << i << " add " << n_features << " Features" << endl;
    }

    for(size_t i = 0; i < m_optimization_block->m_keyframes.size(); ++i)
        m_optimization_block->m_keyframes[i]->m_added = false;

    for(size_t i = 0; i < m_optimization_block->m_features.size(); ++i)
        m_optimization_block->m_features[i]->m_added  = false;

    return true;
}

int PhotomatricCalibration::findKeyFrameID(Frame* frame)
{
    int keyframe_id = -1;
    bool have_found = false;
    for(size_t i = 0; i < m_optimization_block->m_keyframes.size(); ++i)
        if(frame->id_ == m_optimization_block->m_keyframes[i]->id_)
        {
            keyframe_id = i;
            have_found = true;
            break;
        }

    assert(have_found == true);
    return keyframe_id;
}

double PhotomatricCalibration::getNormalizedRadius(Vector2d uv_level0)
{
    double x = uv_level0[0];
    double y = uv_level0[1];
    
    assert(x<m_width && y<m_height);

    double x_norm = x - m_width_2;
    double y_norm = y - m_height_2;
    
    double radius = sqrt(x_norm*x_norm + y_norm*y_norm);
    radius /= m_max_radius;
    assert(radius < 1.000000001);

    return radius;
}

double PhotomatricCalibration::evfOptimization(bool show_debug_prints)
{
    int points_per_patch = (2*m_patch_size+1)*(2*m_patch_size+1);
    int num_residuals = m_optimization_block->getNrResiduals();
    int num_parameters = C_NR_RESPONSE_PARAMS + C_NR_VIGNETTE_PARAMS + m_optimization_block->getNrImages();
    cv::Mat Jacobian_full(num_residuals,num_parameters,CV_64F,0.0);
    cv::Mat Residual_full(num_residuals,1,CV_64F,0.0);
    cv::Mat Weights_Jacobian(num_residuals,num_parameters,CV_64F,0.0);

    if(show_debug_prints)
        cout << "[evfOptimization]: Initialization OK." << endl;

    float huberTH = 1.4826*getMedianError();
    float outlierTH = 8*huberTH;
    if(outlierTH < 8.0 ) outlierTH = 8.0;

    m_huberTH = huberTH;
    m_outlierTH = outlierTH;

    int residual_id = -1;
    double residual_sum = 0;
    const int n_keyframe = static_cast<int>(m_optimization_block->m_keyframes.size());

    for(size_t p = 0; p < m_optimization_block->m_optimized_points.size(); ++p)
        for(int k = 0; k < n_keyframe; ++k)
        {
            if(m_optimization_block->m_optimized_points[p].output_intensities[k].empty())
                continue;

            const double radius = m_optimization_block->m_optimized_points[p].radii.at(k);
            const double vignette_value = applyVignetting(radius);
            const double r2 = radius*radius;
            const double r4 = r2*r2;
            const double r6 = r4*r2;

            double vignette_value_zero=vignette_value;
            if(m_type == kSliding)
                vignette_value_zero = applyVignettingZero(radius);

            const double exposure = m_optimization_block->m_keyframes[k]->m_exposure_time;
            assert(radius > 0 && exposure > 0);

            for(int i = 0; i < points_per_patch; ++i)
            {
                double output = m_optimization_block->m_optimized_points[p].output_intensities[k].at(i);
                if(output > 250 || output < 5) continue;

                double radiance = m_optimization_block->m_optimized_points[p].radiances_mesurement.at(i);
                residual_id++;

                if(m_type == kActive)
                    jacobianRowFillIn(Jacobian_full, vignette_value, r2, r4, r6, exposure, radius, radiance, k, residual_id);
                else
                    JacobianZeroCV(Jacobian_full, vignette_value_zero, r2, r4, r6, exposure, radiance, k, residual_id); 
            
                double residual = getResidualValue(output, radiance, vignette_value, exposure);

                double huber_weight = fabsf(residual) < huberTH ? 1.0 : huberTH / fabsf(residual);
                if(fabsf(residual) > outlierTH) huber_weight /= 10;
                double grad_weight = m_optimization_block->m_optimized_points[p].grad_weights[k].at(i);
                double total_weight = grad_weight*huber_weight;

                Residual_full.at<double>(residual_id,0) = residual*total_weight;

                for(int p = 0;p < num_parameters;p++)
                    Weights_Jacobian.at<double>(residual_id,p) = total_weight;

                residual_sum += abs(residual)*total_weight;
            }
        }

    int real_number_of_residuals = residual_id;

    cv::Mat Jacobian_real = Jacobian_full(cv::Rect(0,0,num_parameters,real_number_of_residuals));
    cv::Mat Residual_real = Residual_full(cv::Rect(0,0,1,real_number_of_residuals));
    cv::Mat Weights_real  = Weights_Jacobian(cv::Rect(0,0,num_parameters,real_number_of_residuals));

    cv::Mat Jacobian_T;
    cv::transpose(Jacobian_real, Jacobian_T);

    cv::Mat A =  Jacobian_T * (Weights_real.mul(Jacobian_real));
    cv::Mat b = -Jacobian_T * Residual_real;


    Eigen::Matrix<double,7,1> bM_top, delta;
    delta[0] = m_c1 - m_c1_eval;
    delta[1] = m_c2 - m_c2_eval;
    delta[2] = m_c3 - m_c3_eval;
    delta[3] = m_c4 - m_c4_eval;
    delta[4] = m_v1 - m_v1_eval;
    delta[5] = m_v2 - m_v2_eval;
    delta[6] = m_v3 - m_v3_eval; 

    bM_top = (b_marg + H_marg * delta);

    for(int i=0;i<7;++i)
    {
        b.at<double>(i,0) += bM_top[i];

        for(int j=0;j<7;++j)
            A.at<double>(i,j) += H_marg(i,j);
    }


    double total_error, avg_error;
    double E_old = getTotalResidual(total_error, avg_error, huberTH, outlierTH);

    cv::Mat Identity = cv::Mat::eye(num_parameters, num_parameters, CV_64F);
    Identity = Identity.mul(A);

    const double response_parameters_backup[4] = {m_c1, m_c2, m_c3, m_c4};
    const double vignette_parameters_backup[3] = {m_v1, m_v2, m_v3};
    vector<double> exposure_time_backup;
    for(size_t k = 0; k < m_optimization_block->m_keyframes.size(); ++k)
        exposure_time_backup.push_back(m_optimization_block->m_keyframes[k]->m_exposure_time);

    cv::Mat best_state_update(num_parameters,1,CV_64F,0.0);

    double current_best_error = total_error;
    double current_best_energy = E_old;
    if(m_type == kSliding)
        current_best_energy += calcMEnergy();

    if(show_debug_prints)
        std::cout << "Energy before ECA adjustment: total: " << current_best_energy << std::endl;
    
    double lm_dampening = 1;
    const int max_rounds = 10;
    for(int round = 0; round < max_rounds; ++round)
    {
        if(show_debug_prints)
            std::cout << "ECA Optimization round with dampening = " << lm_dampening << std::endl;

        cv::Mat H = A + lm_dampening*Identity;
        cv::Mat state_update = H.inv(cv::DECOMP_SVD)*b;

        setResponse(response_parameters_backup[0] + state_update.at<double>(0,0),
                    response_parameters_backup[1] + state_update.at<double>(1,0),
                    response_parameters_backup[2] + state_update.at<double>(2,0),
                    response_parameters_backup[3] + state_update.at<double>(3,0));


        setVignette(vignette_parameters_backup[0] + state_update.at<double>(4,0),
                    vignette_parameters_backup[1] + state_update.at<double>(5,0),
                    vignette_parameters_backup[2] + state_update.at<double>(6,0));

        for(size_t k = 0; k < m_optimization_block->m_keyframes.size(); ++k)
            m_optimization_block->m_keyframes[k]->m_exposure_time = exposure_time_backup[k] + state_update.at<double>(7+k,0);

        double current_error;
        double E_new = getTotalResidual(current_error, avg_error, huberTH, outlierTH);

        double current_enery = E_new;
        if(m_type == kSliding)
            current_enery += calcMEnergy();

        if(show_debug_prints)
            std::cout << "Energy after ECA adjustment: total: " << current_enery << std::endl;
            
        if(current_enery < current_best_energy)
        {
            if(lm_dampening > 0.01) lm_dampening *= 0.5;
            
            current_best_energy = current_enery;
            best_state_update = state_update;
        }
        else
        {
            if(lm_dampening < 1000000)
            {
                lm_dampening *= 4;
                if(lm_dampening < 0.0001) lm_dampening = 0.0001;
            }
            else
            {
                if(show_debug_prints)
                    std::cout << "MAX DAMPING REACHED, BREAK EARLY " << std::endl;
                break;
            }
        }
    }
    
    setResponse(response_parameters_backup[0] + best_state_update.at<double>(0,0),
                response_parameters_backup[1] + best_state_update.at<double>(1,0),
                response_parameters_backup[2] + best_state_update.at<double>(2,0),
                response_parameters_backup[3] + best_state_update.at<double>(3,0));

    setVignette(vignette_parameters_backup[0] + best_state_update.at<double>(4,0),
                vignette_parameters_backup[1] + best_state_update.at<double>(5,0),
                vignette_parameters_backup[2] + best_state_update.at<double>(6,0));

    for(size_t k = 0; k < m_optimization_block->m_keyframes.size(); ++k)
        m_optimization_block->m_keyframes[k]->m_exposure_time = exposure_time_backup[k] + best_state_update.at<double>(7+k,0);

    if(show_debug_prints)
        std::cout << "Best update " << best_state_update << std::endl;
 
    double energy_after_optimization = current_best_energy;

    if(show_debug_prints)
        std::cout << "Energy after ECA adjustment: total: " << energy_after_optimization << std::endl;
        
    return avg_error;
}


void PhotomatricCalibration::jacobianRowFillIn(
    cv::Mat& Jacobian, double vignette_value, double r2, double r4, double r6,
    double exposure_time, double radius, double radiance, int id_in_keyframe, int id_in_Jacobian)
{
    const double eVL = exposure_time*vignette_value*radiance;
    double f0_derivative_value = evaluateGrossbergBaseFunction(0, true, eVL);
    double h1_derivative_value = evaluateGrossbergBaseFunction(1, true, eVL);
    double h2_derivative_value = evaluateGrossbergBaseFunction(2, true, eVL);
    double h3_derivative_value = evaluateGrossbergBaseFunction(3, true, eVL);
    double h4_derivative_value = evaluateGrossbergBaseFunction(4, true, eVL);
    double response_derivative_value = f0_derivative_value      + 
                                       m_c1*h1_derivative_value + 
                                       m_c2*h2_derivative_value + 
                                       m_c3*h3_derivative_value + 
                                       m_c4*h4_derivative_value;

    Jacobian.at<double>(id_in_Jacobian, 0) = 255*evaluateGrossbergBaseFunction(1, false, eVL);
    Jacobian.at<double>(id_in_Jacobian, 1) = 255*evaluateGrossbergBaseFunction(2, false, eVL);
    Jacobian.at<double>(id_in_Jacobian, 2) = 255*evaluateGrossbergBaseFunction(3, false, eVL);
    Jacobian.at<double>(id_in_Jacobian, 3) = 255*evaluateGrossbergBaseFunction(4, false, eVL);

    const double eL = exposure_time*radiance;
    Jacobian.at<double>(id_in_Jacobian, 4) = 255*response_derivative_value * eL * r2;
    Jacobian.at<double>(id_in_Jacobian, 5) = 255*response_derivative_value * eL * r4;
    Jacobian.at<double>(id_in_Jacobian, 6) = 255*response_derivative_value * eL * r6;

    Jacobian.at<double>(id_in_Jacobian, 7+id_in_keyframe) = 255*response_derivative_value * vignette_value * radiance;
}


void PhotomatricCalibration::radianceFullOptimization(bool show_debug_prints)
{
    if(show_debug_prints)
    {
        double total_error_before, avg_error_before;
        getTotalResidual(total_error_before, avg_error_before);
        cout << "Error before radiance adjustment: total: " << total_error_before << " avg: " << avg_error_before << endl;
    }
    
    const int n_keyframe = static_cast<int>(m_optimization_block->m_keyframes.size());
    const double huberTH = m_huberTH;
    const double outlierTH = m_outlierTH;

    for(size_t p = 0; p < m_optimization_block->m_optimized_points.size(); ++p)
    {
        OptimizedPoint op = m_optimization_block->m_optimized_points[p];
        for(size_t r = 0; r < op.radiances_mesurement.size(); ++r) 
        {
            const double radiance_mesurement = op.radiances_mesurement[r];

            double H=0,b=0;
            double init_residual = getTotalPointResidual(op, r);

            for(int k = 0; k < n_keyframe; ++k)
            {
                if(op.output_intensities[k].empty()) continue;

                double output = op.output_intensities[k].at(r);

                double exposure_time = m_optimization_block->m_keyframes[k]->m_exposure_time;
                double radius = op.radii[k];
                double vignette_value = applyVignetting(radius);
                assert(radius > 0 && exposure_time > 0);

                double residual = getResidualValue(output, radiance_mesurement, vignette_value, exposure_time);

                double huber_weight = fabsf(residual) < huberTH ? 1.0 : huberTH / fabsf(residual);
                if(fabsf(residual) > outlierTH) huber_weight /= 10;
                
                const double eVL = exposure_time*vignette_value*radiance_mesurement;
                double f0_derivative_value = evaluateGrossbergBaseFunction(0, true, eVL);
                double h1_derivative_value = evaluateGrossbergBaseFunction(1, true, eVL);
                double h2_derivative_value = evaluateGrossbergBaseFunction(2, true, eVL);
                double h3_derivative_value = evaluateGrossbergBaseFunction(3, true, eVL);
                double h4_derivative_value = evaluateGrossbergBaseFunction(4, true, eVL);
                double response_derivative_value = f0_derivative_value + 
                                                   m_c1*h1_derivative_value + 
                                                   m_c2*h2_derivative_value + 
                                                   m_c3*h3_derivative_value + 
                                                   m_c4*h4_derivative_value;

                double Jacobian_rad = 255*response_derivative_value*exposure_time*vignette_value;
                Jacobian_rad *= 255;
                

                H += Jacobian_rad*Jacobian_rad*huber_weight;
                b -= Jacobian_rad*residual*huber_weight;
            }

            double lambda = 1;
            double new_error = init_residual+1;
            int max_iterations = 5;
            int curr_iteration = 0;

            const double step = b/H;

            while(new_error > init_residual)
            {
                if(curr_iteration == max_iterations)
                    lambda = 0;

                m_optimization_block->m_optimized_points[p].radiances_mesurement[r] = radiance_mesurement + lambda * step;

                if(op.radiances_mesurement[r] < 0.001)
                    op.radiances_mesurement[r] = 0.001;

                if(op.radiances_mesurement[r] > 0.999)
                    op.radiances_mesurement[r] = 0.999;
                    
                lambda /= 2; 
                curr_iteration++;
                    
                if(lambda < 0.001) break;

                new_error = getTotalPointResidual(op, r);
            }
        }
    }

    if(show_debug_prints)
    {
        double total_error,avg_error;
        getTotalResidual(total_error,avg_error);
        std::cout << "error after Radiance adjustment: total: " << total_error << " avg: " << avg_error << std::endl;
    }
}

double PhotomatricCalibration::getTotalResidual(double& total_error, double& avg_error, float huber_th, float outlier_th)
{
    double residual_sum = 0;
    double Energy = 0;
    int residual_id = 0;
    int points_per_patch = (2*m_patch_size+1)*(2*m_patch_size+1);

    const int n_keyframe = static_cast<int>(m_optimization_block->m_keyframes.size());
    const double huberTH = huber_th;
    const double outlierTH = outlier_th;

    std::vector<float> errors;

    for(size_t p = 0; p < m_optimization_block->m_optimized_points.size(); ++p)
    {
        assert(m_optimization_block->m_keyframes.size() == m_optimization_block->m_optimized_points[p].output_intensities.size());
        for(int k = 0; k < n_keyframe; ++k)
        {
            if(m_optimization_block->m_optimized_points[p].output_intensities[k].empty())
                continue;

            const double radius = m_optimization_block->m_optimized_points[p].radii.at(k);
            const double vignette = applyVignetting(radius);
            const double exposure = m_optimization_block->m_keyframes[k]->m_exposure_time;
           
            if(radius < 0 || exposure < 0)
            {
                cout << "r = " << radius << "\t" << "e = " << exposure << endl;
                assert(false);
            }

            for(int i = 0; i < points_per_patch; ++i)
            {
                double output = m_optimization_block->m_optimized_points[p].output_intensities[k].at(i);
                if(output > 250 || output < 5) continue;

                double radiance = m_optimization_block->m_optimized_points[p].radiances_mesurement.at(i);

                double residual = getResidualValue(output, radiance, vignette, exposure);

                errors.push_back(fabsf(residual));

                double grad_weight = m_optimization_block->m_optimized_points[p].grad_weights[k].at(i);
                double huber_weight = fabsf(residual) < huberTH ? 1.0 : huberTH / fabsf(residual);
                if(fabsf(residual) > outlierTH) huber_weight /= 10;
                double total_weight = grad_weight*huber_weight;


                residual_sum += total_weight*abs(residual);
                Energy += residual*residual*grad_weight*grad_weight*huber_weight*(2-huber_weight);
                residual_id++;
            }
        }
    }

    total_error = residual_sum;
    avg_error   = total_error/residual_id;

    return Energy;
}

float PhotomatricCalibration::getMedianError()
{
    const int points_per_patch = (2*m_patch_size+1)*(2*m_patch_size+1);
    const int n_keyframe = static_cast<int>(m_optimization_block->m_keyframes.size());

    std::vector<float> errors;
    for(size_t p = 0; p < m_optimization_block->m_optimized_points.size(); ++p)
        for(int k = 0; k < n_keyframe; ++k)
        {
            if(m_optimization_block->m_optimized_points[p].output_intensities[k].empty())
                continue;

            const double radius = m_optimization_block->m_optimized_points[p].radii.at(k);
            const double vignette = applyVignetting(radius);
            const double exposure = m_optimization_block->m_keyframes[k]->m_exposure_time;
            if(radius < 0 || exposure < 0)
            {
                cout << "r = " << radius << "\t" << "e = " << exposure << endl;
                assert(false);
            }

            for(int i = 0; i < points_per_patch; ++i)
            {
                double output = m_optimization_block->m_optimized_points[p].output_intensities[k].at(i);
                if(output > 250 || output < 5) continue;

                double radiance = m_optimization_block->m_optimized_points[p].radiances_mesurement.at(i);

                double residual = getResidualValue(output, radiance, vignette, exposure);

                errors.push_back(fabsf(residual));
            }
        }

    return vilo::getMedian(errors);
}

double PhotomatricCalibration::getTotalPointResidual(OptimizedPoint& p, int patch_id)
{
    double radiance_guess = p.radiances_mesurement.at(patch_id);
    double n_keyframe = m_optimization_block->m_keyframes.size();

    const double huberTH = m_huberTH;
    const double outlierTH = m_outlierTH;

    double total_error = 0;
    for(int k = 0; k < n_keyframe; ++k)
    {
        if(p.output_intensities[k].empty())
            continue;

        double output = p.output_intensities[k].at(patch_id);
        double exposure_time = m_optimization_block->m_keyframes[k]->m_exposure_time;
        double radius = p.radii[k];
        double vignette = applyVignetting(radius);
        assert(radius > 0);

        double res = getResidualValue(output, radiance_guess, vignette, exposure_time);
    
        double huber_weight = fabsf(res) < huberTH ? 1.0 : huberTH / fabsf(res);
        if(fabsf(res) > outlierTH) huber_weight /= 10;

        total_error += abs(res)*huber_weight;
    }

    return total_error;
}

double PhotomatricCalibration::getResidualValue(double output, double radiance, double vignette, double exposure)
{
    double eVL = exposure*vignette*radiance;
    double response_value;

    if(eVL < 0)
        response_value = 0;
    else if(eVL > 1)
        response_value = 255;
    else
        response_value = applyResponse(eVL);

    return response_value-output;
}

double PhotomatricCalibration::applyVignetting(double radius)
{
    std::unique_lock<mutex> lock(m_mutex_v);

    assert(radius < 1.000001);

    double r_2 = radius*radius;
    double r_4 = r_2*r_2;
    double r_6 = r_2*r_4;

    double result_vignette = 1.0 + m_v1*r_2 + m_v2*r_4 + m_v3*r_6;

    if(result_vignette < 0) result_vignette = 0;
    if(result_vignette > 1) result_vignette = 1;

    return result_vignette;
}

double PhotomatricCalibration::applyVignetting(Vector2d uv)
{
    return applyVignetting(getNormalizedRadius(uv));
}

double PhotomatricCalibration::applyResponse(double evl)
{
    std::unique_lock<mutex> lock(m_mutex_r);

    if(evl > 1.0000001 || evl < 0)
    {
        cout << "Assertion fail in applyResponse(...) the eVL is " << evl << endl;
        assert(1 == 2);
    }

    double f0 = evaluateGrossbergBaseFunction(0, false, evl);
    double h1 = evaluateGrossbergBaseFunction(1, false, evl);
    double h2 = evaluateGrossbergBaseFunction(2, false, evl);
    double h3 = evaluateGrossbergBaseFunction(3, false, evl);
    double h4 = evaluateGrossbergBaseFunction(4, false, evl);

    double response_value = f0 + m_c1*h1 + m_c2*h2 + m_c3*h3 + m_c4*h4;

    return 255*response_value;
}

double PhotomatricCalibration::evaluateGrossbergBaseFunction(int base_function_index, bool is_derivative, double x)
{
    if(x < 0) x = 0;
    if(x > 1) x = 1;

    int x_int     = std::round(x*1023);
    int x_der_int = std::round(x*1022);

    if(base_function_index == 0)
    {
        if(!is_derivative)
            return m_f_0[x_int];
        else
            return m_f_0_der[x_der_int];
    }
    
    if(base_function_index == 1)
    {
        if(!is_derivative)
            return m_h_1[x_int];
        else
            return m_h_1_der[x_der_int];
    }
    
    if(base_function_index == 2)
    {
        if(!is_derivative)
            return m_h_2[x_int];
        else
            return m_h_2_der[x_der_int];
    }
    
    if(base_function_index == 3)
    {
        if(!is_derivative)
            return m_h_3[x_int];
        else
            return m_h_3_der[x_der_int];
    }
    
    if(base_function_index == 4)
    {
        if(!is_derivative)
            return m_h_4[x_int];
        else
            return m_h_4_der[x_der_int];
    }
    
    throw std::runtime_error("Error, evaluateGrossbergBaseFunction(..)");
}

void PhotomatricCalibration::readResponse(double& c1, double& c2, double& c3, double& c4)
{
    std::unique_lock<mutex> lock(m_mutex_r);
    c1 = m_c1;
    c2 = m_c2;
    c3 = m_c3;
    c4 = m_c4;
}

void PhotomatricCalibration::readVignette(double& v1, double& v2, double& v3)
{
    std::unique_lock<mutex> lock(m_mutex_v);
    v1 = m_v1;
    v2 = m_v2;
    v3 = m_v3;
}

void PhotomatricCalibration::setResponse(double c1, double c2, double c3, double c4)
{
    std::unique_lock<mutex> lock(m_mutex_r);
    m_c1 = c1;
    m_c2 = c2;
    m_c3 = c3;
    m_c4 = c4;
}


void PhotomatricCalibration::setVignette(double v1, double v2, double v3)
{
    std::unique_lock<mutex> lock(m_mutex_v);
    m_v1 = v1;
    m_v2 = v2;
    m_v3 = v3;
}


void PhotomatricCalibration::setInverseResponseRaw()
{
    std::unique_lock<mutex> lock(m_mutex_inverse_response);

    m_inverse_response[0]   = 0;
    m_inverse_response[255] = 255;
    
    for(int i=1;i<255;i++)
    {
        bool inversion_found = false;        
        for(int s=0;s<255;s++)
        {
            double response_s1 = applyResponse(s/255.0f);
            double response_s2 = applyResponse((s+1)/255.0f);
            if(response_s1 <= i && response_s2 >= i)
            {
                m_inverse_response[i] = s+(i - response_s1) / (response_s2-response_s1);
                inversion_found = true;
                assert(m_inverse_response[i] > m_inverse_response[i-1]);
                break;
            }
        }
        
        if(!inversion_found)
            throw std::runtime_error("Error, no inversion found in getInverseResponse(..)");
    }
}

void PhotomatricCalibration::readInverseResponseRaw(double* inv)
{
    std::unique_lock<mutex> lock(m_mutex_inverse_response);
    std::memcpy(inv, m_inverse_response, 256*sizeof(double));
}

void PhotomatricCalibration::photometricallyCorrectImage(cv::Mat& corrected_frame)
{
    std::unique_lock<mutex> lock_response(m_mutex_inverse_response);
    std::unique_lock<mutex> lock_vignette(m_mutex_inverse_vignette);

    for(int r = 0;r < corrected_frame.rows;r++)
        for(int c = 0;c < corrected_frame.cols;c++)
        {
            int o_value = corrected_frame.at<uchar>(r,c);

            double radiance = o_value;
            
            radiance /= m_inverse_vignette_image.at<double>(r,c);

            if(radiance > 255) radiance = 255;
            if(radiance < 0)   radiance = 0;

            corrected_frame.at<uchar>(r,c) = (uchar)radiance;
        }
}

double PhotomatricCalibration::removeResponse(int output)
{
    return m_inverse_response[output];
}

void PhotomatricCalibration::setInverseVignette()
{
    std::unique_lock<mutex> lock(m_mutex_inverse_vignette);
    for(int r = 0; r < m_height; ++r)
        for(int c = 0; c < m_width; ++c)
            m_inverse_vignette_image.at<double>(r,c) = applyVignetting(Vector2d(c,r));
}

cv::Mat PhotomatricCalibration::readInverseVignette()
{
    std::unique_lock<mutex> lock(m_mutex_inverse_vignette);
    return m_inverse_vignette_image;
}

void PhotomatricCalibration::visualize()
{
    double inverse_response[256];
    readInverseResponseRaw(inverse_response);

    for(int i = 0;i < 256;i++)
        inverse_response[i] /= 255;

    double exponent = determineGammaFixResponseAt(inverse_response, 148, 0.3);
    for(int i = 0;i < 256;i++)
        inverse_response[i] = 255*std::pow(inverse_response[i], exponent);

    double response_function[256];
    response_function[0] = 0.0;
    response_function[255] = 255.0;

    for(int i=1;i<255;i++)
    {
        bool response_found = false;   
        for(int s=0;s<255;s++)
            if(inverse_response[s] <= i && inverse_response[s+1] >= i)
            {
                response_function[i] = s+(i - inverse_response[s]) / (inverse_response[s+1]-inverse_response[s]);
                assert(response_function[i] > response_function[i-1]);
                response_found = true;
                break;
            }
        if(!response_found)
            throw std::runtime_error("Error, no response found in visualize(..)");
    }        

    int bgc = 0;
    if(m_type == kSliding) bgc = 50;
    cv::Mat response_vis_image(256,256,CV_8UC3,cv::Scalar(bgc,bgc,bgc));
    double response_error = 0;
    for(int i = 0;i < 256;i++)
    {
        int response_value = static_cast<int>(round(response_function[i]));
        int inv_response_value = static_cast<int>(round(inverse_response[i]));

        if(response_value < 0)
            response_value = 0;
        if(response_value > 255)
            response_value = 255;
        if(inv_response_value < 0)
            inv_response_value = 0;
        if(inv_response_value > 255)
            inv_response_value = 255;

        response_vis_image.at<cv::Vec3b>(255-response_value,i)[0] = 0;
        response_vis_image.at<cv::Vec3b>(255-response_value,i)[1] = 0;
        response_vis_image.at<cv::Vec3b>(255-response_value,i)[2] = 255;    
        cv::circle(response_vis_image, cv::Point2i(i,255-response_value), 1, cv::Scalar (0,0,255), -1); 

        double x = i/255.0;
        double m;
        double t;
        
        if(x < 0.1)
        {
            m = (90/255.0)/0.1;
            t = 0.0f;
        }
        else if(x < 0.48)
        {
            m = (110.0/255)/0.38;
            t = (200.0/255.0) - m*0.48;
        }
        else
        {
            m = (55.0/255)/0.52;
            t = 1 - m*1;
        }
        double dso_value_f = m*x + t;
        int dso_value = static_cast<int>(dso_value_f*255);
        if(dso_value > 255)
            dso_value = 255;

        response_vis_image.at<cv::Vec3b>(255-dso_value,i)[0] = 255;
        response_vis_image.at<cv::Vec3b>(255-dso_value,i)[1] = 255;
        response_vis_image.at<cv::Vec3b>(255-dso_value,i)[2] = 0;
        cv::circle(response_vis_image, cv::Point2i(i,255-dso_value), 1, cv::Scalar (255,255,0), -1);

        response_error += fabs(inv_response_value-dso_value);
    }
    cv::imshow("Estimated Response", response_vis_image);
    cv::moveWindow("Estimated Response", 20,20);

    cv::Mat vignette_vis_image(256,256,CV_8UC3,cv::Scalar(bgc,bgc,bgc));
    for(int i = 0;i < 256;i++)
    {
        double r = i/255.0f;
        
        double r_2 = r*r;
        double r_4 = r_2 * r_2;
        double r_6 = r_4 * r_2;
        
        double vignette = 1 + m_v1*r_2 + m_v2*r_4 + m_v3*r_6;
        vignette = pow(vignette,exponent);


        int y_pos = 245 - std::round(235*vignette);
        if(y_pos < 0)
            y_pos = 0;
        if(y_pos > 255)
            y_pos = 255;
        
        cv::circle(vignette_vis_image, cv::Point2i(i,y_pos), 1, cv::Scalar (0,0,255), -1);
        
        double dso_vignette_47 = 0.971 + 0.1891*r - 1.5958*r_2 + 1.4473*r_2*r - 0.5143* r_4;
        y_pos = 245 - round(235*dso_vignette_47  );
        cv::circle(vignette_vis_image, cv::Point2i(i,y_pos), 1, cv::Scalar (255,255,0), -1);
 
    }
    cv::imshow("Estimated Vignetting", vignette_vis_image);
    cv::moveWindow("Estimated Vignetting", 20,20+50+256);

}

double PhotomatricCalibration::determineGammaFixResponseAt(double*inverse_response,int x,double y)
{
    double v_y = inverse_response[x];
    double gamma = log(y) / log(v_y);
    return gamma;
}

void PhotomatricCalibration::responseInitialize(const cv::Mat& image)
{
    typedef Eigen::Matrix<double,4,1> Vector4d;

    Eigen::Matrix<double,4,4> H; H.setZero();
    Vector4d b; b.setZero();

    double energy_old=0;
    for(int i=0; i<1024; ++i)
    {
        double eVL = i/1023.0;
        double diag = 255*eVL;
        double response_value = applyResponse(eVL);
        double residual = response_value-diag;

        Vector4d J;
        J[0] = 255*evaluateGrossbergBaseFunction(1,false,eVL);
        J[1] = 255*evaluateGrossbergBaseFunction(2,false,eVL);
        J[2] = 255*evaluateGrossbergBaseFunction(3,false,eVL);
        J[3] = 255*evaluateGrossbergBaseFunction(4,false,eVL);

        H.noalias() += J*J.transpose();
        b.noalias() -= J*residual;

        energy_old += abs(residual);
    }

    Vector4d step = H.inverse()*b;
    m_c1 += step[0];
    m_c2 += step[1];
    m_c3 += step[2];
    m_c4 += step[3];

    double energy_new=0;
    for(int i=0; i<1024; ++i)
    {
        double eVL = i/1023.0;
        double diag = 255*eVL;
        double response_value = applyResponse(eVL);
        double residual = response_value-diag;
        energy_new += abs(residual);
    }

    if(energy_new > energy_old)
    {
        m_c1 -= step[0];
        m_c2 -= step[1];
        m_c3 -= step[2];
        m_c4 -= step[3];
    }




    cout << "New Response Initialize: " << m_c1 << "\t" << m_c2 << "\t" << m_c3 << "\t" << m_c4 << "\t" << endl;
}



/******************************  windowed  ******************************/
void PhotomatricCalibration::modeConversion()
{
    EF_frames.clear();

    size_t n_keyframe = m_optimization_block->m_keyframes.size();
    assert(n_keyframe > 0);

    for(size_t i=0;i<n_keyframe;++i)
    {
        EF_frames.push_back(m_optimization_block->m_keyframes[i]);
        if(i > 0) assert(EF_frames[i]->id_ > EF_frames[i-1]->id_);

        EF_frames[i]->m_kf_pc = true;
    }

    m_v1_eval = m_v1;
    m_v2_eval = m_v2;
    m_v3_eval = m_v3;

    m_c1_eval = m_c1;
    m_c2_eval = m_c2;
    m_c3_eval = m_c3;
    m_c4_eval = m_c4;

    m_type = kSliding;

    marginalizeFrame();
}

void PhotomatricCalibration::marginalizeFrame()
{
    FramePtr frame = EF_frames[0];
    assert(frame == m_optimization_block->m_keyframes[0]);
    assert(frame->m_kf_pc == true);

    const double exposure = frame->m_exposure_time;

    int points_per_patch = (2*m_patch_size+1)*(2*m_patch_size+1);

    Eigen::Matrix<double,8,8> H; H.setZero();
    Eigen::Matrix<double,8,1> b; b.setZero();
    for(size_t p = 0; p < m_optimization_block->m_optimized_points.size(); ++p)
    {
        if(m_optimization_block->m_optimized_points[p].output_intensities[0].empty())
            continue;

        const double radius = m_optimization_block->m_optimized_points[p].radii.at(0);
        const double vignette_value_zero = applyVignettingZero(radius);
        const double r2 = radius*radius;
        const double r4 = r2*r2;
        const double r6 = r4*r2;

        for(int i = 0; i < points_per_patch; ++i)
        {
            double output = m_optimization_block->m_optimized_points[p].output_intensities[0].at(i);
            if(output > 250 || output < 5) continue;

            double radiance = m_optimization_block->m_optimized_points[p].radiances_mesurement.at(i);

            Eigen::Matrix<double,1,8> J; J.setZero();
            JacobianZero(J, vignette_value_zero, r2, r4, r6, exposure, radiance);

            double res_toZero = getResidualZero(output, radiance, vignette_value_zero, exposure);

            double huber_weight = fabsf(res_toZero) < m_huberTH ? 1.0 : m_huberTH / fabsf(res_toZero);
            if(fabsf(res_toZero) > m_outlierTH) huber_weight /= 10;
            double grad_weight = m_optimization_block->m_optimized_points[p].grad_weights[0].at(i);
            double total_weight = grad_weight*huber_weight;

            H += J.transpose()*J*total_weight;
            b -= J.transpose()*res_toZero*total_weight;
        }
    }

    Eigen::Matrix<double,7,1> H_pe; H_pe.setZero();
    H_pe << H(0,7),H(1,7),H(2,7),H(3,7),H(4,7),H(5,7),H(6,7);

    double H_ee_inv = 1./H(7,7);

    Eigen::Matrix<double,7,7> H_sc = H_pe*H_ee_inv*H_pe.transpose();
    Eigen::Matrix<double,7,1> b_sc = H_pe*H_ee_inv*b(7,0);

    Eigen::Matrix<double,7,7> HM = H.block<7,7>(0,0) - H_sc;
    Eigen::Matrix<double,7,1> bM = b.head<7>() - b_sc;

    H_marg += 0.25*HM;
    b_marg += 0.25*bM;

    frame->m_kf_pc = false;

    assert(frame->id_ == (*EF_frames.begin())->id_);
    EF_frames.erase(EF_frames.begin());
}

double PhotomatricCalibration::applyVignettingZero(double radius)
{
    assert(radius < 1.000001);

    double r_2 = radius*radius;
    double r_4 = r_2*r_2;
    double r_6 = r_2*r_4;

    double result_vignette = 1.0 + m_v1_eval*r_2 + m_v2_eval*r_4 + m_v3_eval*r_6;

    if(result_vignette < 0) result_vignette = 0;
    if(result_vignette > 1) result_vignette = 1;

    return result_vignette;
}

double PhotomatricCalibration::applyResponseZero(double evl)
{
    double f0 = evaluateGrossbergBaseFunction(0, false, evl);
    double h1 = evaluateGrossbergBaseFunction(1, false, evl);
    double h2 = evaluateGrossbergBaseFunction(2, false, evl);
    double h3 = evaluateGrossbergBaseFunction(3, false, evl);
    double h4 = evaluateGrossbergBaseFunction(4, false, evl);

    double response_value = f0 + m_c1_eval*h1 + m_c2_eval*h2 + m_c3_eval*h3 + m_c4_eval*h4;

    return 255*response_value;
}

void PhotomatricCalibration::JacobianZero(Eigen::Matrix<double,1,8>& Jacobian, 
                                       double vignette_value_zero, 
                                       double r2, 
                                       double r4, 
                                       double r6, 
                                       double exposure_time, 
                                       double radiance)
{
    const double eVL = exposure_time*vignette_value_zero*radiance;
    double f0_derivative_value = evaluateGrossbergBaseFunction(0, true, eVL);
    double h1_derivative_value = evaluateGrossbergBaseFunction(1, true, eVL);
    double h2_derivative_value = evaluateGrossbergBaseFunction(2, true, eVL);
    double h3_derivative_value = evaluateGrossbergBaseFunction(3, true, eVL);
    double h4_derivative_value = evaluateGrossbergBaseFunction(4, true, eVL);
    double response_derivative_value_zero = f0_derivative_value      + 
                                       m_c1_eval*h1_derivative_value + 
                                       m_c2_eval*h2_derivative_value + 
                                       m_c3_eval*h3_derivative_value + 
                                       m_c4_eval*h4_derivative_value;

    Jacobian(0, 0) = 255*evaluateGrossbergBaseFunction(1, false, eVL);
    Jacobian(0, 1) = 255*evaluateGrossbergBaseFunction(2, false, eVL);
    Jacobian(0, 2) = 255*evaluateGrossbergBaseFunction(3, false, eVL);
    Jacobian(0, 3) = 255*evaluateGrossbergBaseFunction(4, false, eVL);

    const double eL = exposure_time*radiance;
    Jacobian(0, 4) = 255*response_derivative_value_zero * eL * r2;
    Jacobian(0, 5) = 255*response_derivative_value_zero * eL * r4;
    Jacobian(0, 6) = 255*response_derivative_value_zero * eL * r6;

    Jacobian(0, 7) = 255*response_derivative_value_zero * vignette_value_zero * radiance;
}

void PhotomatricCalibration::JacobianZeroCV(cv::Mat& Jacobian, 
                                             double vignette_value_zero, 
                                             double r2, 
                                             double r4, 
                                             double r6, 
                                             double exposure_time, 
                                             double radiance,
                                             int id_in_keyframe, 
                                             int id_in_Jacobian)
{
    const double eVL = exposure_time*vignette_value_zero*radiance;
    double f0_derivative_value = evaluateGrossbergBaseFunction(0, true, eVL);
    double h1_derivative_value = evaluateGrossbergBaseFunction(1, true, eVL);
    double h2_derivative_value = evaluateGrossbergBaseFunction(2, true, eVL);
    double h3_derivative_value = evaluateGrossbergBaseFunction(3, true, eVL);
    double h4_derivative_value = evaluateGrossbergBaseFunction(4, true, eVL);
    double response_derivative_value_zero = f0_derivative_value      + 
                                       m_c1_eval*h1_derivative_value + 
                                       m_c2_eval*h2_derivative_value + 
                                       m_c3_eval*h3_derivative_value + 
                                       m_c4_eval*h4_derivative_value;

    Jacobian.at<double>(id_in_Jacobian, 0) = 255*evaluateGrossbergBaseFunction(1, false, eVL);
    Jacobian.at<double>(id_in_Jacobian, 1) = 255*evaluateGrossbergBaseFunction(2, false, eVL);
    Jacobian.at<double>(id_in_Jacobian, 2) = 255*evaluateGrossbergBaseFunction(3, false, eVL);
    Jacobian.at<double>(id_in_Jacobian, 3) = 255*evaluateGrossbergBaseFunction(4, false, eVL);

    const double eL = exposure_time*radiance;
    Jacobian.at<double>(id_in_Jacobian, 4) = 255*response_derivative_value_zero * eL * r2;
    Jacobian.at<double>(id_in_Jacobian, 5) = 255*response_derivative_value_zero * eL * r4;
    Jacobian.at<double>(id_in_Jacobian, 6) = 255*response_derivative_value_zero * eL * r6;

    Jacobian.at<double>(id_in_Jacobian, 7+id_in_keyframe) = 255*response_derivative_value_zero * vignette_value_zero * radiance;
}

double PhotomatricCalibration::getResidualZero(
    double output, double radiance, double vignette_zero, double exposure)
{
    double eVL = exposure*vignette_zero*radiance;
    double response_value;

    if(eVL < 0)
        response_value = 0;
    else if(eVL > 1)
        response_value = 255;
    else
        response_value = applyResponseZero(eVL);

    return response_value-output;
}

double PhotomatricCalibration::calcMEnergy()
{
    Eigen::Matrix<double,7,1> v_delta;
    v_delta(0,0) = m_c1 - m_c1_eval;
    v_delta(1,0) = m_c2 - m_c2_eval;
    v_delta(2,0) = m_c3 - m_c3_eval;
    v_delta(3,0) = m_c4 - m_c4_eval;

    v_delta(4,0) = m_v1 - m_v1_eval;
    v_delta(5,0) = m_v2 - m_v2_eval;
    v_delta(6,0) = m_v3 - m_v3_eval;

    double energy_marg = v_delta.dot(2*b_marg + H_marg*v_delta);

    return energy_marg;
}


void PhotomatricCalibration::saveResult(std::string dir, int sequence_num)
{
    cout << "Saving Photomatric Calibration Result..." << endl;

    cv::Mat vignette_image  = cv::Mat(m_height, m_width, CV_8UC1);
    for(int r = 0; r < m_height; ++r)
        for(int c = 0; c < m_width; ++c)
            vignette_image.at<uchar>(r,c) = uchar(255.f * m_inverse_vignette_image.at<double>(r,c));
    cv::imwrite("vignette.png", vignette_image);

}
} //namespace vilo