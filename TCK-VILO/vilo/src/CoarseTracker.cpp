#include "vilo/CoarseTracker.h"
#include "vilo/frame.h"
#include "vilo/feature.h"
#include "vilo/point.h"

#include "vilo/vikit/math_utils.h"

namespace vilo {

CoarseTracker::CoarseTracker(bool inverse_composition, int max_level, int min_level, int n_iter, bool verbose): 
	m_inverse_composition(inverse_composition), 
    m_max_level(max_level), 
    m_min_level(min_level), 
    m_n_iter(n_iter), 
    m_verbose(verbose),
    m_iter(0), 
    m_total_terms(0), 
    m_saturated_terms(0)
{}

CoarseTracker::~CoarseTracker()
{

}

size_t CoarseTracker::run(FramePtr ref_frame, FramePtr cur_frame)
{
	if(ref_frame->fts_.empty())
		return 0;

	m_ref_frame = ref_frame;
	m_cur_frame = cur_frame;

	m_exposure_rat = m_cur_frame->integralImage_ / m_ref_frame->integralImage_;
    m_b = 0;

    Sophus::SE3 Tfw_ref = m_ref_frame->getFramePose();
    Sophus::SE3 Tfw_cur = m_cur_frame->getFramePose();
    m_T_cur_ref = Tfw_cur * Tfw_ref.inverse();
    makeDepthRef();

	for(m_level = m_max_level; m_level >= m_min_level; --m_level)
	{
		std::fill(m_visible_fts.begin(), m_visible_fts.end(), false);

        m_offset_all    = m_max_level-m_level+m_pattern_offset;
        HALF_PATCH_SIZE = staticPatternPadding[m_offset_all];
        PATCH_AREA      = staticPatternNum[m_offset_all];

        m_ref_patch_cache = cv::Mat(m_ref_frame->fts_.size(), PATCH_AREA, CV_32F);
        m_visible_fts.resize(m_ref_frame->fts_.size(), false);
        m_jacobian_cache_true.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);
        m_jacobian_cache_raw.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);

        precomputeReferencePatches();

        selectRobustFunctionLevel(m_T_cur_ref, m_exposure_rat);

        const double cutoff_error = m_outlier_thresh;

		double energy_old = computeResiduals(m_T_cur_ref, m_exposure_rat, cutoff_error);

		Matrix7d H; Vector7d b;
        computeGS(H,b);

		float lambda = 0.1;
		for(m_iter=0; m_iter < m_n_iter; m_iter++)
		{
            
			Matrix7d Hl = H;
			for(int i=0;i<7;i++) Hl(i,i) *= (1+lambda);
			Vector7d step = Hl.ldlt().solve(b);

			float extrap_fac = 1;
			if(lambda < 0.001) extrap_fac = sqrt(sqrt(0.001 / lambda));
			step *= extrap_fac;

			if(!std::isfinite(step.sum()) || std::isnan(step[0])) step.setZero();

            float new_exposure_rat = m_exposure_rat + step[0];

			SE3 new_T_cur_ref;
			if(!m_inverse_composition)
				new_T_cur_ref = Sophus::SE3::exp(-step.segment<6>(1))*m_T_cur_ref;
			else
				new_T_cur_ref = m_T_cur_ref*Sophus::SE3::exp(-step.segment<6>(1));
            
			double energy_new = computeResiduals(new_T_cur_ref, new_exposure_rat, cutoff_error);
            
			if(energy_new < energy_old)
			{
				if(m_verbose)
		        {
		          cout << "It. " << m_iter
		               << "\t Success"
		               << "\t n_meas = " << m_total_terms
		               << "\t rejected = " << m_saturated_terms
		               << "\t new_chi2 = " << energy_new
		               << "\t exposure = " << new_exposure_rat
		               << "\t mu = " << lambda
		               << endl;
		        }
                
				computeGS(H,b);
                
				energy_old = energy_new;
				m_exposure_rat = new_exposure_rat;
				m_T_cur_ref = new_T_cur_ref;
                lambda *= 0.5;
                
			}
			else
			{
				if(m_verbose)
		        {
                    cout << "It. " << m_iter
                         << "\t Failure"
                         << "\t n_meas = " << m_total_terms
                         << "\t rejected = " << m_saturated_terms
                         << "\t new_chi2 = " << energy_new
                         << "\t exposure = " << new_exposure_rat
                         << "\t mu = " << lambda
                         << endl;
		        }

                lambda *= 4;
				if(lambda < 0.001) lambda = 0.001;
			}

            if(!(step.norm() > 1e-4))
            {
                if(m_verbose)
                    printf("inc too small, break!\n");
                break;
            }
            
		}
	}

    Tfw_cur = m_T_cur_ref * Tfw_ref;
    m_cur_frame->setFramePose(Tfw_cur);

    m_exposure_rat = 1;
    while(!m_ref_frame->m_exposure_finish && m_cur_frame->m_pc != NULL){cv::waitKey(1);}
    m_cur_frame->m_exposure_time = m_exposure_rat*m_ref_frame->m_exposure_time;

    if(m_exposure_rat > 0.99 && m_exposure_rat < 1.01) m_cur_frame->m_exposure_time = m_ref_frame->m_exposure_time;

    return float(m_total_terms) / PATCH_AREA;
}

void CoarseTracker::makeDepthRef()
{
    m_pt_ref.resize(m_ref_frame->fts_.size(), -1);

    size_t feature_counter = 0;
    Sophus::SE3 Tfw_ref = m_ref_frame->getFramePose();
    for(auto it_ft=m_ref_frame->fts_.begin(); it_ft!=m_ref_frame->fts_.end(); ++it_ft, ++feature_counter)
    {
        if((*it_ft)->point == NULL) 
            continue;

        double idist = (*it_ft)->point->getPointIdist();
        Vector3d p_host = (*it_ft)->point->hostFeature_->f * (1.0/idist);
        Sophus::SE3 Tfw_host = (*it_ft)->point->hostFeature_->frame->getFramePose();
        SE3 T_r_h = Tfw_ref * Tfw_host.inverse();
        Vector3d p_ref = T_r_h*p_host;
        if(p_ref[2] < 0.00001) continue;

        m_pt_ref[feature_counter] = p_ref.norm();
    }
}

double CoarseTracker::computeResiduals(const SE3& T_cur_ref, float exposure_rat, double cutoff_error, float b)
{
	if(m_inverse_composition)
		m_jacobian_cache_true = exposure_rat*m_jacobian_cache_raw;

	const cv::Mat& cur_img = m_cur_frame->img_pyr_.at(m_level);
	const int stride = cur_img.cols;
    const int border = HALF_PATCH_SIZE+1;
    const float scale = 1.0f/(1<<m_level);
    const double fxl = m_ref_frame->cam_->focal_length().x()*scale;
    const double fyl = m_ref_frame->cam_->focal_length().y()*scale;


    float setting_huberTH = m_huber_thresh;
    const float max_energy = 2*setting_huberTH*cutoff_error-setting_huberTH*setting_huberTH;
    const int pattern_offset = m_offset_all;

    m_buf_jacobian.clear();
    m_buf_weight.clear();
    m_buf_error.clear();
    m_total_terms = m_saturated_terms = 0;

    float E = 0;
    m_color_cur.clear(); m_color_ref.clear();

    size_t feature_counter = 0;
    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();
    for(auto it_ft=m_ref_frame->fts_.begin(); it_ft!=m_ref_frame->fts_.end(); ++it_ft, ++feature_counter, ++visiblity_it)
    {
    	if(!*visiblity_it) 
            continue;

        double dist = m_pt_ref[feature_counter]; if(dist < 0) continue;

        Vector3d xyz_ref((*it_ft)->f*dist);
        Vector3d xyz_cur(T_cur_ref * xyz_ref);

        if(xyz_cur[2] < 0) continue;

        Vector2f uv_cur_0(m_cur_frame->cam_->world2cam(xyz_cur).cast<float>());
        Vector2f uv_cur_pyr(uv_cur_0 * scale);
        float u_cur = uv_cur_pyr[0];
        float v_cur = uv_cur_pyr[1];
        int u_cur_i = floorf(u_cur);
        int v_cur_i = floorf(v_cur);

        if(u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
            continue;

        Matrix<double,2,6> frame_jac;
        if(!m_inverse_composition)
    		Frame::jacobian_xyz2uv(xyz_cur, frame_jac);

        float subpix_u_cur = u_cur-u_cur_i;
        float subpix_v_cur = v_cur-v_cur_i;
        float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        float w_cur_br = subpix_u_cur * subpix_v_cur;

        float* ref_patch_cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;
        size_t pixel_counter = 0;

        for(int n=0; n<PATCH_AREA; ++n, ++ref_patch_cache_ptr, ++pixel_counter)
    	{
            uint8_t* cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i + staticPattern[pattern_offset][n][1])*stride + u_cur_i + staticPattern[pattern_offset][n][0];

    		float cur_color = w_cur_tl*cur_img_ptr[0] 
    						+ w_cur_tr*cur_img_ptr[1] 
    						+ w_cur_bl*cur_img_ptr[stride] 
    						+ w_cur_br*cur_img_ptr[stride+1];
    		if(!std::isfinite(cur_color)) continue;

    		float residual = cur_color - (exposure_rat*(*ref_patch_cache_ptr) + b);


    		float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

    		if(fabs(residual) > cutoff_error && m_level < m_max_level) 
    		{
    			E += max_energy;
    			m_total_terms++;
    			m_saturated_terms++;
    		}
    		else
    		{ 
                if(m_level == m_max_level)
                    E += hw *residual*residual;
                else
                    E += hw *residual*residual*(2-hw);

				m_total_terms++;

				if(!m_inverse_composition)
				{
					float dx = 0.5f * ((w_cur_tl*cur_img_ptr[1]       + w_cur_tr*cur_img_ptr[2]        + w_cur_bl*cur_img_ptr[stride+1] + w_cur_br*cur_img_ptr[stride+2])
                                  	  -(w_cur_tl*cur_img_ptr[-1]      + w_cur_tr*cur_img_ptr[0]        + w_cur_bl*cur_img_ptr[stride-1] + w_cur_br*cur_img_ptr[stride]));
	        		float dy = 0.5f * ((w_cur_tl*cur_img_ptr[stride]  + w_cur_tr*cur_img_ptr[1+stride] + w_cur_bl*cur_img_ptr[stride*2] + w_cur_br*cur_img_ptr[stride*2+1])
	                                  -(w_cur_tl*cur_img_ptr[-stride] + w_cur_tr*cur_img_ptr[1-stride] + w_cur_bl*cur_img_ptr[0]        + w_cur_br*cur_img_ptr[1]));
        			Vector6d J_T = dx*frame_jac.row(0)*fxl + dy*frame_jac.row(1)*fyl;

                    double J_e = -(*ref_patch_cache_ptr);

                    Vector7d J; J[0] = J_e;
                    J.segment<6>(1) = J_T;

        			m_buf_jacobian.push_back(J); 
        			m_buf_weight.push_back(hw);
        			m_buf_error.push_back(residual);
				}
				else
				{
					Vector6d J_T(m_jacobian_cache_true.col(feature_counter*PATCH_AREA + pixel_counter));

                    double J_e = -(*ref_patch_cache_ptr);
                    Vector7d J; J[0] = J_e;
                    J.segment<6>(1) = J_T;

					m_buf_jacobian.push_back(J); 
        			m_buf_weight.push_back(hw);
        			m_buf_error.push_back(residual);
				}
    		}

            m_color_cur.push_back(cur_color);
            m_color_ref.push_back(*ref_patch_cache_ptr);
    	}
    }

    return E/m_total_terms;
}

void CoarseTracker::precomputeReferencePatches()
{
	const int border = HALF_PATCH_SIZE+1;
    const cv::Mat& ref_img = m_ref_frame->img_pyr_[m_level];
    const int stride = ref_img.cols;
    const float scale = 1.0f/(1<<m_level);

    const double fxl = m_ref_frame->cam_->focal_length().x()*scale;
    const double fyl = m_ref_frame->cam_->focal_length().y()*scale;
    const int pattern_offset = m_offset_all;

    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();
    size_t feature_counter = 0;
    for(auto ft_it = m_ref_frame->fts_.begin(); ft_it != m_ref_frame->fts_.end(); ++ft_it, ++visiblity_it, ++feature_counter)
    {
        if((*ft_it)->point == NULL)
            continue;

        float u_ref = (*ft_it)->px[0]*scale;
        float v_ref = (*ft_it)->px[1]*scale;
        int u_ref_i = floorf(u_ref);
        int v_ref_i = floorf(v_ref);
        if(u_ref_i-border < 0 || v_ref_i-border < 0 || u_ref_i+border >= ref_img.cols || v_ref_i+border >= ref_img.rows)
            continue;

        *visiblity_it = true;

        Matrix<double,2,6> frame_jac;
        if(m_inverse_composition)
        {
            double dist = m_pt_ref[feature_counter];
            if(dist < 0) continue;
            Vector3d xyz_ref((*ft_it)->f*dist);

        	Frame::jacobian_xyz2uv(xyz_ref, frame_jac);
        }

        float subpix_u_ref = u_ref-u_ref_i;
        float subpix_v_ref = v_ref-v_ref_i;
        float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
        float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
        float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
        float w_ref_br = 1.0-(w_ref_tl+w_ref_tr+w_ref_bl);

        size_t pixel_counter = 0;
        float* cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;
        for(int n=0; n<PATCH_AREA; ++n, ++cache_ptr, ++pixel_counter)
    	{
            uint8_t* ref_img_ptr = (uint8_t*)ref_img.data + (v_ref_i + staticPattern[pattern_offset][n][1])*stride + u_ref_i + staticPattern[pattern_offset][n][0];


    		*cache_ptr = w_ref_tl*ref_img_ptr[0] 
    				   + w_ref_tr*ref_img_ptr[1] 
    				   + w_ref_bl*ref_img_ptr[stride] 
    				   + w_ref_br*ref_img_ptr[stride+1];

    		if(m_inverse_composition)
    		{
    			float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1]       + w_ref_tr*ref_img_ptr[2]        + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                                  -(w_ref_tl*ref_img_ptr[-1]      + w_ref_tr*ref_img_ptr[0]        + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
            	float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride]  + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                              	  -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0]        + w_ref_br*ref_img_ptr[1]));
            	m_jacobian_cache_raw.col(feature_counter*PATCH_AREA + pixel_counter) = dx*frame_jac.row(0)*fxl + dy*frame_jac.row(1)*fyl;
    		}
    	}
    }
}

void CoarseTracker::computeGS(Matrix7d& H_out, Vector7d& b_out)
{
	assert(m_buf_jacobian.size() == m_buf_weight.size());

    m_acc7.initialize();

    b_out.setZero();
	for(size_t i=0; i<m_buf_jacobian.size(); ++i)
	{
        m_acc7.updateSingleWeighted(m_buf_jacobian[i][0],
                                    m_buf_jacobian[i][1],
                                    m_buf_jacobian[i][2],
                                    m_buf_jacobian[i][3],
                                    m_buf_jacobian[i][4],
                                    m_buf_jacobian[i][5],
                                    m_buf_jacobian[i][6],
                                    m_buf_weight[i], 0);

        b_out.noalias() -= m_buf_jacobian[i]*m_buf_error[i]*m_buf_weight[i];
	}

    m_acc7.finish();
    H_out = m_acc7.H.cast<double>();
}

void CoarseTracker::selectRobustFunctionLevel(const SE3& T_cur_ref, float exposure_rat, float b)
{
    const cv::Mat& cur_img = m_cur_frame->img_pyr_.at(m_level);
    const int stride = cur_img.cols;
    const int border = HALF_PATCH_SIZE+1;
    const float scale = 1.0f/(1<<m_level);
    const int pattern_offset = m_offset_all;

    std::vector<float> errors;
    size_t feature_counter = 0;
    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();

    for(auto it_ft=m_ref_frame->fts_.begin(); it_ft!=m_ref_frame->fts_.end(); ++it_ft, ++feature_counter, ++visiblity_it)
    {
        if(!*visiblity_it) 
            continue;

        double dist = m_pt_ref[feature_counter]; if(dist < 0) continue;
        
        Vector3d xyz_ref((*it_ft)->f*dist);
        Vector3d xyz_cur(T_cur_ref * xyz_ref);

        if(xyz_cur[2] < 0) continue;

        if(xyz_cur[2] < 0.01) continue;
        Vector2f uv_cur_pyr(m_cur_frame->cam_->world2cam(xyz_cur).cast<float>() * scale);
        
        float u_cur = uv_cur_pyr[0];
        float v_cur = uv_cur_pyr[1];
        int u_cur_i = floorf(u_cur);
        int v_cur_i = floorf(v_cur);

        if(u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
            continue;
        
        float subpix_u_cur = u_cur-u_cur_i;
        float subpix_v_cur = v_cur-v_cur_i;
        float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        float w_cur_br = subpix_u_cur * subpix_v_cur;

        float* ref_patch_cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;
        for(int n=0; n<PATCH_AREA; ++n, ++ref_patch_cache_ptr)
        {
            uint8_t* cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i + staticPattern[pattern_offset][n][1])*stride + u_cur_i + staticPattern[pattern_offset][n][0];

            float cur_color = w_cur_tl*cur_img_ptr[0] 
                            + w_cur_tr*cur_img_ptr[1] 
                            + w_cur_bl*cur_img_ptr[stride] 
                            + w_cur_br*cur_img_ptr[stride+1];
            float residual = cur_color - (exposure_rat*(*ref_patch_cache_ptr) + b);

            errors.push_back(fabsf(residual));
        }

    }
    
    if(errors.size() < 30)
    {
        m_huber_thresh = 5.2;
        m_outlier_thresh = 100;
        return;
    }

    float residual_median = vilo::getMedian(errors);
    vector<float> absolute_deviation;
    for(size_t i=0; i<errors.size(); ++i)
        absolute_deviation.push_back(fabs(errors[i]-residual_median));

    float standard_deviation = 1.4826*vilo::getMedian(absolute_deviation);


    m_huber_thresh = residual_median + standard_deviation;
    m_outlier_thresh = 3*m_huber_thresh;
    if(m_outlier_thresh < 10) m_outlier_thresh = 10;

    if(m_verbose)
    {
        printf("\nPYRAMID LEVEL %i\n---------------\n", m_level);
        cout << "Mid = "        << residual_median 
             << "\tStd = "      << standard_deviation 
             << "\tHuber = "    << m_huber_thresh 
             << "\tOutliers = " << m_outlier_thresh << endl;
    }
}

Vector2f CoarseTracker::lineFit(vector<float>& cur, vector<float>& ref, float a, float b)
{
    float sxx=0, syy=0, sxy=0, sx=0, sy=0, sw=0;
    for(size_t i = 0; i < cur.size(); ++i)
    {
        if(cur[i] < 5 || ref[i] < 5 || cur[i] > 250 || ref[i] > 250)
            continue;

        float res = cur[i] - a*ref[i] - b;

        const float cutoff_thresh = m_level == m_max_level? 80 : 25;
        const float weight_aff = fabsf(res) < cutoff_thresh? fabsf(res) < 8.0f? 1.0 : 8.0f / fabsf(res) : 0;  
        sxx += ref[i]*ref[i]*weight_aff; 
        syy += cur[i]*cur[i]*weight_aff;
        sx += ref[i]*weight_aff;
        sy += cur[i]*weight_aff;
        sw += weight_aff;
    }

    float aff_a = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));

    return Vector2f(aff_a, (sy - aff_a*sx)/sw);
}
} //namespace vilo