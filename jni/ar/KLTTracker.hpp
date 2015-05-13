#ifndef __Artsy__KLTTracker__
#define __Artsy__KLTTracker__

#include "ar/Common.hpp"

namespace ar
{

class KLTTracker
{
    
public:
    
    KLTTracker(int max_points=800, cv::Size win_size=cv::Size(9, 9));
    
    virtual ~KLTTracker() = default;
    
    void initialize(const GrayscaleImage& frame, const PointArray& keypoints = PointArray());
    
    void track(const GrayscaleImage& frame, MatchHandler& match_handler);
    void debug_corner_detector();
    
private:
    
    int max_points_;
    
    PointArray pre_points_;
    
    GrayscaleImage pre_frame_;
    
    cv::Size win_size_;
    
    cv::TermCriteria term_crit_;
    
    void harris_corner_detector(const GrayscaleImage& image, cv::Mat& corners, int block_size, float k);
    
    void detect_features(const GrayscaleImage& image,
                         PointArray& corners,
                         double quality_level,
                         double min_distance,
                         int block_size);
    
    //void debug_corner_detector();
};

}

#endif /* defined(__Artsy__KLTTracker__) */
