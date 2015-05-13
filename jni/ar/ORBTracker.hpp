#ifndef __Artsy__ORBTracker__
#define __Artsy__ORBTracker__

#include "ar/Common.hpp"

namespace ar
{

class ORBTracker
{
public:

    ORBTracker();

    virtual ~ORBTracker() = default;

    void initialize(const GrayscaleImage& frame);

    void track(const GrayscaleImage& frame, MatchHandler& match_handler);

private:
    GrayscaleImage ref_frame_;
    
    std::vector<cv::KeyPoint> ref_keyptrs_;
    
    cv::Mat ref_dscrptr_;
    cv::OrbFeatureDetector detector;
};

}

#endif /* defined(__Artsy__ORBTracker__) */
