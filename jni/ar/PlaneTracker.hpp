#ifndef __Artsy__PlaneTracker__
#define __Artsy__PlaneTracker__

#include "ar/Common.hpp"
#include "ar/KLTTracker.hpp"
#include "ar/ORBTracker.hpp"

namespace ar
{
    class PlaneTracker
    {
    public:
        
        PlaneTracker() = default;
        
        virtual ~PlaneTracker() = default;
        
        void initialize(const GrayscaleImage& frame);
        
        bool track(const GrayscaleImage& frame, Homography& H);
        
    private:        
        
        KLTTracker klt_tracker_;
        
        ORBTracker orb_tracker_;
        
        GrayscaleImage ref_frame_;
        
        Homography internal_H_;
        
        PointArray ransac_inliers_;
        
        bool is_lost_;
                
        bool estimate_homography(const PointArray& src_points, const PointArray& dst_points, Homography& H);
        
    };
}

#endif /* defined(__Artsy__PlaneTracker__) */
