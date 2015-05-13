#ifndef __Artsy__Augmentor__
#define __Artsy__Augmentor__

#include "ar/Common.hpp"

namespace ar
{
    
class Augmentor
{
    
public:
    
    Augmentor();
    
    virtual ~Augmentor() = default;

    void initialize(const PointArray& target_bounds);
    
    void augment(ColorImage& frame, const Homography& H, bool lost);
    
    void set_title(const std::string& title);
    
private:
    
    PointArray reference_bounds_;
    
    PointArray current_bounds_;
    
    int font_face_;
    
    double font_scale_;
    
    std::string title_;
    
    void render_bounds(ColorImage& frame, const Homography& H);
    
    void render_title(ColorImage& frame);
    
    void render_text(ColorImage& frame, const std::string& text, const cv::Point2f& pt);
};
    
}

#endif /* defined(__Artsy__Augmentor__) */
