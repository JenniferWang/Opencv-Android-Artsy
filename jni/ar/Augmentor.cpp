#include "ar/Augmentor.hpp"

namespace ar
{
    
static const int TEXT_OUTER_THICKNESS = 10;

static const int TEXT_INNER_THICKNESS = 2;

Augmentor::Augmentor() : font_face_(CV_FONT_HERSHEY_SIMPLEX) , font_scale_(1.0) {
    
    
}

void Augmentor::initialize(const PointArray& target_bounds) {
    
    reference_bounds_ = target_bounds;
    current_bounds_ = target_bounds;
}
    
void Augmentor::set_title(const std::string& title) {
    
    title_ = title;
}

// Draw the current bounding box for the painting by transforming
// the reference points using the provided homography.
// The reference_bounds_ variable contains the locations of the four
// corners in the reference coordinate frame.
// Make sure you update the current_bounds_ variable!
void Augmentor::render_bounds(ColorImage& frame, const Homography& H) {
    perspectiveTransform(reference_bounds_, current_bounds_, H);
    cv::line(frame, current_bounds_[0], current_bounds_[1], cv::Scalar(0, 255, 0), 2);
    cv::line(frame, current_bounds_[1], current_bounds_[2], cv::Scalar(0, 255, 0), 2);
    cv::line(frame, current_bounds_[2], current_bounds_[3], cv::Scalar(0, 255, 0), 2);
    cv::line(frame, current_bounds_[3], current_bounds_[0], cv::Scalar(0, 255, 0), 2);
    
}

void Augmentor::render_title(ColorImage& frame) {
    
    if(current_bounds_.size()) {
        
        render_text(frame, title_, current_bounds_[0]);
    }
}
    
void Augmentor::augment(ColorImage &frame, const Homography& H, bool lost)
{
    if(!lost) {
        
        render_bounds(frame, H);
        render_title(frame);
    }
    else {
        
        std::string lost_msg = "LOST!";
        cv::Size text_size = cv::getTextSize(lost_msg, font_face_, font_scale_, TEXT_OUTER_THICKNESS, nullptr);
        cv::Point text_origin = cv::Point((frame.cols - text_size.width)/2, (frame.rows - text_size.height)/2);
        render_text(frame, lost_msg, text_origin);
    }
}

void Augmentor::render_text(ColorImage& frame, const std::string& text, const cv::Point2f& pt) {
    
    // A quick and dirty trick for rendering bordered text.
    // Feel free to replace this with a more sophisticated method.
    cv::putText(frame, text, pt, font_face_, font_scale_, cv::Scalar(0, 0, 0), TEXT_OUTER_THICKNESS);
    cv::putText(frame, text, pt, font_face_, font_scale_, cv::Scalar(255, 255, 255), TEXT_INNER_THICKNESS);
}

}