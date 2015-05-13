#include "ar/ARApp.hpp"
#include "NativeLogging.hpp"

namespace ar
{
    
    static const char* TAG = "ARApp";

    ARApp::ARApp() {
        has_been_initialized_ = false;
    }
        
    void ARApp::initialize(ColorImage &frame)
    {
        //Initialize tracking subsystem.
        convert_to_grayscale(frame, grayscaleFrame_);
        tracker_.initialize(grayscaleFrame_);
        
        //Classify artwork.
        classifier_ = Classifier(0.3);
        int class_label = classifier_.classify(frame);
        ArtInfo art_info = classifier_.get_art_info(class_label);
        
        //Initialize augmentor.
        PointArray pts;
        int inset = 5;
        cv::Size frame_size = frame.size();
        pts.push_back(cv::Point2f(inset, inset));
        pts.push_back(cv::Point2f(frame_size.width - 2*inset, inset));
        pts.push_back(cv::Point2f(frame_size.width - 2*inset, frame_size.height - 2*inset));
        pts.push_back(cv::Point2f(inset, frame_size.height - 2*inset));
        augmentor_.initialize(pts);
        augmentor_.set_title(art_info.title);
        
        has_been_initialized_ = true;
        LOG_DEBUG(TAG, "Initialized.");
    }

    void ARApp::process_frame(ColorImage& frame)
    {
        assert(has_been_initialized_ && "Must be initialized before calling process_frame!");
        assert(frame.type()==CV_8UC4 && "Frame type must be BGRA!");


        // Track target.
        convert_to_grayscale(frame, grayscaleFrame_);
        Homography H;
        bool tracking_successful = tracker_.track(grayscaleFrame_, H);
        
        // Augment frame.
        augmentor_.augment(frame, H, !tracking_successful);
    }

    void ARApp::convert_to_grayscale(const ColorImage& colorImage, GrayscaleImage& grayscaleImage)
    {
        cv::cvtColor(colorImage, grayscaleImage, CV_BGRA2GRAY);
    }

}