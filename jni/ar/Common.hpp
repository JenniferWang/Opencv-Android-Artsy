#ifndef Artsy_Common_hpp
#define Artsy_Common_hpp

#include <opencv2/opencv.hpp>
#include <cassert>
#include <functional>
#include <memory>

//#include "NativeLogging.hpp"

namespace ar
{

    typedef cv::Matx33f Homography;

    typedef cv::Mat ColorImage;
    
    typedef cv::Mat GrayscaleImage;
    
    typedef std::vector<cv::Point2f> PointArray;
    
    typedef std::function<void(const PointArray&, const PointArray&)> MatchHandler;
    
    struct ConvNetFeatures
    {
        float* features;
        int length;
    };
}

#endif
