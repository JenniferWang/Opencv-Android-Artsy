#ifndef __Artsy__CornerDetection__
#define __Artsy__CornerDetection__

#include "ar/Common.hpp"

namespace ar
{

typedef std::function<void(const GrayscaleImage&, cv::Mat&, int)> HarrisCornerDetector;

void DetectGoodFeaturesToTrack(const GrayscaleImage& image,
                               PointArray& _corners,
                               int maxCorners,
                               double qualityLevel,
                               double minDistance,
                               int blockSize,
                               HarrisCornerDetector harrisCornerDetector);
}

#endif /* defined(__Artsy__CornerDetection__) */
