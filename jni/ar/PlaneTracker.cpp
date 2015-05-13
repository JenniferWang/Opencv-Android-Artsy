#include "ar/PlaneTracker.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "NativeLogging.hpp"
#include <iostream>

namespace ar
{
static const char* TAG = "PlaneTracker";

void PlaneTracker::initialize(const GrayscaleImage &frame) {
    int max_points = 8000;
    cv::Size win_size = cv::Size(5, 5);
    
    klt_tracker_ = KLTTracker(max_points, win_size);
    klt_tracker_.initialize(frame);
    orb_tracker_.initialize(frame);
    is_lost_ = false;
    frame.copyTo(ref_frame_);
    internal_H_ = Homography(1, 0, 0,
                             0, 1, 0,
                             0, 0, 1);
}

// Given a set of correspondences, compute the homography using RANSAC.
// Compute the inlier ratio (number of RANSAC inliers/number of correspondences),
// and return true if it's above a certain threshold. Otherwise, return false.
// Use the test videos to determine a reasonable threshold.
bool PlaneTracker::estimate_homography(const PointArray& src_points, const PointArray& dst_points, Homography& H) {
    
    assert(src_points.size() == dst_points.size());
    
    float inlier_ratio = 0.6; //TODO //0.6
    int ransac_threshold = 8; // 2
    
    ransac_inliers_.clear();
    cv::Mat mask;
    H = cv::findHomography(src_points, dst_points, CV_RANSAC, ransac_threshold, mask);
    
    if (cv::sum(mask)[0]/src_points.size() < inlier_ratio) {
        return false;
    }
    
    for (int i = 0; i < src_points.size(); i++) {
        if (mask.at<uchar>(0, i)) {
            ransac_inliers_.push_back(src_points[i]);
        }
    }
    
    internal_H_ = H * internal_H_ ;
    H = internal_H_;
    return true;
}

// Implement the tracking logic as described on the project page.
// Your implementation should follow the logic described in the flowchart.
bool PlaneTracker::track(const GrayscaleImage &frame, Homography &H)
{
    bool estimation_successful = false;
    MatchHandler match_handler = [&](const PointArray& src_points, const PointArray& dst_points) {
        assert(src_points.size() == dst_points.size());
        if(dst_points.size()>=4) {
            estimation_successful = this->estimate_homography(src_points, dst_points, H);
        }
        else {

            LOG_ERROR(TAG, "Tracker provided less than 4 correspondences for homography estimation!");
        }
    };
    
    if (!is_lost_) {
        klt_tracker_.track(frame, match_handler);
    }
    
    if (!estimation_successful || is_lost_) {
        internal_H_ = Homography(1, 0, 0,
                                 0, 1, 0,
                                 0, 0, 1);
        orb_tracker_.track(frame, match_handler);
        if (estimation_successful) {
                klt_tracker_.initialize(frame, ransac_inliers_);
            is_lost_ = false;
        }
        else
            is_lost_ = true;
    }
    return estimation_successful;
}
} 