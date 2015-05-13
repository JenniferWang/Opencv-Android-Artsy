#include "ar/KLTTracker.hpp"
#include "ar/CornerDetection.hpp"

using namespace cv;

namespace ar
{

KLTTracker::KLTTracker(int max_points, cv::Size win_size) {
    max_points_ = max_points;
    win_size_ = win_size;
    term_crit_ = cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01);
}

// Initialize your tracking subsystem using the provided frame.
// If the provided keypoints array is empty, detect them using a corner detector.
// Otherwise, use the provided points to initialize your tracker.
// Note that this function may be called multiple times to restart tracking.
// Make sure you reset all necessary internal state!
void KLTTracker::initialize(const GrayscaleImage& frame, const PointArray& keypoints) {
    
    pre_points_.clear();
    if(keypoints.empty()) {
        const bool use_my_harris_detector = false;
        
        double qualityLevel = 0.01; //0.01
        double minDistance = 5; //10
        int blockSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        
        if(!use_my_harris_detector) {
            goodFeaturesToTrack(frame,
                                pre_points_,
                                max_points_,
                                qualityLevel,
                                minDistance,
                                Mat(),
                                blockSize,
                                useHarrisDetector,
                                k );
            
        }
        else {
            detect_features(frame, pre_points_, qualityLevel, minDistance, blockSize);
        }
        
        // Optional: use cornerSubPix to refine your corners.
        Size winSize = Size(3, 3);
        Size zeroZone = Size(-1, -1);
        TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
        
        cornerSubPix( frame, pre_points_, winSize, zeroZone, criteria );
    }
    else {
        pre_points_ = keypoints;
    }
    frame.copyTo(pre_frame_);
    
}

void KLTTracker::track(const GrayscaleImage& frame, MatchHandler& match_handler) {
    
    int track_threshold = 50;
    int pyramidLevel = 5;
    
    PointArray tracked_points, pruned_tracked_ponits, pruned_prev_points;
    std::vector<unsigned char> status;
    std::vector<float> err;
    calcOpticalFlowPyrLK(pre_frame_, frame, pre_points_, tracked_points, status, err,
                         win_size_, pyramidLevel, term_crit_);
    
    if (sum(status)[0] > track_threshold) {
        for (int i = 0; i < status.size(); i++) {
            if (status[i]) {
                pruned_tracked_ponits.push_back(tracked_points[i]);
                pruned_prev_points.push_back(pre_points_[i]);
            }
        }
        
    }

    match_handler(pruned_prev_points, pruned_tracked_ponits);
    
    pre_points_ = pruned_tracked_ponits;
    
    frame.copyTo(pre_frame_);

}

void KLTTracker::harris_corner_detector(const GrayscaleImage& src, Mat& score_matrix, int block_size, float k) {

    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    int ksize = 3;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    Sobel( src, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    Sobel( src, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    vector<Mat> I_grad = {grad_x.mul(grad_x), grad_x.mul(grad_y), grad_y.mul(grad_y)};
    vector<Mat> A(3);
    
    Mat kernel = Mat::ones( block_size, block_size, CV_32F ) / (float)(block_size * block_size);
    
    for (int i = 0; i < 3; i++) {
        filter2D(I_grad[i], A[i], ddepth , kernel, Point( -1, -1 ), delta, BORDER_DEFAULT );
    }
    score_matrix = A[0].mul(A[2]) - A[1].mul(A[1])- k * (A[0] + A[2]).mul(A[0] + A[2]);
}

void KLTTracker::detect_features(const GrayscaleImage& image,
                                 PointArray& corners,
                                 double quality_level,
                                 double min_distance,
                                 int block_size) {
    HarrisCornerDetector harris = [&](const GrayscaleImage& hc_img, cv::Mat& hc_eig, int hc_block_size) {
        harris_corner_detector(hc_img, hc_eig, hc_block_size, 0.04);
    };
    DetectGoodFeaturesToTrack(image, corners, max_points_, quality_level, min_distance, block_size, harris);
}

// Debug corner detection algorithm.
void KLTTracker::debug_corner_detector() {
    
    // Create a checkerboard pattern.
    int pattern_size = 600;
    int block_size = 60;
    GrayscaleImage pattern(pattern_size, pattern_size, CV_8UC1);
    for(int i=0; i<pattern_size; ++i) {
        for(int j=0; j<pattern_size; ++j) {
            pattern.at<uint8_t>(i, j) = 255*(((i/block_size)%2)==((j/block_size)%2));
        }
    }
    
    PointArray corners;
    detect_features(pattern, corners, 0.01, 10, 3);
    
    ColorImage debug_image;
    cvtColor(pattern, debug_image, CV_GRAY2BGR);
    for (auto& pt: corners) {
        cv::circle(debug_image, pt, 3, cv::Scalar(0, 0, 255), 3);
    }
    
    cv::imshow("Detected Corners", debug_image);
    waitKey();
}
}