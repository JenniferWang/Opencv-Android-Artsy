#include "ar/ORBTracker.hpp"
#include "NativeLogging.hpp"
#include <iostream>

using namespace cv;
using namespace std;

namespace ar
{

static const char* TAG = "ORBTracker";

    ORBTracker::ORBTracker() {
    }

    // Initialize your tracking using the provided image.
    void ORBTracker::initialize(const GrayscaleImage& frame) {
        
        int nfeatures = 5000;
        float scaleFactor = 1.2;
        int nlevels = 10;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        int scoreType = ORB::HARRIS_SCORE;
        int patchSize = 31;
        
        OrbFeatureDetector detector(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                    firstLevel, WTA_K, scoreType, patchSize);
        
        detector.detect(frame, ref_keyptrs_);
        OrbDescriptorExtractor extractor;
        extractor.compute(frame, ref_keyptrs_, ref_dscrptr_);
        
        frame.copyTo(ref_frame_);
        LOG_DEBUG(TAG, "Initialized.");
    }

    // Initialize your tracking subsystem using the provided frame.
    // Note that this function may be called multiple times to restart tracking.
    // Make sure you reset all necessary internal state!
    void ORBTracker::track(const GrayscaleImage& frame, MatchHandler& match_handler) {
        
        // Step 1: Detect ORB features.
        vector<KeyPoint> keyptrs;
        detector.detect(frame, keyptrs);
        
        OrbFeatureDetector extractor;
        Mat dscrptr;
        extractor.compute(frame, keyptrs, dscrptr);

        // Step 2: Find matches (using Hamming distance).
        BFMatcher matcher = BFMatcher(NORM_HAMMING);
        vector<vector< DMatch >> matches;
        matcher.knnMatch(dscrptr, ref_dscrptr_, matches, 2);
        
        // Step 3: Prune matches based on some heuristic (like the ratio test).
        vector< DMatch > good_matches;
        for (int i = 0; i < matches.size(); i ++) {
            float rejectRatio = 0.95;
            if (matches[i][0].distance / matches[i][1].distance > rejectRatio)
                continue;
            good_matches.push_back(matches[i][0]);
        }
        
        // Step 4: Call match_handler if sufficiently many correspondences were found.
        PointArray good_keyptrs_ref, good_keyptrs;
        for (int i = 0; i < good_matches.size(); i ++) {
            good_keyptrs_ref.push_back(ref_keyptrs_[good_matches[i].trainIdx].pt);
            good_keyptrs.push_back(keyptrs[good_matches[i].queryIdx].pt);
        }
        
        match_handler(good_keyptrs_ref, good_keyptrs);
    }
}