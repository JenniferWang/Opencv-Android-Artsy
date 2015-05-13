#include "ar/Classifier.hpp"
#include "ar/ResourceLocator.hpp"
#include "NativeLogging.hpp"
#include <sstream>
#include <string>

namespace ar {

    static const char* TAG = "Classifier";
    
    Classifier::Classifier(int confidence_threshold) : confidence_threshold_(confidence_threshold)
    {
        //Build art info db.
        art_info_.push_back({.title = "Convergence", .artist = "Jackson Pollock"});
        art_info_.push_back({.title = "Concetto Spaziale", .artist = "Lucio Fontana"});
        art_info_.push_back({.title = "Mona Lisa", .artist = "Leonardo da Vinci"});
        art_info_.push_back({.title = "Nighthawks", .artist = "Edward Hopper"});
        art_info_.push_back({.title = "The Scream", .artist = "Edvard Munch"});

        //Construct convolutional neural network.
        conv_net_ = std::make_shared<ConvNet>(PathForResource("jetpac.ntwk"));

        //Load binary SVMs
        size_t num_classes = art_info_.size();
        for(int i=0; i<num_classes; ++i) {
            std::ostringstream s;
            s << "svm-params-" << i << ".txt";
            std::string params_path = PathForResource(s.str());
            svm_classifiers_.push_back(std::make_shared<BinarySVM>(params_path));
        }
        LOG_DEBUG(TAG, "Initialized.");
    }
        
    ArtInfo Classifier::get_art_info(int label) const {
        if(label==-1) {
            return {.title = "Unknown", .artist="Unknown"};
        }
        assert(label<art_info_.size());
        return art_info_[label];
    }

    // Classify the image and return the class index (0<=index<5).
    int Classifier::classify(const ColorImage& image) const {
        // Use the conv_net_ object to extract features
        // using the convolutional neural network.
        ConvNetFeatures features = conv_net_->extract_features(image);
        
        // Step 2.
        // Use our bank of SVM classifiers to detect the class.
        // Pick the index with the highest probability.
        // Enforce a lower bound (confidence_threshold_) on the
        // probability: if none of the classifiers are
        // sufficiently confident, return -1.
        int res_idx = 0;
        float max_prob = 0, curr_prob;
        for (int i = 0; i < svm_classifiers_.size(); i++) {
            curr_prob = svm_classifiers_[i]->classify(features);
            LOG_DEBUG(TAG, "Probability for index %d is %f.", res_idx, curr_prob);
            if (curr_prob > max_prob) {
                res_idx = i;
                max_prob = curr_prob;
            }
        }
        if (max_prob > confidence_threshold_) {
            LOG_DEBUG(TAG, "Find index %d.", res_idx);
            return res_idx;
        }
        else {
            LOG_DEBUG(TAG, "Cannot classifiy.");
            return -1;
        }
    }

}