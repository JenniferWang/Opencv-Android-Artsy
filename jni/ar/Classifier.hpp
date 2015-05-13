#ifndef __Artsy__Classifier__
#define __Artsy__Classifier__

#include "ar/Common.hpp"
#include "ar/ConvNet.hpp"
#include "ar/BinarySVM.hpp"

namespace ar {
    
    struct ArtInfo {
        std::string title;
        std::string artist;
    };
        
    class Classifier {
        
    public:
        
        Classifier(int confidence_threshold=0.1);
        
        virtual ~Classifier() = default;
        
        int classify(const ColorImage& image) const;
        
        ArtInfo get_art_info(int label) const;
        
    private:
        
        int confidence_threshold_;
        
        std::shared_ptr<ConvNet> conv_net_;
        
        std::vector<std::shared_ptr<BinarySVM>> svm_classifiers_;
        
        std::vector<ArtInfo> art_info_;
    };
    
}


#endif /* defined(__Artsy__Classifier__) */
