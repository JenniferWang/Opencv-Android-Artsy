#ifndef __Artsy__ConvNet__
#define __Artsy__ConvNet__

#include "ar/Common.hpp"

namespace ar
{
    // Our Convolutional Neural Network is powered by
    // DeepBeliefSDK from Jetpac (recently acquired by Google).
    // More info: https://github.com/jetpacapp/DeepBeliefSDK
    class ConvNet
    {
    public:
        
        ConvNet(const std::string& network_path);
        
        virtual ~ConvNet();
        
        ConvNetFeatures extract_features(const ColorImage& image);
        
        ConvNet(const ConvNet& other) = delete;
        
        ConvNet& operator=( const ConvNet&) = delete;
        
    private:
        
        void* net_;
    };
    
}

#endif /* defined(__Artsy__ConvNet__) */
