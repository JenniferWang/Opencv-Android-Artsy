#include "ar/ConvNet.hpp"
#include "DeepBelief/libjpcnn.h"
//#import <DeepBelief/DeepBelief.h>

namespace ar
{
    
class ConvNetInput
{
public:
    
    void* data;
    
    ConvNetInput(const ColorImage& img)
    {
        bool has_alpha = (img.type()==CV_8UC4);
        assert((img.type()==CV_8UC3) || has_alpha);
        const cv::Size size = img.size();
        const int width = size.width;
        const int height = size.height;
        cv::Mat rgbImage;
        cv::cvtColor(img, rgbImage, has_alpha?CV_BGRA2RGB:CV_BGR2RGB);
        uint8_t* rgbPixels = (uint8_t*)rgbImage.data;
        this->data = jpcnn_create_image_buffer_from_uint8_data(rgbPixels, width, height, 3, (3 * width), 0, 0);
    }
    
    ~ConvNetInput()
    {
        jpcnn_destroy_image_buffer(data);
    }
};
    
ConvNet::ConvNet(const std::string& network_path)
{
    net_ = jpcnn_create_network(network_path.c_str());
    assert(net_!=nullptr);
}
    
ConvNet::~ConvNet()
{
    jpcnn_destroy_network(net_);
}
    
ConvNetFeatures ConvNet::extract_features(const ColorImage& image)
{
    ConvNetFeatures features = {.features = nullptr, .length = 0};
    ConvNetInput input(image);
    char** predictionsLabels;
    int predictionsLabelsLength;
    jpcnn_classify_image(net_,
                         input.data,
                         0,
                         -2,
                         &(features.features),
                         &(features.length),
                         &predictionsLabels,
                         &predictionsLabelsLength);
    return features;
}
    
}
