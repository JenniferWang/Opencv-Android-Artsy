#include "ar/BinarySVM.hpp"
#include "DeepBelief/libjpcnn.h"

namespace ar
{
    
BinarySVM::BinarySVM(const std::string& parameters_path)
{
    predictor_ = jpcnn_load_predictor(parameters_path.c_str());
    assert(predictor_ && "Failed to load SVM parameters!");
}
    
BinarySVM::~BinarySVM()
{
    jpcnn_destroy_predictor(predictor_);
}
    
float BinarySVM::classify(const ConvNetFeatures& features) const
{
    return jpcnn_predict(predictor_, features.features, features.length);
}
    
}
