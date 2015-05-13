#ifndef __Artsy__BinarySVM__
#define __Artsy__BinarySVM__

#include "ar/Common.hpp"

namespace ar
{
    
class BinarySVM
{
public:
    
    BinarySVM(const std::string& parameters_path);
    
    virtual ~BinarySVM();
    
    // Returns the probability of the given features belonging
    // to painting it was trained to recognize.
    float classify(const ConvNetFeatures& features) const;
    
    BinarySVM(const BinarySVM& other) = delete;
    
    BinarySVM& operator=( const BinarySVM&) = delete;
    
private:
    
    void* predictor_;
};
    
}

#endif /* defined(__Artsy__BinarySVM__) */
