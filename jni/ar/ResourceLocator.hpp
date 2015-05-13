#ifndef __Artsy__ResourceLocator__
#define __Artsy__ResourceLocator__

#include "ar/Common.hpp"

namespace ar
{

void SetResourceBasePath(std::string path);
    
std::string PathForResource(std::string name);

}

#endif /* defined(__Artsy__ResourceLocator__) */
