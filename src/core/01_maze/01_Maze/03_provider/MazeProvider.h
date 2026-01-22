#pragma once
#include "Provider_uchar4_A.h"

class MazeProvider : public Provider_uchar4_A
{
public:
    virtual Grid grid() override;
    virtual Animable_I<uchar4>* createAnimable(const Grid& grid, bool isVerbose = false) override;
    virtual Image_I* createImageGL() override;
};
