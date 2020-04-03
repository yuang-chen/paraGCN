#pragma once

#include <immintrin.h>
#include "sparse.h"


class Module {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module() {};
};


