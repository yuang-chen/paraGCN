#pragma once

#include <vector>
#include <utility>
#include "variable.h"
#include <cmath>
#include <cstdlib>

using namespace std;

struct AdamParams {
    float lr, beta1, beta2, eps, weight_decay;
    static AdamParams get_default();
};


struct AdamVariable
{
    vector<float> *data, *grad, m, v;
    bool decay;
public:
    int size();
    AdamVariable(Variable *, bool);
};

class Adam {
    AdamParams params;
    int step_count;
    vector<AdamVariable> adamVars;
public:
    Adam() {};
    Adam(vector<pair<Variable*, bool>> vars, AdamParams params);
    void step();
};
