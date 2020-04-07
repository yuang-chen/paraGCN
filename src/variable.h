#pragma once

#include <vector>
#include "rand.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <iostream>


using namespace std;

struct Variable
{
    vector<float> data, grad;
  //  vector<vector<float>> local_grad;

    Variable(int size, bool require_grad = true, bool thread_local_grad = false);
    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    float grad_norm(); 
    void print(int col=0x7fffffff);
};
