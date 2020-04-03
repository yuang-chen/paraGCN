#include "variable.h"

Variable::Variable(int size, bool require_grad, bool thread_local_grad):
    data(size), grad(require_grad? size : 0) {}


void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f /(in_size + out_size));

    for(int i =0; i < data.size(); i++) {
        data[i] = i;
    }
}

void Variable::zero() {
    for(int i = 0; i < data.size(); i++) {
        data[i] = 0;
    }
}

void Variable::zero_grad() {
    for(int i = 0; i < grad.size(); i++) {
        grad[i] = 0;
    } 
}

float Variable::grad_norm() {
    float norm = 0;
    for(float x: grad) norm += x * x;
    return sqrtf(norm);
}



