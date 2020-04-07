#pragma once

#include <immintrin.h>
#include "sparse.h"
#include "variable.h"

class Module {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module() {};
};

class Matmul: public Module {
    Variable *x, *y, *z;
    int m, n, l;

public:
    Matmul(Variable *x, Variable *y, Variable *z, int m, int n, int l);
    ~Matmul() {}
    void forward(bool);
    void backward();
};



class SparseMatmul: public Module {
    Variable *x, *y, *z;
    int m, n, l;
    SparseIndex *sp;

public:
    SparseMatmul(Variable *x, Variable *y, Variable *z, int m, int n, int l, SparseIndex *sp);
    ~SparseMatmul() {}
    void forward(bool);
    void backward();
};

class GraphSum: public Module {
    Variable *in, *out;
    SparseIndex *graph;
    int dim;

public:
    GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim);
    ~GraphSum() {}
    void forward(bool);
    void backward();
};

class CrossEntropyLoss: public Module {
    Variable *logits;
    Variable *truth;
    float *loss;
    int num_classes;
public:
    CrossEntropyLoss(Variable *logits, Variable *truth, float *loss, int num_classes):
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}
    ~CrossEntropyLoss() {}
    void forward(bool);
    void backward();
};