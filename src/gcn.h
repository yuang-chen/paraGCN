#pragma once

#include <vector>
#include "sparse.h"

struct GCNParams {
    int num_nodes, input_dim, hidden_dim, output_dim;
    float dropout, learning_rate, weight_decay;
    int epochs, early_stopping;
    static GCNParams get_default();
};



class GCNData {
public:
    SparseIndex feature_index, graph;    
    std::vector<int> split;
    std::vector<int> label;
    std::vector<float> feature_value;
};



class GCN {
    GCNData *data;
public:
    GCNParams params;
    GCN(GCNParams params, GCNData *data);
    ~GCN();
    void run();
};