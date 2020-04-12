#pragma once

#include <vector>
#include "sparse.h"
#include "rand.h"
#include "module.h"
#include "optim.h"
#include "timer.h"
#include <tuple>

using namespace std;

struct GCNParams {
    int num_nodes, input_dim, hidden_dim, output_dim;
    float dropout, learning_rate, weight_decay;
    int epochs, early_stopping;
    static GCNParams get_default();
};



class GCNData {
public:
    SparseIndex feature_index, graph;    
    vector<int> split;
    vector<int> label;
    vector<float> feature_value;
};



class GCN {
    GCNData *data;
    vector<Module*> modules;
    vector<Variable> variables;
    Variable *input, *output;
    vector<int> truth;
    float loss;
    Adam optimizer;

    void set_input();
    void set_truth(int current_split);
    float get_accuracy();
    float get_l2_penalty();
    pair<float, float> train_epoch();
    pair<float, float> eval(int current_split);

public:
    GCNParams params;
    GCN(GCNParams params, GCNData *data);
    ~GCN();
    void run();
};