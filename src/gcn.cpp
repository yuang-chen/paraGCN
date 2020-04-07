#include "gcn.h"

GCNParams GCNParams::get_default() {
    return {2708, 1433, 16, 7, 0.5, 0.01, 5e-4, 100, 0};
};


GCN::GCN(GCNParams params, GCNData *data) {
    this->params = params;
    this->data = data;
    modules.reserve(8);
    variables.reserve(7);
    // register and initialize the input data
    variables.emplace_back(data->feature_index.indices.size(), false);
    input = &variables.back();
    // register and get the * at layer 1
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var1 = &variables.back();
    // register weights at layer 1 and randomize them
    variables.emplace_back(params.input_dim * params.hidden_dim, true, true);
    Variable *layer1_weight = &variables.back();
    layer1_weight->glorot(params.input_dim, params.hidden_dim);
    // register the 2nd set parameters at layer 1
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var2 = &variables.back();
    // do it for the 2nd layer
    variables.emplace_back(params.hidden_dim * params.output_dim);
    Variable *layer2_var1 = &variables.back();
    // weights of the 2nd layer
    variables.emplace_back(params.hidden_dim * params.output_dim, true, true);
    Variable *layer2_weight = &variables.back();
     // for the output
    variables.emplace_back(params.num_nodes * params.output_dim);
    Variable *output = &variables.back();
    /******************  register the network module ***********************/
    modules.emplace_back(new Dropout(input, params.dropout));
    
}

GCN::~GCN() {

}