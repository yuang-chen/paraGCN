#include "parser.h"

Parser::Parser(GCNParams *gcnParams, GCNData *gcnData, string graph_name) {
    this->graph_file.open("/data/cppGCN/" + graph_name + ".graph");
    this->split_file.open("/data/cppGCN/" + graph_name + ".split");
    this->svmlight_file.open("/data/cppGCN/" + graph_name + ".svmlight");
    this->gcnParams = gcnParams;
    this->gcnData = gcnData;
}


void Parser::parseGraph() {
    int node = 0; 
    string line;

    auto &graph_sparse_index = this->gcnData->graph;
    graph_sparse_index.indptr.push_back(0);
 
    while(!this->graph_file.eof()) {    // get the line out of the graph file, which iilutrates node relation
        getline(this->graph_file, line);
        graph_sparse_index.indices.push_back(node);  
        graph_sparse_index.indptr.push_back(graph_sparse_index.indptr.back() + 1);
        node++;
        
        istringstream ss(line);    // analyze the line 
        int number;
        while(!ss.eof()) {       // get the number out of the line
            ss >> number;
            graph_sparse_index.indices.push_back(number);
            graph_sparse_index.indptr.back() += 1;
        }
    }
    this->gcnParams->num_nodes = node;
}

void Parser::parseNode() {
    string line;
    int max_idx = 0, max_label = 0;

    auto &feature_sparse_index = this->gcnData->feature_index;
    auto &feature_val = this->gcnData->feature_value;
    auto &labels = this->gcnData->label;

    feature_sparse_index.indptr.push_back(0);

    while(!svmlight_file.eof()) {
        getline(svmlight_file, line);
        feature_sparse_index.indptr.push_back(feature_sparse_index.indptr.back());

        istringstream ss(line);
        int label;
        ss >> label;
        max_label = max(max_label, label);
        while(!ss.eof()) {
            string target_value;
            ss >> target_value;
            istringstream tv_ss(target_value);
            int target;
            char col;
            float value;
            tv_ss >> target >> col >> value;

            feature_val.push_back(value);
            feature_sparse_index.indices.push_back(target);
            feature_sparse_index.indptr.back() += 1;
            max_idx = max(max_idx, target);
        }        
    }
    gcnParams->input_dim = max_idx + 1;
    gcnParams->output_dim = max_label + 1;
}

void Parser::parseSplit() {
    auto &split = this->gcnData->split;
    string line;

    while(!split_file.eof()) {
        getline(split_file, line);
        split.push_back(stoi(line));
    }
}

bool Parser::isValidInput() {
    return graph_file.is_open() && split_file.is_open() && svmlight_file.is_open();
}


bool Parser::parse() {
    if(!isValidInput()) return false;
    this->parseGraph();
    cout << "parse Graph successfully" << endl;
    this->parseNode();
    cout << "parse Node successfully" << endl;
    this->parseSplit();
    cout << "parse Split successfully" << endl;
    return true;
}
