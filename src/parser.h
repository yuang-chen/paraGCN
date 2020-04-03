#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include "gcn.h"

using namespace std;

class Parser {
public:
    Parser(GCNParams *gcnParams, GCNData *gcnData, string graph_name);
    bool parse();

private:
    ifstream graph_file;
    ifstream split_file;
    ifstream svmlight_file;
    GCNParams *gcnParams;
    GCNData *gcnData;
    void parseGraph();
    void parseSplit();
    void parseNode();
    bool isValidInput();
};