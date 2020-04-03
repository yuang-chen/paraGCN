#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <sstream>

class SparseIndex {
public:
    std::vector<int> indices;
    std::vector<int> indptr;
    void print();
};