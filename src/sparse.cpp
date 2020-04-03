#include "sparse.h"

using namespace std;

void SparseIndex::print() {
    cout << "----sparse index info----" << endl;

    cout << "indptr: ";
    for(auto i: indptr) {
        cout << i << " ";
    } 
    cout << endl;

    
    cout << "indices: ";
    for(auto i: indices) {
        cout << i << " ";
    } 
    cout << endl;
}