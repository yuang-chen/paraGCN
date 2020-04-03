#include "gcn.h"

GCNParams GCNParams::get_default() {
    return {2708, 1433, 16, 7, 0.5, 0.01, 5e-4, 100, 0};
};


GCN::GCN(GCNParams params, GCNData *data) {
    this->params = params;
    this->data = data;
}

GCN::~GCN() {

}