#include "module.h"

Matmul::Matmul(Variable *x, Variable *y, Variable *z, int m, int n, int l):
    x(x), y(y), z(z), m(m), n(n), l(l) {}

void Matmul::forward(bool training) {
    z->zero();
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < l; k++) {
                z->data[i * l + k] += x->data[i * n + j] * y->data[j * l + k];
            }
        }
    }
}

void Matmul::backward() {
    x->zero_grad();
    y->zero_grad();
    for(int i =0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float tmp = 0;

            for(int k = 0; k < l; k++) {
                tmp += z->grad[i * l + k] * y->data[j * l + k];
                y->grad[j * l + k] += z->grad[i * l + k] * x->data[i * n + j];
            }
            x->grad[i * n + j] = tmp;
        }
    }
}

SparseMatmul::SparseMatmul(Variable *x, Variable *y, Variable *z, int m, int n, int l, SparseIndex *sp):
        x(x), y(y), z(z), m(m), n(n), l(l), sp(sp) {}

void SparseMatmul::forward(bool training) {
    z->zero();
    
    for(int i = 0; i < sp->indptr.size() - 1; i++) {
        for(int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for(int k = 0; k < l; k++) {
                z->data[i * l + k] += x->data[jj] * y->data[j * l + k];
            }
        }
    }
}

void SparseMatmul::backward() {
    y->zero_grad();

    for(int i = 0; i < sp->indptr.size() - 1; i++) {
        for(int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for(int k = 0; k < l; k++) {
                y->grad[j * l + k] += z->grad[i * l + k] * x->data[jj];
            }
        }
    }
}

// compute the normalization coefficients of adj matrix
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) :
                in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training) {
    out->zero();
    for(int src = 0; src < graph->indptr.size() - 1; src++) {
        for(int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );

            for(int j = 0; j < dim; j++) {
                out->data[src*dim + j] += coef * in->data[dst * dim + j];
            }
        }
    }
}

void GraphSum::backward() {
    in->zero_grad();
    for(int src = 0; src < graph->indptr.size() - 1; src++) {
        for(int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );

            for(int j = 0; j < dim; j++) {
                in->grad[src*dim + j] += coef * in->data[dst * dim + j];
            }
        }
    }
}


void CrossEntropyLoss()
