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


CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes):
    logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training) {
    float total_loss = 0;
    int count = 0;
    if(training) {
        logits->zero_grad();
    }

    for(int i = 0; i < logits->data.size() / num_classes; i++) {
        if(truth[i] < 0) {
            continue;
        }
        count++;
        float *logit = &logits->data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;

        for(int j = 0; j < num_classes; j++) {
            max_logit = fmax(max_logit, logit[i]);
        }

        for(int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }
        total_loss += logf(sum_exp) - logit[truth[i]];

        if(training) {
            for(int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }
    *loss = total_loss / count;
    if(training) {
        for(int i = 0; i < logits->grad.size(); i++) {
            logits->grad[i] /= count;
        }
    }
}

void CrossEntropyLoss::backward() {}

ReLU::ReLU(Variable *in): in(in) {
    mask = new bool[in->data.size()];
}

ReLU::~ReLU() {
    delete[] mask;
}
void ReLU::forward(bool training) {
    for(int i = 0; i < in->data.size[]; i++) {
        bool keep = in->data[i] > 0;
        if(training) {
            mask[i] = keep;
        }
        if(!keep) {
            in->data[i] = 0;
        }
    }
}

void ReLU::backward() {
    for(int i = 0; i < in->data.size(); i++) {
        if(!mask[i]) {
            in->grad[i] = 0;
        }
    }
}

Dropout::Dropout(Variable *in, float p): in(in), p(p) {
    if(!in->grad.empty()) {
        mask = new int[in->data.size()];
    } else {
        mask = nullptr;
    }
}

Dropout::~Dropout() {
    delete[] mask;
}

void Dropout::forward(bool training) {
    const int threshold = int(p * GCN_RAND_MAX);
    float scale = 1 / (1 - p);

    for(int i = 0; i < in->data.size(); i++) {
        bool keep = int(RAND()) >= threshold;
        in->data[i] *= keep ? scale : 0;
        if(mask) {
            mask[i] = keep;
        }
    }
}

void Dropout::backward() {
    if(!mask) return;

    float scale = 1 / (1 - p);
    for(int i = 0; i < in->data.size(); i++) {
        in->grad[i] *= mask[i] ? scale : 0;
    }
}

