#pragma once
#include <chrono>
#include <vector>

using namespace std::chrono;

typedef enum {
    TMR_TRAIN = 0,
    TMR_TEST,
    TMR_MATMUL_FW,
    TMR_MATMUL_BW,
    TMR_SPMATMUL_FW,
    TMR_SPMATMUL_BW,
    TMR_GRAPHSUM_FW,
    TMR_GRAPHSUM_BW,
    TMR_LOSS_FW,
    TMR_RELU_FW,
    TMR_RELU_BW,
    TMR_DROPOUT_FW,
    TMR_DROPOUT_BW,
    _NUM_TMR
} gcn_timer_t;

void timer_start(gcn_timer_t t);
float timer_stop(gcn_timer_t t);
float timer_total(gcn_timer_t t);

#define PRINT_TIMER_AVERAGE(T, E) printf(#T " average time: %.3fms\n", timer_total(T) * 1000 / E);