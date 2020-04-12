#include "timer.h"

time_point<high_resolution_clock> tmr_start[_NUM_TMR];
float tmr_sum[_NUM_TMR];

void timer_start(gcn_timer_t t) {
    tmr_start[t] = high_resolution_clock::now();
}

float timer_stop(gcn_timer_t t) {
    float count = duration_cast<duration<float>>(high_resolution_clock::now() - tmr_start[t]).count();
    tmr_sum[t] += count;
    return count;
}

float timer_total(gcn_timer_t t) {
    return tmr_sum[t];
}
