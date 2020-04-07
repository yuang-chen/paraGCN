#include "rand.h"

//uint64_t rand_state[2];

/*
void init_rand_state() {
    rand_state[0] = rand();
    rand_state[1] = rand();
}*/

uint32_t xorshift128plus() {
    uint64_t state[2];
    state[0] = rand();
    state[1] = rand();
    uint64_t t = state[0];
    uint64_t const s = state[1];
    assert(t && s);
    state[0] = t;
    t ^= t << 23;
    t ^= t >> 17;
    t ^= s ^ (s >> 26);
    state[1] = t;
    uint32_t res = (t + s) & 0x7fffffff;
    return res;
}




