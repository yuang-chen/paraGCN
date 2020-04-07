#pragma once

#include <cstdlib>
#include <cstdint>
#include <assert.h>

#define GCN_RAND_MAX 0x7fffffff

//void init_rand_state();
uint32_t xorshift128plus();

#define RAND() xorshift128plus()
