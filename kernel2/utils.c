#include "utils.h"
#include <stdlib.h>
#include <math.h>

unsigned long long rdtsc(void) {
    unsigned a, d;
    __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));
    return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

void fill_random_array(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)(rand() % 256) - 128.0f; // Range: -128 to 127
    }
}

float compare_arrays(float *ref, float *test, int size) {
    float diff = 0.0f;
    for (int i = 0; i < size; i++) {
        diff += fabsf(ref[i] - test[i]);
    }
    return diff / size;
}

