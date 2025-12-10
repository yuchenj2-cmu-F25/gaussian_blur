#ifndef UTILS_H
#define UTILS_H

#include "config.h"

/* Timing function */
unsigned long long rdtsc(void);

/* Utility functions */
void fill_random_array(float *arr, int size);
float compare_arrays(float *ref, float *test, int size);

#endif /* UTILS_H */

